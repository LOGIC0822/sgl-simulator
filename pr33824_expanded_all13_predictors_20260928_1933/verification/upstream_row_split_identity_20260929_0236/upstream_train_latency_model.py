"""Train the 18-feature regressor consumed by ``MLTimePredictor``.

The input is one or more baseline-collection directories containing
``TP0*.schedule_batch.jsonl``. Every retained row represents one real,
device-synchronized SGLang forward iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)

FEATURE_NAMES = list(ML_FEATURE_NAMES)
MONOTONIC_FEATURES = {
    "batch_size",
    "sum_extend",
    "sum_past",
    "sum_extend_x_past",
    "sum_extend_squared",
    "sum_past_squared",
    "sum_attn_flops",
    "sum_extend_x_max_past",
    "log1p_sum_past",
    "log1p_sum_attn_flops",
    "batch_size_x_sum_extend",
}


def parse_case_metadata(name: str) -> dict[str, str]:
    """Best-effort grouping metadata for common baseline directory names."""
    metadata = {
        "case": name,
        "bucket": "unknown",
        "node": "unknown",
        "pod": "unknown",
        "rate": "unknown",
    }

    match = re.match(r"(?P<bucket>.+?)_(?P<node>node\d+)_pod-(?P<pod>.+)$", name)
    if match:
        metadata.update(match.groupdict())

    match = re.search(r"bucket-(?P<bucket>[0-9a-zA-Z+\-]+)", name)
    if match:
        metadata["bucket"] = match.group("bucket")

    match = re.search(r"(?:^|_)node(?P<number>\d+)(?:_|$)", name)
    if match:
        metadata["node"] = f"node{match.group('number')}"

    match = re.search(r"multinode(?P<number>\d+)", name)
    if match:
        metadata["node"] = f"node{match.group('number')}"

    match = re.search(r"pod-(?P<pod>[^_]+)", name)
    if match:
        metadata["pod"] = match.group("pod")

    match = re.search(r"_x(?P<rate>[0-9.]+)_", name)
    if match:
        metadata["rate"] = f"x{match.group('rate')}"
    elif "maxtps" in name or "max-tps" in name:
        metadata["rate"] = "maxtps"

    return metadata


def iter_schedule_files(data_roots: list[str], schedule_glob: str) -> list[Path]:
    files: set[Path] = set()
    for value in data_roots:
        root = Path(value).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"training data root does not exist: {root}")
        if root.is_file():
            files.add(root.resolve())
        else:
            files.update(path.resolve() for path in root.rglob(schedule_glob))
    return sorted(files)


def _load_json_line(schedule: Path, line: str, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {schedule}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {schedule}:{line_number}")
    return value


def collect_rows(
    data_roots: list[str],
    schedule_glob: str,
    forward_modes: set[int],
    max_latency: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []

    for schedule in iter_schedule_files(data_roots, schedule_glob):
        metadata = parse_case_metadata(schedule.parent.name)
        case_id = str(schedule.parent.resolve())
        kept = 0
        skipped_latency = 0
        skipped_mode = 0
        skipped_empty = 0

        with schedule.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                item = _load_json_line(schedule, line, line_number)
                forward_mode = item.get("forward_mode")
                if forward_mode not in forward_modes:
                    skipped_mode += 1
                    continue

                requests = item.get("request_infos") or []
                if not requests:
                    skipped_empty += 1
                    continue

                latency = float(item.get("iter_latency", 0.0))
                if latency <= 0 or latency > max_latency:
                    skipped_latency += 1
                    continue

                try:
                    extend_lengths = [
                        int(request["extend_input_len"]) for request in requests
                    ]
                    past_lengths = [
                        int(request["prefix_indices_len"]) for request in requests
                    ]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid request_infos at {schedule}:{line_number}: {exc}"
                    ) from exc

                rows.append(
                    {
                        **metadata,
                        "case_id": case_id,
                        "source_schedule": str(schedule),
                        "line_number": line_number,
                        "forward_mode": int(forward_mode),
                        "features": extract_ml_features(extend_lengths, past_lengths),
                        "latency": latency,
                        "batch_size": len(extend_lengths),
                        "sum_extend": sum(extend_lengths),
                        "sum_past": sum(past_lengths),
                        "total_tokens": sum(extend_lengths) + sum(past_lengths),
                    }
                )
                kept += 1

        per_case.append(
            {
                **metadata,
                "case_id": case_id,
                "schedule_path": str(schedule),
                "rows": kept,
                "skipped_mode": skipped_mode,
                "skipped_empty": skipped_empty,
                "skipped_latency": skipped_latency,
            }
        )

    return rows, per_case


def as_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["features"] for row in rows], dtype=np.float64),
        np.asarray([row["latency"] for row in rows], dtype=np.float64),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return {
            "n": 0,
            "R2": "",
            "MAE_ms": "",
            "MAPE_pct": "",
            "P95APE_pct": "",
        }

    absolute_percentage_error = (
        np.abs((y_pred - y_true) / np.maximum(y_true, 1e-9)) * 100
    )
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    return {
        "n": int(len(y_true)),
        "R2": float(r2),
        "MAE_ms": float(mean_absolute_error(y_true, y_pred) * 1000),
        "MAPE_pct": float(np.mean(absolute_percentage_error)),
        "P95APE_pct": float(np.percentile(absolute_percentage_error, 95)),
    }


def train_model(
    rows: list[dict[str, Any]], loss_name: str, seed: int
) -> tuple[HistGradientBoostingRegressor, dict[str, Any], list[int]]:
    features, latency = as_arrays(rows)
    monotonic = [
        int(feature_name in MONOTONIC_FEATURES) for feature_name in FEATURE_NAMES
    ]
    hyperparameters: dict[str, Any] = {
        "max_iter": 400,
        "max_depth": None,
        "learning_rate": 0.04,
        "max_leaf_nodes": 63,
        "l2_regularization": 0.1,
        "monotonic_cst": monotonic,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "random_state": seed,
    }
    if loss_name == "p50":
        hyperparameters.update(loss="quantile", quantile=0.5)
    elif loss_name == "l2":
        hyperparameters.update(loss="squared_error")
    else:
        raise ValueError(f"unknown loss: {loss_name}")

    model = HistGradientBoostingRegressor(**hyperparameters)
    model.fit(features, latency)
    return model, hyperparameters, monotonic


def build_case_stratified_split(
    rows: list[dict[str, Any]], seed: int, test_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    random = np.random.RandomState(seed)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    for case_rows in by_case.values():
        if len(case_rows) == 1:
            train.extend(case_rows)
            continue
        order = random.permutation(len(case_rows))
        test_count = int(round(len(case_rows) * test_fraction))
        test_count = min(max(test_count, 1), len(case_rows) - 1)
        test.extend(case_rows[index] for index in order[:test_count])
        train.extend(case_rows[index] for index in order[test_count:])
    return train, test


def evaluate_model(
    model: HistGradientBoostingRegressor,
    rows: list[dict[str, Any]],
    split_name: str,
    loss_name: str,
) -> list[dict[str, Any]]:
    features, latency = as_arrays(rows)
    prediction = model.predict(features)
    output: list[dict[str, Any]] = []

    def add(group: str, value: str, indices: np.ndarray) -> None:
        output.append(
            {
                "loss": loss_name,
                "split": split_name,
                "group": group,
                "value": value,
                **regression_metrics(latency[indices], prediction[indices]),
            }
        )

    add("all", "all", np.arange(len(rows)))
    for key in ("bucket", "node", "rate", "case"):
        for value in sorted({row[key] for row in rows}):
            indices = np.asarray(
                [index for index, row in enumerate(rows) if row[key] == value],
                dtype=int,
            )
            add(key, value, indices)

    token_bins = (
        (0, 1024),
        (1024, 2048),
        (2048, 4096),
        (4096, 8192),
        (8192, 16384),
        (16384, 32768),
        (32768, 65536),
        (65536, 131072),
        (131072, 262144),
        (262144, 10**18),
    )
    for lower, upper in token_bins:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if lower <= row["total_tokens"] < upper
            ],
            dtype=int,
        )
        if len(indices):
            label = f"{lower}-{upper if upper < 10**18 else 'inf'}"
            add("total_tokens", label, indices)

    batch_bins = (
        (1, 2, "1"),
        (2, 3, "2"),
        (3, 5, "3-4"),
        (5, 9, "5-8"),
        (9, 17, "9-16"),
        (17, 33, "17-32"),
        (33, 10**18, "33+"),
    )
    for lower, upper, label in batch_bins:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if lower <= row["batch_size"] < upper
            ],
            dtype=int,
        )
        if len(indices):
            add("batch_size", label, indices)

    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_training_log(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    per_case: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write("# ML latency predictor training log\n\n")
        output.write(f"- data_roots: {args.data_root}\n")
        output.write(f"- schedule_glob: {args.schedule_glob}\n")
        output.write(f"- rows: {len(rows)}\n")
        output.write(f"- cases: {len(per_case)}\n")
        output.write(
            f"- forward_modes: {sorted(parse_forward_modes(args.forward_modes))}\n"
        )
        output.write(f"- max_latency_seconds: {args.max_latency}\n")
        output.write(f"- features: {', '.join(FEATURE_NAMES)}\n\n")
        output.write("## Case-stratified holdout\n\n")
        for summary in summaries:
            output.write(
                f"- {summary['loss']}: MAPE={summary['MAPE_pct']:.3f}%, "  # codespell:ignore mape
                f"MAE={summary['MAE_ms']:.3f}ms, R2={summary['R2']:.5f}\n"
            )


def parse_forward_modes(value: str) -> set[int]:
    modes = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not modes:
        raise argparse.ArgumentTypeError("at least one forward mode is required")
    return modes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        help=(
            "Baseline directory or schedule-batch JSONL file. Repeat for "
            "multiple independent data roots."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-prefix", default="latency_model_hgbmono")
    parser.add_argument(
        "--schedule-glob",
        default="TP0*.schedule_batch.jsonl",
        help="Recursive filename glob used for each directory data root.",
    )
    parser.add_argument("--forward-modes", default="1")
    parser.add_argument("--max-latency", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument(
        "--loss",
        choices=("l2", "p50", "both"),
        default="both",
        help="Train squared-error, median-quantile, or both model candidates.",
    )
    args = parser.parse_args(argv)
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.max_latency <= 0:
        parser.error("--max-latency must be positive")
    if Path(args.model_prefix).name != args.model_prefix:
        parser.error("--model-prefix must be a filename prefix, not a path")
    parse_forward_modes(args.forward_modes)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.out_dir).expanduser().resolve()
    evaluation_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    forward_modes = parse_forward_modes(args.forward_modes)
    rows, per_case = collect_rows(
        args.data_root, args.schedule_glob, forward_modes, args.max_latency
    )
    if not per_case:
        raise RuntimeError(
            f"no files matching {args.schedule_glob!r} under {args.data_root}"
        )
    if not rows:
        raise RuntimeError(f"no usable training rows found under {args.data_root}")

    write_csv(
        evaluation_dir / "dataset_cases.csv",
        per_case,
        [
            "case",
            "case_id",
            "bucket",
            "node",
            "pod",
            "rate",
            "schedule_path",
            "rows",
            "skipped_mode",
            "skipped_empty",
            "skipped_latency",
        ],
    )

    train_rows, test_rows = build_case_stratified_split(
        rows, args.seed, args.test_fraction
    )
    if not train_rows or not test_rows:
        raise RuntimeError(
            "case-stratified evaluation requires at least one case with two "
            "or more usable rows"
        )

    losses = ("l2", "p50") if args.loss == "both" else (args.loss,)
    evaluation_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for loss_name in losses:
        evaluation_model, _, _ = train_model(train_rows, loss_name, args.seed)
        split_evaluation = evaluate_model(
            evaluation_model, test_rows, "case_stratified", loss_name
        )
        evaluation_rows.extend(split_evaluation)
        overall = next(row for row in split_evaluation if row["group"] == "all")
        summaries.append(
            {
                "loss": loss_name,
                "split": "case_stratified",
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "R2": overall["R2"],
                "MAE_ms": overall["MAE_ms"],
                "MAPE_pct": overall["MAPE_pct"],
                "P95APE_pct": overall["P95APE_pct"],
            }
        )

        final_model, hyperparameters, monotonic = train_model(
            rows, loss_name, args.seed
        )
        model_path = output_dir / f"{args.model_prefix}_{loss_name}.pkl"
        joblib.dump(
            {
                "model": final_model,
                "features": FEATURE_NAMES,
                "monotonic": monotonic,
                "best_params": hyperparameters,
                "version": f"{args.model_prefix}_{loss_name}",
                "format_version": 1,
                "train_rows": len(rows),
                "train_cases": len(per_case),
                "source_data_roots": args.data_root,
                "target": "iter_latency_seconds",
                "forward_modes": sorted(forward_modes),
                "max_latency_seconds": args.max_latency,
                "input_contract": ("[[extend_input_len, prefix_indices_len], ...]"),
            },
            model_path,
        )
        print(f"saved {model_path} ({model_path.stat().st_size / 1024:.1f} KiB)")

    write_csv(
        evaluation_dir / "split_summary.csv",
        summaries,
        [
            "loss",
            "split",
            "train_rows",
            "test_rows",
            "R2",
            "MAE_ms",
            "MAPE_pct",
            "P95APE_pct",
        ],
    )
    write_csv(
        evaluation_dir / "split_metrics_by_group.csv",
        evaluation_rows,
        [
            "loss",
            "split",
            "group",
            "value",
            "n",
            "R2",
            "MAE_ms",
            "MAPE_pct",
            "P95APE_pct",
        ],
    )
    write_training_log(
        evaluation_dir / "training_log.md", args, rows, per_case, summaries
    )

    print(f"rows={len(rows)} cases={len(per_case)}")
    for summary in summaries:
        print(
            f"{summary['loss']:>3s} case_stratified "
            f"R2={summary['R2']:.4f} MAE={summary['MAE_ms']:.2f}ms "
            f"MAPE={summary['MAPE_pct']:.2f}% "  # codespell:ignore mape
            f"P95APE={summary['P95APE_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
