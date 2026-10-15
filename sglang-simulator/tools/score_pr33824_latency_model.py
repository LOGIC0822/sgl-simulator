#!/usr/bin/env python3
"""Score an ML latency bundle against device-synchronized collected batches.

This is a teacher-forced, scheduler-iteration comparison: each prediction uses
the exact batch composition observed on the real server.  It is intentionally
separate from end-to-end replay, where a prediction error can change later
scheduler batches and therefore compound over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)


def parse_forward_modes(value: str) -> set[int]:
    modes = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not modes:
        raise argparse.ArgumentTypeError("at least one forward mode is required")
    return modes


def schedule_files(data_roots: list[str], pattern: str) -> list[Path]:
    files: set[Path] = set()
    for value in data_roots:
        root = Path(value).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"data root does not exist: {root}")
        if root.is_file():
            files.add(root.resolve())
        else:
            files.update(path.resolve() for path in root.rglob(pattern))
    return sorted(files)


def read_rows(
    data_roots: list[str],
    pattern: str,
    forward_modes: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schedule_path in schedule_files(data_roots, pattern):
        case = schedule_path.parent.name
        with schedule_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON at {schedule_path}:{line_number}: {error}"
                    ) from error
                mode = value.get("forward_mode")
                if mode not in forward_modes:
                    continue
                requests = value.get("request_infos") or []
                if not requests:
                    continue
                try:
                    actual = float(value["iter_latency"])
                    extend = [int(item["extend_input_len"]) for item in requests]
                    past = [int(item["prefix_indices_len"]) for item in requests]
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid batch at {schedule_path}:{line_number}: {error}"
                    ) from error
                if not math.isfinite(actual) or actual <= 0:
                    continue
                rows.append(
                    {
                        "case": case,
                        "schedule_path": str(schedule_path),
                        "line_number": line_number,
                        "forward_mode": int(mode),
                        "batch_size": len(extend),
                        "sum_extend": sum(extend),
                        "sum_past": sum(past),
                        "features": extract_ml_features(extend, past),
                        "actual_seconds": actual,
                    }
                )
    if not rows:
        raise RuntimeError("no usable synchronized batch rows found")
    return rows


def load_model(path: Path) -> Any:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("ML bundle must contain model and ordered features")
    if list(bundle["features"]) != list(ML_FEATURE_NAMES):
        raise ValueError("ML bundle feature ABI does not match MLTimePredictor")
    model = bundle["model"]
    if not callable(getattr(model, "predict", None)):
        raise TypeError("ML bundle model must provide predict()")
    return model


def score_rows(rows: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    predictions = np.asarray(model.predict(features), dtype=np.float64)
    if len(predictions) != len(rows):
        raise ValueError("model returned a prediction count different from input rows")

    scored: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        if not math.isfinite(float(prediction)) or prediction < 0:
            raise ValueError("model returned a negative or non-finite latency")
        actual = float(row["actual_seconds"])
        signed = (float(prediction) - actual) / actual * 100.0
        scored.append(
            {
                **{key: value for key, value in row.items() if key != "features"},
                "predicted_seconds": float(prediction),
                "signed_error_pct": signed,
                "absolute_percentage_error_pct": abs(signed),
            }
        )
    return scored


def summary_row(
    group: str, value: str, rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    selected = list(rows)
    actual = np.asarray([row["actual_seconds"] for row in selected], dtype=np.float64)
    predicted = np.asarray(
        [row["predicted_seconds"] for row in selected], dtype=np.float64
    )
    signed = (predicted - actual) / actual * 100.0
    absolute = np.abs(signed)
    return {
        "group": group,
        "value": value,
        "n": len(selected),
        "mean_actual_ms": float(np.mean(actual) * 1000.0),
        "mean_predicted_ms": float(np.mean(predicted) * 1000.0),
        "mae_ms": float(np.mean(np.abs(predicted - actual)) * 1000.0),
        "rmse_ms": float(np.sqrt(np.mean((predicted - actual) ** 2)) * 1000.0),
        "mean_signed_error_pct": float(np.mean(signed)),
        "mape_pct": float(np.mean(absolute)),
        "median_ape_pct": float(np.median(absolute)),
        "p95_ape_pct": float(np.percentile(absolute, 95)),
    }


def summarize(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [summary_row("all", "all", scored)]
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "case": defaultdict(list),
        "forward_mode": defaultdict(list),
        "case_forward_mode": defaultdict(list),
    }
    for row in scored:
        groups["case"][row["case"]].append(row)
        groups["forward_mode"][str(row["forward_mode"])].append(row)
        groups["case_forward_mode"][
            f"{row['case']}:mode{row['forward_mode']}"
        ].append(row)
    for group, values in groups.items():
        rows.extend(summary_row(group, value, selected) for value, selected in sorted(values.items()))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", action="append", required=True)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--schedule-glob", default="TP0*.schedule_batch.jsonl")
    parser.add_argument("--forward-modes", default="1,2")
    args = parser.parse_args(argv)
    args.model_bundle = Path(args.model_bundle).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    if not args.model_bundle.is_file():
        parser.error(f"ML bundle does not exist: {args.model_bundle}")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.out_dir}")
    parse_forward_modes(args.forward_modes)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    modes = parse_forward_modes(args.forward_modes)
    model = load_model(args.model_bundle)
    rows = read_rows(args.data_root, args.schedule_glob, modes)
    scored = score_rows(rows, model)
    summary = summarize(scored)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(
        args.out_dir / "per_batch.csv",
        scored,
        [
            "case",
            "schedule_path",
            "line_number",
            "forward_mode",
            "batch_size",
            "sum_extend",
            "sum_past",
            "actual_seconds",
            "predicted_seconds",
            "signed_error_pct",
            "absolute_percentage_error_pct",
        ],
    )
    write_csv(
        args.out_dir / "summary.csv",
        summary,
        [
            "group",
            "value",
            "n",
            "mean_actual_ms",
            "mean_predicted_ms",
            "mae_ms",
            "rmse_ms",
            "mean_signed_error_pct",
            "mape_pct",
            "median_ape_pct",
            "p95_ape_pct",
        ],
    )
    overall = summary[0]
    (args.out_dir / "README.md").write_text(
        "# ML scheduler-batch scoring\n\n"
        "This is teacher-forced scoring against real, device-synchronized "
        "ScheduleBatch rows. It does not replay a new scheduler sequence.\n\n"
        f"- bundle: `{args.model_bundle}`\n"
        f"- forward modes: `{sorted(modes)}`\n"
        f"- batches: {overall['n']}\n"
        f"- MAPE: {overall['mape_pct']:.3f}%\n"
        f"- MAE: {overall['mae_ms']:.3f} ms\n",
        encoding="utf-8",
    )
    print(
        f"batches={overall['n']} MAPE={overall['mape_pct']:.3f}% "
        f"MAE={overall['mae_ms']:.3f}ms out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
