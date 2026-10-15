#!/usr/bin/env python3
"""Train the simulator's ML latency predictor from native SGLang measurements.

``sglang.benchmark.one_batch`` records a prefill latency and, when invoked
with ``--log-decode-step 1``, one measured latency for every decode step.  This
tool converts those records to the exact 18-feature ABI consumed by
``MLTimePredictor``, validates an estimator on a separately measured grid, and
writes a sklearn/joblib bundle that the simulator can load directly.

The input files are intentionally kept separate from AIConfigurator.  They are
native CUDA measurements made by the same SGLang source, CUDA-graph settings,
and model checkpoint as the target simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_NAMES = [
    "batch_size",
    "sum_extend",
    "max_extend",
    "min_extend",
    "sum_past",
    "max_past",
    "min_past",
    "sum_extend_x_past",
    "sum_extend_squared",
    "sum_past_squared",
    "sum_attn_flops",
    "sum_extend_x_max_past",
    "log1p_sum_past",
    "log1p_sum_attn_flops",
    "batch_size_x_sum_extend",
    "max_past_minus_min_past",
    "is_decode",
    "is_prefill",
]

DECODE_RE = re.compile(
    r"^Decode\s+(?P<step>\d+)\.\s+Batch size:\s+(?P<batch>\d+),\s+"
    r"latency:\s*(?P<latency>[0-9]+(?:\.[0-9]+)?)\s+s,"
)
TOTAL_RE = re.compile(r"^Total\.\s+latency:")


@dataclass(frozen=True)
class Sample:
    """One feature/latency observation for a measured forward pass."""

    split: str
    stage: str
    batch_size: int
    input_len: int
    output_len: int
    decode_step: int | None
    target_seconds: float
    features: tuple[float, ...]

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "split": self.split,
            "stage": self.stage,
            "batch_size_source": self.batch_size,
            "input_len_source": self.input_len,
            "output_len_source": self.output_len,
            "decode_step": "" if self.decode_step is None else self.decode_step,
            "target_seconds": self.target_seconds,
        }
        row.update(dict(zip(FEATURE_NAMES, self.features, strict=True)))
        return row


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def make_features(extends: Sequence[int], pasts: Sequence[int]) -> tuple[float, ...]:
    """Return MLTimePredictor's documented feature vector exactly.

    The matching implementation in ``time_predictor/ml.py`` accepts integer
    request lengths.  We keep the calculation here explicit so this trainer
    does not silently depend on an installed editable package.
    """
    if not extends or len(extends) != len(pasts):
        raise ValueError("extends and pasts must be non-empty sequences of equal length")
    if any(item < 0 for item in (*extends, *pasts)):
        raise ValueError("request lengths must be non-negative")

    batch_size = len(extends)
    sum_extend = sum(extends)
    sum_past = sum(pasts)
    sum_extend_x_past = sum(ext * past for ext, past in zip(extends, pasts))
    sum_extend_squared = sum(ext * ext for ext in extends)
    sum_past_squared = sum(past * past for past in pasts)
    sum_attn_flops = sum(
        ext * (past + ext / 2) for ext, past in zip(extends, pasts)
    )
    max_extend = max(extends)
    min_extend = min(extends)
    max_past = max(pasts)
    min_past = min(pasts)

    return (
        float(batch_size),
        float(sum_extend),
        float(max_extend),
        float(min_extend),
        float(sum_past),
        float(max_past),
        float(min_past),
        float(sum_extend_x_past),
        float(sum_extend_squared),
        float(sum_past_squared),
        float(sum_attn_flops),
        float(sum_extend * max_past),
        float(math.log1p(sum_past)),
        float(math.log1p(sum_attn_flops)),
        float(batch_size * sum_extend),
        float(max_past - min_past),
        float(int(all(ext == 1 for ext in extends))),
        float(int(any(ext > 1 for ext in extends))),
    )


def read_result_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"Missing one_batch result file: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object on {path}:{line_number}")
        for field in ("batch_size", "input_len", "output_len", "prefill_latency"):
            if field not in row:
                raise ValueError(f"Missing {field!r} on {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No one_batch result rows in {path}")
    return rows


def read_decode_groups(path: Path) -> list[list[tuple[int, int, float]]]:
    """Read each benchmark decode sequence after the ``Benchmark ...`` marker.

    A group is closed by the matching ``Total.`` line.  The initial one_batch
    warmup is deliberately excluded, which is important because it often
    contains graph-capture work that is not part of the steady-state sweep.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing one_batch log: {path}")
    after_benchmark = False
    current: list[tuple[int, int, float]] | None = None
    groups: list[list[tuple[int, int, float]]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        # ``run_grouped_one_batch_grid`` concatenates independent one_batch
        # processes into one log.  Each process has its own warmup before its
        # own ``Benchmark ...`` marker, so the marker must reset the parsing
        # state rather than inheriting the preceding group's steady-state mode.
        if line.startswith("# grouped measurement:"):
            if current:
                raise ValueError(f"Unclosed decode group before {line!r} in {path}")
            after_benchmark = False
            continue
        if line == "Benchmark ...":
            after_benchmark = True
            current = None
            continue
        if not after_benchmark:
            continue
        match = DECODE_RE.match(line)
        if match:
            step = int(match.group("step"))
            batch_size = int(match.group("batch"))
            latency = _finite_positive(match.group("latency"))
            if latency is None:
                raise ValueError(f"Invalid decode latency in {path}: {line}")
            if step == 0:
                if current:
                    raise ValueError(
                        f"Decode group in {path} began before previous group closed"
                    )
                current = []
            if current is None:
                raise ValueError(f"Decode step before Decode 0 in {path}: {line}")
            current.append((step, batch_size, latency))
            continue
        if TOTAL_RE.match(line):
            if current is not None:
                groups.append(current)
                current = None
    if current:
        raise ValueError(f"Unclosed decode group in {path}")
    return groups


def extract_samples(result_path: Path, log_path: Path, split: str) -> list[Sample]:
    """Convert measured static batches into prefill and per-step decode rows."""
    results = read_result_rows(result_path)
    groups = read_decode_groups(log_path)
    if len(groups) != len(results):
        raise ValueError(
            "one_batch log/result mismatch: "
            f"{len(groups)} completed decode groups but {len(results)} JSON rows; "
            "rerun the measurement without truncated logging"
        )

    samples: list[Sample] = []
    for result, decode_group in zip(results, groups, strict=True):
        try:
            batch_size = int(result["batch_size"])
            input_len = int(result["input_len"])
            output_len = int(result["output_len"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid shape row in {result_path}: {result}") from exc
        if min(batch_size, input_len, output_len) <= 0:
            raise ValueError(f"Non-positive shape row in {result_path}: {result}")

        prefill_latency = _finite_positive(result["prefill_latency"])
        if prefill_latency is None:
            raise ValueError(f"Invalid prefill latency in {result_path}: {result}")
        samples.append(
            Sample(
                split=split,
                stage="prefill",
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
                decode_step=None,
                target_seconds=prefill_latency,
                features=make_features([input_len] * batch_size, [0] * batch_size),
            )
        )

        expected_steps = list(range(max(output_len - 1, 0)))
        actual_steps = [item[0] for item in decode_group]
        if actual_steps != expected_steps:
            raise ValueError(
                f"Decode steps do not match output_len={output_len} in {log_path}; "
                f"expected {len(expected_steps)} steps, got {len(actual_steps)}"
            )
        for step, logged_batch_size, latency in decode_group:
            if logged_batch_size != batch_size:
                raise ValueError(
                    f"Decode batch size mismatch in {log_path}: "
                    f"result={batch_size}, logged={logged_batch_size}"
                )
            # ``one_batch`` has already produced one token after extend before
            # calling decode(step=0).  SchedulerHook builds its decode feature
            # from prefix + generated-token count, giving input_len + 1 here.
            past = input_len + 1 + step
            samples.append(
                Sample(
                    split=split,
                    stage="decode",
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
                    decode_step=step,
                    target_seconds=latency,
                    features=make_features([1] * batch_size, [past] * batch_size),
                )
            )
    return samples


def prediction_metrics(samples: Sequence[Sample], predicted: Sequence[float]) -> dict[str, Any]:
    if len(samples) != len(predicted):
        raise ValueError("sample/prediction length mismatch")
    if not samples:
        return {"n": 0}
    target = np.asarray([sample.target_seconds for sample in samples], dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    ape = np.abs(estimate - target) / target * 100.0
    return {
        "n": int(len(samples)),
        "mae_ms": float(mean_absolute_error(target, estimate) * 1000.0),
        "rmse_ms": float(mean_squared_error(target, estimate) ** 0.5 * 1000.0),
        "mape_pct": float(np.mean(ape)),
        "median_ape_pct": float(np.median(ape)),
        "p90_ape_pct": float(np.percentile(ape, 90)),
        "mean_target_ms": float(np.mean(target) * 1000.0),
        "mean_prediction_ms": float(np.mean(estimate) * 1000.0),
    }


def grouped_metrics(samples: Sequence[Sample], predicted: Sequence[float]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[tuple[Sample, float]]] = {"all": list(zip(samples, predicted))}
    for stage in ("prefill", "decode"):
        groups[stage] = [pair for pair in groups["all"] if pair[0].stage == stage]
    return {
        name: prediction_metrics(
            [sample for sample, _ in values], [value for _, value in values]
        )
        for name, values in groups.items()
    }


def save_samples(path: Path, samples: Iterable[Sample], predicted: Sequence[float] | None = None) -> None:
    rows = [sample.as_row() for sample in samples]
    if predicted is not None:
        if len(rows) != len(predicted):
            raise ValueError("sample/prediction length mismatch while writing CSV")
        for row, estimate in zip(rows, predicted, strict=True):
            row["predicted_seconds"] = float(estimate)
            target = float(row["target_seconds"])
            row["absolute_error_pct"] = abs(float(estimate) - target) / target * 100.0
    fieldnames = [
        "split",
        "stage",
        "batch_size_source",
        "input_len_source",
        "output_len_source",
        "decode_step",
        "target_seconds",
        *FEATURE_NAMES,
    ]
    if predicted is not None:
        fieldnames.extend(("predicted_seconds", "absolute_error_pct"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_simulator_config(path: Path, model_path: Path) -> None:
    """Write a standalone ML configuration for the local single-A100 target."""
    payload = {
        "platform": {
            "accelerator": {
                "name": "a100_pcie_local",
                "hbm_capacity_gb": 80,
                "hbm_bandwidth_gb": 1935,
                "intra_node_bandwidth_gb": 32,
                "inter_node_bandwidth_gb": 25,
            },
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 1,
        },
        "predictor": {
            "name": "ml",
            "database_path": str(model_path.resolve()),
            "latency_scale": 1.0,
        },
        "scheduler": {
            "tp_size": 1,
            "ep_size": 1,
            "dp_size": 1,
            "backend_name": "sglang",
            "backend_version": "0.5.10",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--validation-results", type=Path, required=True)
    parser.add_argument("--validation-log", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=20260915)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    if args.trees <= 0:
        raise SystemExit("--trees must be positive")
    train = extract_samples(args.train_results, args.train_log, "calibration")
    validation = extract_samples(args.validation_results, args.validation_log, "holdout")
    if not train or not validation:
        raise SystemExit("both calibration and holdout measurement sets must be non-empty")

    x_train = np.asarray([sample.features for sample in train], dtype=float)
    y_train = np.asarray([sample.target_seconds for sample in train], dtype=float)
    x_validation = np.asarray([sample.features for sample in validation], dtype=float)
    estimator = ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=args.random_state,
    )
    estimator.fit(x_train, y_train)
    validation_prediction = estimator.predict(x_validation)
    train_prediction = estimator.predict(x_train)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": estimator,
        "features": FEATURE_NAMES,
        "metadata": {
            "model_name": args.model_name,
            "target": "native_sglang_one_batch_elapsed_seconds",
            "estimator": type(estimator).__name__,
            "random_state": args.random_state,
            "trees": args.trees,
            "calibration_samples": len(train),
            "holdout_samples": len(validation),
            "feature_abi": "MLTimePredictor.FEATURE_NAMES/v1",
            "aiconfigurator_used": False,
        },
    }
    joblib.dump(bundle, args.output_model)
    write_simulator_config(args.simulator_config, args.output_model)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_samples(output_dir / "calibration_samples.csv", train, train_prediction)
    save_samples(output_dir / "holdout_samples.csv", validation, validation_prediction)
    summary = {
        "model_name": args.model_name,
        "predictor": "ml",
        "aiconfigurator_used": False,
        "feature_names": FEATURE_NAMES,
        "model_file": str(args.output_model.resolve()),
        "simulator_config": str(args.simulator_config.resolve()),
        "estimator": type(estimator).__name__,
        "hyperparameters": {
            "n_estimators": args.trees,
            "min_samples_leaf": 2,
            "random_state": args.random_state,
        },
        "calibration_metrics": grouped_metrics(train, train_prediction),
        "holdout_metrics": grouped_metrics(validation, validation_prediction),
        "notes": [
            "Targets are native SGLang one_batch elapsed times with CUDA synchronization.",
            "The holdout grid is measured separately and is excluded from fitting.",
            "A separate bundle is trained for each checkpoint; model architecture is not an MLTimePredictor feature.",
            "Decode past length is input_len + 1 + decode_step, matching the post-prefill scheduler state.",
        ],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
