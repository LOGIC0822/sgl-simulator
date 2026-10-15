#!/usr/bin/env python3
"""Teacher-force an AIC Silicon predictor on real ScheduleBatch observations.

Unlike end-to-end replay, this keeps every real scheduler batch fixed and
therefore measures the AIC static operator estimate against the exact
device-synchronized iteration that produced it.  The held-out trace is never
used to choose AIC calibration points; it is only read here for scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sglang.srt.configs.model_config import ModelConfig

from sglang_simulator.simulation.sglang.utils import resolve_model_info
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.data_type import DataType
from sglang_simulator.time_predictor.aiconfigurator import AIConfiguratorTimePredictor
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest


class ObservedScheduleBatch(ScheduleBatch):
    """Preserve the SGLang forward-mode label when an extend batch has one token."""

    def __init__(self, forward_mode: int, reqs: list[ScheduleRequest]) -> None:
        super().__init__(reqs=reqs)
        self.forward_mode = forward_mode

    def is_decode(self) -> bool:
        return self.forward_mode == 2


def parse_modes(value: str) -> set[int]:
    modes = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not modes or not modes <= {1, 2}:
        raise argparse.ArgumentTypeError("--forward-modes must be a nonempty subset of 1,2")
    return modes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", action="append", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--schedule-glob", default="TP0*.schedule_batch.jsonl")
    parser.add_argument("--forward-modes", default="1,2")
    args = parser.parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    args.sim_config = args.sim_config.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.model_path.is_dir():
        parser.error(f"model directory does not exist: {args.model_path}")
    if not args.sim_config.is_file():
        parser.error(f"simulator config does not exist: {args.sim_config}")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.out_dir}")
    parse_modes(args.forward_modes)
    return args


def schedule_files(data_roots: list[str], pattern: str) -> list[Path]:
    files: set[Path] = set()
    for raw_root in data_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"data root does not exist: {root}")
        if root.is_file():
            files.add(root)
        else:
            files.update(path.resolve() for path in root.rglob(pattern))
    result = sorted(files)
    if not result:
        raise RuntimeError("no ScheduleBatch JSONL files found")
    return result


def read_rows(data_roots: list[str], pattern: str, modes: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schedule_path in schedule_files(data_roots, pattern):
        with schedule_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {schedule_path}:{line_number}: {exc}"
                    ) from exc
                mode = value.get("forward_mode")
                if mode not in modes:
                    continue
                requests = value.get("request_infos") or []
                if not requests:
                    continue
                try:
                    actual = float(value["iter_latency"])
                    extend = [int(item["extend_input_len"]) for item in requests]
                    past = [int(item["prefix_indices_len"]) for item in requests]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid ScheduleBatch at {schedule_path}:{line_number}: {exc}"
                    ) from exc
                if not math.isfinite(actual) or actual <= 0:
                    continue
                if any(length < 0 for length in [*extend, *past]):
                    raise ValueError(f"negative token length at {schedule_path}:{line_number}")
                rows.append(
                    {
                        "case": schedule_path.parent.name,
                        "schedule_path": str(schedule_path),
                        "line_number": line_number,
                        "forward_mode": int(mode),
                        "extend": extend,
                        "past": past,
                        "batch_size": len(extend),
                        "sum_extend": sum(extend),
                        "sum_past": sum(past),
                        "actual_seconds": actual,
                    }
                )
    if not rows:
        raise RuntimeError("no usable synchronized ScheduleBatch rows found")
    return rows


def load_predictor(config_path: Path, model_path: Path) -> AIConfiguratorTimePredictor:
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("simulator config must be a JSON object")
    predictor_config = raw_config.get("predictor")
    platform = raw_config.get("platform")
    scheduler = raw_config.get("scheduler", {})
    if not isinstance(predictor_config, dict) or predictor_config.get("name") != "aiconfigurator":
        raise ValueError("--sim-config must select predictor.name=aiconfigurator")
    if not isinstance(platform, dict) or not isinstance(platform.get("accelerator"), dict):
        raise ValueError("simulator config is missing platform.accelerator")
    database_path = predictor_config.get("database_path")
    if not database_path:
        raise ValueError("simulator config is missing predictor.database_path")
    accelerator_raw = platform["accelerator"]
    accelerator = AcceleratorInfo.from_dict(
        {
            "name": str(accelerator_raw["name"]),
            "vendor": str(accelerator_raw.get("vendor", "NVIDIA")),
            "hbm_capacity_gb": int(accelerator_raw["hbm_capacity_gb"]),
            "hbm_bandwidth_gb": int(accelerator_raw["hbm_bandwidth_gb"]),
            "intra_node_bandwidth_gb": int(accelerator_raw.get("intra_node_bandwidth_gb", 32)),
            "inter_node_bandwidth_gb": int(accelerator_raw.get("inter_node_bandwidth_gb", 25)),
        }
    )
    scheduler_config = SchedulerConfig(
        data_type=DataType.BF16,
        kv_cache_data_type=DataType.BF16,
        tp_size=int(scheduler.get("tp_size", 1)),
        ep_size=int(scheduler.get("ep_size", 1)),
        dp_size=int(scheduler.get("dp_size", 1)),
        pp_size=int(scheduler.get("pp_size", 1)),
        backend_name=str(scheduler.get("backend_name", "sglang")),
        backend_version=str(scheduler.get("backend_version", "0.5.10")),
    )
    model_config = ModelConfig(model_path=str(model_path), trust_remote_code=False)
    model = resolve_model_info(model_config)
    return AIConfiguratorTimePredictor(
        model,
        hw=accelerator,
        config=scheduler_config,
        database_path=str(Path(database_path).expanduser().resolve()),
        database_mode=str(predictor_config.get("database_mode", "SILICON")),
    )


def cache_key(row: dict[str, Any]) -> tuple[Any, ...]:
    batch_size = int(row["batch_size"])
    extend = list(row["extend"])
    past = list(row["past"])
    mean_past = sum(past) // batch_size
    if row["forward_mode"] == 2:
        return (2, batch_size, mean_past)
    mean_extend = sum(extend) // batch_size
    avg_flops = (mean_past + mean_past + mean_extend) * mean_extend / 2 * batch_size
    actual_flops = sum(
        (past_length + past_length + extend_length) * extend_length / 2
        for extend_length, past_length in zip(extend, past)
    )
    correction = actual_flops / avg_flops if avg_flops > 0 else 1.0
    return (
        1,
        batch_size,
        mean_past + mean_extend,
        mean_past,
        round(correction, 12) if correction >= 0.4 else None,
    )


def predict_rows(rows: list[dict[str, Any]], predictor: AIConfiguratorTimePredictor) -> tuple[list[dict[str, Any]], int]:
    cache: dict[tuple[Any, ...], float] = {}
    scored: list[dict[str, Any]] = []
    for row in rows:
        key = cache_key(row)
        prediction = cache.get(key)
        if prediction is None:
            batch = ObservedScheduleBatch(
                int(row["forward_mode"]),
                [
                    ScheduleRequest(extend_length=extend, past_kv_length=past)
                    for extend, past in zip(row["extend"], row["past"])
                ],
            )
            prediction = float(predictor.predict_infer_time(batch))
            if not math.isfinite(prediction) or prediction < 0:
                raise RuntimeError(f"AIC returned invalid latency for cache key {key}: {prediction}")
            cache[key] = prediction
        actual = float(row["actual_seconds"])
        signed = (prediction - actual) / actual * 100.0
        scored.append(
            {
                **{name: value for name, value in row.items() if name not in {"extend", "past"}},
                "predicted_seconds": prediction,
                "signed_error_pct": signed,
                "absolute_percentage_error_pct": abs(signed),
            }
        )
    return scored, len(cache)


def summary_row(group: str, value: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    actual = np.asarray([row["actual_seconds"] for row in selected], dtype=np.float64)
    predicted = np.asarray([row["predicted_seconds"] for row in selected], dtype=np.float64)
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [summary_row("all", "all", rows)]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "case": defaultdict(list),
        "forward_mode": defaultdict(list),
        "case_forward_mode": defaultdict(list),
    }
    for row in rows:
        grouped["case"][row["case"]].append(row)
        grouped["forward_mode"][str(row["forward_mode"])].append(row)
        grouped["case_forward_mode"][f"{row['case']}:mode{row['forward_mode']}"] .append(row)
    for group, values in grouped.items():
        result.extend(summary_row(group, value, selected) for value, selected in sorted(values.items()))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    modes = parse_modes(args.forward_modes)
    rows = read_rows(args.data_root, args.schedule_glob, modes)
    predictor = load_predictor(args.sim_config, args.model_path)
    scored, unique_queries = predict_rows(rows, predictor)
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
        "# AIC Silicon scheduler-batch scoring\n\n"
        "Teacher-forced comparison against real device-synchronized ScheduleBatch rows.\n\n"
        f"- sim config: `{args.sim_config}`\n"
        f"- model: `{args.model_path}`\n"
        f"- forward modes: `{sorted(modes)}`\n"
        f"- batches: {overall['n']}\n"
        f"- unique AIC static queries: {unique_queries}\n"
        f"- MAPE: {overall['mape_pct']:.3f}%\n"
        f"- MAE: {overall['mae_ms']:.3f} ms\n",
        encoding="utf-8",
    )
    print(
        f"batches={overall['n']} unique_queries={unique_queries} "
        f"MAPE={overall['mape_pct']:.3f}% MAE={overall['mae_ms']:.3f}ms out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
