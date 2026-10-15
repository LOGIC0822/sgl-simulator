#!/usr/bin/env python3
"""Query the configured AIC Silicon table for one dense-model forward step.

This is intentionally a direct predictor query rather than a CPU simulator
request: its two shapes match ``sglang.benchmark.one_batch`` exactly:

* prefill: one request with ``input_len`` new tokens and no prefix;
* decode: the first generated token after that prefill (AIC ``osl=2``).

The caller is responsible for running this under the same checked-out SGLang
source and AIConfigurator database used by the real benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sglang.srt.configs.model_config import ModelConfig

from sglang_simulator.simulation.sglang.utils import resolve_model_info
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator.base import AcceleratorInfo
from sglang_simulator.spec.data_type import DataType
from sglang_simulator.time_predictor.aiconfigurator import AIConfiguratorTimePredictor
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read simulator config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Simulator config {path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input_len <= 1 or args.batch_size <= 0:
        raise SystemExit("--input-len must be > 1 and --batch-size must be positive")
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model_path}")

    config_path = args.sim_config.resolve()
    raw_config = load_config(config_path)
    platform = raw_config.get("platform", {})
    predictor_config = raw_config.get("predictor", {})
    scheduler = raw_config.get("scheduler", {})
    accelerator_config = platform.get("accelerator", {})
    database_path = predictor_config.get("database_path")
    if not isinstance(accelerator_config, dict) or not database_path:
        raise SystemExit("Simulator config is missing platform.accelerator or predictor.database_path")

    model_config = ModelConfig(
        model_path=str(args.model_path.resolve()), trust_remote_code=False
    )
    model = resolve_model_info(model_config)
    accelerator = AcceleratorInfo.from_dict(
        {
            "name": str(accelerator_config["name"]),
            "vendor": str(accelerator_config.get("vendor", "NVIDIA")),
            "hbm_capacity_gb": int(accelerator_config["hbm_capacity_gb"]),
            "hbm_bandwidth_gb": int(accelerator_config["hbm_bandwidth_gb"]),
            "intra_node_bandwidth_gb": int(
                accelerator_config.get("intra_node_bandwidth_gb", 32)
            ),
            "inter_node_bandwidth_gb": int(
                accelerator_config.get("inter_node_bandwidth_gb", 25)
            ),
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
    predictor = AIConfiguratorTimePredictor(
        model,
        hw=accelerator,
        config=scheduler_config,
        database_path=str(Path(database_path).resolve()),
        database_mode=str(predictor_config.get("database_mode", "SILICON")),
    )

    prefill_batch = ScheduleBatch(
        [
            ScheduleRequest(extend_length=args.input_len, past_kv_length=0)
            for _ in range(args.batch_size)
        ]
    )
    # ``one_batch`` first decode occurs after the prefill and its sampled token;
    # AIC's ``osl=2`` convention internally represents that one decode step.
    decode_batch = ScheduleBatch(
        [
            ScheduleRequest(extend_length=1, past_kv_length=args.input_len)
            for _ in range(args.batch_size)
        ]
    )

    phases: dict[str, dict[str, Any]] = {}
    for name, batch in (("prefill", prefill_batch), ("decode", decode_batch)):
        breakdown = predictor.predict_infer_latency_dict(batch)
        predicted_ms = predictor.predict_infer_time(batch) * 1e3
        phases[name] = {
            "predicted_ms": predicted_ms,
            "breakdown_ms": {key: float(value) for key, value in breakdown.items()},
            "request_shape": batch.request_info(),
        }

    result = {
        "schema_version": 1,
        "measurement": "AIC Silicon static operator composition; CUDA Graph is not replayed",
        "model_path": str(args.model_path.resolve()),
        "simulator_config": str(config_path),
        "database_path": str(Path(database_path).resolve()),
        "database_mode": str(predictor_config.get("database_mode", "SILICON")),
        "backend": scheduler_config.backend_name,
        "backend_version": scheduler_config.backend_version,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "phases": phases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: phase["predicted_ms"] for name, phase in phases.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
