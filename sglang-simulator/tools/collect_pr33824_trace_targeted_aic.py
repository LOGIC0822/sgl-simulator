#!/usr/bin/env python3
"""Append trace-targeted local A100 Silicon points to an AIC performance DB.

The input plan is generated from *training* traces.  Each target is collected
once with AIC's normal 3 warm-up / 20 timed-iteration operator routine; the
resulting table is therefore a hardware calibration artifact, not a copy of
the real scheduler latencies used for held-out accuracy scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# The existing model-shape helper intentionally lives with the AIC launcher
# scripts.  Make this standalone tool import it without depending on a caller
# to add ``scripts/`` to PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_pr33824_aic_gemm_cases import gemm_shapes


ATTENTION_HEADER = (
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "batch_size",
    "isl",
    "num_heads",
    "num_key_value_heads",
    "head_dim",
    "beam_width",
    "attn_dtype",
    "kv_cache_dtype",
    "step",
    "window_size",
    "latency",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="AIC data/<system>/sglang/<version> directory to extend.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.plan = args.plan.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if not args.plan.is_file():
        parser.error(f"plan does not exist: {args.plan}")
    if not (args.model_path / "config.json").is_file():
        parser.error(f"missing model config: {args.model_path / 'config.json'}")
    if args.manifest.exists():
        parser.error(f"refusing to overwrite manifest: {args.manifest}")
    if args.data_dir.exists() and not args.resume:
        parser.error("--data-dir already exists; pass --resume to append measured keys")
    return args


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"plan must be a JSON object: {path}")
    return value


def checked_topology(plan: dict[str, Any], model_path: Path) -> tuple[int, int, int]:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // heads)
    expected = {
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
    }
    if plan.get("topology") != expected:
        raise ValueError(
            "plan topology does not match model config: "
            f"plan={plan.get('topology')} model={expected}"
        )
    return heads, kv_heads, head_dim


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def existing_attention_keys(path: Path) -> set[tuple[int, int, int, int, int, int]]:
    keys: set[tuple[int, int, int, int, int, int]] = set()
    for row in read_csv(path):
        if row.get("kernel_source") != "triton":
            continue
        try:
            keys.add(
                (
                    int(row["batch_size"]),
                    int(row["isl"]),
                    int(row["num_heads"]),
                    int(row["num_key_value_heads"]),
                    int(row["head_dim"]),
                    int(row["step"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def existing_gemm_keys(path: Path) -> set[tuple[int, int, int]]:
    keys: set[tuple[int, int, int]] = set()
    for row in read_csv(path):
        if row.get("gemm_dtype") != "bfloat16":
            continue
        try:
            keys.add((int(row["m"]), int(row["n"]), int(row["k"])))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def is_awq(config: dict[str, Any]) -> bool:
    quant = config.get("quantization_config")
    return isinstance(quant, dict) and str(quant.get("quant_method", "")).lower() == "awq"


def is_moe(config: dict[str, Any]) -> bool:
    return bool(config.get("num_experts") or config.get("num_local_experts"))


def has_dual_chunk(config: dict[str, Any]) -> bool:
    return config.get("dual_chunk_attention_config") is not None


def install_triton_compat() -> None:
    if os.environ.get("AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND") != "triton":
        raise RuntimeError(
            "trace-targeted AIC collection requires "
            "AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND=triton"
        )
    from run_aiconfigurator_collector_compat import (
        install_pr33824_triton_attention_compat,
    )

    install_pr33824_triton_attention_compat()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = read_json(args.plan)
    heads, kv_heads, head_dim = checked_topology(plan, args.model_path)
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    targets = plan.get("attention_targets")
    gemm_m_values = plan.get("gemm_m_values")
    if not isinstance(targets, list) or not isinstance(gemm_m_values, list):
        raise ValueError("plan is missing attention_targets or gemm_m_values")
    if not targets or not gemm_m_values:
        raise ValueError("plan has no requested attention or GEMM targets")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local AIC Silicon calibration")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    context_path = args.data_dir / "context_attention_perf.txt"
    generation_path = args.data_dir / "generation_attention_perf.txt"
    gemm_path = args.data_dir / "gemm_perf.txt"
    existing_context = existing_attention_keys(context_path)
    existing_generation = existing_attention_keys(generation_path)
    existing_gemm = existing_gemm_keys(gemm_path)

    install_triton_compat()
    from collector.sglang.collect_attn import run_attention_torch
    from collector.sglang.collect_gemm import run_gemm

    attention_collected = {"context": 0, "generation": 0}
    attention_reused = {"context": 0, "generation": 0}
    for index, target in enumerate(targets, start=1):
        if not isinstance(target, dict):
            raise ValueError(f"invalid attention target at index {index - 1}")
        phase = str(target.get("phase"))
        batch_size = int(target["batch_size"])
        input_len = int(target["input_len"])
        if phase not in attention_collected or batch_size <= 0 or input_len <= 0:
            raise ValueError(f"invalid attention target: {target}")
        is_context = phase == "context"
        # AIC's generation table stores isl=1 and the history length in step.
        key = (
            batch_size,
            input_len if is_context else 1,
            heads,
            kv_heads,
            head_dim,
            0 if is_context else input_len,
        )
        existing = existing_context if is_context else existing_generation
        if key in existing:
            attention_reused[phase] += 1
            continue
        output = context_path if is_context else generation_path
        print(
            f"[attention {index}/{len(targets)}] {phase} batch={batch_size} "
            f"input_len={input_len}",
            flush=True,
        )
        run_attention_torch(
            batch_size,
            input_len,
            heads,
            kv_heads,
            head_dim,
            False,
            False,
            is_context,
            perf_filename=str(output),
            device=args.device,
        )
        existing.add(key)
        attention_collected[phase] += 1

    model_shapes = sorted(gemm_shapes(config))
    requested_m = sorted({int(value) for value in gemm_m_values if int(value) > 0})
    gemm_collected = 0
    gemm_reused = 0
    total_gemm = len(model_shapes) * len(requested_m)
    gemm_index = 0
    for m in requested_m:
        for n, k in model_shapes:
            gemm_index += 1
            key = (m, n, k)
            if key in existing_gemm:
                gemm_reused += 1
                continue
            print(
                f"[gemm {gemm_index}/{total_gemm}] bfloat16 m={m} n={n} k={k}",
                flush=True,
            )
            run_gemm("bfloat16", m, n, k, perf_filename=str(gemm_path), device=args.device)
            existing_gemm.add(key)
            gemm_collected += 1

    result = {
        "schema_version": 1,
        "plan": str(args.plan),
        "model_path": str(args.model_path),
        "data_dir": str(args.data_dir),
        "device": args.device,
        "attention_backend": "triton",
        "timing_policy": "attention: 3 warmups/20 timed iterations; GEMM: 3 warmups/6 timed cache-aware op-list iterations; CUDA-event timing and collector CUDA graph replay",
        "attention": {
            "requested": {
                "context": sum(1 for target in targets if target.get("phase") == "context"),
                "generation": sum(1 for target in targets if target.get("phase") == "generation"),
            },
            "new": attention_collected,
            "reused": attention_reused,
        },
        "gemm": {
            "m_values": requested_m,
            "matrix_shapes": [{"n": n, "k": k} for n, k in model_shapes],
            "requested_points": total_gemm,
            "new": gemm_collected,
            "reused": gemm_reused,
        },
        "compatibility": {
            "awq_bf16_proxy": is_awq(config),
            "dual_chunk_attention_triton_proxy": has_dual_chunk(config),
            "moe_trace_targeted_expert_profile_not_collected": is_moe(config),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["attention"]), flush=True)
    print(json.dumps(result["gemm"]), flush=True)


if __name__ == "__main__":
    main()
