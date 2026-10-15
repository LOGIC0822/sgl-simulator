#!/usr/bin/env python3
"""Append a regular, measured AIC attention interpolation-support grid.

The plan is generated from training-only trace plans and fixed scheduler
bounds.  Measurements are direct local-A100 Triton attention timings, not
synthetic latency values.  Existing matching table rows are kept and counted
as reused so this tool can resume after an interruption.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.plan = args.plan.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if not args.plan.is_file():
        parser.error(f"plan does not exist: {args.plan}")
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


def normalized_kv_heads(heads: int, kv_heads: int) -> int:
    return 0 if heads == kv_heads else kv_heads


def checked_plan(plan: dict[str, Any]) -> tuple[list[int], dict[str, list[int]], list[dict[str, Any]]]:
    if plan.get("purpose") != "AIC Silicon regular interpolation-support attention grid":
        raise ValueError("unexpected interpolation support plan purpose")
    policy = plan.get("selection_policy")
    if not isinstance(policy, dict) or "holdout" not in str(policy.get("holdout_policy", "")).lower():
        raise ValueError("plan does not record an explicit no-holdout policy")
    grid = plan.get("grid")
    slices = plan.get("topology_slices")
    if not isinstance(grid, dict) or not isinstance(slices, list) or not slices:
        raise ValueError("plan is missing grid or topology_slices")
    batch_sizes = [int(value) for value in grid.get("batch_sizes", [])]
    lengths = {
        "context": [int(value) for value in grid.get("context_input_lengths", [])],
        "generation": [int(value) for value in grid.get("generation_input_lengths", [])],
    }
    if len(batch_sizes) < 2 or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch support grid must contain at least two positive values")
    if any(len(values) < 2 or any(value <= 0 for value in values) for values in lengths.values()):
        raise ValueError("each sequence support grid must contain at least two positive values")
    for item in slices:
        if not isinstance(item, dict):
            raise ValueError(f"invalid topology slice: {item!r}")
        lookup = int(item.get("aic_num_key_value_heads_lookup", -1))
        head_dim = int(item.get("head_dim", 0))
        configs = item.get("head_configs")
        if lookup < 0 or head_dim <= 0 or not isinstance(configs, list) or len(configs) < 2:
            raise ValueError(f"invalid topology slice: {item!r}")
        heads = set()
        for config in configs:
            if not isinstance(config, dict):
                raise ValueError(f"invalid head config: {config!r}")
            n = int(config.get("num_heads", 0))
            n_kv = int(config.get("num_key_value_heads", 0))
            if n <= 0 or n_kv <= 0 or n_kv > n or normalized_kv_heads(n, n_kv) != lookup:
                raise ValueError(f"invalid AIC head config: {config!r}")
            heads.add(n)
        if len(heads) < 2:
            raise ValueError(f"slice does not vary across AIC num_heads: {item!r}")
    return batch_sizes, lengths, slices


def iter_targets(
    batch_sizes: list[int], lengths: dict[str, list[int]], slices: list[dict[str, Any]]
) -> Iterator[tuple[str, int, int, int, int, int, str]]:
    for item in slices:
        lookup = int(item["aic_num_key_value_heads_lookup"])
        head_dim = int(item["head_dim"])
        slice_label = f"nkv_lookup={lookup},head_dim={head_dim}"
        for config in item["head_configs"]:
            heads = int(config["num_heads"])
            kv_heads = int(config["num_key_value_heads"])
            for phase in ("context", "generation"):
                for batch_size in batch_sizes:
                    for input_len in lengths[phase]:
                        yield phase, batch_size, input_len, heads, kv_heads, head_dim, slice_label


def attention_key(
    phase: str, batch_size: int, input_len: int, heads: int, kv_heads: int, head_dim: int
) -> tuple[int, int, int, int, int, int]:
    return (
        batch_size,
        input_len if phase == "context" else 1,
        heads,
        kv_heads,
        head_dim,
        0 if phase == "context" else input_len,
    )


def install_triton_compat() -> None:
    if os.environ.get("AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND") != "triton":
        raise RuntimeError(
            "interpolation support collection requires "
            "AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND=triton"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_aiconfigurator_collector_compat import install_pr33824_triton_attention_compat

    install_pr33824_triton_attention_compat()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = read_json(args.plan)
    batch_sizes, lengths, slices = checked_plan(plan)
    targets = list(iter_targets(batch_sizes, lengths, slices))
    expected = plan.get("expected_attention_targets", {})
    if int(expected.get("total", -1)) != len(targets):
        raise ValueError(f"plan expected {expected.get('total')} targets but expands to {len(targets)}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local AIC Silicon calibration")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    context_path = args.data_dir / "context_attention_perf.txt"
    generation_path = args.data_dir / "generation_attention_perf.txt"
    existing = {
        "context": existing_attention_keys(context_path),
        "generation": existing_attention_keys(generation_path),
    }
    install_triton_compat()
    from collector.sglang.collect_attn import run_attention_torch

    collected = {"context": 0, "generation": 0}
    reused = {"context": 0, "generation": 0}
    per_slice: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {"new": {"context": 0, "generation": 0}, "reused": {"context": 0, "generation": 0}}
    )
    for index, (phase, batch_size, input_len, heads, kv_heads, head_dim, slice_label) in enumerate(targets, start=1):
        key = attention_key(phase, batch_size, input_len, heads, kv_heads, head_dim)
        if key in existing[phase]:
            reused[phase] += 1
            per_slice[slice_label]["reused"][phase] += 1
            continue
        print(
            f"[support {index}/{len(targets)}] {phase} batch={batch_size} input_len={input_len} "
            f"heads={heads} kv_heads={kv_heads} head_dim={head_dim}",
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
            phase == "context",
            perf_filename=str(context_path if phase == "context" else generation_path),
            device=args.device,
        )
        existing[phase].add(key)
        collected[phase] += 1
        per_slice[slice_label]["new"][phase] += 1

    result = {
        "schema_version": 1,
        "plan": str(args.plan),
        "data_dir": str(args.data_dir),
        "device": args.device,
        "attention_backend": "triton",
        "timing_policy": "AIC collector default: 3 warmups and 20 timed iterations per operator point",
        "selection_policy": plan["selection_policy"],
        "declared_workload_bounds": plan["declared_workload_bounds"],
        "grid": plan["grid"],
        "expected": expected,
        "new": collected,
        "reused": reused,
        "per_slice": dict(per_slice),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"new": collected, "reused": reused, "total": len(targets)}), flush=True)


if __name__ == "__main__":
    main()
