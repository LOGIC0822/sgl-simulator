#!/usr/bin/env python3
"""Build a non-leaking regular attention grid for AIC Silicon interpolation.

Trace-targeted operator points faithfully cover the shapes seen while collecting
the training traces, but they are intentionally sparse.  AIC's Silicon lookup
interpolates over ``(num_heads, sequence_length, batch_size)`` for prefill and
``(num_heads, batch_size, sequence_length)`` for decode.  This tool adds a
measured *support* grid from the declared scheduler limits, not from holdout
traces.  The resulting plan is shared by all selected model topologies.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


_CONTEXT_BASE_GRID = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    384,
    512,
    640,
    768,
    896,
    1024,
    1152,
    1280,
    1408,
    1536,
    1792,
    2048,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", action="append", type=Path, required=True)
    parser.add_argument(
        "--training-plan",
        action="append",
        type=Path,
        required=True,
        help="A plan produced by generate_pr33824_trace_aic_plan.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--sequence-grid-step", type=int, default=0,
                        help="Add regularly spaced sequence lengths; 0 preserves the original grid")
    args = parser.parse_args(argv)
    args.model_path = [path.expanduser().resolve() for path in args.model_path]
    args.training_plan = [path.expanduser().resolve() for path in args.training_plan]
    args.output = args.output.expanduser().resolve()
    if len(args.model_path) != len(args.training_plan):
        parser.error("--model-path and --training-plan must have the same count")
    if len(set(args.model_path)) != len(args.model_path):
        parser.error("each --model-path must be supplied once")
    if args.output.exists():
        parser.error(f"refusing to overwrite plan: {args.output}")
    if args.max_context_tokens < 2:
        parser.error("--max-context-tokens must be at least 2")
    if args.max_batch_size < 2:
        parser.error("--max-batch-size must be at least 2")
    if args.sequence_grid_step < 0:
        parser.error("--sequence-grid-step must be nonnegative")
    for model_path, training_plan in zip(args.model_path, args.training_plan, strict=True):
        if not (model_path / "config.json").is_file():
            parser.error(f"missing model config: {model_path / 'config.json'}")
        if not training_plan.is_file():
            parser.error(f"missing training plan: {training_plan}")
    return args


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def model_topology(model_path: Path) -> dict[str, int]:
    config = read_json(model_path / "config.json")
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // heads)
    if heads <= 0 or kv_heads <= 0 or kv_heads > heads or head_dim <= 0:
        raise ValueError(f"invalid attention topology in {model_path / 'config.json'}")
    return {
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
    }


def normalized_kv_heads(heads: int, kv_heads: int) -> int:
    """Match AIC's MHA normalization: n_kv == n is stored as lookup key zero."""
    return 0 if heads == kv_heads else kv_heads


def grid_up_to(values: tuple[int, ...], upper: int) -> list[int]:
    result = {value for value in values if value <= upper}
    result.add(upper)
    return sorted(result)


def generation_upper(context_upper: int) -> int:
    """Cover AIC decode's built-in 0.9--1.1 sequence sampling window."""
    return int(math.ceil(context_upper * 1.1 / 256.0) * 256)


def sequence_grid(upper: int, step: int = 0) -> list[int]:
    if upper < 2 or step < 0:
        raise ValueError("positive sequence bound and nonnegative grid step required")
    values = set(grid_up_to(_CONTEXT_BASE_GRID, upper))
    if step:
        values.update(range(step, upper + 1, step))
    return sorted(values)


def pick_synthetic_neighbor(heads: int, minimum_kv_heads: int, used_heads: set[int]) -> int:
    """Choose one valid, nearby measured head count when a slice has one model."""
    candidates = [heads // 2, heads * 2, heads - 1, heads + 1]
    for candidate in candidates:
        if candidate >= minimum_kv_heads and candidate > 0 and candidate not in used_heads:
            return candidate
    raise ValueError(
        "cannot construct a second AIC interpolation head-count anchor for "
        f"heads={heads}, minimum_kv_heads={minimum_kv_heads}"
    )


def verify_training_plan(plan: dict[str, Any], model_path: Path, plan_path: Path) -> dict[str, Any]:
    purpose = str(plan.get("purpose", "")).lower()
    if "training" not in purpose or "holdout" in purpose:
        raise ValueError(f"plan is not a training-only trace plan: {plan_path}")
    plan_model = Path(str(plan.get("model_path", ""))).expanduser().resolve()
    if plan_model != model_path:
        raise ValueError(f"training plan model mismatch: plan={plan_model} expected={model_path}")
    roots = plan.get("input_data_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"training plan has no input_data_roots: {plan_path}")
    if any("holdout" in str(root).lower() for root in roots):
        raise ValueError(f"training plan names a holdout path: {plan_path}")
    targets = plan.get("attention_targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"training plan has no attention targets: {plan_path}")
    schedule_rows = plan.get("schedule_rows")
    if not isinstance(schedule_rows, dict) or int(schedule_rows.get("total", 0)) <= 0:
        raise ValueError(f"training plan has no schedule rows: {plan_path}")
    return {
        "schedule_rows": int(schedule_rows["total"]),
        "context_targets": sum(1 for target in targets if target.get("phase") == "context"),
        "generation_targets": sum(1 for target in targets if target.get("phase") == "generation"),
        "observed_batch_sizes": sorted({int(target["batch_size"]) for target in targets}),
        "observed_input_length_min": min(int(target["input_len"]) for target in targets),
        "observed_input_length_max": max(int(target["input_len"]) for target in targets),
    }


def make_topology_slices(entries: list[tuple[Path, dict[str, int]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for model_path, topology in entries:
        heads = topology["num_attention_heads"]
        kv_heads = topology["num_key_value_heads"]
        lookup = normalized_kv_heads(heads, kv_heads)
        grouped[(lookup, topology["head_dim"])].append(
            {
                "num_heads": heads,
                "num_key_value_heads": kv_heads,
                "origin": "model",
                "models": [model_path.name],
            }
        )

    slices: list[dict[str, Any]] = []
    for (lookup, head_dim), raw_configs in sorted(grouped.items()):
        by_shape: dict[tuple[int, int], dict[str, Any]] = {}
        for raw in raw_configs:
            key = (int(raw["num_heads"]), int(raw["num_key_value_heads"]))
            item = by_shape.setdefault(
                key,
                {
                    "num_heads": key[0],
                    "num_key_value_heads": key[1],
                    "origin": "model",
                    "models": [],
                },
            )
            item["models"].extend(raw["models"])
        configs = list(by_shape.values())
        used_heads = {int(config["num_heads"]) for config in configs}
        if len(used_heads) < 2:
            original = configs[0]
            original_heads = int(original["num_heads"])
            original_kv = int(original["num_key_value_heads"])
            # For MHA the synthetic point changes n and n_kv together, so it
            # is not constrained by the original n_kv.  XQA/GQA must retain
            # its fixed n_kv slice.
            synthetic_heads = pick_synthetic_neighbor(
                original_heads,
                1 if lookup == 0 else original_kv,
                used_heads,
            )
            synthetic_kv = synthetic_heads if lookup == 0 else original_kv
            configs.append(
                {
                    "num_heads": synthetic_heads,
                    "num_key_value_heads": synthetic_kv,
                    "origin": "synthetic_interpolation_neighbor",
                    "models": [],
                    "anchor_for_num_heads": original_heads,
                }
            )
        configs.sort(key=lambda value: int(value["num_heads"]))
        for config in configs:
            actual_lookup = normalized_kv_heads(
                int(config["num_heads"]), int(config["num_key_value_heads"])
            )
            if actual_lookup != lookup:
                raise AssertionError(f"normalization mismatch: {config} lookup={lookup}")
        slices.append(
            {
                "aic_num_key_value_heads_lookup": lookup,
                "head_dim": head_dim,
                "head_configs": configs,
            }
        )
    return slices


def expected_target_counts(
    topology_slices: list[dict[str, Any]], batch_sizes: list[int], context_lengths: list[int], generation_lengths: list[int]
) -> dict[str, int]:
    head_configs = sum(len(item["head_configs"]) for item in topology_slices)
    context = head_configs * len(batch_sizes) * len(context_lengths)
    generation = head_configs * len(batch_sizes) * len(generation_lengths)
    return {"context": context, "generation": generation, "total": context + generation}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    entries: list[tuple[Path, dict[str, int]]] = []
    training_inputs: list[dict[str, Any]] = []
    for model_path, plan_path in zip(args.model_path, args.training_plan, strict=True):
        topology = model_topology(model_path)
        plan = read_json(plan_path)
        plan_topology = plan.get("topology")
        if plan_topology != topology:
            raise ValueError(f"training plan topology mismatch: {plan_path}")
        training_summary = verify_training_plan(plan, model_path, plan_path)
        entries.append((model_path, topology))
        training_inputs.append(
            {
                "model": model_path.name,
                "model_path": str(model_path),
                "training_plan": str(plan_path),
                **training_summary,
            }
        )

    context_lengths = sequence_grid(args.max_context_tokens, args.sequence_grid_step)
    generation_limit = generation_upper(args.max_context_tokens)
    generation_lengths = sequence_grid(generation_limit, args.sequence_grid_step)
    batch_sizes = list(range(1, args.max_batch_size + 1))
    topology_slices = make_topology_slices(entries)
    counts = expected_target_counts(
        topology_slices, batch_sizes, context_lengths, generation_lengths
    )
    plan = {
        "schema_version": 1,
        "purpose": "AIC Silicon regular interpolation-support attention grid",
        "selection_policy": {
            "source": "training trace plans plus declared scheduler limits only",
            "holdout_policy": "holdout ScheduleBatch rows and holdout measured latencies are not read",
            "why": "AIC Silicon requires support along every three-dimensional interpolation axis for unseen batch and sequence combinations",
        },
        "declared_workload_bounds": {
            "max_context_tokens": args.max_context_tokens,
            "max_running_requests": args.max_batch_size,
            "generation_query_coverage_upper": generation_limit,
            "generation_coverage_reason": "AIC samples 0.9--1.1 times each decode sequence length",
        },
        "grid": {
            "batch_sizes": batch_sizes,
            "context_input_lengths": context_lengths,
            "generation_input_lengths": generation_lengths,
            "additional_sequence_grid_step": args.sequence_grid_step,
        },
        "training_inputs": training_inputs,
        "topology_slices": topology_slices,
        "expected_attention_targets": counts,
        "timing_policy": "collected later with AIC default: 3 warmups and 20 timed iterations per operator point",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        "models={} slices={} batch_grid={} context_grid={} generation_grid={} targets={} out={}".format(
            len(entries),
            len(topology_slices),
            len(batch_sizes),
            len(context_lengths),
            len(generation_lengths),
            counts["total"],
            args.output,
        )
    )


if __name__ == "__main__":
    main()
