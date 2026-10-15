#!/usr/bin/env python3
"""Create an AIC scoring DB whose relevant context slices are rectangular.

AIC 0.10's SILICON context lookup chooses the nearest sequence coordinates
before it interpolates batch size.  Training-trace rows are intentionally
scattered, so a nearby one-batch row can shadow an otherwise complete regular
grid.  This builder leaves the source database untouched, copies every other
operator table verbatim, and retains only the directly measured regular-grid
rows for context-attention slices used by the selected models.

The support plan itself is generated from training-only plans and declared
workload bounds.  This builder never reads a holdout trace or a holdout latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


CONTEXT_RELATIVE_PATH = Path("data/a100_pcie_local/sglang/0.5.10/context_attention_perf.txt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-systems", type=Path, required=True)
    parser.add_argument("--support-plan", type=Path, required=True)
    parser.add_argument("--source-sim-config", type=Path, required=True)
    parser.add_argument("--output-systems", type=Path, required=True)
    parser.add_argument("--output-sim-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in (
        "source_systems",
        "support_plan",
        "source_sim_config",
        "output_systems",
        "output_sim_config",
        "manifest",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.source_systems.is_dir():
        parser.error(f"source systems directory does not exist: {args.source_systems}")
    if not args.support_plan.is_file():
        parser.error(f"support plan does not exist: {args.support_plan}")
    if not args.source_sim_config.is_file():
        parser.error(f"source simulator config does not exist: {args.source_sim_config}")
    source_table = args.source_systems / CONTEXT_RELATIVE_PATH
    if not source_table.is_file():
        parser.error(f"source context table does not exist: {source_table}")
    for output in (args.output_systems, args.output_sim_config, args.manifest):
        if output.exists():
            parser.error(f"refusing to overwrite output: {output}")
    return args


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def support_keys(plan: dict[str, Any]) -> tuple[set[tuple[int, int, int, int, int]], set[tuple[int, int, int]]]:
    if plan.get("purpose") != "AIC Silicon regular interpolation-support attention grid":
        raise ValueError("unexpected support-plan purpose")
    policy = plan.get("selection_policy")
    if not isinstance(policy, dict) or "holdout" not in str(policy.get("holdout_policy", "")).lower():
        raise ValueError("support plan does not declare a no-holdout policy")
    grid = plan.get("grid")
    slices = plan.get("topology_slices")
    if not isinstance(grid, dict) or not isinstance(slices, list) or not slices:
        raise ValueError("support plan is missing grid or topology_slices")
    batches = {int(value) for value in grid.get("batch_sizes", [])}
    lengths = {int(value) for value in grid.get("context_input_lengths", [])}
    if not batches or not lengths:
        raise ValueError("support plan has an empty context grid")
    keys: set[tuple[int, int, int, int, int]] = set()
    configs: set[tuple[int, int, int]] = set()
    for item in slices:
        if not isinstance(item, dict):
            raise ValueError(f"invalid topology slice: {item!r}")
        head_dim = int(item.get("head_dim", 0))
        head_configs = item.get("head_configs")
        if head_dim <= 0 or not isinstance(head_configs, list):
            raise ValueError(f"invalid topology slice: {item!r}")
        for config in head_configs:
            if not isinstance(config, dict):
                raise ValueError(f"invalid head config: {config!r}")
            heads = int(config.get("num_heads", 0))
            kv_heads = int(config.get("num_key_value_heads", 0))
            if heads <= 0 or kv_heads <= 0 or kv_heads > heads:
                raise ValueError(f"invalid head config: {config!r}")
            configs.add((heads, kv_heads, head_dim))
            keys.update((batch, length, heads, kv_heads, head_dim) for batch in batches for length in lengths)
    expected = plan.get("expected_attention_targets", {})
    if len(keys) != int(expected.get("context", -1)):
        raise ValueError(f"support plan context count mismatch: keys={len(keys)} expected={expected.get('context')}")
    return keys, configs


def row_key(row: dict[str, str]) -> tuple[int, int, int, int, int]:
    return (
        int(row["batch_size"]),
        int(row["isl"]),
        int(row["num_heads"]),
        int(row["num_key_value_heads"]),
        int(row["head_dim"]),
    )


def row_config(row: dict[str, str]) -> tuple[int, int, int]:
    return (int(row["num_heads"]), int(row["num_key_value_heads"]), int(row["head_dim"]))


def is_direct_regular_context_row(row: dict[str, str], keys: set[tuple[int, int, int, int, int]]) -> bool:
    return (
        row.get("kernel_source") == "triton"
        and row.get("attn_dtype") == "bfloat16"
        and row.get("kv_cache_dtype") == "bfloat16"
        and int(row.get("window_size", "0")) == 0
        and int(row.get("step", "0")) == 0
        and row_key(row) in keys
    )


def filter_context_rows(
    rows: list[dict[str, str]], keys: set[tuple[int, int, int, int, int]], configs: set[tuple[int, int, int]]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    retained: list[dict[str, str]] = []
    seen_support: set[tuple[int, int, int, int, int]] = set()
    removed_sparse = 0
    retained_other = 0
    for row in rows:
        try:
            config = row_config(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid context row: {row!r}") from exc
        if config not in configs:
            retained.append(row)
            retained_other += 1
            continue
        if not is_direct_regular_context_row(row, keys):
            removed_sparse += 1
            continue
        key = row_key(row)
        if key in seen_support:
            # The AIC loader is first-wins for a duplicate coordinate. Keep
            # exactly that one row in the isolated table as well.
            continue
        seen_support.add(key)
        retained.append(row)
    missing = keys - seen_support
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(f"regular context table is incomplete; missing {len(missing)} keys, sample={sample}")
    return retained, {
        "source_rows": len(rows),
        "retained_rows": len(retained),
        "retained_regular_context_rows": len(seen_support),
        "retained_other_rows": retained_other,
        "removed_sparse_or_incompatible_rows": removed_sparse,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = read_json(args.support_plan)
    keys, configs = support_keys(plan)
    source_table = args.source_systems / CONTEXT_RELATIVE_PATH
    with source_table.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"context table has no header: {source_table}")
        rows = list(reader)
    filtered_rows, counts = filter_context_rows(rows, keys, configs)

    shutil.copytree(args.source_systems, args.output_systems)
    output_table = args.output_systems / CONTEXT_RELATIVE_PATH
    with output_table.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(filtered_rows)

    sim_config = read_json(args.source_sim_config)
    predictor = sim_config.get("predictor")
    if not isinstance(predictor, dict) or predictor.get("name") != "aiconfigurator":
        raise ValueError("source simulator config is not an AIC config")
    predictor["database_path"] = str(args.output_systems)
    sim_config["predictor"] = predictor
    args.output_sim_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_sim_config.write_text(json.dumps(sim_config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "purpose": "AIC Silicon scoring database with regularized context-attention slices",
        "source_systems": str(args.source_systems),
        "output_systems": str(args.output_systems),
        "source_sim_config": str(args.source_sim_config),
        "output_sim_config": str(args.output_sim_config),
        "support_plan": str(args.support_plan),
        "selection_policy": plan["selection_policy"],
        "context_table": str(output_table),
        "regular_context_keys": len(keys),
        "regular_context_configs": [
            {"num_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim}
            for heads, kv_heads, head_dim in sorted(configs)
        ],
        "row_counts": counts,
        "other_operator_tables": "copied verbatim from source AIC systems directory",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"regular_context_keys": len(keys), **counts}), flush=True)


if __name__ == "__main__":
    main()
