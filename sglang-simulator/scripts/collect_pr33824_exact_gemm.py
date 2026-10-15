#!/usr/bin/env python3
"""Measure the exact BF16 GEMM keys needed by the 13-model PR #33824 matrix.

AIConfigurator v0.10 model-case YAML is a selector over its base sweep; it
cannot add arbitrary N,K pairs.  This driver intentionally calls the same
``collector.sglang.collect_gemm.run_gemm`` routine used by AIC, but enumerates
the local model shapes directly.  It resumes from its output file and can seed
already-measured rows from a broader A100 collection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from generate_pr33824_aic_gemm_cases import M_VALUES, gemm_shapes, parse_models


HEADER = [
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "gemm_dtype",
    "m",
    "n",
    "k",
    "latency",
]


def required_keys(checkpoint_root: Path, models: tuple[str, ...]) -> set[tuple[int, int, int]]:
    shapes: set[tuple[int, int]] = set()
    for model in models:
        config_path = checkpoint_root / model / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing model config: {config_path}")
        shapes.update(gemm_shapes(json.loads(config_path.read_text(encoding="utf-8"))))
    return {(m, n, k) for n, k in shapes for m in M_VALUES}


def rows_and_keys(path: Path) -> tuple[list[dict[str, str]], set[tuple[int, int, int]]]:
    if not path.is_file() or path.stat().st_size == 0:
        return [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(int(row["m"]), int(row["n"]), int(row["k"])) for row in rows}
    return rows, keys


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-perf", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/mnt/data/models"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--models",
        help="comma-separated checkpoint directory names; defaults to the historical all-13 set",
    )
    args = parser.parse_args()

    required = required_keys(args.checkpoint_root, parse_models(args.models))
    current_rows, current_keys = rows_and_keys(args.output)
    by_key = {
        (int(row["m"]), int(row["n"]), int(row["k"])): row
        for row in current_rows
        if (int(row["m"]), int(row["n"]), int(row["k"])) in required
    }
    for seed_path in args.seed_perf:
        rows, _ = rows_and_keys(seed_path)
        for row in rows:
            key = (int(row["m"]), int(row["n"]), int(row["k"]))
            if key in required and key not in by_key:
                by_key[key] = row

    # Make the output self-contained before collecting; interrupted runs retain
    # all completed rows and only unmeasured shapes are appended afterwards.
    write_rows(args.output, [by_key[key] for key in sorted(by_key)])
    missing = sorted(required - set(by_key))
    print(f"required={len(required)} reused={len(by_key)} remaining={len(missing)}", flush=True)
    if not missing:
        return

    from collector.sglang.collect_gemm import run_gemm

    for index, (m, n, k) in enumerate(missing, start=1):
        print(f"[{index}/{len(missing)}] bfloat16 m={m} n={n} k={k}", flush=True)
        run_gemm("bfloat16", m, n, k, perf_filename=str(args.output), device=args.device)

    _, completed = rows_and_keys(args.output)
    unresolved = required - completed
    if unresolved:
        raise SystemExit(f"incomplete collection: {len(unresolved)} required GEMM keys are absent")
    print(f"completed={len(completed & required)}", flush=True)


if __name__ == "__main__":
    main()
