#!/usr/bin/env python3
"""Derive non-leaking AIC operator targets from collected training traces.

The plan contains only shapes observed in real *training* ScheduleBatch rows.
It deliberately does not inspect the independent holdout traces.  AIC models a
batch through a compact static representation, so the conversion below mirrors
``AIConfiguratorTimePredictor`` rather than claiming to preserve every ragged
request length in the original scheduler batch.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="A training collection root or a directory containing ScheduleBatch JSONL files.",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument(
        "--max-attention-targets-per-phase",
        type=int,
        default=0,
        help="Keep only the most frequent targets per phase; 0 keeps every observed target.",
    )
    parser.add_argument(
        "--max-gemm-m-values",
        type=int,
        default=0,
        help="Keep only the most frequent GEMM M values; 0 keeps every observed value.",
    )
    args = parser.parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.summary_csv is not None:
        args.summary_csv = args.summary_csv.expanduser().resolve()
    if not args.model_path.is_dir():
        parser.error(f"model directory does not exist: {args.model_path}")
    if not (args.model_path / "config.json").is_file():
        parser.error(f"missing model config: {args.model_path / 'config.json'}")
    if args.output.exists():
        parser.error(f"refusing to overwrite plan: {args.output}")
    if args.summary_csv is not None and args.summary_csv.exists():
        parser.error(f"refusing to overwrite summary CSV: {args.summary_csv}")
    if args.max_attention_targets_per_phase < 0:
        parser.error("--max-attention-targets-per-phase must be non-negative")
    if args.max_gemm_m_values < 0:
        parser.error("--max-gemm-m-values must be non-negative")
    return args


def schedule_files(data_roots: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw_root in data_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"training data root does not exist: {root}")
        if root.is_file():
            files.add(root)
        else:
            files.update(path.resolve() for path in root.rglob("TP0*.schedule_batch.jsonl"))
    result = sorted(files)
    if not result:
        raise RuntimeError("no ScheduleBatch JSONL files found below --data-root")
    return result


def topology(model_path: Path) -> dict[str, int]:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    return {
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
    }


def parse_batch(
    value: dict[str, Any], source: Path, line_number: int
) -> tuple[int, list[int], list[int]] | None:
    mode = value.get("forward_mode")
    if mode not in (1, 2):
        return None
    requests = value.get("request_infos") or []
    if not requests:
        return None
    try:
        extend = [int(item["extend_input_len"]) for item in requests]
        past = [int(item["prefix_indices_len"]) for item in requests]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid request_infos at {source}:{line_number}: {exc}") from exc
    if any(value < 0 for value in [*extend, *past]):
        raise ValueError(f"negative token length at {source}:{line_number}")
    return int(mode), extend, past


def effective_aic_shape(mode: int, extend: list[int], past: list[int]) -> tuple[str, int, int]:
    """Match the shape reduction performed by AIConfiguratorTimePredictor."""
    batch_size = len(extend)
    mean_past = sum(past) // batch_size
    if mode == 2:
        # AIC generation uses RuntimeConfig(batch_size, isl=mean(past), osl=2).
        return "generation", batch_size, mean_past
    mean_extend = sum(extend) // batch_size
    # AIC context uses RuntimeConfig(batch_size, isl=mean(past)+mean(extend),
    # prefix=mean(past), osl=1).
    return "context", batch_size, mean_past + mean_extend


def target_gemm_m_values(mode: int, extend: list[int], past: list[int]) -> set[int]:
    """Return both physical and AIC-reduced GEMM M values for safe coverage."""
    batch_size = len(extend)
    if mode == 2:
        return {batch_size}
    mean_extend = sum(extend) // batch_size
    # The first value is emitted by the ragged SGLang batch; the second is the
    # uniform static shape AIC may ask for after reducing to mean sequence
    # lengths. They agree for uniform batches and harmlessly add coverage when
    # the batch is ragged.
    return {sum(extend), batch_size * mean_extend}


def retained_targets(
    counter: Counter[tuple[str, int, int]], max_per_phase: int
) -> tuple[list[dict[str, int | str]], dict[str, int]]:
    by_phase: dict[str, list[tuple[tuple[str, int, int], int]]] = {
        "context": [],
        "generation": [],
    }
    for target, occurrences in counter.items():
        by_phase[target[0]].append((target, occurrences))

    targets: list[dict[str, int | str]] = []
    omitted: dict[str, int] = {}
    for phase in ("context", "generation"):
        values = sorted(by_phase[phase], key=lambda item: (-item[1], item[0][1:]))
        selected = values if max_per_phase == 0 else values[:max_per_phase]
        omitted[phase] = len(values) - len(selected)
        targets.extend(
            {
                "phase": target[0],
                "batch_size": target[1],
                "input_len": target[2],
                "occurrences": occurrences,
            }
            for target, occurrences in selected
        )
    return sorted(targets, key=lambda row: (str(row["phase"]), int(row["batch_size"]), int(row["input_len"]))), omitted


def retained_gemm_m_values(
    counter: Counter[int], max_values: int
) -> tuple[list[int], int]:
    """Keep all observed M values by default, or the most common subset."""
    values = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    selected = values if max_values == 0 else values[:max_values]
    return sorted(value for value, _ in selected), len(values) - len(selected)


def write_summary(path: Path, targets: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["phase", "batch_size", "input_len", "occurrences"],
        )
        writer.writeheader()
        writer.writerows(targets)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    files = schedule_files(args.data_root)
    attention: Counter[tuple[str, int, int]] = Counter()
    gemm_m_values: Counter[int] = Counter()
    rows_by_mode: Counter[int] = Counter()
    skipped_empty_aic_shape: Counter[int] = Counter()

    for schedule_path in files:
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
                parsed = parse_batch(value, schedule_path, line_number)
                if parsed is None:
                    continue
                mode, extend, past = parsed
                rows_by_mode[mode] += 1
                phase, batch_size, input_len = effective_aic_shape(mode, extend, past)
                if input_len <= 0:
                    skipped_empty_aic_shape[mode] += 1
                    continue
                attention[(phase, batch_size, input_len)] += 1
                gemm_m_values.update(
                    value for value in target_gemm_m_values(mode, extend, past) if value > 0
                )

    if not attention:
        raise RuntimeError("no usable AIC attention targets were derived from the training traces")
    targets, omitted = retained_targets(attention, args.max_attention_targets_per_phase)
    retained_gemm_m, omitted_gemm_m = retained_gemm_m_values(
        gemm_m_values, args.max_gemm_m_values
    )
    plan = {
        "schema_version": 1,
        "purpose": "AIC Silicon calibration targets derived from training traces only",
        "model_path": str(args.model_path),
        "topology": topology(args.model_path),
        "input_data_roots": [str(Path(root).expanduser().resolve()) for root in args.data_root],
        "schedule_files": [str(path) for path in files],
        "schedule_rows": {
            "forward_mode_1": rows_by_mode[1],
            "forward_mode_2": rows_by_mode[2],
            "total": rows_by_mode[1] + rows_by_mode[2],
            "skipped_zero_length_aic_shape": dict(sorted(skipped_empty_aic_shape.items())),
        },
        "attention_targets": targets,
        "attention_targets_observed": {
            "context": sum(1 for phase, _, _ in attention if phase == "context"),
            "generation": sum(1 for phase, _, _ in attention if phase == "generation"),
        },
        "attention_targets_omitted_by_cap": omitted,
        "gemm_m_values": retained_gemm_m,
        "gemm_m_values_observed": len(gemm_m_values),
        "gemm_m_values_omitted_by_cap": omitted_gemm_m,
        "selection_policy": {
            "source": "training ScheduleBatch rows only",
            "aic_shape_reduction": "matches AIConfiguratorTimePredictor mean-length RuntimeConfig construction",
            "max_attention_targets_per_phase": args.max_attention_targets_per_phase,
            "max_gemm_m_values": args.max_gemm_m_values,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.summary_csv is not None:
        write_summary(args.summary_csv, targets)
    print(
        "schedule_rows={} context_targets={} generation_targets={} gemm_m_values={} out={}".format(
            plan["schedule_rows"]["total"],
            sum(1 for target in targets if target["phase"] == "context"),
            sum(1 for target in targets if target["phase"] == "generation"),
            len(retained_gemm_m),
            args.output,
        )
    )


if __name__ == "__main__":
    main()
