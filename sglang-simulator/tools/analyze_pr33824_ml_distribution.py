#!/usr/bin/env python3
"""Audit scheduler-batch feature coverage between ML train and held-out traces.

This tool compares only real, device-synchronized ScheduleBatch rows.  It is
intended to document the input contract of a per-model ML latency predictor;
it neither loads an ML bundle nor scores predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


FEATURE_NAMES = (
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
)


def extract_features(extends: list[int], pasts: list[int]) -> tuple[float, ...]:
    """Match the 18-feature ML predictor ABI without importing its runtime."""
    batch_size = len(extends)
    sum_extend = sum(extends)
    sum_past = sum(pasts)
    sum_extend_x_past = sum(
        extend * past for extend, past in zip(extends, pasts, strict=True)
    )
    sum_extend_squared = sum(value * value for value in extends)
    sum_past_squared = sum(value * value for value in pasts)
    sum_attn_flops = sum(
        extend * (past + extend / 2)
        for extend, past in zip(extends, pasts, strict=True)
    )
    max_extend = max(extends)
    min_extend = min(extends)
    max_past = max(pasts)
    min_past = min(pasts)
    is_decode = int(all(extend == 1 for extend in extends))
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
        float(is_decode),
        float(1 - is_decode),
    )


def schedule_files(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"data root does not exist: {root}")
        files.update(root.rglob("TP0.schedule_batch.jsonl"))
    if not files:
        raise FileNotFoundError("no TP0.schedule_batch.jsonl files found")
    return sorted(files)


def load_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schedule in schedule_files(roots):
        with schedule.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                item = json.loads(line)
                forward_mode = int(item.get("forward_mode", 0))
                latency = float(item.get("iter_latency", 0.0))
                requests = item.get("request_infos") or []
                if forward_mode not in (1, 2) or not requests or not 0 < latency <= 30:
                    continue
                extends = [int(request["extend_input_len"]) for request in requests]
                pasts = [int(request["prefix_indices_len"]) for request in requests]
                if any(value < 0 for value in (*extends, *pasts)):
                    raise ValueError(f"negative length at {schedule}:{line_number}")
                rows.append(
                    {
                        "phase": "decode" if forward_mode == 2 else "prefill",
                        "features": extract_features(extends, pasts),
                    }
                )
    if not rows:
        raise RuntimeError("no usable forward-mode 1/2 rows")
    return rows


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def format_number(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"


def summarize(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> list[dict[str, str]]:
    groups: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {
        "all": (train, test),
    }
    for phase in ("prefill", "decode"):
        groups[phase] = (
            [row for row in train if row["phase"] == phase],
            [row for row in test if row["phase"] == phase],
        )

    output: list[dict[str, str]] = []
    for phase, (train_rows, test_rows) in groups.items():
        if not train_rows or not test_rows:
            continue
        train_features = [row["features"] for row in train_rows]
        test_features = [row["features"] for row in test_rows]
        train_set = set(train_features)
        minima = [min(row[index] for row in train_features) for index in range(len(FEATURE_NAMES))]
        maxima = [max(row[index] for row in train_features) for index in range(len(FEATURE_NAMES))]
        exact = sum(row in train_set for row in test_features)
        outside = sum(
            any(value < lower or value > upper for value, lower, upper in zip(row, minima, maxima, strict=True))
            for row in test_features
        )
        row = {
            "phase": phase,
            "train_rows": str(len(train_rows)),
            "test_rows": str(len(test_rows)),
            "exact_feature_match_pct": f"{100 * exact / len(test_rows):.2f}",
            "test_outside_train_feature_range_pct": f"{100 * outside / len(test_rows):.2f}",
        }
        for name, index in (
            ("batch_size", 0),
            ("sum_extend", 1),
            ("sum_past", 4),
        ):
            train_values = [value[index] for value in train_features]
            test_values = [value[index] for value in test_features]
            row[f"train_{name}_p50"] = format_number(quantile(train_values, 0.5))
            row[f"test_{name}_p50"] = format_number(quantile(test_values, 0.5))
            row[f"train_{name}_min"] = format_number(min(train_values))
            row[f"train_{name}_max"] = format_number(max(train_values))
            row[f"test_{name}_min"] = format_number(min(test_values))
            row[f"test_{name}_max"] = format_number(max(test_values))
        output.append(row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", action="append", required=True)
    parser.add_argument("--test-root", action="append", required=True)
    parser.add_argument("--out", required=True, help="New CSV output path")
    parser.add_argument("--markdown", help="Optional new Markdown summary path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_roots = [Path(value).expanduser().resolve() for value in args.train_root]
    test_roots = [Path(value).expanduser().resolve() for value in args.test_root]
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if args.markdown and Path(args.markdown).expanduser().exists():
        raise FileExistsError(f"refusing to overwrite: {args.markdown}")

    rows = summarize(load_rows(train_roots), load_rows(test_roots))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.markdown:
        markdown = Path(args.markdown).expanduser().resolve()
        lines = [
            "# ML 训练/测试 Scheduler batch 分布核对",
            "",
            "统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。",
            "",
            "| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for row in rows:
            lines.append(
                "| {phase} | {train_rows} | {test_rows} | {exact_feature_match_pct}% | {test_outside_train_feature_range_pct}% | {train_batch_size_p50} / {test_batch_size_p50} | {train_sum_extend_p50} / {test_sum_extend_p50} | {train_sum_past_p50} / {test_sum_past_p50} |".format(**row)
            )
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
