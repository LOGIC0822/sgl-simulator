#!/usr/bin/env python3
"""Compare collector-trained ML batch scores with historical AIC single steps.

The two inputs are intentionally kept separate.  AIC rows are fixed batch=1
measurements at 1024-token prefill and the first decode token at context 1025.
The collector-trained ML rows are teacher-forced scores on real Scheduler
batches from the held-out saturation service workload.  Their MAPE difference
is useful for orientation, but is not a controlled replacement comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


AIC_METHODS = (
    ("aic_degraded", "AIC degraded backend"),
    ("aic_compatible", "AIC SM80-compatible kernel"),
)
PHASES = (("prefill", "1"), ("decode", "2"))


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    phase: str
    ml_mape_pct: float
    aic_degraded_mape_pct: float
    aic_compatible_mape_pct: float
    batches: int

    @property
    def degraded_delta_pp(self) -> float:
        return self.ml_mape_pct - self.aic_degraded_mape_pct

    @property
    def compatible_delta_pp(self) -> float:
        return self.ml_mape_pct - self.aic_compatible_mape_pct


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def required_row(rows: Iterable[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row matching {criteria}, found {len(matches)}")
    return matches[0]


def collect_rows(
    aic_csv: Path, ml_root: Path, models: list[str]
) -> list[ComparisonRow]:
    historical_rows = [
        row for row in read_csv(aic_csv) if row.get("comparison") == "single_step"
    ]
    output: list[ComparisonRow] = []
    for model in models:
        score_csv = (
            ml_root / "models" / model / "holdout" / "teacher_forced" / "summary.csv"
        )
        scores = read_csv(score_csv)
        for phase, mode in PHASES:
            ml_row = required_row(scores, group="forward_mode", value=mode)
            values = {
                method: float(
                    required_row(
                        historical_rows,
                        model=model,
                        phase=phase,
                        method=method,
                    )["absolute_error_pct"]
                )
                for method, _ in AIC_METHODS
            }
            output.append(
                ComparisonRow(
                    model=model,
                    phase=phase,
                    ml_mape_pct=float(ml_row["mape_pct"]),
                    aic_degraded_mape_pct=values["aic_degraded"],
                    aic_compatible_mape_pct=values["aic_compatible"],
                    batches=int(float(ml_row["n"])),
                )
            )
    return output


def write_csv(path: Path, rows: list[ComparisonRow]) -> None:
    fields = (
        "model",
        "phase",
        "ml_teacher_forced_mape_pct",
        "aic_degraded_mape_pct",
        "aic_compatible_mape_pct",
        "ml_minus_aic_degraded_pp",
        "ml_minus_aic_compatible_pp",
        "ml_scheduler_batches",
    )
    with path.open("x", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "phase": row.phase,
                    "ml_teacher_forced_mape_pct": row.ml_mape_pct,
                    "aic_degraded_mape_pct": row.aic_degraded_mape_pct,
                    "aic_compatible_mape_pct": row.aic_compatible_mape_pct,
                    "ml_minus_aic_degraded_pp": row.degraded_delta_pp,
                    "ml_minus_aic_compatible_pp": row.compatible_delta_pp,
                    "ml_scheduler_batches": row.batches,
                }
            )


def short_model_name(model: str) -> str:
    return model.replace("-Instruct", "")


def plot(rows: list[ComparisonRow], output: Path) -> tuple[Path, Path]:
    # Keep the Matplotlib cache outside source and result directories.
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sglang-simulator-matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np

    by_phase = {
        phase: [row for row in rows if row.phase == phase] for phase, _ in PHASES
    }
    labels = [short_model_name(row.model) for row in by_phase["prefill"]]
    positions = np.arange(len(labels))
    width = 0.24

    absolute_path = output / "ml_vs_aic_single_step_mape.png"
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=False)
    for axis, (phase, _) in zip(axes, PHASES):
        selected = by_phase[phase]
        series = (
            ("AIC degraded", [row.aic_degraded_mape_pct for row in selected]),
            ("AIC SM80-compatible", [row.aic_compatible_mape_pct for row in selected]),
            ("Collector ML (held-out batches)", [row.ml_mape_pct for row in selected]),
        )
        for index, (label, values) in enumerate(series):
            bars = axis.bar(positions + (index - 1) * width, values, width, label=label)
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, rotation=90)
        aic_shape = "batch=1, fixed token shape" \
            if phase == "prefill" else "batch=1, fixed first-token shape"
        axis.set_title(f"{phase.capitalize()}: AIC {aic_shape}")
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.set_ylabel("MAPE (%)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Historical AIC fixed-point MAPE vs. collector ML Scheduler-batch MAPE\n"
        "Different input distributions: visual comparison only, not a controlled ranking",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(absolute_path, dpi=200)
    plt.close(figure)

    delta_path = output / "ml_minus_aic_single_step_delta_pp.png"
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for axis, (phase, _) in zip(axes, PHASES):
        selected = by_phase[phase]
        degraded = [row.degraded_delta_pp for row in selected]
        compatible = [row.compatible_delta_pp for row in selected]
        bars_a = axis.bar(
            positions - width / 2,
            degraded,
            width,
            label="ML − AIC degraded",
        )
        bars_b = axis.bar(
            positions + width / 2,
            compatible,
            width,
            label="ML − AIC SM80-compatible",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.bar_label(bars_a, fmt="%+.1f", padding=2, fontsize=8, rotation=90)
        axis.bar_label(bars_b, fmt="%+.1f", padding=2, fontsize=8, rotation=90)
        axis.set_title(phase.capitalize())
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.set_ylabel("MAPE difference (percentage points)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle(
        "Negative values mean lower ML MAPE; values remain descriptive across different workloads",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(delta_path, dpi=200)
    plt.close(figure)
    return absolute_path, delta_path


def write_readme(path: Path, rows: list[ComparisonRow]) -> None:
    lines = [
        "# Collector ML 与 AIC 单步误差对照",
        "",
        "## 比较口径",
        "",
        "AIC 数值来自历史固定点实验：batch=1，prefill 为 1024 tokens，decode 为上下文 1025 的首个 token；每项为 5 次真机测量相对 AIC Silicon 预测的 MAPE。",
        "",
        "ML 数值来自 2026-09-17 的 collector + HGB predictor：在独立的 saturation_1024x256_inf 服务负载中，对真实记录的 Scheduler batch 做 teacher-forced 打分。每个模型每个 phase 有 155 个 prefill batch 或 1,040 个 decode batch；batch 大小和 token 长度均为动态值。",
        "",
        "因此 `ML − AIC` 是用于观察量级的 MAPE 差（百分点），不是同一输入上的受控精度差。旧 ExtraTrees 固定点结果未纳入本图，因为 1024-token 点位落在它的训练网格内。",
        "",
        "## 数值（MAPE %）",
        "",
        "| 模型 | phase | ML Scheduler-batch | AIC 退化后端 | ML−AIC 退化（pp） | AIC SM80 兼容 kernel | ML−AIC 兼容（pp） | ML batch 数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {phase} | {ml:.2f} | {degraded:.2f} | {delta_degraded:+.2f} | "
            "{compatible:.2f} | {delta_compatible:+.2f} | {batches} |".format(
                model=row.model,
                phase=row.phase,
                ml=row.ml_mape_pct,
                degraded=row.aic_degraded_mape_pct,
                delta_degraded=row.degraded_delta_pp,
                compatible=row.aic_compatible_mape_pct,
                delta_compatible=row.compatible_delta_pp,
                batches=row.batches,
            )
        )
    lines.extend(
        [
            "",
            "![单步 MAPE 对照](ml_vs_aic_single_step_mape.png)",
            "",
            "![ML 相对 AIC 的 MAPE 差](ml_minus_aic_single_step_delta_pp.png)",
            "",
            "## 来源",
            "",
            "- `comparison.csv`：逐项 MAPE 与百分点差。",
            "- 历史 AIC：`pr33824_predictor_method_comparison_dense3_20260915_231314/comparison_data.csv`。",
            "- 当前 ML：`pr33824_data_tools_dense3_20260917/models/*/holdout/teacher_forced/summary.csv`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aic-comparison-csv", type=Path, required=True)
    parser.add_argument("--ml-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.aic_comparison_csv = args.aic_comparison_csv.expanduser().resolve()
    args.ml_root = args.ml_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not args.aic_comparison_csv.is_file():
        parser.error(f"missing AIC comparison CSV: {args.aic_comparison_csv}")
    if not args.ml_root.is_dir():
        parser.error(f"missing ML result root: {args.ml_root}")
    if not args.models:
        parser.error("--models must name at least one checkpoint")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.out_dir}")
    return args


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.aic_comparison_csv, args.ml_root, args.models)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.out_dir / "comparison.csv", rows)
    absolute_path, delta_path = plot(rows, args.out_dir)
    write_readme(args.out_dir / "README.md", rows)
    print(f"rows={len(rows)} absolute={absolute_path} delta={delta_path}")


if __name__ == "__main__":
    main()
