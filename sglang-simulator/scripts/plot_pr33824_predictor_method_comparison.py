#!/usr/bin/env python3
"""Plot a like-for-like three-method predictor comparison for the dense models.

The input experiments are deliberately kept as command-line arguments (with
the completed experiment directories as defaults).  The script produces two
figures and a compact README that records the comparison boundary instead of
silently treating different real-workload configurations as one A/B test.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


SIMULATOR_ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3-32B",
)
MODEL_LABELS = {
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen3-32B": "Qwen3-32B",
}
METHODS = (
    ("aic_degraded", "AIC：退化后端", "#7B8794"),
    ("aic_compatible", "AIC：SM80 兼容 kernel", "#3F78A8"),
    ("ml_compatible", "ML：SM80 兼容 kernel", "#D77941"),
)
PHASES = ("prefill", "decode")
CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a non-empty UTF-8 CSV file and give errors with its source path."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input CSV has no data rows: {path}")
    return rows


def indexed_single_step_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load exactly one prefill and decode row for every expected model."""
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in csv_rows(path):
        model = row.get("model", "")
        phase = row.get("phase", "")
        if model not in MODELS or phase not in PHASES:
            continue
        key = (model, phase)
        if key in output:
            raise ValueError(f"Duplicate single-step result for {key} in {path}")
        for field in ("real_mean_ms", "aic_predicted_ms", "absolute_error_pct"):
            if field not in row:
                raise ValueError(f"Missing {field!r} in {path}")
        output[key] = row
    expected = {(model, phase) for model in MODELS for phase in PHASES}
    if output.keys() != expected:
        missing = sorted(expected - output.keys())
        unexpected = sorted(output.keys() - expected)
        raise ValueError(
            f"Unexpected single-step coverage in {path}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return output


def indexed_e2e_rows(
    path: Path, mape_field: str
) -> dict[str, dict[str, str]]:
    """Select the per-model mean E2E latency MAPE rows from an experiment."""
    output: dict[str, dict[str, str]] = {}
    for row in csv_rows(path):
        if row.get("model") not in MODELS:
            continue
        if row.get("metric") != "mean_e2e_latency_ms":
            continue
        if mape_field not in row:
            raise ValueError(f"Missing {mape_field!r} in {path}")
        model = row["model"]
        if model in output:
            raise ValueError(f"Duplicate E2E result for {model} in {path}")
        output[model] = row
    if set(output) != set(MODELS):
        raise ValueError(
            f"Expected E2E rows for {list(MODELS)} in {path}; got {sorted(output)}"
        )
    return output


def make_features(extends: Sequence[int], pasts: Sequence[int]) -> tuple[float, ...]:
    """Build the v1 MLTimePredictor vector used by the trained ExtraTrees model."""
    if not extends or len(extends) != len(pasts):
        raise ValueError("extends and pasts must be non-empty sequences of equal length")
    if any(item < 0 for item in (*extends, *pasts)):
        raise ValueError("request lengths must be non-negative")

    batch_size = len(extends)
    sum_extend = sum(extends)
    sum_past = sum(pasts)
    sum_extend_x_past = sum(ext * past for ext, past in zip(extends, pasts))
    sum_extend_squared = sum(ext * ext for ext in extends)
    sum_past_squared = sum(past * past for past in pasts)
    sum_attn_flops = sum(
        ext * (past + ext / 2) for ext, past in zip(extends, pasts)
    )
    max_extend = max(extends)
    min_extend = min(extends)
    max_past = max(pasts)
    min_past = min(pasts)

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
        float(int(all(ext == 1 for ext in extends))),
        float(int(any(ext > 1 for ext in extends))),
    )


def ml_predictions_ms(model_root: Path) -> dict[tuple[str, str], float]:
    """Query each trained ML bundle at the fixed AIC single-step workload."""
    inputs = {
        "prefill": ([1024], [0]),
        # The first generated token has one new token and a 1025-token context.
        "decode": ([1], [1025]),
    }
    output: dict[tuple[str, str], float] = {}
    for model in MODELS:
        model_path = model_root / model / "model" / "latency_model.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing trained ML bundle: {model_path}")
        bundle = joblib.load(model_path)
        estimator = bundle.get("model")
        feature_names = bundle.get("features")
        if estimator is None or not isinstance(feature_names, list):
            raise ValueError(f"Invalid ML bundle: {model_path}")
        for phase, (extends, pasts) in inputs.items():
            features = make_features(extends, pasts)
            if len(feature_names) != len(features):
                raise ValueError(
                    f"Feature ABI mismatch in {model_path}: "
                    f"bundle={len(feature_names)}, script={len(features)}"
                )
            seconds = float(estimator.predict(np.asarray([features], dtype=float))[0])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Invalid ML prediction from {model_path}: {seconds}")
            output[(model, phase)] = seconds * 1000.0
        del bundle
        gc.collect()
    return output


def as_float(row: dict[str, str], field: str, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field!r} in {source}: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field!r} in {source}: {value}")
    return value


def pct_error(predicted_ms: float, real_ms: float) -> float:
    if real_ms <= 0:
        raise ValueError(f"real time must be positive, got {real_ms}")
    return abs(predicted_ms - real_ms) / real_ms * 100.0


def build_single_step_records(
    degraded_rows: dict[tuple[str, str], dict[str, str]],
    compatible_rows: dict[tuple[str, str], dict[str, str]],
    ml_predicted_ms: dict[tuple[str, str], float],
    degraded_path: Path,
    compatible_path: Path,
    ml_root: Path,
) -> list[dict[str, Any]]:
    """Create records with a measured/predicted value behind each plotted MAPE."""
    records: list[dict[str, Any]] = []
    for model in MODELS:
        for phase in PHASES:
            degraded = degraded_rows[(model, phase)]
            compatible = compatible_rows[(model, phase)]
            for method_key, row, path in (
                ("aic_degraded", degraded, degraded_path),
                ("aic_compatible", compatible, compatible_path),
            ):
                records.append(
                    {
                        "comparison": "single_step",
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "phase": phase,
                        "method": method_key,
                        "absolute_error_pct": as_float(row, "absolute_error_pct", path),
                        "real_mean_ms": as_float(row, "real_mean_ms", path),
                        "predicted_ms": as_float(row, "aic_predicted_ms", path),
                        "paired_runs": int(row["real_samples"]),
                        "source": str(path),
                        "note": "AIC Silicon static composition",
                    }
                )

            real_ms = as_float(compatible, "real_mean_ms", compatible_path)
            predicted_ms = ml_predicted_ms[(model, phase)]
            records.append(
                {
                    "comparison": "single_step",
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "phase": phase,
                    "method": "ml_compatible",
                    "absolute_error_pct": pct_error(predicted_ms, real_ms),
                    "real_mean_ms": real_ms,
                    "predicted_ms": predicted_ms,
                    "paired_runs": int(compatible["real_samples"]),
                    "source": str(ml_root),
                    "note": (
                        "ExtraTrees prediction; uses compatible single-step real mean; "
                        "input length 1024 is inside the ML calibration grid"
                    ),
                }
            )
    return records


def build_e2e_records(
    degraded_rows: dict[str, dict[str, str]],
    compatible_rows: dict[str, dict[str, str]],
    ml_rows: dict[str, dict[str, str]],
    degraded_path: Path,
    compatible_path: Path,
    ml_path: Path,
) -> list[dict[str, Any]]:
    """Create the 3 workloads x 5 paired-run E2E MAPE records."""
    records: list[dict[str, Any]] = []
    sources = (
        ("aic_degraded", degraded_rows, "mape_mean_pct", degraded_path),
        ("aic_compatible", compatible_rows, "mape_mean_pct", compatible_path),
        ("ml_compatible", ml_rows, "mape_pct", ml_path),
    )
    for model in MODELS:
        for method_key, rows, mape_field, source in sources:
            row = rows[model]
            record: dict[str, Any] = {
                "comparison": "end_to_end",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "phase": "mean_e2e_latency_ms",
                "method": method_key,
                "absolute_error_pct": as_float(row, mape_field, source),
                "real_mean_ms": row.get("real_mean", ""),
                "predicted_ms": row.get("predicted_mean", ""),
                "paired_runs": int(row["paired_runs"]),
                "source": str(source),
                "note": "Per-model mean across 3 workloads x 5 paired runs",
            }
            records.append(record)
    return records


def records_by_key(records: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    output: dict[tuple[str, str, str], float] = {}
    for record in records:
        key = (record["model"], record["phase"], record["method"])
        if key in output:
            raise ValueError(f"Duplicate plotted result: {key}")
        output[key] = float(record["absolute_error_pct"])
    return output


def setup_plot_style() -> None:
    """Use the system CJK font by file path; its family is not in all caches."""
    if CJK_FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(CJK_FONT_PATH))
        cjk_family = font_manager.FontProperties(fname=str(CJK_FONT_PATH)).get_name()
    else:
        cjk_family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": [cjk_family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFBFC",
        }
    )


def plot_single_step(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Render prefill and first-token decode MAPEs in two aligned panels."""
    indexed = records_by_key(records)
    setup_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.3), sharey=True)
    x = np.arange(len(MODELS), dtype=float)
    width = 0.23
    max_value = max(indexed[(model, phase, method)] for model in MODELS for phase in PHASES for method, _, _ in METHODS)
    y_limit = max(20.0, math.ceil(max_value / 10.0) * 10.0 + 15.0)

    for axis, phase, title in zip(
        axes,
        PHASES,
        ("Prefill：输入 1024 tokens", "Decode：首个生成 token（上下文 1025）"),
        strict=True,
    ):
        for method_index, (method, label, color) in enumerate(METHODS):
            values = [indexed[(model, phase, method)] for model in MODELS]
            positions = x + (method_index - 1) * width
            bars = axis.bar(
                positions,
                values,
                width=width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + y_limit * 0.015,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        axis.set_title(title, pad=12)
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
        axis.set_ylim(0, y_limit)
        axis.grid(axis="y", color="#D9DDE3", linewidth=0.7, alpha=0.85)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("绝对误差 / MAPE (%)")
    figure.suptitle("固定单步预测误差对比（batch=1）", fontsize=16, fontweight="bold", y=0.985)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    figure.text(
        0.5,
        0.025,
        "† ML 的 input=1024 固定点位位于其校准网格内；独立 holdout 结果见 README。",
        ha="center",
        fontsize=9.5,
        color="#4F5965",
    )
    figure.tight_layout(rect=(0.02, 0.075, 0.98, 0.84))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_e2e(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Render the wide-range E2E MAPE values on a clearly labelled log scale."""
    indexed = records_by_key(records)
    setup_plot_style()
    figure, axis = plt.subplots(figsize=(10.5, 6.3))
    x = np.arange(len(MODELS), dtype=float)
    width = 0.23
    max_value = max(indexed[(model, "mean_e2e_latency_ms", method)] for model in MODELS for method, _, _ in METHODS)
    upper_limit = 10 ** math.ceil(math.log10(max_value * 1.8))

    for method_index, (method, label, color) in enumerate(METHODS):
        values = [indexed[(model, "mean_e2e_latency_ms", method)] for model in MODELS]
        positions = x + (method_index - 1) * width
        bars = axis.bar(
            positions,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.16,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_yscale("log")
    axis.set_ylim(1, upper_limit)
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
    axis.set_ylabel("平均 E2E 延迟 MAPE (%)，对数刻度")
    axis.set_title("端到端预测误差对比（3 个负载 × 每负载 5 次配对）", pad=16)
    axis.grid(axis="y", which="both", color="#D9DDE3", linewidth=0.7, alpha=0.85)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False)
    figure.text(
        0.5,
        0.025,
        "AIC 两组结果各自对应其运行时的真实基线；ML 复用 SM80 兼容 kernel 的真实基线。",
        ha="center",
        fontsize=9.5,
        color="#4F5965",
    )
    figure.tight_layout(rect=(0.02, 0.07, 0.98, 0.98))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def markdown_table(
    records: Sequence[dict[str, Any]], phase: str
) -> list[str]:
    indexed = records_by_key(records)
    lines = [
        "| 模型 | AIC：退化后端 | AIC：SM80 兼容 kernel | ML：SM80 兼容 kernel |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        values = [indexed[(model, phase, method)] for method, _, _ in METHODS]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            + " | ".join(f"{value:.3f}%" for value in values)
            + " |"
        )
    return lines


def write_readme(
    output_path: Path,
    single_step: Sequence[dict[str, Any]],
    e2e: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write the facts needed to read the charts without overclaiming a causal A/B."""
    lines = [
        "# 三种预测方式的误差对比",
        "",
        "本目录比较三个 dense 模型上的单步和端到端预测误差。图中的数值和来源见 `comparison_data.csv`。",
        "",
        "- [单步误差图](single_step_mape.png)：batch=1，1024-token prefill，以及首个 decode token（上下文长度 1025）。",
        "- [端到端误差图](end_to_end_e2e_mape.png)：每个模型的 3 个负载、每负载 5 次真实/预测配对结果；纵轴为对数刻度。",
        "",
        "## 三种方式",
        "",
        "| 图例 | 预测方式和真实负载 |",
        "| --- | --- |",
        "| AIC：退化后端 | AIConfigurator Silicon 表；真实负载使用 Triton attention 和 Torch BF16 GEMM fallback。 |",
        "| AIC：SM80 兼容 kernel | AIConfigurator Silicon 表；真实负载使用 SM80 AOT 兼容 kernel。AIC 本身不执行 CUDA kernel。 |",
        "| ML：SM80 兼容 kernel | 不调用 AIConfigurator；用本机原生 SGLang `one_batch` 采样训练的 ExtraTrees，端到端预测也使用该 ML predictor。 |",
        "",
        "## 单步绝对误差 / MAPE",
        "",
        "### Prefill（1024 tokens）",
        "",
        *markdown_table(single_step, "prefill"),
        "",
        "### 首个 Decode token（上下文 1025）",
        "",
        *markdown_table(single_step, "decode"),
        "",
        "ML 的这两个固定点位中，input=1024 已在其校准网格内，因此它们不是独立 holdout。该事实在图中以 † 标出；不要把 3B 的低单点误差外推成端到端精度。",
        "",
        "## 端到端平均 E2E 延迟 MAPE",
        "",
        *markdown_table(e2e, "mean_e2e_latency_ms"),
        "",
        "## 比较边界",
        "",
        "两组 AIC 实验分别使用其各自采集时的真实基线，因而图表能展示三种已完成实验的预测误差，但不能把“退化后端”和“兼容 kernel”的差异单独归因为 kernel；两批真实运行的时间和配置并非严格只差一个变量。ML 端到端结果复用了兼容 kernel 实验的真实基线。",
        "",
        "## 输入来源",
        "",
        f"- AIC 退化后端单步：`{args.aic_degraded_step}`",
        f"- AIC SM80 兼容 kernel 单步：`{args.aic_compatible_step}`",
        f"- AIC 退化后端端到端：`{args.aic_degraded_e2e}`",
        f"- AIC SM80 兼容 kernel 端到端：`{args.aic_compatible_e2e}`",
        f"- ML 端到端：`{args.ml_e2e}`",
        f"- ML 模型：`{args.ml_model_root}`",
        "",
        "生成脚本：`scripts/plot_pr33824_predictor_method_comparison.py`。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = (
        "comparison",
        "model",
        "model_label",
        "phase",
        "method",
        "absolute_error_pct",
        "real_mean_ms",
        "predicted_ms",
        "paired_runs",
        "source",
        "note",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def default_path(*parts: str) -> Path:
    return SIMULATOR_ROOT.joinpath(*parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--aic-degraded-step",
        type=Path,
        default=default_path(
            "test-results",
            "pr33824_aic_vs_real_single_step_dense3_20260908_155211",
            "analysis",
            "single_step_comparison.csv",
        ),
    )
    parser.add_argument(
        "--aic-compatible-step",
        type=Path,
        default=default_path(
            "test-results",
            "pr33824_aic_vs_real_single_step_dense3_sm80compat_20260910_181204",
            "analysis",
            "single_step_comparison.csv",
        ),
    )
    parser.add_argument(
        "--aic-degraded-e2e",
        type=Path,
        default=default_path(
            "test-results",
            "pr33824_piecewise_a100_dense3_matrix_20260908_134523",
            "analysis",
            "model_metric_statistics.csv",
        ),
    )
    parser.add_argument(
        "--aic-compatible-e2e",
        type=Path,
        default=default_path(
            "test-results",
            "pr33824_a100_dense3_sm80compat_matrix_20260910_164000",
            "analysis",
            "model_metric_statistics.csv",
        ),
    )
    parser.add_argument(
        "--ml-e2e",
        type=Path,
        default=default_path(
            "test-results",
            "pr33824_ml_dense3_matrix_20260915_131100",
            "analysis",
            "model_metric_summary.csv",
        ),
    )
    parser.add_argument(
        "--ml-model-root",
        type=Path,
        default=default_path(
            "test-results",
            "pr33824_ml_native_calibration_dense3_20260915_130105",
            "models",
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    degraded_step = indexed_single_step_rows(args.aic_degraded_step)
    compatible_step = indexed_single_step_rows(args.aic_compatible_step)
    ml_step = ml_predictions_ms(args.ml_model_root)
    single_step = build_single_step_records(
        degraded_step,
        compatible_step,
        ml_step,
        args.aic_degraded_step,
        args.aic_compatible_step,
        args.ml_model_root,
    )
    e2e = build_e2e_records(
        indexed_e2e_rows(args.aic_degraded_e2e, "mape_mean_pct"),
        indexed_e2e_rows(args.aic_compatible_e2e, "mape_mean_pct"),
        indexed_e2e_rows(args.ml_e2e, "mape_pct"),
        args.aic_degraded_e2e,
        args.aic_compatible_e2e,
        args.ml_e2e,
    )

    write_csv(output_dir / "comparison_data.csv", [*single_step, *e2e])
    plot_single_step(single_step, output_dir / "single_step_mape.png")
    plot_e2e(e2e, output_dir / "end_to_end_e2e_mape.png")
    write_readme(output_dir / "README.md", single_step, e2e, args)
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "sources": {
            "aic_degraded_step": str(args.aic_degraded_step),
            "aic_compatible_step": str(args.aic_compatible_step),
            "aic_degraded_e2e": str(args.aic_degraded_e2e),
            "aic_compatible_e2e": str(args.aic_compatible_e2e),
            "ml_e2e": str(args.ml_e2e),
            "ml_model_root": str(args.ml_model_root),
        },
        "models": list(MODELS),
        "methods": [method for method, _, _ in METHODS],
        "single_step": {
            "batch_size": 1,
            "prefill_input_tokens": 1024,
            "decode_extend_tokens": 1,
            "decode_past_tokens": 1025,
            "ml_fixed_point_is_in_calibration_grid": True,
        },
        "e2e": {"workloads_per_model": 3, "paired_runs_per_workload": 5},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote comparison figures and data to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
