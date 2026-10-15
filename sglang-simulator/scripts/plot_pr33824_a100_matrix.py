#!/usr/bin/env python3
"""Create parameter-aware accuracy plots from PR #33824 matrix summaries."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("mean_ttft_ms", "TTFT MAPE (%)"),
    ("mean_tpot_ms", "TPOT MAPE (%)"),
    ("mean_e2e_latency_ms", "E2E MAPE (%)"),
    ("total_throughput", "Total throughput MAPE (%)"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def annotate_points(axis, x, y, labels) -> None:
    for x_value, y_value, label in zip(x, y, labels):
        axis.annotate(label, (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.summary_root.resolve()
    metrics = read_csv(root / "model_metric_statistics.csv")
    parameters = read_csv(root / "model_parameters.csv")
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    params = {row["model"]: row for row in parameters}
    models = sorted(params)
    metric_map = {(row["model"], row["metric"]): row for row in metrics}
    plotted_models = [
        model
        for model in models
        if all(
            as_float(metric_map.get((model, metric), {}).get("mape_mean_pct"))
            is not None
            for metric, _ in METRICS
        )
    ]

    # Grouped model comparison includes sample error bars from the five repeats
    # across each of the three fixed workload families.
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(plotted_models)))
    positions = np.arange(len(plotted_models))
    for axis, (metric, label) in zip(axes.flat, METRICS):
        means = []
        errors = []
        for model in plotted_models:
            row = metric_map.get((model, metric), {})
            means.append(as_float(row.get("mape_mean_pct")))
            errors.append(as_float(row.get("absolute_error_std_pct")) or 0.0)
        axis.bar(positions, means, yerr=errors, capsize=3, color=colors, alpha=0.85)
        axis.set_title(label)
        axis.set_ylabel("MAPE (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(positions, plotted_models, rotation=42, ha="right", fontsize=8)
    fig.suptitle(
        "PR #33824 simulator accuracy by model (mean ± sample std; incomplete models omitted)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(figures / "mape_by_model_and_metric.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot model parameters rather than treating model name as the only factor.
    parameter_specs = (
        ("checkpoint_size_gb", "Checkpoint size (GB)"),
        ("hidden_size", "Hidden size"),
        ("num_hidden_layers", "Number of layers"),
        ("num_attention_heads", "Attention heads"),
        ("num_local_experts", "MoE experts"),
        ("max_position_embeddings", "Max context length"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    e2e_rows = {model: metric_map.get((model, "mean_e2e_latency_ms"), {}) for model in models}
    architectures = sorted({params[model].get("model_type", "unknown") for model in models})
    palette = {architecture: color for architecture, color in zip(architectures, plt.cm.Set2(np.linspace(0, 1, len(architectures))))}
    for axis, (field, label) in zip(axes.flat, parameter_specs):
        points = []
        for model in models:
            x_value = as_float(params[model].get(field))
            y_value = as_float(e2e_rows[model].get("mape_mean_pct"))
            if x_value is not None and y_value is not None:
                points.append((model, x_value, y_value, params[model].get("model_type", "unknown")))
        for architecture in architectures:
            selected = [point for point in points if point[3] == architecture]
            if selected:
                axis.scatter([point[1] for point in selected], [point[2] for point in selected], label=architecture, color=palette[architecture], s=58, alpha=0.9)
        annotate_points(axis, [point[1] for point in points], [point[2] for point in points], [point[0] for point in points])
        axis.set_xlabel(label)
        axis.set_ylabel("E2E MAPE (%)")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(title="model_type", fontsize=8)
    fig.suptitle("E2E prediction error versus checkpoint and architecture parameters", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures / "e2e_mape_vs_model_parameters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # The same model can behave differently under the three load regimes.
    case_rows = read_csv(root / "case_metric_statistics.csv")
    workloads = sorted({row["case"] for row in case_rows})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 4.8), sharex=True)
    for axis, (metric, label) in zip(axes, METRICS):
        values = defaultdict(list)
        for row in case_rows:
            value = as_float(row.get(f"absolute_error_pct_{metric}_mean"))
            if value is not None:
                values[row["case"]].append(value)
        nonempty_workloads = [workload for workload in workloads if values[workload]]
        plot_values = [values[workload] for workload in nonempty_workloads]
        axis.boxplot(plot_values, tick_labels=nonempty_workloads, showmeans=True)
        axis.set_title(label)
        axis.set_ylabel("per-model MAPE (%)")
        axis.tick_params(axis="x", rotation=28, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Error distribution across model parameters, grouped by workload", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures / "mape_distribution_by_workload.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures to {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
