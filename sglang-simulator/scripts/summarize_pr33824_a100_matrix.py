#!/usr/bin/env python3
"""Summarize matched A100-versus-PR#33824 simulator repetitions.

The raw run directories are authoritative.  This script is deliberately
stdlib-only: it produces a repeat-level paired table and aggregate statistics
without changing benchmark artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, variance
from typing import Any, Iterable


METRICS = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p99_e2e_latency_ms",
)

DISPLAY_METRICS = (
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
    "total_throughput",
    "request_throughput",
)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_benchmark(path: Path) -> dict[str, Any] | None:
    """Read either SGLang's JSONL output or a single JSON result file."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def is_successful_run(path: Path) -> bool:
    """Return true only for an explicitly completed raw runner directory.

    A benchmark or metrics file can exist when a client exits part-way through
    a run.  The runner writes ``status.txt`` only after its output has passed
    the success checks, so that marker is the authority for pairing samples.
    """
    try:
        return path.read_text(encoding="utf-8").strip() == "ok"
    except OSError:
        return False


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def relative_error(real: float | None, predicted: float | None) -> float | None:
    if real is None or predicted is None or real == 0:
        return None
    return (predicted - real) * 100.0 / real


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    data = list(values)
    if not data:
        return {"n": 0, "mean": None, "min": None, "max": None, "std": None, "variance": None}
    return {
        "n": len(data),
        "mean": mean(data),
        "min": min(data),
        "max": max(data),
        "std": stdev(data) if len(data) > 1 else 0.0,
        "variance": variance(data) if len(data) > 1 else 0.0,
    }


def model_parameters(model_dir: Path) -> dict[str, Any]:
    config = load_json(model_dir / "model-config.json") or {}
    bytes_path = model_dir / "checkpoint-bytes.txt"
    try:
        checkpoint_bytes = int(bytes_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        checkpoint_bytes = None
    quantization = (config.get("quantization_config") or {}).get("quant_method")
    # Config field names differ between dense and MoE checkpoint families.
    # Store both the normalized field used by the plots and the original
    # architecture details so the report does not silently label an MoE model
    # as dense just because it uses ``num_experts`` instead of
    # ``num_local_experts``.
    local_experts = config.get("num_local_experts", config.get("num_experts", 0))
    return {
        "model": model_dir.name,
        "model_type": config.get("model_type", "unknown"),
        "architectures": ";".join(config.get("architectures", [])),
        "quantization": quantization or "none",
        "checkpoint_size_gb": checkpoint_bytes / 1e9 if checkpoint_bytes else None,
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "intermediate_size": config.get("intermediate_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "num_local_experts": local_experts,
        "num_experts": config.get("num_experts", local_experts),
        "num_experts_per_tok": config.get("num_experts_per_tok", 0),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
    }


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def predictor_scope_note(root: Path) -> str:
    """Describe the selected predictor without assuming an AIC-only run."""
    config = load_json(root / "simulator-config.json") or {}
    predictor = config.get("predictor") if isinstance(config, dict) else {}
    name = predictor.get("name") if isinstance(predictor, dict) else None
    if name == "aiconfigurator":
        return "`simulator` 是 PR #33824 的 CPU 模拟器，用 AIC Silicon 数据库预测 A100，不执行模型 GPU forward。"
    if name == "ml":
        return "`simulator` 是 PR #33824 的 CPU 模拟器，加载 MLTimePredictor 的 sklearn/joblib bundle 预测 A100，不执行模型 GPU forward。"
    if name == "replay":
        return "`simulator` 是 PR #33824 的 CPU 模拟器，加载 replay 时延表预测 A100，不执行模型 GPU forward。"
    return f"`simulator` 是 PR #33824 的 CPU 模拟器，使用 predictor={name or 'unknown'} 预测 A100，不执行模型 GPU forward。"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--prediction-aggregation",
        choices=("matched", "mean-per-case"),
        default="matched",
        help=(
            "matched pairs same-numbered real/prediction repeats; mean-per-case "
            "uses the mean of the completed prediction repeats against each real repeat"
        ),
    )
    args = parser.parse_args()
    root = args.results_root.resolve()
    results = root / "results"
    if not results.is_dir():
        raise SystemExit(f"Missing results directory: {results}")
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    repeat_rows: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in results.iterdir() if path.is_dir()):
        parameters.append(model_parameters(model_dir))
        for case_path in sorted((model_dir / "cases").glob("*.json")):
            case = load_json(case_path) or {}
            case_name = case_path.stem
            real_dir = model_dir / "real" / case_name
            sim_dir = model_dir / "simulator" / case_name
            repeat_names = {entry.name for entry in real_dir.glob("repeat-*")} if real_dir.is_dir() else set()
            if sim_dir.is_dir():
                repeat_names.update(entry.name for entry in sim_dir.glob("repeat-*"))
            predicted_by_repeat: dict[str, dict[str, Any]] = {}
            if sim_dir.is_dir():
                for prediction_dir in sim_dir.glob("repeat-*"):
                    if not is_successful_run(prediction_dir / "status.txt"):
                        continue
                    prediction = load_json(prediction_dir / "simulator-output" / "metrics.json")
                    if prediction is not None:
                        predicted_by_repeat[prediction_dir.name] = prediction
            prediction_mean: dict[str, float] = {}
            if args.prediction_aggregation == "mean-per-case":
                for metric in METRICS:
                    values = [
                        number(prediction.get(metric))
                        for prediction in predicted_by_repeat.values()
                    ]
                    valid = [value for value in values if value is not None]
                    if valid:
                        prediction_mean[metric] = mean(valid)
            for repeat_name in sorted(repeat_names):
                real_run_dir = real_dir / repeat_name
                sim_run_dir = sim_dir / repeat_name
                real_ok = is_successful_run(real_run_dir / "status.txt")
                sim_ok = is_successful_run(sim_run_dir / "status.txt")
                real = load_benchmark(real_run_dir / "benchmark.jsonl") if real_ok else None
                matched_predicted = (
                    load_json(sim_run_dir / "simulator-output" / "metrics.json") if sim_ok else None
                )
                predicted = (
                    prediction_mean or None
                    if args.prediction_aggregation == "mean-per-case"
                    else matched_predicted
                )
                prediction_samples = (
                    len(predicted_by_repeat)
                    if args.prediction_aggregation == "mean-per-case"
                    else (1 if matched_predicted is not None else 0)
                )
                row: dict[str, Any] = {
                    "model": model_dir.name,
                    "case": case_name,
                    "repeat": repeat_name.removeprefix("repeat-"),
                    "status": "ok" if real_ok and real is not None and predicted is not None else "incomplete",
                    "prediction_repeat_policy": args.prediction_aggregation,
                    "prediction_samples": prediction_samples,
                    "num_prompts": case.get("num_prompts"),
                    "input_tokens_target": case.get("input_tokens"),
                    "output_tokens_target": case.get("output_tokens"),
                    "request_rate": case.get("request_rate"),
                }
                for metric in METRICS:
                    real_value = number(real.get(metric)) if real else None
                    predicted_value = number(predicted.get(metric)) if predicted else None
                    signed = relative_error(real_value, predicted_value)
                    row[f"real_{metric}"] = real_value
                    row[f"simulated_{metric}"] = predicted_value
                    row[f"signed_error_pct_{metric}"] = signed
                    row[f"absolute_error_pct_{metric}"] = abs(signed) if signed is not None else None
                repeat_rows.append(row)

    repeat_fields = [
        "model", "case", "repeat", "status", "prediction_repeat_policy", "prediction_samples", "num_prompts", "input_tokens_target", "output_tokens_target", "request_rate"
    ]
    for metric in METRICS:
        repeat_fields += [
            f"real_{metric}", f"simulated_{metric}", f"signed_error_pct_{metric}", f"absolute_error_pct_{metric}"
        ]
    write_csv(analysis / "paired_repeat_metrics.csv", repeat_rows, repeat_fields)

    parameter_fields = list(parameters[0]) if parameters else ["model"]
    write_csv(analysis / "model_parameters.csv", parameters, parameter_fields)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in repeat_rows:
        grouped[(row["model"], row["case"])].append(row)
    case_rows: list[dict[str, Any]] = []
    for (model, case), rows in sorted(grouped.items()):
        base = rows[0]
        summary: dict[str, Any] = {
            "model": model,
            "case": case,
            "runs": len(rows),
            "complete_pairs": sum(row["status"] == "ok" for row in rows),
            "prediction_repeat_policy": args.prediction_aggregation,
            "prediction_samples": max((int(row["prediction_samples"]) for row in rows), default=0),
            "num_prompts": base["num_prompts"],
            "input_tokens_target": base["input_tokens_target"],
            "output_tokens_target": base["output_tokens_target"],
            "request_rate": base["request_rate"],
        }
        for metric in METRICS:
            for source, prefix in (("real", "real"), ("simulated", "simulated"), ("signed_error_pct", "signed_error_pct"), ("absolute_error_pct", "absolute_error_pct")):
                values = [
                    number(row.get(f"{source}_{metric}"))
                    for row in rows
                    if row["status"] == "ok"
                ]
                result = stats(value for value in values if value is not None)
                for field, value in result.items():
                    summary[f"{prefix}_{metric}_{field}"] = value
        case_rows.append(summary)

    case_fields = [
        "model", "case", "runs", "complete_pairs", "prediction_repeat_policy", "prediction_samples", "num_prompts", "input_tokens_target", "output_tokens_target", "request_rate"
    ]
    for metric in METRICS:
        for prefix in ("real", "simulated", "signed_error_pct", "absolute_error_pct"):
            case_fields += [f"{prefix}_{metric}_{field}" for field in ("n", "mean", "min", "max", "std", "variance")]
    write_csv(analysis / "case_metric_statistics.csv", case_rows, case_fields)

    model_metric_rows: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in repeat_rows}):
        model_records = [row for row in repeat_rows if row["model"] == model and row["status"] == "ok"]
        for metric in METRICS:
            signed = stats(number(row.get(f"signed_error_pct_{metric}")) for row in model_records if number(row.get(f"signed_error_pct_{metric}")) is not None)
            absolute = stats(number(row.get(f"absolute_error_pct_{metric}")) for row in model_records if number(row.get(f"absolute_error_pct_{metric}")) is not None)
            model_metric_rows.append({
                "model": model,
                "metric": metric,
                "paired_runs": len(model_records),
                "signed_error_mean_pct": signed["mean"],
                "signed_error_min_pct": signed["min"],
                "signed_error_max_pct": signed["max"],
                "signed_error_std_pct": signed["std"],
                "signed_error_variance_pct2": signed["variance"],
                "mape_mean_pct": absolute["mean"],
                "absolute_error_min_pct": absolute["min"],
                "absolute_error_max_pct": absolute["max"],
                "absolute_error_std_pct": absolute["std"],
                "absolute_error_variance_pct2": absolute["variance"],
            })
    model_metric_fields = list(model_metric_rows[0]) if model_metric_rows else ["model", "metric"]
    write_csv(analysis / "model_metric_statistics.csv", model_metric_rows, model_metric_fields)

    params_by_model = {row["model"]: row for row in parameters}
    predictor_note = predictor_scope_note(root)
    report = [
        "# PR #33824 A100 精度矩阵结果",
        "",
        "## 统计口径",
        "",
        "- 每个模型的三个固定 `random-ids` 工作负载各运行 5 次；同一模型/工作负载的五次运行保持相同 seed。",
        f"- `real` 是 A100 PCIe 上加载完整 checkpoint 的原生 SGLang；{predictor_note}",
        "- signed error = `(预测 − 实机) / 实机 × 100%`；MAPE 是 absolute error 的均值。方差和标准差采用样本统计（n−1）。",
        "- `incomplete` 运行保留在逐次 CSV 中，但不进入误差统计，也不会被填为 0。",
        "",
        "## 模型级误差（15 个配对运行；若有失败则少于 15）",
        "",
        "| 模型 | 架构 | Checkpoint GB | TTFT MAPE / 方差 | TPOT MAPE / 方差 | E2E MAPE / 方差 | 总吞吐 MAPE / 方差 |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    metric_map = {(row["model"], row["metric"]): row for row in model_metric_rows}
    for model in sorted(params_by_model):
        param = params_by_model[model]
        cells = []
        for metric in ("mean_ttft_ms", "mean_tpot_ms", "mean_e2e_latency_ms", "total_throughput"):
            item = metric_map.get((model, metric), {})
            cells.append(f"{fmt(item.get('mape_mean_pct'))}% / {fmt(item.get('absolute_error_variance_pct2'))}")
        report.append(
            f"| {model} | {param['model_type']} | {fmt(param['checkpoint_size_gb'])} | " + " | ".join(cells) + " |"
        )
    report += [
        "",
        "## 按模型和工作负载的统计",
        "",
        "完整的实机/预测均值、最小值、最大值、标准差和样本方差见 `case_metric_statistics.csv`；每次配对原值见 `paired_repeat_metrics.csv`。",
        "",
        "| 模型 | 工作负载 | 配对 | TTFT MAPE ± std | TPOT MAPE ± std | E2E MAPE ± std | 总吞吐 MAPE ± std |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in case_rows:
        report.append(
            "| {model} | {case} | {pairs}/{runs} | {ttft}% ± {ttft_std} | {tpot}% ± {tpot_std} | {e2e}% ± {e2e_std} | {throughput}% ± {throughput_std} |".format(
                model=row["model"], case=row["case"], pairs=row["complete_pairs"], runs=row["runs"],
                ttft=fmt(row["absolute_error_pct_mean_ttft_ms_mean"]), ttft_std=fmt(row["absolute_error_pct_mean_ttft_ms_std"]),
                tpot=fmt(row["absolute_error_pct_mean_tpot_ms_mean"]), tpot_std=fmt(row["absolute_error_pct_mean_tpot_ms_std"]),
                e2e=fmt(row["absolute_error_pct_mean_e2e_latency_ms_mean"]), e2e_std=fmt(row["absolute_error_pct_mean_e2e_latency_ms_std"]),
                throughput=fmt(row["absolute_error_pct_total_throughput_mean"]), throughput_std=fmt(row["absolute_error_pct_total_throughput_std"]),
            )
        )
    (analysis / "accuracy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote analysis to {analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
