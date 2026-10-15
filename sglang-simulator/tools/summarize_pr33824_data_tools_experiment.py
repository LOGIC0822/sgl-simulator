#!/usr/bin/env python3
"""Aggregate the PR #33824 data-collector ML experiment into a report.

The report deliberately keeps calibration split metrics, teacher-forced batch
scoring, and end-to-end replay metrics separate.  They answer different
questions and should not be averaged into one "accuracy" value.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRICS = (
    ("mean_ttft_ms", "TTFT (ms)"),
    ("mean_tpot_ms", "TPOT (ms)"),
    ("mean_e2e_latency_ms", "E2E (ms)"),
    ("total_throughput", "总吞吐 (token/s)"),
)


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def require_complete(path: Path, marker: str) -> None:
    if not path.is_file() or path.read_text(encoding="utf-8").strip() != "complete":
        raise RuntimeError(f"missing completion marker: {marker}")


def required_row(
    rows: list[dict[str, str]],
    **criteria: str,
) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row matching {criteria}, found {len(matches)}")
    return matches[0]


def as_float(value: str) -> float:
    return float(value)


def as_int(value: str) -> int:
    return int(float(value))


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


@dataclass(frozen=True)
class ModelResult:
    model: str
    root: Path
    experiment_metadata: dict[str, str]
    calibration_metadata: dict[str, str]
    holdout_metadata: dict[str, str]
    package_versions: dict[str, str]
    gpu_identity: str
    calibration_rows: int
    calibration_validation: dict[str, str]
    teacher_forced: dict[str, str]
    e2e: dict[str, dict[str, str]]
    distribution_audit: dict[str, dict[str, str]]


def collect_model(root: Path, model: str, repeats: int) -> ModelResult:
    model_root = root / model
    calibration = model_root / "calibration"
    single_calibration = calibration / "BASELINE_COLLECTION_COMPLETE"
    if single_calibration.is_file():
        calibration_run = calibration
    else:
        calibration_runs = sorted((calibration / "real").glob("repeat-*"))
        if not calibration_runs:
            raise RuntimeError(f"missing calibration collection: {calibration}")
        for run in calibration_runs:
            require_complete(run / "BASELINE_COLLECTION_COMPLETE", str(run))
        calibration_run = calibration_runs[0]
    calibration_metadata = read_key_values(calibration_run / "run_metadata.txt")
    package_versions = read_key_values(calibration_run / "package_versions.txt")
    gpu_identity = (calibration_run / "hardware" / "gpu_identity.csv").read_text(
        encoding="utf-8"
    ).strip()
    experiment_metadata_path = root.parent / "run_metadata.txt"
    experiment_metadata = (
        read_key_values(experiment_metadata_path)
        if experiment_metadata_path.is_file()
        else {}
    )

    dataset_rows = read_csv(calibration / "training" / "eval" / "dataset_cases.csv")
    calibration_rows = sum(as_int(row["rows"]) for row in dataset_rows)
    split_rows = read_csv(calibration / "training" / "eval" / "split_summary.csv")
    calibration_validation = required_row(
        split_rows, loss="p50", split="case_stratified"
    )

    holdout_real = model_root / "holdout" / "real"
    holdout_simulator = model_root / "holdout" / "simulator"
    for repeat in range(1, repeats + 1):
        require_complete(
            holdout_real / f"repeat-{repeat}" / "BASELINE_COLLECTION_COMPLETE",
            f"{model}/real/repeat-{repeat}",
        )
        require_complete(
            holdout_simulator / f"repeat-{repeat}" / "ML_REPLAY_COMPLETE",
            f"{model}/simulator/repeat-{repeat}",
        )
    holdout_metadata = read_key_values(holdout_real / "repeat-1" / "run_metadata.txt")

    teacher_rows = read_csv(model_root / "holdout" / "teacher_forced" / "summary.csv")
    teacher_forced = required_row(teacher_rows, group="all", value="all")
    e2e_rows = read_csv(model_root / "holdout" / "e2e_comparison" / "summary.csv")
    e2e: dict[str, dict[str, str]] = {}
    for metric, _ in METRICS:
        row = required_row(e2e_rows, group="metric", value=metric)
        if as_int(row["n"]) != repeats:
            raise RuntimeError(
                f"{model}/{metric}: expected {repeats} paired replays, got {row['n']}"
            )
        e2e[metric] = row
    distribution_path = model_root / "holdout" / "distribution_audit.csv"
    distribution_audit = (
        {row["phase"]: row for row in read_csv(distribution_path)}
        if distribution_path.is_file()
        else {}
    )
    return ModelResult(
        model=model,
        root=model_root,
        experiment_metadata=experiment_metadata,
        calibration_metadata=calibration_metadata,
        holdout_metadata=holdout_metadata,
        package_versions=package_versions,
        gpu_identity=gpu_identity,
        calibration_rows=calibration_rows,
        calibration_validation=calibration_validation,
        teacher_forced=teacher_forced,
        e2e=e2e,
        distribution_audit=distribution_audit,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(results: list[ModelResult], figures: Path) -> list[Path]:
    # The shared login environment does not make ~/.config/matplotlib writable.
    # Keep Matplotlib's small cache out of the experiment and document roots.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sglang-simulator-matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np

    figures.mkdir(parents=True, exist_ok=False)
    labels = [result.model.replace("-Instruct", "") for result in results]
    positions = np.arange(len(results))

    direct = [as_float(result.teacher_forced["mape_pct"]) for result in results]
    e2e = [as_float(result.e2e["mean_e2e_latency_ms"]["mape_pct"]) for result in results]
    figure, axis = plt.subplots(figsize=(max(8.2, len(results) * 1.15), 4.5))
    width = 0.36
    axis.bar(positions - width / 2, direct, width, label="Teacher-forced batch")
    axis.bar(positions + width / 2, e2e, width, label="End-to-end E2E")
    axis.set_ylabel("MAPE (%)")
    axis.set_xticks(positions, labels, rotation=30 if len(results) > 5 else 12, ha="right")
    axis.set_title("ML prediction: batch scoring vs. end-to-end E2E")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    direct_e2e = figures / "teacher_forced_vs_e2e_mape.png"
    figure.savefig(direct_e2e, dpi=180)
    plt.close(figure)

    metric_names = ["TTFT", "TPOT", "E2E", "Total throughput"]
    figure, axis = plt.subplots(figsize=(max(9.2, len(results) * 0.9), 4.8))
    width = min(0.22, 0.78 / len(results))
    for index, result in enumerate(results):
        values = [as_float(result.e2e[key]["mape_pct"]) for key, _ in METRICS]
        offset = (index - (len(results) - 1) / 2) * width
        axis.bar(np.arange(len(METRICS)) + offset, values, width, label=labels[index])
    axis.set_ylabel("MAPE (%)")
    axis.set_xticks(np.arange(len(METRICS)), metric_names)
    axis.set_title("End-to-end replay MAPE by metric")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    metric_plot = figures / "e2e_metric_mape.png"
    figure.savefig(metric_plot, dpi=180)
    plt.close(figure)
    return [direct_e2e, metric_plot]


def make_markdown(
    results: list[ModelResult],
    markdown_path: Path,
    figures: list[Path],
    historical_root: Path | None,
    repeats: int,
) -> str:
    first = results[0]
    metadata = first.calibration_metadata
    package = first.package_versions
    same_distribution = (
        first.experiment_metadata.get("protocol")
        == "same_workload_independent_train_holdout"
    )
    title = (
        "# A100 PCIe：SGLang 数据采集式 ML 预测器同分布复测"
        if same_distribution
        else "# A100 PCIe：SGLang 数据采集式 ML 预测器复测"
    )
    if same_distribution:
        scope_detail = (
            "每个 checkpoint 用 {train_runs} 条新采集的 "
            "saturation_1024x256_inf trace 训练，并只在来自独立参考目录的 "
            "{test_runs} 条同工作负载 trace 上评分和回放。训练与测试的 "
            "ScheduleBatch 行没有复用。"
        ).format(
            train_runs=first.experiment_metadata.get("training_real_repeats", "unknown"),
            test_runs=first.experiment_metadata.get("heldout_real_repeats", repeats),
        )
    else:
        scope_detail = (
            f"每个模型的真实服务运行 {repeats} 次 saturation_1024x256_inf 负载。"
            "每次真实运行产生独立 raw request trace；CPU 模拟器回放同一次 "
            "trace 的 token IDs、完成输出长度和相对到达时间。"
        )
    lines = [
        title,
        "",
        "## 范围",
        "",
        "本轮覆盖 {count} 个 checkpoint：{models}。每个 checkpoint 单独采集、单独训练、单独回放；没有跨模型复用 ML bundle。".format(
            count=len(results), models="、".join(result.model for result in results)
        ),
        "",
        scope_detail,
        "",
        "## 本轮结果",
        "",
        "| 模型 | 校准 batch 数 | 实际 batch 评分 MAPE | 实际 batch MAE (ms) | E2E 实测均值 (ms) | E2E 预测均值 (ms) | E2E MAPE | TTFT MAPE | TPOT MAPE | 总吞吐 MAPE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        e2e = result.e2e["mean_e2e_latency_ms"]
        lines.append(
            "| {model} | {calibration_rows} | {batch_mape}% | {batch_mae} | {real} | {predicted} | "
            "{e2e_mape}% | {ttft}% | {tpot}% | {throughput}% |".format(
                model=result.model,
                calibration_rows=result.calibration_rows,
                batch_mape=fmt(as_float(result.teacher_forced["mape_pct"])),
                batch_mae=fmt(as_float(result.teacher_forced["mae_ms"])),
                real=fmt(as_float(e2e["real_mean"])),
                predicted=fmt(as_float(e2e["predicted_mean"])),
                e2e_mape=fmt(as_float(e2e["mape_pct"])),
                ttft=fmt(as_float(result.e2e["mean_ttft_ms"]["mape_pct"])),
                tpot=fmt(as_float(result.e2e["mean_tpot_ms"]["mape_pct"])),
                throughput=fmt(as_float(result.e2e["total_throughput"]["mape_pct"])),
            )
        )
    lines.extend(
        [
            "",
            "实际 batch 评分使用真实服务记录下来的 Scheduler batch 组成，预测不会改变后续 batch。端到端回放会让预测时延参与调度，因而两行结果不应互相替代。",
            "",
            f"![实际 batch 与端到端 E2E 的 MAPE 对比]({relpath(figures[0], markdown_path.parent)})",
            "",
            f"![端到端各指标的 MAPE 对比]({relpath(figures[1], markdown_path.parent)})",
            "",
            "## 端到端真实值与预测值",
            "",
            "表中均值和标准差来自 5 个独立真机运行及其一一对应的回放。带符号偏差为 `(预测−实测)/实测×100%`。",
            "",
            "| 模型 | 指标 | 实测均值 | 实测标准差 | 预测均值 | 预测标准差 | 带符号偏差 | MAPE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        for key, label in METRICS:
            row = result.e2e[key]
            lines.append(
                "| {model} | {label} | {real} | {real_std} | {predicted} | {predicted_std} | {signed}% | {mape}% |".format(
                    model=result.model,
                    label=label,
                    real=fmt(as_float(row["real_mean"])),
                    real_std=fmt(as_float(row["real_std"])),
                    predicted=fmt(as_float(row["predicted_mean"])),
                    predicted_std=fmt(as_float(row["predicted_std"])),
                    signed=fmt(as_float(row["mean_signed_error_pct"])),
                    mape=fmt(as_float(row["mape_pct"])),
                )
            )
    lines.extend(
        [
            "",
            "## 校准与运行配置",
            "",
            "| 项目 | 设置 |",
            "| --- | --- |",
            f"| 真实硬件 | {first.gpu_identity} |",
            f"| 源码 revision | {metadata.get('source_revision', 'unknown')} |",
            f"| 采集器来源 | {metadata.get('collector_source', 'unknown')} @ {metadata.get('collector_commit', 'unknown')} |",
            f"| Python / PyTorch | {package.get('python', 'unknown')} / {package.get('torch', 'unknown')}（CUDA {package.get('torch_cuda', 'unknown')}） |",
            f"| SGLang 包版本记录 | {package.get('sglang', 'unknown')}；实际代码以 revision 为准 |",
            f"| sgl-kernel 包版本记录 | {package.get('sglang-kernel', 'unknown')}；运行时覆盖路径见各模型 package_versions.txt |",
            f"| Attention / kernel | {metadata.get('attention_backend', 'unknown')} / {metadata.get('kernel_policy', 'unknown')} |",
            f"| 调度设置 | overlap disabled；CUDA graph disabled；context={metadata.get('context_length', 'unknown')}；max_running_requests={metadata.get('max_running_requests', 'unknown')}；chunked_prefill_size={metadata.get('chunked_prefill_size', 'unknown')} |",
            f"| 长度采样 | random_range_ratio={metadata.get('random_range_ratio', 'unknown')}：输入和输出长度分别在 1 到命名上限之间采样 |",
            "| 训练负载 | {workload}；真实训练 trace 数={train_runs}；训练 seed={seeds}；warmup_requests={warmup}；forward modes={modes} |".format(
                workload=first.experiment_metadata.get(
                    "training_case", "latency_256x128_r1 与 service_1024x256_r4"
                ),
                train_runs=first.experiment_metadata.get("training_real_repeats", "1"),
                seeds=first.experiment_metadata.get("training_seeds", metadata.get("seed", "unknown")),
                warmup=metadata.get("warmup_requests", "unknown"),
                modes=metadata.get("train_forward_modes", "unknown"),
            ),
            "| 测试负载 | {workload}；真实 trace 数={test_runs}；每条 trace 单独回放 |".format(
                workload=first.experiment_metadata.get("heldout_case", "saturation_1024x256_inf"),
                test_runs=first.experiment_metadata.get("heldout_real_repeats", repeats),
            ),
            "",
            "| 模型 | 静态显存比例 | 校准集内 p50 分层切分 MAPE | 校准集内 p50 MAE (ms) | 持出实际 batch 数 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        lines.append(
            "| {model} | {memory} | {split_mape}% | {split_mae} | {batches} |".format(
                model=result.model,
                memory=result.calibration_metadata.get("mem_fraction_static", "unknown"),
                split_mape=fmt(as_float(result.calibration_validation["MAPE_pct"])),
                split_mae=fmt(as_float(result.calibration_validation["MAE_ms"])),
                batches=result.teacher_forced["n"],
            )
        )
    lines.extend(
        [
            "",
            "校准集内分层切分只用于检查训练数据拟合情况；最终表中的实际 batch 评分和端到端结果来自未用于训练的 held-out trace。",
            "",
        ]
    )
    if any(result.distribution_audit for result in results):
        lines.extend(
            [
                "## 训练/测试 Scheduler batch 覆盖",
                "",
                "下表来自真实设备同步的 Scheduler batch。`测试超出训练范围`表示测试行至少有一个 18 维输入特征落在训练逐维范围外；它描述支持集覆盖，不等同于预测误差。",
                "",
                "| 模型 | phase | 训练 batch | 测试 batch | 完全同形状 | 测试超出训练范围 | 训练/测试 batch p50 | 训练/测试 extend p50 | 训练/测试 past p50 |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for result in results:
            for phase in ("all", "prefill", "decode"):
                row = result.distribution_audit.get(phase)
                if not row:
                    continue
                lines.append(
                    "| {model} | {phase} | {train_rows} | {test_rows} | {exact_feature_match_pct}% | {test_outside_train_feature_range_pct}% | {train_batch_size_p50} / {test_batch_size_p50} | {train_sum_extend_p50} / {test_sum_extend_p50} | {train_sum_past_p50} / {test_sum_past_p50} |".format(
                        model=result.model, **row
                    )
                )
        lines.append("")
    lines.extend(
        [
            "## 结果边界",
            "",
            "- 结果只适用于表中 checkpoint、A100 PCIe、当前源码 revision、Triton attention、关闭 overlap 和 CUDA graph 的组合。",
            "- ML predictor 只给 forward 逻辑时长；真实请求处理、CPU 调度和网络处理不会被单独建模。端到端误差包含这些因素与调度反馈的共同影响。",
            "- 同一 seed 的五次重复用于观察同一负载下的运行波动，不代表不同请求分布下的泛化误差。",
            "- 每个 checkpoint 的结果只适用于其自身的模型格式、量化方式和上下文设置；不同架构之间不共享 ML bundle。",
            "",
        ]
    )
    if historical_root is not None:
        old_single = historical_root / "single_step_mape.png"
        old_e2e = historical_root / "end_to_end_e2e_mape.png"
        lines.extend(
            [
                "## 已有三种预测方式图表（历史实验）",
                "",
                "下图保留此前 3 个 dense checkpoint 上的 AIC Silicon（退化后端、SM80 兼容 kernel）和旧 ExtraTrees ML 实验记录。两批 AIC 实验使用的真实基线并非只改变一个变量；旧 ML 也不是本轮的 collector + HGB bundle。因此图仅用于追溯历史结果，不能与本轮表格作单一变量因果比较。",
                "",
                f"![三种预测方式的固定单步误差对比]({relpath(old_single, markdown_path.parent)})",
                "",
                f"![三种预测方式的端到端 E2E 误差对比]({relpath(old_e2e, markdown_path.parent)})",
                "",
                f"历史图表的逐项数据：`{relpath(historical_root / 'comparison_data.csv', markdown_path.parent)}`。",
                "",
            ]
        )
    lines.extend(
        [
            "## 结果文件",
            "",
            "- `aggregate/model_summary.csv`：模型级汇总。",
            "- `aggregate/e2e_detail.csv`：各指标的实测、预测、标准差和误差。",
            "- 每个模型的 `holdout/teacher_forced/`：实际 Scheduler batch 的逐 batch 评分。",
            "- 每个模型的 `holdout/e2e_comparison/`：5 次真实 trace 与回放的逐次配对。",
        ]
    )
    return "\n".join(lines) + "\n"


def summary_rows(results: list[ModelResult]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for result in results:
        model_rows.append(
            {
                "model": result.model,
                "calibration_rows": result.calibration_rows,
                "heldout_batches": result.teacher_forced["n"],
                "teacher_forced_mape_pct": result.teacher_forced["mape_pct"],
                "teacher_forced_mae_ms": result.teacher_forced["mae_ms"],
                "e2e_mape_pct": result.e2e["mean_e2e_latency_ms"]["mape_pct"],
                "ttft_mape_pct": result.e2e["mean_ttft_ms"]["mape_pct"],
                "tpot_mape_pct": result.e2e["mean_tpot_ms"]["mape_pct"],
                "throughput_mape_pct": result.e2e["total_throughput"]["mape_pct"],
                "source_revision": result.calibration_metadata.get("source_revision", ""),
                "mem_fraction_static": result.calibration_metadata.get("mem_fraction_static", ""),
                "exact_feature_match_pct": result.distribution_audit.get("all", {}).get(
                    "exact_feature_match_pct", ""
                ),
                "test_outside_train_feature_range_pct": result.distribution_audit.get(
                    "all", {}
                ).get("test_outside_train_feature_range_pct", ""),
            }
        )
        for metric, label in METRICS:
            detail_rows.append(
                {
                    "model": result.model,
                    "metric": metric,
                    "label": label,
                    **result.e2e[metric],
                }
            )
    return model_rows, detail_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--method-comparison-root", type=Path)
    args = parser.parse_args(argv)
    args.models_root = args.models_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.markdown = args.markdown.expanduser().resolve()
    args.method_comparison_root = (
        args.method_comparison_root.expanduser().resolve()
        if args.method_comparison_root
        else None
    )
    args.models = [value.strip() for value in args.models.split(",") if value.strip()]
    if not args.models:
        parser.error("--models must name at least one model")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.out_dir}")
    if args.markdown.exists():
        parser.error(f"refusing to overwrite Markdown report: {args.markdown}")
    if args.method_comparison_root:
        for filename in ("comparison_data.csv", "single_step_mape.png", "end_to_end_e2e_mape.png"):
            if not (args.method_comparison_root / filename).is_file():
                parser.error(f"missing historical comparison artifact: {filename}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = [collect_model(args.models_root, model, args.repeats) for model in args.models]
    revisions = {result.calibration_metadata.get("source_revision") for result in results}
    if len(revisions) != 1:
        raise RuntimeError(f"models used different source revisions: {sorted(revisions)}")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    figures = make_figures(results, args.out_dir / "figures")
    model_rows, detail_rows = summary_rows(results)
    write_csv(
        args.out_dir / "model_summary.csv",
        model_rows,
        list(model_rows[0]),
    )
    write_csv(
        args.out_dir / "e2e_detail.csv",
        detail_rows,
        list(detail_rows[0]),
    )
    markdown = make_markdown(
        results,
        args.markdown,
        figures,
        args.method_comparison_root,
        args.repeats,
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown, encoding="utf-8")
    print(f"models={len(results)} out={args.out_dir} markdown={args.markdown}")


if __name__ == "__main__":
    main()
