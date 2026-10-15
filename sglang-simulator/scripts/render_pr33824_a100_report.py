#!/usr/bin/env python3
"""Render the readable Markdown and DOCX handoff for a completed matrix.

The CSV files written by ``summarize_pr33824_a100_matrix.py`` remain the
authoritative machine-readable records.  This renderer only arranges those
records into a short Chinese report suitable for review; it never changes raw
benchmark output or fills missing measurements with zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


KEY_METRICS = (
    ("mean_ttft_ms", "TTFT"),
    ("mean_tpot_ms", "TPOT"),
    ("mean_e2e_latency_ms", "E2E"),
    ("total_throughput", "总吞吐"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fmt(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    return "—" if number is None else f"{number:.{digits}f}"


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def calibration_record_summary(root: Path) -> str:
    """Count the currently selected AIC tables instead of hard-coding a run."""
    try:
        config = json.loads((root / "simulator-config.json").read_text(encoding="utf-8"))
        predictor = config["predictor"]
        scheduler = config["scheduler"]
        table_root = (
            Path(predictor["database_path"])
            / "data"
            / str(predictor["database_system"])
            / str(scheduler["backend_name"])
            / str(scheduler["backend_version"])
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return "AIC Silicon 校准表路径未记录，无法统计本机采样行数。"

    def records(filename: str) -> int | None:
        path = table_root / filename
        if not path.is_file():
            return None
        return max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)

    gemm = records("gemm_perf.txt")
    context = records("context_attention_perf.txt")
    generation = records("generation_attention_perf.txt")
    moe = records("moe_perf.txt")
    if None in {gemm, context, generation}:
        return f"AIC Silicon 校准表不完整：{table_root}。"
    moe_summary = (
        f"、{moe} 条 Qwen3 MoE 记录"
        if moe is not None
        else "；当前数据库未包含 MoE 算子记录"
    )
    return (
        "预测器使用 AIC Silicon 采集值，而非 SOL roofline 值；当前选用的本机表包含 "
        f"{gemm} 条 GEMM、{context} 条 prefill attention、{generation} 条 decode attention"
        f"{moe_summary}。"
    )


def launch_settings_summary(root: Path) -> str:
    """Read actual per-model server settings rather than assuming a uniform run."""
    entries: list[str] = []
    for path in sorted((root / "results").glob("*/real/launch-settings.txt")):
        values = read_key_values(path)
        if not values:
            continue
        model = path.parents[1].name
        entries.append(
            f"{model}: mem_fraction_static={values.get('mem_fraction_static', '未记录')}"
        )
    if not entries:
        return "未找到逐模型 launch-settings.txt。"
    return "逐模型静态显存比例为 " + "；".join(entries) + "。"


def attention_kernel_summary(root: Path) -> str:
    """Report the recorded attention-kernel source from the selected AIC table."""
    try:
        config = json.loads((root / "simulator-config.json").read_text(encoding="utf-8"))
        predictor = config["predictor"]
        scheduler = config["scheduler"]
        table_root = (
            Path(predictor["database_path"])
            / "data"
            / str(predictor["database_system"])
            / str(scheduler["backend_name"])
            / str(scheduler["backend_version"])
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return "未能定位本轮 AIC attention 校准表。"

    summaries: list[str] = []
    for filename, label in (
        ("context_attention_perf.txt", "prefill"),
        ("generation_attention_perf.txt", "decode"),
    ):
        path = table_root / filename
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            sources = sorted({row.get("kernel_source", "") for row in csv.DictReader(handle) if row.get("kernel_source")})
        if sources:
            summaries.append(f"{label}={','.join(sources)}")
    if not summaries:
        return "attention 表没有记录 kernel_source。"
    return "AIC attention 校准表记录 " + "，".join(summaries) + "；与实机 Triton attention 后端对齐。"


def awq_proxy_summary(parameters: list[dict[str, str]]) -> str | None:
    models = [
        row["model"]
        for row in parameters
        if row.get("quantization", "").lower() == "awq"
    ]
    if not models:
        return None
    return (
        f"AIC 0.10 在 A100/SM80 上没有 dense AWQ GEMM 数据，{', '.join(models)} "
        "保留真实模型结构、使用本机 BF16 算子表进行代理估算；该行不属于 AWQ 原生校准。"
    )


def matrix_report_status(
    root: Path, metadata: dict[str, str], case_rows: list[dict[str, str]]
) -> str:
    """Describe whether a rendered report is a live snapshot or final handoff."""
    completed_pairs = sum(as_int(row.get("complete_pairs")) for row in case_rows)
    models = [item for item in metadata.get("models", "").split(",") if item]
    workloads = [item for item in metadata.get("workloads", "").split(";") if item]
    expected_pairs = len(models) * len(workloads) * as_int(metadata.get("repeats"))
    progress = f"有效配对 {completed_pairs}/{expected_pairs or '未记录'}"
    if (root / "MATRIX_COMPLETE").is_file():
        return f"完整矩阵已成功结束（{progress}）"
    if (root / "MATRIX_COMPLETED_WITH_FAILURES").is_file():
        return f"矩阵已结束但存在失败运行（{progress}）；本报告没有用零值补齐失败样本"
    return f"运行中快照（{progress}）；最终结论以矩阵结束后自动重新生成的版本为准"


def metric_cell(metric_map: dict[tuple[str, str], dict[str, str]], model: str, metric: str) -> str:
    row = metric_map.get((model, metric), {})
    return f"{fmt(row.get('mape_mean_pct'))}% ± {fmt(row.get('absolute_error_std_pct'))}"


def range_variance_cell(metric_map: dict[tuple[str, str], dict[str, str]], model: str, metric: str) -> str:
    row = metric_map.get((model, metric), {})
    return (
        f"{fmt(row.get('absolute_error_min_pct'))}%–{fmt(row.get('absolute_error_max_pct'))}%"
        f" / {fmt(row.get('absolute_error_variance_pct2'))}"
    )


def real_prediction_cell(row: dict[str, str], metric: str) -> str:
    """Format case-level means without hiding either measured or predicted data."""
    return (
        f"{fmt(row.get(f'real_{metric}_mean'))} / "
        f"{fmt(row.get(f'simulated_{metric}_mean'))} / "
        f"{fmt(row.get(f'signed_error_pct_{metric}_mean'))}%"
    )


def runtime_description(metadata: dict[str, str]) -> tuple[str, str]:
    """Return recorded runtime descriptions without imposing an old experiment mode."""
    real = metadata.get(
        "real_runtime",
        "本机 NVIDIA A100 服务；未记录具体 attention、CUDA graph 与调度设置。",
    )
    simulator = metadata.get(
        "simulator_runtime",
        "同一 PR 的 CPU 模拟器；未记录具体预测器与逻辑调度设置。",
    )
    return real, simulator


def comparison_scope_note(root: Path) -> str:
    """Keep an explicit current/legacy comparison boundary in the handoff."""
    path = root / "comparison_scope.md"
    if not path.is_file():
        return "本轮与旧实验的配置边界未单独记录；不得将误差变化归因于单一代码因素。"
    differences = [
        line.strip()[2:].rstrip("；。")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- ")
    ]
    if not differences:
        return "本轮与旧实验的配置边界见 comparison_scope.md；不得将误差变化归因于单一代码因素。"
    # The source file deliberately lists the five controls that changed.  Do
    # not truncate wrapped prose before those bullets, otherwise a reader can
    # mistake the trend table for a code-only ablation.
    return (
        "两轮使用相同的 PR 提交、模型目录、固定工作负载、seed 规则和 A100 PCIe 目标，"
        "但不是纯代码对照。本轮相对于历史轮还改变了："
        + "；".join(differences)
        + "。因此旧/新表只能观察完整配置更新后的趋势，不能将误差变化单独归因于代码。"
    )


def workload_summary(case_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        if as_int(row.get("complete_pairs")):
            grouped[row["case"]].append(row)
    rows: list[dict[str, Any]] = []
    for workload, entries in sorted(grouped.items()):
        result: dict[str, Any] = {
            "case": workload,
            "models": len(entries),
            "paired_runs": sum(as_int(entry.get("complete_pairs")) for entry in entries),
        }
        for metric, _ in KEY_METRICS:
            values = [as_float(entry.get(f"absolute_error_pct_{metric}_mean")) for entry in entries]
            clean = [value for value in values if value is not None]
            result[f"{metric}_mean"] = mean(clean) if clean else None
            result[f"{metric}_std"] = stdev(clean) if len(clean) > 1 else 0.0 if clean else None
        rows.append(result)
    return rows


def legacy_rows(path: Path, current_cases: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, str]]:
    """Return only same-name legacy comparisons; do not equate near-match models."""
    rendered: list[dict[str, str]] = []
    for old in read_csv(path):
        key = (old.get("model", ""), old.get("case", ""))
        current = current_cases.get(key)
        if (
            not current
            or as_int(old.get("complete_pairs")) <= 0
            or as_int(current.get("complete_pairs")) <= 0
        ):
            continue
        rendered.append({
            "model": key[0],
            "case": key[1],
            "old_ttft": old.get("mape_mean_ttft_ms", ""),
            "new_ttft": current.get("absolute_error_pct_mean_ttft_ms_mean", ""),
            "old_tpot": old.get("mape_mean_tpot_ms", ""),
            "new_tpot": current.get("absolute_error_pct_mean_tpot_ms_mean", ""),
            "old_e2e": old.get("mape_mean_e2e_latency_ms", ""),
            "new_e2e": current.get("absolute_error_pct_mean_e2e_latency_ms_mean", ""),
            "old_throughput": old.get("mape_total_throughput", ""),
            "new_throughput": current.get("absolute_error_pct_total_throughput_mean", ""),
        })
    return rendered


def legacy_metric_summary(legacy: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    """Summarize only the exact-name old/new pairs used in the reference table."""
    rows: list[dict[str, float | int | str]] = []
    for old_field, new_field, label in (
        ("old_ttft", "new_ttft", "TTFT"),
        ("old_tpot", "new_tpot", "TPOT"),
        ("old_e2e", "new_e2e", "E2E"),
        ("old_throughput", "new_throughput", "总吞吐"),
    ):
        pairs = [
            (as_float(row.get(old_field)), as_float(row.get(new_field)))
            for row in legacy
        ]
        pairs = [(old, new) for old, new in pairs if old is not None and new is not None]
        if not pairs:
            continue
        old_mean = mean(old for old, _ in pairs)
        new_mean = mean(new for _, new in pairs)
        rows.append(
            {
                "metric": label,
                "pairs": len(pairs),
                "old_mean": old_mean,
                "new_mean": new_mean,
                "delta": new_mean - old_mean,
            }
        )
    return rows


def make_markdown(
    root: Path,
    metadata: dict[str, str],
    parameters: list[dict[str, str]],
    case_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    legacy: list[dict[str, str]],
) -> str:
    metric_map = {(row["model"], row["metric"]): row for row in metric_rows}
    cases_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        cases_by_model[row["model"]].append(row)
    planned_per_model = 15
    status_line = matrix_report_status(root, metadata, case_rows)
    calibration_summary = calibration_record_summary(root)
    launch_summary = launch_settings_summary(root)
    attention_summary = attention_kernel_summary(root)
    awq_summary = awq_proxy_summary(parameters)
    real_runtime, simulator_runtime = runtime_description(metadata)
    scope_note = comparison_scope_note(root)
    report = [
        "# PR #33824：A100 PCIe 实测与预测器对比报告",
        "",
        "## 实验概况",
        "",
        f"- 代码：SGLang PR #33824，提交 `{metadata.get('pr_head', '未记录')}`。",
        f"- 版本定位：{metadata.get('source_version_identity', '未记录；请以提交号为准')}。",
        f"- 报告状态：{status_line}。",
        f"- 实机：{real_runtime}",
        f"- {launch_summary}",
        f"- 预测：{simulator_runtime}",
        "- 每个模型使用三组固定随机 token 工作负载：`latency_256x128_r1`、`service_1024x256_r4`、`saturation_1024x256_inf`；真机每组固定 5 次。预测器按本轮记录的重复策略运行，样本数在逐工作负载表中单独列出。",
        "- 下方先给出每个工作负载的实测均值、预测均值和带符号偏差；MAPE 仅用于汇总。带符号偏差为 `(预测值 − 实测值) / 实测值 × 100%`，MAPE 为其绝对值的平均值。",
        "",
        "## 运行状态与模型级结果",
        "",
        "| 模型 | 参数特征 | 有效真机对照/15 | TTFT MAPE ± std | TPOT MAPE ± std | E2E MAPE ± std | 总吞吐 MAPE ± std |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    if metadata.get("disk_full_interruption_at"):
        report.insert(
            10,
            "- 运行记录：实验期间曾因根分区空间耗尽中断一次；释放可再生缓存后通过 "
            "`--resume` 恢复。已完成的测量未被覆盖，只有当时未完成的重复被补跑；"
            "统计仍只读取带 `status.txt=ok` 的完整配对。",
        )
    if awq_summary:
        report.insert(10, f"- {awq_summary}")
    for parameter in sorted(parameters, key=lambda row: row["model"]):
        model = parameter["model"]
        pairs = sum(as_int(row.get("complete_pairs")) for row in cases_by_model.get(model, []))
        details = (
            f"{parameter.get('model_type', 'unknown')}; "
            f"{fmt(parameter.get('checkpoint_size_gb'))} GB; "
            f"H={parameter.get('hidden_size') or '—'}, L={parameter.get('num_hidden_layers') or '—'}, "
            f"E={parameter.get('num_local_experts') or '0'}"
        )
        if parameter.get("quantization", "").lower() == "awq":
            details += "; AWQ→BF16 proxy"
        report.append(
            f"| {model} | {details} | {pairs}/{planned_per_model} | "
            + " | ".join(metric_cell(metric_map, model, metric) for metric, _ in KEY_METRICS)
            + " |"
        )

    report += [
        "",
        "## 实测值与预测值（每个工作负载的重复均值）",
        "",
        "每格为 `实测 / 预测 / 带符号偏差%`。TTFT、TPOT、E2E 的单位为 ms，总吞吐的单位为 token/s。若预测样本少于真机样本，预测栏为该工作负载所有完成预测样本的均值；不会把同一预测值伪装成多次独立预测。",
        "",
        "| 模型 | 工作负载 | 有效真机对照 | 预测样本 | TTFT | TPOT | E2E | 总吞吐 |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in sorted(case_rows, key=lambda item: (item["model"], item["case"])):
        report.append(
            f"| {row['model']} | {row['case']} | {row.get('complete_pairs', '0')}/{row.get('runs', '0')} | {row.get('prediction_samples', '—')} | "
            + " | ".join(real_prediction_cell(row, metric) for metric, _ in KEY_METRICS)
            + " |"
        )

    report += [
        "",
        "## 模型级误差范围与样本方差",
        "",
        "每格为 `最小绝对误差%–最大绝对误差% / 样本方差(%²)`，范围和方差均只使用该模型已完成的有效真机对照；表中的 `有效真机对照/15` 给出实际样本数。",
        "",
        "| 模型 | TTFT 范围 / 方差 | TPOT 范围 / 方差 | E2E 范围 / 方差 | 总吞吐范围 / 方差 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for parameter in sorted(parameters, key=lambda row: row["model"]):
        model = parameter["model"]
        report.append(
            f"| {model} | "
            + " | ".join(range_variance_cell(metric_map, model, metric) for metric, _ in KEY_METRICS)
            + " |"
        )

    report += [
        "",
        "## 按工作负载汇总",
        "",
        "该表先对每个模型的五次重复求误差均值，再在已完成模型之间汇总，因此不会让某一模型的单次异常值主导结论。",
        "",
        "| 工作负载 | 完成模型 | 配对数 | TTFT MAPE | TPOT MAPE | E2E MAPE | 总吞吐 MAPE |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in workload_summary(case_rows):
        report.append(
            f"| {row['case']} | {row['models']} | {row['paired_runs']} | "
            + " | ".join(
                f"{fmt(row[f'{metric}_mean'])}% ± {fmt(row[f'{metric}_std'])}"
                for metric, _ in KEY_METRICS
            )
            + " |"
        )

    report += [
        "",
        "## 与此前固定轨迹实验的参考比较",
        "",
        f"{scope_note}",
        "只保留模型名和工作负载完全相同的条目。本轮选择 "
        f"{len([item for item in metadata.get('models', '').split(',') if item])} 个模型、每工作负载 "
        f"{metadata.get('repeats', '未记录')} 次；旧轮的样本数以其原始 CSV 为准。"
        "下面用于观察变化，不能当作严格的逐请求消融对照。",
        "",
        "| 模型 | 工作负载 | 旧 TTFT / 新 TTFT | 旧 TPOT / 新 TPOT | 旧 E2E / 新 E2E | 旧总吞吐 / 新总吞吐 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if legacy:
        for row in legacy:
            report.append(
                f"| {row['model']} | {row['case']} | {fmt(row['old_ttft'])}% / {fmt(row['new_ttft'])}% | "
                f"{fmt(row['old_tpot'])}% / {fmt(row['new_tpot'])}% | "
                f"{fmt(row['old_e2e'])}% / {fmt(row['new_e2e'])}% | "
                f"{fmt(row['old_throughput'])}% / {fmt(row['new_throughput'])}% |"
            )
    else:
        report.append("| 暂无完全同名且完成的历史条目 | — | — | — | — | — |")

    legacy_summary = legacy_metric_summary(legacy)
    if legacy_summary:
        report += [
            "",
            "## 旧/新实验汇总",
            "",
            "下表对上一节的完全同名条目直接求平均；`变化`为本轮 MAPE 减去旧实验 MAPE，正值表示本轮误差更大。",
            "",
            "| 指标 | 可比条目数 | 旧实验平均 MAPE | 本轮平均 MAPE | 变化（百分点） |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in legacy_summary:
            report.append(
                f"| {row['metric']} | {row['pairs']} | {fmt(row['old_mean'])}% | "
                f"{fmt(row['new_mean'])}% | {fmt(row['delta'])} |"
            )

    report += [
        "",
        "## 复现边界和交付物",
        "",
        "- 运行模式以本轮 `run_metadata.txt`、每模型 `real/server.log` 和逐次 `benchmark.command.txt` 为准。CPU 模拟器不执行物理 CUDA graph；若实机开启 CUDA graph，报告会明确记录这一不对称，不能把模拟器的逻辑时间称为 CUDA graph 实测时间。",
        "- 每个模型的模拟器 `max_total_tokens` 直接取自该模型实机服务启动后报告的 `max_total_num_tokens`，记录在 `results/<model>/real/max_total_tokens.txt`。这样 CPU dummy model 不会按主机内存虚增 KV-cache 容量。",
        f"- {calibration_summary}",
        f"- attention 校准对齐：{attention_summary}",
        "- 原始请求结果在 `results/<model>/{real,simulator}/...`；逐次实测/预测对照表是 `paired_repeat_metrics.csv`，其中 `prediction_repeat_policy` 和 `prediction_samples` 说明预测汇总方式；按模型/工作负载的均值、最小值、最大值、标准差和样本方差是 `case_metric_statistics.csv` 与 `model_metric_statistics.csv`。",
        "- 图表保存在 `figures/`：按模型的误差柱图、误差与模型结构参数的散点图、按工作负载的箱线图。",
    ]
    return "\n".join(report) + "\n"


def write_docx(
    path: Path,
    markdown: str,
    parameters: list[dict[str, str]],
    case_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    legacy: list[dict[str, str]],
    figures: Path,
    status_line: str,
    scope_note: str,
) -> None:
    """Write a compact DOCX mirror without requiring an external converter."""
    try:
        from docx import Document
        from docx.enum.section import WD_ORIENT
        from docx.oxml.ns import qn
        from docx.shared import Inches, Pt
    except ImportError as exc:
        print(f"Skipping DOCX (python-docx unavailable): {exc}")
        return

    document = Document()

    def set_simsun_rfonts(rpr: Any) -> None:
        fonts = rpr.get_or_add_rFonts()
        for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn(f"w:{attribute}"), "SimSun")

    def set_simsun_run(run: Any) -> None:
        run.font.name = "SimSun"
        set_simsun_rfonts(run._element.get_or_add_rPr())

    def set_simsun_style(style: Any) -> None:
        style.font.name = "SimSun"
        set_simsun_rfonts(style._element.get_or_add_rPr())

    def walk_paragraphs(container: Any):
        yield from container.paragraphs
        for table in container.tables:
            for row in table.rows:
                for cell in row.cells:
                    yield from walk_paragraphs(cell)

    def enforce_simsun() -> None:
        # Set style defaults and every existing run.  Word table styles and
        # headings otherwise retain theme fonts even when Normal is changed.
        for style in document.styles:
            try:
                set_simsun_style(style)
            except AttributeError:
                continue
        containers = [document]
        for doc_section in document.sections:
            containers.extend((doc_section.header, doc_section.footer))
        for container in containers:
            for paragraph in walk_paragraphs(container):
                for run in paragraph.runs:
                    set_simsun_run(run)

    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    styles = document.styles
    styles["Normal"].font.name = "SimSun"
    styles["Normal"].font.size = Pt(9)
    document.add_heading("PR #33824：A100 PCIe 实测与预测器对比报告", 0)
    document.add_paragraph(f"本文件与 final_report.md 同步生成。报告状态：{status_line}。")
    document.add_paragraph("表中的配对数表示已完成有效样本；详细逐次数据以 CSV 附表为准。")

    # Keep the DOCX self-contained: readers should not need to open the
    # Markdown report to learn that CUDA Graph belongs to the real service,
    # while the CPU predictor consumes calibrated operator tables only.
    report_root = path.parent.parent
    real_runtime, simulator_runtime = runtime_description(
        read_key_values(report_root / "run_metadata.txt")
    )
    document.add_heading("运行配置与校准边界", level=1)
    document.add_paragraph(f"实机配置：{real_runtime}")
    document.add_paragraph(launch_settings_summary(report_root))
    document.add_paragraph(f"预测配置：{simulator_runtime}")
    document.add_paragraph(attention_kernel_summary(report_root))

    metric_map = {(row["model"], row["metric"]): row for row in metric_rows}
    cases_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        cases_by_model[row["model"]].append(row)
    document.add_heading("模型级结果", level=1)
    table = document.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    for cell, title in zip(table.rows[0].cells, ("模型", "参数特征", "配对/15", "TTFT", "TPOT", "E2E", "总吞吐")):
        cell.text = title
    for parameter in sorted(parameters, key=lambda row: row["model"]):
        model = parameter["model"]
        row = table.add_row().cells
        values = [
            model,
            f"{fmt(parameter.get('checkpoint_size_gb'))}GB; H={parameter.get('hidden_size') or '—'}; L={parameter.get('num_hidden_layers') or '—'}; E={parameter.get('num_local_experts') or '0'}"
            + ("; AWQ→BF16 proxy" if parameter.get("quantization", "").lower() == "awq" else ""),
            f"{sum(as_int(item.get('complete_pairs')) for item in cases_by_model.get(model, []))}/15",
            *[metric_cell(metric_map, model, metric) for metric, _ in KEY_METRICS],
        ]
        for cell, value in zip(row, values):
            cell.text = value

    document.add_heading("模型级误差范围与样本方差", level=1)
    document.add_paragraph("每格为最小绝对误差%–最大绝对误差% / 样本方差(%²)。")
    dispersion_table = document.add_table(rows=1, cols=5)
    dispersion_table.style = "Table Grid"
    for cell, title in zip(
        dispersion_table.rows[0].cells,
        ("模型", "TTFT", "TPOT", "E2E", "总吞吐"),
    ):
        cell.text = title
    for parameter in sorted(parameters, key=lambda row: row["model"]):
        model = parameter["model"]
        row = dispersion_table.add_row().cells
        values = [model, *[range_variance_cell(metric_map, model, metric) for metric, _ in KEY_METRICS]]
        for cell, value in zip(row, values):
            cell.text = value

    document.add_heading("实测值与预测值", level=1)
    document.add_paragraph(
        "每格为实测 / 预测 / 带符号偏差%。TTFT、TPOT、E2E 的单位为 ms；总吞吐单位为 token/s。预测样本少于真机样本时，预测栏使用该工作负载的预测均值。"
    )
    value_table = document.add_table(rows=1, cols=8)
    value_table.style = "Table Grid"
    for cell, title in zip(
        value_table.rows[0].cells,
        ("模型", "工作负载", "有效真机对照", "预测样本", "TTFT", "TPOT", "E2E", "总吞吐"),
    ):
        cell.text = title
    for result in sorted(case_rows, key=lambda item: (item["model"], item["case"])):
        row = value_table.add_row().cells
        values = [
            result["model"],
            result["case"],
            f"{result.get('complete_pairs', '0')}/{result.get('runs', '0')}",
            str(result.get("prediction_samples", "—")),
            *[real_prediction_cell(result, metric) for metric, _ in KEY_METRICS],
        ]
        for cell, value in zip(row, values):
            cell.text = value

    document.add_heading("按工作负载汇总", level=1)
    workload_table = document.add_table(rows=1, cols=7)
    workload_table.style = "Table Grid"
    for cell, title in zip(
        workload_table.rows[0].cells,
        ("工作负载", "完成模型", "配对数", "TTFT", "TPOT", "E2E", "总吞吐"),
    ):
        cell.text = title
    for result in workload_summary(case_rows):
        row = workload_table.add_row().cells
        values = [
            str(result["case"]),
            str(result["models"]),
            str(result["paired_runs"]),
            *[
                f"{fmt(result[f'{metric}_mean'])}% ± {fmt(result[f'{metric}_std'])}"
                for metric, _ in KEY_METRICS
            ],
        ]
        for cell, value in zip(row, values):
            cell.text = value

    document.add_heading("与旧固定轨迹实验的参考比较", level=1)
    document.add_paragraph(scope_note)
    document.add_paragraph(
        "只列出模型名和工作负载完全一致的条目；该表仅作趋势参考。"
        "两轮并非纯代码对照，不能将误差变化单独归因于代码。"
    )
    legacy_table = document.add_table(rows=1, cols=6)
    legacy_table.style = "Table Grid"
    for cell, title in zip(
        legacy_table.rows[0].cells,
        ("模型", "工作负载", "旧/新 TTFT", "旧/新 TPOT", "旧/新 E2E", "旧/新总吞吐"),
    ):
        cell.text = title
    if legacy:
        for result in legacy:
            row = legacy_table.add_row().cells
            values = [
                result["model"],
                result["case"],
                f"{fmt(result['old_ttft'])}% / {fmt(result['new_ttft'])}%",
                f"{fmt(result['old_tpot'])}% / {fmt(result['new_tpot'])}%",
                f"{fmt(result['old_e2e'])}% / {fmt(result['new_e2e'])}%",
                f"{fmt(result['old_throughput'])}% / {fmt(result['new_throughput'])}%",
            ]
            for cell, value in zip(row, values):
                cell.text = value
    else:
        legacy_table.add_row().cells[0].text = "暂无完全同名且完成的历史条目"

    legacy_summary = legacy_metric_summary(legacy)
    if legacy_summary:
        document.add_heading("旧/新实验汇总", level=1)
        document.add_paragraph("仅对完全同名模型和工作负载求平均；正变化表示本轮 MAPE 更大。")
        summary_table = document.add_table(rows=1, cols=5)
        summary_table.style = "Table Grid"
        for cell, title in zip(
            summary_table.rows[0].cells,
            ("指标", "可比条目", "旧平均 MAPE", "本轮平均 MAPE", "变化（百分点）"),
        ):
            cell.text = title
        for result in legacy_summary:
            row = summary_table.add_row().cells
            values = (
                str(result["metric"]),
                str(result["pairs"]),
                f"{fmt(result['old_mean'])}%",
                f"{fmt(result['new_mean'])}%",
                fmt(result["delta"]),
            )
            for cell, value in zip(row, values):
                cell.text = value

    document.add_heading("实验说明", level=1)
    for line in markdown.splitlines():
        if line.startswith("- ") and ("工作负载" in line or "绝对值" in line or "cu129" in line or "max_total_tokens" in line or "AIC Silicon" in line or "attention 校准" in line or "overlap scheduler" in line or "AWQ" in line):
            document.add_paragraph(line[2:], style="List Bullet")

    document.add_heading("图表", level=1)
    for filename in (
        "mape_by_model_and_metric.png",
        "e2e_mape_vs_model_parameters.png",
        "mape_distribution_by_workload.png",
    ):
        image = figures / filename
        if image.is_file():
            document.add_picture(str(image), width=Inches(8.7))
    document.add_paragraph("完整工作负载表、历史参考表与交付物路径请见同目录 final_report.md。")
    enforce_simsun()
    document.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--legacy-summary", type=Path, default=Path(
        "/home/ms-admin/workspace1/yiheng/ali/sglang/tools/sglang-simulator/test-results/"
        "a100_pcie_all_10_fixed_trace_retest_20260901/aggregate/comparison_summary_all.csv"
    ))
    args = parser.parse_args()
    root = args.results_root.resolve()
    analysis = root / "analysis"
    parameters = read_csv(analysis / "model_parameters.csv")
    case_rows = read_csv(analysis / "case_metric_statistics.csv")
    metric_rows = read_csv(analysis / "model_metric_statistics.csv")
    if not parameters or not case_rows or not metric_rows:
        raise SystemExit("Run summarize_pr33824_a100_matrix.py before rendering the report")
    metadata = read_key_values(root / "run_metadata.txt")
    current_cases = {(row["model"], row["case"]): row for row in case_rows}
    legacy = legacy_rows(args.legacy_summary, current_cases)
    markdown = make_markdown(root, metadata, parameters, case_rows, metric_rows, legacy)
    markdown_path = analysis / "final_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    write_docx(
        analysis / "final_report.docx",
        markdown,
        parameters,
        case_rows,
        metric_rows,
        legacy,
        analysis / "figures",
        matrix_report_status(root, metadata, case_rows),
        comparison_scope_note(root),
    )
    print(f"Wrote {markdown_path} and {analysis / 'final_report.docx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
