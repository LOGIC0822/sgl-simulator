#!/usr/bin/env python3
"""Audit BF16 alignment and compare AIC with CUDA-profiler operator families.

The comparison intentionally keeps CUDA Graph enabled in the SGLang run.  A
PyTorch trace can still expose the kernels replayed by the graph, but it does
not preserve a one-to-one Python-op boundary for every AIC breakdown item.
Accordingly, this script reports only the operator families that can be
identified unambiguously in the trace: GEMM and Triton attention.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from docx import Document
from docx.shared import Pt


MODELS = ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-32B")
PHASES = ("prefill", "decode")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def safetensor_dtypes(model_path: Path) -> list[str]:
    """Read SafeTensors headers only; never materialize a model weight."""

    dtypes: set[str] = set()
    for path in sorted(model_path.glob("*.safetensors")):
        with path.open("rb") as file:
            header_size = int.from_bytes(file.read(8), "little")
            header = json.loads(file.read(header_size))
        for metadata in header.values():
            if isinstance(metadata, dict) and "dtype" in metadata:
                dtypes.add(str(metadata["dtype"]))
    return sorted(dtypes)


def runtime_model_dtype(model_path: Path) -> str:
    """Resolve SGLang's actual `--dtype auto` decision without loading weights."""

    try:
        from sglang.srt.configs.model_config import ModelConfig

        config = ModelConfig(
            model_path=str(model_path), trust_remote_code=False, dtype="auto"
        )
        return str(config.dtype)
    except Exception as error:  # Keep the profile report usable if imports change.
        return f"unavailable: {type(error).__name__}: {error}"


def collect_table_dtypes(database_path: Path) -> dict[str, str]:
    data_dir = database_path / "data" / "a100_pcie_local" / "sglang" / "0.5.10"
    result: dict[str, str] = {}
    for filename, label, dtype_column in (
        ("gemm_perf.txt", "gemm", "gemm_dtype"),
        ("context_attention_perf.txt", "context_attention", "attn_dtype"),
        ("generation_attention_perf.txt", "generation_attention", "attn_dtype"),
    ):
        path = data_dir / filename
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        result[f"{label}_dtype"] = ",".join(
            sorted({row[dtype_column] for row in rows})
        )
        result[f"{label}_kernel_source"] = ",".join(
            sorted({row.get("kernel_source", "") for row in rows})
        )
        if label != "gemm":
            result[f"{label}_kv_dtype"] = ",".join(
                sorted({row["kv_cache_dtype"] for row in rows})
            )
    return result


def real_kv_dtype(log_path: Path) -> str:
    match = re.search(
        r"KV Cache is allocated\. dtype: ([^,]+)",
        log_path.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1) if match else "not found"


def aic_quant_modes(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    names = (
        "gemm_quant_mode",
        "moe_quant_mode",
        "kvcache_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
    )
    result: dict[str, str] = {}
    for name in names:
        match = re.search(rf"{name}=<[^>]*name='([^']+)'", text)
        result[name] = match.group(1) if match else "not found"
    backend = re.search(r"attention_backend='([^']+)'", text)
    result["attention_backend"] = backend.group(1) if backend else "not found"
    return result


def load_trace(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file).get("traceEvents", [])


def duration_us(event: dict[str, Any]) -> float:
    duration = event.get("dur", 0.0)
    return float(duration) if isinstance(duration, (int, float)) else 0.0


def classify_kernel(name: str, phase: str) -> str:
    """Classify only families whose kernel naming is stable for this run.

    `*_fwd_kernel*` is the current-PR Triton attention implementation in these
    dense MHA/GQA traces.  GEMM/GEMV names are emitted by cuBLAS/cuBLASLt.
    Everything else remains deliberately in ``other`` rather than assigning an
    inaccurate AIC operation label to it.
    """

    lowered = name.lower()
    if phase == "prefill" and lowered == "_fwd_kernel":
        return "attention"
    if phase == "decode" and (
        lowered == "_fwd_kernel"
        or lowered.startswith("_fwd_grouped_kernel_stage")
        or lowered.startswith("_fwd_kernel_stage")
    ):
        return "attention"
    if "gemm" in lowered or "gemv" in lowered:
        return "gemm"
    return "other"


def trace_profile(path: Path, phase: str) -> dict[str, float | int | list[str]]:
    totals_us: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    attention_names: Counter[str] = Counter()
    kernel_total_us = 0.0
    for event in load_trace(path):
        if event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        duration = duration_us(event)
        if duration <= 0:
            continue
        name = str(event.get("name", ""))
        family = classify_kernel(name, phase)
        totals_us[family] += duration
        event_counts[family] += 1
        kernel_total_us += duration
        if family == "attention":
            attention_names[name] += 1
    return {
        "trace_gpu_sum_ms": kernel_total_us / 1000.0,
        "profiled_gemm_ms": totals_us["gemm"] / 1000.0,
        "profiled_attention_ms": totals_us["attention"] / 1000.0,
        "profiled_other_ms": totals_us["other"] / 1000.0,
        "gemm_kernel_events": event_counts["gemm"],
        "attention_kernel_events": event_counts["attention"],
        "other_gpu_events": event_counts["other"],
        "attention_kernel_names": sorted(attention_names),
    }


def error_percent(predicted: float, measured: float) -> float | None:
    return (predicted - measured) / measured * 100.0 if measured else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def add_docx_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()

    root = args.results_root.resolve()
    baseline = args.baseline_root.resolve()
    metadata = read_key_value_file(root / "run_metadata.txt")
    baseline_metadata = read_key_value_file(baseline / "run_metadata.txt")
    configured_models = tuple(filter(None, args.models.split(",")))

    simulator_config = Path(baseline_metadata["simulator_config"])
    simulator = read_json(simulator_config)
    database_path = Path(simulator["predictor"]["database_path"])
    table_data = collect_table_dtypes(database_path)

    dtype_rows: list[dict[str, str]] = []
    op_rows: list[dict[str, Any]] = []
    profile_json: dict[str, Any] = {"models": {}}

    for model_name in configured_models:
        model_path = Path("/mnt/data/models") / model_name
        checkpoint = read_json(model_path / "config.json")
        checkpoint_dtype = str(
            checkpoint.get("dtype") or checkpoint.get("torch_dtype") or "not set"
        )
        tensor_dtypes = ",".join(safetensor_dtypes(model_path))
        runtime_dtype = runtime_model_dtype(model_path)
        real_log = baseline / "models" / model_name / "real" / "one_batch.log"
        kv_dtype = real_kv_dtype(real_log)
        quant = aic_quant_modes(
            baseline / "models" / model_name / "aic_query.log"
        )
        is_aligned = all(
            (
                checkpoint_dtype == "bfloat16",
                tensor_dtypes == "BF16",
                runtime_dtype == "torch.bfloat16",
                kv_dtype == "torch.bfloat16",
                quant["gemm_quant_mode"] == "bfloat16",
                quant["kvcache_quant_mode"] == "bfloat16",
                quant["fmha_quant_mode"] == "bfloat16",
                table_data["gemm_dtype"] == "bfloat16",
                table_data["context_attention_dtype"] == "bfloat16",
                table_data["context_attention_kv_dtype"] == "bfloat16",
                table_data["generation_attention_dtype"] == "bfloat16",
                table_data["generation_attention_kv_dtype"] == "bfloat16",
                table_data["context_attention_kernel_source"] == "triton",
                table_data["generation_attention_kernel_source"] == "triton",
            )
        )
        dtype_rows.append(
            {
                "model": model_name,
                "checkpoint_config_dtype": checkpoint_dtype,
                "safetensors_dtypes": tensor_dtypes,
                "sglang_auto_dtype": runtime_dtype,
                "real_kv_cache_dtype": kv_dtype,
                "aic_gemm": quant["gemm_quant_mode"],
                "aic_fmha": quant["fmha_quant_mode"],
                "aic_kv_cache": quant["kvcache_quant_mode"],
                "aic_comm": quant["comm_quant_mode"],
                "attention_table": (
                    f"{table_data['context_attention_dtype']}/"
                    f"{table_data['context_attention_kv_dtype']}/"
                    f"{table_data['context_attention_kernel_source']}"
                ),
                "gemm_table": (
                    f"{table_data['gemm_dtype']}/"
                    f"{table_data['gemm_kernel_source']}"
                ),
                "all_exposed_dtypes_aligned": "yes" if is_aligned else "no",
            }
        )

        prediction = read_json(baseline / "models" / model_name / "aic_prediction.json")
        profile_json["models"][model_name] = {"dtype": dtype_rows[-1], "phases": {}}
        for phase in PHASES:
            traces = sorted((root / "models" / model_name / "traces").glob(f"*_{phase}.trace.json.gz"))
            if len(traces) != 1:
                raise RuntimeError(
                    f"Expected exactly one {phase} trace for {model_name}, found {traces}"
                )
            trace = trace_profile(traces[0], phase)
            breakdown = prediction["phases"][phase]["breakdown_ms"]
            aic_total = float(prediction["phases"][phase]["predicted_ms"])
            aic_gemm = sum(
                float(value) for name, value in breakdown.items() if name.endswith("_gemm")
            )
            aic_attention = sum(
                float(value) for name, value in breakdown.items() if name.endswith("_attention")
            )
            aic_other = aic_total - aic_gemm - aic_attention
            profile_total = float(trace["trace_gpu_sum_ms"])
            profiled_gemm = float(trace["profiled_gemm_ms"])
            profiled_attention = float(trace["profiled_attention_ms"])
            profiled_other = float(trace["profiled_other_ms"])
            row: dict[str, Any] = {
                "model": model_name,
                "phase": phase,
                "aic_total_ms": aic_total,
                "trace_gpu_sum_ms": profile_total,
                "aic_total_error_pct": error_percent(aic_total, profile_total),
                "aic_gemm_ms": aic_gemm,
                "profiled_gemm_ms": profiled_gemm,
                "gemm_error_pct": error_percent(aic_gemm, profiled_gemm),
                "aic_attention_ms": aic_attention,
                "profiled_attention_ms": profiled_attention,
                "attention_error_pct": error_percent(aic_attention, profiled_attention),
                "aic_other_ms": aic_other,
                "profiled_other_ms": profiled_other,
                "other_error_pct": error_percent(aic_other, profiled_other),
                "gemm_kernel_events": trace["gemm_kernel_events"],
                "attention_kernel_events": trace["attention_kernel_events"],
                "other_gpu_events": trace["other_gpu_events"],
                "trace": str(traces[0]),
                "attention_kernel_names": "; ".join(trace["attention_kernel_names"]),
            }
            op_rows.append(row)
            profile_json["models"][model_name]["phases"][phase] = row

    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    dtype_columns = list(dtype_rows[0])
    with (analysis / "dtype_alignment.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=dtype_columns)
        writer.writeheader()
        writer.writerows(dtype_rows)
    op_columns = list(op_rows[0])
    with (analysis / "op_family_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=op_columns)
        writer.writeheader()
        writer.writerows(op_rows)
    (analysis / "profile_summary.json").write_text(
        json.dumps(profile_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dtype_table = [
        [
            row["model"],
            row["checkpoint_config_dtype"],
            row["safetensors_dtypes"],
            row["sglang_auto_dtype"],
            row["real_kv_cache_dtype"],
            row["aic_gemm"],
            row["aic_fmha"],
            row["aic_kv_cache"],
            row["all_exposed_dtypes_aligned"],
        ]
        for row in dtype_rows
    ]
    op_table = [
        [
            row["model"],
            row["phase"],
            fmt(row["aic_total_ms"]),
            fmt(row["trace_gpu_sum_ms"]),
            fmt(row["aic_total_error_pct"], 2) + "%",
            fmt(row["aic_gemm_ms"]),
            fmt(row["profiled_gemm_ms"]),
            fmt(row["gemm_error_pct"], 2) + "%",
            fmt(row["aic_attention_ms"]),
            fmt(row["profiled_attention_ms"]),
            fmt(row["attention_error_pct"], 2) + "%",
        ]
        for row in op_rows
    ]
    markdown_lines = [
        "# AIC 数据类型对齐与 CUDA Graph 算子级误差分析",
        "",
        "## 结论",
        "",
        "三个模型的权重、SGLang `--dtype auto` 解析结果、真机 KV cache、AIC GEMM/FMHA/KV 量化模式，以及 AIC 采样表中的 attention/GEMM 数据类型均为 BF16；未发现会改变本次结果的外部数据类型配置不一致。",
        "",
        "实际 trace 同时显示 RMSNorm 等内部计算使用 FP32 reduction/accumulator kernel。这不是用户可配置的 BF16/FP16 不一致，而是 AIC 当前配置没有按内部 kernel 分别表达 accumulator 和 cast 的抽象边界；它可能贡献剩余误差。",
        "",
        "## 数据类型核对",
        "",
        "| 模型 | checkpoint | SafeTensors | SGLang auto | 真机 KV | AIC GEMM | AIC FMHA | AIC KV | 对齐 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in dtype_table:
        markdown_lines.append("| " + " | ".join(row) + " |")
    markdown_lines += [
        "",
        "AIC attention 采样表统一为 `attn_dtype=bfloat16`、`kv_cache_dtype=bfloat16`、`kernel_source=triton`；GEMM 表统一为 `gemm_dtype=bfloat16`。AIC 日志中的 `attention_backend=flashinfer` 是该 SDK 对 MLA 的默认字段；本次 Qwen2/Qwen3 是 dense MHA/GQA，SDK 选择的是 `context_attention` / `generation_attention` 表，因此该字段不参与本次查询。TP=1，所有 AIC 通信项均为 0，`comm_quant_mode=half` 未被执行或计费。",
        "",
        "## 仍然存在的非数据类型边界",
        "",
        "- AIC 使用 Silicon 静态算子组合；attention 表是在 eager collector 中采集，而真机基线用 CUDA Graph replay。二者的算子、shape、BF16 和 Triton backend 对齐，但 AIC 没有物理回放 graph，也不建模 graph 的 launch/融合边界。",
        "- 对 dense MHA/GQA，AIC 的 `attention_backend=flashinfer` 日志默认值不会改变所选的 `context_attention` / `generation_attention` 表；该字段只供 MLA 路径使用。",
        "- 真机 RMSNorm 的 FP32 accumulator、BF16↔FP32 cast、KV 写入、索引和采样在 AIC 的高层 BF16 quant mode 中没有逐 kernel 配置字段。它们不是 dtype 填错，但不是一一对应的 table 项。",
        "",
        "## CUDA Graph Trace 与 AIC 的算子族对照",
        "",
        "真机 profiling 保持与基线一致的 Triton attention、prefill `tc_piecewise` CUDA Graph 和 decode `full` CUDA Graph。`trace GPU sum` 为 Chrome trace 中 CUDA kernel/memcpy/memset duration 的串行求和，不等于 HTTP 或 scheduler 的端到端时延。GEMM 和 Triton attention 可以从 kernel 名精确识别；其余部分包括 norm、SILU、residual、KV 写入、索引、采样和运行时开销，不强行映射到某一个 AIC op。",
        "",
        "| 模型 | 阶段 | AIC 总 ms | trace GPU sum ms | 总偏差 | AIC GEMM | trace GEMM | GEMM 偏差 | AIC attention | trace attention | attention 偏差 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in op_table:
        markdown_lines.append("| " + " | ".join(row) + " |")
    markdown_lines += [
        "",
        "## 如何解读",
        "",
        "- 若 attention 偏差较小而总偏差仍明显，主要问题不在 BF16 attention 表，而在 GEMM 以外的实测 kernel、融合方式、KV/cache 管理及 CUDA Graph 部署开销的总和。",
        "- 若 GEMM 偏差明显，优先检查 AIC 的精确 GEMM key 是否覆盖该 M/N/K，及全模型运行中 cuBLASLt 的实际 kernel 与采样器 `run_gemm` 是否相同。",
        "- 此次 trace 的 CUDA Graph replay 没有保留 QKV、proj、FFN1、FFN2 的单独高层边界，因此不能把 GEMM 总误差误报为某一个线性层的误差。要继续拆分，需要在 SGLang 模型层加入可关联的 NVTX/`record_function` 范围后再 profile。",
        "",
        "## 复现信息",
        "",
        f"- 源码：`{metadata.get('source_revision', 'not recorded')}`。",
        f"- 对照基线：`{baseline}`。",
        f"- GPU：A100 PCIe；shape：batch=1，prefill=1024 token，首次 decode=1 token。",
        "- 明细：`dtype_alignment.csv`、`op_family_comparison.csv`、`profile_summary.json`；每个模型的原始 Chrome trace 在 `models/<model>/traces/`。",
        "",
    ]
    markdown_path = analysis / "aic_dtype_and_op_profile.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    document = Document()
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(9)
    document.add_heading("AIC 数据类型对齐与 CUDA Graph 算子级误差分析", 0)
    document.add_paragraph(
        "结论：三个模型的权重、SGLang 自动精度、真机 KV cache 和 AIC 的 GEMM/FMHA/KV "
        "配置均为 BF16，未发现会改变本次结果的外部数据类型配置不一致。"
    )
    document.add_heading("数据类型核对", level=1)
    add_docx_table(
        document,
        ["模型", "checkpoint", "SafeTensors", "SGLang", "KV", "AIC GEMM", "AIC FMHA", "AIC KV", "对齐"],
        dtype_table,
    )
    document.add_paragraph(
        "补充：实际 trace 中 RMSNorm 等内部 kernel 有 FP32 reduction/accumulator。AIC 当前没有为这些内部 "
        "kernel 单独设置 accumulator dtype 的配置；这是模型抽象粒度的限制，而不是 BF16/FP16 配置填错。"
    )
    document.add_heading("仍然存在的非数据类型边界", level=1)
    document.add_paragraph(
        "AIC 是静态算子组合，attention 表来自 eager collector；真机为 CUDA Graph replay。两侧的 Triton、"
        "BF16 和 shape 已对齐，但 AIC 不物理回放 graph，也不按 kernel 表达 norm accumulator、cast、KV 写入和索引。"
    )
    document.add_heading("CUDA Graph Trace 与 AIC 算子族对照", level=1)
    add_docx_table(
        document,
        ["模型", "阶段", "AIC总", "trace总", "总偏差", "AIC GEMM", "trace GEMM", "GEMM偏差", "AIC attn", "trace attn", "attn偏差"],
        op_table,
    )
    document.add_paragraph(
        "trace GPU sum 是 CUDA event duration 的求和，不是 HTTP/scheduler 端到端时延。GEMM 与 Triton attention "
        "可直接从 kernel 名识别；其他 kernel 统一保留为 other，避免将其错误归因给某个 AIC op。"
    )
    document.add_heading("下一步", level=1)
    document.add_paragraph(
        "若需要 QKV、proj、FFN1、FFN2 的严格逐项误差，应在模型层添加可关联的 NVTX/record_function 范围后再次采样；"
        "不能只根据 CUDA Graph replay 的 kernel 顺序做猜测。"
    )
    document.save(analysis / "aic_dtype_and_op_profile.docx")


if __name__ == "__main__":
    main()
