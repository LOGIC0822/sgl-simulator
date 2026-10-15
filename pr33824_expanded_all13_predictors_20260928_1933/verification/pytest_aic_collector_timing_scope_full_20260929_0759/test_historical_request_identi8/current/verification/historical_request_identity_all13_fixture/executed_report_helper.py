#!/usr/bin/env python3
"""Append workload-separated tables, learning curves and measured/predicted plots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import tarfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty result table: {path}")
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path, models: list[str], cases: list[str], repeats: int) -> tuple[list[dict], list[dict], list[dict]]:
    batch, e2e, curves = [], [], []
    for model in models:
        model_root = root / "models" / model
        for method, path in (("ML", "ml"), ("AIC", "aic")):
            rows = read_csv(model_root / "holdout" / f"{path}_teacher_forced" / "summary.csv")
            if {row["value"] for row in rows if row["group"] == "case"} != set(cases):
                raise ValueError(f"missing or extra batch workloads for {model}/{method}")
            phases = [row["value"] for row in rows if row["group"] == "case_forward_mode"]
            expected_phases = {f"{case}:mode{mode}" for case in cases for mode in (1, 2)}
            if set(phases) != expected_phases or len(phases) != len(expected_phases):
                raise ValueError(f"missing, duplicate or extra batch phases for {model}/{method}")
            for row in rows:
                if row["group"] not in ("case", "case_forward_mode"):
                    continue
                case, phase = row["value"], "all"
                if row["group"] == "case_forward_mode":
                    case, mode = case.rsplit(":", 1)
                    phase = {"mode1": "prefill", "mode2": "decode"}[mode]
                batch.append({"model": model, "method": method, "case": case, "phase": phase,
                              **{key: row[key] for key in ("n", "mean_actual_ms", "mean_predicted_ms", "mape_pct", "mae_ms", "p95_ape_pct", "mean_signed_error_pct")}})
            paired = read_csv(model_root / "holdout" / f"{path}_e2e_comparison" / "summary.csv")
            expected = {(case, metric) for case in cases for metric in ("mean_e2e_latency_ms", "mean_ttft_ms", "mean_tpot_ms", "total_throughput")}
            actual = set()
            for row in paired:
                if row["group"] != "case_metric":
                    continue
                case, metric = row["value"].rsplit(":", 1)
                if int(row["n"]) != repeats:
                    raise ValueError(f"wrong real repeat count for {model}/{case}/{method}")
                actual.add((case, metric))
                e2e.append({"model": model, "method": method, "case": case, "metric": metric,
                            **{key: row[key] for key in ("n", "real_mean", "real_std", "predicted_mean", "predicted_std", "mape_pct", "mean_signed_error_pct")}})
            if actual != expected:
                raise ValueError(f"incomplete paired workloads for {model}/{method}")
        for row in read_csv(model_root / "learning_curve" / "learning_curve.csv"):
            curves.append(dict(row))
        for case in cases:
            for phase in ("all", "prefill", "decode"):
                compared = [row for row in batch if row["model"] == model and row["case"] == case and row["phase"] == phase]
                if len(compared) != 2 or int(compared[0]["n"]) != int(compared[1]["n"]) or not np.isclose(float(compared[0]["mean_actual_ms"]), float(compared[1]["mean_actual_ms"]), rtol=1e-10):
                    raise ValueError(f"ML/AIC score different real batch observations: {model}/{case}/{phase}")
            for metric in ("mean_e2e_latency_ms", "mean_ttft_ms", "mean_tpot_ms", "total_throughput"):
                compared = [row for row in e2e if row["model"] == model and row["case"] == case and row["metric"] == metric]
                if len(compared) != 2 or not all(np.isclose(float(compared[0][field]), float(compared[1][field]), rtol=1e-10) for field in ("real_mean", "real_std")):
                    raise ValueError(f"ML/AIC use different real E2E baselines: {model}/{case}/{metric}")
    return batch, e2e, curves


def residual_concentration_details(
    root: Path, models: list[str], out: Path, methods: tuple[str, ...] = ("ML", "AIC")
) -> list[str]:
    """Describe squared-error tails without excluding any scored observation."""
    required = (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file()
    first = root / "models" / models[0] / "holdout" / f"{methods[0].lower()}_teacher_forced/per_batch.csv"
    if not first.is_file() and not required:
        return []  # Presentation fixtures have summaries, not measured batches.
    rows, largest = [], []
    for model in models:
        reference = None
        for method in methods:
            directory = root / "models" / model / "holdout" / f"{method.lower()}_teacher_forced"
            scored = read_csv(directory / "per_batch.csv")
            if not scored:
                raise ValueError(f"empty scored batches: {model}/{method}")
            observed = {}
            for row in scored:
                key = (str(Path(row["schedule_path"]).resolve()), int(row["line_number"]))
                actual, predicted = float(row["actual_seconds"]), float(row["predicted_seconds"])
                if key in observed or key[1] < 1:
                    raise ValueError(f"duplicate or invalid scored observation: {model}/{method}/{key}")
                if not np.isfinite(actual) or actual <= 0 or not np.isfinite(predicted) or predicted < 0:
                    raise ValueError(f"nonfinite or invalid scored latency: {model}/{method}/{key}")
                observed[key] = (row["case"], int(row["forward_mode"]), int(row["batch_size"]),
                                 int(row["sum_extend"]), int(row["sum_past"]), actual)
            if reference is not None and observed != reference:
                raise ValueError(f"ML/AIC score different individual observations: {model}/{method}")
            reference = observed
            actual = np.asarray([float(row["actual_seconds"]) for row in scored])
            predicted = np.asarray([float(row["predicted_seconds"]) for row in scored])
            squared = (predicted - actual) ** 2
            absolute = np.abs(predicted - actual) / actual * 100
            overall = [row for row in read_csv(directory / "summary.csv") if row["group"] == "all"]
            computed = {"mean_actual_ms": float(actual.mean() * 1000),
                        "mean_predicted_ms": float(predicted.mean() * 1000),
                        "rmse_ms": float(np.sqrt(squared.mean()) * 1000),
                        "mape_pct": float(absolute.mean())}
            if len(overall) != 1 or int(overall[0]["n"]) != len(scored) or any(
                not np.isclose(float(overall[0][key]), value, rtol=1e-9, atol=1e-12)
                for key, value in computed.items()
            ):
                raise ValueError(f"per-batch rows disagree with summary: {model}/{method}")
            ordering = np.argsort(-squared, kind="stable")
            total_squared = float(squared.sum())
            total_absolute = float(absolute.sum())
            result = {"model": model, "method": method, "n": len(scored),
                      "actual_median_ms": float(np.median(actual) * 1000),
                      "actual_max_ms": float(actual.max() * 1000), **computed}
            for count in (1, 10, 100):
                selected = min(count, len(scored))
                result[f"top_{count}_count"] = selected
                result[f"top_{count}_row_pct"] = 100 * selected / len(scored)
                result[f"top_{count}_squared_error_pct"] = (
                    float(squared[ordering[:selected]].sum() / total_squared * 100)
                    if total_squared else 0.0
                )
                selected_absolute = float(absolute[ordering[:selected]].sum())
                result[f"top_{count}_mape_contribution_pp"] = selected_absolute / len(scored)
                result[f"top_{count}_absolute_percentage_error_share_pct"] = (
                    selected_absolute / total_absolute * 100 if total_absolute else 0.0
                )
            rows.append(result)
            for rank, index in enumerate(ordering[:10], start=1):
                source = scored[int(index)]
                largest.append({"model": model, "method": method, "rank": rank,
                                **{key: source[key] for key in ("case", "schedule_path", "line_number",
                                                               "forward_mode", "batch_size", "sum_extend", "sum_past")},
                                "actual_ms": float(actual[index] * 1000),
                                "predicted_ms": float(predicted[index] * 1000),
                                "absolute_percentage_error_pct": float(absolute[index])})
    write_csv(out / "residual_concentration.csv", rows)
    write_csv(out / "largest_batch_residuals.csv", largest)
    paired_note = (
        "ML/AIC 已逐条核对来源文件、行号、batch 特征和真实时延，不能仅以样本数与均值相同代替这项检查。"
        if {"ML", "AIC"} <= set(methods)
        else "此表只检查所列方法，不表示已经完成 ML/AIC 的逐条对照。"
    )
    lines = ["## 单步误差的尾部集中情况", "",
             "按平方残差排序检查误差集中程度，所有样本仍计入正式评分。表中的占比是总平方误差占比，不是 MAPE 占比；少量长时延可能影响 RMSE，不能据此判断平均相对误差或确定其成因。" + paired_note, "",
             "同一组步骤对整体 MAPE 的贡献为其 APE 之和除以全部步骤数，单位为百分点；它们按平方残差选出，不是按 APE 选出。这不是删除样本后 MAPE 会下降的幅度，因为删除会改变分母。", "",
             "| 模型 | 方法 | 全部步骤 | 实测中位数 / 最大值 ms | RMSE ms | 最大 100 项的实际数量 / 样本占比 | 平方误差占比 | 对整体 MAPE 贡献（百分点） |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['method']} | {row['n']} | {row['actual_median_ms']:.2f} / {row['actual_max_ms']:.2f} | {row['rmse_ms']:.2f} | {row['top_100_count']} / {row['top_100_row_pct']:.3f}% | {row['top_100_squared_error_pct']:.2f}% | {row['top_100_mape_contribution_pp']:.4f} |")
    return lines + ["", "完整排序占比见 residual_concentration.csv；每种方法最大的十个残差及其原始文件/行号见 largest_batch_residuals.csv。未删除异常值、重新拟合或修正任何预测。零总平方误差时，占比记为 0。", ""]


def scored_observation_identity(directory: Path) -> dict[tuple[str, int], tuple]:
    """Identify measured steps independently of predictor values and CSV order."""
    scored = read_csv(directory / "per_batch.csv")
    if not scored:
        raise ValueError(f"empty scored batches: {directory}")
    observations = {}
    resolved_paths = {}
    for row in scored:
        path = Path(row["schedule_path"])
        if row["schedule_path"] not in resolved_paths:
            resolved_paths[row["schedule_path"]] = str(path.resolve())
        key = (resolved_paths[row["schedule_path"]], int(row["line_number"]))
        mode, size = int(row["forward_mode"]), int(row["batch_size"])
        extend, past = int(row["sum_extend"]), int(row["sum_past"])
        actual, prediction = float(row["actual_seconds"]), float(row["predicted_seconds"])
        if (not path.is_absolute() or key[1] < 1 or mode not in (1, 2)
                or size < 1 or extend < 0 or past < 0
                or not np.isfinite(actual) or actual <= 0
                or not np.isfinite(prediction) or prediction < 0):
            raise ValueError(f"invalid scored observation: {directory}/{key}")
        if key in observations:
            raise ValueError(f"duplicate scored observation: {directory}/{key}")
        observations[key] = (row["case"], mode, size, extend, past, actual)
    return observations


def kv_input_contract_details(root: Path, models: list[str], out: Path, rel: Path) -> list[str]:
    """Disclose and quantify the input discrepancy with unchanged ML bundles."""
    first = root / "models" / models[0] / "holdout/real"
    if not list(first.glob("repeat-*/cases/*/TP0.schedule_batch.jsonl")):
        if (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
            raise FileNotFoundError(f"missing formal KV input audit data: {first}")
        return []
    path = Path(__file__).resolve().with_name("audit_pr33824_kv_feature_contract.py")
    spec = importlib.util.spec_from_file_location("kv_input_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = module.source_contract(Path(__file__).resolve().parents[1])
    counts, sources = module.audit(root.resolve(), models)
    target = out / "kv_input_contract"
    target.mkdir()
    write_csv(target / "input_contract_audit.csv", counts)
    scores, frozen = module.compare_frozen_inputs(root.resolve(), models, target)
    (target / "source_manifest.json").write_text(json.dumps({"source_contract": contract,
        "measured_sources": sources, "frozen_models": frozen,
        "scope": "Frozen input-only ML comparison; no refit, E2E rerun, AIC sensitivity estimate or independent GPU seq_lens check."}, indent=2), encoding="utf-8")
    lines = ["## 单步与端到端预测的 KV 输入口径", "",
             "本轮离线训练、ML 单步评分及 AIC 单步评分读取 prefix_indices_len；端到端调用预测器时使用 prefix_indices_len + output_ids_len。因此原单步表和端到端回放并非完全相同的输入口径。此处保留原始结果，使用同一冻结 ML bundle、同一真实步骤和时延，只将评分输入改为当前端到端表达式；没有重新训练、调整参数或重跑真机。",
             "", "这个对照检验的是调用口径，不是独立测量 GPU seq_lens；也不是按新口径重新训练后的准确率。AIC 的上下文查询也受该输入影响，但本表只测 ML，不能把 ML 的变化量当成 AIC 的变化量。原始记录保留已生成 token 数，后续口径统一可复用这些采集数据。", "",
             "| 模型 | 独立步骤数 | 实测 ms | prefix-only / 端到端长度口径预测 ms | MAPE（旧 / 新） | 改善（百分点） |",
             "| --- | --- | --- | --- | --- | --- |"]
    overall = {row["model"]: row for row in scores if row["group"] == "all"}
    for model in models:
        row = overall[model]
        lines.append(f"| {model} | {row['n']} | {row['mean_actual_ms']:.2f} | {row['mean_legacy_prediction_ms']:.2f} / {row['mean_runtime_prediction_ms']:.2f} | {row['legacy_mape_pct']:.2f}% / {row['runtime_mape_pct']:.2f}% | {row['improvement_pp']:+.3f} |")
    fig, ax = plt.subplots(figsize=(max(7, len(models) * .8), 4.5))
    positions = np.arange(len(models))
    ax.bar(positions - .18, [overall[model]["legacy_mape_pct"] for model in models], .36, label="Prefix-only input")
    ax.bar(positions + .18, [overall[model]["runtime_mape_pct"] for model in models], .36, label="Runtime prefix + output input")
    ax.set_xticks(positions, models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ML batch MAPE (%)")
    ax.set_title("Frozen ML models; identical measured steps; input-only comparison")
    ax.legend()
    fig.tight_layout()
    filename = "ml_frozen_kv_input_mape.png"
    fig.savefig(target / filename, dpi=180)
    plt.close(fig)
    return lines + ["", f"![冻结模型 KV 输入口径对照]({rel / 'kv_input_contract' / filename})", "",
                    "kv_input_contract/input_contract_audit.csv 分列训练/测试、负载、阶段和输入变化量；frozen_input_comparison.csv 与 frozen_input_per_batch/ 保留全部分组和逐步预测。source_manifest.json 保存原始文件、源码、模型及评分文件校验值。原口径预测已逐条复现，原始文件和 bundle 在对照前后保持不变。", ""]


def aic_cache_contract_details(root: Path, models: list[str], out: Path) -> list[str]:
    """Check all collected rows for cache aliases without calculating latency."""
    provenance = root / "provenance/formal_start_after_ninja_fix/identity.json"
    if not provenance.is_file():
        return []  # Presentation-only fixtures do not declare an actual AIC source.
    identity = json.loads(provenance.read_text())
    path = Path(__file__).with_name("audit_pr33824_aic_cache_contract.py")
    spec = importlib.util.spec_from_file_location("aic_cache_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = module.run(root.resolve(), models, Path(__file__).resolve().parents[1],
                        Path(identity["aic_root"]), out / "aic_cache_contract")
    lines = ["## AIC 评分缓存与实际查询参数核对", "",
             "评分缓存键与预测调用对平均长度的处理存在差异。这里执行当前源码中的实际方法，仅捕获查询参数，按原评分文件顺序检查后续步骤是否错误复用首个查询；没有修改评分器、重新采样或计算新的预测时延。不同请求被汇总为相同的实际查询，不算缓存冲突。", "",
             "| 模型 | 训练 / 测试步骤数 | 训练 / 测试参数不同的复用步骤 |",
             "| --- | --- | --- |"]
    lookup = {(row["model"], row["role"]): row for row in counts}
    for model in models:
        train, test = lookup[model, "calibration"], lookup[model, "holdout"]
        lines.append(f"| {model} | {train['rows']} / {test['rows']} | {train['material_alias_rows']} / {test['material_alias_rows']} |")
    affected = sum(row["material_alias_rows"] for row in counts)
    if affected:
        lines += ["", f"有 {affected} 个后续步骤复用了查询参数不同的结果。此计数不能直接换算成预测误差；需要保持数据库不变，对这些步骤分别查询才能量化影响。原始评分结果保留，不能根据此表直接归因。"]
    else:
        lines += ["", "本次已采集步骤未出现这种错误复用，因此不能用该缓存键差异解释本次预测误差。这不表示所有未测试输入都没有冲突。"]
    return lines + ["", "aic_cache_contract/ 保留逐模型计数、冲突样例（若有）、原数据及调用源码校验值。浮点差异同时记录严格相等与 1e-12 容差口径；表格采用后者。", ""]


def run_final_cpu_diagnostic(root: Path, models: list[str], out: Path, folder: str, tool: str) -> Path:
    """Keep final-only CPU diagnostics in explicit, isolated subprocesses."""
    import subprocess

    provenance = root / "provenance/formal_start_after_ninja_fix/identity.json"
    identity = json.loads(provenance.read_text())
    metadata = dict(line.split("=", 1) for line in (root / "run_metadata.txt").read_text().splitlines() if "=" in line)
    sim = Path(__file__).resolve().parents[1]
    target = out / folder
    command = ["nice", "-n", "10", metadata["python"],
               str(sim / "tools" / tool),
               "--results-root", str(root.resolve()), "--models", ",".join(models), "--out-dir", str(target.resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    env["PYTHONPATH"] = ":".join([metadata["kernel_compat_stage"], str(sim / "src"),
        str(sim.parents[1] / "python"), str(Path(identity["aic_root"]) / "src"),
        env.get("SGLANG_EXPERIMENT_EXTRA_PYTHONPATH", "")])
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec and torch_spec.origin:
        env["LD_LIBRARY_PATH"] = str(Path(torch_spec.origin).parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["SGLANG_CACHE_DIR"] = str(root / "aic_calibration/cache")
    env["SGLANG_JIT_CACHE_DIR"] = str(root / "aic_calibration/jit-cache")
    env["FLASHINFER_WORKSPACE_BASE"] = str(root / "aic_calibration/flashinfer-workspace")
    (out / f"{folder}.command.json").write_text(json.dumps({"command": command,
        "environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "OMP_NUM_THREADS", "PYTHONPATH")}}, indent=2))
    with (out / f"{folder}.log").open("x") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return target


def run_final_native_profile_diagnostic(root: Path, out: Path) -> Path:
    """Run GPU diagnostics only after formal collection and old CPU scoring."""
    import subprocess

    spec = importlib.util.spec_from_file_location("matched_profile_runner",
        Path(__file__).with_name("run_pr33824_matched_native_profiles.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.require_complete(root)
    metadata = dict(line.split("=", 1) for line in (root / "run_metadata.txt").read_text().splitlines() if "=" in line)
    target = out / "matched_native_profiles"
    command = [metadata["python"], str(Path(runner.__file__)), "--results-root", str(root.resolve()),
               "--out-dir", str(target.resolve())]
    (out / "matched_native_profiles.command.json").write_text(json.dumps({"command": command,
        "scope": "GPU diagnostic after formal native/ML/AIC stages and historical CPU comparison; no new training."}, indent=2))
    with (out / "matched_native_profiles.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if (target / "MATCHED_NATIVE_PROFILE_COMPLETE").read_text().strip() != "complete":
        raise ValueError("matched native profiling did not complete")
    return target


def native_profile_details(root: Path, out: Path, rel: Path) -> list[str]:
    """Keep instrumented CUDA findings separate from all-model accuracy tables."""
    if not (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
        return []
    target = run_final_native_profile_diagnostic(root, out)
    rows = read_csv(target / "analysis/summary.csv")
    models = ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-32B")
    expected = {(model, phase) for model in models for phase in ("prefill", "decode")}
    if len(rows) != 6 or {(row['model'], row['phase']) for row in rows} != expected or any(int(row['n']) != 15 for row in rows):
        raise ValueError("incomplete 3-model, 5-workload, 3-sample native profiling coverage")
    lines = ["", "## 原生单步 CUDA 诊断", "",
             "沿用 3B、7B、32B 正式真实基线的模型、kernel、attention 后端、显存比例、上下文、并行设置及客户端请求参数。每类负载跳过前两个 prefill/decode 步骤，各记录随后三个，共 90 个样本。此处每负载只有一次带 profiler 的运行，不替代前面五次正式实测；采样和导出可能改变动态批次与墙钟时间。", "",
             "按 CPU launch correlation 归属 GPU 事件，GPU busy 时间使用时间区间并集；算子族使用持续时间之和，两个量不混用。AIC 在同一条诊断样本上分别查询原 prefix、runtime prefix+output、实际 prepared seq_lens−extend 三种输入，保持原生阶段标签。generation 查询内部使用 isl+1，因此第三种输入扣除本次新 token。三种结果都保留，不按测试误差选择口径。", "",
             "| 模型 | 阶段 / 样本 | 带 profiling 墙钟 / GPU busy ms | AIC 原 prefix / runtime / 实际 seq 输入 ms | 实际 seq 输入对墙钟 / GPU busy MAPE | GPU 事件时长归属比例 |",
             "| --- | --- | --- | --- | --- | --- |"]
    ordered = [next(row for row in rows if row['model'] == model and row['phase'] == phase)
               for model in models for phase in ("prefill", "decode")]
    for row in ordered:
        lines.append(f"| {row['model']} | {row['phase']} / {row['n']} | {float(row['instrumented_wall_ms']):.3f} / {float(row['native_gpu_union_ms']):.3f} | {float(row['prefix_aic_ms']):.3f} / {float(row['runtime_aic_ms']):.3f} / {float(row['physical_before_aic_ms']):.3f} | {float(row['physical_before_ape_vs_wall_pct']):.2f}% / {float(row['physical_before_ape_vs_gpu_union_pct']):.2f}% | {float(row['gpu_attribution_duration_pct']):.2f}% |")
    lines.extend(["", "墙钟−GPU busy 包含 CPU 调度、launch 间隙、同步、未归属事件及 profiler 开销，不能直接称为纯 CPU 时间。未归属事件及实际 KV 长度不一致的请求数保留在 per_sample.csv 和逐样本 JSON 中。若归属不完整，相应 busy 时间只代表已归属部分，不作完整 GPU 执行时间结论。", "",
                  "### 同类算子时长对照", "",
                  "| 模型 | 阶段 | Native GEMM/GEMV / AIC GEMM ms | Native Triton attention / AIC attention ms | Native 其他 kernel ms | AIC 其他项 ms |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for row in ordered:
        attention = float(row['triton_prefill_attention_duration_sum_ms'])+float(row['triton_decode_attention_duration_sum_ms'])
        lines.append(f"| {row['model']} | {row['phase']} | {float(row['gemm_gemv_duration_sum_ms']):.3f} / {float(row['physical_aic_gemm_sum_ms']):.3f} | {attention:.3f} / {float(row['physical_aic_attention_sum_ms']):.3f} | {float(row['other_kernel_duration_sum_ms']):.3f} | {float(row['physical_aic_other_sum_ms']):.3f} |")
    lines.extend(["", "Native 按 CUDA kernel 名称分族；AIC GEMM 为 *_gemm，attention 为 context_attention/generation_attention。融合、准备及采样边界不完全相同，这张表是同类时长定位，不是逐算子一一对应的准确率。完整 kernel 名、correlation、流、区间与 AIC 每项预测都保留，不能把 other 项差值直接归因为某个算子。", ""])
    figure_dir = target / "analysis"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(6)
    labels = [f"{row['model'].split('-')[1]}\n{row['phase']}" for row in ordered]
    for offset, (key, label, color) in enumerate((("instrumented_wall_ms", "Instrumented wall", "#455a64"),
            ("native_gpu_union_ms", "Attributed GPU busy", "#009688"), ("physical_before_aic_ms", "AIC physical-seq input", "#ff9800"))):
        axes[0].bar(x+(offset-1)*.23, [float(row[key]) for row in ordered], .23, label=label, color=color)
    axes[0].set_ylabel("Mean diagnostic latency (ms)")
    for offset, (key, label) in enumerate((("physical_before_ape_vs_wall_pct", "vs instrumented wall"),
                                         ("physical_before_ape_vs_gpu_union_pct", "vs attributed GPU busy"))):
        axes[1].bar(x+(offset-.5)*.34, [float(row[key]) for row in ordered], .34, label=label)
    axes[1].set_ylabel("Diagnostic MAPE (%)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    filename = "native_gpu_vs_aic_diagnostic.png"
    fig.savefig(figure_dir / filename, dpi=160)
    plt.close(fig)
    lines.extend([f"![原生 GPU 单步与 AIC 诊断]({rel / 'matched_native_profiles/analysis' / filename})", "",
                  "matched_native_profiles/ 保存原命令校验值、诊断命令/环境、版本源码归档、GPU UUID/容量检查、原生 collector 数据、trace、退出码及完成标记。所有正式单步/E2E 表仍使用未启用本 profiler 的原始实验结果。", ""])
    return lines


def ml_runtime_refit_details(root: Path, models: list[str], out: Path, rel: Path) -> list[str]:
    """Compare an independent, predeclared input-corrected CPU refit."""
    if not (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
        return []
    if (root / "REAL_ML_ALL_MODELS_COMPLETE").read_text().strip() != "complete":
        raise ValueError("runtime-KV refit must wait for all native collection")
    target = run_final_cpu_diagnostic(root, models, out, "ml_runtime_refit", "refit_pr33824_runtime_kv_ml.py")
    rows = read_csv(target / "ml_runtime_refit_comparison.csv")
    overall = {row["model"]: row for row in rows if row["group"] == "all" and row["value"] == "all"}
    if set(overall) != set(models) or len(overall) != sum(row["group"] == "all" for row in rows):
        raise ValueError("runtime-KV ML refit model coverage differs")
    manifest = json.loads((target / "source_manifest.json").read_text())
    identities = {entry["model"]: entry for entry in manifest["models"]}
    if set(identities) != set(models) or len(identities) != len(manifest["models"]):
        raise ValueError("runtime-KV ML refit source manifest coverage differs")
    lines = ["### 复用原训练数据的 KV 口径修正拟合", "",
             "冻结模型的输入对照不等于修正训练后的结果。这里另做一次 CPU 拟合：复用每个模型原有训练时延、步骤顺序、HGB p50 参数、seed 和 early-stopping 规则，只将训练特征的 past 改为 prefix+output。独立测试请求不进入拟合；测试记录在拟合、保存 bundle 后才读取，也不用于选参数或选择是否采用修正。原始采集文件和预测器保持不变。", "",
             "| 模型 | 训练 / 测试时延条数 | 实测 ms | 预测 ms（原口径 / 冻结模型新口径 / 新口径重拟合） | MAPE（原 / 冻结 / 重拟合） |",
             "| --- | --- | --- | --- | --- |"]
    for model in models:
        row, identity = overall[model], identities[model]
        lines.append(f"| {model} | {identity['training_rows']} / {row['n']} | {float(row['mean_actual_ms']):.2f} | {float(row['baseline_prefix_mean_predicted_ms']):.2f} / {float(row['frozen_runtime_mean_predicted_ms']):.2f} / {float(row['refit_runtime_mean_predicted_ms']):.2f} | {float(row['baseline_prefix_mape_pct']):.2f}% / {float(row['frozen_runtime_mape_pct']):.2f}% / {float(row['refit_runtime_mape_pct']):.2f}% |")
    fig, ax = plt.subplots(figsize=(10, max(4, len(models)*.55+1.5)))
    positions = np.arange(len(models))
    for offset, key, label in ((-.25, "baseline_prefix", "Original prefix input"),
                              (0, "frozen_runtime", "Frozen model, runtime input"),
                              (.25, "refit_runtime", "Refit on runtime input")):
        ax.barh(positions+offset, [float(overall[model][f"{key}_mape_pct"]) for model in models], .25, label=label)
    ax.set_yticks(positions, models, fontsize=8)
    ax.set_xlim(left=0)
    ax.set_xlabel("Independent-test ML batch MAPE (%)")
    ax.set_title("Same saved training/holdout rows and fit parameters; KV input correction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    filename = "ml_runtime_kv_refit_mape.png"
    fig.savefig(target / filename, dpi=180)
    plt.close(fig)
    return lines + ["", f"![ML KV 口径修正拟合对照]({rel / 'ml_runtime_refit' / filename})", "",
                    "ml_runtime_refit/ 保存完整分负载/阶段统计、逐步骤三个预测值、新 bundle、构造参数、实际训练轮数、CPU 拟合用时、拟合特征/目标及原始文件校验值。此表仅检验修正口径后重新拟合的单步误差，没有用新 bundle 重做端到端回放，不能把单步改善直接写成 E2E 改善；也不是独立验证 GPU seq_lens。", ""]


def aic_kv_input_contract_details(root: Path, models: list[str], out: Path, rel: Path) -> list[str]:
    """Run the separate frozen-AIC CPU diagnostic only at final report time."""
    provenance = root / "provenance/formal_start_after_ninja_fix/identity.json"
    config = root / "aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"
    if not provenance.is_file():
        # Presentation-only tests have no native environment or performance DB.
        return []
    if not config.is_file():
        raise FileNotFoundError(f"missing formal AIC input audit config: {config}")
    if (root / "AIC_ALL_MODELS_COMPLETE").read_text().strip() != "complete":
        raise ValueError("AIC input diagnostic must wait for the formal AIC stage")
    target = run_final_cpu_diagnostic(root, models, out, "aic_kv_input_contract", "audit_pr33824_aic_kv_inputs.py")
    rows = read_csv(target / "aic_input_comparison.csv")
    overall = {row["model"]: row for row in rows if row["group"] == "all" and row["value"] == "all"}
    if set(overall) != set(models) or len(overall) != sum(row["group"] == "all" for row in rows):
        raise ValueError("frozen AIC input comparison model coverage differs")
    lines = ["### AIC 使用相同实测步骤的输入对照", "",
             "下面固定本轮 AIC Silicon 数据库、模型配置和实测步骤，先只把 past 长度从 prefix-only 改为 prefix+output，再单独计入阶段判断的差异。原口径预测先逐条复现；没有用测试时延调整算子数据、比例系数或选择模型。该结果与上面的 ML 输入对照分别计算，不互相代替。", "",
             "离线 AIC 按真机 forward_mode 判断 prefill/decode；当前回放的 ScheduleBatch.is_decode() 则在所有 extend 长度都不大于 1 时判为 decode，单 token 的 prefill 因而可能走另一条查询路径。表中依次列出原评分、只改长度、长度与阶段都按回放输入的结果。分组仍按真机阶段，未改写真实标签。", "",
             "| 模型 | 独立步骤数 / 阶段不一致数 | 实测 ms | 预测 ms（原 / 只改长度 / 长度+阶段） | MAPE（原 / 只改长度 / 长度+阶段） | 只改长度改善（百分点） |",
             "| --- | --- | --- | --- | --- | --- |"]
    for model in models:
        row = overall[model]
        lines.append(f"| {model} | {row['n']} / {row['runtime_phase_disagreement_rows']} | {float(row['mean_actual_ms']):.2f} | {float(row['mean_legacy_prediction_ms']):.2f} / {float(row['mean_runtime_prediction_ms']):.2f} / {float(row['mean_runtime_inferredphase_prediction_ms']):.2f} | {float(row['legacy_mape_pct']):.2f}% / {float(row['runtime_mape_pct']):.2f}% / {float(row['runtime_inferredphase_mape_pct']):.2f}% | {float(row['improvement_pp']):+.3f} |")
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(models)*1.0), 4.5), squeeze=False)
    positions = np.arange(len(models))
    for axis, mode, label in zip(axes[0], ("1", "2"), ("Prefill", "Decode")):
        selected = {row["model"]: row for row in rows if row["group"] == "phase" and row["value"] == mode}
        if set(selected) != set(models):
            raise ValueError(f"frozen AIC input comparison missing phase {mode}")
        axis.bar(positions-.25, [float(selected[model]["legacy_mape_pct"]) for model in models], .25, label="Original scoring")
        axis.bar(positions, [float(selected[model]["runtime_mape_pct"]) for model in models], .25, label="Runtime lengths")
        axis.bar(positions+.25, [float(selected[model]["runtime_inferredphase_mape_pct"]) for model in models], .25, label="Runtime lengths + phase")
        axis.set_xticks(positions, models, rotation=45, ha="right", fontsize=7)
        axis.set_ylim(bottom=0)
        axis.set_ylabel("AIC batch MAPE (%)")
        axis.set_title(label)
        axis.legend(fontsize=8)
    fig.suptitle("Frozen AIC database; identical measured steps; input-only comparison")
    fig.tight_layout()
    filename = "aic_frozen_kv_input_mape.png"
    fig.savefig(target / filename, dpi=180)
    plt.close(fig)
    return lines + ["", f"![AIC KV 输入口径对照]({rel / 'aic_kv_input_contract' / filename})", "",
                    "aic_kv_input_contract/aic_input_comparison.csv 保留全部模型、负载和阶段；per_batch/ 保留逐步骤真实值和三种输入解释的预测。source_manifest.json 记录原评分、实测记录、checkpoint config、数据库、kernel 二进制和调用源码校验值。这里只检验输入变化对静态预测的影响，没有重新做端到端回放，也没有独立测量 GPU seq_lens；剩余误差不能仅凭此表归因于 CPU 或某个算子。", ""]


def previous_calibration_rows(root: Path, models: list[str]) -> list[dict]:
    """Pair old/new predictors only on the exact same measured batch rows."""
    rows = []
    for model in models:
        reference = None
        for method in ("ml", "aic"):
            current_root = root / "models" / model / "holdout" / f"{method}_teacher_forced"
            old_root = root / "previous_calibration/models" / model / f"{method}_teacher_forced"
            observed = scored_observation_identity(current_root)
            old_observed = scored_observation_identity(old_root)
            if observed != old_observed or (reference is not None and observed != reference):
                raise ValueError(f"old/new score different individual real observations: {model}/{method}")
            real_root = (root / "models" / model / "holdout/real").resolve()
            if any(real_root not in Path(path).parents for path in {key[0] for key in observed}):
                raise ValueError(f"comparison source is not the new holdout: {model}/{method}")
            reference = observed
            current = read_csv(current_root / "summary.csv")
            old = read_csv(old_root / "summary.csv")
            previous = {(row["group"], row["value"]): row for row in old}
            current_keys = [(row["group"], row["value"]) for row in current]
            if len(previous) != len(old) or len(set(current_keys)) != len(current_keys):
                raise ValueError(f"duplicate scoring summary group: {model}/{method}")
            if set(current_keys) != set(previous):
                raise ValueError(f"old/new scoring summary groups differ: {model}/{method}")
            for row in current:
                if row["group"] not in ("all", "forward_mode", "case", "case_forward_mode"):
                    continue
                matched = previous[(row["group"], row["value"])]
                if int(row["n"]) != int(matched["n"]) or not np.isclose(float(row["mean_actual_ms"]), float(matched["mean_actual_ms"]), rtol=1e-10):
                    raise ValueError(f"old/new score different real observations: {model}/{method}/{row['value']}")
                rows.append({"model": model, "method": method.upper(), "group": row["group"], "value": row["value"],
                             "n": int(row["n"]), "mean_actual_ms": float(row["mean_actual_ms"]),
                             "old_predicted_ms": float(matched["mean_predicted_ms"]), "new_predicted_ms": float(row["mean_predicted_ms"]),
                             "old_mape_pct": float(matched["mape_pct"]), "new_mape_pct": float(row["mape_pct"]),
                             "improvement_pp": float(matched["mape_pct"]) - float(row["mape_pct"]),
                             "old_mae_ms": float(matched["mae_ms"]), "new_mae_ms": float(row["mae_ms"]),
                             "old_p95_ape_pct": float(matched["p95_ape_pct"]), "new_p95_ape_pct": float(row["p95_ape_pct"])})
    return rows


def append_previous_comparison(root: Path, models: list[str], cases: list[str], out: Path, rel: Path) -> list[str]:
    rows = previous_calibration_rows(root, models)
    write_csv(out / "old_new_calibration_same_test.csv", rows)
    lines = ["## 固定新测试集的历史校准对照（仅单步）", "",
             "下面将 9 月 18 日的 ML bundle / AIC Silicon 表和本轮扩充结果评分于完全相同的本轮真实 batch；逐条核对原始文件、行号、phase、batch 大小、token 统计和实测时延，再核对分组样本数及实测均值。所有来源必须指向本轮测试目录；样本数和均值相同不能替代逐条核对。旧方法原有 E2E 不移入此表。ML 的训练负载覆盖和请求量同时改变，不能把改善全部归因于样本数量；前面的 160/480/800 请求曲线用于固定覆盖下的训练量检查。AIC 重新采集、加密规则长度网格，并将 Triton page size 从旧采集的 64 对齐为普通真机基线的 1；比较的是整体校准效果，不能单独归因于多采样。1M/量化/MoE 的代理限制保持单列。", "",
             "| 模型 | 方法 | 实测 ms | 旧预测 / 新预测 ms | 旧 MAPE / 新 MAPE | 改善（百分点） |",
             "| --- | --- | --- | --- | --- | --- |"]
    selected = [row for row in rows if row["group"] == "all"]
    for row in selected:
        lines.append(f"| {row['model']} | {row['method']} | {row['mean_actual_ms']:.2f} | {row['old_predicted_ms']:.2f} / {row['new_predicted_ms']:.2f} | {row['old_mape_pct']:.2f}% / {row['new_mape_pct']:.2f}% | {row['improvement_pp']:+.2f} |")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ax, method in zip(axes, ("ML", "AIC")):
        subset = {row["model"]: row for row in selected if row["method"] == method}
        x = np.arange(len(models))
        ax.bar(x - .18, [subset[m]["old_mape_pct"] for m in models], .36, label="Previous calibration")
        ax.bar(x + .18, [subset[m]["new_mape_pct"] for m in models], .36, label="Expanded calibration")
        ax.set_xticks(x, models, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{method}: old/new calibration on identical new real batches")
        ax.set_ylabel("Batch MAPE (%)")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "old_new_calibration_mape.png", dpi=180)
    plt.close(fig)
    lines.extend(["", f"![固定新测试集的校准对照]({rel / 'old_new_calibration_mape.png'})", ""])
    for method in ("ML", "AIC"):
        subset = {(row["model"], row["value"]): row for row in rows if row["method"] == method and row["group"] == "case"}
        values = np.asarray([[subset[(model, case)]["improvement_pp"] for case in cases] for model in models])
        limit = max(1., float(np.max(np.abs(values))))
        fig, ax = plt.subplots(figsize=(9, max(5, len(models) * .38)))
        picture = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(cases)), cases, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)), models, fontsize=8)
        for i in range(len(models)):
            for j in range(len(cases)):
                ax.text(j, i, f"{values[i,j]:+.1f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{method}: old MAPE - new MAPE (percentage points)")
        fig.colorbar(picture, ax=ax, label="Positive = lower error after expanded calibration")
        fig.tight_layout()
        filename = f"{method.lower()}_old_new_calibration_workloads.png"
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
        lines.extend([f"![{method} 分负载校准差值]({rel / filename})", ""])
    return lines


def wait_previous_comparison(root: Path) -> bool:
    previous = root / "previous_calibration"
    if not (previous / "reference.json").is_file():
        return False
    while not (previous / "PREVIOUS_CALIBRATION_COMPLETE").is_file():
        if (previous / "COMPARISON_FAILED").is_file():
            raise RuntimeError((previous / "COMPARISON_FAILED").read_text())
        owner = json.loads((previous / "owner.json").read_text())
        proc = Path(f"/proc/{int(owner['pid'])}/cmdline")
        if not proc.exists() or b"run_pr33824_previous_calibration_comparison.py" not in proc.read_bytes():
            raise RuntimeError("historical calibration scorer is not running; resume it before report generation")
        print("Waiting for historical calibration scoring; recheck in 600 seconds", flush=True)
        time.sleep(600)
    return True


def historical_aic_baseline_details(root: Path, models: list[str], out: Path, rel: Path) -> list[str]:
    """Keep the earlier low-error baselines separate from the current matrix."""
    selected = [name for name in ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-32B") if name in models]
    if not selected:
        return []
    sources = (("2026-09-08 fallback", "pr33824_piecewise_a100_dense3_matrix_20260908_134523"),
               ("2026-09-10 compatible", "pr33824_a100_dense3_sm80compat_matrix_20260910_164000"))
    if not all((root.parent / folder / "analysis/case_metric_statistics.csv").is_file() for _, folder in sources):
        if (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
            raise FileNotFoundError("historical AIC baseline CSVs are missing; do not replace them with remembered percentages")
        return []
    rows, summaries = [], []
    for label, folder in sources:
        source = root.parent / folder
        metadata = key_values(source / "run_metadata.txt")
        if not (source / "MATRIX_COMPLETE").is_file():
            raise ValueError(f"historical matrix lacks its completion marker: {source}")
        for entry in read_csv(source / "analysis/model_metric_statistics.csv"):
            if entry["model"] in selected and entry["metric"] in ("mean_e2e_latency_ms", "mean_ttft_ms"):
                summaries.append({"experiment": label, "model": entry["model"], "metric": entry["metric"],
                                  "paired_runs": int(entry["paired_runs"]), "mape_pct": float(entry["mape_mean_pct"])})
        for entry in read_csv(source / "analysis/case_metric_statistics.csv"):
            model = entry["model"]
            if model not in selected:
                continue
            if int(entry["complete_pairs"]) != 5:
                raise ValueError(f"incomplete historical real pairs: {folder}/{model}/{entry['case']}")
            settings = source / "results" / model
            rows.append({"experiment": label, "source_root": str(source), "model": model, "case": entry["case"],
                         "requests_per_run": int(entry["num_prompts"]), "paired_runs": int(entry["complete_pairs"]),
                         "real_e2e_ms": float(entry["real_mean_e2e_latency_ms_mean"]),
                         "predicted_e2e_ms": float(entry["simulated_mean_e2e_latency_ms_mean"]),
                         "e2e_mape_pct": float(entry["absolute_error_pct_mean_e2e_latency_ms_mean"]),
                         "context_length": int((settings / "context-length.txt").read_text()),
                         "mem_fraction_static": float((settings / "real-mem-fraction-static.txt").read_text()),
                         "source_revision": metadata["source_revision"],
                         "real_cuda_graph_config": metadata["real_cuda_graph_config"],
                         "real_overlap_schedule_enabled": metadata["real_overlap_schedule_enabled"],
                         "trace_policy": metadata["trace_policy"]})
    if len(rows) != 6 * len(selected) or len(summaries) != 4 * len(selected):
        raise ValueError("historical baseline model/workload coverage is incomplete")
    write_csv(out / "historical_aic_baselines.csv", rows)
    write_csv(out / "historical_aic_metric_scope.csv", summaries)
    lines = ["## 历史低误差结果的基线变化", "",
             "9 月 8 日的低 E2E 误差来自 Torch 回退算子路径；9 月 10 日采用 SM80 兼容 native kernel 并更换算子表。两次历史实验都启用 CUDA Graph 和 overlap，context=8192，同一用例重复固定 seed；显存比例从 0.80 变为 0.96。本轮则禁用 graph/overlap、context=2048，并重新采集五个独立 seed。源码 revision 相同不代表 dirty 补丁、kernel 和执行路径相同。这些数据是配置变化的历史对照，不是固定真机基线上的单因素校准实验。", "",
             "先列饱和用例的实际时延与预测时延。两次历史实验均每轮 32 请求；本轮同名用例为 64 请求，不能仅按用例名称直接相减。", "",
             "| 历史实验 | 模型 | 真机 E2E ms | AIC E2E ms | E2E MAPE |", "| --- | --- | --- | --- | --- |"]
    saturation = [row for row in rows if row["case"] == "saturation_1024x256_inf"]
    for row in saturation:
        lines.append(f"| {row['experiment']} | {row['model']} | {row['real_e2e_ms']:.2f} | {row['predicted_e2e_ms']:.2f} | {row['e2e_mape_pct']:.2f}% |")
    lines.extend(["", "三个历史负载等权汇总如下。E2E 小于 20%并不表示首 token 或单步也小于 20%；表中不把饱和用例误差与三负载总误差混写。", "",
                  "| 历史实验 | 模型 | E2E MAPE（三负载） | TTFT MAPE（三负载） | 配对数 |", "| --- | --- | --- | --- | --- |"])
    for label, _ in sources:
        for model in selected:
            matching = {row["metric"]: row for row in summaries if row["experiment"] == label and row["model"] == model}
            a, b = matching["mean_e2e_latency_ms"], matching["mean_ttft_ms"]
            lines.append(f"| {label} | {model} | {a['mape_pct']:.2f}% | {b['mape_pct']:.2f}% | {a['paired_runs']} |")
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4), squeeze=False)
    for ax, model in zip(axes[0], selected):
        points = [next(row for row in saturation if row["experiment"] == label and row["model"] == model) for label, _ in sources]
        x = np.arange(len(points))
        ax.bar(x - .18, [row["real_e2e_ms"] for row in points], .36, label="Real baseline")
        ax.bar(x + .18, [row["predicted_e2e_ms"] for row in points], .36, label="AIC prediction")
        ax.set_xticks(x, ["Sep 08 fallback", "Sep 10 compatible"], rotation=15, ha="right", fontsize=8)
        ax.set_title(model)
        ax.set_ylabel("Saturation mean E2E (ms); 32 requests")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Historical protocol changes: each prediction has its own real baseline")
    fig.tight_layout()
    fig.savefig(out / "historical_aic_real_and_predicted.png", dpi=180)
    plt.close(fig)
    return lines + ["", f"![历史基线与预测值变化]({rel / 'historical_aic_real_and_predicted.png'})", "",
                    "历史各负载数值、源码、graph 配置、context、显存比例和固定 seed 策略保存于 historical_aic_baselines.csv。版本、实际算子路径、请求量及单步计时边界都需要对齐后才可归因；仅有同名模型和同一源码提交不够。", ""]


def historical_aic_one_batch_details(root: Path, models: list[str], out: Path, relative: Path) -> list[str]:
    """Rebuild historical fixed-shape errors from five raw measurements."""
    import subprocess

    selected = [model for model in ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-32B") if model in models]
    if not selected:
        return []
    sources = (("Sep 08 fallback", "pr33824_aic_vs_real_single_step_dense3_20260908_155211"),
               ("Sep 10 compatible", "pr33824_aic_vs_real_single_step_dense3_sm80compat_20260910_181204"))
    if not all((root.parent / folder / "analysis/single_step_comparison.csv").is_file() for _, folder in sources):
        if (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
            raise FileNotFoundError("missing historical fixed-shape AIC single-step results")
        return []
    required = (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file()
    timer_sha = "presentation fixture, not verified source"
    if required:
        repo = Path(__file__).resolve().parents[3]
        revision = key_values(root / "run_metadata.txt")["source_revision"]
        name = "python/sglang/benchmark/one_batch.py"
        timer_bytes = (repo / name).read_bytes()
        if timer_bytes != subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{name}"]):
            raise ValueError("one_batch timing source differs from recorded source revision")
        timer_sha = hashlib.sha256(timer_bytes).hexdigest()
    rows = []
    for label, folder in sources:
        source = root.parent / folder
        if key_values(source / "finalization.status").get("status") != "complete":
            raise ValueError(f"historical fixed-shape experiment incomplete: {folder}")
        meta = key_values(source / "run_metadata.txt")
        if required and meta["source_revision"] != revision:
            raise ValueError("historical one_batch source revision differs from current experiment")
        if (int(meta["input_len"]) != 1024 or int(meta["output_len"]) != 2 or int(meta["repeats"]) != 5):
            raise ValueError(f"historical fixed-shape protocol changed: {folder}")
        existing = [row for row in read_csv(source / "analysis/single_step_comparison.csv") if row["model"] in selected]
        keys = {(row["model"], row["phase"]) for row in existing}
        if len(existing) != 2 * len(selected) or keys != {(model, phase) for model in selected for phase in ("prefill", "decode")}:
            raise ValueError(f"missing/duplicate historical fixed-shape model phases: {folder}")
        for model in selected:
            model_root = source / "models" / model
            if (model_root / "real/status.txt").read_text().strip() != "ok" or (model_root / "aic_query.status.txt").read_text().strip() != "ok":
                raise ValueError(f"historical fixed-shape model failed: {folder}/{model}")
            raw = [json.loads(line) for line in (model_root / "real/one_batch.jsonl").read_text().splitlines() if line.strip()]
            prediction = json.loads((model_root / "aic_prediction.json").read_text())
            if (len(raw) != 5 or any((item["batch_size"], item["input_len"], item["output_len"]) != (1, 1024, 2) for item in raw)
                    or (prediction["batch_size"], prediction["input_len"]) != (1, 1024)):
                raise ValueError(f"wrong historical fixed-shape raw samples: {folder}/{model}")
            command = (model_root / "real/command.txt").read_text()
            for phase, key in (("prefill", "prefill_latency"), ("decode", "median_decode_latency")):
                samples = np.asarray([float(item[key]) * 1000 for item in raw])
                predicted = float(prediction["phases"][phase]["predicted_ms"])
                if not np.all(np.isfinite(samples) & (samples > 0)) or not np.isfinite(predicted) or predicted <= 0:
                    raise ValueError(f"invalid historical fixed-shape latency: {folder}/{model}/{phase}")
                mean, std = float(samples.mean()), float(samples.std(ddof=1))
                signed = (predicted - mean) / mean * 100
                recorded = next(row for row in existing if (row["model"], row["phase"]) == (model, phase))
                computed = {"real_mean_ms": mean, "real_stddev_ms": std, "aic_predicted_ms": predicted,
                            "signed_error_pct": signed, "absolute_error_pct": abs(signed)}
                if int(recorded["real_samples"]) != 5 or any(not np.isclose(float(recorded[name]), value, rtol=1e-9, atol=1e-10) for name, value in computed.items()):
                    raise ValueError(f"historical fixed-shape CSV disagrees with raw data: {folder}/{model}/{phase}")
                rows.append({"experiment": label, "model": model, "phase": phase, "real_samples": 5,
                             "real_mean_ms": mean, "real_sample_std_ms": std, "predicted_ms": predicted,
                             "signed_error_of_mean_pct": signed, "absolute_error_of_mean_pct": abs(signed),
                             "individual_repeat_mape_pct": float(np.mean(np.abs((predicted - samples) / samples)) * 100),
                             "source_root": str(source), "source_revision": meta["source_revision"],
                             "context_length": int(meta["context_length"]), "cuda_graph_config": meta["real_cuda_graph_config"],
                             "kernel_stage": meta.get("kernel_compat_stage", "historical Torch fallback"),
                             "real_command": command.strip(), "prediction_database": prediction["database_path"],
                             "timer": "device synchronization plus perf_counter wall time", "timer_source_sha256": timer_sha})
    write_csv(out / "historical_aic_fixed_shape_steps.csv", rows)
    lines = ["## 历史固定形状单步结果与本轮统计口径", "",
             "以下重新读取两次历史实验的原始五次 one_batch 测量和 AIC 预测 JSON，复算真实均值、样本标准差及误差，并核验与原 CSV 一致。B=1、输入=1024、输出=2，每轮只含一个 prefill 和一个首 decode；启用 prefill piecewise/decode full graph，context=8192。它不是本轮 graph 关闭、动态 batch、随机长度服务请求的同输入对照。", "",
             "原 single_step_comparison.csv 的 absolute_error_pct 是 |预测−五次真实均值|/真实均值，不是逐次 APE 的平均，也不是本轮逐 batch MAPE。早期 ML/AIC 比较表曾将该字段标成 MAPE；这里按原始公式更正名称，不改写历史文件。逐次 APE 平均另列在新 CSV 的 individual_repeat_mape_pct，仍只代表这五次固定形状测量。历史 std 为样本标准差（n−1），与本轮 E2E 表采用的总体 std 不同。", "",
             "| 历史实验 | 模型 | 阶段 | 真机均值 ± 样本 std ms | AIC 预测 ms | 均值相对误差 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['model']} | {row['phase']} | {row['real_mean_ms']:.3f} ± {row['real_sample_std_ms']:.3f} | {row['predicted_ms']:.3f} | {row['absolute_error_of_mean_pct']:.2f}% |")
    scope_difference = max(abs(row["absolute_error_of_mean_pct"] - row["individual_repeat_mape_pct"]) for row in rows)
    lines.extend(["", f"在本表这些固定样本中，两种误差公式的最大数值差为 {scope_difference:.4f} 个百分点；这是同一批历史重复的复算结果，不是新旧服务实验的精度差。", ""])
    lines.extend(["", "历史单步也存在超过 20%的误差；不能把某次饱和负载的低 E2E 误差推广到全部阶段与形状。当前 one_batch 的计时源码与所记录 revision 一致：它同样使用同步墙钟，包含 batch 准备、forward 和 sampling，并非仅 CUDA event 算子时间。当前 collector 则包围 Scheduler.run_batch，二者的调用范围和 graph 设置均不同；没有同配置 profile 时，不能给 CPU、编译或 kernel 差异分配具体误差比例。", ""])
    fig, axes = plt.subplots(2, len(selected), figsize=(5 * len(selected), 7), squeeze=False)
    for column, model in enumerate(selected):
        for row_index, phase in enumerate(("prefill", "decode")):
            ax = axes[row_index, column]
            points = [next(row for row in rows if row["experiment"] == label and row["model"] == model and row["phase"] == phase) for label, _ in sources]
            x = np.arange(2)
            ax.bar(x - .18, [row["real_mean_ms"] for row in points], .36, label="Real mean")
            ax.bar(x + .18, [row["predicted_ms"] for row in points], .36, label="AIC prediction")
            ax.set_xticks(x, [label for label, _ in sources], rotation=12, fontsize=8)
            ax.set_title(f"{model}: {phase}", fontsize=10)
            ax.set_ylabel("Fixed-shape synchronized wall time (ms)")
            ax.grid(axis="y", alpha=.2)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "historical_aic_one_batch_actual_vs_predicted.png", dpi=180)
    plt.close(fig)
    return lines + [f"![历史固定形状的真实值与预测值]({relative / 'historical_aic_one_batch_actual_vs_predicted.png'})", "",
                    "原始数据位置、五次样本误差、graph、context、kernel stage、实际命令和预测表来源保存于 historical_aic_fixed_shape_steps.csv；它与本轮独立服务请求表分开列示。", ""]


def training_details(root: Path, models: list[str], out: Path) -> list[str]:
    import joblib
    rows = []
    for model in models:
        bundle = joblib.load(root / "models" / model / "calibration/training" / f"{model}_hgbmono_p50.pkl")
        rows.append({"model": model, "fit_input_rows": bundle["train_rows"], "training_cases": bundle["train_cases"],
                     "actual_iterations": int(bundle["model"].n_iter_),
                     "parameters": json.dumps(bundle["best_params"], ensure_ascii=False),
                     "features": json.dumps(bundle["features"], ensure_ascii=False),
                     "monotonic_constraints": json.dumps(bundle["monotonic"], ensure_ascii=False),
                     "max_training_latency_seconds": bundle["max_latency_seconds"]})
    write_csv(out / "training_parameters.csv", rows)
    lines = ["## ML 训练记录", "",
             "正式 predictor 为每模型独立 HGB p50，18 维 batch/长度特征；不按独立测试误差选择 p50 或 L2。训练集合内先按每条 trace 用例随机划分 80%/20%作内部检查，再将全部训练行传给最终 fit；HGB 另取训练侧 15%作 early-stopping 验证，不使用正式测试请求。", "",
             "上游 data-tools 的 trainer 读取已采集的 batch 文件，不负责生成独立测试请求。其默认内部检查按每个用例随机分行，比例 50%/50%、seed=42、forward_modes=1；检查后保存的模型重新使用全部输入行拟合。这个划分没有按完整请求隔离，同一请求的多个 batch 可能落在两侧，不能作为跨请求或跨负载泛化结果。本轮正式测试另行采集完整请求，内部检查只用于核查拟合过程。[上游训练脚本](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/tools/train_latency_model.py)，[上游验证说明](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/README.md#train-an-ml-latency-predictor)。", "",
             "上游 serving 单测与上述训练验证是两件事：它使用 examples 中的 ML 配置，在 CPU 模拟器上发送 3 条 ShareGPT 请求，每条输出 4 token，断言完成数和指标有效性，没有真机误差对照。配套资产生成脚本仅以一行全零特征、0.001 秒标签构建固定 1 ms 的 DummyRegressor；它是功能示例，不是 GPU 校准结果。本轮不使用该示例预测器，而是每模型分别采集、训练并在独立请求上评价。[上游 serving 单测](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/test/test_simulation_sglang_serving.py)，[ML 示例配置](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/examples/sim_configs/ml.json)，[示例资产生成脚本](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/examples/build_example_assets.py)。", "",
             "| 参数 | 设置 |", "| --- | --- |",
             "| loss / quantile | quantile / 0.5 |", "| max_iter / max_leaf_nodes | 400 / 63 |",
             "| learning_rate / l2_regularization | 0.04 / 0.1 |",
             "| early_stopping / validation_fraction / n_iter_no_change | True / 0.15 / 20 |",
             "| 标签与过滤 | iter_latency，秒；训练只保留有效 mode=1/2、非空 batch、0 < latency ≤ 30 秒；正式评分保留所有有限正时延 |", "",
             "| 模型 | 最终 fit 输入行 | 训练 trace 用例 | 实际迭代数 |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['fit_input_rows']} | {row['training_cases']} | {row['actual_iterations']} |")
    lines.extend(["", "全部特征名称、单调约束、random_state 和实际参数另存 training_parameters.csv；原始 bundle 和训练命令同时保留。", ""])
    return lines + upstream_split_request_details(root, models, out) + historical_request_identity_details(root, models, out)


def upstream_split_request_details(root: Path, models: list[str], out: Path) -> list[str]:
    """Present the pinned split-function demonstration, not upstream accuracy."""
    candidates = list(root.glob("verification/upstream_row_split_identity_*/manifest.json"))
    if not candidates:
        return []
    if len(candidates) != 1:
        raise ValueError("ambiguous upstream split demonstration")
    manifest_path = candidates[0]
    manifest = json.loads(manifest_path.read_text())
    if (manifest["upstream_branch_head"] != "9cfddcedc9d1e00885c28db75a45513eddbb6271"
            or manifest["parsed_defaults"] != {"--forward-modes": "1", "--seed": 42, "--test-fraction": .5}
            or not manifest["no_fit_gpu_or_formal_split_mutation"]):
        raise ValueError("upstream split source/defaults differ")
    folder = manifest_path.parent
    for filename, key in (("upstream_train_latency_model.py", "upstream_source_sha256"),
                          ("upstream_split_request_overlap.csv", "split_summary_sha256")):
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"upstream split artifact checksum differs: {filename}")
    sources = read_csv(folder / "source_sha256.csv")
    if not sources or len(sources) != manifest["source_files_verified_unchanged"]:
        raise ValueError("upstream split input checksum coverage differs")
    for source in sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("upstream split demonstration input changed")
    rows = read_csv(folder / "upstream_split_request_overlap.csv")
    lookup = {(row["model"], row["forward_modes"]): row for row in rows}
    audited = sorted({row["model"] for row in rows})
    if (not audited or not set(audited) <= set(models) or len(rows) != len(lookup)
            or set(lookup) != {(model, mode) for model in audited for mode in ("1", "1,2")}):
        raise ValueError("upstream split demonstration model/mode coverage differs")
    for row in rows:
        shared, union = int(row["shared_request_ids"]), int(row["unique_request_ids"])
        train_ids, test_ids = int(row["train_unique_request_ids"]), int(row["test_unique_request_ids"])
        if (union <= 0 or not 0 <= shared <= min(train_ids, test_ids)
                or train_ids + test_ids - shared != union
                or int(row["train_rows"]) + int(row["internal_test_rows"]) != int(row["rows"])
                or int(row["identical_rows_in_both_splits"]) != 0
                or not np.isclose(float(row["shared_pct_of_all_requests"]), 100 * shared / union, rtol=0, atol=1e-10)):
            raise ValueError("upstream split demonstration counts differ")
    write_csv(out / "upstream_split_request_overlap.csv", rows)
    lines = ["### 上游分行方法的请求重叠核对", "",
             "将上游提交中的原始划分函数应用到本轮下列模型的训练记录，seed=42、50%/50%，只检查 request ID 和行索引，不拟合。这不是上游作者的数据或准确率，也没有改变本轮训练与独立测试的划分。默认仅保留 prefill；第二列对照显式加入 decode，并非上游默认设置。比例以两侧请求 ID 的并集为分母。", "",
             "| 模型 | 默认 prefill：两侧共享请求 / 全部请求 | 包含 decode：两侧共享请求 / 全部请求 |",
             "| --- | --- | --- |"]
    for model in audited:
        cells = [f"{lookup[model, mode]['shared_request_ids']} / {lookup[model, mode]['unique_request_ids']}（{float(lookup[model, mode]['shared_pct_of_all_requests']):.2f}%）" for mode in ("1", "1,2")]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return lines + ["", "两侧没有相同的原始行，但同一请求的不同步骤仍可能分到两侧；因此这种内部检查不等于独立请求测试。此表不测量它使误差降低多少，不能拿来替代前面的冻结预测器对照。原始函数、执行命令和输入校验值保存在 verification/upstream_row_split_identity_*/；明细复制到 upstream_split_request_overlap.csv。", ""]


def key_values(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def ml_fit_input_comparison_details(
    root: Path, models: list[str], out: Path, relative: Path
) -> list[str]:
    """Rescore fit inputs with frozen bundles; never train or launch a server."""
    import importlib.util
    import joblib

    first = root / "models" / models[0] / "calibration/real"
    if not list(first.glob("repeat-*/cases/*/TP0.schedule_batch.jsonl")):
        if (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
            raise FileNotFoundError("missing formal training batches for frozen-model comparison")
        return []
    metadata = key_values(root / "run_metadata.txt")
    training_cases = set(metadata["training_case"].split(","))
    testing_cases = set(metadata["heldout_case"].split(","))
    if not training_cases < testing_cases:
        raise ValueError("frozen-model comparison requires seen and unseen workloads")
    spec = importlib.util.spec_from_file_location(
        "expanded_report_ml_scorer", Path(__file__).with_name("score_pr33824_latency_model.py")
    )
    assert spec and spec.loader
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    detailed, macro = [], []
    stages = (("fit_input", training_cases), ("independent_seen", training_cases),
              ("independent_unseen", testing_cases - training_cases))
    for model in models:
        model_root = root / "models" / model
        training_root = (model_root / "calibration/real").resolve()
        bundle_path = model_root / "calibration/training" / f"{model}_hgbmono_p50.pkl"
        bundle = joblib.load(bundle_path)
        if (list(bundle["features"]) != list(scorer.ML_FEATURE_NAMES)
                or bundle["target"] != "iter_latency_seconds"
                or set(bundle["forward_modes"]) != {1, 2}):
            raise ValueError(f"incompatible frozen ML bundle: {model}")
        if not bundle["source_data_roots"] or any(
            not Path(path).resolve().is_relative_to(training_root)
            for path in bundle["source_data_roots"]
        ):
            raise ValueError(f"bundle fitted outside declared training data: {model}")
        files = sorted(training_root.glob("repeat-*/cases/*/TP0.schedule_batch.jsonl"))
        if len(files) != int(bundle["train_cases"]):
            raise ValueError(f"frozen-model training case count mismatch: {model}")
        limit = float(bundle["max_latency_seconds"])
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(f"invalid frozen-model training filter: {model}")
        inputs = [row for row in scorer.read_rows([str(path) for path in files], "", {1, 2})
                  if row["actual_seconds"] <= limit]
        if len(inputs) != int(bundle["train_rows"]) or {row["case"] for row in inputs} != training_cases:
            raise ValueError(f"frozen-model fit input count/workload mismatch: {model}")
        fitted = scorer.score_rows(inputs, bundle["model"])
        # Recompute held-out predictions with this same frozen bundle, without
        # fitting. This verifies model identity rather than trusting file names.
        test_files = sorted((model_root / "holdout/real").glob("repeat-*/cases/*/TP0.schedule_batch.jsonl"))
        heldout = scorer.score_rows(scorer.read_rows([str(path) for path in test_files], "", {1, 2}), bundle["model"])
        recorded = read_csv(model_root / "holdout/ml_teacher_forced/per_batch.csv")
        def identities(rows):
            values = {}
            for row in rows:
                key = (str(Path(row["schedule_path"]).resolve()), int(row["line_number"]))
                if key in values:
                    raise ValueError(f"duplicate frozen-model observation: {model}")
                values[key] = (str(row["case"]), int(row["forward_mode"]), int(row["batch_size"]),
                               int(row["sum_extend"]), int(row["sum_past"]),
                               float(row["actual_seconds"]), float(row["predicted_seconds"]))
            return values
        if identities(heldout) != identities(recorded) or {row["case"] for row in heldout} != testing_cases:
            raise ValueError(f"held-out scores do not use the same frozen model/observations: {model}")
        destination = out / "ml_fit_input_readback" / model
        destination.mkdir(parents=True, exist_ok=False)
        scorer.write_csv(destination / "per_batch.csv", fitted, list(fitted[0]))
        summary = scorer.summarize(fitted)
        scorer.write_csv(destination / "summary.csv", summary, list(summary[0]))
        bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        for phase, mode in (("all", None), ("prefill", 1), ("decode", 2)):
            fit_mape = None
            for stage, cases in stages:
                source = fitted if stage == "fit_input" else heldout
                points = []
                for case in sorted(cases):
                    selected = [row for row in source if row["case"] == case and (mode is None or row["forward_mode"] == mode)]
                    if not selected:
                        raise ValueError(f"missing frozen-model comparison phase: {model}/{stage}/{case}/{phase}")
                    point = {"model": model, "stage": stage, "phase": phase, "case": case,
                             "bundle_sha256": bundle_sha, **scorer.summary_row("case", case, selected)}
                    detailed.append(point)
                    points.append(point)
                mean = lambda key: float(np.mean([float(row[key]) for row in points]))
                value = mean("mape_pct")
                if stage == "fit_input":
                    fit_mape = value
                macro.append({"model": model, "stage": stage, "phase": phase,
                              "workloads": len(points), "n": sum(row["n"] for row in points),
                              "mean_actual_ms": mean("mean_actual_ms"), "mean_predicted_ms": mean("mean_predicted_ms"),
                              "mape_pct": value, "mape_change_from_fit_input_pp": value - fit_mape,
                              "bundle_sha256": bundle_sha})
    write_csv(out / "ml_fit_input_vs_independent_by_workload.csv", detailed)
    write_csv(out / "ml_fit_input_vs_independent_macro.csv", macro)
    labels = {"fit_input": "Fit-input readback (not independent)",
              "independent_seen": "Independent requests, seen workloads",
              "independent_unseen": "Independent requests, unseen workloads"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    x = np.arange(len(models))
    for ax, phase in zip(axes, ("all", "prefill", "decode")):
        for index, (stage, _) in enumerate(stages):
            values = [next(row["mape_pct"] for row in macro if row["model"] == model and row["stage"] == stage and row["phase"] == phase) for model in models]
            ax.bar(x + (index - 1) * .25, values, .25, label=labels[stage])
        ax.set_xticks(x, models, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(f"{phase}: workload-macro MAPE (%)")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ml_fit_input_vs_independent_mape.png", dpi=180)
    plt.close(fig)
    lines = ["## 同一 ML 预测器的训练样本回评分与独立测试", "",
             "固定每模型最终 p50 bundle，回评分传入最终 fit 的训练样本，并与同一 bundle 的独立测试比较；不重新训练、不选择新的参数。训练侧沿用 bundle 的时延过滤，核对行数/用例数及数据来源；测试侧保留全部有效正时延，逐行复算并核验与正式评分完全一致。", "",
             "训练样本回评分不是独立验证；最终 HGB fit 内仍有 early-stopping 保留部分，因此也不表示每行都用于构建树。它与前文 80%/20% 内部分行验证的模型不同。下表各格为负载等权的实测/预测均值 ms（MAPE）；差距同时包含请求组成、batch 分布与采集时段变化，不能归因于请求不同这一单一因素。此处仅比较单步，不补跑训练请求的端到端服务。", "",
             "| 模型 | 训练样本回评分 ms（MAPE） | 同负载独立测试 ms（MAPE） | 跨负载独立测试 ms（MAPE） |",
             "| --- | --- | --- | --- |"]
    for model in models:
        values = [next(row for row in macro if row["model"] == model and row["stage"] == stage and row["phase"] == "all") for stage, _ in stages]
        cells = [f"{row['mean_actual_ms']:.3f}/{row['mean_predicted_ms']:.3f}（{row['mape_pct']:.2f}%）" for row in values]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return lines + ["", f"![同一冻结 ML 预测器的三类样本对照]({relative / 'ml_fit_input_vs_independent_mape.png'})", "",
                    "逐负载、prefill/decode、样本数和 bundle SHA256 见 ml_fit_input_vs_independent_by_workload.csv；等权汇总及相对训练回评分的百分点差见 ml_fit_input_vs_independent_macro.csv。训练逐行预测保存在 ml_fit_input_readback/，正式独立测试文件不修改。", ""]


def request_identity_details(root: Path, models: list[str], out: Path) -> list[str]:
    """Audit saved real prompts and replay payloads, not just seed labels."""
    experiment = key_values(root / "run_metadata.txt")
    first = root / "models" / models[0] / "calibration/real"
    if not list(first.glob("repeat-*/cases/*/TP0.raw_request.jsonl")):
        if int(experiment.get("training_requests_total", "0")):
            raise FileNotFoundError("missing real requests for train/test identity audit")
        return []  # Small report fixtures do not collect actual requests.
    rows = []
    for model in models:
        roles = {}
        for role, folder in (("training", "calibration"), ("holdout", "holdout")):
            values, ids = [], set()
            for raw_file in sorted((root / "models" / model / folder / "real").glob("repeat-*/cases/*/TP0.raw_request.jsonl")):
                raw = [json.loads(line) for line in raw_file.read_text().splitlines() if line.strip()]
                trace_file = raw_file.parent / "autobench_trace.jsonl"
                trace = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
                replay = {item["metadata"]["source_rid"]: item for item in trace}
                if len(raw) != len(trace) or len(replay) != len(trace):
                    raise ValueError(f"real/replay request count mismatch: {raw_file}")
                for item in raw:
                    rid = item["rid"]
                    prompt = item["input_ids"]
                    length = int(item["output_length"])
                    if rid in ids or len(prompt) != int(item["input_length"]):
                        raise ValueError(f"duplicate request ID or wrong saved input length: {raw_file}")
                    copied = replay.get(rid)
                    if not copied or copied["prompt"] != prompt or int(copied["output_len"]) != length or int(copied["prompt_len"]) != len(prompt):
                        raise ValueError(f"replay changes actual input IDs/output length: {raw_file}/{rid}")
                    ids.add(rid)
                    prompt_hash = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
                    values.append((prompt_hash, length))
            key = "training_requests_total" if role == "training" else "heldout_requests_total"
            if not values or len(values) != int(experiment.get(key, len(values))):
                raise ValueError(f"actual request budget mismatch: {model}/{role}")
            roles[role] = (values, ids)
        train, train_ids = roles["training"]
        test, test_ids = roles["holdout"]
        if train_ids & test_ids:
            raise ValueError(f"training/holdout reuse request IDs: {model}")
        train_prompts, train_payloads = {item[0] for item in train}, set(train)
        rows.append({"model": model, "training_requests": len(train), "heldout_requests": len(test),
                     "unique_training_prompts": len(train_prompts),
                     "unique_heldout_prompts": len({item[0] for item in test}),
                     "heldout_prompt_overlap_pct": 100 * sum(item[0] in train_prompts for item in test) / len(test),
                     "heldout_input_and_output_length_overlap_pct": 100 * sum(item in train_payloads for item in test) / len(test),
                     "reused_request_ids": 0, "real_to_replay_ids_verified": len(train) + len(test)})
    write_csv(out / "train_test_request_identity.csv", rows)
    lines = ["## 训练与测试请求核对", "",
             "直接检查真机保存的 input_ids、实际输出长度和 request ID，不只依靠 seed 声明；逐条核验回放 trace 的输入 ID/输出长度未改变。下表的重叠比例以测试请求数为分母。相同 prompt、相同 18 维特征与同一条时延标签是不同概念；少量 prompt 重复不自动构成时延标签泄漏。正式拟合仍只读取训练目录。", "",
             "| 模型 | 训练/测试请求 | 测试 prompt 与训练重叠 | 输入 ID+输出长度同时重叠 | 回放 ID 核验请求 |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['training_requests']} / {row['heldout_requests']} | {row['heldout_prompt_overlap_pct']:.2f}% | {row['heldout_input_and_output_length_overlap_pct']:.2f}% | {row['real_to_replay_ids_verified']} |")
    return lines + [""]


def historical_request_identity_details(root: Path, models: list[str], out: Path) -> list[str]:
    """Distinguish a verified historical independent matrix from row holdout."""
    candidates = list(root.glob("verification/historical_request_identity_all13_*/manifest.json"))
    if not candidates:
        if (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
            raise FileNotFoundError("missing historical request-identity audit")
        return []
    if len(candidates) != 1:
        raise ValueError("ambiguous historical request-identity audit")
    folder = candidates[0].parent
    manifest = json.loads(candidates[0].read_text())
    reference = json.loads((root / "previous_calibration/reference.json").read_text())
    previous = Path(manifest["previous_root"]).resolve()
    if previous != Path(reference["previous_root"]).resolve():
        raise ValueError("historical identity audit uses a different reference experiment")
    old = key_values(previous / "run_metadata.txt")
    if ((previous / "FINAL_REPORT_COMPLETE").read_text().strip() != "complete"
            or manifest["protocol"] != old["protocol"]
            or old["protocol"] != "same_workload_independent_train_holdout"
            or manifest["no_fit_prediction_gpu_or_historical_mutation"] is not True):
        raise ValueError("incomplete or different historical identity protocol")
    if len(manifest["models"]) != len(set(manifest["models"])) or set(manifest["models"]) != set(models):
        raise ValueError("historical request-identity model coverage differs")
    for role, count_key in (("training", "training_real_repeats"), ("heldout", "heldout_real_repeats")):
        seeds = manifest[f"{role}_seeds"]
        if seeds != old[f"{role}_seeds"].split(",") or len(set(seeds)) != int(old[count_key]):
            raise ValueError("historical identity seeds differ from recorded collection")
    if set(manifest["training_seeds"]) & set(manifest["heldout_seeds"]):
        raise ValueError("historical training/test seeds overlap")
    expected_cases = len(models) * sum(int(old[count]) * len(old[cases].split(","))
        for count, cases in (("training_real_repeats", "training_case"), ("heldout_real_repeats", "heldout_case")))
    if manifest["closed_cases"] != expected_cases:
        raise ValueError("historical identity case coverage differs")
    summary = folder / "train_test_request_identity.csv"
    helper = folder / "executed_report_helper.py"
    if (hashlib.sha256(summary.read_bytes()).hexdigest() != manifest["summary_sha256"]
            or hashlib.sha256(helper.read_bytes()).hexdigest() != manifest["executed_helper_source_sha256"]
            or manifest["executed_helper_function"] != "request_identity_details"):
        raise ValueError("historical identity summary or executed helper changed")
    sources = read_csv(folder / "source_sha256.csv")
    if (len(sources) != manifest["input_files_verified_unchanged"] or not sources
            or len({str(Path(row["path"]).resolve()) for row in sources}) != len(sources)
            or any(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"] for row in sources)):
        raise ValueError("historical identity source files changed or coverage differs")
    rows = read_csv(summary)
    if len(rows) != len(models) or {row["model"] for row in rows} != set(models):
        raise ValueError("historical request-identity summary coverage differs")
    for row in rows:
        train, test = int(row["training_requests"]), int(row["heldout_requests"])
        if (min(train, test) <= 0 or train != int(old["training_requests_total"])
                or test != int(old["heldout_requests_total"])
                or int(row["real_to_replay_ids_verified"]) != train + test
                or int(row["reused_request_ids"]) != 0
                or not 0 < int(row["unique_training_prompts"]) <= train
                or not 0 < int(row["unique_heldout_prompts"]) <= test):
            raise ValueError("historical identity request counts differ")
        overlap = [float(row[key]) for key in ("heldout_prompt_overlap_pct", "heldout_input_and_output_length_overlap_pct")]
        if not all(np.isfinite(value) and 0 <= value <= 100 for value in overlap) or overlap[1] > overlap[0]:
            raise ValueError("invalid historical prompt-overlap percentages")
    if sum(int(row["real_to_replay_ids_verified"]) for row in rows) != manifest["verified_payload_requests"]:
        raise ValueError("historical verified request total differs")
    write_csv(out / "historical_train_test_request_identity.csv", rows)
    lines = ["## 历史 ML 请求与本轮变化", "",
             f"另行核对了 {old['created_at'][:10]} 全模型矩阵的原始请求与回放 trace，共 {manifest['closed_cases']} 个闭合用例、{manifest['verified_payload_requests']} 个请求。那轮已经采用独立训练/测试 seed，并非上游脚本按 batch 行划分的内部验证；两者不能统称为同请求测试。此处核对请求身份，不将其当作拟合过程或准确率的证明。", "",
             f"历史训练 seed：{', '.join(manifest['training_seeds'])}；测试 seed：{', '.join(manifest['heldout_seeds'])}。两侧 request ID 无复用，输入 token 与输出长度均与各自回放 trace 对应。下表 prompt 重叠以测试请求数为分母。", "",
             "| 历史模型 | 训练/测试请求 | 测试 prompt 与训练重叠 | 输入 ID+输出长度重叠 | 回放 ID 核验请求 |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['training_requests']} / {row['heldout_requests']} | {float(row['heldout_prompt_overlap_pct']):.2f}% | {float(row['heldout_input_and_output_length_overlap_pct']):.2f}% | {row['real_to_replay_ids_verified']} |")
    return lines + ["", "本轮相对该历史矩阵同时增加请求量和训练负载覆盖，不能把两轮的误差变化仅解释成测试请求不同。训练侧回评分、上游内部划分验证、历史独立请求和本轮独立请求分别列示；旧预测器在本轮相同真实步骤上的评分另作配对比较。", "",
                    "明细见 historical_train_test_request_identity.csv；原始审计命令、执行时源码及输入校验值保存在 verification/historical_request_identity_all13_*/。未重新训练、预测或修改历史数据。", ""]


def aic_source_snapshot_details(root: Path, revision: str, *, required: bool) -> list[str]:
    """Check archived collector bytes, not just the checkout's HEAD label."""
    folder = root / "aic_calibration/provenance_before_wide_sweep"
    if not (folder / "identity.json").is_file():
        if required:
            raise FileNotFoundError("missing formal AIC collector source snapshot")
        return []
    identity = json.loads((folder / "identity.json").read_text())
    if identity["aic_revision"] != revision:
        raise ValueError("AIC source snapshot revision differs from calibration metadata")
    details = identity["aic_source_snapshot"]
    checksums = json.loads((folder / "aic_source_sha256.json").read_text())
    if len(checksums) != details["source_files"]:
        raise ValueError("AIC source snapshot file count mismatch")
    for name in ("aic_git_status.txt", "aic_tracked_changes.patch"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"missing AIC source provenance: {name}")
    with tarfile.open(folder / "aic_collector_and_sdk_sources.tar.gz") as archive:
        names = archive.getnames()
        if len(names) != len(checksums) or set(names) != set(checksums):
            raise ValueError("AIC source archive/manifest mismatch")
        for name, expected in checksums.items():
            stream = archive.extractfile(name)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                raise ValueError(f"AIC source archive checksum mismatch: {name}")
    imported = identity["aic_import_source_match"]
    if imported["mismatches"] or imported["checked_python_files"] <= 0:
        raise ValueError("AIC SDK import differs from archived checkout")
    return ["## AIC 采集源码记录", "",
            f"采集前归档了 {len(checksums)} 个 collector/SDK/配置源文件，并核验归档内容与 SHA256 清单一致；实际导入的 SDK 与声明 checkout 的 {imported['checked_python_files']} 个 Python 文件逐一相同。checkout dirty={details['dirty']}，因此源码身份由 revision、dirty patch 和文件归档共同确定。原生 kernel 的一致性另按二进制 SHA256 检查。", "",
            "归档位于 aic_calibration/provenance_before_wide_sweep/；算子性能表与本次重新测量记录位于同一校准目录，不混入源码归档。", ""]


def runtime_identity_details(root: Path, models: list[str], out: Path) -> list[str]:
    """Check recorded real imports against the actual operator collector."""
    reference = root / "provenance/formal_start_after_ninja_fix/identity.json"
    if not reference.is_file():
        return []  # Small report fixtures do not run a real collection.
    experiment = key_values(root / "run_metadata.txt")
    aic = key_values(root / "aic_calibration/environment/collector_runtime_versions.txt")
    aic_source = key_values(root / "aic_calibration/trace_targeted_run_metadata.txt")
    if aic_source["source_revision"] != experiment["source_revision"]:
        raise ValueError("AIC collector and real baseline source revisions differ")
    rows = []
    for model in models:
        versions = []
        for role, folder, repeat_key in (("training", "calibration", "training_real_repeats"), ("holdout", "holdout", "heldout_real_repeats")):
            groups = sorted((root / "models" / model / folder / "real").glob("repeat-*"))
            if len(groups) != int(experiment[repeat_key]):
                raise ValueError(f"incomplete recorded real repeats: {model}/{role}")
            for group in groups:
                meta = key_values(group / "run_metadata.txt")
                if meta["source_revision"] != experiment["source_revision"]:
                    raise ValueError(f"changed baseline source revision: {group}")
                versions.append(key_values(group / "package_versions.txt"))
        keys = ("python", "sglang", "torch", "torch_cuda", "sgl_kernel_runtime_version", "sglang-kernel")
        unique = {tuple(version[key] for key in keys) for version in versions}
        if len(unique) != 1:
            raise ValueError(f"baseline package versions changed between real repeats: {model}")
        current = versions[0]
        for version in versions:
            for key in ("torch", "torch_cuda", "sgl_kernel_runtime_version"):
                if version[key] != aic[key]:
                    raise ValueError(f"AIC collector/real version mismatch: {model}/{key}")
            if Path(version["sgl_kernel_file"]).resolve() != Path(aic["sgl_kernel_file"]).resolve():
                raise ValueError(f"AIC collector imports a different native stage: {model}")
        rows.append({"model": model, "recorded_real_groups": len(versions), **{key: current[key] for key in keys},
                     "source_revision": experiment["source_revision"], "native_import": current["sgl_kernel_file"],
                     "aic_collector_native_import": aic["sgl_kernel_file"]})
    write_csv(out / "runtime_identity_checks.csv", rows)
    lines = ["## 实际运行版本核验", "",
             "下表来自每组真实采集的 package_versions.txt，而不是 conda 环境名称。核对了训练与独立测试组版本一致、AIC 采集的 Torch/CUDA/kernel 运行版本及实际 native 导入路径一致。采集前另以原生二进制 SHA256 对照正式基线快照。SGLang 元数据为 0.0.0，源码身份以上文 revision 和 dirty 源码快照为准。", "",
             "| 模型 | 核验真实组数 | SGLang 元数据 | Torch / CUDA | kernel 运行 / 包元数据 |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['recorded_real_groups']} | {row['sglang']} | {row['torch']} / {row['torch_cuda']} | {row['sgl_kernel_runtime_version']} / {row['sglang-kernel']} |")
    lines.extend(["", f"AIC attention 采集 page size 为 {aic_source['collector_triton_page_size']}，KV split 上限为 8（非固定 split 数）。普通 BF16 Triton 基线按 page=1 对齐；1M dual-chunk/page=256、AWQ 精度和 MoE 路由的代理差异不因版本核验而消失。", ""])
    lines.extend(aic_source_snapshot_details(root, aic_source.get("aic_source_revision", ""),
                                           required=int(experiment.get("training_requests_total", "0")) >= 800))
    return lines


def workload_macro_details(batch: list[dict], e2e: list[dict], models: list[str], cases: list[str], training_cases: list[str], out: Path) -> list[str]:
    rows = []
    for model in models:
        for method in ("ML", "AIC"):
            for domain, selected in (("seen_workloads", [case for case in cases if case in training_cases]),
                                     ("unseen_workloads", [case for case in cases if case not in training_cases])):
                if not selected:
                    continue
                row = {"model": model, "method": method, "domain": domain, "workload_count": len(selected)}
                for phase in ("all", "prefill", "decode"):
                    values = [float(entry["mape_pct"]) for entry in batch if entry["model"] == model and entry["method"] == method and entry["phase"] == phase and entry["case"] in selected]
                    if len(values) != len(selected):
                        raise ValueError(f"incomplete macro workload cells: {model}/{method}/{domain}/{phase}")
                    row[f"{phase}_workload_macro_mape_pct"] = float(np.mean(values))
                values = [float(entry["mape_pct"]) for entry in e2e if entry["model"] == model and entry["method"] == method and entry["metric"] == "mean_e2e_latency_ms" and entry["case"] in selected]
                row["e2e_workload_macro_mape_pct"] = float(np.mean(values))
                rows.append(row)
    write_csv(out / "workload_macro_seen_unseen.csv", rows)
    lines = ["## 按负载等权的同负载/跨负载结果", "",
             "以下先分别计算每类负载的误差，再对负载等权平均，避免低到达率用例较多的单请求 decode 行主导总览。前三种是同负载独立请求，后两种是跨负载；这个划分不等同于严格的特征分布内/分布外。原始逐行加权结果仍在前文保留。", "",
             "| 模型 | 方法 | 同负载单步 MAPE | 跨负载单步 MAPE | 同负载 E2E MAPE | 跨负载 E2E MAPE |",
             "| --- | --- | --- | --- | --- | --- |"]
    for model in models:
        for method in ("ML", "AIC"):
            matching = {row["domain"]: row for row in rows if row["model"] == model and row["method"] == method}
            if set(matching) != {"seen_workloads", "unseen_workloads"}:
                continue
            a, b = matching["seen_workloads"], matching["unseen_workloads"]
            lines.append(f"| {model} | {method} | {a['all_workload_macro_mape_pct']:.2f}% | {b['all_workload_macro_mape_pct']:.2f}% | {a['e2e_workload_macro_mape_pct']:.2f}% | {b['e2e_workload_macro_mape_pct']:.2f}% |")
    return lines + [""]


def workload_feature_coverage_details(root: Path, models: list[str], cases: list[str], batch_metrics: list[dict], out: Path, rel: Path) -> list[str]:
    """Pair phase-specific training support with the same held-out batch errors.

    Bounds only describe marginal support; they do not establish identical
    distributions or a causal explanation of the prediction error.
    """
    from sglang_simulator.time_predictor.ml_features import extract_ml_features

    def load(folder: Path, *, training: bool) -> list[dict]:
        files = sorted(folder.glob("repeat-*/cases/*/TP0.schedule_batch.jsonl"))
        if not files:
            raise FileNotFoundError(f"no real schedule batches for feature coverage: {folder}")
        values = []
        for path in files:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    raw = json.loads(line)
                    requests = raw.get("request_infos") or []
                    mode = raw.get("forward_mode")
                    latency = float(raw.get("iter_latency", 0))
                    if mode not in (1, 2) or not requests:
                        continue
                    if not np.isfinite(latency) or latency <= 0:
                        raise ValueError(f"invalid measured latency in feature audit: {path}")
                    if training and latency > 30:
                        continue  # Match the actual trainer's input filter.
                    features = extract_ml_features([int(item["extend_input_len"]) for item in requests],
                                                   [int(item["prefix_indices_len"]) for item in requests])
                    values.append({"case": path.parent.name, "phase": {1: "prefill", 2: "decode"}[mode], "features": features})
        return values

    # Unit report fixtures have no real collection. A formal run must not
    # silently skip a missing data source merely because its CSVs exist.
    first_training = root / "models" / models[0] / "calibration/real"
    if not list(first_training.glob("repeat-*/cases/*/TP0.schedule_batch.jsonl")):
        if (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
            raise FileNotFoundError(f"missing formal training data: {first_training}")
        return []
    rows = []
    for model in models:
        model_root = root / "models" / model
        train, test = load(model_root / "calibration/real", training=True), load(model_root / "holdout/real", training=False)
        if {row["case"] for row in test} != set(cases):
            raise ValueError(f"feature coverage workloads differ from scoring: {model}")
        for phase in ("all", "prefill", "decode"):
            reference = np.asarray([row["features"] for row in train if phase == "all" or row["phase"] == phase])
            if not len(reference):
                raise ValueError(f"missing training phase for feature coverage: {model}/{phase}")
            lower, upper = reference.min(axis=0), reference.max(axis=0)
            for case in cases:
                actual = np.asarray([row["features"] for row in test if row["case"] == case and (phase == "all" or row["phase"] == phase)])
                compared = [row for row in batch_metrics if row["model"] == model and row["case"] == case and row["phase"] == phase]
                if len(compared) != 2 or any(int(row["n"]) != len(actual) for row in compared) or not len(actual):
                    raise ValueError(f"coverage and scoring batch counts differ: {model}/{case}/{phase}")
                metrics = {row["method"]: float(row["mape_pct"]) for row in compared}
                row = {"model": model, "case": case, "phase": phase, "train_rows": len(reference), "test_rows": len(actual),
                       "outside_train_range_pct": 100 * float(np.mean(np.any((actual < lower) | (actual > upper), axis=1))),
                       "ml_mape_pct": metrics["ML"], "aic_mape_pct": metrics["AIC"]}
                for name, index in (("batch_size", 0), ("sum_extend", 1), ("sum_past", 4)):
                    row[f"train_{name}_p50"] = float(np.median(reference[:, index]))
                    row[f"test_{name}_p50"] = float(np.median(actual[:, index]))
                    row[f"train_{name}_min"] = float(lower[index])
                    row[f"train_{name}_max"] = float(upper[index])
                rows.append(row)
    write_csv(out / "workload_feature_coverage.csv", rows)
    lines = ["## 分负载的训练特征覆盖", "",
             "下面按 all/prefill/decode 分别计算训练集 18 维特征的逐维最小/最大值，统计每种测试负载至少一维超出范围的 batch 比例。训练范围采用最终 fit 的输入过滤口径；测试保留所有有效正时延，且核对每格样本数与 ML/AIC 正式评分一致。范围内不代表同分布，范围外也不是误差的唯一原因。batch_size、sum_extend、sum_past 的训练/测试中位数和训练范围保存在 workload_feature_coverage.csv。", ""]
    lookup = {(row["model"], row["case"], row["phase"]): row for row in rows}
    for phase in ("all", "prefill", "decode"):
        values = np.asarray([[lookup[(model, case, phase)]["outside_train_range_pct"] for case in cases] for model in models])
        fig, ax = plt.subplots(figsize=(9, max(5, len(models) * .38)))
        picture = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(cases)), cases, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)), models, fontsize=8)
        for i in range(len(models)):
            for j in range(len(cases)):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{phase}: held-out batches outside marginal training-feature ranges (%)")
        fig.colorbar(picture, ax=ax, label="Outside training-feature ranges (%)")
        fig.tight_layout()
        filename = f"workload_feature_coverage_{phase}.png"
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
        lines.extend([f"![{phase} 分负载训练范围覆盖]({rel / filename})", ""])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, phase in zip(axes, ("all", "prefill", "decode")):
        for case in cases:
            points = [lookup[(model, case, phase)] for model in models]
            ax.scatter([row["outside_train_range_pct"] for row in points], [row["ml_mape_pct"] for row in points], s=26, alpha=.75, label=case)
        ax.set_title(phase)
        ax.set_xlabel("Outside training-feature ranges (%)")
        ax.set_ylabel("ML batch MAPE (%)")
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(cases)), fontsize=7)
    fig.suptitle("Training support and held-out ML error: descriptive comparison, not causal attribution")
    fig.tight_layout(rect=(0, .13, 1, .93))
    fig.savefig(out / "feature_coverage_vs_ml_error.png", dpi=180)
    plt.close(fig)
    return lines + [f"![训练覆盖率与 ML 误差]({rel / 'feature_coverage_vs_ml_error.png'})", ""]


def telemetry_details(root: Path, models: list[str], out: Path) -> list[str]:
    log = root / "hardware/gpu_telemetry.jsonl"
    if not log.is_file():
        return ["## GPU 状态记录", "", "本轮没有连续 GPU 遥测文件；启动快照不足以排除运行期间干扰。", ""]
    snapshot = log.read_text()
    # The watcher remains active until the report marker is written. A read
    # racing its final append may see a partial line; retain only complete lines.
    if snapshot and not snapshot.endswith("\n"):
        snapshot = snapshot.rsplit("\n", 1)[0] if "\n" in snapshot else ""
    samples = [json.loads(line) for line in snapshot.splitlines() if line.strip()]
    rows = []
    zone = datetime.now().astimezone().tzinfo
    pattern = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
    for model in models:
        model_root = root / "models" / model
        for role, folder in (("training", "calibration"), ("holdout", "holdout")):
            for server_log in model_root.glob(f"{folder}/real/repeat-*/cases/*.server.log"):
                starts, ends, compilations = [], [], []
                for line in server_log.read_text().splitlines():
                    match = pattern.match(line)
                    if not match:
                        continue
                    stamp = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).timestamp()
                    if '"POST /start_profile HTTP' in line:
                        starts.append(stamp)
                    if '"POST /stop_profile HTTP' in line:
                        ends.append(stamp)
                    if "compile after serving started" in line:
                        compilations.append(stamp)
                if len(starts) != 1 or len(ends) != 1:
                    raise ValueError(f"expected exactly one successful profiling window: {server_log}")
                window = [sample for sample in samples if starts[0] <= sample["wall_time"] <= ends[0] and "gpu" in sample]
                others = [sample for sample in window if any(not process["own_real_collector"] for process in sample.get("processes", []))]
                rows.append({"model": model, "role": role, "repeat": server_log.parents[1].name,
                             "case": server_log.name.removesuffix(".server.log"),
                             "telemetry_samples": len(window), "samples_with_other_resident_processes": len(others),
                             "profile_logged_slow_compilations": sum(starts[0] <= stamp <= ends[0] for stamp in compilations),
                             "profile_start_wall_time": starts[0], "profile_end_wall_time": ends[0]})
    if rows:
        write_csv(out / "gpu_collection_windows.csv", rows)
    lines = ["## GPU 状态与采集质量", "",
             "每次服务启动前记录显存和驻留进程，另每 30 秒采样一次 GPU 功率、频率、温度和进程。下表只统计真实 profile 时间窗；未采到的窗口记为未覆盖，不当作没有干扰。其他进程驻留不等同于同时执行计算，30 秒采样也不能排除短时干扰。", "",
             "| 模型 | 真实用例总数 | 遥测覆盖用例 | 采到其他驻留进程的用例 | profile 内日志记录的慢编译 |",
             "| --- | --- | --- | --- | --- |"]
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        lines.append(f"| {model} | {len(subset)} | {sum(row['telemetry_samples'] > 0 for row in subset)} | {sum(row['samples_with_other_resident_processes'] > 0 for row in subset)} | {sum(row['profile_logged_slow_compilations'] for row in subset)} |")
    return lines + ["", "原始遥测和每用例统计均保留；日志慢编译统计不是所有编译事件的完整计数，不据此自动删去较慢样本。", ""]


def replay_overhead_details(
    root: Path,
    models: list[str],
    out: Path,
    methods: tuple[str, ...] = ("ML", "AIC"),
) -> list[str]:
    """Describe recorded simulator wall time, without correcting predictions."""
    if not (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
        return []
    metadata = key_values(root / "run_metadata.txt")
    repeats = int(metadata["heldout_real_repeats"])
    cases = metadata["heldout_case"].split(",")
    rows = []
    for model in models:
        for method in methods:
            folder = {"ML": "ml_simulator", "AIC": "aic_simulator"}[method]
            for repeat in range(1, repeats + 1):
                for case in cases:
                    path = root / "models" / model / "holdout" / folder / f"repeat-{repeat}" / "cases" / case / "simulator-output"
                    metrics = json.loads((path / "metrics.json").read_text())
                    stats = [json.loads(line) for line in (path / "iteration.jsonl").read_text().splitlines() if line.strip()]
                    iterations = int(metrics["iterations"])
                    if not iterations or iterations != len(stats):
                        raise ValueError(f"replay iteration count mismatch: {path}")
                    predictor = float(metrics["predictor_time_cost"])
                    cpu = [float(row["cpu_overhead"]) for row in stats]
                    forward = [float(row["forward_latency"]) for row in stats]
                    if not all(np.isfinite(value) and value >= 0 for value in [predictor, *cpu, *forward]):
                        raise ValueError(f"invalid replay timing component: {path}")
                    rows.append({"model": model, "method": method, "repeat": repeat, "case": case,
                                 "iterations": iterations, "predictor_wall_seconds": predictor,
                                 "accounted_cpu_seconds": sum(cpu), "predicted_forward_seconds": sum(forward),
                                 "predictor_wall_ms_per_iteration": predictor * 1000 / iterations,
                                 "accounted_cpu_ms_per_iteration": sum(cpu) * 1000 / iterations,
                                 "predicted_forward_ms_per_iteration": sum(forward) * 1000 / iterations})
    write_csv(out / "replay_wall_time_components.csv", rows)
    lines = ["## 离线回放的计算开销", "",
             "回放日志同时记录 predicted forward、cpu_overhead 和 predictor_time_cost。当前 OFFLINE 路径以相邻处理点的墙钟间隔推进 CPU 时间，预测函数在该间隔内执行；predictor_time_cost 另行汇总，但没有从 cpu_overhead 中扣除。所以下表是模拟器自身的执行开销，不是独立测得的原生服务 CPU 时延，也不是 GPU 算子时延。依据为本轮归档的 src/sglang_simulator/simulation/sglang/scheduler.py。", "",
             "均值按回放 iteration 数加权；不同预测方法形成的动态 batch 数可能不同。每个用例、每次重复的明细保存在 replay_wall_time_components.csv。保留原始端到端结果，没有事后扣除平均开销；扣除会改变后续调度，效果需要另行回放验证。", "",
             "| 模型 | 方法 | 核对用例数 | 回放 iteration 数 | 预测函数墙钟 ms/iter | 计入时钟的 CPU ms/iter | predicted forward ms/iter |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in models:
        for method in methods:
            selected = [row for row in rows if row["model"] == model and row["method"] == method]
            iterations = sum(row["iterations"] for row in selected)
            mean = lambda key: sum(row[key] for row in selected) * 1000 / iterations
            lines.append(f"| {model} | {method} | {len(selected)} | {iterations} | {mean('predictor_wall_seconds'):.3f} | {mean('accounted_cpu_seconds'):.3f} | {mean('predicted_forward_seconds'):.3f} |")
    return lines + [""]


def regression_verification_details(root: Path, out: Path) -> list[str]:
    """Expose recorded regression failures separately from GPU accuracy results."""
    record_file = root / "verification/full_suite_record.json"
    if not record_file.is_file():
        if (root / "provenance/formal_start_after_ninja_fix/identity.json").is_file():
            raise FileNotFoundError("missing full-suite verification record")
        return []
    record = json.loads(record_file.read_text())
    result = record["result"]
    failed, exit_code = int(result["failed"]), int(result["exit_code"])
    if failed < 0 or int(result["passed"]) < 0 or (failed > 0 and exit_code == 0):
        raise ValueError("inconsistent full-suite result")
    lines = ["## 工程回归验证（非 GPU 精度实验）", "",
             f"{record['date']} 在 hbf-cxl 执行 `{record['command']}`："
             f"{result['passed']} 项通过、{failed} 项失败、{result['warnings']} 条警告，"
             f"耗时 {float(result['elapsed_seconds']):.2f} 秒，退出码 {exit_code}。", "",
             record["scope_note"], ""]
    if failed:
        lines.extend([f"失败用例：`{record['failure']['test']}`。以下从保存的 fixture 输出重新读取，未修改测量值或容差。", "",
                      "| 指标 | offline | blocking | 相对差异 | 测试容差 |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        failure = record["failure"]
        metrics = {mode: json.loads((root / failure["outputs"][mode]).read_text())
                   for mode in ("offline", "blocking")}
        rows = []
        for metric, tolerance in failure["tolerances"].items():
            actual, expected = (float(metrics[mode][metric]) for mode in ("offline", "blocking"))
            if not all(np.isfinite(value) and value > 0 for value in (actual, expected)):
                raise ValueError(f"invalid saved regression metric: {metric}")
            difference = abs(actual - expected) / abs(expected) * 100
            rows.append({"metric": metric, "offline": actual, "blocking": expected,
                         "relative_difference_pct": difference, "tolerance_pct": float(tolerance) * 100})
            lines.append(f"| {metric} | {actual:.6f} | {expected:.6f} | {difference:.4f}% | {float(tolerance) * 100:g}% |")
        if not any(row["relative_difference_pct"] > row["tolerance_pct"] for row in rows):
            raise ValueError("saved fixtures do not reproduce the recorded regression failure")
        write_csv(out / "full_suite_fixture_comparison.csv", rows)
        lines.extend(["", failure["interpretation_note"], ""])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = dict(line.split("=", 1) for line in (args.results_root / "run_metadata.txt").read_text().splitlines() if "=" in line)
    models = metadata["models"].split(",")
    train_cases = metadata["training_case"].split(",")
    cases = metadata["heldout_case"].split(",")
    include_previous = wait_previous_comparison(args.results_root)
    batch, e2e, curves = collect(args.results_root, models, cases, int(metadata["heldout_real_repeats"]))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.out_dir / "workload_batch_metrics.csv", batch)
    write_csv(args.out_dir / "workload_end_to_end_metrics.csv", e2e)
    write_csv(args.out_dir / "learning_curves.csv", curves)
    lines = ["", "## 同负载与跨负载测试", "",
             "前三类负载出现在训练数据中，但测试请求和 seed 独立；后两类负载不用于训练。跨负载不代表所有特征都在训练范围之外。以下按模型与负载列出实测值和预测值，避免把五类负载混成一个精度结论。", "",
             "总 token 吞吐采用 collector/trace 的实际输入+输出 token 数，除以原始 benchmark duration；原始输入计数和原报吞吐保留在 comparison.csv。随机 ID 解码成文本后再编码可能改变输入长度，此处统一 token 分子，不消除 client/模拟器时间窗差异。std 为五次重复的总体标准差（分母 n），不是置信区间。", ""]
    specs = {"latency_256x128_r1": (256, 128, "1 请求/秒", 32),
             "service_1024x256_r4": (1024, 256, "4 请求/秒", 64),
             "saturation_1024x256_inf": (1024, 256, "一次提交，不限速", 64),
             "latency_128x64_r1": (128, 64, "1 请求/秒", 32),
             "saturation_1536x384_inf": (1536, 384, "一次提交，不限速", 64)}
    lines.extend(["| 用例名称 | 生成输入/输出长度上限 | 到达设置 | 每组请求 | 用于训练 |",
                  "| --- | --- | --- | --- | --- |"])
    for case in cases:
        a, b, rate, requests = specs[case]
        lines.append(f"| {case} | {a} / {b} token | {rate} | {requests} | {'是' if case in train_cases else '否'} |")
    lines.extend(["", "长度是随机生成上限，不是每个请求的固定长度；range_ratio=0 从 1 到上限采样。1536/384 长用例使用 --tokenize-prompt 直接发送随机 ID，避免文本再编码使输入+输出超出 2048 上下文；其他用例沿用随机 ID 解码成文本再发送，实际输入 ID 保存在 trace。长用例相对训练还改变了输入路径，不能将其差异只归因于长度。r1/r4 为平均到达率，有限速请求间隔为指数分布；inf 是突发提交，不是无限并发，服务最多同时运行 32 个请求。", ""])
    rel = Path(os.path.relpath(args.out_dir, args.markdown.parent))
    lines = runtime_identity_details(args.results_root, models, args.out_dir) + training_details(args.results_root, models, args.out_dir) + request_identity_details(args.results_root, models, args.out_dir) + replay_overhead_details(args.results_root, models, args.out_dir) + kv_input_contract_details(args.results_root, models, args.out_dir, rel) + ml_runtime_refit_details(args.results_root, models, args.out_dir, rel) + aic_kv_input_contract_details(args.results_root, models, args.out_dir, rel) + aic_cache_contract_details(args.results_root, models, args.out_dir) + native_profile_details(args.results_root, args.out_dir, rel) + lines
    for case in cases:
        label = "同负载独立测试" if case in train_cases else "跨负载测试"
        lines.extend([f"### {case}（{label}）", "",
                      "| 模型 | 方法 | 单步实测/预测 ms | 单步 MAPE / MAE ms | E2E 实测 ± std / 预测 ms | E2E MAPE | TTFT 实测 ± std / 预测 ms（MAPE） | TPOT 实测 ± std / 预测 ms（MAPE） | 总 token 吞吐实测/预测 tok/s |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
        for model in models:
            for method in ("ML", "AIC"):
                b = next(row for row in batch if row["model"] == model and row["method"] == method and row["case"] == case and row["phase"] == "all")
                metrics = {row["metric"]: row for row in e2e if row["model"] == model and row["method"] == method and row["case"] == case}
                r = metrics["mean_e2e_latency_ms"]
                t = metrics["total_throughput"]
                ttft, tpot = metrics["mean_ttft_ms"], metrics["mean_tpot_ms"]
                f = lambda row, key: float(row[key])
                lines.append(f"| {model} | {method} | {f(b,'mean_actual_ms'):.2f} / {f(b,'mean_predicted_ms'):.2f} | {f(b,'mape_pct'):.2f}% / {f(b,'mae_ms'):.2f} | {f(r,'real_mean'):.2f} ± {f(r,'real_std'):.2f} / {f(r,'predicted_mean'):.2f} | {f(r,'mape_pct'):.2f}% | {f(ttft,'real_mean'):.2f} ± {f(ttft,'real_std'):.2f} / {f(ttft,'predicted_mean'):.2f}（{f(ttft,'mape_pct'):.2f}%） | {f(tpot,'real_mean'):.2f} ± {f(tpot,'real_std'):.2f} / {f(tpot,'predicted_mean'):.2f}（{f(tpot,'mape_pct'):.2f}%） | {f(t,'real_mean'):.2f} / {f(t,'predicted_mean'):.2f} |")
        lines.extend(["", "按阶段展开单步结果。n 为五次独立真机实验合并后的 batch 条数，P95 为逐 batch 绝对百分比误差的 95 分位。", "",
                      "| 模型 | 方法 | n（prefill/decode） | prefill 实测/预测 ms | prefill MAPE / MAE ms | decode 实测/预测 ms | decode MAPE / MAE ms | P95 APE（prefill/decode） |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"])
        for model in models:
            for method in ("ML", "AIC"):
                phases = {row["phase"]: row for row in batch if row["model"] == model and row["method"] == method and row["case"] == case}
                p, d = phases["prefill"], phases["decode"]
                lines.append(f"| {model} | {method} | {p['n']} / {d['n']} | {float(p['mean_actual_ms']):.2f} / {float(p['mean_predicted_ms']):.2f} | {float(p['mape_pct']):.2f}% / {float(p['mae_ms']):.2f} | {float(d['mean_actual_ms']):.2f} / {float(d['mean_predicted_ms']):.2f} | {float(d['mape_pct']):.2f}% / {float(d['mae_ms']):.2f} | {float(p['p95_ape_pct']):.2f}% / {float(d['p95_ape_pct']):.2f}% |")
        lines.append("")
    for method in ("ML", "AIC"):
        for category in ("batch", "e2e"):
            rows = batch if category == "batch" else e2e
            selected = [row for row in rows if row["method"] == method and row.get("phase", "all") == "all" and row.get("metric", "mean_e2e_latency_ms") == "mean_e2e_latency_ms"]
            values = np.asarray([[float(next(row["mape_pct"] for row in selected if row["model"] == model and row["case"] == case)) for case in cases] for model in models])
            fig, ax = plt.subplots(figsize=(9, max(5, len(models) * .38)))
            picture = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0)
            ax.set_xticks(range(len(cases)), cases, rotation=25, ha="right", fontsize=8)
            ax.set_yticks(range(len(models)), models, fontsize=8)
            for i in range(len(models)):
                for j in range(len(cases)):
                    ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"{method}: {category} MAPE (%) by workload")
            fig.colorbar(picture, ax=ax, label="MAPE (%)")
            fig.tight_layout()
            filename = f"{method.lower()}_{category}_workload_heatmap.png"
            fig.savefig(args.out_dir / filename, dpi=180)
            plt.close(fig)
            lines.extend([f"![{method} {category} 分负载误差]({rel / filename})", ""])
    requests_per_group = int(metadata.get("requests_per_training_repeat", "160"))
    lines.extend(["## 训练量递增对比", "", f"测试集保持不变，以前 1、3、5 组训练 trace 分别拟合，训练请求数为 {requests_per_group}、{requests_per_group * 3}、{requests_per_group * 5}；下面曲线按负载展示。增加样本是否有效以实测误差变化为准，不预设单调改善。", ""])
    labels = {"latency_256x128_r1": "256/128 @1/s", "service_1024x256_r4": "1024/256 @4/s",
              "saturation_1024x256_inf": "1024/256 burst", "latency_128x64_r1": "128/64 @1/s",
              "saturation_1536x384_inf": "1536/384 burst"}
    for offset in range(0, len(models), 4):
        group = models[offset:offset + 4]
        single = len(group) == 1
        fig, axes = plt.subplots(1 if single else 2, 1 if single else 2, figsize=(7, 4.5) if single else (10, 7), squeeze=False)
        for ax, model in zip(axes.flat, group):
            for case in cases:
                rows = sorted([row for row in curves if row["model"] == model and row.get("case") == case], key=lambda row: int(row["training_traces"]))
                if len(rows) != 3:
                    raise ValueError(f"incomplete per-workload learning curve: {model}/{case}")
                ax.plot([int(row["training_traces"]) * requests_per_group for row in rows], [float(row["mape_pct"]) for row in rows], marker="o", label=labels.get(case, case))
            ax.set_title(model, fontsize=11)
            ax.set_xlabel("Training requests", fontsize=10)
            ax.set_ylabel("Batch MAPE (%)", fontsize=10)
            ax.grid(alpha=.25)
        for ax in list(axes.flat)[len(group):]:
            ax.set_visible(False)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=10)
        fig.tight_layout(rect=(0, .1, 1, 1))
        filename = f"learning_curves_models_{offset + 1}_{offset + len(group)}.png"
        fig.savefig(args.out_dir / filename, dpi=180)
        plt.close(fig)
        lines.extend([f"![模型 {offset + 1}–{offset + len(group)} 训练量曲线]({rel / filename})", ""])
    lines.extend([
                  "完整 prefill/decode、P95 APE、偏差与样本数见同目录 workload_batch_metrics.csv；全部 E2E、TTFT、TPOT、吞吐及实机标准差见 workload_end_to_end_metrics.csv。", "",
                  "## 结果解释的边界", "",
                  "同步墙钟单步时延包含 run_batch 内 CPU 发射、采样/拷贝和同步等待，AIC 算子 GPU 时延求和不包含全部这些开销。因此增加 AIC 算子点与提高端到端准确性不是同一件事。AWQ BF16 代理、dual-chunk attention 代理与 MoE 路由覆盖限制沿用前文单列说明。当前 GPU 其他进程及显存状态已记录；本轮不是无干扰的硬件性能标定。", ""])
    lines.extend([
        "AIC 的 Silicon 模式也不是所有算子都查实测表：此版本将 RMSNorm、SwiGLU 等归为 ElementWise，并调用 query_mem_op 的带宽/固定延迟公式。当前配置为带宽 1.935 TB/s、经验缩放 0.8、固定延迟 3 μs；重新采集 attention/GEMM 不会更新这些项。这里保留标准 SDK 实现，解析项与本机实测表分别说明。源码：[ElementWise](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/operations/elementwise.py)、[query_mem_op](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/perf_database.py)。", "",
        "attention harness 在计时前调用 init_forward_metadata，测量时仅重复执行 RadixAttention，并先 warmup 三次；真机 run_batch 的同步墙钟标签包含相关元数据准备。两条路径都构造 TritonAttnBackend，KV-split 上限 8 不代表每个请求固定使用 8 个 split。元数据准备、首次 shape 编译和稳态 GPU 算子测量的边界应区分；尚无同配置逐项 profile，不能为这些开销分配具体误差比例。", "",
        "算子采样默认使用 CUDA Graph：benchmark_with_power 在 capture 前后分别预热，再用 CUDA event 计时重放；本轮真机则关闭 graph，测量完整 run_batch 的同步墙钟。GEMM harness 按显存预算建立 1–6 组独立矩阵，依次执行后按组数取平均；attention harness 重用一组层输入与 KV 池。计时循环内没有显式缓存驱逐步骤，但这不足以证明实际缓存命中率或它造成的误差，不能把所有 GEMM 采样概括为反复读取同一份权重，也不能据此给预测值补一个固定延迟。不同层权重、缓存和发射开销的影响仍需同配置 GPU profile 验证。依据：[采样计时](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/collector/helper.py)、[GEMM harness](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/collector/sglang/collect_gemm.py)；实际 collector 修改与源码校验和已存入本轮 provenance。", "",
        "Qwen3 的 QK norm 还存在阶段差异：ContextAttention 用解析项附加其时延；GenerationAttention 虽保留 use_qk_norm 标志，query 中没有对应附加项。本机 attention harness 计时的是 RadixAttention 调用，不包含模型层前面的 QK norm/旋转位置编码调用，因此加密这张表不等于补采这些操作。CPU 调用核验在 B=1、S=128 和 B=8、S=1024 下，开关 QK norm 的 prefill 差值均为约 0.0264 ms/层，decode 差值为零；这是 SDK 公式/调用行为，不是 GPU 实测时延，也不用于修改预测值。没有当前逐项 profile 证据，不能据此量化它占总误差的比例。依据：[attention 查询](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/operations/attention.py)、[attention harness](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/collector/sglang/collect_attn.py)；调用记录见 preflight/aic_attention_query_scope_20260928.json。", "",
        "AIC checkout 在本轮开始前已有三处 collector 修改：两份 attention YAML 增补 head=28，以及 collect_attn.py 的 runner 兼容字段和可选 Torch SDPA 路径。正式采集显式选择 Triton，不进入可选 SDPA 分支；这些原有修改未回滚，也未在本轮改写。provenance 同时保存 AIC dirty patch、collector/SDK 源文件及 SHA256，不能仅用 HEAD 表示实际采集代码。", "",
    ])
    lines.extend(workload_macro_details(batch, e2e, models, cases, train_cases, args.out_dir))
    lines.extend(ml_fit_input_comparison_details(args.results_root, models, args.out_dir, rel))
    lines.extend(residual_concentration_details(args.results_root, models, args.out_dir))
    lines.extend(workload_feature_coverage_details(args.results_root, models, cases, batch, args.out_dir, rel))
    if include_previous:
        lines.extend(append_previous_comparison(args.results_root, models, cases, args.out_dir, rel))
    lines.extend(historical_aic_baseline_details(args.results_root, models, args.out_dir, rel))
    lines.extend(historical_aic_one_batch_details(args.results_root, models, args.out_dir, rel))
    lines.extend(telemetry_details(args.results_root, models, args.out_dir))
    lines.extend(regression_verification_details(args.results_root, args.out_dir))
    args.markdown.write_text(args.markdown.read_text(encoding="utf-8") + "\n".join(lines), encoding="utf-8")
    print(f"workload batch rows={len(batch)} e2e rows={len(e2e)} curve rows={len(curves)}")


if __name__ == "__main__":
    main()
