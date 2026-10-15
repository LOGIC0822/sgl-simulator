#!/usr/bin/env python3
"""Summarize the standardized all-model ML/AIC predictor experiment.

This report tool consumes only recorded experiment files.  It does not launch a
server, re-query a predictor, or fill a missing value with zero.  A missing
per-model artifact is treated as an incomplete experiment unless explicitly
allowed for an interim progress report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = (
    "Qwen3-0.6B",
    "Qwen2.5-3B-Instruct",
    "llama-2-7b",
    "Mistral-7B-Instruct-v0.2",
    "Qwen2.5-7B-Instruct",
    "Qwen3-8B",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-14B-Instruct-1M",
    "Qwen3-14B",
    "Meta-Llama-3.1-70B-Instruct-AWQ",
    "Qwen3-30B-A3B",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-32B",
)
METHODS = ("ml", "aic_silicon")
METHOD_LABELS = {"ml": "ML (HGB)", "aic_silicon": "AIC Silicon"}
COLORS = {"ml": "#3F78A8", "aic_silicon": "#C9783A"}


@dataclass(frozen=True)
class BatchCounts:
    requests: int
    total: int
    prefill: int
    decode: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    args.results_root = args.results_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.markdown = args.markdown.expanduser().resolve()
    args.models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    if not args.results_root.is_dir():
        parser.error(f"results root does not exist: {args.results_root}")
    if not args.models:
        parser.error("--models must name at least one model")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite analysis directory: {args.out_dir}")
    if args.markdown.exists():
        parser.error(f"refusing to overwrite report: {args.markdown}")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def float_or_blank(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def parse_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def count_collection(collection_root: Path) -> BatchCounts:
    if not (collection_root / "BASELINE_COLLECTION_COMPLETE").is_file():
        raise FileNotFoundError(collection_root / "BASELINE_COLLECTION_COMPLETE")
    manifest = collection_root / "case_manifest.tsv"
    requests = 0
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                requests += int(row["prompts"])
    else:
        for trace in collection_root.rglob("autobench_trace.jsonl"):
            requests += sum(1 for line in trace.read_text(encoding="utf-8").splitlines() if line.strip())
    total = prefill = decode = 0
    for schedule in collection_root.rglob("TP0*.schedule_batch.jsonl"):
        with schedule.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                value = json.loads(line)
                mode = value.get("forward_mode")
                if mode not in (1, 2) or not value.get("request_infos"):
                    continue
                total += 1
                if mode == 1:
                    prefill += 1
                else:
                    decode += 1
    if total == 0:
        raise RuntimeError(f"no usable ScheduleBatch rows below {collection_root}")
    return BatchCounts(requests=requests, total=total, prefill=prefill, decode=decode)


def count_repeats(root: Path) -> BatchCounts:
    aggregate = BatchCounts(0, 0, 0, 0)
    repeats = sorted(path for path in root.glob("repeat-*") if path.is_dir())
    if not repeats:
        raise FileNotFoundError(f"no repeat directories under {root}")
    for repeat in repeats:
        count = count_collection(repeat)
        aggregate = BatchCounts(
            requests=aggregate.requests + count.requests,
            total=aggregate.total + count.total,
            prefill=aggregate.prefill + count.prefill,
            decode=aggregate.decode + count.decode,
        )
    return aggregate


def teacher_forced_metrics(path: Path) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for row in read_csv(path):
        if row.get("group") == "all" and row.get("value") == "all":
            key = "all"
        elif row.get("group") == "forward_mode" and row.get("value") == "1":
            key = "prefill"
        elif row.get("group") == "forward_mode" and row.get("value") == "2":
            key = "decode"
        else:
            continue
        result[key] = {
            "n": int(row["n"]),
            "mean_actual_ms": float_or_blank(row.get("mean_actual_ms")),
            "mean_predicted_ms": float_or_blank(row.get("mean_predicted_ms")),
            "mae_ms": float_or_blank(row.get("mae_ms")),
            "mean_signed_error_pct": float_or_blank(row.get("mean_signed_error_pct")),
            "mape_pct": float_or_blank(row.get("mape_pct")),
            "p95_ape_pct": float_or_blank(row.get("p95_ape_pct")),
        }
    if set(result) != {"all", "prefill", "decode"}:
        raise RuntimeError(f"incomplete teacher-forced summary: {path}")
    return result


def e2e_metrics(path: Path) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for row in read_csv(path):
        if row.get("group") != "metric":
            continue
        metric = row.get("value")
        if metric not in {
            "mean_e2e_latency_ms",
            "mean_tpot_ms",
            "mean_ttft_ms",
            "total_throughput",
        }:
            continue
        result[metric] = {
            "n": int(row["n"]),
            "real_mean": float_or_blank(row.get("real_mean")),
            "predicted_mean": float_or_blank(row.get("predicted_mean")),
            "mape_pct": float_or_blank(row.get("mape_pct")),
            "mean_signed_error_pct": float_or_blank(row.get("mean_signed_error_pct")),
        }
    required = {
        "mean_e2e_latency_ms",
        "mean_tpot_ms",
        "mean_ttft_ms",
        "total_throughput",
    }
    if set(result) != required:
        raise RuntimeError(f"incomplete E2E summary: {path}")
    return result


def blank_metrics() -> dict[str, dict[str, float | int | None]]:
    return {}


def distribution_metrics(path: Path) -> dict[str, float | int | None]:
    for row in read_csv(path):
        if row.get("phase") != "all":
            continue
        return {
            "train_rows": int(row["train_rows"]),
            "test_rows": int(row["test_rows"]),
            "exact_feature_match_pct": float_or_blank(row.get("exact_feature_match_pct")),
            "test_outside_train_feature_range_pct": float_or_blank(
                row.get("test_outside_train_feature_range_pct")
            ),
        }
    raise RuntimeError(f"missing all-phase distribution row: {path}")


def training_validation_metrics(path: Path) -> dict[str, float | int | None]:
    for row in read_csv(path):
        if row.get("loss") != "p50" or row.get("split") != "case_stratified":
            continue
        return {
            "train_rows": int(row["train_rows"]),
            "internal_test_rows": int(row["test_rows"]),
            "mape_pct": float_or_blank(row.get("MAPE_pct")),
            "mae_ms": float_or_blank(row.get("MAE_ms")),
            "r2": float_or_blank(row.get("R2")),
        }
    raise RuntimeError(f"missing p50 case-stratified validation row: {path}")


def interpolation_support_metrics(path: Path) -> dict[str, Any]:
    """Read and validate the shared no-holdout AIC attention support sweep."""
    manifest = read_json(path)
    expected = manifest.get("expected")
    collected = manifest.get("new")
    reused = manifest.get("reused")
    if not all(isinstance(value, dict) for value in (expected, collected, reused)):
        raise RuntimeError(f"incomplete interpolation support manifest: {path}")
    for phase in ("context", "generation"):
        target = int(expected.get(phase, -1))
        actual = int(collected.get(phase, -1)) + int(reused.get(phase, -1))
        if target <= 0 or actual != target:
            raise RuntimeError(
                f"interpolation support coverage mismatch for {phase}: expected={target} actual={actual}"
            )
    if int(expected.get("total", -1)) != int(expected["context"]) + int(expected["generation"]):
        raise RuntimeError(f"interpolation support total mismatch: {path}")
    return manifest


def interpolation_support_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_slice = manifest.get("per_slice")
    if not isinstance(per_slice, dict):
        return rows
    for slice_name, values in sorted(per_slice.items()):
        if not isinstance(values, dict):
            continue
        new = values.get("new", {})
        reused = values.get("reused", {})
        if not isinstance(new, dict) or not isinstance(reused, dict):
            continue
        for phase in ("context", "generation"):
            rows.append(
                {
                    "slice": slice_name,
                    "phase": phase,
                    "new_points": int(new.get(phase, 0)),
                    "reused_points": int(reused.get(phase, 0)),
                    "measured_or_reused_total": int(new.get(phase, 0)) + int(reused.get(phase, 0)),
                }
            )
    return rows


def regular_context_scoring_database_metrics(path: Path, support_manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("purpose") != "AIC Silicon scoring database with regularized context-attention slices":
        raise RuntimeError(f"unexpected AIC scoring database manifest: {path}")
    expected_context = int(support_manifest["expected"]["context"])
    if int(manifest.get("regular_context_keys", -1)) != expected_context:
        raise RuntimeError(
            "regular-context scoring DB does not cover the complete support grid: "
            f"expected={expected_context} actual={manifest.get('regular_context_keys')}"
        )
    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, dict) or int(row_counts.get("retained_regular_context_rows", -1)) != expected_context:
        raise RuntimeError(f"regular-context scoring DB row counts are incomplete: {path}")
    return manifest


def model_record(root: Path, model: str, allow_incomplete: bool) -> dict[str, Any]:
    model_root = root / "models" / model
    try:
        train = count_repeats(model_root / "calibration" / "real")
        holdout = count_repeats(model_root / "holdout" / "real")
        model_settings = parse_metadata(model_root / "model_settings.txt")
        aic_manifest = read_json(model_root / "aic_calibration" / "aic_trace_collection_manifest.json")
        aic_plan = read_json(model_root / "aic_calibration" / "training_trace_aic_plan.json")
        distribution = distribution_metrics(model_root / "holdout" / "distribution_audit.csv")
        training_validation = training_validation_metrics(
            model_root / "calibration" / "training" / "eval" / "split_summary.csv"
        )
        teacher = {
            "ml": teacher_forced_metrics(model_root / "holdout" / "ml_teacher_forced" / "summary.csv"),
            "aic_silicon": teacher_forced_metrics(
                model_root / "holdout" / "aic_teacher_forced" / "summary.csv"
            ),
        }
        e2e = {
            "ml": e2e_metrics(model_root / "holdout" / "ml_e2e_comparison" / "summary.csv"),
            "aic_silicon": e2e_metrics(
                model_root / "holdout" / "aic_e2e_comparison" / "summary.csv"
            ),
        }
        return {
            "model": model,
            "complete": True,
            "train": train,
            "holdout": holdout,
            "settings": model_settings,
            "aic_manifest": aic_manifest,
            "aic_plan": aic_plan,
            "distribution": distribution,
            "training_validation": training_validation,
            "teacher": teacher,
            "e2e": e2e,
            "missing": "",
        }
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        if not allow_incomplete:
            raise RuntimeError(f"{model} is incomplete: {exc}") from exc
        return {
            "model": model,
            "complete": False,
            "train": BatchCounts(0, 0, 0, 0),
            "holdout": BatchCounts(0, 0, 0, 0),
            "settings": {},
            "aic_manifest": {},
            "aic_plan": {},
            "distribution": {},
            "training_validation": {},
            "teacher": {method: blank_metrics() for method in METHODS},
            "e2e": {method: blank_metrics() for method in METHODS},
            "missing": str(exc),
        }


def metric_value(metrics: dict[str, Any], phase: str, field: str) -> float | None:
    value = metrics.get(phase, {}).get(field)
    return float(value) if isinstance(value, (float, int)) else None


def create_rows(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    model_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for record in records:
        aic = record["aic_manifest"]
        attention = aic.get("attention", {}) if isinstance(aic, dict) else {}
        gemm = aic.get("gemm", {}) if isinstance(aic, dict) else {}
        compatibility = aic.get("compatibility", {}) if isinstance(aic, dict) else {}
        model_rows.append(
            {
                "model": record["model"],
                "complete": record["complete"],
                "train_requests": record["train"].requests,
                "holdout_requests": record["holdout"].requests,
                "train_scheduler_batches": record["train"].total,
                "holdout_scheduler_batches": record["holdout"].total,
                "train_prefill_batches": record["train"].prefill,
                "train_decode_batches": record["train"].decode,
                "holdout_prefill_batches": record["holdout"].prefill,
                "holdout_decode_batches": record["holdout"].decode,
                "aic_context_targets": attention.get("requested", {}).get("context", ""),
                "aic_generation_targets": attention.get("requested", {}).get("generation", ""),
                "aic_attention_new": sum(attention.get("new", {}).values()) if isinstance(attention.get("new"), dict) else "",
                "aic_attention_reused": sum(attention.get("reused", {}).values()) if isinstance(attention.get("reused"), dict) else "",
                "aic_gemm_new": gemm.get("new", ""),
                "aic_gemm_reused": gemm.get("reused", ""),
                "awq_bf16_proxy": compatibility.get("awq_bf16_proxy", ""),
                "dual_chunk_attention_triton_proxy": compatibility.get("dual_chunk_attention_triton_proxy", ""),
                "moe_trace_targeted_expert_profile_not_collected": compatibility.get("moe_trace_targeted_expert_profile_not_collected", ""),
                "ml_internal_validation_mape_pct": record["training_validation"].get("mape_pct", ""),
                "ml_internal_validation_mae_ms": record["training_validation"].get("mae_ms", ""),
                "missing": record["missing"],
            }
        )
        if not record["complete"]:
            continue
        distribution_rows.append({"model": record["model"], **record["distribution"]})
        for method in METHODS:
            for phase in ("all", "prefill", "decode"):
                metrics = record["teacher"][method][phase]
                phase_rows.append(
                    {
                        "model": record["model"],
                        "method": method,
                        "phase": phase,
                        **metrics,
                    }
                )
            for metric, values in record["e2e"][method].items():
                e2e_rows.append({"model": record["model"], "method": method, "metric": metric, **values})
    return model_rows, phase_rows, e2e_rows, distribution_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def values_for(records: list[dict[str, Any]], method: str, phase: str, field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        if not record["complete"]:
            values.append(float("nan"))
            continue
        value = metric_value(record["teacher"][method], phase, field)
        values.append(float("nan") if value is None else value)
    return values


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_data_volume(records: list[dict[str, Any]], figures: Path) -> None:
    names = [record["model"] for record in records]
    positions = np.arange(len(names))
    train = [record["train"].total if record["complete"] else np.nan for record in records]
    holdout = [record["holdout"].total if record["complete"] else np.nan for record in records]
    plt.figure(figsize=(14, 5))
    plt.bar(positions - 0.18, train, 0.36, label="training ScheduleBatch rows", color="#3F78A8")
    plt.bar(positions + 0.18, holdout, 0.36, label="holdout ScheduleBatch rows", color="#C9783A")
    plt.xticks(positions, names, rotation=35, ha="right")
    plt.ylabel("device-synchronized batch samples")
    plt.title("Recorded scheduler latency samples (same request budget per model)")
    plt.legend()
    save_figure(figures / "scheduler_batch_samples.png")


def plot_teacher_mape(records: list[dict[str, Any]], figures: Path, phase: str, filename: str) -> None:
    names = [record["model"] for record in records]
    positions = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(14, 5))
    for offset, method in ((-width / 2, "ml"), (width / 2, "aic_silicon")):
        plt.bar(
            positions + offset,
            values_for(records, method, phase, "mape_pct"),
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    plt.xticks(positions, names, rotation=35, ha="right")
    plt.ylabel("MAPE (%)")
    plt.title(f"Teacher-forced batch latency error: {phase}")
    plt.legend()
    save_figure(figures / filename)


def e2e_value(records: list[dict[str, Any]], method: str, metric: str, field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record["e2e"].get(method, {}).get(metric, {}).get(field)
        values.append(float(value) if isinstance(value, (int, float)) else float("nan"))
    return values


def plot_e2e_mape(records: list[dict[str, Any]], figures: Path) -> None:
    names = [record["model"] for record in records]
    metrics = ("mean_e2e_latency_ms", "mean_tpot_ms", "mean_ttft_ms")
    titles = ("E2E", "TPOT", "TTFT")
    filenames = ("end_to_end_mape.png", "tpot_mape.png", "ttft_mape.png")
    positions = np.arange(len(names))
    for metric, title, filename in zip(metrics, titles, filenames):
        figure, axis = plt.subplots(figsize=(14, 5))
        axis.bar(positions - 0.18, e2e_value(records, "ml", metric, "mape_pct"), 0.36, label="ML", color=COLORS["ml"])
        axis.bar(positions + 0.18, e2e_value(records, "aic_silicon", metric, "mape_pct"), 0.36, label="AIC Silicon", color=COLORS["aic_silicon"])
        axis.set_title(title)
        axis.set_ylabel("MAPE (%)")
        axis.set_xticks(positions)
        axis.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
        axis.legend()
        save_figure(figures / filename)


def plot_e2e_actual_predicted(records: list[dict[str, Any]], figures: Path) -> None:
    plt.figure(figsize=(6.5, 6))
    maxima: list[float] = []
    for method in METHODS:
        actual = e2e_value(records, method, "mean_e2e_latency_ms", "real_mean")
        predicted = e2e_value(records, method, "mean_e2e_latency_ms", "predicted_mean")
        pairs = [(left, right) for left, right in zip(actual, predicted) if math.isfinite(left) and math.isfinite(right)]
        if not pairs:
            continue
        left, right = zip(*pairs)
        maxima.extend([*left, *right])
        plt.scatter(left, right, label=METHOD_LABELS[method], color=COLORS[method], s=42)
    if maxima:
        upper = max(maxima) * 1.08
        plt.plot([0, upper], [0, upper], "--", color="#555555", linewidth=1)
        plt.xlim(0, upper)
        plt.ylim(0, upper)
    plt.xlabel("real mean E2E latency (ms)")
    plt.ylabel("predicted mean E2E latency (ms)")
    plt.title("End-to-end mean latency: real versus prediction")
    plt.legend()
    save_figure(figures / "end_to_end_actual_vs_predicted.png")


def plot_aic_points(records: list[dict[str, Any]], figures: Path) -> None:
    names = [record["model"] for record in records]
    positions = np.arange(len(names))
    attention = []
    gemm = []
    for record in records:
        manifest = record["aic_manifest"]
        attention.append(
            sum(manifest.get("attention", {}).get("new", {}).values())
            if record["complete"]
            else np.nan
        )
        gemm.append(manifest.get("gemm", {}).get("new", np.nan) if record["complete"] else np.nan)
    plt.figure(figsize=(14, 5))
    plt.bar(positions - 0.18, attention, 0.36, label="new attention points", color="#668F80")
    plt.bar(positions + 0.18, gemm, 0.36, label="new BF16 GEMM points", color="#8C6BB1")
    plt.xticks(positions, names, rotation=35, ha="right")
    plt.ylabel("AIC measured operator points")
    plt.title("Trace-targeted AIC Silicon calibration coverage")
    plt.legend()
    save_figure(figures / "aic_trace_targeted_points.png")


def plot_aic_interpolation_support(manifest: dict[str, Any], figures: Path) -> None:
    expected = manifest["expected"]
    collected = manifest["new"]
    reused = manifest["reused"]
    phases = ("context", "generation")
    positions = np.arange(len(phases))
    new = [int(collected[phase]) for phase in phases]
    old = [int(reused[phase]) for phase in phases]
    total = [int(expected[phase]) for phase in phases]
    plt.figure(figsize=(7.2, 4.5))
    plt.bar(positions, new, label="new direct measurements", color="#668F80")
    plt.bar(positions, old, bottom=new, label="existing direct measurements reused", color="#B6C9B7")
    for position, value in zip(positions, total):
        plt.text(position, value, str(value), ha="center", va="bottom", fontsize=9)
    plt.xticks(positions, ["prefill context", "decode generation"])
    plt.ylabel("AIC attention operator points")
    plt.title("Regular Silicon interpolation-support grid")
    plt.legend()
    save_figure(figures / "aic_regular_interpolation_support.png")


def plot_distribution_coverage(records: list[dict[str, Any]], figures: Path) -> None:
    names = [record["model"] for record in records]
    positions = np.arange(len(names))
    exact = [
        record["distribution"].get("exact_feature_match_pct", np.nan)
        if record["complete"]
        else np.nan
        for record in records
    ]
    outside = [
        record["distribution"].get("test_outside_train_feature_range_pct", np.nan)
        if record["complete"]
        else np.nan
        for record in records
    ]
    plt.figure(figsize=(14, 5))
    plt.bar(positions - 0.18, exact, 0.36, label="exact 18-feature match", color="#3F78A8")
    plt.bar(positions + 0.18, outside, 0.36, label="holdout outside train feature range", color="#C9783A")
    plt.xticks(positions, names, rotation=35, ha="right")
    plt.ylabel("holdout SchedulerBatch rows (%)")
    plt.title("ML training/holdout feature-support check")
    plt.legend()
    save_figure(figures / "ml_feature_support.png")


def plot_teacher_actual_predicted(records: list[dict[str, Any]], figures: Path) -> None:
    names = [record["model"] for record in records]
    positions = np.arange(len(names))
    actual = [
        metric_value(record["teacher"]["ml"], "all", "mean_actual_ms")
        if record["complete"]
        else np.nan
        for record in records
    ]
    ml = values_for(records, "ml", "all", "mean_predicted_ms")
    aic = values_for(records, "aic_silicon", "all", "mean_predicted_ms")
    plt.figure(figsize=(14, 5))
    plt.bar(positions - 0.25, ml, 0.25, label="ML predicted", color=COLORS["ml"])
    plt.bar(positions, aic, 0.25, label="AIC Silicon predicted", color=COLORS["aic_silicon"])
    plt.bar(positions + 0.25, actual, 0.25, label="real", color="#555555")
    plt.xticks(positions, names, rotation=35, ha="right")
    plt.ylabel("mean device-synchronized batch latency (ms)")
    plt.title("Teacher-forced batch latency: actual and predicted")
    plt.legend()
    save_figure(figures / "teacher_forced_actual_vs_predicted.png")


def plot_teacher_signed_error(records: list[dict[str, Any]], figures: Path) -> None:
    names = [record["model"] for record in records]
    positions = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(14, 5))
    for offset, method in ((-width / 2, "ml"), (width / 2, "aic_silicon")):
        values = [
            metric_value(record["teacher"].get(method, {}), "all", "mean_signed_error_pct")
            if record["complete"]
            else np.nan
            for record in records
        ]
        plt.bar(positions + offset, values, width, label=METHOD_LABELS[method], color=COLORS[method])
    plt.axhline(0.0, color="#555555", linewidth=1)
    plt.xticks(positions, names, rotation=35, ha="right")
    plt.ylabel("mean per-batch signed relative error (%)")
    plt.title("Teacher-forced mean signed relative error")
    plt.legend()
    save_figure(figures / "teacher_forced_signed_error.png")


def fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def table(lines: list[list[str]]) -> list[str]:
    if not lines:
        return []
    divider = ["---"] * len(lines[0])
    return [
        "| " + " | ".join(row) + " |"
        for row in [lines[0], divider, *lines[1:]]
    ]


def build_markdown(
    root: Path,
    records: list[dict[str, Any]],
    metadata: dict[str, str],
    aic_metadata: dict[str, str],
    support_manifest: dict[str, Any],
    scoring_database_manifest: dict[str, Any],
    relative_figures: Path,
) -> str:
    lines = [
        "# 全模型预测方法对比实验",
        "",
        "## 实验口径",
        "",
        f"本轮对 {len(records)} 个本地 checkpoint 分别进行真实 SGLang 采集、ML 训练与 AIConfigurator Silicon 预测。每个模型训练请求 {metadata.get('training_requests_total', '未记录')} 个，独立测试请求 {metadata.get('heldout_requests_total', '未记录')} 个；训练与测试 seed 分开。训练负载：`{metadata.get('training_case', '未记录')}`；测试负载：`{metadata.get('heldout_case', '未记录')}`。真实服务每轮加载完整 checkpoint，记录同步墙钟 scheduler iteration 时延。",
        "",
        "ML 仅使用训练 trace 的 ScheduleBatch 时延拟合 HGB 模型。AIC 的宽覆盖表和 trace-targeted 算子表也只使用训练 trace 的形状；留出 trace 的真实时延不参与 ML 拟合或 AIC 选点。批级比较采用 teacher-forced 方式，端到端比较则使用同一条留出原始请求 trace 分别运行 CPU 逻辑模拟器。",
        "",
        "本轮比较的预测方法是 ML（HGB）和 AIC Silicon。此前的“退化后端”记录对应另一套真实 kernel 路径，不能与本轮同一兼容 kernel 的基线构成受控比较，因此不作为第三种方法并入本表。",
        "",
        "每条不同 seed 的留出 trace 运行一次预测；真实 GPU 对每类负载保留五轮。每负载 n=5；多负载汇总 n 为实际配对数，不能混称为同一负载的五次结果。多负载汇总表为描述性总览，同负载与跨负载差异须查看分负载结果。",
        "",
        "## 版本与运行设置",
        "",
        f"- SGLang 源码 revision：`{metadata.get('source_revision', '未记录')}`",
        f"- Python：`{metadata.get('python', '未记录')}`",
        f"- 训练请求：{metadata.get('training_requests_total', '未记录')}；留出请求：{metadata.get('heldout_requests_total', '未记录')}。",
        f"- 训练 seed：`{metadata.get('training_seeds', '未记录')}`；留出 seed：`{metadata.get('heldout_seeds', '未记录')}`。",
        f"- 调度设置：{metadata.get('scheduler_settings', '未记录')}。",
        f"- kernel 设置：{metadata.get('kernel_policy', '未记录')}；阶段目录：`{metadata.get('kernel_compat_stage', '未记录')}`。",
        f"- AIC：AIConfigurator 0.10，backend schema `{aic_metadata.get('collector_backend_version', '未记录')}`，源码 revision `{aic_metadata.get('aic_source_revision', '未记录')}`。",
        f"- AIC 算子采集：{aic_metadata.get('operator_timing_policy', '未记录')}。",
        f"- AIC 规则插值支撑：{aic_metadata.get('interpolation_support_policy', '未记录')}。",
        f"- AIC 评分表处理：{aic_metadata.get('scoring_database_policy', '未记录')}。",
        "- AIC 的 backend schema 是其性能表查询命名空间；真实 SGLang 服务的身份以上述源码 revision、每条 trace 的 package_versions.txt 和兼容 kernel 路径为准，二者不混写为同一个版本号。",
        "",
        "## 数据量与 AIC 采样覆盖",
        "",
    ]
    support_grid = support_manifest["grid"]
    support_expected = support_manifest["expected"]
    support_new = support_manifest["new"]
    support_reused = support_manifest["reused"]
    support_table = [
        ["项目", "数值"],
        ["规则 batch 网格", "1–{}（{} 个值）".format(max(support_grid["batch_sizes"]), len(support_grid["batch_sizes"]))],
        ["prefill 长度网格", "{} 个值，最大 {}".format(len(support_grid["context_input_lengths"]), max(support_grid["context_input_lengths"]))],
        ["decode 长度网格", "{} 个值，最大 {}".format(len(support_grid["generation_input_lengths"]), max(support_grid["generation_input_lengths"]))],
        ["prefill 支撑点", "{}（新测 {}，复用 {}）".format(support_expected["context"], support_new["context"], support_reused["context"])],
        ["decode 支撑点", "{}（新测 {}，复用 {}）".format(support_expected["generation"], support_new["generation"], support_reused["generation"])],
        ["规则网格总点", str(support_expected["total"])],
    ]
    lines.extend(table(support_table))
    scoring_counts = scoring_database_manifest["row_counts"]
    scoring_table = [
        ["评分库 context 表处理", "行数"],
        ["原始 context 表", str(scoring_counts["source_rows"])],
        ["评分库保留的完整规则 context 行", str(scoring_counts["retained_regular_context_rows"])],
        ["保留的非本轮模型 context 行", str(scoring_counts["retained_other_rows"])],
        ["从评分 context 切片排除的散点/不兼容行", str(scoring_counts["removed_sparse_or_incompatible_rows"])],
    ]
    lines.extend([""])
    lines.extend(table(scoring_table))
    lines.extend(
        [
            "",
            "规则网格只由训练 trace 计划和固定的 `context_length=2048`、`max_running_requests=32` 生成。它用于保证 AIC Silicon 对独立留出 trace 的批大小和长度组合有完整的三维插值支撑；留出 trace 不参与该选点过程。",
            "原始 AIC 表仍保留所有 trace-targeted 行。正式评分使用独立评分库：对本轮 10 个 attention 形状，只保留完整规则 context 网格，避免 AIC 先选到一个单 batch 的相邻长度行；generation、GEMM 和 MoE 表没有改写。",
            "",
        ]
    )
    data_table = [
        [
            "模型",
            "训练请求",
            "留出请求",
            "训练 batch",
            "留出 batch",
            "留出超出训练特征范围",
            "AIC attention 目标",
            "AIC attention 新测 / 复用",
            "AIC GEMM 新测 / 复用",
        ]
    ]
    for record in records:
        manifest = record["aic_manifest"]
        attention = manifest.get("attention", {})
        attention_requested = attention.get("requested", {})
        attention_new = attention.get("new", {})
        attention_reused = attention.get("reused", {})
        gemm = manifest.get("gemm", {})
        data_table.append(
            [
                record["model"],
                str(record["train"].requests) if record["complete"] else "—",
                str(record["holdout"].requests) if record["complete"] else "—",
                str(record["train"].total) if record["complete"] else "—",
                str(record["holdout"].total) if record["complete"] else "—",
                fmt(record["distribution"].get("test_outside_train_feature_range_pct")) + "%" if record["complete"] else "—",
                str(sum(attention_requested.values())) if record["complete"] else "—",
                "{} / {}".format(sum(attention_new.values()), sum(attention_reused.values()))
                if record["complete"]
                else "—",
                "{} / {}".format(gemm.get("new", "—"), gemm.get("reused", "—"))
                if record["complete"]
                else "—",
            ]
        )
    lines.extend(table(data_table))
    lines.extend(
        [
            "",
            "同样的请求预算并不保证产生同样数量的 scheduler batch：较快或较慢的模型在饱和队列下会形成不同的动态 batch 组合。因此训练和留出 batch 数均单独列出，作为实际时延样本量。",
            "AIC 的“复用”表示该算子形状已在本轮相同 A100、相同采集后端下直接测得，因而不重复执行同一测点；它不是使用留出 trace，也不是把 ML 训练样本在模型间转移。",
            "",
            f"![scheduler batch samples]({relative_figures / 'scheduler_batch_samples.png'})",
            "",
            f"![aic points]({relative_figures / 'aic_trace_targeted_points.png'})",
            "",
            f"![aic regular support]({relative_figures / 'aic_regular_interpolation_support.png'})",
            "",
            f"![ml feature support]({relative_figures / 'ml_feature_support.png'})",
            "",
            "训练侧还保留了一个按 trace 分层的内部检查，用于确认模型拟合过程；它与后面的独立留出 trace 结果分开报告，不能替代正式测试。",
            "",
            "## ML 训练内核查（非正式留出结果）",
            "",
        ]
    )
    internal_table = [["模型", "最终 fit 输入行", "内部训练 / 检查行", "p50 HGB MAPE", "MAE ms"]]
    for record in records:
        validation = record["training_validation"]
        internal_table.append(
            [
                record["model"],
                str(record["train"].total) if record["complete"] else "—",
                f"{validation.get('train_rows', '—')} / {validation.get('internal_test_rows', '—')}",
                fmt(validation.get("mape_pct")) + "%",
                fmt(validation.get("mae_ms")),
            ]
        )
    lines.extend(table(internal_table))
    lines.extend(
        [
            "",
            "每条 real trace 的完整启动命令、SGLang/sgl-kernel 包版本和原始 server log 均保存在对应的 `calibration/real/repeat-*/` 或 `holdout/real/repeat-*/` 目录。下表给出模型间不同的运行参数；其余 scheduler 设置统一采用本报告前述口径。",
            "",
            "## 模型级运行参数",
            "",
        ]
    )
    settings_table = [["模型", "显存比例", "attention 后端", "page size", "FA2 兼容层", "运行时 shim"]]
    for record in records:
        settings = record["settings"]
        settings_table.append(
            [
                record["model"],
                settings.get("mem_fraction_static", "—"),
                settings.get("attention_backend", "—"),
                settings.get("page_size") or "默认",
                "是" if settings.get("fa2_compat_stage") else "否",
                "是" if settings.get("runtime_shim") else "否",
            ]
        )
    lines.extend(table(settings_table))
    lines.extend(
        [
            "",
            "## 批级时延结果（teacher-forced）",
            "",
        ]
    )
    teacher_table = [["模型", "ML 全部 MAPE", "AIC 全部 MAPE", "ML prefill", "AIC prefill", "ML decode", "AIC decode"]]
    for record in records:
        row = [record["model"]]
        for phase in ("all", "prefill", "decode"):
            for method in METHODS:
                row.append(fmt(metric_value(record["teacher"].get(method, {}), phase, "mape_pct")) + "%")
        teacher_table.append(row)
    lines.extend(table(teacher_table))
    lines.extend(["", "批级平均实测值与预测值（全部 ScheduleBatch）：", ""])
    teacher_value_table = [["模型", "实测 ms", "ML 预测 ms", "ML 平均相对偏差", "AIC 预测 ms", "AIC 平均相对偏差"]]
    for record in records:
        ml = record["teacher"].get("ml", {}).get("all", {})
        aic = record["teacher"].get("aic_silicon", {}).get("all", {})
        teacher_value_table.append(
            [
                record["model"],
                fmt(ml.get("mean_actual_ms")),
                fmt(ml.get("mean_predicted_ms")),
                fmt(ml.get("mean_signed_error_pct")) + "%",
                fmt(aic.get("mean_predicted_ms")),
                fmt(aic.get("mean_signed_error_pct")) + "%",
            ]
        )
    lines.extend(table(teacher_value_table))
    lines.extend(["", "平均相对偏差为逐 batch 的 (预测−实测)/实测 的平均值，正值表示平均相对高估；它不是 (预测均值−实测均值)/实测均值，两者可能符号不同。MAPE 对逐 batch 相对误差取绝对值后平均。", ""])
    lines.extend(
        [
            "",
            f"![teacher forced mape]({relative_figures / 'teacher_forced_all_mape.png'})",
            "",
            f"![teacher forced prefill mape]({relative_figures / 'teacher_forced_prefill_mape.png'})",
            "",
            f"![teacher forced decode mape]({relative_figures / 'teacher_forced_decode_mape.png'})",
            "",
            f"![teacher forced actual predicted]({relative_figures / 'teacher_forced_actual_vs_predicted.png'})",
            "",
            f"![teacher forced signed error]({relative_figures / 'teacher_forced_signed_error.png'})",
            "",
            "批级实测为同步墙钟 run_batch 时延，包含函数内 CPU 发射、采样/拷贝和同步等待，但不包含函数外全部调度及网络时延。AIC 为 GPU 算子组合估计，ML 拟合上述同步墙钟标签；两者计时口径并非天然相同。",
            "",
            "## 端到端留出 trace 结果",
            "",
        ]
    )
    e2e_table = [["模型", "实测 E2E ms", "ML 预测 ms / MAPE", "AIC 预测 ms / MAPE", "ML TTFT MAPE", "AIC TTFT MAPE"]]
    for record in records:
        ml_e2e = record["e2e"].get("ml", {}).get("mean_e2e_latency_ms", {})
        aic_e2e = record["e2e"].get("aic_silicon", {}).get("mean_e2e_latency_ms", {})
        ml_ttft = record["e2e"].get("ml", {}).get("mean_ttft_ms", {})
        aic_ttft = record["e2e"].get("aic_silicon", {}).get("mean_ttft_ms", {})
        e2e_table.append(
            [
                record["model"],
                fmt(ml_e2e.get("real_mean")),
                f"{fmt(ml_e2e.get('predicted_mean'))} / {fmt(ml_e2e.get('mape_pct'))}%",
                f"{fmt(aic_e2e.get('predicted_mean'))} / {fmt(aic_e2e.get('mape_pct'))}%",
                fmt(ml_ttft.get("mape_pct")) + "%",
                fmt(aic_ttft.get("mape_pct")) + "%",
            ]
        )
    lines.extend(table(e2e_table))
    lines.extend(
        [
            "",
            f"![end to end mape]({relative_figures / 'end_to_end_mape.png'})",
            "",
            f"![TTFT mape]({relative_figures / 'ttft_mape.png'})",
            "",
            f"![TPOT mape]({relative_figures / 'tpot_mape.png'})",
            "",
            f"![end to end actual predicted]({relative_figures / 'end_to_end_actual_vs_predicted.png'})",
            "",
            "端到端误差会包含单步预测偏差在后续调度中的累积影响。表格同时保留实测均值和预测均值，避免只看误差百分比而忽略绝对时延规模。",
            "",
            "## 汇总比较",
            "",
        ]
    )
    all_ml_mape = [
        metric_value(record["teacher"]["ml"], "all", "mape_pct")
        for record in records
        if record["complete"]
    ]
    all_aic_mape = [
        metric_value(record["teacher"]["aic_silicon"], "all", "mape_pct")
        for record in records
        if record["complete"]
    ]
    ml_values = [value for value in all_ml_mape if value is not None and math.isfinite(value)]
    aic_values = [value for value in all_aic_mape if value is not None and math.isfinite(value)]
    paired = [
        (ml, aic)
        for ml, aic in zip(all_ml_mape, all_aic_mape)
        if ml is not None and aic is not None and math.isfinite(ml) and math.isfinite(aic)
    ]
    ml_wins = sum(ml < aic for ml, aic in paired)
    aic_wins = sum(aic < ml for ml, aic in paired)
    e2e_ml = [
        e2e_value(records, "ml", "mean_e2e_latency_ms", "mape_pct")[index]
        for index, record in enumerate(records)
        if record["complete"]
    ]
    e2e_aic = [
        e2e_value(records, "aic_silicon", "mean_e2e_latency_ms", "mape_pct")[index]
        for index, record in enumerate(records)
        if record["complete"]
    ]
    e2e_paired = [
        (ml, aic)
        for ml, aic in zip(e2e_ml, e2e_aic)
        if ml is not None and aic is not None and math.isfinite(ml) and math.isfinite(aic)
    ]
    e2e_ml_wins = sum(ml < aic for ml, aic in e2e_paired)
    e2e_aic_wins = sum(aic < ml for ml, aic in e2e_paired)
    lines.extend(
        [
            "- 批级全部行 MAPE 按模型等权平均：ML {}%，AIC Silicon {}%；中位数分别为 {}% 和 {}%。".format(
                fmt(float(np.mean(ml_values))),
                fmt(float(np.mean(aic_values))),
                fmt(float(np.median(ml_values))),
                fmt(float(np.median(aic_values))),
            ),
            "- 批级 MAPE 较小的模型数：ML {} 个，AIC Silicon {} 个；其余为并列。".format(
                ml_wins,
                aic_wins,
            ),
            "- 端到端平均 E2E MAPE 按模型等权平均：ML {}%，AIC Silicon {}%。".format(
                fmt(float(np.nanmean(e2e_ml))),
                fmt(float(np.nanmean(e2e_aic))),
            ),
            "- 端到端 E2E MAPE 较小的模型数：ML {} 个，AIC Silicon {} 个；其余为并列。".format(
                e2e_ml_wins,
                e2e_aic_wins,
            ),
            "- 上述统计仅覆盖本报告实际采集的工作负载、A100 PCIe 和当前兼容 kernel，不外推至未测试配置。",
            "",
            "## 兼容性说明",
            "",
        ]
    )
    compatibility_table = [["模型", "AWQ BF16 代理", "dual-chunk Triton 代理", "MoE expert trace-targeted 表缺失"]]
    for record in records:
        flags = record["aic_manifest"].get("compatibility", {})
        compatibility_table.append(
            [
                record["model"],
                fmt(flags.get("awq_bf16_proxy"), 0),
                fmt(flags.get("dual_chunk_attention_triton_proxy"), 0),
                fmt(flags.get("moe_trace_targeted_expert_profile_not_collected"), 0),
            ]
        )
    lines.extend(table(compatibility_table))
    lines.extend(
        [
            "",
            "AWQ checkpoint 保留原始权重参与真实基线，但 AIC 0.10 在本机 A100/SM80 没有原生 dense-AWQ GEMM 表，因此该行使用 BF16 算子代理。1M checkpoint 的真实短上下文路径使用 dual-chunk FlashAttention；AIC trace-targeted attention 表为 Triton 代理。Qwen3 MoE 两行保留 AIC 的通用 MoE 表，当前没有按 trace 动态路由分布补采 expert 表。上述行仍纳入结果表，但不应与原生 BF16 dense 行作“仅差模型大小”的解释。",
            "",
            "## 结果位置",
            "",
            f"- 原始实验根目录：`{root}`",
            "- 模型级 real、ML、AIC 命令、日志、原始请求 trace 和 metrics 均在 `models/<模型名>/` 下。",
            "- 本报告的结构化汇总在同级 `analysis/` CSV 文件中；图位于 `analysis/figures/`。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = [model_record(args.results_root, model, args.allow_incomplete) for model in args.models]
    support_manifest = interpolation_support_metrics(
        args.results_root / "aic_calibration" / "aic_interpolation_support_manifest.json"
    )
    scoring_database_manifest = regular_context_scoring_database_metrics(
        args.results_root / "aic_calibration" / "aic_regular_context_scoring_database_manifest.json",
        support_manifest,
    )
    support_rows = interpolation_support_rows(support_manifest)
    model_rows, phase_rows, e2e_rows, distribution_rows = create_rows(records)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    figures = args.out_dir / "figures"
    figures.mkdir()
    write_csv(args.out_dir / "model_data_and_aic_coverage.csv", model_rows, model_rows[0].keys())
    write_csv(args.out_dir / "teacher_forced_metrics.csv", phase_rows, phase_rows[0].keys() if phase_rows else ["model", "method", "phase"])
    write_csv(args.out_dir / "end_to_end_metrics.csv", e2e_rows, e2e_rows[0].keys() if e2e_rows else ["model", "method", "metric"])
    write_csv(
        args.out_dir / "distribution_coverage.csv",
        distribution_rows,
        distribution_rows[0].keys()
        if distribution_rows
        else [
            "model",
            "train_rows",
            "test_rows",
            "exact_feature_match_pct",
            "test_outside_train_feature_range_pct",
        ],
    )
    write_csv(
        args.out_dir / "aic_interpolation_support_coverage.csv",
        support_rows,
        support_rows[0].keys()
        if support_rows
        else ["slice", "phase", "new_points", "reused_points", "measured_or_reused_total"],
    )
    write_csv(
        args.out_dir / "aic_regular_context_scoring_database.csv",
        [
            {
                "regular_context_keys": scoring_database_manifest["regular_context_keys"],
                **scoring_database_manifest["row_counts"],
                "source_systems": scoring_database_manifest["source_systems"],
                "output_systems": scoring_database_manifest["output_systems"],
            }
        ],
        [
            "regular_context_keys",
            "source_rows",
            "retained_rows",
            "retained_regular_context_rows",
            "retained_other_rows",
            "removed_sparse_or_incompatible_rows",
            "source_systems",
            "output_systems",
        ],
    )
    plot_data_volume(records, figures)
    plot_teacher_mape(records, figures, "all", "teacher_forced_all_mape.png")
    plot_teacher_mape(records, figures, "prefill", "teacher_forced_prefill_mape.png")
    plot_teacher_mape(records, figures, "decode", "teacher_forced_decode_mape.png")
    plot_e2e_mape(records, figures)
    plot_e2e_actual_predicted(records, figures)
    plot_aic_points(records, figures)
    plot_aic_interpolation_support(support_manifest, figures)
    plot_distribution_coverage(records, figures)
    plot_teacher_actual_predicted(records, figures)
    plot_teacher_signed_error(records, figures)
    metadata = parse_metadata(args.results_root / "run_metadata.txt")
    aic_metadata = parse_metadata(
        args.results_root / "aic_calibration" / "trace_targeted_run_metadata.txt"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    relative_figures = Path(os.path.relpath(figures, start=args.markdown.parent))
    args.markdown.write_text(
        build_markdown(
            args.results_root,
            records,
            metadata,
            aic_metadata,
            support_manifest,
            scoring_database_manifest,
            relative_figures,
        )
        + "\n",
        encoding="utf-8",
    )
    complete = sum(1 for record in records if record["complete"])
    print(f"models={len(records)} complete={complete} analysis={args.out_dir} markdown={args.markdown}")


if __name__ == "__main__":
    main()
