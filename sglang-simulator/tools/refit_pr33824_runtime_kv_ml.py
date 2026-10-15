#!/usr/bin/env python3
"""Refit saved per-model ML calibration with runtime-compatible KV features.

Reuse existing native measurements, exact train rows/order, and frozen HGB
parameters. Only the interpretation of past lengths changes. Independent
holdout data are read after fitting and never select parameters or predictors.
This is a CPU-only single-step diagnostic, not new GPU collection or E2E replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from sglang_simulator.time_predictor.ml_features import ML_FEATURE_NAMES, extract_ml_features


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty result table: {path}")
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_tool():
    path = Path(__file__).resolve().with_name("audit_pr33824_kv_feature_contract.py")
    spec = importlib.util.spec_from_file_location("runtime_kv_source_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_collection(model_root: Path, role: str, max_latency: float) -> tuple[list[dict], dict[str, str], list[str]]:
    """Keep original source identities; never create transformed raw traces."""
    if role not in ("calibration", "holdout"):
        raise ValueError(f"unknown data role: {role}")
    directory = (model_root / role / "real").resolve()
    groups = sorted(directory.glob("repeat-*"))
    if not groups:
        raise ValueError(f"no collected groups: {directory}")
    tool = audit_tool()
    rows, sources, roots = [], {}, []
    for group in groups:
        if (group / "BASELINE_COLLECTION_COMPLETE").read_text().strip() != "complete":
            raise ValueError(f"incomplete collected group: {group}")
        roots.append(str((group / "cases").resolve()))
        files = sorted((group / "cases").glob("*/TP0.schedule_batch.jsonl"))
        if not files:
            raise ValueError(f"no measured cases: {group}")
        for path in files:
            canonical = path.resolve()
            if not canonical.is_relative_to(directory):
                raise ValueError(f"measured file escapes its data role: {path}")
            if (path.parent / "status.txt").read_text().strip() != "ok":
                raise ValueError(f"incomplete collected case: {path.parent}")
            sources[str(canonical)] = digest(path)
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                mode = item["forward_mode"]
                if mode not in (1, 2):
                    continue
                requests = item["request_infos"]
                actual = float(item["iter_latency"])
                if not requests or not math.isfinite(actual) or actual <= 0:
                    raise ValueError(f"invalid measured batch: {canonical}:{number}")
                # The original trainer filters <= max_latency; original scoring
                # retains all finite positive holdout targets. Keep both policies.
                if role == "calibration" and actual > max_latency:
                    continue
                extend = [tool.length(req, "extend_input_len") for req in requests]
                prefix = [tool.length(req, "prefix_indices_len") for req in requests]
                output = [tool.length(req, "output_ids_len") for req in requests]
                runtime = [past + generated for past, generated in zip(prefix, output)]
                rows.append({"model": model_root.name, "case": path.parent.name,
                             "schedule_path": str(canonical), "line_number": number,
                             "forward_mode": mode, "batch_size": len(requests), "sum_extend": sum(extend),
                             "legacy_sum_past": sum(prefix), "runtime_sum_past": sum(runtime),
                             "actual_seconds": actual, "legacy_features": extract_ml_features(extend, prefix),
                             "runtime_features": extract_ml_features(extend, runtime)})
    if not rows:
        raise ValueError(f"no measured rows: {directory}")
    return rows, sources, roots


def original_scores(path: Path) -> dict:
    result, resolved = {}, {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["schedule_path"] not in resolved:
                resolved[row["schedule_path"]] = str(Path(row["schedule_path"]).resolve())
            key = (resolved[row["schedule_path"]], int(row["line_number"]))
            if key in result:
                raise ValueError(f"duplicate original ML score: {key}")
            result[key] = row
    return result


def predict(model, rows: list[dict], feature_key: str) -> np.ndarray:
    values = np.asarray(model.predict(np.asarray([row[feature_key] for row in rows], dtype=float)), dtype=float)
    if values.shape != (len(rows),) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("invalid ML predictions")
    return values


def refit_model(model_root: Path, out: Path) -> tuple[list[dict], dict]:
    """One predeclared refit; no access to test rows before fit() completes."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite an existing refit directory: {out}")
    bundle_path = model_root / "calibration/training" / f"{model_root.name}_hgbmono_p50.pkl"
    original_digest = digest(bundle_path)
    saved = joblib.load(bundle_path)
    if (list(saved.get("features", [])) != list(ML_FEATURE_NAMES)
            or saved.get("target") != "iter_latency_seconds"
            or saved.get("input_contract") != "[[extend_input_len, prefix_indices_len], ...]"
            or saved.get("forward_modes") != [1, 2]):
        raise ValueError("original bundle feature/target/input contract differs")
    original = saved["model"]
    if not isinstance(original, HistGradientBoostingRegressor):
        raise ValueError("expected the original per-model HGB p50 regressor")
    params = original.get_params()
    if params["loss"] != "quantile" or params["quantile"] != .5:
        raise ValueError("expected the predeclared p50 objective")
    for key, value in saved["best_params"].items():
        if not np.array_equal(params[key], value):
            raise ValueError(f"saved fitting parameter differs from frozen model: {key}")
    maximum = float(saved["max_latency_seconds"])
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("invalid original training latency filter")
    training, train_sources, roots = read_collection(model_root, "calibration", maximum)
    if set(roots) != {str(Path(path).resolve()) for path in saved["source_data_roots"]}:
        raise ValueError("saved training roots differ from this model's calibration")
    if len(training) != int(saved["train_rows"]) or len(train_sources) != int(saved["train_cases"]):
        raise ValueError("original versus runtime-input training row/case coverage differs")
    if len({(row["schedule_path"], row["line_number"]) for row in training}) != len(training):
        raise ValueError("duplicate calibration step identity")
    # Constructor parameters, row order, labels and early-stopping seed are
    # identical to the original final model; only past-derived features change.
    corrected = HistGradientBoostingRegressor(**params)
    train_features = np.asarray([row["runtime_features"] for row in training], dtype=float)
    train_target = np.asarray([row["actual_seconds"] for row in training], dtype=float)
    started = time.perf_counter()
    corrected.fit(train_features, train_target)
    fit_seconds = time.perf_counter() - started
    out.mkdir(parents=True)
    corrected_path = out / f"{model_root.name}_runtime_kv_p50.pkl"
    joblib.dump({**saved, "model": corrected, "version": f"{model_root.name}_runtime_kv_p50",
                 "input_contract": "[[extend_input_len, prefix_indices_len + output_ids_len], ...]",
                 "original_bundle_sha256": original_digest,
                 "refit_scope": "Only saved calibration rows; frozen parameters; runtime KV feature interpretation"}, corrected_path)
    # Only after fit and bundle serialization do we read independent targets.
    holdout, test_sources, _ = read_collection(model_root, "holdout", maximum)
    if set(train_sources) & set(test_sources):
        raise ValueError("calibration and holdout share raw step files")
    score_path = model_root / "holdout/ml_teacher_forced/per_batch.csv"
    score_digest = digest(score_path)
    official = original_scores(score_path)
    if len(official) != len(holdout):
        raise ValueError("original ML score versus measured holdout coverage differs")
    baseline = predict(original, holdout, "legacy_features")
    frozen = predict(original, holdout, "runtime_features")
    refitted = predict(corrected, holdout, "runtime_features")
    details = []
    for source, before, unchanged, after in zip(holdout, baseline, frozen, refitted):
        key = (source["schedule_path"], source["line_number"])
        previous = official.get(key)
        identity = (source["case"], source["forward_mode"], source["batch_size"],
                    source["sum_extend"], source["legacy_sum_past"], source["actual_seconds"])
        if previous is None or identity != (previous["case"], int(previous["forward_mode"]),
                int(previous["batch_size"]), int(previous["sum_extend"]), int(previous["sum_past"]),
                float(previous["actual_seconds"])):
            raise ValueError(f"original ML score mismatches measured step: {key}")
        if not math.isclose(float(before), float(previous["predicted_seconds"]), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("original bundle does not reproduce its official scores")
        actual = source["actual_seconds"]
        row = {key: value for key, value in source.items() if key not in ("legacy_features", "runtime_features")}
        for label, value in (("baseline_prefix", before), ("frozen_runtime", unchanged), ("refit_runtime", after)):
            row[f"{label}_predicted_seconds"] = float(value)
            row[f"{label}_ape_pct"] = abs(float(value)-actual)/actual*100
        details.append(row)
    groups = {}
    for row in details:
        for key in (("all", "all"), ("case", row["case"]), ("phase", str(row["forward_mode"])),
                    ("case_phase", f"{row['case']}:mode{row['forward_mode']}")):
            groups.setdefault(key, []).append(row)
    summaries = []
    for (kind, value), members in groups.items():
        row = {"model": model_root.name, "group": kind, "value": value, "n": len(members),
               "mean_actual_ms": sum(item["actual_seconds"] for item in members)/len(members)*1000}
        for label in ("baseline_prefix", "frozen_runtime", "refit_runtime"):
            row[f"{label}_mean_predicted_ms"] = sum(item[f"{label}_predicted_seconds"] for item in members)/len(members)*1000
            row[f"{label}_mape_pct"] = sum(item[f"{label}_ape_pct"] for item in members)/len(members)
        summaries.append(row)
    unchanged = {str(bundle_path.resolve()): original_digest, str(score_path.resolve()): score_digest,
                 **train_sources, **test_sources}
    if any(digest(Path(path)) != checksum for path, checksum in unchanged.items()):
        raise ValueError("original calibration/bundle/scores/holdout changed during refit")
    write_csv(out / "per_batch.csv", details)
    write_csv(out / "summary.csv", summaries)
    manifest = {"model": model_root.name, "training_rows": len(training), "training_cases": len(train_sources),
                "holdout_rows": len(holdout), "holdout_cases": len(test_sources),
                "constructor_parameters": params, "fit_input": "calibration only; prefix + output lengths",
                "fit_wall_seconds": fit_seconds, "original_n_iter": int(original.n_iter_),
                "refit_n_iter": int(corrected.n_iter_),
                "fit_features_float64_sha256": hashlib.sha256(train_features.tobytes()).hexdigest(),
                "fit_targets_float64_sha256": hashlib.sha256(train_target.tobytes()).hexdigest(),
                "test_read_after_fit": True, "original_predictions_reproduced": True,
                "refit_bundle_path": str(corrected_path.resolve()), "refit_bundle_sha256": digest(corrected_path),
                "source_sha256": unchanged, "scope": "Single-step CPU refit comparison; no GPU recollection, E2E replay, model/parameter selection or physical seq_lens measurement."}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summaries, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    models = args.models.split(",")
    if len(set(models)) != len(models) or any(not name or "/" in name or name in (".", "..") for name in models):
        parser.error("--models must contain unique checkpoint directory names")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("OMP_NUM_THREADS") != "1":
        parser.error("use CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 for this CPU-only diagnostic")
    root = args.results_root.resolve()
    if (root / "REAL_ML_ALL_MODELS_COMPLETE").read_text().strip() != "complete":
        raise ValueError("wait for all native collection before CPU refitting")
    if args.out_dir.exists():
        parser.error("refusing to overwrite an existing refit directory")
    contract = audit_tool().source_contract(Path(__file__).resolve().parents[1])
    args.out_dir.mkdir(parents=True)
    all_rows, manifests = [], []
    for model in models:
        rows, manifest = refit_model(root / "models" / model, args.out_dir / "models" / model)
        all_rows.extend(rows)
        manifests.append(manifest)
        print(f"{model}: refit {manifest['training_rows']} saved training rows; score {manifest['holdout_rows']} independent rows", flush=True)
    write_csv(args.out_dir / "ml_runtime_refit_comparison.csv", all_rows)
    (args.out_dir / "source_manifest.json").write_text(json.dumps({"source_contract": contract,
        "tool_sha256": digest(Path(__file__)), "models": manifests,
        "scope": "Predeclared runtime-KV correction only; original protocol artifacts preserved; no test-target tuning."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
