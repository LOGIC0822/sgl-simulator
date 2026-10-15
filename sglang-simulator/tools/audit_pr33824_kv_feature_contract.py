#!/usr/bin/env python3
"""Audit saved token-length inputs against the live predictor call contract.

The audit does not fit/predict; optional frozen comparison uses CPU predict()
only. Neither path refits, recollects, or changes source traces. The runtime
expression is an input-ABI reference, not independent GPU seq_lens evidence.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

from sglang_simulator.time_predictor.ml_features import ML_FEATURE_NAMES, extract_ml_features


def length(request: dict, key: str) -> int:
    value = request[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {key}: {value!r}")
    return value


def source_contract(sim: Path) -> dict:
    paths = {
        "runtime": "src/sglang_simulator/simulation/sglang/scheduler.py",
        "trainer": "tools/train_latency_model.py",
        "ml_scorer": "tools/score_pr33824_latency_model.py",
        "aic_scorer": "tools/score_pr33824_aic_latency_model.py",
    }
    output = {}
    for kind, relative in paths.items():
        path = sim / relative
        content = path.read_bytes()
        tree = ast.parse(content.decode())
        if kind == "runtime":
            expressions = [ast.unparse(keyword.value) for node in ast.walk(tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id == "ScheduleRequest" for keyword in node.keywords
                           if keyword.arg == "past_kv_length"]
            if expressions != ["len(req.prefix_indices) + len(req.output_ids)"] * 2:
                raise ValueError("runtime length expression changed; reassess the audit contract")
        else:
            expressions = [ast.unparse(node.value.elt) for node in ast.walk(tree)
                           if isinstance(node, ast.Assign) and isinstance(node.value, ast.ListComp)
                           and any(isinstance(target, ast.Name) and target.id in ("past", "past_lengths")
                                   for target in node.targets)]
            if len(expressions) != 1 or "prefix_indices_len" not in expressions[0] or "output_ids_len" in expressions[0]:
                raise ValueError(f"{kind} length expression changed; reassess the audit contract")
        output[kind] = {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest(),
                        "length_expressions": expressions}
    return output


def audit(root: Path, models: list[str]) -> tuple[list[dict], list[dict]]:
    summaries, manifest = [], []
    for model in models:
        for role in ("calibration", "holdout"):
            directory = root / "models" / model / role / "real"
            groups = sorted(directory.glob("repeat-*"))
            if not groups:
                raise FileNotFoundError(f"no collected groups: {directory}")
            accumulators = {}
            for group in groups:
                if (group / "BASELINE_COLLECTION_COMPLETE").read_text().strip() != "complete":
                    raise ValueError(f"incomplete collected group: {group}")
                schedules = sorted((group / "cases").glob("*/TP0.schedule_batch.jsonl"))
                if not schedules:
                    raise FileNotFoundError(f"no schedule data: {group}")
                for path in schedules:
                    if (path.parent / "status.txt").read_text().strip() != "ok":
                        raise ValueError(f"incomplete collected case: {path.parent}")
                    content = path.read_bytes()
                    manifest.append({"model": model, "role": role, "path": str(path.resolve()),
                                     "sha256": hashlib.sha256(content).hexdigest()})
                    for line_number, line in enumerate(content.splitlines(), start=1):
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        mode = item["forward_mode"]
                        if mode not in (1, 2):
                            continue
                        requests = item["request_infos"]
                        latency = float(item["iter_latency"])
                        if not requests or not math.isfinite(latency) or latency <= 0:
                            raise ValueError(f"invalid measured batch: {path}:{line_number}")
                        extend = [length(request, "extend_input_len") for request in requests]
                        prefix = [length(request, "prefix_indices_len") for request in requests]
                        generated = [length(request, "output_ids_len") for request in requests]
                        runtime = [past + count for past, count in zip(prefix, generated)]
                        key = (path.parent.name, mode)
                        values = accumulators.setdefault(key, {"rows": 0, "changed": 0, "requests": 0,
                                                                "prefix": 0, "runtime": 0, "generated": 0,
                                                                "max_generated": 0, "latency": 0.0})
                        values["rows"] += 1
                        values["changed"] += int(extract_ml_features(extend, prefix) != extract_ml_features(extend, runtime))
                        values["requests"] += len(requests)
                        values["prefix"] += sum(prefix)
                        values["runtime"] += sum(runtime)
                        values["generated"] += sum(generated)
                        values["max_generated"] = max(values["max_generated"], max(generated))
                        values["latency"] += latency
            if not accumulators:
                raise ValueError(f"no measured prefill/decode batches: {directory}")
            for (case, mode), value in sorted(accumulators.items()):
                summaries.append({"model": model, "role": role, "case": case, "forward_mode": mode,
                                  "rows": value["rows"], "changed_rows": value["changed"],
                                  "changed_18_feature_vector_pct": 100 * value["changed"] / value["rows"],
                                  "request_occurrences": value["requests"],
                                  "mean_offline_prefix_tokens_per_request": value["prefix"] / value["requests"],
                                  "mean_runtime_prefix_plus_output_tokens_per_request": value["runtime"] / value["requests"],
                                  "mean_added_output_tokens_per_request": value["generated"] / value["requests"],
                                  "max_added_output_tokens_per_request": value["max_generated"],
                                  "mean_actual_ms": value["latency"] / value["rows"] * 1000})
    return summaries, manifest


def compare_frozen_inputs(root: Path, models: list[str], out: Path) -> tuple[list[dict], list[dict]]:
    """Change only the input interpretation; never refit the saved regressor."""
    import joblib
    import numpy as np

    summaries, identities = [], []
    detail_root = out / "frozen_input_per_batch"
    detail_root.mkdir()
    for name in models:
        model_root = root / "models" / name
        bundle_path = model_root / "calibration/training" / f"{name}_hgbmono_p50.pkl"
        bundle_before = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        bundle = joblib.load(bundle_path)
        if list(bundle.get("features", [])) != list(ML_FEATURE_NAMES) or bundle.get("target") != "iter_latency_seconds":
            raise ValueError(f"frozen model input/target ABI differs: {name}")
        score_path = model_root / "holdout/ml_teacher_forced/per_batch.csv"
        score_before = hashlib.sha256(score_path.read_bytes()).hexdigest()
        with score_path.open(newline="", encoding="utf-8") as source:
            official = {}
            resolved_paths = {}
            for row in csv.DictReader(source):
                if row["schedule_path"] not in resolved_paths:
                    resolved_paths[row["schedule_path"]] = str(Path(row["schedule_path"]).resolve())
                key = (resolved_paths[row["schedule_path"]], int(row["line_number"]))
                if key in official:
                    raise ValueError(f"duplicate official score: {name}/{key}")
                official[key] = row
        legacy_features, runtime_features, steps, source_hashes = [], [], [], {}
        for path in sorted((model_root / "holdout/real").glob("repeat-*/cases/*/TP0.schedule_batch.jsonl")):
            content = path.read_bytes()
            source_hashes[path] = hashlib.sha256(content).hexdigest()
            resolved_path = str(path.resolve())
            for number, line in enumerate(content.splitlines(), start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                mode = item["forward_mode"]
                if mode not in (1, 2):
                    continue
                requests = item["request_infos"]
                actual = float(item["iter_latency"])
                if not requests or not math.isfinite(actual) or actual <= 0:
                    raise ValueError(f"invalid frozen-input measurement: {path}:{number}")
                extend = [length(request, "extend_input_len") for request in requests]
                prefix = [length(request, "prefix_indices_len") for request in requests]
                generated = [length(request, "output_ids_len") for request in requests]
                runtime = [past + count for past, count in zip(prefix, generated)]
                key = (resolved_path, number)
                previous = official.get(key)
                if previous is None or (previous["case"], int(previous["forward_mode"]),
                        int(previous["batch_size"]), int(previous["sum_extend"]),
                        int(previous["sum_past"]), float(previous["actual_seconds"])) != (
                        path.parent.name, mode, len(requests), sum(extend), sum(prefix), actual):
                    raise ValueError(f"official score does not match measured batch: {name}/{key}")
                legacy_features.append(extract_ml_features(extend, prefix))
                runtime_features.append(extract_ml_features(extend, runtime))
                steps.append({"model": name, "case": path.parent.name, "schedule_path": key[0],
                              "line_number": number, "forward_mode": mode, "batch_size": len(requests),
                              "sum_extend": sum(extend), "legacy_sum_past": sum(prefix),
                              "runtime_sum_past": sum(runtime), "actual_seconds": actual})
        if not steps or len(steps) != len(official):
            raise ValueError(f"official score coverage differs: {name}")
        frozen = bundle["model"]
        legacy = np.asarray(frozen.predict(np.asarray(legacy_features, dtype=float)), dtype=float)
        runtime = np.asarray(frozen.predict(np.asarray(runtime_features, dtype=float)), dtype=float)
        saved = np.asarray([float(official[(row["schedule_path"], row["line_number"])]["predicted_seconds"]) for row in steps])
        if (legacy.shape != (len(steps),) or runtime.shape != legacy.shape
                or not np.all(np.isfinite(legacy)) or not np.all(np.isfinite(runtime))
                or np.any(legacy < 0) or np.any(runtime < 0)):
            raise ValueError(f"invalid frozen predictions: {name}")
        if not np.allclose(legacy, saved, rtol=1e-10, atol=1e-12):
            raise ValueError(f"frozen bundle does not reproduce official prefix-only predictions: {name}")
        groups = {}
        for row, before, after in zip(steps, legacy, runtime):
            row.update(legacy_predicted_seconds=float(before), runtime_predicted_seconds=float(after),
                       legacy_ape_pct=abs(float(before)-row["actual_seconds"])/row["actual_seconds"]*100,
                       runtime_ape_pct=abs(float(after)-row["actual_seconds"])/row["actual_seconds"]*100)
            for key in (("all", "all"), ("case", row["case"]), ("phase", str(row["forward_mode"])),
                        ("case_phase", f"{row['case']}:mode{row['forward_mode']}")):
                groups.setdefault(key, []).append(row)
        for (kind, value), members in groups.items():
            old_mape = float(np.mean([row["legacy_ape_pct"] for row in members]))
            new_mape = float(np.mean([row["runtime_ape_pct"] for row in members]))
            summaries.append({"model": name, "group": kind, "value": value, "n": len(members),
                              "mean_actual_ms": float(np.mean([row["actual_seconds"] for row in members])*1000),
                              "mean_legacy_prediction_ms": float(np.mean([row["legacy_predicted_seconds"] for row in members])*1000),
                              "mean_runtime_prediction_ms": float(np.mean([row["runtime_predicted_seconds"] for row in members])*1000),
                              "legacy_mape_pct": old_mape, "runtime_mape_pct": new_mape,
                              "improvement_pp": old_mape-new_mape})
        if (hashlib.sha256(bundle_path.read_bytes()).hexdigest() != bundle_before
                or hashlib.sha256(score_path.read_bytes()).hexdigest() != score_before
                or any(hashlib.sha256(path.read_bytes()).hexdigest() != checksum for path, checksum in source_hashes.items())):
            raise ValueError(f"frozen artifacts changed during comparison: {name}")
        with (detail_root / f"{name}.csv").open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(steps[0]))
            writer.writeheader()
            writer.writerows(steps)
        identities.append({"model": name, "bundle_path": str(bundle_path.resolve()),
                           "bundle_sha256": bundle_before, "official_score_sha256": score_before,
                           "n": len(steps), "measured_source_files": len(source_hashes),
                           "official_predictions_reproduced": True, "fit_invoked": False})
    with (out / "frozen_input_comparison.csv").open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries, identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--compare-frozen-inputs", action="store_true",
                        help="CPU inference only: reuse final p50 bundles, change prefix-only to prefix-plus-output inputs")
    args = parser.parse_args()
    models = args.models.split(",")
    if any(not model or "/" in model or model in (".", "..") for model in models) or len(set(models)) != len(models):
        parser.error("--models must contain unique checkpoint directory names")
    if args.out_dir.exists():
        parser.error("refusing to overwrite an existing audit directory")
    contract = source_contract(Path(__file__).resolve().parents[1])
    rows, sources = audit(args.results_root.resolve(), models)
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "input_contract_audit.csv").open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons, frozen = compare_frozen_inputs(args.results_root.resolve(), models, args.out_dir) if args.compare_frozen_inputs else ([], [])
    scope = "Input-ABI consistency only; not independent physical seq_lens verification or model refitting."
    (args.out_dir / "source_manifest.json").write_text(json.dumps({"source_contract": contract,
        "audit_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measured_sources": sources, "frozen_models": frozen, "scope": scope}, indent=2), encoding="utf-8")
    print(f"Saved input-contract audit: {len(models)} models, {len(sources)} measured files, {sum(row['rows'] for row in rows)} steps; no fit() or GPU invocation.")
    if comparisons:
        print("Optional frozen-input comparison invoked CPU predict() twice per model, no fit() or GPU job; original bundles, scores and raw traces unchanged.")


if __name__ == "__main__":
    main()
