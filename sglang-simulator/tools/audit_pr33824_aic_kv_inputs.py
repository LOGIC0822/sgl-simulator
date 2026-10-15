#!/usr/bin/env python3
"""Compare AIC length/phase input contracts on identical saved native batches.

This CPU-only diagnostic reproduces original prefix-only scores before querying
prefix-plus-output inputs, then runtime-inferred phases on mismatching steps.
It does not tune, recollect, replay, or refit. The
runtime expression is an input-contract reference, not a physical seq_lens
measurement. Run after the formal AIC stage, with its kernel-stage PYTHONPATH.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tool(name: str):
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty comparison: {path}")
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def database_snapshot(config: Path) -> dict[str, str]:
    """Fingerprint this local systems directory, including its performance files."""
    raw = json.loads(config.read_text())
    predictor = raw["predictor"]
    if (predictor.get("name") != "aiconfigurator"
            or str(predictor.get("database_mode", "")).upper() != "SILICON"):
        raise ValueError("comparison requires an explicit AIC SILICON config")
    directory = Path(predictor["database_path"]).expanduser().resolve()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files or not any(path.name.endswith("_perf.txt") for path in files):
        raise ValueError(f"no local AIC performance tables: {directory}")
    return {str(path): digest(path) for path in [config.resolve(), *files]}


def measured_rows(model_root: Path) -> tuple[list[dict], dict, dict[str, str]]:
    """Read complete groups and pair every measured row with its original score."""
    audit = load_tool("audit_pr33824_kv_feature_contract")
    score = model_root / "holdout/aic_teacher_forced/per_batch.csv"
    checksums = {str(score.resolve()): digest(score)}
    official, resolved = {}, {}
    with score.open(newline="", encoding="utf-8") as source:
        for value in csv.DictReader(source):
            raw_path = value["schedule_path"]
            if raw_path not in resolved:
                resolved[raw_path] = str(Path(raw_path).resolve())
            key = (resolved[raw_path], int(value["line_number"]))
            if key in official:
                raise ValueError(f"duplicate original AIC step: {key}")
            prediction = float(value["predicted_seconds"])
            if not math.isfinite(prediction) or prediction < 0:
                raise ValueError(f"invalid original AIC prediction: {key}")
            official[key] = value
    groups = sorted((model_root / "holdout/real").glob("repeat-*"))
    if not groups:
        raise ValueError(f"no measured holdout groups: {model_root}")
    rows = []
    for group in groups:
        if (group / "BASELINE_COLLECTION_COMPLETE").read_text().strip() != "complete":
            raise ValueError(f"incomplete collected group: {group}")
        files = sorted((group / "cases").glob("*/TP0.schedule_batch.jsonl"))
        if not files:
            raise ValueError(f"no measured cases: {group}")
        for path in files:
            if (path.parent / "status.txt").read_text().strip() != "ok":
                raise ValueError(f"incomplete collected case: {path.parent}")
            canonical = str(path.resolve())
            checksums[canonical] = digest(path)
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                mode = item["forward_mode"]
                if mode not in (1, 2):
                    continue
                actual = float(item["iter_latency"])
                requests = item["request_infos"]
                if not requests or not math.isfinite(actual) or actual <= 0:
                    raise ValueError(f"invalid measured step: {canonical}:{number}")
                extend = [audit.length(req, "extend_input_len") for req in requests]
                prefix = [audit.length(req, "prefix_indices_len") for req in requests]
                output = [audit.length(req, "output_ids_len") for req in requests]
                previous = official.get((canonical, number))
                identity = (path.parent.name, mode, len(requests), sum(extend), sum(prefix), actual)
                if previous is None or identity != (
                    previous["case"], int(previous["forward_mode"]), int(previous["batch_size"]),
                    int(previous["sum_extend"]), int(previous["sum_past"]), float(previous["actual_seconds"])
                ):
                    raise ValueError(f"original AIC score mismatches measured step: {canonical}:{number}")
                rows.append({"case": path.parent.name, "schedule_path": canonical,
                             "line_number": number, "forward_mode": mode, "batch_size": len(requests),
                             "sum_extend": sum(extend), "sum_past": sum(prefix),
                             "extend": extend, "past": prefix,
                             "runtime_past": [past + generated for past, generated in zip(prefix, output)],
                             "actual_seconds": actual})
    if not rows or len(rows) != len(official):
        raise ValueError(f"original AIC score coverage differs: {model_root}")
    return rows, official, checksums


def compare_model(model_root: Path, predictor, scorer=None) -> tuple[list[dict], list[dict], dict]:
    """Use an unchanged predictor and exact native row targets for both inputs."""
    from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest

    scorer = scorer or load_tool("score_pr33824_aic_latency_model")
    rows, official, checksums = measured_rows(model_root)
    legacy_rows = [{key: value for key, value in row.items() if key != "runtime_past"} for row in rows]
    runtime_rows = [{**row, "past": source["runtime_past"], "sum_past": sum(source["runtime_past"])}
                    for row, source in zip(legacy_rows, rows)]
    legacy, legacy_queries = scorer.predict_rows(legacy_rows, predictor)
    runtime, runtime_queries = scorer.predict_rows(runtime_rows, predictor)
    if len(legacy) != len(rows) or len(runtime) != len(rows):
        raise ValueError("AIC scorer changed row coverage")
    # Offline AIC preserves native forward_mode; runtime infers it from lengths.
    # Query only mismatching rows for a separate phase+length input comparison.
    inferred_modes = [2 if ScheduleBatch(reqs=[ScheduleRequest(extend_length=value)
                                             for value in row["extend"]]).is_decode() else 1 for row in rows]
    mismatching = [index for index, row in enumerate(rows) if inferred_modes[index] != row["forward_mode"]]
    phase_rows = [{**runtime_rows[index], "forward_mode": inferred_modes[index]} for index in mismatching]
    phase_scores, phase_queries = scorer.predict_rows(phase_rows, predictor) if phase_rows else ([], 0)
    if len(phase_scores) != len(phase_rows):
        raise ValueError("AIC phase comparison changed row coverage")
    phase_predictions = {}
    for index, scored in zip(mismatching, phase_scores):
        source = rows[index]
        if (scored["schedule_path"], scored["line_number"], scored["forward_mode"], scored["actual_seconds"]) != (
                source["schedule_path"], source["line_number"], inferred_modes[index], source["actual_seconds"]):
            raise ValueError("AIC phase comparison changed measured identity")
        value = float(scored["predicted_seconds"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("AIC phase comparison returned invalid prediction")
        phase_predictions[index] = value
    details = []
    for index, (source, before, after) in enumerate(zip(rows, legacy, runtime)):
        key = (source["schedule_path"], source["line_number"])
        for scored in (before, after):
            if (scored["schedule_path"], scored["line_number"], scored["forward_mode"],
                    scored["actual_seconds"]) != (*key, source["forward_mode"], source["actual_seconds"]):
                raise ValueError(f"AIC scoring changed measured identity: {key}")
            prediction = float(scored["predicted_seconds"])
            if not math.isfinite(prediction) or prediction < 0:
                raise ValueError(f"AIC returned invalid prediction: {key}")
        if not math.isclose(float(before["predicted_seconds"]), float(official[key]["predicted_seconds"]),
                            rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"unchanged AIC does not reproduce original prefix-only score: {key}")
        actual = source["actual_seconds"]
        old, new = float(before["predicted_seconds"]), float(after["predicted_seconds"])
        phase_prediction = phase_predictions.get(index, new)
        details.append({"model": model_root.name, "case": source["case"], "schedule_path": key[0],
                        "line_number": key[1], "forward_mode": source["forward_mode"],
                        "runtime_inferred_forward_mode": inferred_modes[index],
                        "batch_size": source["batch_size"], "sum_extend": source["sum_extend"],
                        "legacy_sum_past": source["sum_past"], "runtime_sum_past": sum(source["runtime_past"]),
                        "actual_seconds": actual, "legacy_predicted_seconds": old,
                        "runtime_predicted_seconds": new,
                        "runtime_inferredphase_predicted_seconds": phase_prediction,
                        "legacy_ape_pct": abs(old-actual)/actual*100,
                        "runtime_ape_pct": abs(new-actual)/actual*100,
                        "runtime_inferredphase_ape_pct": abs(phase_prediction-actual)/actual*100})
    groups = {}
    for row in details:
        for key in (("all", "all"), ("case", row["case"]), ("phase", str(row["forward_mode"])),
                    ("case_phase", f"{row['case']}:mode{row['forward_mode']}")):
            groups.setdefault(key, []).append(row)
    summaries = []
    for (kind, value), members in groups.items():
        mean = lambda key: sum(row[key] for row in members) / len(members)
        summaries.append({"model": model_root.name, "group": kind, "value": value, "n": len(members),
                          "mean_actual_ms": mean("actual_seconds")*1000,
                          "mean_legacy_prediction_ms": mean("legacy_predicted_seconds")*1000,
                          "mean_runtime_prediction_ms": mean("runtime_predicted_seconds")*1000,
                          "mean_runtime_inferredphase_prediction_ms": mean("runtime_inferredphase_predicted_seconds")*1000,
                          "legacy_mape_pct": mean("legacy_ape_pct"), "runtime_mape_pct": mean("runtime_ape_pct"),
                          "runtime_inferredphase_mape_pct": mean("runtime_inferredphase_ape_pct"),
                          "runtime_phase_disagreement_rows": sum(row["forward_mode"] != row["runtime_inferred_forward_mode"] for row in members),
                          "improvement_pp": mean("legacy_ape_pct")-mean("runtime_ape_pct")})
    if any(digest(Path(path)) != checksum for path, checksum in checksums.items()):
        raise ValueError("measured traces/original AIC scores changed during comparison")
    return summaries, details, {"model": model_root.name, "n": len(details), "source_sha256": checksums,
                                "legacy_unique_queries": legacy_queries, "runtime_unique_queries": runtime_queries,
                                "runtime_phase_disagreement_rows": len(mismatching), "phase_unique_queries": phase_queries,
                                "original_predictions_reproduced": True, "recollect_or_fit_invoked": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    models = args.models.split(",")
    if not models or len(set(models)) != len(models) or any(not name or "/" in name or name in (".", "..") for name in models):
        parser.error("--models must contain unique checkpoint directory names")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("set CUDA_VISIBLE_DEVICES='' for this CPU-only diagnostic")
    if (root / "AIC_ALL_MODELS_COMPLETE").read_text().strip() != "complete":
        raise ValueError("wait for the formal AIC stage before this diagnostic")
    if args.out_dir.exists():
        parser.error("refusing to overwrite an existing comparison directory")
    metadata = dict(line.split("=", 1) for line in (root / "run_metadata.txt").read_text().splitlines() if "=" in line)
    stage = Path(metadata["kernel_compat_stage"]).resolve()
    kernel_spec = importlib.util.find_spec("sgl_kernel")
    if not kernel_spec or not kernel_spec.origin or not Path(kernel_spec.origin).resolve().is_relative_to(stage):
        raise ValueError("sgl_kernel resolution does not match the declared native compatibility stage")
    native = {str(path.resolve()): digest(path) for path in stage.rglob("*.so")}
    if not native:
        raise ValueError("no native binary in the declared compatibility stage")
    reference = json.loads((root / "provenance/formal_start_after_ninja_fix/sha256.json").read_text())
    reference_native = {path: checksum for path, checksum in reference.items() if path.startswith("/") and path.endswith(".so")}
    if not reference_native or any(native.get(path) != checksum for path, checksum in reference_native.items()):
        raise ValueError("kernel binary differs from the archived real-baseline reference")
    config = root / "aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"
    database = database_snapshot(config)
    with (root / "model_manifest.tsv").open() as source:
        entries = list(csv.DictReader(source, delimiter="\t"))
    paths = {row["model"]: Path(row["model_path"]) for row in entries}
    if len(paths) != len(entries) or not set(models) <= set(paths):
        raise ValueError("duplicate/missing model-manifest entries")
    contract = load_tool("audit_pr33824_kv_feature_contract").source_contract(Path(__file__).resolve().parents[1])
    runtime_batch = Path(__file__).resolve().parents[1] / "src/sglang_simulator/time_predictor/base.py"
    contract["runtime_batch"] = {"path": str(runtime_batch.resolve()), "sha256": digest(runtime_batch),
                                 "phase_reference": "ScheduleBatch.is_decode() called on saved extend lengths"}
    scorer = load_tool("score_pr33824_aic_latency_model")
    args.out_dir.mkdir(parents=True)
    details_root = args.out_dir / "per_batch"
    details_root.mkdir()
    summaries, identities = [], []
    for model in models:
        model_root = root / "models" / model
        if (model_root / "AIC_PREDICTOR_COMPLETE").read_text().strip() != "complete":
            raise ValueError(f"incomplete formal AIC model: {model}")
        checkpoint_config = paths[model] / "config.json"
        checkpoint_digest = digest(checkpoint_config)
        predictor = scorer.load_predictor(config, paths[model])
        result, details, identity = compare_model(model_root, predictor, scorer)
        if digest(checkpoint_config) != checkpoint_digest:
            raise ValueError(f"checkpoint config changed: {model}")
        identity.update(model_path=str(paths[model].resolve()), checkpoint_config_sha256=checkpoint_digest)
        identities.append(identity)
        summaries.extend(result)
        write_csv(details_root / f"{model}.csv", details)
        print(f"{model}: {len(details)} identical measured steps; original AIC scores reproduced", flush=True)
    if (database_snapshot(config) != database
            or any(digest(Path(path)) != checksum for path, checksum in native.items())):
        raise ValueError("AIC database/config or kernel binary changed during comparison")
    write_csv(args.out_dir / "aic_input_comparison.csv", summaries)
    (args.out_dir / "source_manifest.json").write_text(json.dumps({"source_contract": contract,
        "tool_sha256": digest(Path(__file__)), "aic_scorer_sha256": digest(Path(scorer.__file__)),
        "database_sha256": database, "kernel_sha256": native, "models": identities,
        "scope": "Frozen AIC database; input-only CPU queries; no fitting/recollection/E2E replay or independent physical seq_lens check."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
