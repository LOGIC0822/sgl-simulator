#!/usr/bin/env python3
"""Read-only audit of closed Qwen7 native requests and saved ML scores."""
import csv, hashlib, json, math, statistics
from pathlib import Path
root = Path("test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen2.5-7B-Instruct").resolve()
import numpy as np
audit_dir = Path(__file__).resolve().parent
out = audit_dir / "results"
assert not out.exists(), "refusing to overwrite earlier diagnostic outputs"
source_paths = sorted({
    path.resolve()
    for folder in ("calibration/real", "holdout/real", "holdout/ml_teacher_forced")
    for path in (root / folder).rglob("*")
    if path.is_file()
} | {
    (root / "calibration/training/Qwen2.5-7B-Instruct_hgbmono_p50.pkl").resolve(),
    Path(__file__).resolve(),
    Path("tools/score_pr33824_latency_model.py").resolve(),
})
before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def digest(item):
    return hashlib.sha256(json.dumps(item["input_ids"],separators=(",",":")).encode()).hexdigest()
training = [item for path in (root/"calibration/real").glob("repeat-*/cases/*/TP0.raw_request.jsonl") for item in load(path)]
assert len(training) == 800
train_ids = {item["rid"] for item in training}
assert len(train_ids) == 800
train_prompts = {digest(item) for item in training}
groups, test_ids, test_prompts, observed = [], set(), set(), {}
expected_cases = {"latency_256x128_r1":32, "service_1024x256_r4":64, "saturation_1024x256_inf":64, "latency_128x64_r1":32, "saturation_1536x384_inf":64}
for repeat in range(1,6):
    group = root / f"holdout/real/repeat-{repeat}"
    assert (group/"BASELINE_COLLECTION_COMPLETE").read_text().strip() == "complete"
    assert (group/"package_versions.txt").read_bytes() == (root/"calibration/real/repeat-1/package_versions.txt").read_bytes()
    meta = dict(line.split("=",1) for line in (group/"run_metadata.txt").read_text().splitlines() if "=" in line)
    assert int(meta["seed"]) == 202609379+repeat
    cases = {}
    for path in sorted((group/"cases").glob("*/TP0.raw_request.jsonl")):
        case = path.parent
        assert (case/"status.txt").read_text().strip() == "ok"
        manifest = json.loads((case/"TP0.manifest.json").read_text())
        assert manifest["batch_latency_unit"] == "seconds"
        assert manifest["batch_timing"] == "device_synchronized"
        raw, trace = load(path), load(case/"autobench_trace.jsonl")
        schedule = case/"TP0.schedule_batch.jsonl"
        batches = load(schedule)
        benchmark = load(case/"benchmark.jsonl")[-1]
        assert len(raw) == manifest["request_count"] == len(trace) == benchmark["completed"] == expected_cases[case.name]
        assert len(batches) == manifest["schedule_batch_count"]
        copies = {item["metadata"]["source_rid"]:item for item in trace}
        assert len(copies) == len(raw)
        for item in raw:
            assert item["rid"] not in train_ids|test_ids
            assert digest(item) not in train_prompts
            copied = copies[item["rid"]]
            assert len(item["input_ids"]) == item["input_length"] == copied["prompt_len"]
            assert item["input_ids"] == copied["prompt"] and len(item["output_ids"]) == item["output_length"] == copied["output_len"]
            test_ids.add(item["rid"])
            test_prompts.add(digest(item))
        for line, item in enumerate(batches,1):
            assert math.isfinite(item["iter_latency"]) and 0 < item["iter_latency"] <= 30
            assert math.isclose(item["iter_latency"],item["end_timestamp"]-item["start_timestamp"],rel_tol=1e-12)
            key = (str(schedule.resolve()),line)
            assert key not in observed
            requests = item["request_infos"]
            assert requests and int(item["forward_mode"]) in (1, 2)
            assert all(request["rid"] in copies for request in requests)
            observed[key] = (case.name, int(item["forward_mode"]), float(item["iter_latency"]),
                             len(requests), sum(int(request["extend_input_len"]) for request in requests),
                             sum(int(request["prefix_indices_len"]) for request in requests))
        cases[case.name] = {"requests":len(raw),"timing_rows":len(batches)}
    assert set(cases) == set(expected_cases)
    groups.append({"repeat":repeat,"seed":int(meta["seed"]),"cases":cases,"requests":sum(item["requests"] for item in cases.values()),"timing_rows":sum(item["timing_rows"] for item in cases.values())})
assert len(test_ids) == 1280 and len(observed) == 19018
directory = root/"holdout/ml_teacher_forced"
with (directory/"per_batch.csv").open() as source:
    scored = list(csv.DictReader(source))
assert len(scored) == len(observed)
score_keys = set()
by_case = {}
actuals, predictions, apes = [], [], []
for row in scored:
    key = (str(Path(row["schedule_path"]).resolve()),int(row["line_number"]))
    assert key in observed and key not in score_keys
    score_keys.add(key)
    case, mode, actual, batch_size, sum_extend, sum_past = observed[key]
    assert (int(row["batch_size"]), int(row["sum_extend"]), int(row["sum_past"])) == (batch_size, sum_extend, sum_past)
    assert row["case"] == case and int(row["forward_mode"]) == mode
    assert math.isclose(float(row["actual_seconds"]),actual,rel_tol=1e-12)
    prediction = float(row["predicted_seconds"])
    assert math.isfinite(prediction) and prediction >= 0
    signed = (prediction - actual) / actual * 100
    ape = abs(signed)
    assert math.isclose(float(row["signed_error_pct"]), signed, rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(float(row["absolute_percentage_error_pct"]), ape, rel_tol=1e-10, abs_tol=1e-10)
    by_case.setdefault(case,[]).append(ape)
    actuals.append(actual); predictions.append(prediction); apes.append(ape)
with (directory/"summary.csv").open() as source:
    summary = list(csv.DictReader(source))
overall = [item for item in summary if item["group"]=="all"]
assert len(overall)==1 and int(overall[0]["n"])==len(scored)
for field,value in {"mean_actual_ms":statistics.mean(actuals)*1000,"mean_predicted_ms":statistics.mean(predictions)*1000,"mape_pct":statistics.mean(apes)}.items():
    assert math.isclose(float(overall[0][field]),value,rel_tol=1e-10)
seen_cases = ("latency_256x128_r1","service_1024x256_r4","saturation_1024x256_inf")
unseen_cases = ("latency_128x64_r1","saturation_1536x384_inf")
result = {"groups":groups,"training_requests":len(training),"training_unique_prompts":len(train_prompts),
          "independent_test_requests":len(test_ids),"independent_test_unique_prompts":len(test_prompts),
          "test_timing_rows":len(observed),"training_prompt_overlap":0,"reused_request_ids":0,
          "actual_replay_payloads_verified":len(test_ids),"raw_score_identity_rows_verified":len(score_keys),
          "mean_actual_ms":statistics.mean(actuals)*1000,"mean_predicted_ms":statistics.mean(predictions)*1000,
          "all_row_weighted_mape_pct":statistics.mean(apes),
          "seen_workload_equal_case_mape_pct":statistics.mean(statistics.mean(by_case[case]) for case in seen_cases),
          "unseen_workload_equal_case_mape_pct":statistics.mean(statistics.mean(by_case[case]) for case in unseen_cases),
          "claim_scope":"Full five-group collection and saved single-step scoring verified; no E2E completion or fresh AIC claim."}
partitions = {("all", "all"): scored}
for row in scored:
    for key in (("case", row["case"]), ("forward_mode", row["forward_mode"]),
                ("case_forward_mode", row["case"] + ":mode" + row["forward_mode"])):
        partitions.setdefault(key, []).append(row)
assert {(row["group"], row["value"]) for row in summary} == set(partitions)
assert len(summary) == len(partitions) == 18
for row in summary:
    rows = partitions[(row["group"], row["value"])]
    actual = np.asarray([float(item["actual_seconds"]) for item in rows])
    predicted = np.asarray([float(item["predicted_seconds"]) for item in rows])
    signed = (predicted - actual) / actual * 100
    absolute = np.abs(signed)
    expected = {
        "mean_actual_ms": float(actual.mean() * 1000),
        "mean_predicted_ms": float(predicted.mean() * 1000),
        "mae_ms": float(np.abs(predicted - actual).mean() * 1000),
        "rmse_ms": float(np.sqrt(np.mean((predicted - actual) ** 2)) * 1000),
        "mean_signed_error_pct": float(signed.mean()),
        "mape_pct": float(absolute.mean()),
        "median_ape_pct": float(np.median(absolute)),
        "p95_ape_pct": float(np.percentile(absolute, 95)),
    }
    assert int(row["n"]) == len(rows)
    for field, value in expected.items():
        assert math.isclose(float(row[field]), value, rel_tol=1e-10, abs_tol=1e-10), (row, field, value)
assert score_keys == set(observed)
after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
assert before == after, "closed original inputs changed during diagnostic"
result.update({
    "saved_summary_groups_verified": len(summary),
    "unchanged_source_files": len(source_paths),
    "bundle_sha256": before[(root / "calibration/training/Qwen2.5-7B-Instruct_hgbmono_p50.pkl").resolve()],
    "prefill": next(row for row in summary if row["group"] == "forward_mode" and row["value"] == "1"),
    "decode": next(row for row in summary if row["group"] == "forward_mode" and row["value"] == "2"),
    "source_scope": "completed native calibration/holdout files, saved score files, frozen bundle, executed audit and original scorer",
    "version_check": "all five complete holdout package-version records byte-match completed calibration group 1",
    "claim_scope": "request/trace identity and saved-score statistical audit, not a re-execution of prediction, fit-input readback, E2E completion or fresh AIC scoring",
})
out.mkdir()
with (out / "source_sha256.csv").open("x", newline="") as destination:
    writer = csv.writer(destination)
    writer.writerow(("path", "sha256"))
    writer.writerows((str(path), before[path]) for path in source_paths)
(out / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result,indent=2,sort_keys=True))
print("PASS: all five independent real groups and every saved ML score agree; no fit, new prediction invocation or extra GPU job.")

