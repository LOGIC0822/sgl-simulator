import csv, hashlib, json, math, os, statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
started = datetime.now().astimezone().isoformat()
root = Path("test-results/pr33824_expanded_all13_predictors_20260928_1933").resolve()
training = root/"verification/frozen_fit_input_qwen25_14b_1m_20260929_0802/results/ml_fit_input_readback/Qwen2.5-14B-Instruct-1M/per_batch.csv"
testing = root/"models/Qwen2.5-14B-Instruct-1M/holdout/ml_teacher_forced/per_batch.csv"
sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (training, testing)}
def read(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in ("forward_mode","batch_size","sum_extend","sum_past","line_number"):
            row[name] = int(row[name])
        for name in ("actual_seconds","predicted_seconds","absolute_percentage_error_pct"):
            row[name] = float(row[name])
        computed = abs(row["predicted_seconds"] - row["actual_seconds"])/row["actual_seconds"]*100
        assert math.isclose(computed,row["absolute_percentage_error_pct"],rel_tol=1e-10,abs_tol=1e-10)
        row["repeat"] = Path(row["schedule_path"]).parents[2].name
    return rows
train, held = read(training), read(testing)
assert len(train) == 5862 and len(held) == 9974
high = [row for row in held if row["absolute_percentage_error_pct"] >= 100]
assert len(high) == 96 and all(row["forward_mode"] == 2 for row in high)
index = defaultdict(list)
for row in train:
    index[(row["forward_mode"],row["batch_size"],row["sum_extend"])].append(row)
def stats(values):
    if not values: return None
    return {"n":len(values),"min":min(values),"median":statistics.median(values),"mean":statistics.mean(values),"max":max(values)}
groups=[]
for case in sorted({row["case"] for row in high}):
    selected = [row for row in high if row["case"] == case]
    matches=[]; unsupported=[]
    for row in selected:
        candidates = index[(row["forward_mode"],row["batch_size"],row["sum_extend"])]
        if not candidates:
            unsupported.append(row["line_number"]); continue
        gap = min(abs(candidate["sum_past"]-row["sum_past"]) for candidate in candidates)
        nearest = [candidate for candidate in candidates if abs(candidate["sum_past"]-row["sum_past"]) == gap]
        matches.append({"repeat":row["repeat"],"line_number":row["line_number"],"batch_size":row["batch_size"],"held_sum_past":row["sum_past"],"held_actual_ms":row["actual_seconds"]*1000,"held_prediction_ms":row["predicted_seconds"]*1000,"nearest_sum_past_gap_tokens":gap,"relative_sum_past_gap_pct":gap/max(1,row["sum_past"])*100,"nearest_training_rows":len(nearest),"nearest_training_actual_mean_ms":statistics.mean(n["actual_seconds"]*1000 for n in nearest),"nearest_training_actual_min_ms":min(n["actual_seconds"]*1000 for n in nearest),"nearest_training_actual_max_ms":max(n["actual_seconds"]*1000 for n in nearest)})
    groups.append({"case":case,"high_error_rows":len(selected),"repeat_counts":dict(sorted(Counter(row["repeat"] for row in selected).items())),"matching_mode_batch_extend_rows":len(matches),"missing_mode_batch_extend_rows":len(unsupported),"exact_sum_past_match_rows":sum(row["nearest_sum_past_gap_tokens"] == 0 for row in matches),"nearest_training_actual_mean_ms":stats([row["nearest_training_actual_mean_ms"] for row in matches]),"relative_sum_past_gap_pct":stats([row["relative_sum_past_gap_pct"] for row in matches]),"matching_records":matches})
for path, digest in sources.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
result={"started_at":started,"finished_at":datetime.now().astimezone().isoformat(),"exit_code":0,"model":"Qwen2.5-14B-Instruct-1M","training_rows":len(train),"test_rows":len(held),"descriptive_ape_threshold_pct":100,"high_error_rows":len(high),"training_decode_actual_ms":stats([row["actual_seconds"]*1000 for row in train if row["forward_mode"] == 2]),"cohort_projected_nearest_training_checks":groups,"source_sha256_unchanged":sources,"fit_prediction_gpu_or_measurement_change_invoked":False,"claim_scope":"Read-only descriptive matching on native forward mode, batch size and sum_extend, then nearest sum_past. These four descriptors are not the full 18-feature vector: no full-feature coverage, causal, latency correction, new training or all-model conclusion. Tied nearest training labels are averaged, not a prediction method."}
print(json.dumps(result,ensure_ascii=False))
