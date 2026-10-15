import csv, hashlib, importlib.util, json, math, os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
started = datetime.now().astimezone().isoformat()
sim=Path.cwd(); root=(sim/"test-results/pr33824_expanded_all13_predictors_20260928_1933").resolve()
train_path=root/"verification/frozen_fit_input_qwen25_14b_1m_20260929_0802/results/ml_fit_input_readback/Qwen2.5-14B-Instruct-1M/per_batch.csv"
held_path=root/"models/Qwen2.5-14B-Instruct-1M/holdout/ml_teacher_forced/per_batch.csv"
feature_path=sim/"src/sglang_simulator/time_predictor/ml_features.py"
hashes={}
def preserve(path):
    path=path.resolve()
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path
spec=importlib.util.spec_from_file_location("standalone_feature_contract",preserve(feature_path))
features=importlib.util.module_from_spec(spec); spec.loader.exec_module(features)
def read(path):
    with preserve(path).open(newline="") as f: rows=list(csv.DictReader(f))
    wanted=defaultdict(dict)
    for row in rows: wanted[Path(row["schedule_path"])][int(row["line_number"])]=row
    for path, identities in wanted.items():
        with preserve(path).open() as f:
            for number,line in enumerate(f,1):
                if number not in identities: continue
                raw=json.loads(line); row=identities.pop(number)
                info=raw["request_infos"]
                assert int(raw["forward_mode"])==int(row["forward_mode"])
                assert float(raw["iter_latency"])==float(row["actual_seconds"])
                ex=[int(r["extend_input_len"]) for r in info]
                past=[int(r["prefix_indices_len"]) for r in info]
                assert len(ex)==int(row["batch_size"]) and sum(ex)==int(row["sum_extend"]) and sum(past)==int(row["sum_past"])
                row["feature_vector"]=features.extract_ml_features(ex,past)
                row["forward_mode"]=int(row["forward_mode"])
                row["actual_ms"]=float(row["actual_seconds"])*1000
                row["predicted_ms"]=float(row["predicted_seconds"])*1000
                row["ape_pct"]=float(row["absolute_percentage_error_pct"])
                assert math.isclose(abs(row["predicted_ms"]-row["actual_ms"])/row["actual_ms"]*100,row["ape_pct"],rel_tol=1e-10,abs_tol=1e-10)
        assert not identities
    return rows
train, held=read(train_path),read(held_path)
assert len(train)==5862 and len(held)==9974
high=[r for r in held if r["ape_pct"]>=100]
assert len(high)==96
dec=[r for r in train if r["forward_mode"]==2]
pref_one=[r for r in train if r["forward_mode"]==1 and r["feature_vector"][-2]==1]
limits=[(min(r["feature_vector"][i] for r in dec),max(r["feature_vector"][i] for r in dec)) for i in range(18)]
high_features=[]
for i,name in enumerate(features.ML_FEATURE_NAMES):
    vals=[r["feature_vector"][i] for r in high]
    lo,hi=limits[i]
    high_features.append({"feature":name,"training_decode_min":lo,"training_decode_max":hi,"high_error_min":min(vals),"high_error_max":max(vals),"high_rows_below_training_decode_min":sum(v<lo for v in vals),"high_rows_above_training_decode_max":sum(v>hi for v in vals)})
    if name=="min_past": past_idx=i
all_outside=Counter()
for row in held:
    if row["forward_mode"]!=2: continue
    outside=any(v<lo or v>hi for v,(lo,hi) in zip(row["feature_vector"],limits))
    all_outside[("high" if row["ape_pct"]>=100 else "other",outside)]+=1
phase_by_feature=defaultdict(set)
for row in train: phase_by_feature[tuple(row["feature_vector"])].add(row["forward_mode"])
collisions=[v for v,modes in phase_by_feature.items() if len(modes)>1]
result={"started_at":started,"finished_at":datetime.now().astimezone().isoformat(),"exit_code":0,"model":"Qwen2.5-14B-Instruct-1M","training_rows":len(train),"test_rows":len(held),"high_error_rows":len(high),"high_error_repeat_counts":dict(Counter(Path(r["schedule_path"]).parents[2].name for r in high)),"training_mode1_inferred_decode_rows":len(pref_one),"training_mode1_inferred_decode_actual_ms":[r["actual_ms"] for r in pref_one],"training_mode1_inferred_decode_features":[dict(zip(features.ML_FEATURE_NAMES,r["feature_vector"])) for r in pref_one],"training_exact_feature_vectors_shared_by_native_phases":len(collisions),"high_error_full_feature_exact_training_matches":sum(tuple(r["feature_vector"]) in phase_by_feature for r in high),"decode_train_box_counts":[{"cohort":cohort,"outside_train_decode_box":outside,"rows":n} for (cohort,outside),n in sorted(all_outside.items())],"high_error_feature_ranges":high_features,"source_sha256_unchanged":hashes,"fit_prediction_gpu_or_measurement_change_invoked":False,"claim_scope":"Descriptive raw-data readback of all 18 legacy offline features and native phase, not refit or predictive validation. One-token prefill can encode is_decode=1, but no causal/error correction attribution is proved by these counts or bounding boxes."}
for path,digest in hashes.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
print(json.dumps(result,ensure_ascii=False))
