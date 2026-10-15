# Read-only first-independent-group audit for the second MoE checkpoint.
import hashlib,json,math,os,shlex,time
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
started=time.time()
root=Path("test-results/pr33824_expanded_all13_predictors_20260928_1933").resolve()
model=root/"models"/"Qwen3-30B-A3B-Instruct-2507"
training_files=[model/"calibration"/"real"/f"repeat-{repeat}"/"cases"/case/"TP0.raw_request.jsonl" for repeat in range(1,6) for case in ("latency_256x128_r1","service_1024x256_r4","saturation_1024x256_inf")]
held=model/"holdout"/"real"/"repeat-1"
cases={"latency_256x128_r1":32,"service_1024x256_r4":64,"saturation_1024x256_inf":64,"latency_128x64_r1":32,"saturation_1536x384_inf":64}
proof=root/"verification"/"moe30_instruct_full_training_input_readback_20260929_2228"/"manifest.json"
bundle=model/"calibration"/"training"/"Qwen3-30B-A3B-Instruct-2507_hgbmono_p50.pkl"
files=training_files+[proof,bundle,held/"run_metadata.txt",held/"package_versions.txt",held/"BASELINE_COLLECTION_COMPLETE"]
for case in cases:
    folder=held/"cases"/case
    files += [folder/name for name in ("status.txt","TP0.raw_request.jsonl","TP0.schedule_batch.jsonl","TP0.manifest.json","autobench_trace.jsonl","autobench_trace.manifest.json","benchmark.jsonl")]
    files += [held/"cases"/f"{case}.benchmark.command.txt"]
def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def digest(ids): return hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()
before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
assert before[str(proof)]=="4488c67183d3e927cf462057eae7bb84a5db331bd841f734d0d84329dadc1e0c"
assert before[str(bundle)]=="5b13bad77cae8f3421c988eeeeba45ecbaadbffbbf6bf1d23f8dabd26a7620d8"
metadata=dict(line.split("=",1) for line in (held/"run_metadata.txt").read_text().splitlines() if "=" in line)
assert (held/"BASELINE_COLLECTION_COMPLETE").read_text().strip()=="complete"
assert metadata["model"]=="Qwen3-30B-A3B-Instruct-2507" and metadata["seed"]=="202609380"
assert metadata["source_revision"]=="59799a368793b9f795baf59b067233a65ad8e38e"
train_ids=set();train_inputs=set();train_count=0
for path in training_files:
    for row in rows(path):
        assert row["rid"] not in train_ids
        train_ids.add(row["rid"]);train_inputs.add(digest(row["input_ids"]));train_count+=1
assert train_count==len(train_ids)==800 and len(train_inputs)==480
test_ids=set();test_inputs=set();case_results={};timing_count=0
for case,count in cases.items():
    folder=held/"cases"/case
    assert (folder/"status.txt").read_text().strip()=="ok"
    manifest=json.loads((folder/"TP0.manifest.json").read_text())
    raw=rows(folder/"TP0.raw_request.jsonl");batches=rows(folder/"TP0.schedule_batch.jsonl");trace=rows(folder/"autobench_trace.jsonl")
    benchmark=rows(folder/"benchmark.jsonl")[-1]
    assert len(raw)==len(trace)==manifest["request_count"]==benchmark["completed"]==count
    assert benchmark["server_info"]["device"]=="cuda"
    assert len(batches)==manifest["schedule_batch_count"]
    assert manifest["batch_timing"]=="device_synchronized" and manifest["batch_latency_unit"]=="seconds"
    client=shlex.split((held/"cases"/f"{case}.benchmark.command.txt").read_text())
    assert client[client.index("--seed")+1]=="202609380"
    copies={item["metadata"]["source_rid"]:item for item in trace}
    assert len(copies)==count
    case_ids=set()
    for row in raw:
        rid=row["rid"];assert rid not in test_ids
        test_ids.add(rid);case_ids.add(rid);test_inputs.add(digest(row["input_ids"]))
        copy=copies[rid]
        assert row["input_ids"]==copy["prompt"]
        assert row["input_length"]==len(copy["prompt"])==copy["prompt_len"]
        assert row["output_length"]==len(row["output_ids"])==copy["output_len"]
    last_end=-math.inf
    for row in batches:
        start,end,latency=row["start_timestamp"],row["end_timestamp"],row["iter_latency"]
        assert all(math.isfinite(value) for value in (start,end,latency))
        assert latency>0 and end>start and start>=last_end
        assert math.isclose(latency,end-start,rel_tol=1e-12,abs_tol=1e-12)
        last_end=end
        assert all(info["rid"] in case_ids for info in row["request_infos"])
    timing_count+=len(batches)
    case_results[case]={"requests":count,"timing_rows":len(batches)}
assert len(test_ids)==256 and timing_count==2530 and not (train_ids & test_ids) and not (train_inputs & test_inputs)
after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
assert before==after
result={"model":"Qwen3-30B-A3B-Instruct-2507","training_requests":train_count,"training_distinct_complete_inputs":len(train_inputs),"partial_holdout_repeat":1,"partial_holdout_cases":case_results,"partial_test_requests":len(test_ids),"partial_test_distinct_complete_inputs":len(test_inputs),"partial_test_timing_rows":timing_count,"shared_request_ids":len(train_ids&test_ids),"shared_complete_input_sequences":len(train_inputs&test_inputs),"frozen_p50_sha256":before[str(bundle)],"source_files_unchanged":len(before),"source_sha256":before,"started_unix":started,"finished_unix":time.time(),"scope":"The entire completed first holdout group, all five workloads. Does not establish completion or independence of the remaining four test groups, nor any independent prediction error. No model loading, fit/predict, new GPU, replay, AIC or profile call."}
print(json.dumps(result,sort_keys=True))
