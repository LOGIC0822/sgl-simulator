import csv, hashlib, json, math, os, shlex, statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
sim=Path("/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator")
root=sim/"test-results/pr33824_expanded_all13_predictors_20260928_1933"
model=root/"models/Qwen3-30B-A3B-Instruct-2507"
case_counts={"latency_256x128_r1":32,"service_1024x256_r4":64,"saturation_1024x256_inf":64,"latency_128x64_r1":32,"saturation_1536x384_inf":64}
bundle=model/"calibration/training/Qwen3-30B-A3B-Instruct-2507_hgbmono_p50.pkl"
training_proof=root/"verification/moe30_instruct_full_training_input_readback_20260929_2228/manifest.json"
score=model/"holdout/ml_teacher_forced/per_batch.csv"
summary=model/"holdout/ml_teacher_forced/summary.csv"
readme=model/"holdout/ml_teacher_forced/README.md"
train_files=sorted((model/"calibration/real").glob("repeat-*/cases/*/TP0.raw_request.jsonl"))
assert len(train_files)==15
sources=train_files+[bundle,training_proof,score,summary,readme,root/"run_metadata.txt",sim/"tools/score_pr33824_latency_model.py"]
train_package=model/"calibration/real/repeat-1/package_versions.txt"
sources.append(train_package)
for repeat in range(1,6):
    group=model/f"holdout/real/repeat-{repeat}"
    sources.extend(group/name for name in ("BASELINE_COLLECTION_COMPLETE","run_metadata.txt","package_versions.txt"))
    for case in case_counts:
        directory=group/"cases"/case
        sources.extend(directory/name for name in ("status.txt","TP0.raw_request.jsonl","TP0.schedule_batch.jsonl","TP0.manifest.json","autobench_trace.jsonl","benchmark.jsonl"))
        sources.extend(group/"cases"/(case+suffix) for suffix in (".server.command.txt",".benchmark.command.txt"))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def prompt_digest(row):
    return hashlib.sha256(json.dumps(row["input_ids"],separators=(",",":")).encode()).hexdigest()
before={str(path):digest(path) for path in sources}
assert before[str(bundle)]=="5b13bad77cae8f3421c988eeeeba45ecbaadbffbbf6bf1d23f8dabd26a7620d8"
assert before[str(training_proof)]=="4488c67183d3e927cf462057eae7bb84a5db331bd841f734d0d84329dadc1e0c"
assert str(bundle.resolve()) in readme.read_text()
train_ids=set();train_prompts=set()
for path in train_files:
    for row in rows(path):
        assert row["rid"] not in train_ids
        train_ids.add(row["rid"]);train_prompts.add(prompt_digest(row))
assert len(train_ids)==800 and len(train_prompts)==480
test_ids=set();test_prompts=set();observed={};group_counts=[]
for repeat in range(1,6):
    group=model/f"holdout/real/repeat-{repeat}"
    assert (group/"BASELINE_COLLECTION_COMPLETE").read_text().strip()=="complete"
    assert (group/"package_versions.txt").read_bytes()==train_package.read_bytes()
    metadata=dict(line.split("=",1) for line in (group/"run_metadata.txt").read_text().splitlines() if "=" in line)
    assert metadata["model"]=="Qwen3-30B-A3B-Instruct-2507" and int(metadata["seed"])==202609379+repeat
    assert metadata["source_revision"]=="59799a368793b9f795baf59b067233a65ad8e38e"
    assert set(metadata["requested_cases"].split(","))==set(case_counts)
    counts={}
    for case,expected in case_counts.items():
        directory=group/"cases"/case
        assert (directory/"status.txt").read_text().strip()=="ok"
        manifest=json.loads((directory/"TP0.manifest.json").read_text())
        raw=rows(directory/"TP0.raw_request.jsonl")
        traces=rows(directory/"autobench_trace.jsonl")
        batches=rows(directory/"TP0.schedule_batch.jsonl")
        benchmark=rows(directory/"benchmark.jsonl")[-1]
        assert len(raw)==len(traces)==manifest["request_count"]==benchmark["completed"]==expected
        assert len(batches)==manifest["schedule_batch_count"]
        assert manifest["batch_timing"]=="device_synchronized" and manifest["batch_latency_unit"]=="seconds"
        info=benchmark["server_info"]
        assert info["device"]=="cuda" and info["speculative_algorithm"] is None
        assert info["disable_overlap_schedule"] is True and info["disable_cuda_graph"] is True
        server=shlex.split((group/"cases"/(case+".server.command.txt")).read_text())
        client=shlex.split((group/"cases"/(case+".benchmark.command.txt")).read_text())
        assert server[server.index("-m")+1]=="sglang_simulator.baseline.launch_server"
        assert server[server.index("-m")-1]=="/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python"
        assert server[server.index("--device")+1]=="cuda"
        assert server[server.index("--mem-fraction-static")+1]=="0.90"
        assert "--disable-overlap-schedule" in server and "--disable-cuda-graph" in server
        assert client[client.index("-m")+1]=="sglang.benchmark.serving"
        assert int(client[client.index("--seed")+1])==202609379+repeat
        assert Path(server[server.index("--model-path")+1]).resolve()==Path("/home/ms-admin/data/models/Qwen3-30B-A3B-Instruct-2507").resolve()
        copied={item["metadata"]["source_rid"]:item for item in traces}
        assert len(copied)==expected
        case_ids=set()
        for row in raw:
            rid=row["rid"];assert rid not in test_ids
            test_ids.add(rid);case_ids.add(rid);test_prompts.add(prompt_digest(row))
            copy=copied[rid]
            assert row["input_ids"]==copy["prompt"]
            assert row["input_length"]==len(row["input_ids"])==copy["prompt_len"]
            assert row["output_length"]==len(row["output_ids"])==copy["output_len"]
        schedule=directory/"TP0.schedule_batch.jsonl"
        last_end=-math.inf
        for line_number,line in enumerate(schedule.read_text().splitlines(),1):
            if not line.strip():continue
            batch=json.loads(line)
            assert int(batch["forward_mode"]) in (1,2) and batch["request_infos"]
            start,end,actual=batch["start_timestamp"],batch["end_timestamp"],batch["iter_latency"]
            assert all(math.isfinite(value) for value in (start,end,actual))
            assert end>start>=last_end and actual>0
            assert math.isclose(end-start,actual,rel_tol=1e-12,abs_tol=1e-12)
            last_end=end
            reqs=batch["request_infos"]
            assert all(req["rid"] in case_ids for req in reqs)
            key=(str(schedule.resolve()),line_number)
            assert key not in observed
            observed[key]={"case":case,"forward_mode":int(batch["forward_mode"]),"batch_size":len(reqs),
                           "sum_extend":sum(req["extend_input_len"] for req in reqs),
                           "sum_past":sum(req["prefix_indices_len"] for req in reqs),
                           "actual_seconds":actual}
        counts[case]={"requests":expected,"timing_rows":len(batches)}
    group_counts.append({"repeat":repeat,"cases":counts,"requests":sum(v["requests"] for v in counts.values()),
                         "timing_rows":sum(v["timing_rows"] for v in counts.values())})
assert len(test_ids)==1280 and len(observed)==12188
assert not (train_ids&test_ids) and not (train_prompts&test_prompts)
with score.open() as stream:
    saved=list(csv.DictReader(stream))
assert len(saved)==len(observed)
seen=set();partitions=defaultdict(list)
for row in saved:
    key=(str(Path(row["schedule_path"]).resolve()),int(row["line_number"]))
    assert key not in seen and key in observed
    seen.add(key)
    for field,expected in observed[key].items():
        actual=row[field] if isinstance(expected,str) else float(row[field])
        assert actual==expected,(key,field)
    actual=float(row["actual_seconds"]);predicted=float(row["predicted_seconds"])
    assert math.isfinite(predicted) and predicted>=0
    signed=(predicted-actual)/actual*100
    assert math.isclose(float(row["signed_error_pct"]),signed,rel_tol=1e-10,abs_tol=1e-8)
    assert math.isclose(float(row["absolute_percentage_error_pct"]),abs(signed),rel_tol=1e-10,abs_tol=1e-8)
    numeric={"actual":actual,"predicted":predicted,"signed":signed,"ape":abs(signed),"error":predicted-actual}
    for key in (("all","all"),("case",row["case"]),("forward_mode",row["forward_mode"]),
                ("case_forward_mode",row["case"]+":mode"+row["forward_mode"])):
        partitions[key].append(numeric)
assert len(seen)==len(observed)
def percentile(values,q):
    ordered=sorted(values);position=(len(ordered)-1)*q
    left=int(math.floor(position));right=int(math.ceil(position))
    return ordered[left]+(ordered[right]-ordered[left])*(position-left)
with summary.open() as stream:
    summaries=list(csv.DictReader(stream))
assert len(summaries)==len(partitions)==18
checked_summary=[]
for row in summaries:
    members=partitions.pop((row["group"],row["value"]))
    mean=lambda field:statistics.fmean(item[field] for item in members)
    expected={"n":len(members),"mean_actual_ms":mean("actual")*1000,"mean_predicted_ms":mean("predicted")*1000,
              "mae_ms":statistics.fmean(abs(item["error"]) for item in members)*1000,
              "rmse_ms":math.sqrt(statistics.fmean(item["error"]**2 for item in members))*1000,
              "mean_signed_error_pct":mean("signed"),"mape_pct":mean("ape"),
              "median_ape_pct":percentile([item["ape"] for item in members],.5),
              "p95_ape_pct":percentile([item["ape"] for item in members],.95)}
    for field,value in expected.items():
        assert math.isclose(float(row[field]),value,rel_tol=1e-10,abs_tol=1e-8),(row["group"],row["value"],field)
    checked_summary.append({"group":row["group"],"value":row["value"],**expected})
assert not partitions
after={str(path):digest(path) for path in sources}
assert before==after
print(json.dumps({"observed_at":datetime.now().astimezone().isoformat(),"model":"Qwen3-30B-A3B-Instruct-2507",
                  "training_requests":len(train_ids),"training_distinct_complete_inputs":len(train_prompts),
                  "test_requests":len(test_ids),"test_distinct_complete_inputs":len(test_prompts),
                  "test_timing_rows":len(observed),"shared_request_ids":0,"shared_complete_inputs":0,
                  "groups":group_counts,"verified_summary_partitions":checked_summary,
                  "frozen_p50_sha256":before[str(bundle)],"source_files_unchanged":len(before),
                  "source_manifest_sha256":hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
                  "source_key_root":str(root),
                  "source_sha256":{str(Path(path).relative_to(root)) if Path(path).is_relative_to(root) else path:value for path,value in before.items()},
                  "scope":"All five completed native test groups; exact raw-to-replay request payloads and original stored score identities and all18 summary statistics checked. No model/bundle loading, fit, predict, GPU, replay, AIC table query, or actual operator profile is performed. This does not complete E2E or fresh AIC results."}))
