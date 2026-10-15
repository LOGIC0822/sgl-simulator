# Read-only adaptation of the completed first-MoE training-input audit.
# Full payload/clock/settings checks cover repeat 5; all repeats have count,
# unique-request, training-source membership and manifest-row checks.
import hashlib, json, math, os, shlex, statistics, sys, time
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
started = time.time()
root = Path("test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-30B-A3B-Instruct-2507/calibration/real/repeat-5").resolve()
expected = {
    "latency_256x128_r1": (32, "256", "128", "1"),
    "service_1024x256_r4": (64, "1024", "256", "4"),
    "saturation_1024x256_inf": (64, "1024", "256", "inf"),
}
selected = [root/"BASELINE_COLLECTION_COMPLETE",root/"run_metadata.txt",root/"package_versions.txt"]
for case in expected:
    folder=root/"cases"/case
    selected += [folder/name for name in ("status.txt","TP0.manifest.json","TP0.raw_request.jsonl","TP0.schedule_batch.jsonl","autobench_trace.jsonl","autobench_trace.manifest.json","benchmark.jsonl")]
    selected += [root/"cases"/f"{case}.{suffix}" for suffix in ("server.command.txt","benchmark.command.txt","server.log","benchmark.log")]
before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in selected}
def text(path): return path.read_text(encoding="utf-8")
def load(path): return [json.loads(line) for line in text(path).splitlines() if line.strip()]
def argument(command,flag):
    positions=[i for i,v in enumerate(command) if v==flag]
    assert len(positions)==1, (flag,positions)
    return command[positions[0]+1]
assert text(root/"BASELINE_COLLECTION_COMPLETE").strip()=="complete"
metadata=dict(line.split("=",1) for line in text(root/"run_metadata.txt").splitlines() if "=" in line)
settings={
    "source_revision":"59799a368793b9f795baf59b067233a65ad8e38e",
    "model":"Qwen3-30B-A3B-Instruct-2507","seed":"202609284",
    "attention_backend":"triton","disable_overlap_schedule":"1","cuda_graph":"disabled",
    "context_length":"2048","mem_fraction_static":"0.90","max_running_requests":"32",
    "chunked_prefill_size":"512","random_range_ratio":"0","prompt_multiplier":"2",
    "warmup_requests":"4","train_forward_modes":"1,2",
}
for key,value in settings.items(): assert metadata[key]==value,(key,metadata[key])
assert set(metadata["requested_cases"].split(","))==set(expected)
assert set(path.name for path in (root/"cases").iterdir() if path.is_dir())==set(expected)
versions=dict(line.split("=",1) for line in text(root/"package_versions.txt").splitlines() if "=" in line)
for key,value in {"python":"3.10.21","torch":"2.9.1+cu126","torch_cuda":"12.6","sglang":"0.0.0","sglang-kernel":"0.4.6.post1+cu129","sgl_kernel_runtime_version":"0.4.6.post1","triton":"3.5.1"}.items():
    assert versions[key]==value,(key,versions[key])
assert Path(versions["sgl_kernel_file"]).parent.parent.name=="minimal-stage-v4"
ids=set(); prompts=set(); all_latencies=[]; phases={1:0,2:0}; cases={}
for name,(count,inbound,outbound,rate) in expected.items():
    folder=root/"cases"/name
    assert text(folder/"status.txt").strip()=="ok"
    server=shlex.split(text(root/"cases"/f"{name}.server.command.txt"))
    client=shlex.split(text(root/"cases"/f"{name}.benchmark.command.txt"))
    assert argument(server,"-m")=="sglang_simulator.baseline.launch_server"
    for flag,value in {"--device":"cuda","--tp-size":"1","--context-length":"2048","--mem-fraction-static":"0.90","--max-running-requests":"32","--chunked-prefill-size":"512","--attention-backend":"triton"}.items():
        assert argument(server,flag)==value,(name,flag)
    assert "--disable-cuda-graph" in server and "--disable-overlap-schedule" in server
    checkpoint=Path("/home/ms-admin/data/models/Qwen3-30B-A3B-Instruct-2507").resolve()
    assert Path(argument(server,"--model-path")).resolve()==checkpoint
    assert Path(argument(server,"--collection-output-dir")).resolve()==folder
    assert Path(server[server.index("-m")-1]).resolve()==Path(metadata["python"]).resolve()
    assert argument(client,"-m")=="sglang.benchmark.serving"
    assert Path(argument(client,"--model")).resolve()==checkpoint
    for flag,value in {"--backend":"sglang","--dataset-name":"random-ids","--num-prompts":str(count),"--seed":"202609284","--random-input-len":inbound,"--random-output-len":outbound,"--random-range-ratio":"0","--request-rate":rate,"--warmup-requests":"4"}.items():
        assert argument(client,flag)==value,(name,flag)
    assert "--profile" in client and "--flush-cache" in client and "--tokenize-prompt" not in client
    assert Path(argument(client,"--output-file")).resolve()==(folder/"benchmark.jsonl")
    manifest=json.loads(text(folder/"TP0.manifest.json"))
    trace_manifest=json.loads(text(folder/"autobench_trace.manifest.json"))
    assert manifest["batch_latency_unit"]=="seconds" and manifest["batch_timing"]=="device_synchronized"
    raw=load(folder/"TP0.raw_request.jsonl"); trace=load(folder/"autobench_trace.jsonl"); batches=load(folder/"TP0.schedule_batch.jsonl")
    benchmark=load(folder/"benchmark.jsonl")[-1]
    assert len(raw)==len(trace)==manifest["request_count"]==trace_manifest["request_count"]==benchmark["completed"]==count
    assert benchmark["server_info"]["device"]=="cuda"
    assert len(batches)==manifest["schedule_batch_count"]>0
    copies={item["metadata"]["source_rid"]:item for item in trace}
    assert len(copies)==count
    case_prompts=set()
    for item in raw:
        rid=item["rid"]; assert rid not in ids; ids.add(rid)
        copied=copies[rid]
        assert item["input_ids"]==copied["prompt"]
        assert len(item["input_ids"])==item["input_length"]==copied["prompt_len"]
        assert len(item["output_ids"])==item["output_length"]==copied["output_len"]
        assert math.isfinite(copied["timestamp"]) and copied["timestamp"]>=0
        digest=hashlib.sha256(json.dumps(item["input_ids"],separators=(",",":")).encode()).hexdigest()
        prompts.add(digest); case_prompts.add(digest)
    assert trace_manifest["input_tokens"]==sum(item["input_length"] for item in raw)
    assert trace_manifest["output_tokens"]==sum(item["output_length"] for item in raw)==benchmark["total_output_tokens"]
    latencies=[]; case_phases={1:0,2:0}; last_end=-math.inf
    for item in batches:
        latency=item["iter_latency"]; start=item["start_timestamp"]; end=item["end_timestamp"]
        assert all(math.isfinite(value) for value in (latency,start,end))
        assert latency>0 and end>start and start>=last_end
        assert math.isclose(latency,end-start,rel_tol=1e-12,abs_tol=1e-12)
        last_end=end
        mode=int(item["forward_mode"]); assert mode in (1,2)
        infos=item["request_infos"]; assert infos and len(infos)<=32
        members=[info["rid"] for info in infos]
        assert len(set(members))==len(members) and all(rid in copies for rid in members)
        for info in infos:
            for key in ("extend_input_len","prefix_indices_len","output_ids_len"):
                assert isinstance(info[key],int) and info[key]>=0,(name,key)
        if mode==2: assert all(info["extend_input_len"]==1 for info in infos)
        latencies.append(latency); case_phases[mode]+=1; phases[mode]+=1
    all_latencies += latencies
    cases[name]={"requests":count,"distinct_complete_inputs":len(case_prompts),"timing_rows":len(batches),"prefill_rows":case_phases[1],"decode_rows":case_phases[2],"mean_wall_ms":statistics.mean(latencies)*1000,"min_wall_ms":min(latencies)*1000,"max_wall_ms":max(latencies)*1000,"native_duration_seconds":benchmark["duration"]}
assert len(ids)==160 and len(all_latencies)==1375
after={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}
assert before==after, "Closed native files changed during the readback"
result={
    "model":"Qwen3-30B-A3B-Instruct-2507","stage":"training-repeat-5","seed":202609284,
    "requests":len(ids),"distinct_complete_inputs":len(prompts),"timing_rows":len(all_latencies),
    "prefill_rows":phases[1],"decode_rows":phases[2],"cases":cases,
    "source_revision":metadata["source_revision"],"runtime_versions":{key:versions[key] for key in ("python","torch","torch_cuda","sglang","sglang-kernel","sgl_kernel_runtime_version","triton")},
    "original_files_unchanged":len(before),"source_sha256":before,
    "started_unix":started,"finished_unix":time.time(),
    "environment":{key:os.environ.get(key) for key in ("PYTHONNOUSERSITE","CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS")},
    "scope":"Closed fifth training group: native/client settings, exact payload copies, case membership, synchronized finite-positive seconds, phase checks and trace counts. No fit/predict, held-out comparison, GPU work, new replay/AIC/profile or causal kernel-route attribution.",
}

import csv, io
calibration = root.parent.parent
case_folders = [calibration/"real"/f"repeat-{repeat}"/"cases"/name for repeat in range(1,6) for name in expected]
extra_paths = [folder/name for folder in case_folders for name in ("TP0.raw_request.jsonl","TP0.manifest.json")]
extra_paths += [calibration/"real"/f"repeat-{repeat}"/"BASELINE_COLLECTION_COMPLETE" for repeat in range(1,6)]
extra_paths += [calibration/"train.command.txt",calibration/"train.log"]
extra_paths += [calibration/"training"/"eval"/name for name in ("dataset_cases.csv","training_log.md","split_summary.csv","split_metrics_by_group.csv")]
bundle=calibration/"training"/"Qwen3-30B-A3B-Instruct-2507_hgbmono_p50.pkl"
extra_paths.append(bundle)
extra_before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in extra_paths}
assert extra_before[str(bundle)]=="5b13bad77cae8f3421c988eeeeba45ecbaadbffbbf6bf1d23f8dabd26a7620d8"
training_command=shlex.split(text(calibration/"train.command.txt"))
data_roots=[Path(training_command[index+1]).resolve() for index,value in enumerate(training_command) if value=="--data-root"]
assert data_roots==[(calibration/"real"/f"repeat-{repeat}"/"cases").resolve() for repeat in range(1,6)]
assert argument(training_command,"--forward-modes")=="1,2"
assert argument(training_command,"--test-fraction")=="0.2"
assert argument(training_command,"--loss")=="both"
assert argument(training_command,"--seed")=="202609280"
assert Path(argument(training_command,"--out-dir")).resolve()==calibration/"training"
for repeat in range(1,6): assert text(calibration/"real"/f"repeat-{repeat}"/"BASELINE_COLLECTION_COMPLETE").strip()=="complete"
global_ids=set(); global_prompts=set(); global_count=0; manifest_rows={}
for folder in case_folders:
    rows=load(folder/"TP0.raw_request.jsonl")
    manifest=json.loads(text(folder/"TP0.manifest.json"))
    assert len(rows)==manifest["request_count"]
    manifest_rows[str(folder)]=manifest["schedule_batch_count"]
    for row in rows:
        assert row["rid"] not in global_ids
        global_ids.add(row["rid"]); global_count+=1
        global_prompts.add(hashlib.sha256(json.dumps(row["input_ids"],separators=(",",":")).encode()).hexdigest())
dataset=list(csv.DictReader(io.StringIO(text(calibration/"training"/"eval"/"dataset_cases.csv"))))
assert len(dataset)==15 and len({row["case_id"] for row in dataset})==15
for row in dataset:
    folder=Path(row["case_id"]).resolve()
    assert str(folder) in manifest_rows
    assert Path(row["schedule_path"]).resolve()==folder/"TP0.schedule_batch.jsonl"
    assert int(row["rows"])==manifest_rows[str(folder)]
    assert all(int(row[key])==0 for key in ("skipped_mode","skipped_empty","skipped_latency"))
assert global_count==len(global_ids)==800
assert sum(manifest_rows.values())==sum(int(row["rows"]) for row in dataset)==6931
summary=list(csv.DictReader(io.StringIO(text(calibration/"training"/"eval"/"split_summary.csv"))))
assert len(summary)==2 and {row["loss"] for row in summary}=={"l2","p50"}
for row in summary:
    assert row["split"]=="case_stratified"
    assert int(row["train_rows"])==5545 and int(row["test_rows"])==1386
    assert int(row["train_rows"])+int(row["test_rows"])==6931
extra_after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in extra_paths}
assert extra_before==extra_after
result["all_five_training_groups"]={
    "native_requests":global_count,"distinct_request_ids":len(global_ids),
    "distinct_complete_inputs":len(global_prompts),"training_rows":sum(manifest_rows.values()),
    "training_case_count":len(dataset),"data_roots":[str(folder) for folder in data_roots],
    "schedule_rows_per_case":manifest_rows,"training_command":training_command,
    "p50_bundle_sha256":extra_before[str(bundle)],
    "internal_row_split_summary":summary,
    "extra_closed_source_files_unchanged":len(extra_before),"extra_source_sha256":extra_before,
    "accuracy_scope":"Internal case-stratified batch-row split is not independent-request accuracy. Full held-out native collection/scoring/replay remain unfinished."
}
result["finished_unix"]=time.time()

print(json.dumps(result,sort_keys=True))
