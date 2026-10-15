"""Read-only request-independence and native-timing audit for closed 32B holdout 1."""

import hashlib
import json
import math
import os
import shlex
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path("test-results/pr33824_expanded_all13_predictors_20260928_1933").resolve()
model = root / "models/Qwen3-32B"
training = model / "calibration/real"
group = model / "holdout/real/repeat-1"
cases = {
    "latency_256x128_r1": 32,
    "service_1024x256_r4": 64,
    "saturation_1024x256_inf": 64,
    "latency_128x64_r1": 32,
    "saturation_1536x384_inf": 64,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def input_digest(row):
    return hashlib.sha256(json.dumps(row["input_ids"], separators=(",", ":")).encode()).hexdigest()


train_paths = sorted(training.glob("repeat-*/cases/*/TP0.raw_request.jsonl"))
assert len(train_paths) == 15
selected = train_paths + [group / name for name in ("BASELINE_COLLECTION_COMPLETE", "run_metadata.txt", "package_versions.txt")]
for case in cases:
    directory = group / "cases" / case
    selected += [directory / name for name in ("status.txt", "TP0.raw_request.jsonl", "TP0.manifest.json", "TP0.schedule_batch.jsonl", "autobench_trace.jsonl", "benchmark.jsonl")]
    selected += [group / "cases" / (case + suffix) for suffix in (".server.command.txt", ".benchmark.command.txt")]
before = {str(path): digest(path) for path in selected}
train_ids = set()
train_inputs = set()
for path in train_paths:
    for row in records(path):
        assert row["rid"] not in train_ids
        train_ids.add(row["rid"])
        train_inputs.add(input_digest(row))
assert len(train_ids) == 800 and len(train_inputs) == 480
assert (group / "BASELINE_COLLECTION_COMPLETE").read_text().strip() == "complete"
metadata = dict(line.split("=", 1) for line in (group / "run_metadata.txt").read_text().splitlines() if "=" in line)
assert metadata["model"] == "Qwen3-32B" and metadata["seed"] == "202609380"
assert metadata["source_revision"] == "59799a368793b9f795baf59b067233a65ad8e38e"
assert metadata["mem_fraction_static"] == "0.93"
assert metadata["attention_backend"] == "triton"
assert metadata["cuda_graph"] == "disabled" and metadata["disable_overlap_schedule"] == "1"
assert set(metadata["requested_cases"].split(",")) == set(cases)
assert (group / "package_versions.txt").read_bytes() == (training / "repeat-1/package_versions.txt").read_bytes()
test_ids = set()
test_inputs = set()
per_case = {}
for case, expected in cases.items():
    directory = group / "cases" / case
    assert (directory / "status.txt").read_text().strip() == "ok"
    manifest = json.loads((directory / "TP0.manifest.json").read_text())
    raw = records(directory / "TP0.raw_request.jsonl")
    trace = records(directory / "autobench_trace.jsonl")
    batches = records(directory / "TP0.schedule_batch.jsonl")
    benchmark = records(directory / "benchmark.jsonl")[-1]
    assert len(raw) == len(trace) == manifest["request_count"] == benchmark["completed"] == expected
    assert len(batches) == manifest["schedule_batch_count"] > 0
    assert manifest["batch_timing"] == "device_synchronized" and manifest["batch_latency_unit"] == "seconds"
    assert benchmark["server_info"]["device"] == "cuda"
    server = shlex.split((group / "cases" / (case + ".server.command.txt")).read_text())
    client = shlex.split((group / "cases" / (case + ".benchmark.command.txt")).read_text())
    assert server[server.index("-m") + 1] == "sglang_simulator.baseline.launch_server"
    assert server[server.index("--device") + 1] == "cuda"
    assert server[server.index("--mem-fraction-static") + 1] == "0.93"
    assert "--disable-overlap-schedule" in server and "--disable-cuda-graph" in server
    assert Path(server[server.index("--model-path") + 1]).resolve() == Path("/home/ms-admin/data/models/Qwen3-32B").resolve()
    assert client[client.index("-m") + 1] == "sglang.benchmark.serving"
    assert client[client.index("--seed") + 1] == "202609380"
    copies = {row["metadata"]["source_rid"]: row for row in trace}
    assert len(copies) == expected
    local_ids = set()
    local_inputs = set()
    for row in raw:
        rid = row["rid"]
        assert rid not in test_ids
        test_ids.add(rid)
        local_ids.add(rid)
        prompt = input_digest(row)
        test_inputs.add(prompt)
        local_inputs.add(prompt)
        copied = copies[rid]
        assert row["input_ids"] == copied["prompt"]
        assert len(row["input_ids"]) == row["input_length"] == copied["prompt_len"]
        assert len(row["output_ids"]) == row["output_length"] == copied["output_len"]
    last_end = -math.inf
    prefill = decode = 0
    for row in batches:
        start, end, latency = row["start_timestamp"], row["end_timestamp"], row["iter_latency"]
        assert all(math.isfinite(value) for value in (start, end, latency))
        assert end > start >= last_end and latency > 0
        assert math.isclose(end - start, latency, rel_tol=1e-12, abs_tol=1e-12)
        last_end = end
        infos = row["request_infos"]
        assert infos and all(info["rid"] in local_ids for info in infos)
        if row["forward_mode"] == 1:
            prefill += 1
        else:
            assert row["forward_mode"] == 2
            decode += 1
    per_case[case] = {"requests": expected, "distinct_complete_inputs": len(local_inputs), "timing_rows": len(batches), "prefill_rows": prefill, "decode_rows": decode}
assert len(test_ids) == 256
assert not (train_ids & test_ids) and not (train_inputs & test_inputs)
after = {str(path): digest(path) for path in selected}
assert before == after
print(json.dumps({
    "model": "Qwen3-32B",
    "holdout_group": 1,
    "training_request_ids": len(train_ids),
    "training_distinct_complete_inputs": len(train_inputs),
    "holdout_request_ids": len(test_ids),
    "holdout_distinct_complete_inputs": len(test_inputs),
    "shared_request_ids": len(train_ids & test_ids),
    "shared_complete_inputs": len(train_inputs & test_inputs),
    "per_case": per_case,
    "source_files_unchanged": len(before),
    "source_manifest_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    "scope": "Only closed holdout repeat 1 vs all five training groups; no predictor query, fit, GPU work, AIC, or E2E accuracy claim.",
}, sort_keys=True))
