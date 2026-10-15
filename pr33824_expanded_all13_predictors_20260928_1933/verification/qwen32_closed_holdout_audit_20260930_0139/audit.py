"""Read-only Qwen3-32B closed-holdout provenance audit; no scoring or GPU work."""

import argparse
import hashlib
import json
import math
import os
import shlex
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
parser = argparse.ArgumentParser()
parser.add_argument("--through", type=int, required=True, choices=range(1, 6))
args = parser.parse_args()
root = Path("test-results/pr33824_expanded_all13_predictors_20260928_1933").resolve()
model = root / "models/Qwen3-32B"
checkpoint = Path("/home/ms-admin/data/models/Qwen3-32B").resolve()
training = model / "calibration/real"
cases = {
    "latency_256x128_r1": 32,
    "service_1024x256_r4": 64,
    "saturation_1024x256_inf": 64,
    "latency_128x64_r1": 32,
    "saturation_1536x384_inf": 64,
}


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_hash(row):
    return hashlib.sha256(json.dumps(row["input_ids"], separators=(",", ":")).encode()).hexdigest()


train_paths = sorted(training.glob("repeat-*/cases/*/TP0.raw_request.jsonl"))
assert len(train_paths) == 15
source_paths = list(train_paths)
for repeat in range(1, args.through + 1):
    group = model / f"holdout/real/repeat-{repeat}"
    source_paths += [group / name for name in ("BASELINE_COLLECTION_COMPLETE", "run_metadata.txt", "package_versions.txt")]
    for case in cases:
        directory = group / "cases" / case
        source_paths += [directory / name for name in ("status.txt", "TP0.raw_request.jsonl", "TP0.manifest.json", "TP0.schedule_batch.jsonl", "autobench_trace.jsonl", "benchmark.jsonl")]
        source_paths += [group / "cases" / (case + suffix) for suffix in (".server.command.txt", ".benchmark.command.txt")]
before = {str(path): digest(path) for path in source_paths}
train_ids = set()
train_inputs = set()
for path in train_paths:
    for row in lines(path):
        assert row["rid"] not in train_ids
        train_ids.add(row["rid"])
        train_inputs.add(prompt_hash(row))
assert len(train_ids) == 800 and len(train_inputs) == 480
train_package = (training / "repeat-1/package_versions.txt").read_bytes()
all_test_ids = set()
all_test_inputs = set()
groups = []
for repeat in range(1, args.through + 1):
    group = model / f"holdout/real/repeat-{repeat}"
    seed = str(202609379 + repeat)
    assert (group / "BASELINE_COLLECTION_COMPLETE").read_text().strip() == "complete"
    assert (group / "package_versions.txt").read_bytes() == train_package
    metadata = dict(line.split("=", 1) for line in (group / "run_metadata.txt").read_text().splitlines() if "=" in line)
    assert metadata["model"] == "Qwen3-32B" and metadata["seed"] == seed
    assert metadata["source_revision"] == "59799a368793b9f795baf59b067233a65ad8e38e"
    assert metadata["attention_backend"] == "triton" and metadata["mem_fraction_static"] == "0.93"
    assert metadata["cuda_graph"] == "disabled" and metadata["disable_overlap_schedule"] == "1"
    assert set(metadata["requested_cases"].split(",")) == set(cases)
    group_ids = set()
    group_inputs = set()
    per_case = {}
    for case, expected in cases.items():
        directory = group / "cases" / case
        assert (directory / "status.txt").read_text().strip() == "ok"
        manifest = json.loads((directory / "TP0.manifest.json").read_text())
        raw = lines(directory / "TP0.raw_request.jsonl")
        trace = lines(directory / "autobench_trace.jsonl")
        batch = lines(directory / "TP0.schedule_batch.jsonl")
        bench = lines(directory / "benchmark.jsonl")[-1]
        assert len(raw) == len(trace) == manifest["request_count"] == bench["completed"] == expected
        assert len(batch) == manifest["schedule_batch_count"] > 0
        assert manifest["batch_timing"] == "device_synchronized" and manifest["batch_latency_unit"] == "seconds"
        assert bench["server_info"]["device"] == "cuda"
        server = shlex.split((group / "cases" / (case + ".server.command.txt")).read_text())
        client = shlex.split((group / "cases" / (case + ".benchmark.command.txt")).read_text())
        assert server[server.index("-m") + 1] == "sglang_simulator.baseline.launch_server"
        assert server[server.index("--device") + 1] == "cuda"
        assert server[server.index("--mem-fraction-static") + 1] == "0.93"
        assert "--disable-overlap-schedule" in server and "--disable-cuda-graph" in server
        assert Path(server[server.index("--model-path") + 1]).resolve() == checkpoint
        assert client[client.index("-m") + 1] == "sglang.benchmark.serving"
        assert client[client.index("--seed") + 1] == seed
        copies = {row["metadata"]["source_rid"]: row for row in trace}
        assert len(copies) == expected
        case_ids = set()
        case_inputs = set()
        for row in raw:
            rid = row["rid"]
            assert rid not in group_ids
            group_ids.add(rid)
            case_ids.add(rid)
            prompt = prompt_hash(row)
            group_inputs.add(prompt)
            case_inputs.add(prompt)
            copy = copies[rid]
            assert row["input_ids"] == copy["prompt"]
            assert len(row["input_ids"]) == row["input_length"] == copy["prompt_len"]
            assert len(row["output_ids"]) == row["output_length"] == copy["output_len"]
        last_end = -math.inf
        prefill = decode = 0
        for row in batch:
            start, end, duration = row["start_timestamp"], row["end_timestamp"], row["iter_latency"]
            assert all(math.isfinite(value) for value in (start, end, duration))
            assert end > start >= last_end and duration > 0
            assert math.isclose(end - start, duration, rel_tol=1e-12, abs_tol=1e-12)
            last_end = end
            infos = row["request_infos"]
            assert infos and all(info["rid"] in case_ids for info in infos)
            if row["forward_mode"] == 1:
                prefill += 1
            else:
                assert row["forward_mode"] == 2
                decode += 1
        per_case[case] = {"requests": expected, "distinct_complete_inputs": len(case_inputs), "timing_rows": len(batch), "prefill_rows": prefill, "decode_rows": decode}
    assert len(group_ids) == 256
    assert not (group_ids & train_ids) and not (group_inputs & train_inputs)
    assert not (group_ids & all_test_ids) and not (group_inputs & all_test_inputs)
    all_test_ids |= group_ids
    all_test_inputs |= group_inputs
    groups.append({"repeat": repeat, "requests": len(group_ids), "distinct_complete_inputs": len(group_inputs), "timing_rows": sum(row["timing_rows"] for row in per_case.values()), "cases": per_case})
after = {str(path): digest(path) for path in source_paths}
assert before == after
print(json.dumps({
    "model": "Qwen3-32B",
    "through_repeat": args.through,
    "training_request_ids": len(train_ids),
    "training_distinct_complete_inputs": len(train_inputs),
    "heldout_request_ids": len(all_test_ids),
    "heldout_distinct_complete_inputs": len(all_test_inputs),
    "heldout_timing_rows": sum(group["timing_rows"] for group in groups),
    "shared_train_request_ids": len(all_test_ids & train_ids),
    "shared_train_complete_inputs": len(all_test_inputs & train_inputs),
    "groups": groups,
    "source_files_unchanged": len(before),
    "source_manifest_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    "scope": "Closed native holdout groups through requested repeat versus all training groups; no fit, predict, GPU work, AIC query, E2E or final-score claim.",
}, sort_keys=True))
