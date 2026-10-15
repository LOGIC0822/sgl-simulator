"""Validate isolated diagnostic plans/gates/ownership; never launch CUDA."""
import importlib.util
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest


SIM = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("matched_profile_runner", SIM / "tools/run_pr33824_matched_native_profiles.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(tmp_path):
    case = "saturation_1536x384_inf"
    model = runner.MODELS[0]
    source = tmp_path / "models" / model / "holdout/real/repeat-1/cases"
    source.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    metadata = {"python": "/fixture/python", "kernel_compat_stage": str(tmp_path / "kernel")}
    imports = ":".join((metadata["kernel_compat_stage"], str(SIM.parents[1] / "python"), str(SIM / "src")))
    server = ["env", "-u", "SGLANG_USE_CPU_ENGINE", f"PYTHONPATH={imports}", "LD_LIBRARY_PATH=/fixture/lib",
              metadata["python"], "-m", "sglang_simulator.baseline.launch_server", "--model-path", str(checkpoint),
              "--collection-output-dir", str(source / case), "--host", "127.0.0.1", "--port", "30227",
              "--device", "cuda", "--tp-size", "1", "--context-length", "2048", "--max-running-requests", "32",
              "--chunked-prefill-size", "512", "--attention-backend", "triton", "--mem-fraction-static", ".8",
              "--disable-cuda-graph", "--disable-overlap-schedule"]
    client = [metadata["python"], "-m", "sglang.benchmark.serving", "--model", str(checkpoint), "--backend", "sglang",
              "--base-url", "http://127.0.0.1:30227", "--output-file", str(source / case / "benchmark.jsonl"),
              "--seed", "202609380", "--dataset-name", "random-ids", "--random-range-ratio", "0", "--num-prompts", "64",
              "--random-input-len", "1536", "--random-output-len", "384", "--request-rate", "inf",
              "--profile", "--tokenize-prompt"]
    server_path, client_path = source / f"{case}.server.command.txt", source / f"{case}.benchmark.command.txt"
    server_path.write_text(shlex.join(server))
    client_path.write_text(shlex.join(client))
    return model, case, checkpoint, metadata, server_path, client_path, server, client


def test_recorded_plan_preserves_workload_and_changes_only_diagnostic_paths(tmp_path):
    model, case, checkpoint, metadata, server_path, client_path, _, original_client = fixture(tmp_path)
    before = server_path.read_bytes(), client_path.read_bytes()
    target = tmp_path / "diagnostic"
    server, client, env, sha = runner.plan_case(tmp_path, model, case, target, 40001, metadata, checkpoint,
        {"SGLANG_FORCE_FUSED_OP_BACKEND": "torch", "SGLANG_USE_CPU_ENGINE": "1", "SGLANG_SIMULATOR_CONFIG_PATH": "wrong"})
    assert server[:3] == ["/fixture/python", "-m", runner.MODULE]
    assert runner.one_value(server, "--mem-fraction-static") == ".8"
    expected_client = list(original_client)
    runner.replace_value(expected_client, "--base-url", "http://127.0.0.1:40001")
    runner.replace_value(expected_client, "--output-file", str(target / "benchmark.jsonl"))
    assert client == expected_client and "--tokenize-prompt" in client
    assert runner.one_value(server, "--native-profile-dir") == str(target / "profiles")
    assert env["OMP_NUM_THREADS"] == "1" and "SGLANG_USE_CPU_ENGINE" not in env
    assert "SGLANG_FORCE_FUSED_OP_BACKEND" not in env and "SGLANG_SIMULATOR_CONFIG_PATH" not in env
    assert env["SGLANG_CACHE_DIR"] == str(target / "cache")
    assert len(sha) == 2 and (server_path.read_bytes(), client_path.read_bytes()) == before
    assert not target.exists()


@pytest.mark.parametrize("kind", ["module", "checkpoint", "backend", "graph", "imports", "output", "env", "url"])
def test_recorded_plan_rejects_nonmatching_baseline(tmp_path, kind):
    model, case, checkpoint, metadata, server_path, client_path, server, client = fixture(tmp_path)
    if kind == "module":
        server[server.index("-m")+1] = "sglang_simulator.simulation.sglang.launch_server"
    elif kind == "checkpoint":
        runner.replace_value(server, "--model-path", "/different/checkpoint")
    elif kind == "backend":
        runner.replace_value(server, "--attention-backend", "flashinfer")
    elif kind == "graph":
        server.remove("--disable-cuda-graph")
    elif kind == "imports":
        server[server.index(next(value for value in server if value.startswith("PYTHONPATH=")))] = "PYTHONPATH=/wrong/kernel"
    elif kind == "output":
        runner.replace_value(client, "--output-file", "/outside/results")
    elif kind == "env":
        server.insert(1, "UNKNOWN_SETTING=1")
    else:
        runner.replace_value(client, "--base-url", "http://127.0.0.1:1")
    server_path.write_text(shlex.join(server))
    client_path.write_text(shlex.join(client))
    with pytest.raises(ValueError):
        runner.plan_case(tmp_path, model, case, tmp_path / "diagnostic", 40001, metadata, checkpoint, {})


def test_gate_requires_native_aic_and_existing_historical_cpu_jobs_complete(tmp_path):
    with pytest.raises(ValueError, match="formal collection"):
        runner.require_complete(tmp_path)
    (tmp_path / "REAL_ML_ALL_MODELS_COMPLETE").write_text("complete\n")
    with pytest.raises(ValueError, match="formal collection"):
        runner.require_complete(tmp_path)
    (tmp_path / "AIC_ALL_MODELS_COMPLETE").write_text("complete\n")
    previous = tmp_path / "previous_calibration"
    previous.mkdir()
    (previous / "reference.json").write_text("{}")
    with pytest.raises(ValueError, match="historical CPU"):
        runner.require_complete(tmp_path)
    (previous / "PREVIOUS_CALIBRATION_COMPLETE").write_text("complete\n")
    runner.require_complete(tmp_path)


def test_cleanup_uses_only_captured_owned_process_handles(monkeypatch):
    signals, checked = [], []
    class Handle:
        def __init__(self, pid):
            self.pid = pid
        def create_time(self):
            checked.append(self.pid)
            return 123
        def is_running(self):
            return True
        def children(self, recursive):
            assert recursive
            return [child] if self.pid == 10 else []
        def terminate(self):
            signals.append(("term", self.pid))
        def kill(self):
            signals.append(("kill", self.pid))
    child, parent = Handle(11), Handle(10)
    monkeypatch.setattr(runner.psutil, "Process", lambda pid: parent if pid == 10 else pytest.fail("unowned PID"))
    waits = []
    def wait(handles, timeout):
        waits.append([value.pid for value in handles])
        for value in handles:
            value.returncode = -15
        return handles, []
    monkeypatch.setattr(runner.psutil, "wait_procs", wait)
    process = SimpleNamespace(pid=10, wait=lambda timeout: 0)
    owner = runner.OwnedTree(process, ["/fixture/python", "-m", runner.MODULE])
    owner.close()
    assert set(signals) == {("term", 10), ("term", 11)}
    assert waits[0] == [10, 11] and 11 in checked
    assert process.returncode == -15
    with pytest.raises(ValueError, match="cleanup ownership"):
        runner.OwnedTree(process, ["/fixture/python", "-m", "unrelated.module"])


@pytest.mark.parametrize("kind", ["ok", "short_client", "duplicate", "unfinished", "foreign_batch"])
def test_completed_request_count_is_checked_independently_of_exit_code(tmp_path, kind):
    folder = tmp_path / "collection"
    folder.mkdir()
    (folder / "TP0.manifest.json").write_text(json.dumps({"rank_prefix": "TP0", "batch_latency_unit": "seconds",
        "batch_timing": "device_synchronized", "request_count": 2, "schedule_batch_count": 1}))
    (tmp_path / "benchmark.jsonl").write_text(json.dumps({"completed": 1 if kind == "short_client" else 2})+"\n")
    raw = [{"rid": "1", "input_ids": [1], "output_ids": [2]}, {"rid": "2", "input_ids": [1], "output_ids": [2]}]
    if kind == "duplicate":
        raw[1]["rid"] = "1"
    elif kind == "unfinished":
        raw[1]["output_ids"] = []
    (folder / "TP0.raw_request.jsonl").write_text("".join(json.dumps(row)+"\n" for row in raw))
    (folder / "TP0.schedule_batch.jsonl").write_text(json.dumps({"request_infos": [{"rid": "foreign" if kind == "foreign_batch" else "1"}]})+"\n")
    if kind == "ok":
        assert runner.verify_collection(tmp_path, 2) == {"completed_requests": 2, "schedule_batch_count": 1}
    else:
        with pytest.raises(ValueError):
            runner.verify_collection(tmp_path, 2)
