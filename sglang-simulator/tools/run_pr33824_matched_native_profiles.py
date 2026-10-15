#!/usr/bin/env python3
"""Run bounded, separately labelled CUDA diagnostics after the formal matrix.

Recorded native server/client commands are replayed without shell evaluation.
Only launcher, diagnostic output/cache paths and port change. Profiling can
alter wall times and scheduling; these runs never replace formal baselines.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request

import psutil


MODELS = ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-32B")
MODULE = "sglang_simulator.baseline.profiled_launcher"
SIM = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_complete(root: Path) -> None:
    for marker in (root / "REAL_ML_ALL_MODELS_COMPLETE", root / "AIC_ALL_MODELS_COMPLETE"):
        if not marker.is_file() or marker.read_text().strip() != "complete":
            raise ValueError(f"formal collection/scoring must finish first: {marker}")
    previous = root / "previous_calibration"
    if (previous / "reference.json").is_file():
        marker = previous / "PREVIOUS_CALIBRATION_COMPLETE"
        if not marker.is_file() or marker.read_text().strip() != "complete":
            raise ValueError("wait for historical CPU comparison before GPU profiling")


def one_value(command: list[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"require exactly one recorded {flag}")
    index = command.index(flag)
    if index+1 == len(command) or command[index+1].startswith("--"):
        raise ValueError(f"missing recorded value for {flag}")
    return command[index+1]


def replace_value(command: list[str], flag: str, value: str) -> None:
    one_value(command, flag)
    command[command.index(flag)+1] = value


def recorded_server(path: Path, inherited: dict) -> tuple[list[str], dict]:
    tokens = shlex.split(path.read_text())
    if not tokens or tokens.pop(0) != "env":
        raise ValueError("expected a recorded env invocation, not a shell command")
    env = dict(inherited)
    allowed = {"SGLANG_SIMULATOR_CONFIG_PATH", "SGLANG_SIMULATOR_OUTPUT_DIR", "SGLANG_USE_CPU_ENGINE"}
    while tokens and tokens[0] == "-u":
        if len(tokens) < 2 or tokens[1] not in allowed:
            raise ValueError("unexpected recorded env unset")
        env.pop(tokens[1], None)
        tokens = tokens[2:]
    assignments = {"SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "PYTHONPATH", "LD_LIBRARY_PATH"}
    while tokens and "=" in tokens[0] and not tokens[0].startswith("/"):
        name, value = tokens.pop(0).split("=", 1)
        if name not in assignments:
            raise ValueError(f"unexpected recorded environment assignment: {name}")
        env[name] = value
    for name in allowed | {"SGLANG_FORCE_FUSED_OP_BACKEND"}:
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    return tokens, env


def plan_case(root: Path, model: str, case: str, target: Path, port: int,
              metadata: dict, model_path: Path, inherited: dict,
              steps: int = 3, skip: int = 2) -> tuple[list[str], list[str], dict, dict]:
    source = root / "models" / model / "holdout/real/repeat-1/cases"
    server_path, client_path = source / f"{case}.server.command.txt", source / f"{case}.benchmark.command.txt"
    server, env = recorded_server(server_path, inherited)
    client = shlex.split(client_path.read_text())
    python = metadata["python"]
    if server[:3] != [python, "-m", "sglang_simulator.baseline.launch_server"] or client[:3] != [python, "-m", "sglang.benchmark.serving"]:
        raise ValueError("recorded executable/module does not match the formal native baseline")
    if Path(one_value(server, "--model-path")).resolve() != model_path.resolve() or Path(one_value(client, "--model")).resolve() != model_path.resolve():
        raise ValueError("server/client checkpoint differs from the matrix manifest")
    expected = {"--device": "cuda", "--tp-size": "1", "--context-length": "2048",
                "--max-running-requests": "32", "--chunked-prefill-size": "512", "--attention-backend": "triton", "--host": "127.0.0.1"}
    if any(one_value(server, flag) != value for flag, value in expected.items()):
        raise ValueError("recorded native settings differ from the predeclared dense diagnostic")
    for flag in ("--disable-cuda-graph", "--disable-overlap-schedule"):
        if server.count(flag) != 1:
            raise ValueError(f"explicit native {flag} required")
    if "--cuda-graph-config" in server or "--profile" not in client:
        raise ValueError("graph override or missing collection activation")
    if one_value(client, "--backend") != "sglang" or one_value(client, "--dataset-name") != "random-ids" or one_value(client, "--random-range-ratio") != "0":
        raise ValueError("recorded client is not the formal random-ID native workload")
    if "heldout_seeds" in metadata and one_value(client, "--seed") != metadata["heldout_seeds"].split(",")[0]:
        raise ValueError("profile request seed differs from formal holdout group 1")
    if one_value(client, "--base-url") != f"http://127.0.0.1:{one_value(server, '--port')}":
        raise ValueError("recorded client/server address mismatch")
    if Path(one_value(server, "--collection-output-dir")).resolve() != (source / case).resolve() or Path(one_value(client, "--output-file")).resolve() != (source / case / "benchmark.jsonl").resolve():
        raise ValueError("recorded formal output paths differ from their own case")
    stage = Path(metadata["kernel_compat_stage"]).resolve()
    paths = [Path(part).resolve() for part in env["PYTHONPATH"].split(":") if part]
    if paths[:3] != [stage, SIM.parents[1] / "python", SIM / "src"]:
        raise ValueError("recorded native import paths do not use the formal source and kernel")
    env.update(SGLANG_CACHE_DIR=str(target / "cache"), SGLANG_JIT_CACHE_DIR=str(root / "shared/jit-cache"),
               FLASHINFER_WORKSPACE_BASE=str(target / "flashinfer-workspace"))
    server[2] = MODULE
    server.extend(["--native-profile-dir", str(target / "profiles"),
                   "--native-profile-steps-per-phase", str(steps), "--native-profile-skip-per-phase", str(skip)])
    for flag, value in (("--port", str(port)), ("--collection-output-dir", str(target / "collection"))):
        replace_value(server, flag, value)
    replace_value(client, "--base-url", f"http://127.0.0.1:{port}")
    replace_value(client, "--output-file", str(target / "benchmark.jsonl"))
    return server, client, env, {str(path.resolve()): digest(path) for path in (server_path, client_path)}


class OwnedTree:
    """Retain psutil handles with create-time checks; never target other servers."""
    def __init__(self, process: subprocess.Popen, command: list[str]) -> None:
        if command[1:3] != ["-m", MODULE]:
            raise ValueError("cleanup ownership requires the diagnostic launcher")
        self.process = process
        self.parent = psutil.Process(process.pid)
        self.handles = {self.parent.pid: self.parent}
        self.capture()

    def capture(self) -> None:
        try:
            if not self.parent.is_running():
                return
            for process in self.parent.children(recursive=True):
                # Cache create_time before any later PID reuse.
                process.create_time()
                self.handles[process.pid] = process
        except psutil.NoSuchProcess:
            pass

    def owns_listener(self, port: int) -> bool:
        self.capture()
        for process in self.handles.values():
            try:
                if process.is_running() and any(connection.status == psutil.CONN_LISTEN
                        and connection.laddr.port == port for connection in process.net_connections(kind="inet")):
                    return True
            except psutil.NoSuchProcess:
                pass
        return False

    def close(self) -> None:
        self.capture()
        processes = list(self.handles.values())
        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=10)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=5)
        # wait_procs may reap our direct child first. Preserve that real signal
        # exit code instead of letting Popen's ECHILD path report a false zero.
        parent_exit = getattr(self.parent, "returncode", None)
        if isinstance(parent_exit, int):
            self.process.returncode = parent_exit
        self.process.wait(timeout=5)
        if alive:
            raise RuntimeError(f"diagnostic descendants still live: {[value.pid for value in alive]}")


def wait_ready(process: subprocess.Popen, owner: OwnedTree, port: int, timeout: float = 600) -> None:
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        owner.capture()
        if process.poll() is not None:
            raise RuntimeError(f"profile server exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    if not owner.owns_listener(port):
                        raise RuntimeError("health endpoint is not owned by this diagnostic process tree")
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("profile server readiness timeout; inspect its own log")


def verify_collection(target: Path, expected: int) -> dict:
    """A successful client exit alone does not prove all requests completed."""
    folder = target / "collection"
    manifest = json.loads((folder / "TP0.manifest.json").read_text())
    benchmark = json.loads((target / "benchmark.jsonl").read_text().splitlines()[-1])
    raw = [json.loads(line) for line in (folder / "TP0.raw_request.jsonl").read_text().splitlines()]
    batches = [json.loads(line) for line in (folder / "TP0.schedule_batch.jsonl").read_text().splitlines()]
    ids = {str(row["rid"]) for row in raw}
    if (manifest["rank_prefix"] != "TP0" or manifest["batch_latency_unit"] != "seconds"
            or manifest["batch_timing"] != "device_synchronized"
            or manifest["request_count"] != expected or int(benchmark["completed"]) != expected
            or len(raw) != expected or len(ids) != expected
            or manifest["schedule_batch_count"] != len(batches) or not batches):
        raise ValueError("diagnostic request completion/count/collection contract differs")
    if any(not row["input_ids"] or not row["output_ids"] for row in raw):
        raise ValueError("diagnostic contains unfinished native requests")
    if any(str(req["rid"]) not in ids for batch in batches for req in batch["request_infos"]):
        raise ValueError("diagnostic batches refer to requests outside their own collection")
    return {"completed_requests": expected, "schedule_batch_count": len(batches)}


def run_case(root: Path, model: str, case: str, target: Path, metadata: dict,
             model_path: Path, inherited: dict, uuid: str, steps: int, skip: int) -> dict:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server, client, env, sources = plan_case(root, model, case, target, port, metadata, model_path, inherited, steps, skip)
    env["CUDA_VISIBLE_DEVICES"] = uuid
    env["PATH"] = str(Path(metadata["python"]).parent)+":"+env.get("PATH", "")
    target.mkdir(parents=True, exist_ok=False)
    config = {"server": server, "client": client, "environment": {name: env[name] for name in
              ("CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "OMP_NUM_THREADS", "PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE")},
              "formal_command_sha256": sources, "model": model, "case": case,
              "scope": "Separate instrumented diagnostic; formal baselines and ML/AIC training are unchanged."}
    (target / "commands.json").write_text(json.dumps(config, indent=2))
    with (target / "server.log").open("x") as log:
        process = subprocess.Popen(server, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owner = OwnedTree(process, server)
        config["server_owner"] = {"pid": process.pid, "create_time": owner.parent.create_time()}
        (target / "owner.json").write_text(json.dumps(config["server_owner"]))
        try:
            wait_ready(process, owner, port)
            with (target / "benchmark.log").open("x") as client_log:
                result = subprocess.run(client, env=env, stdout=client_log, stderr=subprocess.STDOUT, timeout=1800)
            config["benchmark_exit_code"] = result.returncode
            if result.returncode != 0:
                raise RuntimeError(f"profile client failed, exit={result.returncode}: {target}")
        finally:
            owner.close()
            config["server_exit_code_after_owned_cleanup"] = process.returncode
            (target / "commands.json").write_text(json.dumps(config, indent=2))
    records = sorted((target / "profiles").glob("*.metadata.json"))
    if len(records) != 2*steps or any(sum(json.loads(path.read_text())["forward_mode"] == mode for path in records) != steps for mode in (1, 2)):
        raise ValueError(f"incomplete per-phase profiling coverage: {target}")
    schedule = target / "collection/TP0.schedule_batch.jsonl"
    if not schedule.is_file():
        raise ValueError("diagnostic did not export its own native collection")
    config["verified_collection"] = verify_collection(target, int(one_value(client, "--num-prompts")))
    (target / "commands.json").write_text(json.dumps(config, indent=2))
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise ValueError("original formal commands changed during diagnostic")
    (target / "PROFILE_COLLECTION_COMPLETE").write_text("complete\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps-per-phase", type=int, default=3)
    parser.add_argument("--skip-per-phase", type=int, default=2)
    args = parser.parse_args()
    root, out = args.results_root.resolve(), args.out_dir.resolve()
    require_complete(root)
    if out.exists() or args.steps_per_phase < 1 or args.skip_per_phase < 0:
        parser.error("require a new output directory and valid bounded profile counts")
    metadata = dict(line.split("=", 1) for line in (root / "run_metadata.txt").read_text().splitlines() if "=" in line)
    with (root / "model_manifest.tsv").open() as source:
        entries = list(csv.DictReader(source, delimiter="\t"))
    paths = {row["model"]: Path(row["model_path"]) for row in entries}
    if len(paths) != len(entries) or not set(MODELS) <= set(paths):
        raise ValueError("matched dense profiles require the original three model manifests")
    cases = metadata["heldout_case"].split(",")
    expected_cases = {"latency_256x128_r1", "service_1024x256_r4", "saturation_1024x256_inf",
                      "latency_128x64_r1", "saturation_1536x384_inf"}
    if len(cases) != 5 or set(cases) != expected_cases:
        raise ValueError("require all five predeclared formal workloads")
    stage = Path(metadata["kernel_compat_stage"])
    kernels = {str(path.resolve()): digest(path) for path in stage.rglob("*.so")}
    reference = json.loads((root / "provenance/formal_start_after_ninja_fix/sha256.json").read_text())
    expected = {path: value for path, value in reference.items() if path.startswith("/") and path.endswith(".so")}
    if not expected or kernels != expected:
        raise ValueError("native kernel binary differs from formal collection")
    runtime_sources = {str(SIM.parents[1] / path): checksum for path, checksum in reference.items()
                       if path.startswith(("python/sglang/", "tools/sglang-simulator/src/"))}
    if not runtime_sources or any(digest(Path(path)) != checksum for path, checksum in runtime_sources.items()):
        raise ValueError("existing native/simulator runtime sources differ from formal collection")
    identity = json.loads((root / "provenance/formal_start_after_ninja_fix/identity.json").read_text())
    _, _, first_env, _ = plan_case(root, MODELS[0], cases[0], out / MODELS[0] / cases[0], 1, metadata, paths[MODELS[0]], dict(os.environ), args.steps_per_phase, args.skip_per_phase)
    out.mkdir(parents=True)
    capture = [metadata["python"], str(SIM / "tools/capture_pr33824_experiment_provenance.py"),
               "--sim-root", str(SIM), "--aic-source", identity["aic_root"], "--kernel-stage", str(stage),
               "--out-dir", str(out / "provenance"), "--reference-kernel-checksums", str(root / "provenance/formal_start_after_ninja_fix/sha256.json"), "--require-stage-import"]
    cpu_env = dict(first_env, CUDA_VISIBLE_DEVICES="")
    cpu_env["PYTHONPATH"] += ":"+str(Path(identity["aic_root"]) / "src")
    (out / "provenance.command.json").write_text(json.dumps({"command": capture,
        "environment": {key: cpu_env[key] for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONNOUSERSITE", "OMP_NUM_THREADS")}}, indent=2))
    with (out / "provenance.log").open("x") as log:
        subprocess.run(capture, env=cpu_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    diagnostic = json.loads((out / "provenance/identity.json").read_text())
    if diagnostic["source_revision"] != identity["source_revision"] or any(diagnostic["distributions"].get(key) != identity["distributions"].get(key) for key in ("torch", "transformers", "triton", "flashinfer-python")):
        raise ValueError("diagnostic software differs from archived formal environment")
    if digest(out / "provenance/tracked_changes.patch") != digest(root / "provenance/formal_start_after_ninja_fix/tracked_changes.patch"):
        raise ValueError("tracked native/simulator source patch differs from formal collection")
    diagnostic_reference = json.loads((out / "provenance/sha256.json").read_text())
    for relative in ("src/sglang_simulator/baseline/scheduler_profile.py", "src/sglang_simulator/baseline/profiled_launcher.py",
                     "tools/run_pr33824_matched_native_profiles.py", "tools/analyze_pr33824_native_profile_sample.py",
                     "tools/score_pr33824_native_profile_aic.py"):
        key = "tools/sglang-simulator/"+relative
        runtime_sources[str(SIM / relative)] = diagnostic_reference[key]
    spec = importlib.util.spec_from_file_location("profile_capacity", SIM / "tools/wait_pr33824_gpu_capacity.py")
    capacity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capacity)
    configs = []
    for model in MODELS:
        gpu_csv = root / "models" / model / "holdout/real/repeat-1/hardware/gpu_identity.csv"
        cards = list(csv.reader(gpu_csv.read_text().splitlines()))
        if len(cards) != 1:
            raise ValueError("require a single archived native GPU")
        uuid = cards[0][1].strip()
        if capacity.snapshot(0)["uuid"] != uuid:
            raise ValueError("GPU differs from the formal real baseline")
        for case in cases:
            server, _, _, _ = plan_case(root, model, case, out / model / case, 1, metadata, paths[model], dict(os.environ), args.steps_per_phase, args.skip_per_phase)
            original_fraction = next(row["mem_fraction_static"] for row in entries if row["model"] == model)
            if float(one_value(server, "--mem-fraction-static")) != float(original_fraction):
                raise ValueError("recorded memory fraction differs from the formal model manifest")
            wait_command = [metadata["python"], str(SIM / "tools/wait_pr33824_gpu_capacity.py"), "--fraction", ".5", "--model", model,
                            "--interval", "600", "--log", str(out / f"{model}.{case}.capacity.jsonl")]
            subprocess.run(wait_command, env=cpu_env, check=True)
            configs.append(run_case(root, model, case, out / model / case, metadata, paths[model], dict(os.environ), uuid, args.steps_per_phase, args.skip_per_phase))
            print(f"Profile collection complete: {model}/{case}", flush=True)
    if any(digest(Path(path)) != value for path, value in {**kernels, **runtime_sources}.items()):
        raise ValueError("native kernel or runtime source changed during profiling")
    (out / "collection_manifest.json").write_text(json.dumps({"configs": configs, "kernel_sha256": kernels,
        "runtime_source_sha256": runtime_sources,
        "formal_root": str(root), "steps_per_phase": args.steps_per_phase, "skip_per_phase": args.skip_per_phase,
        "scope": "Three dense models, five workloads, one instrumented collection each; not five-repeat formal baselines."}, indent=2))
    score = [metadata["python"], str(SIM / "tools/score_pr33824_native_profile_aic.py"), "--results-root", str(root), "--profile-dir", str(out), "--out-dir", str(out / "analysis")]
    (out / "cpu_scoring.command.json").write_text(json.dumps({"command": score, "environment": {key: cpu_env[key] for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONNOUSERSITE", "OMP_NUM_THREADS")}}, indent=2))
    with (out / "cpu_scoring.log").open("x") as log:
        subprocess.run(score, env=cpu_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (out / "MATCHED_NATIVE_PROFILE_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
