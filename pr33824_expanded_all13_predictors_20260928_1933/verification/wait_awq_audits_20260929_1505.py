"""Wait for closed Meta-Llama-3.1-70B-Instruct-AWQ results and run the three prepared CPU readbacks.

This auxiliary process does not collect requests, fit predictors, replay traces,
query AIC, start GPU jobs or declare the all-thirteen-model experiment complete.
The original matrix owns its experiment. Failed audits are retained, not retried
over their existing output files.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

MODEL = "Meta-Llama-3.1-70B-Instruct-AWQ"
RUN = "pr33824_expanded_all13_predictors_20260928_1933"
CASES = (
    "latency_256x128_r1", "service_1024x256_r4", "saturation_1024x256_inf",
    "latency_128x64_r1", "saturation_1536x384_inf",
)
CALIBRATION = "awq_completed_calibration_20260929_1449"
STAGES = (
    ("saved_score", "awq_full_holdout_saved_score_20260929_1505", "audit.py"),
    ("frozen_fit_input", "frozen_fit_input_awq_20260929_1505", "readback.py"),
    ("endpoints", "awq_full_e2e_score_integrity_20260929_1505", "audit.py"),
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def text_is(path: Path, expected: str) -> bool:
    return path.is_file() and path.read_text().strip() == expected


def successful_audit(directory: Path) -> bool:
    """Accept only successful model-bound evidence for its declared stage."""
    execution = directory / "execution.command.json"
    manifest = directory / "results/manifest.json"
    if not execution.is_file() or not manifest.is_file():
        return False
    completed = read_json(execution)
    result = read_json(manifest)
    if completed.get("exit_code") != 0 or result.get("model") != MODEL:
        return False
    bundle = "5f212c17d088c78a916801fecd7cbc1634418413ed1c3cb60694565ea09301a7"
    if directory.name == CALIBRATION:
        return (result.get("p50_bundle_sha256") == bundle
                and result.get("training_requests") == 800
                and result.get("timing_rows") == 7069
                and result.get("training_cases") == 15)
    if directory.name == STAGES[0][1]:
        return (result.get("bundle_sha256") == bundle
                and result.get("training_requests") == 800
                and result.get("independent_test_requests") == 1280
                and result.get("actual_replay_payloads_verified") == 1280
                and result.get("native_case_command_pairs_verified") == 25
                and result.get("saved_summary_groups_verified") == 18
                and type(result.get("test_timing_rows")) is int
                and result["test_timing_rows"] > 0
                and result.get("raw_score_identity_rows_verified") == result["test_timing_rows"]
                and result.get("training_prompt_overlap") == 0
                and result.get("reused_request_ids") == 0
                and (directory / "results/source_sha256.csv").is_file())
    if directory.name == STAGES[1][1]:
        return (result.get("bundle_sha256") == bundle
                and result.get("training_requests") == 800
                and result.get("independent_test_requests") == 1280
                and result.get("training_rows") == 7069
                and result.get("fit_invoked") is False
                and type(result.get("independent_test_rows")) is int
                and result["independent_test_rows"] > 0
                and (directory / "results/source_sha256.csv").is_file())
    if directory.name == STAGES[2][1]:
        return (result.get("bundle_sha256") == bundle
                and result.get("paired_cases") == 25
                and result.get("paired_metric_rows") == 100
                and result.get("requests_verified") == 1280
                and result.get("matched_native_replay_capacity_cases") == 25
                and result.get("fit_gpu_or_latency_adjustment_invoked") is False
                and (directory / "results/source_sha256.csv").is_file())
    return False


def latest_status(root: Path, stage: str) -> str | None:
    path = root / "status.tsv"
    if not path.is_file():
        return None
    latest = None
    with path.open(newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) >= 4 and row[1:3] == [MODEL, stage]:
                latest = row[3]
    return latest


def native_complete(root: Path) -> bool:
    model = root / "models" / MODEL
    if not successful_audit(root / "verification" / CALIBRATION):
        return False
    for role, cases in (("calibration", CASES[:3]), ("holdout", CASES)):
        real = model / role / "real"
        if {path.name for path in real.glob("repeat-*")} != {f"repeat-{i}" for i in range(1, 6)}:
            return False
        for repeat in range(1, 6):
            group = real / f"repeat-{repeat}"
            if not text_is(group / "BASELINE_COLLECTION_COMPLETE", "complete"):
                return False
            folders = group / "cases"
            if not folders.is_dir():
                return False
            if {path.name for path in folders.iterdir() if path.is_dir()} != set(cases):
                return False
            if not all(text_is(folders / case / "status.txt", "ok") for case in cases):
                return False
    return True


def ready(root: Path, stage: str) -> bool:
    if not native_complete(root):
        return False
    model = root / "models" / MODEL
    if stage == "saved_score":
        folder = model / "holdout/ml_teacher_forced"
        return (latest_status(root, "teacher_forced") == "complete"
                and latest_status(root, "distribution_audit") == "complete"
                and all((folder / name).is_file() for name in ("per_batch.csv", "summary.csv")))
    if not successful_audit(root / "verification" / STAGES[0][1]):
        return False
    if stage == "frozen_fit_input":
        return True
    if stage != "endpoints":
        raise ValueError(stage)
    if not text_is(model / "REAL_ML_COMPLETE", "complete"):
        return False
    if any(latest_status(root, name) != "complete" for name in ("e2e_comparison", "model")):
        return False
    replay = model / "holdout/ml_simulator"
    if {path.name for path in replay.glob("repeat-*")} != {f"repeat-{i}" for i in range(1, 6)}:
        return False
    for repeat in range(1, 6):
        group = replay / f"repeat-{repeat}"
        if not text_is(group / "ML_REPLAY_COMPLETE", "complete"):
            return False
        if not all(text_is(group / "cases" / case / "status.txt", "ok") for case in CASES):
            return False
    return True


def process_identity(pid: int) -> dict | None:
    try:
        proc = Path("/proc") / str(pid)
        stat = (proc / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": int(fields[19]),
                "argv": (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--matrix-pid", type=int, required=True)
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()
    assert args.interval >= 10
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 15
    sim, root = Path.cwd(), args.results_root.resolve()
    assert root == (sim / "test-results" / RUN).resolve()
    owner = process_identity(args.matrix_pid)
    assert owner is not None
    runner = (sim / "scripts/run_pr33824_comprehensive_predictor_matrix.sh").resolve()
    assert any(Path(arg).resolve() == runner for arg in owner["argv"] if arg.startswith("/"))
    assert "--results-root" in owner["argv"]
    assert Path(owner["argv"][owner["argv"].index("--results-root") + 1]).resolve() == root
    out = root / "verification/awq_audit_waiter_20260929_1505"
    out.mkdir(exist_ok=True)
    with (out / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [root / "verification" / folder / name for _, folder, name in STAGES]
        sources += [Path(__file__).resolve(), root / "verification" / CALIBRATION / "results/manifest.json",
                    root / "models" / MODEL / f"calibration/training/{MODEL}_hgbmono_p50.pkl"]
        hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        record = {"started_at": now(), "pid": os.getpid(), "argv": sys.argv,
                  "cwd": str(sim), "owner": owner, "source_sha256": hashes,
                  "interval_seconds": args.interval, "nice": os.getpriority(os.PRIO_PROCESS, 0),
                  "environment": {key: os.environ.get(key) for key in ("PYTHONNOUSERSITE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONPATH")},
                  "scope": "Three existing checkpoint CPU readbacks only; not all-model or experiment completion"}
        with (out / "worker_manifest.json").open("x") as stream:
            json.dump(record, stream, indent=2)
        completed = []
        state = {"completed_readbacks": completed, "state": "running"}
        try:
            for stage, folder, filename in STAGES:
                directory = root / "verification" / folder
                while not successful_audit(directory):
                    if (directory / "execution.command.json").exists() or (directory / "results").exists():
                        raise RuntimeError(f"Retaining failed/incomplete prior audit, not overwriting: {directory}")
                    gate = ready(root, stage)
                    state.update(updated_at=now(), stage=stage, ready=gate,
                                 owner_live=process_identity(args.matrix_pid) == owner)
                    (out / "state.json").write_text(json.dumps(state, indent=2) + "\n")
                    if not gate:
                        if not state["owner_live"]:
                            raise RuntimeError("Original matrix ended before the required results closed")
                        print(json.dumps(state), flush=True)
                        time.sleep(args.interval)
                        continue
                    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
                               for path, value in hashes.items()), "Prepared audit or frozen bundle changed"
                    command = [sys.executable, str(directory / filename)]
                    execution = {"started_at": now(), "argv": command, "cwd": str(sim),
                                 "environment": record["environment"], "nice": record["nice"]}
                    with (directory / "audit.log").open("x") as log:
                        result = subprocess.run(command, cwd=sim, stdout=log, stderr=subprocess.STDOUT)
                    execution.update(finished_at=now(), exit_code=result.returncode)
                    with (out / f"execution_{stage}.json").open("x") as stream:
                        json.dump(execution, stream, indent=2)
                    if result.returncode != 0 or not successful_audit(directory):
                        raise RuntimeError(f"Readback failed; preserved output: {directory}")
                completed.append(stage)
                print(json.dumps({"at": now(), "readback": stage, "state": "verified"}), flush=True)
            state.update(updated_at=now(), state="complete", scope=record["scope"])
            with (out / "READBACKS_COMPLETE.json").open("x") as stream:
                json.dump(state, stream, indent=2)
        except BaseException as error:
            state.update(updated_at=now(), state="failed", error=repr(error))
            raise
        finally:
            (out / "state.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
