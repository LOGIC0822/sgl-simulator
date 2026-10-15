"""Focused CPU-only readiness/schema checks; not experiment accuracy tests."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "awq_audit_waiter", Path(__file__).with_name("wait_awq_audits_20260929_1505.py")
)
WAITER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WAITER)
BUNDLE = "5f212c17d088c78a916801fecd7cbc1634418413ed1c3cb60694565ea09301a7"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def audit(root, stage):
    names = {"calibration": WAITER.CALIBRATION, **{
        key: folder for key, folder, _ in WAITER.STAGES
    }}
    directory = root / "verification" / names[stage]
    value = {"model": WAITER.MODEL}
    if stage == "calibration":
        value.update(p50_bundle_sha256=BUNDLE, training_requests=800,
                     timing_rows=7069, training_cases=15)
    elif stage == "saved_score":
        value.update(bundle_sha256=BUNDLE, training_requests=800,
                     independent_test_requests=1280, actual_replay_payloads_verified=1280,
                     native_case_command_pairs_verified=25, saved_summary_groups_verified=18,
                     test_timing_rows=10000, raw_score_identity_rows_verified=10000,
                     training_prompt_overlap=0, reused_request_ids=0)
    elif stage == "frozen_fit_input":
        value.update(bundle_sha256=BUNDLE, training_requests=800,
                     independent_test_requests=1280, training_rows=7069,
                     independent_test_rows=10000, fit_invoked=False)
    elif stage == "endpoints":
        value.update(bundle_sha256=BUNDLE, paired_cases=25, paired_metric_rows=100,
                     requests_verified=1280, matched_native_replay_capacity_cases=25,
                     fit_gpu_or_latency_adjustment_invoked=False)
    else:
        raise AssertionError(stage)
    write_json(directory / "execution.command.json", {"exit_code": 0})
    write_json(directory / "results/manifest.json", value)
    (directory / "results/source_sha256.csv").write_text("path,sha256\n")
    return directory, value


@pytest.mark.parametrize("stage", ["calibration", "saved_score", "frozen_fit_input", "endpoints"])
def test_valid_stage_schema(tmp_path, stage):
    directory, _ = audit(tmp_path, stage)
    assert WAITER.successful_audit(directory)


@pytest.mark.parametrize("fault", ["wrong_model", "wrong_bundle", "failed_exit",
                                   "missing_execution", "wrong_request_count",
                                   "missing_hash_csv", "mismatched_row_count"])
def test_saved_score_rejects_invalid_evidence(tmp_path, fault):
    directory, value = audit(tmp_path, "saved_score")
    if fault == "wrong_model":
        value["model"] = "Qwen2.5-14B-Instruct-1M"
    elif fault == "wrong_bundle":
        value["bundle_sha256"] = "wrong"
    elif fault == "failed_exit":
        write_json(directory / "execution.command.json", {"exit_code": 1})
    elif fault == "missing_execution":
        (directory / "execution.command.json").unlink()
    elif fault == "wrong_request_count":
        value["independent_test_requests"] = 256
    elif fault == "missing_hash_csv":
        (directory / "results/source_sha256.csv").unlink()
    elif fault == "mismatched_row_count":
        value["raw_score_identity_rows_verified"] = 9999
    write_json(directory / "results/manifest.json", value)
    assert not WAITER.successful_audit(directory)


def complete_fixture(root):
    audit(root, "calibration")
    model = root / "models" / WAITER.MODEL
    for role, cases in (("calibration", WAITER.CASES[:3]), ("holdout", WAITER.CASES)):
        for repeat in range(1, 6):
            group = model / role / "real" / f"repeat-{repeat}"
            group.mkdir(parents=True)
            (group / "BASELINE_COLLECTION_COMPLETE").write_text("complete\n")
            for case in cases:
                folder = group / "cases" / case
                folder.mkdir(parents=True)
                (folder / "status.txt").write_text("ok\n")
    score = model / "holdout/ml_teacher_forced"
    score.mkdir()
    for name in ("per_batch.csv", "summary.csv"):
        (score / name).write_text("fixture\n")
    audit(root, "saved_score")
    for repeat in range(1, 6):
        group = model / "holdout/ml_simulator" / f"repeat-{repeat}"
        group.mkdir(parents=True)
        (group / "ML_REPLAY_COMPLETE").write_text("complete\n")
        for case in WAITER.CASES:
            folder = group / "cases" / case
            folder.mkdir(parents=True)
            (folder / "status.txt").write_text("ok\n")
    (model / "REAL_ML_COMPLETE").write_text("complete\n")
    (root / "status.tsv").write_text("".join(
        f"fixture\t{WAITER.MODEL}\t{stage}\tcomplete\t\n"
        for stage in ("teacher_forced", "distribution_audit", "e2e_comparison", "model")
    ))
    return model


@pytest.mark.parametrize("stage", ["saved_score", "frozen_fit_input", "endpoints"])
def test_ready_requires_all_native_groups(tmp_path, stage):
    model = complete_fixture(tmp_path)
    (model / "holdout/real/repeat-5/BASELINE_COLLECTION_COMPLETE").unlink()
    assert not WAITER.ready(tmp_path, stage)


@pytest.mark.parametrize("stage", ["saved_score", "frozen_fit_input", "endpoints"])
def test_ready_complete_fixture(tmp_path, stage):
    complete_fixture(tmp_path)
    assert WAITER.ready(tmp_path, stage)


def test_endpoints_requires_all_replays(tmp_path):
    model = complete_fixture(tmp_path)
    (model / "holdout/ml_simulator/repeat-5/ML_REPLAY_COMPLETE").unlink()
    assert not WAITER.ready(tmp_path, "endpoints")


def test_status_latest_failure_wins(tmp_path):
    complete_fixture(tmp_path)
    path = tmp_path / "status.tsv"
    with path.open("a") as stream:
        stream.write(f"fixture\t{WAITER.MODEL}\tteacher_forced\tfailed\t\n")
    assert not WAITER.ready(tmp_path, "saved_score")
