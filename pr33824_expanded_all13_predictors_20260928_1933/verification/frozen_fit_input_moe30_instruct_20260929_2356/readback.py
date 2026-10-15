"""Read back the frozen second-MoE predictor; no fitting or GPU work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refuse_fit(*args, **kwargs):
    raise AssertionError("Frozen readback must not fit or refit")


def main() -> None:
    script = Path(__file__).resolve()
    directory = script.parent
    root = directory.parent.parent
    sim = Path.cwd().resolve()
    model = "Qwen3-30B-A3B-Instruct-2507"
    mr = root / "models" / model
    expected = "5b13bad77cae8f3421c988eeeeba45ecbaadbffbbf6bf1d23f8dabd26a7620d8"
    keys = ("PYTHONNOUSERSITE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH", "MPLCONFIGDIR")
    record = {
        "python_argv": [sys.executable, *sys.argv],
        "launch_command": ["env", *[f"{key}={os.environ.get(key, '')}" for key in keys],
                           "nice", "-n", "15", sys.executable, str(script)],
        "cwd": str(sim), "environment": {key: os.environ.get(key) for key in keys},
        "started_unix": time.time(), "exit_code": 1,
        "scope": "Frozen final p50 readback only; no fit, native run, AIC query or new replay",
    }
    try:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert os.environ.get("PYTHONNOUSERSITE") == "1"
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert os.environ.get(key) == "1"
        training_proof = root / "verification/moe30_instruct_full_training_input_readback_20260929_2228/manifest.json"
        test_proof = root / "verification/moe30_instruct_full_independent_saved_score_integrity_20260929_2354/manifest.json"
        train = json.loads(training_proof.read_text())["result"]["all_five_training_groups"]
        held_record = json.loads(test_proof.read_text())
        assert held_record["exit_code"] == 0
        held = held_record["result"]
        assert train["native_requests"] == 800 and train["training_rows"] == 6931
        assert held["test_requests"] == 1280
        assert held["test_timing_rows"] == 12188
        assert held["shared_request_ids"] == held["shared_complete_inputs"] == 0
        assert train["p50_bundle_sha256"] == held["frozen_p50_sha256"] == expected
        bundle = mr / "calibration/training" / f"{model}_hgbmono_p50.pkl"
        assert digest(bundle) == expected
        sources = {script, training_proof, test_proof, bundle, root / "run_metadata.txt",
                   sim / "tools/append_pr33824_expanded_report.py",
                   sim / "tools/score_pr33824_latency_model.py",
                   sim / "src/sglang_simulator/time_predictor/ml_features.py",
                   mr / "holdout/ml_teacher_forced/per_batch.csv",
                   mr / "holdout/ml_teacher_forced/summary.csv"}
        for role, case_count in (("calibration", 3), ("holdout", 5)):
            repeats = sorted((mr / role / "real").glob("repeat-*"))
            assert len(repeats) == 5
            for repeat in repeats:
                marker = repeat / "BASELINE_COLLECTION_COMPLETE"
                assert marker.read_text().strip() == "complete"
                sources.add(marker)
                schedules = sorted((repeat / "cases").glob("*/TP0.schedule_batch.jsonl"))
                assert len(schedules) == case_count
                for schedule in schedules:
                    status = schedule.parent / "status.txt"
                    assert status.read_text().strip() == "ok"
                    sources.update((schedule, status, schedule.parent / "TP0.raw_request.jsonl",
                                    schedule.parent / "autobench_trace.jsonl"))
        before = {str(path.resolve()): digest(path) for path in sources}
        HistGradientBoostingRegressor.fit = refuse_fit
        spec = importlib.util.spec_from_file_location(
            "frozen_moe_readback_report", sim / "tools/append_pr33824_expanded_report.py"
        )
        assert spec and spec.loader
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        output = directory / "results"
        output.mkdir(exist_ok=False)
        lines = report.request_identity_details(root, [model], output)
        lines += report.ml_fit_input_comparison_details(root, [model], output, Path("."))
        (output / "readback_sections.md").write_text("\n".join(lines), encoding="utf-8")
        identity = report.read_csv(output / "train_test_request_identity.csv")
        assert len(identity) == 1
        row = identity[0]
        for key, value in (("training_requests", 800), ("heldout_requests", 1280),
                           ("unique_training_prompts", 480), ("unique_heldout_prompts", 888),
                           ("reused_request_ids", 0), ("real_to_replay_ids_verified", 2080)):
            assert int(row[key]) == value, key
        assert float(row["heldout_prompt_overlap_pct"]) == 0
        assert float(row["heldout_input_and_output_length_overlap_pct"]) == 0
        macro = report.read_csv(output / "ml_fit_input_vs_independent_macro.csv")
        assert len(macro) == 9 and {item["bundle_sha256"] for item in macro} == {expected}
        all_phase = [item for item in macro if item["phase"] == "all"]
        assert sum(int(item["n"]) for item in all_phase if item["stage"] == "fit_input") == 6931
        assert sum(int(item["n"]) for item in all_phase if item["stage"] != "fit_input") == 12188
        after = {path: digest(Path(path)) for path in before}
        assert before == after, "Original files changed during readback"
        report.write_csv(output / "source_sha256.csv", [
            {"path": path, "sha256_before": value, "sha256_after": after[path]}
            for path, value in sorted(before.items())
        ])
        result = {
            "model": model, "bundle_sha256": expected, "training_requests": 800,
            "independent_test_requests": 1280, "training_rows": 6931,
            "independent_test_rows": 12188, "all_phase_macro": all_phase,
            "fit_invoked": False, "fit_guard": "HGB.fit raises",
            "original_files_verified_unchanged": len(before),
            "heldout_predictions": "Recomputed and matched original per-batch identities and values exactly",
            "metric_scope": "Workload-equal single-step macros; not pooled MAPE or E2E",
            "interpretation": "Fit-input rescore is not independent validation; request composition, batch distribution and time all differ",
            "gpu_aic_or_new_replay_invoked": False,
            "auxiliary_context": "Low-priority CPU diagnostic alongside original CPU replay; not a new timing baseline",
        }
        with (output / "manifest.json").open("x", encoding="utf-8") as destination:
            json.dump(result, destination, ensure_ascii=False, indent=2)
        record["exit_code"] = 0
        print(json.dumps(result, ensure_ascii=False))
    finally:
        record["finished_unix"] = time.time()
        with (directory / "execution.command.json").open("x", encoding="utf-8") as destination:
            json.dump(record, destination, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
