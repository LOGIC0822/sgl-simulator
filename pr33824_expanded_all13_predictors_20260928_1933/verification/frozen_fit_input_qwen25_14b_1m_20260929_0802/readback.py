"""Read back one completed checkpoint's frozen fit inputs and independent tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refuse_fit(*args, **kwargs):
    raise AssertionError("This readback must not fit or refit a predictor")


def main() -> None:
    script = Path(__file__).resolve()
    directory = script.parent
    root = directory.parent.parent
    sim = Path.cwd().resolve()
    model = "Qwen2.5-14B-Instruct-1M"
    model_root = root / "models" / model
    record = {
        "python_argv": [sys.executable, *sys.argv],
        "launch_command": ["env", "PYTHONNOUSERSITE=1", "CUDA_VISIBLE_DEVICES=",
                           "OMP_NUM_THREADS=1", "PYTHONPATH=src", "nice", "-n", "15",
                           sys.executable, str(script)],
        "cwd": str(sim),
        "environment": {key: os.environ.get(key) for key in
                        ("PYTHONNOUSERSITE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")},
        "started_unix": time.time(),
        "scope": "Frozen final p50 predictor; no fit, GPU collection, new replay, or AIC query",
    }
    exit_code = 1
    try:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        training_audit_path = root / "verification/qwen25_14b_1m_completed_calibration_20260929_0750/results/manifest.json"
        score_audit_path = root / "verification/qwen25_14b_1m_full_holdout_saved_score_20260929_0802/results/manifest.json"
        training_execution = training_audit_path.parent.parent / "execution.command.json"
        score_execution = score_audit_path.parent.parent / "execution.command.json"
        training_audit = json.loads(training_audit_path.read_text())
        train_rows = int(training_audit["timing_rows"])
        assert training_audit["model"] == model
        score_audit = json.loads(score_audit_path.read_text())
        assert json.loads(training_execution.read_text())["exit_code"] == 0
        assert json.loads(score_execution.read_text())["exit_code"] == 0
        assert train_rows > 0 and training_audit["training_requests"] == 800
        assert score_audit["test_timing_rows"] > 0 and score_audit["independent_test_requests"] == 1280
        test_rows = int(score_audit["test_timing_rows"])
        sources = [root / "run_metadata.txt", training_audit_path, score_audit_path,
                   training_execution, score_execution, script,
                   sim / "tools/append_pr33824_expanded_report.py",
                   sim / "tools/score_pr33824_latency_model.py",
                   sim / "src/sglang_simulator/time_predictor/ml_features.py",
                   model_root / "holdout/ml_teacher_forced/per_batch.csv",
                   model_root / "holdout/ml_teacher_forced/summary.csv"]
        bundle_path = model_root / "calibration/training" / f"{model}_hgbmono_p50.pkl"
        sources.append(bundle_path)
        expected_bundle = training_audit["p50_bundle_sha256"]
        assert digest(bundle_path) == expected_bundle
        assert training_audit["p50_bundle_sha256"] == score_audit["bundle_sha256"] == expected_bundle
        for folder, expected_cases in (("calibration", 3), ("holdout", 5)):
            role = model_root / folder / "real"
            repeats = sorted(role.glob("repeat-*"))
            assert len(repeats) == 5
            for repeat in repeats:
                marker = repeat / "BASELINE_COLLECTION_COMPLETE"
                assert marker.read_text().strip() == "complete"
                sources.append(marker)
                cases = sorted((repeat / "cases").glob("*/TP0.schedule_batch.jsonl"))
                assert len(cases) == expected_cases
                for schedule in cases:
                    status = schedule.parent / "status.txt"
                    assert status.read_text().strip() == "ok"
                    sources.extend([schedule, status, schedule.parent / "TP0.raw_request.jsonl",
                                    schedule.parent / "autobench_trace.jsonl"])
        sources = sorted({path.resolve() for path in sources})
        before = {str(path): digest(path) for path in sources}
        bundle = joblib.load(bundle_path)
        assert isinstance(bundle["model"], HistGradientBoostingRegressor)
        HistGradientBoostingRegressor.fit = refuse_fit
        spec = importlib.util.spec_from_file_location(
            "frozen_readback_report", sim / "tools/append_pr33824_expanded_report.py"
        )
        assert spec and spec.loader
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        output = directory / "results"
        output.mkdir(exist_ok=False)
        identity_lines = report.request_identity_details(root, [model], output)
        lines = report.ml_fit_input_comparison_details(root, [model], output, Path("."))
        (output / "readback_sections.md").write_text(
            "\n".join(identity_lines + lines), encoding="utf-8"
        )
        identity = report.read_csv(output / "train_test_request_identity.csv")
        assert len(identity) == 1
        row = identity[0]
        assert int(row["training_requests"]) == 800
        assert int(row["heldout_requests"]) == 1280
        assert int(row["reused_request_ids"]) == 0
        assert int(row["unique_training_prompts"]) == int(training_audit["unique_training_prompts"])
        assert int(row["unique_heldout_prompts"]) == int(score_audit["independent_test_unique_prompts"])
        assert int(row["real_to_replay_ids_verified"]) == 2080
        assert float(row["heldout_prompt_overlap_pct"]) == 0
        assert float(row["heldout_input_and_output_length_overlap_pct"]) == 0
        macro = report.read_csv(output / "ml_fit_input_vs_independent_macro.csv")
        assert len(macro) == 9
        all_phase = [item for item in macro if item["phase"] == "all"]
        assert sum(int(item["n"]) for item in all_phase if item["stage"] != "fit_input") == test_rows
        assert sum(int(item["n"]) for item in all_phase if item["stage"] == "fit_input") == train_rows
        assert {item["bundle_sha256"] for item in macro} == {expected_bundle}
        after = {str(path): digest(path) for path in sources}
        assert before == after, "Original completed source/data/bundle changed"
        report.write_csv(output / "source_sha256.csv", [
            {"path": path, "sha256_before": value, "sha256_after": after[path]}
            for path, value in before.items()
        ])
        manifest = {
            "model": model, "bundle_path": str(bundle_path), "bundle_sha256": expected_bundle,
            "training_requests": 800, "independent_test_requests": 1280,
            "training_rows": train_rows, "independent_test_rows": test_rows,
            "heldout_predictions": "Recomputed with frozen bundle and exactly matched original per-batch scores",
            "fit_invoked": False, "fit_guard": "HistGradientBoostingRegressor.fit raises",
            "completion_guard": "All ten native collection markers and forty case statuses, executed calibration/saved-score audits; no E2E completion required or claimed",
            "auxiliary_execution_context": "Low-priority CPU readback while original CPU E2E replay remains live; not a new or isolated timing experiment",
            "gpu_collection_invoked": False, "new_e2e_replay_invoked": False,
            "original_files_verified_unchanged": len(before),
            "metric_scope": "Three separate workload-equal macros: fit input, independent seen, independent unseen",
            "interpretation": "Fit-input readback is not independent validation; differences are not a request-ID-only causal ablation",
            "all_phase_macro": all_phase,
        }
        with (output / "manifest.json").open("x", encoding="utf-8") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        record["original_files_verified_unchanged"] = len(before)
        record["fit_invoked"] = False
        exit_code = 0
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        record["exit_code"] = exit_code
        record["finished_unix"] = time.time()
        with (directory / "execution.command.json").open("x", encoding="utf-8") as destination:
            json.dump(record, destination, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
