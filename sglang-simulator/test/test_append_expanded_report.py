"""Validate report completeness without launching servers or GPU workloads."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("expanded_report", ROOT / "tools/append_pr33824_expanded_report.py")
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def upstream_split_fixture(tmp_path):
    folder = tmp_path / "verification/upstream_row_split_identity_fixture"
    folder.mkdir(parents=True)
    source = folder / "upstream_train_latency_model.py"
    source.write_text("# Synthetic source snapshot for artifact-reader tests only\n")
    raw = tmp_path / "sample_schedule.jsonl"
    raw.write_text("{}\n")
    write_csv(folder / "source_sha256.csv", [{"path": str(raw), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}])
    rows = [{"model": "example", "forward_modes": mode, "rows": count,
             "train_rows": count // 2, "internal_test_rows": count // 2,
             "unique_request_ids": 3, "train_unique_request_ids": ids,
             "test_unique_request_ids": ids, "shared_request_ids": shared,
             "shared_pct_of_all_requests": 100 * shared / 3,
             "identical_rows_in_both_splits": 0}
            for mode, count, ids, shared in (("1", 4, 2, 1), ("1,2", 6, 3, 3))]
    write_csv(folder / "upstream_split_request_overlap.csv", rows)
    manifest = {"upstream_branch_head": "9cfddcedc9d1e00885c28db75a45513eddbb6271",
                "upstream_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "split_summary_sha256": hashlib.sha256((folder / "upstream_split_request_overlap.csv").read_bytes()).hexdigest(),
                "parsed_defaults": {"--forward-modes": "1", "--seed": 42, "--test-fraction": .5},
                "source_files_verified_unchanged": 1, "no_fit_gpu_or_formal_split_mutation": True}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "analysis"
    out.mkdir()
    return folder, manifest, rows, raw, out


def test_upstream_split_report_is_qualified_and_has_request_denominators(tmp_path):
    _, _, _, _, out = upstream_split_fixture(tmp_path)
    text = "\n".join(report.upstream_split_request_details(tmp_path, ["example"], out))
    assert "1 / 3（33.33%）" in text and "3 / 3（100.00%）" in text
    assert "这不是上游作者的数据或准确率" in text
    assert "并非上游默认设置" in text
    assert "两侧没有相同的原始行" in text
    assert "不测量它使误差降低多少" in text
    assert (out / "upstream_split_request_overlap.csv").is_file()


@pytest.mark.parametrize("change", ["defaults", "source", "summary", "input", "counts", "coverage"])
def test_upstream_split_report_rejects_different_source_or_evidence(tmp_path, change):
    folder, manifest, rows, raw, out = upstream_split_fixture(tmp_path)
    if change == "defaults":
        manifest["parsed_defaults"]["--seed"] = 0
    elif change == "source":
        (folder / "upstream_train_latency_model.py").write_text("changed")
    elif change == "summary":
        (folder / "upstream_split_request_overlap.csv").write_text("changed")
    elif change == "input":
        raw.write_text("changed")
    else:
        if change == "counts":
            rows[0]["train_unique_request_ids"] = 1
        else:
            rows.pop()
        write_csv(folder / "upstream_split_request_overlap.csv", rows)
        manifest["split_summary_sha256"] = hashlib.sha256((folder / "upstream_split_request_overlap.csv").read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        report.upstream_split_request_details(tmp_path, ["example"], out)


def test_upstream_split_report_does_not_invent_missing_demonstration(tmp_path):
    assert report.upstream_split_request_details(tmp_path, ["unused"], tmp_path) == []


@pytest.mark.parametrize("affected", [0, 2])
def test_aic_cache_report_separates_argument_aliases_from_latency_error(tmp_path, monkeypatch, affected):
    provenance = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"aic_root": "/synthetic-aic"}))
    rows = [{"model": "example", "role": role, "rows": count, "material_alias_rows": alias}
            for role, count, alias in (("calibration", 10, 0), ("holdout", 20, affected))]
    observed = []
    module = SimpleNamespace(run=lambda *args: observed.append(args) or rows)
    monkeypatch.setattr(report.importlib.util, "spec_from_file_location", lambda *args:
                        SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None)))
    monkeypatch.setattr(report.importlib.util, "module_from_spec", lambda _: module)
    text = "\n".join(report.aic_cache_contract_details(tmp_path, ["example"], tmp_path / "analysis"))
    assert f"| example | 10 / 20 | 0 / {affected} |" in text
    assert "没有修改评分器" in text
    assert len(observed) == 1
    assert observed[0][3] == Path("/synthetic-aic")
    assert "不能直接换算成预测误差" in text if affected else "不能用该缓存键差异解释" in text


def test_aic_cache_report_optional_only_without_formal_identity(tmp_path):
    assert report.aic_cache_contract_details(tmp_path, ["unused"], tmp_path) == []
    provenance = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}")
    with pytest.raises(KeyError, match="aic_root"):
        report.aic_cache_contract_details(tmp_path, ["unused"], tmp_path)


def native_profile_report_fixture(tmp_path, monkeypatch):
    provenance = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}")
    target = tmp_path / "analysis/matched_native_profiles"
    rows = []
    for model in ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-32B"):
        for phase in ("prefill", "decode"):
            rows.append({"model": model, "phase": phase, "n": 15, "instrumented_wall_ms": 20,
                         "native_gpu_union_ms": 12, "prefix_aic_ms": 7, "runtime_aic_ms": 8,
                         "physical_before_aic_ms": 9, "physical_before_ape_vs_wall_pct": 55,
                         "physical_before_ape_vs_gpu_union_pct": 25, "gpu_attribution_duration_pct": 98,
                         "gemm_gemv_duration_sum_ms": 8, "physical_aic_gemm_sum_ms": 6,
                         "triton_prefill_attention_duration_sum_ms": 1, "triton_decode_attention_duration_sum_ms": 2,
                         "physical_aic_attention_sum_ms": 2, "other_kernel_duration_sum_ms": 1,
                         "physical_aic_other_sum_ms": 1})
    write_csv(target / "analysis/summary.csv", rows)
    monkeypatch.setattr(report, "run_final_native_profile_diagnostic", lambda root, out: target)
    return target, rows


def test_native_profile_report_keeps_instrumented_and_formal_results_separate(tmp_path, monkeypatch):
    target, _ = native_profile_report_fixture(tmp_path, monkeypatch)
    lines = report.native_profile_details(tmp_path, tmp_path / "analysis", Path("figures"))
    text = "\n".join(lines)
    assert "20.000 / 12.000" in text and "7.000 / 8.000 / 9.000" in text
    assert "55.00% / 25.00%" in text
    assert "不替代前面五次正式实测" in text and "不是逐算子一一对应" in text
    assert "不能直接称为纯 CPU 时间" in text
    assert (target / "analysis/native_gpu_vs_aic_diagnostic.png").is_file()


def test_native_profile_report_rejects_incomplete_coverage(tmp_path, monkeypatch):
    target, rows = native_profile_report_fixture(tmp_path, monkeypatch)
    (target / "analysis/summary.csv").unlink()
    rows[0]["n"] = 14
    write_csv(target / "analysis/summary.csv", rows)
    with pytest.raises(ValueError, match="incomplete 3-model"):
        report.native_profile_details(tmp_path, tmp_path / "analysis", Path("figures"))


def test_native_profile_report_is_optional_only_for_nonformal_fixtures(tmp_path):
    assert report.native_profile_details(tmp_path, tmp_path, Path("figures")) == []
    with pytest.raises(ValueError, match="formal collection"):
        report.run_final_native_profile_diagnostic(tmp_path, tmp_path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_results(tmp_path, repeats=5):
    model = "Qwen3-0.6B"
    case = "latency_128x64_r1"
    model_root = tmp_path / "models" / model
    for method in ("ml", "aic"):
        batch = [{"group": group, "value": value, "n": "15", "mean_actual_ms": "2", "mean_predicted_ms": "1.8", "mape_pct": "10", "mae_ms": "0.2", "p95_ape_pct": "12", "mean_signed_error_pct": "-10"}
                 for group, value in (("case", case), ("case_forward_mode", case + ":mode1"), ("case_forward_mode", case + ":mode2"))]
        write_csv(model_root / "holdout" / f"{method}_teacher_forced" / "summary.csv", batch)
        pairs = [{"group": "case_metric", "value": case + ":" + metric, "n": str(repeats), "real_mean": "2", "real_std": "0.1", "predicted_mean": "1.8", "predicted_std": "0.1", "mape_pct": "10", "mean_signed_error_pct": "-10"}
                 for metric in ("mean_e2e_latency_ms", "mean_ttft_ms", "mean_tpot_ms", "total_throughput")]
        write_csv(model_root / "holdout" / f"{method}_e2e_comparison" / "summary.csv", pairs)
    write_csv(model_root / "learning_curve" / "learning_curve.csv", [{"model": model, "case": case, "training_traces": str(count), "phase": "all", "mape_pct": "10"} for count in (1, 3, 5)])
    return model, case


def test_report_has_batch_phases_and_all_metrics(tmp_path):
    model, case = fixture_results(tmp_path)
    batch, pairs, curves = report.collect(tmp_path, [model], [case], 5)
    assert len(batch) == 6
    assert len(pairs) == 8
    assert {row["phase"] for row in batch} == {"all", "prefill", "decode"}
    assert curves[0]["case"] == case


def fit_input_fixture(root):
    import numpy as np
    from sklearn.dummy import DummyRegressor
    from sglang_simulator.time_predictor.ml_features import ML_FEATURE_NAMES

    model = "frozen_example"
    model_root = root / "models" / model
    training = model_root / "calibration/real/repeat-1/cases"
    bundle_path = model_root / "calibration/training" / f"{model}_hgbmono_p50.pkl"
    (root / "run_metadata.txt").write_text("training_case=seen_a,seen_b\nheldout_case=seen_a,seen_b,unseen_c\n")
    for role, cases in (("calibration", {"seen_a": [(1, .01), (2, .01)],
                                        "seen_b": [(1, .02), (2, .02), (2, .02), (2, .02), (2, 31)]}),
                        ("holdout", {"seen_a": [(1, .02), (2, .02)],
                                     "seen_b": [(1, .01), (2, .01)], "unseen_c": [(1, .005), (2, .005)]})):
        for case, latencies in cases.items():
            path = model_root / role / "real/repeat-1/cases" / case / "TP0.schedule_batch.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps({"forward_mode": mode, "iter_latency": latency,
                                                "request_infos": [{"extend_input_len": 4 if mode == 1 else 1,
                                                                   "prefix_indices_len": index}]}) + "\n"
                                    for index, (mode, latency) in enumerate(latencies)))
    predictor = DummyRegressor(strategy="constant", constant=.01)
    predictor.fit(np.zeros((1, len(ML_FEATURE_NAMES))), [.01])
    bundle_path.parent.mkdir(parents=True)
    joblib.dump({"model": predictor, "features": list(ML_FEATURE_NAMES),
                 "source_data_roots": [str(training)], "target": "iter_latency_seconds",
                 "forward_modes": [1, 2], "train_rows": 6, "train_cases": 2,
                 "max_latency_seconds": 30}, bundle_path)
    scored = []
    for path in sorted((model_root / "holdout/real").glob("repeat-*/cases/*/TP0.schedule_batch.jsonl")):
        for line, value in enumerate(path.read_text().splitlines(), start=1):
            raw = json.loads(value)
            scored.append({"case": path.parent.name, "schedule_path": str(path.resolve()),
                           "line_number": line, "forward_mode": raw["forward_mode"], "batch_size": 1,
                           "sum_extend": raw["request_infos"][0]["extend_input_len"],
                           "sum_past": raw["request_infos"][0]["prefix_indices_len"],
                           "actual_seconds": raw["iter_latency"], "predicted_seconds": .01})
    heldout = model_root / "holdout/ml_teacher_forced/per_batch.csv"
    write_csv(heldout, scored)
    output = root / "fit_readback"
    output.mkdir()
    return model, bundle_path, heldout, output


def test_frozen_fit_input_comparison_never_refits_or_changes_holdout(tmp_path, monkeypatch):
    from sklearn.dummy import DummyRegressor
    model, bundle_path, heldout, output = fit_input_fixture(tmp_path)
    original_bundle, original_test = bundle_path.read_bytes(), heldout.read_bytes()
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("readback must not refit the predictor")
    monkeypatch.setattr(DummyRegressor, "fit", forbidden_fit)
    lines = report.ml_fit_input_comparison_details(tmp_path, [model], output, Path("figures"))
    assert bundle_path.read_bytes() == original_bundle and heldout.read_bytes() == original_test
    macro = report.read_csv(output / "ml_fit_input_vs_independent_macro.csv")
    assert len(macro) == 9
    all_rows = {row["stage"]: row for row in macro if row["phase"] == "all"}
    assert float(all_rows["fit_input"]["mape_pct"]) == 25  # Equal cases, not equal batch rows.
    assert int(all_rows["fit_input"]["n"]) == 6  # The 31-second training row is excluded.
    assert float(all_rows["independent_seen"]["mape_pct"]) == 25
    assert float(all_rows["independent_unseen"]["mape_pct"]) == 100
    assert float(all_rows["independent_unseen"]["mape_change_from_fit_input_pp"]) == 75
    assert len(report.read_csv(output / "ml_fit_input_vs_independent_by_workload.csv")) == 15
    assert len(report.read_csv(output / "ml_fit_input_readback" / model / "per_batch.csv")) == 6
    assert {row["bundle_sha256"] for row in macro} == {hashlib.sha256(original_bundle).hexdigest()}
    assert "训练样本回评分不是独立验证" in "\n".join(lines)
    assert (output / "ml_fit_input_vs_independent_mape.png").is_file()


@pytest.mark.parametrize("changed,expected", [
    ({"features": ["wrong"]}, "incompatible frozen ML bundle"),
    ({"train_rows": 7}, "fit input count/workload mismatch"),
    ({"train_cases": 3}, "training case count mismatch"),
    ({"max_latency_seconds": float("nan")}, "invalid frozen-model training filter"),
    ({"forward_modes": [1]}, "incompatible frozen ML bundle"),
])
def test_frozen_fit_input_comparison_rejects_changed_bundle(tmp_path, changed, expected):
    model, bundle_path, heldout, output = fit_input_fixture(tmp_path)
    bundle = joblib.load(bundle_path)
    bundle.update(changed)
    joblib.dump(bundle, bundle_path)
    with pytest.raises(ValueError, match=expected):
        report.ml_fit_input_comparison_details(tmp_path, [model], output, Path("figures"))


def test_frozen_fit_input_comparison_rejects_test_labels_as_training_source(tmp_path):
    model, bundle_path, heldout, output = fit_input_fixture(tmp_path)
    bundle = joblib.load(bundle_path)
    bundle["source_data_roots"] = [str(heldout.parent)]
    joblib.dump(bundle, bundle_path)
    with pytest.raises(ValueError, match="outside declared training data"):
        report.ml_fit_input_comparison_details(tmp_path, [model], output, Path("figures"))


@pytest.mark.parametrize("duplicate", [False, True])
def test_frozen_fit_input_comparison_rejects_changed_or_duplicate_holdout_score(tmp_path, duplicate):
    model, bundle_path, heldout, output = fit_input_fixture(tmp_path)
    rows = report.read_csv(heldout)
    if duplicate:
        rows.append(dict(rows[0]))
    else:
        rows[0]["predicted_seconds"] = .011
    write_csv(heldout, rows)
    with pytest.raises(ValueError, match="duplicate frozen-model observation" if duplicate else "same frozen model/observations"):
        report.ml_fit_input_comparison_details(tmp_path, [model], output, Path("figures"))


def test_frozen_fit_input_comparison_rejects_missing_formal_inputs(tmp_path):
    identity = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    identity.parent.mkdir(parents=True)
    identity.write_text("{}")
    with pytest.raises(FileNotFoundError, match="missing formal training batches"):
        report.ml_fit_input_comparison_details(tmp_path, ["missing"], tmp_path, Path("figures"))


def residual_fixture(root):
    model = "example"
    source = root / "models" / model / "holdout/real/repeat-1/cases/case/TP0.schedule_batch.jsonl"
    for method in ("ml", "aic"):
        directory = root / "models" / model / "holdout" / f"{method}_teacher_forced"
        rows = [{"case": "case", "schedule_path": str(source), "line_number": index,
                 "forward_mode": 2, "batch_size": 1, "sum_extend": 1, "sum_past": index,
                 "actual_seconds": actual, "predicted_seconds": predicted}
                for index, (actual, predicted) in enumerate(((1.0, 2.0), (2.0, 2.0)), start=1)]
        write_csv(directory / "per_batch.csv", rows[::-1] if method == "aic" else rows)
        write_csv(directory / "summary.csv", [{"group": "all", "value": "all", "n": 2,
                  "mean_actual_ms": 1500, "mean_predicted_ms": 2000,
                  "rmse_ms": 1000 * .5 ** .5, "mape_pct": 50}])
    output = root / "residual_report"
    output.mkdir()
    return model, output


def test_residual_report_keeps_all_rows_and_accepts_reordered_scores(tmp_path):
    model, output = residual_fixture(tmp_path)
    lines = report.residual_concentration_details(tmp_path, [model], output)
    rows = report.read_csv(output / "residual_concentration.csv")
    assert len(rows) == 2 and {row["method"] for row in rows} == {"ML", "AIC"}
    assert all(int(row["n"]) == 2 and int(row["top_100_count"]) == 2 for row in rows)
    assert all(float(row["top_1_squared_error_pct"]) == 100 for row in rows)
    assert all(float(row["top_1_mape_contribution_pp"]) == 50 for row in rows)
    assert all(float(row["top_100_row_pct"]) == 100 for row in rows)
    assert len(report.read_csv(output / "largest_batch_residuals.csv")) == 4
    assert "不是 MAPE 占比" in "\n".join(lines)
    assert "未删除异常值" in "\n".join(lines)


def test_residual_report_distinguishes_squared_tail_from_mape_contribution(tmp_path):
    model, output = residual_fixture(tmp_path)
    for method in ("ml", "aic"):
        directory = tmp_path / "models" / model / "holdout" / f"{method}_teacher_forced"
        rows = sorted(report.read_csv(directory / "per_batch.csv"), key=lambda row: int(row["line_number"]))
        # Squared errors 81/100, but APE 900%/10%: different rankings.
        for row, actual, predicted in zip(rows, (1, 100), (10, 90)):
            row.update(actual_seconds=actual, predicted_seconds=predicted)
        write_csv(directory / "per_batch.csv", rows)
        write_csv(directory / "summary.csv", [{"group": "all", "value": "all", "n": 2,
                  "mean_actual_ms": 50500, "mean_predicted_ms": 50000,
                  "rmse_ms": 1000 * (181 / 2) ** .5, "mape_pct": 455}])
    lines = report.residual_concentration_details(tmp_path, [model], output)
    rows = report.read_csv(output / "residual_concentration.csv")
    assert all(float(row["top_1_mape_contribution_pp"]) == 5 for row in rows)
    assert all(float(row["top_1_squared_error_pct"]) == pytest.approx(100 / 181 * 100) for row in rows)
    assert all(float(row["top_1_absolute_percentage_error_share_pct"]) == pytest.approx(10 / 910 * 100) for row in rows)
    assert "不是删除样本后 MAPE 会下降的幅度" in "\n".join(lines)


def test_single_method_residual_readback_does_not_claim_ml_aic_comparison(tmp_path):
    model, output = residual_fixture(tmp_path)
    lines = report.residual_concentration_details(tmp_path, [model], output, methods=("ML",))
    assert "不表示已经完成 ML/AIC 的逐条对照" in "\n".join(lines)
    assert {row["method"] for row in report.read_csv(output / "residual_concentration.csv")} == {"ML"}


def test_residual_report_rejects_equal_mean_but_different_real_steps(tmp_path):
    model, output = residual_fixture(tmp_path)
    path = tmp_path / "models" / model / "holdout/aic_teacher_forced/per_batch.csv"
    rows = report.read_csv(path)
    rows[0]["actual_seconds"], rows[1]["actual_seconds"] = rows[1]["actual_seconds"], rows[0]["actual_seconds"]
    write_csv(path, rows)
    # Counts, actual/predicted means, RMSE and MAPE remain identical.
    with pytest.raises(ValueError, match="different individual observations"):
        report.residual_concentration_details(tmp_path, [model], output)


def test_residual_report_rejects_duplicate_observations(tmp_path):
    model, output = residual_fixture(tmp_path)
    path = tmp_path / "models" / model / "holdout/ml_teacher_forced/per_batch.csv"
    rows = report.read_csv(path)
    write_csv(path, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate or invalid scored observation"):
        report.residual_concentration_details(tmp_path, [model], output)


def test_residual_report_rejects_nonfinite_prediction(tmp_path):
    model, output = residual_fixture(tmp_path)
    path = tmp_path / "models" / model / "holdout/ml_teacher_forced/per_batch.csv"
    rows = report.read_csv(path)
    rows[0]["predicted_seconds"] = "nan"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="nonfinite or invalid scored latency"):
        report.residual_concentration_details(tmp_path, [model], output)


def test_residual_report_defines_zero_error_share_without_dropping_rows(tmp_path):
    model, output = residual_fixture(tmp_path)
    for method in ("ml", "aic"):
        directory = tmp_path / "models" / model / "holdout" / f"{method}_teacher_forced"
        rows = report.read_csv(directory / "per_batch.csv")
        for row in rows:
            row["predicted_seconds"] = row["actual_seconds"]
        write_csv(directory / "per_batch.csv", rows)
        summary = report.read_csv(directory / "summary.csv")
        summary[0].update(mean_predicted_ms=1500, rmse_ms=0, mape_pct=0)
        write_csv(directory / "summary.csv", summary)
    report.residual_concentration_details(tmp_path, [model], output)
    assert all(float(row["top_100_squared_error_pct"]) == 0 and
               float(row["top_100_mape_contribution_pp"]) == 0 and
               float(row["top_100_absolute_percentage_error_share_pct"]) == 0 and int(row["n"]) == 2
               for row in report.read_csv(output / "residual_concentration.csv"))


def test_residual_report_rejects_summary_that_drops_a_batch(tmp_path):
    model, output = residual_fixture(tmp_path)
    path = tmp_path / "models" / model / "holdout/ml_teacher_forced/summary.csv"
    rows = report.read_csv(path)
    rows[0]["n"] = 1
    write_csv(path, rows)
    with pytest.raises(ValueError, match="per-batch rows disagree with summary"):
        report.residual_concentration_details(tmp_path, [model], output)


def test_formal_residual_report_requires_actual_scored_rows(tmp_path):
    identity = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    identity.parent.mkdir(parents=True)
    identity.write_text("{}")
    with pytest.raises(FileNotFoundError, match="per_batch.csv"):
        report.residual_concentration_details(tmp_path, ["example"], tmp_path)


def regression_record_fixture(root, failed=1, exit_code=1):
    verification = root / "verification"
    verification.mkdir()
    (verification / "full_suite_record.json").write_text(json.dumps({
        "date": "2026-09-28", "command": "python -m pytest -q test",
        "result": {"failed": failed, "passed": 47, "warnings": 5,
                   "elapsed_seconds": 395.2, "exit_code": exit_code},
        "scope_note": "CPU fixture，不是 GPU 精度实验。",
        "failure": {"test": "test_offline_blocking",
                    "outputs": {mode: f"verification/{mode}.json" for mode in ("offline", "blocking")},
                    "tolerances": {"mean_ttft_ms": .1},
                    "interpretation_note": "未放宽容差；尚未隔离原因。"},
    }))
    for mode, value in (("offline", 12.762786), ("blocking", 11.341969)):
        (verification / f"{mode}.json").write_text(json.dumps({"mean_ttft_ms": value}))


def test_report_discloses_full_suite_failure_from_saved_fixtures(tmp_path):
    regression_record_fixture(tmp_path)
    lines = report.regression_verification_details(tmp_path, tmp_path)
    assert "47 项通过、1 项失败" in "\n".join(lines)
    assert "退出码 1" in "\n".join(lines)
    assert "12.5271%" in "\n".join(lines)
    rows = report.read_csv(tmp_path / "full_suite_fixture_comparison.csv")
    assert float(rows[0]["relative_difference_pct"]) > float(rows[0]["tolerance_pct"])


def test_report_rejects_failed_full_suite_marked_success(tmp_path):
    regression_record_fixture(tmp_path, exit_code=0)
    with pytest.raises(ValueError, match="inconsistent full-suite result"):
        report.regression_verification_details(tmp_path, tmp_path)


def test_report_rejects_regression_record_not_reproduced_by_fixtures(tmp_path):
    regression_record_fixture(tmp_path)
    (tmp_path / "verification/offline.json").write_text(json.dumps({"mean_ttft_ms": 11.34}))
    with pytest.raises(ValueError, match="do not reproduce"):
        report.regression_verification_details(tmp_path, tmp_path)


def test_report_distinguishes_relative_bias_from_bias_of_means(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "comprehensive_report_bias_test",
        ROOT / "tools/summarize_pr33824_comprehensive_predictor_experiment.py",
    )
    assert spec and spec.loader
    base = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, base)
    spec.loader.exec_module(base)
    # Actual [1, 100], predictions [10, 90]: relative bias is +445%,
    # despite the prediction mean being below the actual mean.
    metrics = {"mean_actual_ms": 50.5, "mean_predicted_ms": 50.0,
               "mean_signed_error_pct": 445.0, "mape_pct": 455.0}
    record = {"model": "example", "complete": True,
              "train": base.BatchCounts(800, 2, 1, 1),
              "holdout": base.BatchCounts(1280, 2, 1, 1),
              "distribution": {}, "aic_manifest": {},
              "training_validation": {}, "settings": {},
              "teacher": {method: {phase: metrics for phase in ("all", "prefill", "decode")}
                          for method in base.METHODS},
              "e2e": {method: {"mean_e2e_latency_ms": {"mape_pct": 1.0}}
                      for method in base.METHODS}}
    support = {"grid": {"batch_sizes": [1], "context_input_lengths": [1],
                        "generation_input_lengths": [1]},
               **{key: {"context": 0, "generation": 0, "total": 0}
                  for key in ("expected", "new", "reused")}}
    counts = {key: 0 for key in ("source_rows", "retained_regular_context_rows",
                                 "retained_other_rows", "removed_sparse_or_incompatible_rows")}
    markdown = base.build_markdown(tmp_path, [record], {}, {}, support,
                                   {"row_counts": counts}, Path("figures"))
    assert "ML 平均相对偏差" in markdown
    assert "| example | 50.50 | 50.00 | 445.00% |" in markdown
    assert "不是 (预测均值−实测均值)/实测均值，两者可能符号不同" in markdown
    assert metrics["mean_signed_error_pct"] == 445.0


def replay_component_fixture(root, iterations=2):
    reference = root / "provenance/formal_start_after_ninja_fix/identity.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}")
    (root / "run_metadata.txt").write_text("heldout_real_repeats=1\nheldout_case=case\n")
    path = root / "models/example/holdout/ml_simulator/repeat-1/cases/case/simulator-output"
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"iterations": iterations, "predictor_time_cost": .004}))
    (path / "iteration.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"cpu_overhead": .003, "forward_latency": .01},
        {"cpu_overhead": .004, "forward_latency": .02},
    )))


def test_replay_wall_time_audit_preserves_forward_predictions(tmp_path):
    replay_component_fixture(tmp_path)
    output = tmp_path / "audit"
    output.mkdir()
    lines = report.replay_overhead_details(tmp_path, ["example"], output, methods=("ML",))
    rows = report.read_csv(output / "replay_wall_time_components.csv")
    assert len(rows) == 1
    assert float(rows[0]["predictor_wall_ms_per_iteration"]) == 2
    assert float(rows[0]["accounted_cpu_ms_per_iteration"]) == 3.5
    assert float(rows[0]["predicted_forward_ms_per_iteration"]) == 15
    assert "不是独立测得的原生服务 CPU 时延" in "\n".join(lines)
    assert "没有事后扣除平均开销" in "\n".join(lines)


def test_replay_wall_time_audit_rejects_missing_iterations(tmp_path):
    replay_component_fixture(tmp_path, iterations=3)
    with pytest.raises(ValueError, match="iteration count mismatch"):
        report.replay_overhead_details(tmp_path, ["example"], tmp_path, methods=("ML",))


def test_report_rejects_incomplete_real_repeats(tmp_path):
    model, case = fixture_results(tmp_path, repeats=4)
    with pytest.raises(ValueError, match="wrong real repeat count"):
        report.collect(tmp_path, [model], [case], 5)


def test_report_rejects_missing_phase(tmp_path):
    model, case = fixture_results(tmp_path)
    path = tmp_path / "models" / model / "holdout/ml_teacher_forced/summary.csv"
    rows = report.read_csv(path)
    write_csv(path, rows[:-1])
    with pytest.raises(ValueError, match="batch phases"):
        report.collect(tmp_path, [model], [case], 5)


def test_report_rejects_different_ml_aic_real_observations(tmp_path):
    model, case = fixture_results(tmp_path)
    path = tmp_path / "models" / model / "holdout/aic_teacher_forced/summary.csv"
    rows = report.read_csv(path)
    rows[0]["mean_actual_ms"] = "2.1"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="different real batch"):
        report.collect(tmp_path, [model], [case], 5)


def previous_comparison_fixture(tmp_path):
    model, case = "Qwen3-0.6B", "latency_128x64_r1"
    schedule = tmp_path / "models" / model / "holdout/real/repeat-1/cases" / case / "TP0.schedule_batch.jsonl"
    scored = [{"case": case, "schedule_path": str(schedule), "line_number": index,
               "forward_mode": mode, "batch_size": 1, "sum_extend": 4 if mode == 1 else 1,
               "sum_past": index, "actual_seconds": .002, "predicted_seconds": .0018}
              for index, mode in enumerate((1, 2, 2), start=1)]
    summary = [{"group": group, "value": value, "n": n,
                "mean_actual_ms": 2, "mean_predicted_ms": 1.8, "mape_pct": 10,
                "mae_ms": .2, "p95_ape_pct": 10, "mean_signed_error_pct": -10}
               for group, value, n in (("all", "all", 3), ("case", case, 3),
                                       ("forward_mode", "1", 1), ("forward_mode", "2", 2),
                                       ("case_forward_mode", case + ":mode1", 1),
                                       ("case_forward_mode", case + ":mode2", 2))]
    for method in ("ml", "aic"):
        current = tmp_path / "models" / model / "holdout" / f"{method}_teacher_forced"
        previous = tmp_path / "previous_calibration/models" / model / f"{method}_teacher_forced"
        write_csv(current / "summary.csv", summary)
        write_csv(previous / "summary.csv", summary)
        write_csv(current / "per_batch.csv", scored)
        write_csv(previous / "per_batch.csv", scored[::-1])
    return model


def test_previous_comparison_requires_identical_real_samples(tmp_path):
    model = previous_comparison_fixture(tmp_path)
    rows = report.previous_calibration_rows(tmp_path, [model])
    assert len(rows) == 12 and all(row["improvement_pp"] == 0 for row in rows)
    original = report.read_csv(tmp_path / "previous_calibration/models" / model / "aic_teacher_forced/summary.csv")
    original[0]["n"] = "2"
    write_csv(tmp_path / "previous_calibration/models" / model / "aic_teacher_forced/summary.csv", original)
    with pytest.raises(ValueError, match="different real observations"):
        report.previous_calibration_rows(tmp_path, [model])


@pytest.mark.parametrize("field", ["schedule_path", "line_number", "case", "forward_mode",
                                    "batch_size", "sum_extend", "sum_past", "actual_seconds"])
def test_previous_comparison_rejects_changed_rows_despite_equal_summary(tmp_path, field):
    model = previous_comparison_fixture(tmp_path)
    path = tmp_path / "previous_calibration/models" / model / "aic_teacher_forced/per_batch.csv"
    rows = report.read_csv(path)
    if field == "schedule_path":
        rows[0][field] = str(Path(rows[0][field]).with_name("different.jsonl"))
    elif field == "case":
        rows[0][field] = "different_case"
    elif field == "actual_seconds":
        rows[0][field], rows[1][field] = ".003", ".001"  # Same count and global mean.
    elif field == "forward_mode":
        rows[0][field] = "1"
    else:
        rows[0][field] = str(int(rows[0][field]) + 10)
    write_csv(path, rows)
    with pytest.raises(ValueError, match="different individual real observations"):
        report.previous_calibration_rows(tmp_path, [model])


@pytest.mark.parametrize("field,value", [("actual_seconds", "nan"), ("actual_seconds", "0"),
                                         ("predicted_seconds", "inf"), ("predicted_seconds", "-1"),
                                         ("forward_mode", "3"), ("line_number", "0")])
def test_previous_comparison_rejects_invalid_scored_rows(tmp_path, field, value):
    model = previous_comparison_fixture(tmp_path)
    path = tmp_path / "previous_calibration/models" / model / "ml_teacher_forced/per_batch.csv"
    rows = report.read_csv(path)
    rows[0][field] = value
    write_csv(path, rows)
    with pytest.raises(ValueError, match="invalid scored observation"):
        report.previous_calibration_rows(tmp_path, [model])


def test_previous_comparison_rejects_duplicate_scored_rows(tmp_path):
    model = previous_comparison_fixture(tmp_path)
    path = tmp_path / "previous_calibration/models" / model / "ml_teacher_forced/per_batch.csv"
    rows = report.read_csv(path)
    write_csv(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate scored observation"):
        report.previous_calibration_rows(tmp_path, [model])


def test_previous_comparison_accepts_changed_predictions_on_same_steps(tmp_path):
    model = previous_comparison_fixture(tmp_path)
    directory = tmp_path / "previous_calibration/models" / model / "ml_teacher_forced"
    scored = report.read_csv(directory / "per_batch.csv")
    for row in scored:
        row["predicted_seconds"] = ".0015"
    write_csv(directory / "per_batch.csv", scored)
    summary = report.read_csv(directory / "summary.csv")
    for row in summary:
        row.update(mean_predicted_ms="1.5", mape_pct="25", mae_ms=".5", p95_ape_pct="25")
    write_csv(directory / "summary.csv", summary)
    rows = [row for row in report.previous_calibration_rows(tmp_path, [model]) if row["method"] == "ML"]
    assert all(row["improvement_pp"] == 15 for row in rows)


def test_previous_comparison_canonicalizes_equivalent_source_paths(tmp_path):
    model = previous_comparison_fixture(tmp_path)
    path = tmp_path / "previous_calibration/models" / model / "ml_teacher_forced/per_batch.csv"
    rows = report.read_csv(path)
    original = Path(rows[0]["schedule_path"])
    original.parent.mkdir(parents=True)
    alias = tmp_path / "source_alias"
    alias.symlink_to(original.parent, target_is_directory=True)
    for row in rows:
        row["schedule_path"] = str(alias / original.name)
    write_csv(path, rows)
    assert len(report.previous_calibration_rows(tmp_path, [model])) == 12


def test_previous_comparison_requires_shared_ml_aic_real_steps(tmp_path):
    model = previous_comparison_fixture(tmp_path)
    for folder in (tmp_path / "models" / model / "holdout/aic_teacher_forced",
                   tmp_path / "previous_calibration/models" / model / "aic_teacher_forced"):
        rows = report.read_csv(folder / "per_batch.csv")
        rows[0]["sum_past"] = "100"
        write_csv(folder / "per_batch.csv", rows)
    with pytest.raises(ValueError, match="different individual real observations"):
        report.previous_calibration_rows(tmp_path, [model])


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_previous_comparison_rejects_changed_summary_groups(tmp_path, change):
    model = previous_comparison_fixture(tmp_path)
    path = tmp_path / "previous_calibration/models" / model / "ml_teacher_forced/summary.csv"
    rows = report.read_csv(path)
    write_csv(path, rows[:-1] if change == "missing" else rows + [rows[0]])
    with pytest.raises(ValueError, match="summary group"):
        report.previous_calibration_rows(tmp_path, [model])


def test_previous_comparison_requires_individual_scores(tmp_path):
    model = previous_comparison_fixture(tmp_path)
    (tmp_path / "previous_calibration/models" / model / "ml_teacher_forced/per_batch.csv").unlink()
    with pytest.raises(FileNotFoundError, match="per_batch.csv"):
        report.previous_calibration_rows(tmp_path, [model])


def test_kv_input_report_discloses_frozen_input_only_scope(tmp_path, monkeypatch):
    from sklearn.dummy import DummyRegressor
    model, bundle, scored_path, _ = fit_input_fixture(tmp_path)
    for role in ("calibration", "holdout"):
        for group in (tmp_path / "models" / model / role / "real").glob("repeat-*"):
            (group / "BASELINE_COLLECTION_COMPLETE").write_text("complete\n")
            for path in (group / "cases").glob("*/TP0.schedule_batch.jsonl"):
                (path.parent / "status.txt").write_text("ok\n")
                values = [json.loads(line) for line in path.read_text().splitlines()]
                for value in values:
                    value["request_infos"][0]["output_ids_len"] = 0 if value["forward_mode"] == 1 else 3
                path.write_text("".join(json.dumps(value) + "\n" for value in values))
    scored = report.read_csv(scored_path)
    for row in scored:
        value = json.loads(Path(row["schedule_path"]).read_text().splitlines()[int(row["line_number"])-1])
        row.update(batch_size="1", sum_extend=str(value["request_infos"][0]["extend_input_len"]),
                   sum_past=str(value["request_infos"][0]["prefix_indices_len"]))
    write_csv(scored_path, scored)
    before = bundle.read_bytes(), scored_path.read_bytes()
    def refuse_fit(*args, **kwargs):
        raise AssertionError("report must not refit frozen bundles")
    monkeypatch.setattr(DummyRegressor, "fit", refuse_fit)
    output = tmp_path / "kv_report"
    output.mkdir()
    lines = report.kv_input_contract_details(tmp_path, [model], output, Path("figures"))
    text = "\n".join(lines)
    assert "没有重新训练" in text and "不是独立测量 GPU seq_lens" in text
    assert "本表只测 ML" in text and "不是按新口径重新训练后的准确率" in text
    assert (output / "kv_input_contract/ml_frozen_kv_input_mape.png").is_file()
    assert before == (bundle.read_bytes(), scored_path.read_bytes())


def test_kv_input_report_requires_formal_raw_data(tmp_path):
    path = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(FileNotFoundError, match="missing formal KV input audit data"):
        report.kv_input_contract_details(tmp_path, ["model"], tmp_path, Path("figures"))


def test_aic_input_report_calls_cpu_diagnostic_with_declared_stage(tmp_path, monkeypatch):
    import subprocess
    import sys

    provenance = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"aic_root": str(tmp_path / "aic-source")}))
    config = tmp_path / "aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"
    config.parent.mkdir()
    config.write_text("{}")
    (tmp_path / "AIC_ALL_MODELS_COMPLETE").write_text("complete\n")
    stage = str(tmp_path / "kernel-stage")
    (tmp_path / "run_metadata.txt").write_text(f"python={sys.executable}\nkernel_compat_stage={stage}\n")
    output = tmp_path / "aic_report"
    output.mkdir()
    observed = []
    def synthetic_run(command, **kwargs):
        observed.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == stage
        assert kwargs["check"]
        target = Path(command[command.index("--out-dir")+1])
        target.mkdir()
        rows = [{"model": "model", "group": group, "value": value, "n": 3,
                 "mean_actual_ms": 10, "mean_legacy_prediction_ms": 7, "mean_runtime_prediction_ms": 8,
                 "mean_runtime_inferredphase_prediction_ms": 9, "runtime_phase_disagreement_rows": 1,
                 "legacy_mape_pct": 30, "runtime_mape_pct": 20, "runtime_inferredphase_mape_pct": 10, "improvement_pp": 10}
                for group, value in (("all", "all"), ("phase", "1"), ("phase", "2"))]
        write_csv(target / "aic_input_comparison.csv", rows)
    monkeypatch.setattr(subprocess, "run", synthetic_run)
    lines = report.aic_kv_input_contract_details(tmp_path, ["model"], output, Path("figures"))
    text = "\n".join(lines)
    assert len(observed) == 1 and "audit_pr33824_aic_kv_inputs.py" in observed[0][4]
    assert "7.00 / 8.00 / 9.00" in text and "30.00% / 20.00% / 10.00%" in text
    assert "分组仍按真机阶段" in text
    assert "没有重新做端到端回放" in text and "不互相代替" in text
    assert (output / "aic_kv_input_contract/aic_frozen_kv_input_mape.png").is_file()
    assert (output / "aic_kv_input_contract.command.json").is_file()


def test_aic_input_report_requires_completed_formal_aic_stage(tmp_path):
    provenance = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}")
    with pytest.raises(FileNotFoundError, match="missing formal AIC input audit config"):
        report.aic_kv_input_contract_details(tmp_path, ["model"], tmp_path, Path("figures"))
    config = tmp_path / "aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"
    config.parent.mkdir()
    config.write_text("{}")
    (tmp_path / "AIC_ALL_MODELS_COMPLETE").write_text("incomplete\n")
    with pytest.raises(ValueError, match="wait for the formal AIC stage"):
        report.aic_kv_input_contract_details(tmp_path, ["model"], tmp_path, Path("figures"))


def test_runtime_refit_report_requires_completed_native_stage(tmp_path):
    assert report.ml_runtime_refit_details(tmp_path, ["model"], tmp_path, Path("figures")) == []
    provenance = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}")
    (tmp_path / "REAL_ML_ALL_MODELS_COMPLETE").write_text("incomplete\n")
    with pytest.raises(ValueError, match="wait for all native collection"):
        report.ml_runtime_refit_details(tmp_path, ["model"], tmp_path, Path("figures"))


def test_runtime_refit_report_separates_refit_from_frozen_and_e2e(tmp_path, monkeypatch):
    provenance = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}")
    (tmp_path / "REAL_ML_ALL_MODELS_COMPLETE").write_text("complete\n")
    out = tmp_path / "output"
    out.mkdir()
    def synthetic(root, models, directory, folder, script):
        assert folder == "ml_runtime_refit" and script == "refit_pr33824_runtime_kv_ml.py"
        target = directory / folder
        target.mkdir()
        write_csv(target / "ml_runtime_refit_comparison.csv", [{"model": "model", "group": "all", "value": "all",
            "n": 12, "mean_actual_ms": 10, "baseline_prefix_mean_predicted_ms": 7,
            "frozen_runtime_mean_predicted_ms": 8, "refit_runtime_mean_predicted_ms": 9,
            "baseline_prefix_mape_pct": 30, "frozen_runtime_mape_pct": 20, "refit_runtime_mape_pct": 10}])
        (target / "source_manifest.json").write_text(json.dumps({"models": [{"model": "model", "training_rows": 40}]}))
        return target
    monkeypatch.setattr(report, "run_final_cpu_diagnostic", synthetic)
    lines = report.ml_runtime_refit_details(tmp_path, ["model"], out, Path("figures"))
    text = "\n".join(lines)
    assert "7.00 / 8.00 / 9.00" in text and "30.00% / 20.00% / 10.00%" in text
    assert "测试记录在拟合、保存 bundle 后才读取" in text and "没有用新 bundle 重做端到端回放" in text
    assert (out / "ml_runtime_refit/ml_runtime_kv_refit_mape.png").is_file()


def test_previous_comparison_rejects_shared_but_wrong_holdout_source(tmp_path):
    model = previous_comparison_fixture(tmp_path)
    for folder in (tmp_path / "models" / model / "holdout/ml_teacher_forced",
                   tmp_path / "previous_calibration/models" / model / "ml_teacher_forced"):
        rows = report.read_csv(folder / "per_batch.csv")
        for row in rows:
            row["schedule_path"] = str(tmp_path / "old_experiment/case/TP0.schedule_batch.jsonl")
        write_csv(folder / "per_batch.csv", rows)
    with pytest.raises(ValueError, match="source is not the new holdout"):
        report.previous_calibration_rows(tmp_path, [model])


def request_identity_fixture(tmp_path):
    model = "Qwen3-0.6B"
    (tmp_path / "run_metadata.txt").write_text("training_requests_total=2\nheldout_requests_total=2\n")
    for role, payloads in (("calibration", [([1, 2], 8), ([3], 4)]),
                           ("holdout", [([1, 2], 9), ([5], 4)])):
        folder = tmp_path / "models" / model / role / "real/repeat-1/cases/fixture"
        folder.mkdir(parents=True)
        raw = [{"rid": f"{role}-{index}", "input_ids": prompt, "input_length": len(prompt), "output_length": length}
               for index, (prompt, length) in enumerate(payloads)]
        trace = [{"prompt": item["input_ids"], "prompt_len": item["input_length"], "output_len": item["output_length"], "metadata": {"source_rid": item["rid"]}} for item in raw]
        (folder / "TP0.raw_request.jsonl").write_text("".join(json.dumps(item) + "\n" for item in raw))
        (folder / "autobench_trace.jsonl").write_text("".join(json.dumps(item) + "\n" for item in trace))
    out = tmp_path / "identity_report"
    out.mkdir()
    return model, out


def test_request_identity_distinguishes_prompt_and_full_payload_overlap(tmp_path):
    model, out = request_identity_fixture(tmp_path)
    lines = report.request_identity_details(tmp_path, [model], out)
    row = report.read_csv(out / "train_test_request_identity.csv")[0]
    assert float(row["heldout_prompt_overlap_pct"]) == 50
    assert float(row["heldout_input_and_output_length_overlap_pct"]) == 0
    assert row["real_to_replay_ids_verified"] == "4"
    assert any("50.00%" in line for line in lines)


def test_request_identity_rejects_changed_replay_tokens(tmp_path):
    model, out = request_identity_fixture(tmp_path)
    path = tmp_path / "models" / model / "holdout/real/repeat-1/cases/fixture/autobench_trace.jsonl"
    trace = [json.loads(line) for line in path.read_text().splitlines()]
    trace[0]["prompt"] = [99, 100]
    path.write_text("".join(json.dumps(item) + "\n" for item in trace))
    with pytest.raises(ValueError, match="replay changes actual input"):
        report.request_identity_details(tmp_path, [model], out)


def historical_request_identity_fixture(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    model, generated = request_identity_fixture(previous)
    (previous / "run_metadata.txt").write_text(
        "created_at=2026-09-18T15:24:01+08:00\n"
        "protocol=same_workload_independent_train_holdout\n"
        "training_requests_total=2\nheldout_requests_total=2\n"
        "training_real_repeats=1\nheldout_real_repeats=1\n"
        "training_case=fixture\nheldout_case=fixture\n"
        "training_seeds=1\nheldout_seeds=2\n"
    )
    (previous / "FINAL_REPORT_COMPLETE").write_text("complete\n")
    report.request_identity_details(previous, [model], generated)
    current = tmp_path / "current"
    folder = current / "verification/historical_request_identity_all13_fixture"
    folder.mkdir(parents=True)
    summary = folder / "train_test_request_identity.csv"
    summary.write_bytes((generated / summary.name).read_bytes())
    helper = folder / "executed_report_helper.py"
    helper.write_bytes((ROOT / "tools/append_pr33824_expanded_report.py").read_bytes())
    inputs = [previous / "run_metadata.txt", previous / "FINAL_REPORT_COMPLETE",
              *sorted(previous.glob("models/**/TP0.raw_request.jsonl")),
              *sorted(previous.glob("models/**/autobench_trace.jsonl"))]
    write_csv(folder / "source_sha256.csv", [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs])
    manifest = {"previous_root": str(previous), "models": [model], "closed_cases": 2,
                "verified_payload_requests": 4, "input_files_verified_unchanged": len(inputs),
                "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "executed_helper_source_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                "executed_helper_function": "request_identity_details",
                "protocol": "same_workload_independent_train_holdout",
                "training_seeds": ["1"], "heldout_seeds": ["2"],
                "no_fit_prediction_gpu_or_historical_mutation": True}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    reference = current / "previous_calibration/reference.json"
    reference.parent.mkdir()
    reference.write_text(json.dumps({"previous_root": str(previous)}))
    out = current / "analysis"
    out.mkdir()
    return current, previous, model, folder, manifest, out


def test_historical_request_identity_distinguishes_independent_matrix_and_row_split(tmp_path):
    current, _, model, _, _, out = historical_request_identity_fixture(tmp_path)
    text = "\n".join(report.historical_request_identity_details(current, [model], out))
    assert "2026-09-18" in text and "2 个闭合用例、4 个请求" in text
    assert "并非上游脚本按 batch 行划分的内部验证" in text
    assert "50.00%" in text  # Nonzero prompt overlap is not silently replaced by zero.
    assert "不将其当作拟合过程或准确率的证明" in text
    assert "不能把两轮的误差变化仅解释成测试请求不同" in text
    assert report.read_csv(out / "historical_train_test_request_identity.csv")[0]["real_to_replay_ids_verified"] == "4"


@pytest.mark.parametrize("change", ["summary", "helper", "input", "seeds", "counts", "coverage", "reference", "percentage"])
def test_historical_request_identity_rejects_inconsistent_evidence(tmp_path, change):
    current, previous, model, folder, manifest, out = historical_request_identity_fixture(tmp_path)
    if change == "summary":
        (folder / "train_test_request_identity.csv").write_text("changed")
    elif change == "helper":
        (folder / "executed_report_helper.py").write_text("changed")
    elif change == "input":
        next(previous.glob("models/**/autobench_trace.jsonl")).write_text("changed")
    elif change == "seeds":
        manifest["heldout_seeds"] = ["1"]
    elif change == "coverage":
        manifest["closed_cases"] = 1
    elif change == "reference":
        (current / "previous_calibration/reference.json").write_text(json.dumps({"previous_root": str(tmp_path / "other")}))
    else:
        path = folder / "train_test_request_identity.csv"
        rows = report.read_csv(path)
        rows[0]["real_to_replay_ids_verified" if change == "counts" else "heldout_prompt_overlap_pct"] = "99" if change == "counts" else "nan"
        write_csv(path, rows)
        manifest["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        report.historical_request_identity_details(current, [model], out)


def test_historical_request_identity_does_not_invent_missing_audit(tmp_path):
    assert report.historical_request_identity_details(tmp_path, ["unused"], tmp_path) == []
    required = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    required.parent.mkdir(parents=True)
    required.write_text("{}")
    with pytest.raises(FileNotFoundError, match="missing historical request-identity audit"):
        report.historical_request_identity_details(tmp_path, ["unused"], tmp_path)


def test_aic_snapshot_checks_archived_bytes_not_just_revision(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing formal AIC"):
        report.aic_source_snapshot_details(tmp_path, "fixture", required=True)
    folder = tmp_path / "aic_calibration/provenance_before_wide_sweep"
    folder.mkdir(parents=True)
    code = tmp_path / "collect_attn.py"
    code.write_text("# fixture, not measured data\n")
    identity = {"aic_revision": "fixture", "aic_source_snapshot": {"source_files": 1, "dirty": True},
                "aic_import_source_match": {"checked_python_files": 133, "mismatches": []}}
    (folder / "identity.json").write_text(json.dumps(identity))
    (folder / "aic_git_status.txt").write_text(" M collector/sglang/collect_attn.py\n")
    (folder / "aic_tracked_changes.patch").write_text("fixture patch\n")
    name = "collector/sglang/collect_attn.py"
    hashes = {name: hashlib.sha256(code.read_bytes()).hexdigest()}
    (folder / "aic_source_sha256.json").write_text(json.dumps(hashes))
    with tarfile.open(folder / "aic_collector_and_sdk_sources.tar.gz", "w:gz") as archive:
        archive.add(code, arcname=name)
    assert any("133" in line for line in report.aic_source_snapshot_details(tmp_path, "fixture", required=True))
    hashes[name] = "0" * 64
    (folder / "aic_source_sha256.json").write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="checksum mismatch"):
        report.aic_source_snapshot_details(tmp_path, "fixture", required=True)


def test_runtime_identity_checks_collector_against_all_real_groups(tmp_path):
    model = "Qwen3-0.6B"
    reference = tmp_path / "provenance/formal_start_after_ninja_fix/identity.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}")
    (tmp_path / "run_metadata.txt").write_text("source_revision=test-sha\ntraining_real_repeats=1\nheldout_real_repeats=1\n")
    aic = tmp_path / "aic_calibration"
    (aic / "environment").mkdir(parents=True)
    (aic / "trace_targeted_run_metadata.txt").write_text("source_revision=test-sha\ncollector_triton_page_size=1\n")
    common = "torch=2.9.1+cu126\ntorch_cuda=12.6\nsgl_kernel_runtime_version=0.4.6.post1\nsgl_kernel_file=/tmp/fixture-native-stage/sgl_kernel/__init__.py\n"
    (aic / "environment/collector_runtime_versions.txt").write_text(common)
    for role in ("calibration", "holdout"):
        group = tmp_path / "models" / model / role / "real/repeat-1"
        group.mkdir(parents=True)
        (group / "run_metadata.txt").write_text("source_revision=test-sha\n")
        (group / "package_versions.txt").write_text(common + "python=3.10\nsglang=0.0.0\nsglang-kernel=0.4.6.post1+cu129\n")
    out = tmp_path / "report"
    out.mkdir()
    lines = report.runtime_identity_details(tmp_path, [model], out)
    assert any("0.4.6.post1+cu129" in line for line in lines)
    assert report.read_csv(out / "runtime_identity_checks.csv")[0]["recorded_real_groups"] == "2"
    version = tmp_path / "models" / model / "holdout/real/repeat-1/package_versions.txt"
    version.write_text(version.read_text().replace("2.9.1+cu126", "2.14+cpu"))
    with pytest.raises(ValueError, match="versions changed"):
        report.runtime_identity_details(tmp_path, [model], out)


def coverage_fixture(tmp_path):
    model, case = "Qwen3-0.6B", "latency_256x128_r1"
    def sample(mode, extend, past, latency=.01):
        return {"forward_mode": mode, "iter_latency": latency,
                "request_infos": [{"extend_input_len": extend, "prefix_indices_len": past}]}
    for role, values in (("calibration", [sample(1, 2, 0), sample(1, 4, 4), sample(2, 1, 1), sample(2, 1, 10), sample(1, 9999, 9999, 31)]),
                         ("holdout", [sample(1, 3, 2), sample(2, 1, 20)])):
        path = tmp_path / "models" / model / role / "real/repeat-1/cases" / case / "TP0.schedule_batch.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(value) + "\n" for value in values))
    metrics = [{"model": model, "case": case, "phase": phase, "method": method, "n": str(2 if phase == "all" else 1), "mape_pct": "10"}
               for phase in ("all", "prefill", "decode") for method in ("ML", "AIC")]
    out = tmp_path / "coverage"
    out.mkdir()
    return model, case, metrics, out


def test_feature_coverage_matches_phases_and_training_filter(tmp_path):
    model, case, metrics, out = coverage_fixture(tmp_path)
    lines = report.workload_feature_coverage_details(tmp_path, [model], [case], metrics, out, Path("coverage"))
    rows = {row["phase"]: row for row in report.read_csv(out / "workload_feature_coverage.csv")}
    assert float(rows["all"]["outside_train_range_pct"]) == 50
    assert float(rows["prefill"]["outside_train_range_pct"]) == 0
    assert float(rows["decode"]["outside_train_range_pct"]) == 100
    assert int(rows["all"]["train_rows"]) == 4
    assert len(list(out.glob("*.png"))) == 4
    assert any("范围内不代表同分布" in line for line in lines)


def test_feature_coverage_rejects_different_scoring_counts(tmp_path):
    model, case, metrics, out = coverage_fixture(tmp_path)
    metrics[0]["n"] = "3"
    with pytest.raises(ValueError, match="batch counts differ"):
        report.workload_feature_coverage_details(tmp_path, [model], [case], metrics, out, Path("coverage"))


def test_historical_baselines_keep_real_values_and_metric_scopes(tmp_path):
    model = "Qwen2.5-3B-Instruct"
    current = tmp_path / "current"
    current.mkdir()
    out = current / "plots"
    out.mkdir()
    for name in ("pr33824_piecewise_a100_dense3_matrix_20260908_134523", "pr33824_a100_dense3_sm80compat_matrix_20260910_164000"):
        source = tmp_path / name
        source.mkdir()
        (source / "MATRIX_COMPLETE").write_text("complete\n")
        (source / "run_metadata.txt").write_text('source_revision=fixture\nreal_cuda_graph_config={"decode":"full"}\nreal_overlap_schedule_enabled=1\ntrace_policy=fixed seed\n')
        settings = source / "results" / model
        settings.mkdir(parents=True)
        (settings / "context-length.txt").write_text("8192\n")
        (settings / "real-mem-fraction-static.txt").write_text("0.80\n")
        rows = [{"model": model, "case": case, "complete_pairs": "5", "num_prompts": "32", "real_mean_e2e_latency_ms_mean": "100", "simulated_mean_e2e_latency_ms_mean": "120", "absolute_error_pct_mean_e2e_latency_ms_mean": "20"}
                for case in ("latency_256x128_r1", "service_1024x256_r4", "saturation_1024x256_inf")]
        write_csv(source / "analysis/case_metric_statistics.csv", rows)
        write_csv(source / "analysis/model_metric_statistics.csv", [{"model": model, "metric": metric, "paired_runs": "15", "mape_mean_pct": "20"} for metric in ("mean_e2e_latency_ms", "mean_ttft_ms")])
    lines = report.historical_aic_baseline_details(current, [model], out, Path("plots"))
    assert len(report.read_csv(out / "historical_aic_baselines.csv")) == 6
    assert any("100.00 | 120.00" in line for line in lines)
    assert any("不代表" in line or "不能" in line for line in lines)
    assert (out / "historical_aic_real_and_predicted.png").is_file()


def historical_one_batch_fixture(root):
    model = "Qwen2.5-3B-Instruct"
    current = root / "current"
    current.mkdir()
    output = current / "figures"
    output.mkdir()
    paths = []
    for folder in ("pr33824_aic_vs_real_single_step_dense3_20260908_155211",
                   "pr33824_aic_vs_real_single_step_dense3_sm80compat_20260910_181204"):
        source = root / folder
        source.mkdir()
        (source / "finalization.status").write_text("status=complete\n")
        (source / "run_metadata.txt").write_text('source_revision=fixture\ninput_len=1024\noutput_len=2\nrepeats=5\ncontext_length=8192\nreal_cuda_graph_config={"decode":"full","prefill":"piecewise"}\n')
        directory = source / "models" / model
        (directory / "real").mkdir(parents=True)
        (directory / "real/status.txt").write_text("ok\n")
        (directory / "aic_query.status.txt").write_text("ok\n")
        (directory / "real/command.txt").write_text("python -m sglang.benchmark.one_batch --batch-size 1 --input-len 1024 --output-len 2\n")
        raw = [{"batch_size": 1, "input_len": 1024, "output_len": 2,
                "prefill_latency": value / 1000, "median_decode_latency": value * 2 / 1000}
               for value in (1, 2, 3, 2, 2)]
        (directory / "real/one_batch.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
        (directory / "aic_prediction.json").write_text(json.dumps({"batch_size": 1, "input_len": 1024,
              "database_path": "fixture-db", "phases": {"prefill": {"predicted_ms": 2}, "decode": {"predicted_ms": 4}}}))
        rows = [{"model": model, "phase": phase, "real_samples": 5, "real_mean_ms": mean,
                 "real_stddev_ms": mean * (.5 ** .5) / 2, "aic_predicted_ms": mean,
                 "signed_error_pct": 0, "absolute_error_pct": 0}
                for phase, mean in (("prefill", 2), ("decode", 4))]
        write_csv(source / "analysis/single_step_comparison.csv", rows)
        paths.append(source)
    return model, current, output, paths


def test_historical_one_batch_separates_error_of_mean_from_repeat_mape(tmp_path):
    model, current, output, sources = historical_one_batch_fixture(tmp_path)
    originals = [(source / "analysis/single_step_comparison.csv").read_bytes() for source in sources]
    lines = report.historical_aic_one_batch_details(current, [model], output, Path("figures"))
    rows = report.read_csv(output / "historical_aic_fixed_shape_steps.csv")
    assert len(rows) == 4
    assert all(float(row["absolute_error_of_mean_pct"]) == 0 for row in rows)
    assert all(float(row["individual_repeat_mape_pct"]) == pytest.approx(80 / 3) for row in rows)
    assert [(source / "analysis/single_step_comparison.csv").read_bytes() for source in sources] == originals
    assert "不是逐次 APE 的平均" in "\n".join(lines)
    assert "不改写历史文件" in "\n".join(lines)
    assert "26.6667 个百分点" in "\n".join(lines)
    assert (output / "historical_aic_one_batch_actual_vs_predicted.png").is_file()


@pytest.mark.parametrize("change,expected", [("summary", "CSV disagrees with raw data"),
                                            ("duplicate", "missing/duplicate"),
                                            ("samples", "wrong historical fixed-shape raw samples"),
                                            ("status", "experiment incomplete")])
def test_historical_one_batch_rejects_inconsistent_evidence(tmp_path, change, expected):
    model, current, output, sources = historical_one_batch_fixture(tmp_path)
    source = sources[0]
    if change in ("summary", "duplicate"):
        path = source / "analysis/single_step_comparison.csv"
        rows = report.read_csv(path)
        if change == "summary":
            rows[0]["absolute_error_pct"] = 1
        else:
            rows.append(dict(rows[0]))
        write_csv(path, rows)
    elif change == "samples":
        path = source / "models" / model / "real/one_batch.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    else:
        (source / "finalization.status").write_text("status=failed\n")
    with pytest.raises(ValueError, match=expected):
        report.historical_aic_one_batch_details(current, [model], output, Path("figures"))


def test_report_figures_and_simsun_docx(tmp_path, monkeypatch):
    model, case = fixture_results(tmp_path)
    bundle_dir = tmp_path / "models" / model / "calibration/training"
    bundle_dir.mkdir(parents=True)
    joblib.dump({"model": SimpleNamespace(n_iter_=20), "train_rows": 30, "train_cases": 15,
                 "best_params": {"random_state": 42}, "features": ["batch_size"],
                 "monotonic": [1], "max_latency_seconds": 30}, bundle_dir / f"{model}_hgbmono_p50.pkl")
    (tmp_path / "run_metadata.txt").write_text(f"models={model}\ntraining_case={case}\nheldout_case={case}\nheldout_real_repeats=5\n")
    markdown = tmp_path / "fixture.md"
    markdown.write_text("# Fixture, not experiment results\n")
    monkeypatch.setattr(sys, "argv", ["report", "--results-root", str(tmp_path), "--markdown", str(markdown), "--out-dir", str(tmp_path / "plots")])
    report.main()
    assert len(list((tmp_path / "plots").glob("*.png"))) == 5
    assert "tok/s" in markdown.read_text()
    assert "prefill 实测/预测" in markdown.read_text()
    assert "TTFT 实测 ± std / 预测" in markdown.read_text()
    assert "50%/50%、seed=42、forward_modes=1" in markdown.read_text()
    assert "9cfddcedc9d1e00885c28db75a45513eddbb6271" in markdown.read_text()
    assert "3 条 ShareGPT 请求，每条输出 4 token" in markdown.read_text()
    assert "固定 1 ms 的 DummyRegressor" in markdown.read_text()
    assert "本轮不使用该示例预测器" in markdown.read_text()
    assert "examples/build_example_assets.py" in markdown.read_text()
    assert "ElementWise" in markdown.read_text()
    assert "3 μs" in markdown.read_text()
    assert "解析项与本机实测表" in markdown.read_text()
    assert "QK norm" in markdown.read_text()
    assert "不是 GPU 实测时延" in markdown.read_text()
    assert "计时前调用 init_forward_metadata" in markdown.read_text()
    assert "不代表每个请求固定使用 8 个 split" in markdown.read_text()
    assert "1,920 次子函数调用，这不是实测 GPU kernel 数或时延" in markdown.read_text()
    assert "253,952，本轮 context=2,048" in markdown.read_text()
    assert "不据此解释其余普通 dense 模型的误差" in markdown.read_text()
    assert "capture 前后分别预热" in markdown.read_text()
    assert "1–6 组独立矩阵" in markdown.read_text()
    assert "这不足以证明实际缓存命中率或它造成的误差" in markdown.read_text()
    assert "不能据此给预测值补一个固定延迟" in markdown.read_text()
    assert "3B、7B、32B 抽样 profile" in markdown.read_text()
    assert "不是元数据准备、编译等开销的逐项受控对照" in markdown.read_text()
    assert "不单独测量缓存命中率" in markdown.read_text()
    assert "算子族统计不等于隔离 QK norm 的单项计时" in markdown.read_text()
    assert "尚无同配置逐项 profile" not in markdown.read_text()
    assert "没有当前逐项 profile 证据" not in markdown.read_text()
    spec = importlib.util.spec_from_file_location("docx_renderer", ROOT / "scripts/render_markdown_report_docx.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    output = tmp_path / "fixture.docx"
    renderer.render(markdown, output)
    from docx import Document
    from docx.oxml.ns import qn
    document = Document(output)
    assert len(document.inline_shapes) == 5
    fonts = document.element.xpath("//w:rFonts")
    assert fonts and all(node.get(qn("w:ascii")) == "SimSun" for node in fonts)
