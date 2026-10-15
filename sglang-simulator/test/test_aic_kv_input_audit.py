"""Synthetic AIC input-contract tests; no GPU, AIC SDK, or native-model loading."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("aic_kv_audit", Path(__file__).resolve().parents[1] / "tools/audit_pr33824_aic_kv_inputs.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class SyntheticScorer:
    @staticmethod
    def predict_rows(rows, predictor):
        # Test double only: deterministic dependence on past length and phase.
        values = []
        for row in rows:
            prediction = predictor(row)
            values.append({**{key: value for key, value in row.items() if key not in ("extend", "past")},
                           "predicted_seconds": prediction})
        return values, len(rows)


def prediction(row):
    return sum(row["past"]) * .0001 + (0 if row["forward_mode"] == 2 else .001)


def fixture(tmp_path):
    model = tmp_path / "models/model"
    group = model / "holdout/real/repeat-1"
    case = group / "cases/example"
    case.mkdir(parents=True)
    (group / "BASELINE_COLLECTION_COMPLETE").write_text("complete\n")
    (case / "status.txt").write_text("ok\n")
    # One-token extend must remain prefill; it must not be inferred as decode.
    values = [{"forward_mode": mode, "iter_latency": .01, "request_infos": [
        {"extend_input_len": extend, "prefix_indices_len": 20, "output_ids_len": generated}]}
        for mode, extend, generated in ((1, 1, 0), (2, 1, 1), (2, 1, 7))]
    path = case / "TP0.schedule_batch.jsonl"
    path.write_text("".join(json.dumps(value)+"\n" for value in values))
    rows = [{"case": "example", "schedule_path": str(path), "line_number": number,
             "forward_mode": value["forward_mode"], "batch_size": 1, "sum_extend": 1,
             "sum_past": 20, "actual_seconds": .01,
             "predicted_seconds": .003 if value["forward_mode"] == 1 else .002}
            for number, value in enumerate(values, start=1)]
    scored = model / "holdout/aic_teacher_forced/per_batch.csv"
    scored.parent.mkdir(parents=True)
    audit.write_csv(scored, rows)
    return model, path, scored


def test_aic_comparison_changes_only_inputs_with_identical_step_targets(tmp_path):
    model, path, scored = fixture(tmp_path)
    before = path.read_bytes(), scored.read_bytes()
    summaries, details, identity = audit.compare_model(model, prediction, SyntheticScorer)
    overall = next(row for row in summaries if row["group"] == "all")
    assert overall["n"] == 3
    assert overall["mean_actual_ms"] == pytest.approx(10)
    assert overall["mean_legacy_prediction_ms"] == pytest.approx(7/3)
    assert overall["mean_runtime_prediction_ms"] == pytest.approx(7.8/3)
    assert overall["improvement_pp"] == pytest.approx(8/3)
    assert details[0]["forward_mode"] == 1
    assert details[0]["legacy_predicted_seconds"] == details[0]["runtime_predicted_seconds"]
    assert details[0]["runtime_inferred_forward_mode"] == 2
    assert details[0]["runtime_inferredphase_predicted_seconds"] == pytest.approx(.002)
    assert overall["runtime_phase_disagreement_rows"] == 1
    assert overall["runtime_inferredphase_mape_pct"] == pytest.approx(overall["runtime_mape_pct"]+10/3)
    assert details[2]["runtime_sum_past"] == 27
    assert identity["original_predictions_reproduced"]
    assert identity["runtime_phase_disagreement_rows"] == 1
    assert not identity["recollect_or_fit_invoked"]
    assert before == (path.read_bytes(), scored.read_bytes())


@pytest.mark.parametrize("change", ["prediction", "identity", "duplicate", "omit"])
def test_aic_comparison_rejects_different_official_scores_or_coverage(tmp_path, change):
    model, _, scored = fixture(tmp_path)
    with scored.open() as source:
        rows = list(csv.DictReader(source))
    if change == "prediction":
        rows[0]["predicted_seconds"] = .004
    elif change == "identity":
        rows[0]["sum_past"] = 21
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows.pop()
    scored.unlink()
    audit.write_csv(scored, rows)
    with pytest.raises(ValueError):
        audit.compare_model(model, prediction, SyntheticScorer)


@pytest.mark.parametrize("generated", [None, -1, True, 1.5])
def test_aic_comparison_requires_known_nonnegative_integer_output_lengths(tmp_path, generated):
    model, path, _ = fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if generated is None:
        del rows[0]["request_infos"][0]["output_ids_len"]
    else:
        rows[0]["request_infos"][0]["output_ids_len"] = generated
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises((KeyError, ValueError)):
        audit.compare_model(model, prediction, SyntheticScorer)


def test_aic_comparison_rejects_incomplete_collection(tmp_path):
    model, path, _ = fixture(tmp_path)
    (path.parent / "status.txt").write_text("failed\n")
    with pytest.raises(ValueError, match="incomplete collected case"):
        audit.compare_model(model, prediction, SyntheticScorer)


def test_aic_comparison_rejects_mutation_and_invalid_predictions(tmp_path):
    model, path, scored = fixture(tmp_path)
    calls = []
    def mutate(row):
        calls.append(row)
        if len(calls) == 4:
            scored.write_text(scored.read_text()+"\n")
        return prediction(row)
    with pytest.raises(ValueError, match="changed during comparison"):
        audit.compare_model(model, mutate, SyntheticScorer)
    with pytest.raises(ValueError, match="invalid prediction"):
        audit.compare_model(model, lambda row: float("nan"), SyntheticScorer)


def test_database_snapshot_includes_tables_and_refuses_sol(tmp_path):
    directory = tmp_path / "database"
    directory.mkdir()
    table = directory / "gemm_perf.txt"
    table.write_text("synthetic table\n")
    config = tmp_path / "config.json"
    value = {"predictor": {"name": "aiconfigurator", "database_mode": "SILICON", "database_path": str(directory)}}
    config.write_text(json.dumps(value))
    previous = audit.database_snapshot(config)
    assert str(table) in previous
    table.write_text("different table\n")
    assert audit.database_snapshot(config) != previous
    value["predictor"]["database_mode"] = "SOL"
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="explicit AIC SILICON"):
        audit.database_snapshot(config)
