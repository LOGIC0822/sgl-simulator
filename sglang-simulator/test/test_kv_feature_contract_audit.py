"""Input-contract audit tests; no timing predictions or GPU execution."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kv_contract_audit", SIM / "tools/audit_pr33824_kv_feature_contract.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(tmp_path):
    files = []
    for role in ("calibration", "holdout"):
        group = tmp_path / "models/model" / role / "real/repeat-1"
        case = group / "cases/example"
        case.mkdir(parents=True)
        (group / "BASELINE_COLLECTION_COMPLETE").write_text("complete\n")
        (case / "status.txt").write_text("ok\n")
        rows = [{"forward_mode": mode, "iter_latency": .01,
                 "request_infos": [{"extend_input_len": extend, "prefix_indices_len": 20, "output_ids_len": generated}]}
                for mode, extend, generated in ((1, 4, 0), (2, 1, 1), (2, 1, 7))]
        path = case / "TP0.schedule_batch.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        files.append(path)
    return files


def test_audit_counts_changed_decode_inputs_without_mutating_sources(tmp_path):
    files = fixture(tmp_path)
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
    rows, manifest = audit.audit(tmp_path, ["model"])
    assert len(rows) == 4 and len(manifest) == 2
    prefill = next(row for row in rows if row["role"] == "holdout" and row["forward_mode"] == 1)
    decode = next(row for row in rows if row["role"] == "holdout" and row["forward_mode"] == 2)
    assert prefill["changed_rows"] == 0
    assert decode["rows"] == decode["changed_rows"] == 2
    assert decode["changed_18_feature_vector_pct"] == 100
    assert decode["mean_offline_prefix_tokens_per_request"] == 20
    assert decode["mean_runtime_prefix_plus_output_tokens_per_request"] == 24
    assert decode["mean_added_output_tokens_per_request"] == 4
    assert decode["max_added_output_tokens_per_request"] == 7
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]


@pytest.mark.parametrize("change", ["missing", "negative", "bool"])
def test_audit_refuses_unknown_or_invalid_generated_lengths(tmp_path, change):
    path = fixture(tmp_path)[0]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "missing":
        del rows[0]["request_infos"][0]["output_ids_len"]
    else:
        rows[0]["request_infos"][0]["output_ids_len"] = -1 if change == "negative" else True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((KeyError, ValueError)):
        audit.audit(tmp_path, ["model"])


def test_audit_refuses_incomplete_group(tmp_path):
    path = fixture(tmp_path)[0]
    (path.parents[2] / "BASELINE_COLLECTION_COMPLETE").write_text("incomplete\n")
    with pytest.raises(ValueError, match="incomplete collected group"):
        audit.audit(tmp_path, ["model"])


def test_audit_checks_current_runtime_and_offline_source_expressions():
    source = audit.source_contract(SIM)
    assert set(source) == {"runtime", "trainer", "ml_scorer", "aic_scorer"}
    assert source["runtime"]["length_expressions"] == ["len(req.prefix_indices) + len(req.output_ids)"] * 2
    assert all("prefix_indices_len" in source[key]["length_expressions"][0] for key in ("trainer", "ml_scorer", "aic_scorer"))


def frozen_fixture(tmp_path):
    import joblib
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sglang_simulator.time_predictor.ml_features import ML_FEATURE_NAMES

    files = fixture(tmp_path)
    features = np.zeros((2, len(ML_FEATURE_NAMES)))
    features[:, 4] = [0, 20]
    predictor = LinearRegression().fit(features, [0, .002])
    bundle = tmp_path / "models/model/calibration/training/model_hgbmono_p50.pkl"
    bundle.parent.mkdir(parents=True)
    joblib.dump({"model": predictor, "features": list(ML_FEATURE_NAMES), "target": "iter_latency_seconds"}, bundle)
    score = tmp_path / "models/model/holdout/ml_teacher_forced/per_batch.csv"
    score.parent.mkdir(parents=True)
    values = [json.loads(line) for line in files[1].read_text().splitlines()]
    rows = [{"case": "example", "schedule_path": str(files[1]), "line_number": index,
             "forward_mode": value["forward_mode"], "batch_size": 1,
             "sum_extend": value["request_infos"][0]["extend_input_len"], "sum_past": 20,
             "actual_seconds": .01, "predicted_seconds": .002}
            for index, value in enumerate(values, start=1)]
    with score.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "frozen_comparison"
    out.mkdir()
    return files + [bundle, score], out


def test_frozen_comparison_changes_only_inputs_and_never_fits(tmp_path, monkeypatch):
    from sklearn.linear_model import LinearRegression

    files, out = frozen_fixture(tmp_path)
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
    def refuse_fit(*args, **kwargs):
        raise AssertionError("frozen comparison must not call fit")
    monkeypatch.setattr(LinearRegression, "fit", refuse_fit)
    summaries, identities = audit.compare_frozen_inputs(tmp_path, ["model"], out)
    overall = next(row for row in summaries if row["group"] == "all")
    assert overall["legacy_mape_pct"] == pytest.approx(80)
    assert overall["runtime_mape_pct"] == pytest.approx(77 + 1/3)
    assert overall["mean_actual_ms"] == pytest.approx(10)
    prefill = next(row for row in summaries if row["group"] == "phase" and row["value"] == "1")
    assert prefill["legacy_mape_pct"] == prefill["runtime_mape_pct"]
    assert identities[0]["official_predictions_reproduced"]
    assert not identities[0]["fit_invoked"]
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]


def test_frozen_comparison_refuses_changed_official_predictions(tmp_path):
    files, out = frozen_fixture(tmp_path)
    score = files[-1]
    with score.open() as source:
        rows = list(csv.DictReader(source))
    rows[0]["predicted_seconds"] = ".003"
    with score.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="does not reproduce official"):
        audit.compare_frozen_inputs(tmp_path, ["model"], out)
