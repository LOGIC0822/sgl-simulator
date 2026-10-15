"""Saved-data ML refit tests with synthetic timings, no CUDA or native server."""
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from sglang_simulator.time_predictor.ml_features import ML_FEATURE_NAMES

SPEC = importlib.util.spec_from_file_location("runtime_ml_refit", Path(__file__).resolve().parents[1] / "tools/refit_pr33824_runtime_kv_ml.py")
refit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refit)


def fixture(tmp_path):
    model = tmp_path / "models/model"
    for role, count in (("calibration", 40), ("holdout", 12)):
        group = model / role / "real/repeat-1"
        case = group / "cases/example"
        case.mkdir(parents=True)
        (group / "BASELINE_COLLECTION_COMPLETE").write_text("complete\n")
        (case / "status.txt").write_text("ok\n")
        values = [{"forward_mode": 2, "iter_latency": .002 + index*.0001,
                   "request_infos": [{"rid": f"{role}-{index}", "extend_input_len": 1,
                                      "prefix_indices_len": 20, "output_ids_len": index+1}]}
                  for index in range(count)]
        (case / "TP0.schedule_batch.jsonl").write_text("".join(json.dumps(value)+"\n" for value in values))
    train, _, roots = refit.read_collection(model, "calibration", 30)
    params = {"loss": "quantile", "quantile": .5, "max_iter": 3, "random_state": 42,
              "min_samples_leaf": 2, "early_stopping": False}
    predictor = HistGradientBoostingRegressor(**params).fit(
        np.asarray([row["legacy_features"] for row in train]), [row["actual_seconds"] for row in train])
    bundle = model / "calibration/training/model_hgbmono_p50.pkl"
    bundle.parent.mkdir(parents=True)
    joblib.dump({"model": predictor, "features": list(ML_FEATURE_NAMES), "target": "iter_latency_seconds",
                 "input_contract": "[[extend_input_len, prefix_indices_len], ...]", "forward_modes": [1, 2],
                 "train_rows": 40, "train_cases": 1, "source_data_roots": roots,
                 "max_latency_seconds": 30, "best_params": params}, bundle)
    holdout, _, _ = refit.read_collection(model, "holdout", 30)
    scores = []
    for row in holdout:
        scores.append({"case": row["case"], "schedule_path": row["schedule_path"], "line_number": row["line_number"],
                       "forward_mode": row["forward_mode"], "batch_size": row["batch_size"], "sum_extend": row["sum_extend"],
                       "sum_past": row["legacy_sum_past"], "actual_seconds": row["actual_seconds"],
                       "predicted_seconds": float(predictor.predict([row["legacy_features"]])[0])})
    score = model / "holdout/ml_teacher_forced/per_batch.csv"
    score.parent.mkdir(parents=True)
    refit.write_csv(score, scores)
    return model, bundle, score


def test_refit_reuses_only_calibration_and_reads_test_after_fit(tmp_path, monkeypatch):
    model, bundle, score = fixture(tmp_path)
    sources = [bundle, score, *model.glob("*/real/repeat-*/cases/*/TP0.schedule_batch.jsonl")]
    before = {str(path): path.read_bytes() for path in sources}
    roles, inputs = [], []
    original_read, original_fit = refit.read_collection, HistGradientBoostingRegressor.fit
    def read(*args, **kwargs):
        roles.append(args[1])
        return original_read(*args, **kwargs)
    def fit(self, features, target, *args, **kwargs):
        assert roles == ["calibration"]
        inputs.append((features.copy(), np.asarray(target).copy(), self.get_params()))
        return original_fit(self, features, target, *args, **kwargs)
    monkeypatch.setattr(refit, "read_collection", read)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", fit)
    out = tmp_path / "refit"
    rows, identity = refit.refit_model(model, out)
    assert roles == ["calibration", "holdout"] and len(inputs) == 1
    assert inputs[0][0].shape == (40, 18)
    assert inputs[0][0][:, 4].tolist() == list(range(21, 61))
    assert inputs[0][1] == pytest.approx([.002+index*.0001 for index in range(40)])
    assert inputs[0][2] == joblib.load(bundle)["model"].get_params()
    assert identity["training_rows"] == 40 and identity["holdout_rows"] == 12
    assert identity["test_read_after_fit"] and identity["original_predictions_reproduced"]
    overall = next(row for row in rows if row["group"] == "all")
    assert overall["n"] == 12
    corrected = joblib.load(out / "model_runtime_kv_p50.pkl")
    assert "output_ids_len" in corrected["input_contract"]
    assert corrected["source_data_roots"] == joblib.load(bundle)["source_data_roots"]
    assert before == {str(path): path.read_bytes() for path in sources}


@pytest.mark.parametrize("change", ["training_roots", "rows", "params", "features", "objective"])
def test_refit_rejects_inconsistent_original_bundle_before_fitting(tmp_path, monkeypatch, change):
    model, bundle, _ = fixture(tmp_path)
    value = joblib.load(bundle)
    if change == "training_roots":
        value["source_data_roots"] = [str(model / "holdout/real/repeat-1/cases")]
    elif change == "rows":
        value["train_rows"] += 1
    elif change == "params":
        value["best_params"]["max_iter"] += 1
    elif change == "features":
        value["features"].reverse()
    else:
        value["model"].set_params(loss="squared_error")
    joblib.dump(value, bundle)
    def refuse_fit(*args, **kwargs):
        raise AssertionError("invalid source configuration must fail before fitting")
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", refuse_fit)
    with pytest.raises(ValueError):
        refit.refit_model(model, tmp_path / "refit")


def test_refit_rejects_unknown_generated_lengths_and_incomplete_groups(tmp_path):
    model, _, _ = fixture(tmp_path)
    path = next(model.glob("calibration/real/repeat-*/cases/*/TP0.schedule_batch.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del rows[0]["request_infos"][0]["output_ids_len"]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(KeyError):
        refit.refit_model(model, tmp_path / "refit")
    (path.parents[2] / "BASELINE_COLLECTION_COMPLETE").write_text("incomplete\n")
    with pytest.raises(ValueError, match="incomplete collected group"):
        refit.read_collection(model, "calibration", 30)


def test_refit_rejects_original_score_identity_and_reproduction_errors(tmp_path):
    model, _, score = fixture(tmp_path)
    rows = list(refit.original_scores(score).values())
    rows[0]["predicted_seconds"] = .5
    score.unlink()
    refit.write_csv(score, rows)
    with pytest.raises(ValueError, match="does not reproduce"):
        refit.refit_model(model, tmp_path / "first-refit")
    rows[0]["sum_past"] = 999
    score.unlink()
    refit.write_csv(score, rows)
    with pytest.raises(ValueError, match="mismatches measured step"):
        refit.refit_model(model, tmp_path / "second-refit")


def test_refit_keeps_existing_output_untouched_without_fitting(tmp_path, monkeypatch):
    model, _, _ = fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("existing result")
    def refuse_fit(*args, **kwargs):
        raise AssertionError("existing output must be rejected before fitting")
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", refuse_fit)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        refit.refit_model(model, output)
    assert sentinel.read_text() == "existing result"


def test_refit_detects_original_artifact_mutation_during_fitting(tmp_path, monkeypatch):
    model, bundle, _ = fixture(tmp_path)
    original_fit = HistGradientBoostingRegressor.fit
    def mutate(self, *args, **kwargs):
        value = original_fit(self, *args, **kwargs)
        # Appended bytes leave this test's loaded estimator valid but change SHA.
        with bundle.open("ab") as output:
            output.write(b"mutation")
        return value
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", mutate)
    with pytest.raises(ValueError, match="changed during refit"):
        refit.refit_model(model, tmp_path / "output")
