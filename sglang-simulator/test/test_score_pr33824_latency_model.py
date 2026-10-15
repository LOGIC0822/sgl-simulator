"""Tests for teacher-forced ML scheduler-batch scoring."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from sglang_simulator.time_predictor.ml_features import ML_FEATURE_NAMES


SCRIPT = Path(__file__).resolve().parents[1] / "tools/score_pr33824_latency_model.py"
SPEC = importlib.util.spec_from_file_location("latency_model_scorer", SCRIPT)
assert SPEC and SPEC.loader
scorer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scorer
SPEC.loader.exec_module(scorer)


def test_score_rows_and_summary_by_forward_mode(tmp_path):
    schedule = tmp_path / "TP0.schedule_batch.jsonl"
    schedule.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "forward_mode": 1,
                        "iter_latency": 0.01,
                        "request_infos": [
                            {
                                "extend_input_len": 4,
                                "prefix_indices_len": 0,
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "forward_mode": 2,
                        "iter_latency": 0.02,
                        "request_infos": [
                            {
                                "extend_input_len": 1,
                                "prefix_indices_len": 4,
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n"
    )
    model = DummyRegressor(strategy="constant", constant=0.015)
    model.fit(np.zeros((1, len(ML_FEATURE_NAMES))), [0.015])

    rows = scorer.read_rows([str(tmp_path)], "TP0*.schedule_batch.jsonl", {1, 2})
    scored = scorer.score_rows(rows, model)
    summary = scorer.summarize(scored)

    assert len(scored) == 2
    assert scored[0]["absolute_percentage_error_pct"] == pytest.approx(50.0)
    assert summary[0]["n"] == 2
    assert summary[0]["mape_pct"] == pytest.approx(37.5)
    assert {row["value"] for row in summary if row["group"] == "forward_mode"} == {
        "1",
        "2",
    }


def test_load_model_rejects_feature_abi_mismatch(tmp_path):
    bundle_path = tmp_path / "bad.pkl"
    joblib.dump(
        {"model": DummyRegressor(), "features": ["wrong"]},
        bundle_path,
    )

    try:
        scorer.load_model(bundle_path)
    except ValueError as error:
        assert "feature ABI" in str(error)
    else:
        raise AssertionError("expected ABI mismatch")
