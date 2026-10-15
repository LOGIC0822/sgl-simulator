"""Contracts for the copied baseline-data ML trainer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pytest

from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from sglang_simulator.time_predictor.ml import MLTimePredictor
from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)


TRAINER_PATH = Path(__file__).resolve().parents[1] / "tools/train_latency_model.py"
SPEC = importlib.util.spec_from_file_location("baseline_latency_trainer", TRAINER_PATH)
assert SPEC and SPEC.loader
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def _write_case(root: Path, name: str, offset: int = 0) -> Path:
    case_dir = root / name
    case_dir.mkdir()
    schedule = case_dir / "TP0.schedule_batch.jsonl"
    with schedule.open("w", encoding="utf-8") as output:
        for index in range(8):
            extend = index + 1 + offset
            item = {
                "forward_mode": 1,
                "iter_latency": 0.001 * (extend + 1),
                "request_infos": [
                    {
                        "rid": f"{name}-{index}",
                        "extend_input_len": extend,
                        "prefix_indices_len": index,
                        "output_ids_len": 0,
                    }
                ],
            }
            output.write(json.dumps(item) + "\n")
    return schedule


def test_shared_feature_contract_matches_runtime_predictor():
    features = extract_ml_features([1, 3], [2, 4])

    assert len(features) == 18
    assert tuple(MLTimePredictor.FEATURE_NAMES) == ML_FEATURE_NAMES
    assert features[:10] == [2.0, 4.0, 3.0, 1.0, 6.0, 4.0, 2.0, 14.0, 10.0, 20.0]
    assert features[-2:] == [0.0, 1.0]


@pytest.mark.parametrize(
    ("extend", "past", "message"),
    [
        ([], [], "empty batch"),
        ([1], [1, 2], "equal size"),
        ([1], [-1], "non-negative"),
    ],
)
def test_feature_extraction_rejects_invalid_batches(extend, past, message):
    with pytest.raises(ValueError, match=message):
        extract_ml_features(extend, past)


def test_collect_rows_and_split_each_case(tmp_path):
    _write_case(tmp_path, "0-32k_node1_pod-a")
    _write_case(tmp_path, "0-32k_node2_pod-b", offset=2)

    rows, per_case = trainer.collect_rows(
        [str(tmp_path)], "TP0*.schedule_batch.jsonl", {1}, 30.0
    )
    train_rows, test_rows = trainer.build_case_stratified_split(rows, 42, 0.5)

    assert len(rows) == 16
    assert len(per_case) == 2
    assert {row["case_id"] for row in train_rows} == {
        row["case_id"] for row in test_rows
    }
    assert len(train_rows) == len(test_rows) == 8


def test_cli_writes_model_and_evaluation_artifacts(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_case(data_root, "0-32k_node1_pod-a")
    _write_case(data_root, "0-32k_node2_pod-b", offset=2)
    output_dir = tmp_path / "output"

    trainer.main(
        [
            "--data-root",
            str(data_root),
            "--out-dir",
            str(output_dir),
            "--model-prefix",
            "test_model",
            "--loss",
            "l2",
        ]
    )

    bundle = joblib.load(output_dir / "test_model_l2.pkl")
    assert bundle["features"] == list(ML_FEATURE_NAMES)
    assert bundle["format_version"] == 1
    assert bundle["train_rows"] == 16
    assert (output_dir / "eval/dataset_cases.csv").is_file()
    assert (output_dir / "eval/split_summary.csv").is_file()
    assert (output_dir / "eval/split_metrics_by_group.csv").is_file()
    assert (output_dir / "eval/training_log.md").is_file()


def test_trained_bundle_loads_in_ml_time_predictor(tmp_path):
    rows = []
    for index in range(40):
        extend = index % 10 + 1
        past = index * 2
        rows.append(
            {
                "features": extract_ml_features([extend], [past]),
                "latency": 0.001 + extend * 0.0002 + past * 0.00001,
            }
        )

    model, hyperparameters, monotonic = trainer.train_model(rows, "l2", 42)
    model_path = tmp_path / "model.pkl"
    joblib.dump(
        {
            "model": model,
            "features": trainer.FEATURE_NAMES,
            "monotonic": monotonic,
            "best_params": hyperparameters,
        },
        model_path,
    )

    predictor = MLTimePredictor(None, None, None, str(model_path))
    latency = predictor.predict_infer_time(
        ScheduleBatch([ScheduleRequest(extend_length=4, past_kv_length=12)])
    )

    assert latency > 0
