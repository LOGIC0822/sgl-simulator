"""Unit tests for the native-one-batch ML predictor trainer.

The test fixtures mimic the stable text emitted by ``sglang.benchmark.one_batch``
and do not require a CUDA device or an installed AIC database.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_pr33824_ml_latency_model.py"
SPEC = importlib.util.spec_from_file_location("ml_latency_trainer", SCRIPT)
assert SPEC and SPEC.loader
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def write_measurement(path: Path, *, batch_size: int, input_len: int, offset: float) -> tuple[Path, Path]:
    results = path / "result.jsonl"
    log = path / "benchmark.log"
    path.mkdir(parents=True)
    results.write_text(
        json.dumps(
            {
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": 3,
                "prefill_latency": 0.010 + offset,
                "median_decode_latency": 0.002 + offset,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log.write_text(
        "\n".join(
            [
                "Warmup ...",
                "Decode 0. Batch size: 99, latency: 9.00000 s, throughput: 1.00 token/s",
                "Total. latency: 9.000 s, throughput: 1.00 token/s",
                "Benchmark ...",
                "Prefill. latency: 0.01000 s, throughput: 100.00 token/s",
                f"Decode 0. Batch size: {batch_size}, latency: {0.002 + offset:.5f} s, throughput: 1.00 token/s",
                f"Decode 1. Batch size: {batch_size}, latency: {0.003 + offset:.5f} s, throughput: 1.00 token/s",
                "Decode. median latency: 0.00250 s, median throughput: 1.00 token/s",
                "Total. latency: 0.015 s, throughput: 1.00 token/s",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return results, log


def test_extract_samples_uses_exact_ml_feature_contract(tmp_path):
    results, log = write_measurement(tmp_path / "measurement", batch_size=2, input_len=8, offset=0.0)
    samples = trainer.extract_samples(results, log, "calibration")

    assert [sample.stage for sample in samples] == ["prefill", "decode", "decode"]
    assert samples[0].features == trainer.make_features([8, 8], [0, 0])
    assert samples[1].features == trainer.make_features([1, 1], [9, 9])
    assert samples[2].features == trainer.make_features([1, 1], [10, 10])
    assert samples[1].target_seconds == 0.002
    assert samples[2].target_seconds == 0.003
    assert trainer.FEATURE_NAMES == [
        "batch_size",
        "sum_extend",
        "max_extend",
        "min_extend",
        "sum_past",
        "max_past",
        "min_past",
        "sum_extend_x_past",
        "sum_extend_squared",
        "sum_past_squared",
        "sum_attn_flops",
        "sum_extend_x_max_past",
        "log1p_sum_past",
        "log1p_sum_attn_flops",
        "batch_size_x_sum_extend",
        "max_past_minus_min_past",
        "is_decode",
        "is_prefill",
    ]


def test_grouped_measurement_log_resets_each_process_warmup(tmp_path):
    first_results, first_log = write_measurement(
        tmp_path / "first", batch_size=2, input_len=8, offset=0.0
    )
    second_results, second_log = write_measurement(
        tmp_path / "second", batch_size=4, input_len=12, offset=0.001
    )
    results = tmp_path / "result.jsonl"
    log = tmp_path / "benchmark.log"
    results.write_text(
        first_results.read_text(encoding="utf-8")
        + second_results.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    log.write_text(
        "# grouped measurement: first\n"
        + first_log.read_text(encoding="utf-8")
        + "# grouped measurement: second\n"
        + second_log.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    samples = trainer.extract_samples(results, log, "calibration")

    assert len(samples) == 6
    assert [sample.batch_size for sample in samples] == [2, 2, 2, 4, 4, 4]


def test_train_writes_loadable_ml_bundle_and_config(tmp_path, monkeypatch):
    train_results, train_log = write_measurement(
        tmp_path / "train", batch_size=2, input_len=8, offset=0.0
    )
    validation_results, validation_log = write_measurement(
        tmp_path / "validation", batch_size=3, input_len=12, offset=0.001
    )
    model_path = tmp_path / "model.pkl"
    analysis = tmp_path / "analysis"
    config = tmp_path / "simulator-config.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--model-name",
            "fixture-model",
            "--train-results",
            str(train_results),
            "--train-log",
            str(train_log),
            "--validation-results",
            str(validation_results),
            "--validation-log",
            str(validation_log),
            "--output-model",
            str(model_path),
            "--output-dir",
            str(analysis),
            "--simulator-config",
            str(config),
            "--trees",
            "5",
        ],
    )

    assert trainer.main() == 0
    bundle = joblib.load(model_path)
    assert bundle["features"] == trainer.FEATURE_NAMES
    assert callable(bundle["model"].predict)
    assert bundle["metadata"]["aiconfigurator_used"] is False
    assert json.loads(config.read_text(encoding="utf-8"))["predictor"]["name"] == "ml"
    assert (analysis / "holdout_samples.csv").is_file()
    summary = json.loads((analysis / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["holdout_metrics"]["all"]["n"] == 3
