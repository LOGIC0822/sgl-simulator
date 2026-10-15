"""Tests for pairing real collection and simulator replay metrics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "summarize_pr33824_data_tools_replay.py"
)
SPEC = importlib.util.spec_from_file_location("data_tools_replay_summary", SCRIPT)
assert SPEC and SPEC.loader
summary_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_tool
SPEC.loader.exec_module(summary_tool)


def _write_pair(root: Path, *, real: float, predicted: float) -> tuple[Path, Path]:
    real_root = root / "real"
    replay_root = root / "replay"
    case = "saturation"
    real_case = real_root / "cases" / case
    replay_case = replay_root / "cases" / case / "simulator-output"
    real_case.mkdir(parents=True)
    replay_case.mkdir(parents=True)
    real_metrics = {
        metric: real
        for metric in summary_tool.METRICS
    }
    predicted_metrics = {
        metric: predicted
        for metric in summary_tool.METRICS
    }
    (real_case / "benchmark.jsonl").write_text(json.dumps(real_metrics) + "\n")
    (replay_case / "metrics.json").write_text(json.dumps(predicted_metrics))
    (real_case / "autobench_trace.manifest.json").write_text(
        json.dumps({"request_count": 2, "input_tokens": 4, "output_tokens": 3})
    )
    return real_root, replay_root


def test_pair_and_summarize_metrics(tmp_path):
    real_root, replay_root = _write_pair(tmp_path, real=100.0, predicted=110.0)

    rows = summary_tool.pair_roots(real_root, replay_root, "repeat-1")
    summary = summary_tool.summarize(rows)

    assert len(rows) == len(summary_tool.METRICS)
    assert rows[0]["absolute_percentage_error_pct"] == pytest.approx(10.0)
    metric_row = next(
        row
        for row in summary
        if row["group"] == "metric" and row["value"] == "mean_e2e_latency_ms"
    )
    assert metric_row["mape_pct"] == pytest.approx(10.0)


def test_throughput_aligns_actual_tokens_without_overwriting_benchmark(tmp_path):
    real_root, replay_root = _write_pair(tmp_path, real=100.0, predicted=110.0)
    path = real_root / "cases/saturation/benchmark.jsonl"
    original = json.loads(path.read_text())
    original.update(duration=2, total_input_tokens=3, total_output_tokens=3, total_throughput=3)
    path.write_text(json.dumps(original) + "\n")
    before = path.read_bytes()
    row = next(row for row in summary_tool.pair_roots(real_root, replay_root, "repeat-1") if row["metric"] == "total_throughput")
    assert row["real_value"] == 3.5
    assert row["real_reported_value"] == 3
    assert row["real_benchmark_input_tokens"] == 3
    assert row["trace_input_tokens"] == 4
    assert row["throughput_token_source"] == "actual_trace"
    assert path.read_bytes() == before


def test_invalid_real_duration_is_rejected(tmp_path):
    real_root, replay_root = _write_pair(tmp_path, real=100.0, predicted=110.0)
    path = real_root / "cases/saturation/benchmark.jsonl"
    value = json.loads(path.read_text())
    value["duration"] = 0
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="invalid real duration"):
        summary_tool.pair_roots(real_root, replay_root, "repeat-1")
