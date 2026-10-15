"""Test replay completion and exact token preservation without a server."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_replay", ROOT / "tools/validate_pr33824_replay_result.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_valid_replay_and_reject_token_mismatch(tmp_path):
    real = tmp_path / "real"
    replay = tmp_path / "replay"
    real.mkdir()
    (replay / "simulator-output").mkdir(parents=True)
    (real / "autobench_trace.manifest.json").write_text(json.dumps({"request_count": 2, "input_tokens": 8, "output_tokens": 3}))
    metrics = {"num_requests": 2, "completed": 2, "total_input": 8, "total_output": 3,
               "duration": 1, "mean_e2e_latency_ms": 2, "mean_ttft_ms": 1,
               "mean_tpot_ms": 1, "total_throughput": 11, "iterations": 2}
    path = replay / "simulator-output/metrics.json"
    path.write_text(json.dumps(metrics))
    (replay / "benchmark.jsonl").write_text(json.dumps({"completed": 2}))
    MODULE.validate(real, replay)
    metrics["total_output"] = 2
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="total_output"):
        MODULE.validate(real, replay)
