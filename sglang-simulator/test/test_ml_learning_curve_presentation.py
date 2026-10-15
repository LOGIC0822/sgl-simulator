"""Check learning-curve labels without collecting data or fitting models."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ml_learning_curve", ROOT / "tools/run_pr33824_ml_learning_curve.py")
assert SPEC and SPEC.loader
curve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curve)


def test_presentation_labels_workloads_and_actual_request_budget(tmp_path):
    rows = [
        {"training_traces": str(count), "phase": phase, "case": case,
         "heldout_batches": "100", "mape_pct": "6.25", "mae_ms": "1.5", "p95_ape_pct": "12"}
        for count in (1, 3, 5)
        for case, phase in (("all", "all"), ("all", "prefill"), ("all", "decode"), ("unseen_case", "all"))
    ]
    original = [dict(row) for row in rows]
    curve.render_presentation("example", rows, tmp_path, requests_per_group=160)
    text = (tmp_path / "README.md").read_text()
    assert "| 800 | 5 | unseen_case | all | 100 | 6.2500% |" in text
    assert "训练样本量与独立测试误差" in text
    assert "不按测试集误差挑选训练量" in text
    assert (tmp_path / "learning_curve_mape.png").stat().st_size > 1000
    assert rows == original


def test_presentation_does_not_invent_missing_request_budget(tmp_path):
    rows = [{"training_traces": "1", "phase": "all", "case": "all",
             "heldout_batches": "100", "mape_pct": "6.25", "mae_ms": "1.5", "p95_ape_pct": "12"}]
    curve.render_presentation("example", rows, tmp_path)
    assert "| 未记录 | 1 | all | all |" in (tmp_path / "README.md").read_text()
    with pytest.raises(ValueError, match="must be positive"):
        curve.render_presentation("example", rows, tmp_path, requests_per_group=0)
