"""CPU-only contract test for the final ML experiment summarizer."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_pr33824_ml_experiment.py"
SPEC = importlib.util.spec_from_file_location("ml_experiment_summary", SCRIPT)
assert SPEC and SPEC.loader
summary_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_tool
SPEC.loader.exec_module(summary_tool)


MODELS = ["model-3b", "model-7b", "model-32b"]
CASES = ["latency_256x128_r1", "service_1024x256_r4", "saturation_1024x256_inf"]


def holdout_metrics(seed: float) -> dict:
    return {
        stage: {
            "n": 10,
            "mape_pct": seed + offset,
            "median_ape_pct": seed + offset / 2,
            "p90_ape_pct": seed + offset * 2,
            "mae_ms": 0.1 + offset,
            "rmse_ms": 0.2 + offset,
            "mean_target_ms": 1.0,
            "mean_prediction_ms": 1.1,
        }
        for stage, offset in (("all", 0.0), ("prefill", 1.0), ("decode", 2.0))
    }


def paired_rows(model: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_index, case in enumerate(CASES):
        for repeat in range(1, 6):
            row = {
                "model": model,
                "case": case,
                "repeat": str(repeat),
                "status": "ok",
                "prediction_repeat_policy": "mean-per-case",
                "prediction_samples": "2",
                "num_prompts": "8",
                "input_tokens_target": "256",
                "output_tokens_target": "128",
                "request_rate": "1",
            }
            for metric_index, (metric, _) in enumerate(summary_tool.CORE_METRICS):
                real = 10.0 + case_index + repeat + metric_index
                predicted = real * 1.1
                row[f"real_{metric}"] = str(real)
                row[f"simulated_{metric}"] = str(predicted)
                row[f"signed_error_pct_{metric}"] = "10"
                row[f"absolute_error_pct_{metric}"] = "10"
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summary_rejects_aic_and_writes_simsun_docx(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "CALIBRATION_COMPLETE").touch()
    experiment = tmp_path / "experiment"
    real_reference = tmp_path / "real-reference"
    real_reference.mkdir()
    for index, model in enumerate(MODELS):
        calibration_model = calibration / "models" / model
        analysis = calibration_model / "analysis"
        analysis.mkdir(parents=True)
        (analysis / "training_summary.json").write_text(
            json.dumps(
                {
                    "aiconfigurator_used": False,
                    "holdout_metrics": holdout_metrics(float(index + 1)),
                    "model_file": str(calibration_model / "latency.pkl"),
                    "simulator_config": str(calibration_model / "simulator-config.json"),
                }
            ),
            encoding="utf-8",
        )
        root = experiment / "per-model" / model
        root.mkdir(parents=True)
        (root / "MATRIX_COMPLETE").touch()
        (root / "run_metadata.txt").write_text("predictor_name=ml\n", encoding="utf-8")
        (root / "simulator-config.json").write_text(
            json.dumps({"predictor": {"name": "ml", "database_path": str(calibration_model / "latency.pkl")}}),
            encoding="utf-8",
        )
        write_csv(root / "analysis" / "paired_repeat_metrics.csv", paired_rows(model))

    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--experiment-root",
            str(experiment),
            "--calibration-root",
            str(calibration),
            "--real-reference-root",
            str(real_reference),
            "--models",
            ",".join(MODELS),
        ],
    )
    assert summary_tool.main() == 0
    final_md = experiment / "analysis" / "final_report.md"
    final_docx = experiment / "analysis" / "final_report.docx"
    assert "未调用 AIConfigurator" in final_md.read_text(encoding="utf-8")
    assert final_docx.is_file()
    with zipfile.ZipFile(final_docx) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "SimSun" in document_xml
    report_summary = json.loads((experiment / "analysis" / "summary.json").read_text(encoding="utf-8"))
    assert report_summary["aiconfigurator_used"] is False
    assert report_summary["completed_paired_runs"] == 45
