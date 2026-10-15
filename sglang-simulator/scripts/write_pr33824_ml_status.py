#!/usr/bin/env python3
"""Write a concise progress snapshot for a native ML calibration run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


TRAIN_ROWS = 6 * 5 * 3
HOLDOUT_ROWS = 4 * 4


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def read_status(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "not-started"


def expected_rows(stage_root: Path, default: int) -> int:
    """Use a per-stage manifest when a model has a trace-feasible grid."""
    try:
        value = int((stage_root / "expected_result_rows.txt").read_text().strip())
    except (OSError, ValueError):
        return default
    return value if value > 0 else default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    metadata: dict[str, str] = {}
    metadata_path = root / "run_metadata.txt"
    if metadata_path.is_file():
        for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                metadata[key] = value
    models = [item.strip() for item in metadata.get("models", "").split(",") if item.strip()]
    lines = [
        "# Native SGLang ML calibration status",
        "",
        f"更新时间：{datetime.now(timezone.utc).astimezone().isoformat()}",
        f"根目录：`{root}`",
        f"总状态：`{'complete' if (root / 'CALIBRATION_COMPLETE').is_file() else 'running-or-incomplete'}`",
        "",
        "| 模型 | calibration JSON 行 | holdout JSON 行 | 训练状态 | bundle |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for model in models:
        model_root = root / "models" / model
        calibration_root = model_root / "calibration"
        holdout_root = model_root / "holdout"
        calibration = line_count(calibration_root / "result.jsonl")
        holdout = line_count(holdout_root / "result.jsonl")
        calibration_expected = expected_rows(calibration_root, TRAIN_ROWS)
        holdout_expected = expected_rows(holdout_root, HOLDOUT_ROWS)
        status = read_status(model_root / "status.txt")
        bundle = "yes" if (model_root / "model" / "latency_model.pkl").is_file() else "no"
        lines.append(
            f"| {model} | {calibration} / {calibration_expected} | "
            f"{holdout} / {holdout_expected} | {status} | {bundle} |"
        )
    summary_path = root / "models"
    if not models and summary_path.is_dir():
        lines.append("| 未读取到 metadata 中的模型列表 | — | — | — | — |")
    (root / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results_root": str(root), "models": models, "complete": (root / "CALIBRATION_COMPLETE").is_file()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
