#!/usr/bin/env python3
"""Verify that a PR #33824 A100 accuracy matrix is complete and readable.

The matrix launcher creates the primary output.  This independent verifier is
intended for the final handoff: it does not alter raw runs, but records exactly
which required experiment or summary artifact is missing if a run ends early.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


CASES = (
    "latency_256x128_r1",
    "service_1024x256_r4",
    "saturation_1024x256_inf",
)
REPEATS = range(1, 6)
METRIC_COUNT = 11
FIGURES = (
    "mape_by_model_and_metric.png",
    "e2e_mape_vs_model_parameters.png",
    "mape_distribution_by_workload.png",
)


def models_from_metadata(path: Path) -> list[str]:
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("models="):
            return [name for name in line.removeprefix("models=").split(",") if name]
    return []


def successful_run(path: Path, phase: str) -> bool:
    try:
        if (path / "status.txt").read_text(encoding="utf-8").strip() != "ok":
            return False
    except OSError:
        return False
    if not (path / "benchmark.jsonl").is_file():
        return False
    return phase != "simulator" or (path / "simulator-output" / "metrics.json").is_file()


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    models = models_from_metadata(root / "run_metadata.txt")
    failures: list[str] = []
    expected_pairs = len(models) * len(CASES) * len(REPEATS)
    complete_pairs = 0

    if not models:
        failures.append("run_metadata.txt does not declare the model list")
    if not (root / "MATRIX_COMPLETE").is_file():
        failures.append("MATRIX_COMPLETE marker is absent (a failed/partial run is not a final matrix)")
    if (root / "MATRIX_COMPLETED_WITH_FAILURES").is_file():
        failures.append("MATRIX_COMPLETED_WITH_FAILURES marker is present")

    for model in models:
        for case in CASES:
            for repeat in REPEATS:
                real = root / "results" / model / "real" / case / f"repeat-{repeat}"
                simulator = root / "results" / model / "simulator" / case / f"repeat-{repeat}"
                real_ok = successful_run(real, "real")
                simulator_ok = successful_run(simulator, "simulator")
                if real_ok and simulator_ok:
                    complete_pairs += 1
                else:
                    failures.append(
                        f"missing valid pair: {model}/{case}/repeat-{repeat} "
                        f"(real={real_ok}, simulator={simulator_ok})"
                    )

    analysis = root / "analysis"
    paired = csv_rows(analysis / "paired_repeat_metrics.csv")
    valid_paired = sum(row.get("status") == "ok" for row in paired)
    if valid_paired != expected_pairs:
        failures.append(f"paired_repeat_metrics.csv has {valid_paired}/{expected_pairs} successful rows")

    case_stats = csv_rows(analysis / "case_metric_statistics.csv")
    expected_case_rows = len(models) * len(CASES)
    complete_case_rows = sum(row.get("complete_pairs") == "5" for row in case_stats)
    if len(case_stats) != expected_case_rows or complete_case_rows != expected_case_rows:
        failures.append(
            "case_metric_statistics.csv lacks complete five-repeat rows "
            f"({complete_case_rows}/{expected_case_rows}; total={len(case_stats)})"
        )

    model_stats = csv_rows(analysis / "model_metric_statistics.csv")
    expected_model_rows = len(models) * METRIC_COUNT
    complete_model_rows = sum(row.get("paired_runs") == "15" for row in model_stats)
    if len(model_stats) != expected_model_rows or complete_model_rows != expected_model_rows:
        failures.append(
            "model_metric_statistics.csv lacks complete fifteen-pair metrics "
            f"({complete_model_rows}/{expected_model_rows}; total={len(model_stats)})"
        )

    for filename in FIGURES:
        if not (analysis / "figures" / filename).is_file() or (analysis / "figures" / filename).stat().st_size == 0:
            failures.append(f"missing figure: figures/{filename}")
    for filename in ("final_report.md", "final_report.docx", "accuracy_report.md"):
        if not (analysis / filename).is_file() or (analysis / filename).stat().st_size == 0:
            failures.append(f"missing report artifact: analysis/{filename}")

    audit = [
        "# PR #33824 A100 矩阵完成审计",
        "",
        f"审计时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"计划配对：{expected_pairs}；有效配对：{complete_pairs}。",
        "",
    ]
    if failures:
        audit += ["## 未通过", ""]
        audit.extend(f"- {failure}" for failure in failures)
    else:
        audit += [
            "## 通过",
            "",
            "13 个模型、3 个工作负载、每工作负载 5 个实机/模拟器配对均完整；"
            "统计 CSV、三张参数相关图和 Markdown/DOCX 报告均已存在。",
        ]
    (analysis / "completion_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(f"Wrote {analysis / 'completion_audit.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
