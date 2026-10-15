#!/usr/bin/env python3
"""Write a small, current-state status page for an in-progress PR #33824 run."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


CASES = (
    "latency_256x128_r1",
    "service_1024x256_r4",
    "saturation_1024x256_inf",
)
PHASES = ("real", "simulator")


def models_from_metadata(path: Path) -> list[str]:
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("models="):
            return [item for item in line.removeprefix("models=").split(",") if item]
    return []


def complete(phase: str, run_dir: Path) -> bool:
    status = run_dir / "status.txt"
    if not status.is_file() or status.read_text(encoding="utf-8").strip() != "ok":
        return False
    if not (run_dir / "benchmark.jsonl").is_file():
        return False
    return phase != "simulator" or (run_dir / "simulator-output" / "metrics.json").is_file()


def matrix_runner_is_live(root: Path) -> bool:
    """Look for the detached matrix process instead of inferring from files."""
    expected_root = str(root)
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if "run_pr33824_a100_matrix.sh" in command and expected_root in command:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    models = models_from_metadata(root / "run_metadata.txt")
    repeats = 5
    expected = len(models) * len(PHASES) * len(CASES) * repeats
    completed = 0
    incomplete = 0
    rows: list[tuple[str, int, int]] = []
    for model in models:
        model_dir = root / "results" / model
        model_completed = 0
        model_incomplete = 0
        for phase in PHASES:
            for case in CASES:
                for repeat in range(1, repeats + 1):
                    run_dir = model_dir / phase / case / f"repeat-{repeat}"
                    if complete(phase, run_dir):
                        completed += 1
                        model_completed += 1
                    elif run_dir.exists():
                        incomplete += 1
                        model_incomplete += 1
        rows.append((model, model_completed, model_incomplete))

    markers = [name for name in ("MATRIX_COMPLETE", "MATRIX_COMPLETED_WITH_FAILURES") if (root / name).is_file()]
    state = markers[0] if markers else ("RUNNING" if matrix_runner_is_live(root) else "INTERRUPTED_OR_NOT_STARTED")
    report = [
        "# PR #33824 实验状态",
        "",
        f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"状态：`{state}`",
        f"当前有效原始运行完成：**{completed}/{expected}**；存在目录但尚未完成：**{incomplete}**。",
        "`superseded_*` 目录是已保留的配置不匹配尝试，不计入这里的完成数和最终统计。",
        "",
        "| 模型 | 当前有效完成/30 | 已创建但未完成 |",
        "| --- | ---: | ---: |",
    ]
    report.extend(f"| {model} | {done} | {partial} |" for model, done, partial in rows)
    report += [
        "",
        "每个模型计划为 3 个工作负载 × 5 次 ×（实机、模拟器）= 30 个原始运行。",
        "最终统计只读取 `results/<model>/real` 和 `results/<model>/simulator` 下带 `status.txt=ok` 的完整配对。",
    ]
    (root / "status.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {root / 'status.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
