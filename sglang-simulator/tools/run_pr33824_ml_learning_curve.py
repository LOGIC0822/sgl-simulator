#!/usr/bin/env python3
"""Measure how nested ML training trace groups affect independent-test MAPE.

The tool retrains the same HGB p50 predictor on nested prefixes of already
collected training traces and scores each bundle against a fixed, disjoint
held-out SchedulerBatch set.  It does not launch a GPU workload.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def required_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row matching {criteria}, found {len(matches)}")
    return matches[0]


def parse_counts(value: str, maximum: int) -> list[int]:
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts or any(count < 1 or count > maximum for count in counts):
        raise argparse.ArgumentTypeError(
            f"--train-counts must be positive values no larger than {maximum}"
        )
    if counts != sorted(set(counts)):
        raise argparse.ArgumentTypeError("--train-counts must be ascending and unique")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--python", dest="python_bin", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-counts", default="1,3,5")
    args = parser.parse_args()
    args.sim_root = args.sim_root.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.python_bin.is_file():
        parser.error(f"missing Python executable: {args.python_bin}")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.out_dir}")
    return args


def run(command: list[str], environment: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(command, check=True, env=environment, stdout=output, stderr=subprocess.STDOUT)


def render_presentation(
    model_name: str,
    rows: list[dict[str, str]],
    out_dir: Path,
    requests_per_group: int | None = None,
) -> None:
    """Render existing scores without fitting or selecting a predictor."""
    if requests_per_group is not None and requests_per_group < 1:
        raise ValueError("requests_per_group must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = requests_per_group or 1
    counts = sorted({int(row["training_traces"]) for row in rows})
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    for phase in ("all", "prefill", "decode"):
        phase_rows = [row for row in rows if row["phase"] == phase and row["case"] == "all"]
        axis.plot(
            [int(row["training_traces"]) * scale for row in phase_rows],
            [float(row["mape_pct"]) for row in phase_rows],
            marker="o",
            label=phase,
        )
    axis.set_xlabel("Training requests" if requests_per_group else "Independent training trace groups")
    axis.set_ylabel("Held-out SchedulerBatch MAPE (%)")
    axis.set_title(f"{model_name}: sample-count learning curve")
    axis.set_xticks([count * scale for count in counts])
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "learning_curve_mape.png", dpi=180)
    plt.close(figure)

    markdown = [
        f"# {model_name}：训练样本量与独立测试误差",
        "",
        "训练集为前 N 组独立采集的真机 trace（每组可含多种负载），测试集固定为未参与训练的五组真实 trace。数值为 teacher-forced SchedulerBatch MAPE，不包含调度反馈。训练和测试的负载设置以采集 metadata 为准，不将所有测试用例统称为同分布。正式模型使用预先确定的全量训练集，不按测试集误差挑选训练量。",
        "",
        "| 训练请求数 | 训练 trace 组数 | case | phase | 测试 batch 数 | MAPE | MAE (ms) | P95 APE |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        request_count = str(int(row["training_traces"]) * requests_per_group) if requests_per_group else "未记录"
        markdown.append(
            f"| {request_count} | {row['training_traces']} | {row['case']} | {row['phase']} | "
            f"{row['heldout_batches']} | {float(row['mape_pct']):.4f}% | "
            f"{float(row['mae_ms']):.4f} | {float(row['p95_ape_pct']):.4f}% |"
        )
    markdown.extend(["", "![训练样本量与测试误差](learning_curve_mape.png)", ""])
    (out_dir / "README.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_root = args.model_root / "calibration" / "real"
    heldout_root = args.model_root / "holdout" / "real"
    train_repeats = sorted(path for path in train_root.glob("repeat-*") if path.is_dir())
    if not train_repeats:
        raise RuntimeError(f"no training repeats found: {train_root}")
    for repeat in train_repeats:
        marker = repeat / "BASELINE_COLLECTION_COMPLETE"
        if marker.read_text(encoding="utf-8").strip() != "complete":
            raise RuntimeError(f"incomplete training repeat: {repeat}")
    for repeat in range(1, 6):
        marker = heldout_root / f"repeat-{repeat}" / "BASELINE_COLLECTION_COMPLETE"
        if marker.read_text(encoding="utf-8").strip() != "complete":
            raise RuntimeError(f"incomplete held-out repeat: {marker.parent}")
    counts = parse_counts(args.train_counts, len(train_repeats))

    model_name = args.model_root.name
    metadata_path = args.model_root.parent.parent / "run_metadata.txt"
    metadata = dict(line.split("=", 1) for line in metadata_path.read_text().splitlines() if "=" in line)
    fit_seed = metadata.get("training_fit_seed", metadata.get("training_seeds", "20260918").split(",")[0])
    safe_model = "".join(char if char.isalnum() or char in ".-_" else "_" for char in model_name)
    sglang_root = args.sim_root.parent.parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{args.sim_root / 'src'}:{sglang_root / 'python'}"
    if environment.get("SGLANG_EXPERIMENT_EXTRA_PYTHONPATH"):
        environment["PYTHONPATH"] += ":" + environment["SGLANG_EXPERIMENT_EXTRA_PYTHONPATH"]

    args.out_dir.mkdir(parents=True, exist_ok=False)
    matplotlib_cache = args.out_dir / "matplotlib-cache"
    matplotlib_cache.mkdir()
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    rows: list[dict[str, str]] = []
    for count in counts:
        stage = args.out_dir / f"train_{count}_traces"
        bundle_dir = stage / "bundle"
        score_dir = stage / "teacher_forced"
        bundle = bundle_dir / f"{safe_model}_hgbmono_p50.pkl"
        train_command = [
            str(args.python_bin),
            str(args.sim_root / "tools" / "train_latency_model.py"),
        ]
        for repeat in train_repeats[:count]:
            train_command.extend(["--data-root", str(repeat / "cases")])
        train_command.extend(
            [
                "--out-dir",
                str(bundle_dir),
                "--model-prefix",
                f"{safe_model}_hgbmono",
                "--loss",
                "p50",
                "--forward-modes",
                "1,2",
                "--seed",
                fit_seed,
                "--test-fraction",
                "0.2",
            ]
        )
        run(train_command, environment, stage / "train.log")
        if not bundle.is_file():
            raise RuntimeError(f"training did not produce bundle: {bundle}")
        score_command = [
            str(args.python_bin),
            str(args.sim_root / "tools" / "score_pr33824_latency_model.py"),
            "--data-root",
            str(heldout_root),
            "--model-bundle",
            str(bundle),
            "--out-dir",
            str(score_dir),
            "--forward-modes",
            "1,2",
        ]
        run(score_command, environment, stage / "score.log")
        summary = read_csv(score_dir / "summary.csv")
        for phase, group, value in (
            ("all", "all", "all"),
            ("prefill", "forward_mode", "1"),
            ("decode", "forward_mode", "2"),
        ):
            row = required_row(summary, group=group, value=value)
            rows.append(
                {
                    "model": model_name,
                    "case": "all",
                    "training_traces": str(count),
                    "phase": phase,
                    "heldout_batches": row["n"],
                    "mape_pct": row["mape_pct"],
                    "mae_ms": row["mae_ms"],
                    "p95_ape_pct": row["p95_ape_pct"],
                    "bundle": str(bundle),
                }
            )
        for row in summary:
            if row["group"] != "case":
                continue
            rows.append({"model": model_name, "case": row["value"],
                         "training_traces": str(count), "phase": "all",
                         "heldout_batches": row["n"], "mape_pct": row["mape_pct"],
                         "mae_ms": row["mae_ms"], "p95_ape_pct": row["p95_ape_pct"],
                         "bundle": str(bundle)})

    csv_path = args.out_dir / "learning_curve.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    requests_per_group = metadata.get("requests_per_training_repeat")
    render_presentation(model_name, rows, args.out_dir, int(requests_per_group) if requests_per_group else None)
    print(f"out={args.out_dir} rows={len(rows)}")


if __name__ == "__main__":
    main()
