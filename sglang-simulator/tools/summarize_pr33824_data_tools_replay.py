#!/usr/bin/env python3
"""Pair collected real-server metrics with ML simulator replay metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


METRICS = (
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
    "total_throughput",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def final_jsonl_object(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError(f"empty benchmark JSONL: {path}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def pair_roots(real_root: Path, replay_root: Path, pair_name: str) -> list[dict[str, Any]]:
    real_cases = real_root / "cases"
    replay_cases = replay_root / "cases"
    if not real_cases.is_dir() or not replay_cases.is_dir():
        raise FileNotFoundError(
            f"both roots must contain cases/: real={real_root}, replay={replay_root}"
        )
    rows: list[dict[str, Any]] = []
    names = sorted(
        path.name
        for path in real_cases.iterdir()
        if path.is_dir() and (replay_cases / path.name).is_dir()
    )
    if not names:
        raise RuntimeError(f"no common cases for pair {pair_name}")
    for name in names:
        real_case = real_cases / name
        replay_case = replay_cases / name
        real = final_jsonl_object(real_case / "benchmark.jsonl")
        predicted = read_json(replay_case / "simulator-output" / "metrics.json")
        trace_manifest_path = real_case / "autobench_trace.manifest.json"
        trace_manifest = read_json(trace_manifest_path) if trace_manifest_path.is_file() else {}
        # Random token IDs may be decoded to text before serving. Dataset
        # prompt_len then counts the pre-decode IDs rather than actual server
        # input IDs. Keep the raw counter, but use the replay trace's actual
        # token numerator for both throughputs; do not alter measured duration.
        trace_throughput = None
        if trace_manifest and "duration" in real:
            duration = float(real["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"invalid real duration for {pair_name}/{name}")
            trace_throughput = (
                int(trace_manifest["input_tokens"]) + int(trace_manifest["output_tokens"])
            ) / duration
        for metric in METRICS:
            reported = float(real[metric])
            actual = trace_throughput if metric == "total_throughput" and trace_throughput is not None else reported
            estimate = float(predicted[metric])
            if not (math.isfinite(actual) and math.isfinite(estimate)):
                raise ValueError(f"non-finite {metric} for {pair_name}/{name}")
            signed = (estimate - actual) / actual * 100.0
            rows.append(
                {
                    "pair": pair_name,
                    "case": name,
                    "metric": metric,
                    "real_value": actual,
                    "real_reported_value": reported,
                    "predicted_value": estimate,
                    "signed_error_pct": signed,
                    "absolute_percentage_error_pct": abs(signed),
                    "trace_request_count": trace_manifest.get("request_count", ""),
                    "trace_input_tokens": trace_manifest.get("input_tokens", ""),
                    "trace_output_tokens": trace_manifest.get("output_tokens", ""),
                    "real_benchmark_input_tokens": real.get("total_input_tokens", ""),
                    "real_benchmark_output_tokens": real.get("total_output_tokens", ""),
                    "throughput_token_source": "actual_trace" if trace_throughput is not None else "benchmark_legacy",
                    "real_root": str(real_root),
                    "replay_root": str(replay_root),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[("metric", row["metric"])].append(row)
        groups[("case_metric", f"{row['case']}:{row['metric']}")].append(row)
    output = []
    for (group, value), selected in sorted(groups.items()):
        real = [float(row["real_value"]) for row in selected]
        predicted = [float(row["predicted_value"]) for row in selected]
        signed = [float(row["signed_error_pct"]) for row in selected]
        absolute = [float(row["absolute_percentage_error_pct"]) for row in selected]
        output.append(
            {
                "group": group,
                "value": value,
                "n": len(selected),
                "real_mean": fmean(real),
                "real_std": pstdev(real) if len(real) > 1 else 0.0,
                "predicted_mean": fmean(predicted),
                "predicted_std": pstdev(predicted) if len(predicted) > 1 else 0.0,
                "mean_signed_error_pct": fmean(signed),
                "mape_pct": fmean(absolute),
                "median_ape_pct": sorted(absolute)[len(absolute) // 2],
                "max_ape_pct": max(absolute),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def optional_teacher_forced_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    overall = next(
        (row for row in rows if row.get("group") == "all" and row.get("value") == "all"),
        None,
    )
    if overall is None:
        return []
    return [
        "## Teacher-forced scheduler-batch score",
        "",
        "This score uses the real observed batch composition and therefore does not include scheduler feedback.",
        "",
        f"- batches: {overall['n']}",
        f"- MAPE: {float(overall['mape_pct']):.3f}%",
        f"- MAE: {float(overall['mae_ms']):.3f} ms",
        "",
    ]


def render_markdown(
    pairs: list[tuple[Path, Path]], summary_rows: list[dict[str, Any]], teacher_forced: Path | None
) -> str:
    lines = [
        "# Collected SGLang ML replay comparison",
        "",
        "## Scope",
        "",
        "Each row pairs a real SGLang serving result with a CPU simulator replay of that run's converted raw request trace. The replay uses the same token IDs, completed output lengths, relative arrival times, context length, request limit, chunked-prefill size, and real service token-pool limit. The real server uses device-synchronized collection with overlap scheduling and CUDA graph disabled.",
        "",
        "Total token throughput uses actual trace input+output counts divided by the original measured benchmark duration; the benchmark's nominal pre-decode input count can differ after text re-tokenization. Original reported throughput and token counts are retained in comparison.csv. Client and simulator timing boundaries remain different; this correction aligns only token counting.",
        "",
        "## End-to-end paired metrics",
        "",
        "| Metric | Pairs | Real mean | Real std | Predicted mean | MAPE | Mean signed error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if row["group"] != "metric":
            continue
        lines.append(
            "| {value} | {n} | {real_mean:.3f} | {real_std:.3f} | "
            "{predicted_mean:.3f} | {mape_pct:.3f}% | {mean_signed_error_pct:.3f}% |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
        ]
    )
    for index, (real_root, replay_root) in enumerate(pairs, start=1):
        lines.append(f"{index}. real: `{real_root}`")
        lines.append(f"   replay: `{replay_root}`")
    lines.append("")
    lines.extend(optional_teacher_forced_lines(teacher_forced))
    lines.extend(
        [
            "## Reading the numbers",
            "",
            "The table is descriptive. Teacher-forced batch error and end-to-end replay error answer different questions: the latter also changes future batch formation when a predicted latency differs from the real one. These rows do not isolate every runtime component, such as request handling and host scheduling, from GPU forward time.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", action="append", required=True)
    parser.add_argument("--replay-root", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--teacher-forced-summary", default=None)
    args = parser.parse_args(argv)
    if len(args.real_root) != len(args.replay_root):
        parser.error("--real-root and --replay-root must be supplied equally")
    args.real_root = [Path(value).expanduser().resolve() for value in args.real_root]
    args.replay_root = [Path(value).expanduser().resolve() for value in args.replay_root]
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    args.teacher_forced_summary = (
        Path(args.teacher_forced_summary).expanduser().resolve()
        if args.teacher_forced_summary
        else None
    )
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.out_dir}")
    if args.teacher_forced_summary and not args.teacher_forced_summary.is_file():
        parser.error(f"teacher-forced summary does not exist: {args.teacher_forced_summary}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pairs = list(zip(args.real_root, args.replay_root))
    rows: list[dict[str, Any]] = []
    for index, (real_root, replay_root) in enumerate(pairs, start=1):
        rows.extend(pair_roots(real_root, replay_root, f"repeat-{index}"))
    summary_rows = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(
        args.out_dir / "comparison.csv",
        rows,
        [
            "pair",
            "case",
            "metric",
            "real_value",
            "real_reported_value",
            "predicted_value",
            "signed_error_pct",
            "absolute_percentage_error_pct",
            "trace_request_count",
            "trace_input_tokens",
            "trace_output_tokens",
            "real_benchmark_input_tokens",
            "real_benchmark_output_tokens",
            "throughput_token_source",
            "real_root",
            "replay_root",
        ],
    )
    write_csv(
        args.out_dir / "summary.csv",
        summary_rows,
        [
            "group",
            "value",
            "n",
            "real_mean",
            "real_std",
            "predicted_mean",
            "predicted_std",
            "mean_signed_error_pct",
            "mape_pct",
            "median_ape_pct",
            "max_ape_pct",
        ],
    )
    (args.out_dir / "comparison.md").write_text(
        render_markdown(pairs, summary_rows, args.teacher_forced_summary),
        encoding="utf-8",
    )
    print(f"paired_rows={len(rows)} out={args.out_dir}")


if __name__ == "__main__":
    main()
