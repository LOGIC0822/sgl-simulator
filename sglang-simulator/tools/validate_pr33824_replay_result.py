#!/usr/bin/env python3
"""Reject incomplete or token-mismatched offline predictor replays."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate(real_case: Path, replay_case: Path) -> None:
    trace = json.loads((real_case / "autobench_trace.manifest.json").read_text())
    predicted = json.loads((replay_case / "simulator-output/metrics.json").read_text())
    benchmark = json.loads((replay_case / "benchmark.jsonl").read_text().splitlines()[-1])
    count = int(trace["request_count"])
    for key in ("num_requests", "completed"):
        if int(predicted[key]) != count:
            raise ValueError(f"replay {key}={predicted[key]}, expected {count}")
    if int(benchmark["completed"]) != count:
        raise ValueError(f"replay benchmark completed={benchmark['completed']}, expected {count}")
    for metric, key in (("total_input", "input_tokens"), ("total_output", "output_tokens")):
        if int(predicted[metric]) != int(trace[key]):
            raise ValueError(f"replay {metric}={predicted[metric]}, expected {trace[key]}")
    for key in ("duration", "mean_e2e_latency_ms", "mean_ttft_ms", "mean_tpot_ms", "total_throughput"):
        value = float(predicted[key])
        if not math.isfinite(value) or value < 0 or (value == 0 and key != "mean_tpot_ms"):
            raise ValueError(f"invalid replay metric {key}={value}")
    if int(predicted["iterations"]) <= 0:
        raise ValueError("replay contains no predictor iterations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-case", type=Path, required=True)
    parser.add_argument("--replay-case", type=Path, required=True)
    args = parser.parse_args()
    validate(args.real_case, args.replay_case)
    print(f"Verified complete predictor replay and matching token totals: {args.replay_case}")


if __name__ == "__main__":
    main()
