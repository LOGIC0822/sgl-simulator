#!/usr/bin/env python3
"""Attribute GPU events by launch correlation to one native CPU record range.

GPU interval union, duration sum and unknown coverage are distinct. CPU range
duration and iteration wall minus GPU union are not pure CPU execution time.
This tool never tunes a predictor or substitutes instrumented baseline values.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
HOST_CATEGORIES = {"cuda_runtime", "cuda_driver"}


def interval(event: dict) -> tuple[float, float]:
    start, duration = float(event["ts"]), float(event["dur"])
    if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration < 0:
        raise ValueError("invalid profiler time interval")
    return start, start+duration


def union_duration(intervals: list[tuple[float, float]]) -> float:
    end, result = None, 0.0
    for start, stop in sorted(intervals):
        if end is None or start >= end:
            result += stop-start
        elif stop > end:
            result += stop-end
        end = stop if end is None else max(end, stop)
    return result


def analyze(trace: dict, marker: str, iteration_seconds: float) -> dict:
    if not math.isfinite(iteration_seconds) or iteration_seconds <= 0:
        raise ValueError("positive finite measured iteration time required")
    events = trace["traceEvents"]
    markers = [event for event in events if event.get("ph") == "X" and event.get("name") == marker]
    if len(markers) != 1:
        raise ValueError("require exactly one matching native record range")
    root = markers[0]
    root_start, root_end = interval(root)
    host = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in HOST_CATEGORIES:
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            interval(event)
            host.setdefault(correlation, []).append(event)
    assigned, unknown, seen = [], [], set()
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in GPU_CATEGORIES:
            continue
        start, stop = interval(event)
        info = event.get("args", {})
        key = (event["cat"], event["name"], start, stop, info.get("device"), info.get("stream"), info.get("correlation"))
        if key in seen:
            raise ValueError("duplicate profiler GPU event")
        seen.add(key)
        launches = host.get(info.get("correlation"), [])
        within = [entry for entry in launches if entry.get("pid") == root.get("pid")
                  and entry.get("tid") == root.get("tid") and root_start <= interval(entry)[0] <= root_end]
        if within and len(within) != len(launches):
            raise ValueError("ambiguous GPU launch correlation across native boundaries")
        target = assigned if within else unknown
        target.append({"name": event["name"], "category": event["cat"], "start_us": start,
                       "end_us": stop, "duration_us": stop-start, "device": info.get("device"),
                       "stream": info.get("stream"), "correlation": info.get("correlation")})
    if not assigned:
        raise ValueError("no GPU events reliably attributed to the native call")
    # This diagnostic is a single-device TP1 run. Do not union timelines across
    # independent devices and call them one GPU's busy time.
    if len({event["device"] for event in assigned}) != 1 or any(event["device"] is None for event in assigned):
        raise ValueError("native sample spans multiple or inconsistently identified devices")
    union = union_duration([(event["start_us"], event["end_us"]) for event in assigned])
    total = sum(event["duration_us"] for event in assigned)
    unknown_total = sum(event["duration_us"] for event in unknown)
    families = {}
    for event in assigned:
        name = event["name"].lower()
        if event["category"] != "kernel":
            family = event["category"]
        elif "gemm" in name or "gemv" in name:
            family = "gemm_gemv"
        elif name.startswith("_fwd_kernel_stage") or name.startswith("_fwd_grouped_kernel_stage") or name == "_lean_attention_decode_kernel":
            family = "triton_decode_attention"
        elif name in ("_fwd_kernel", "_fwd_kernel_unified"):
            family = "triton_prefill_attention"
        else:
            family = "other_kernel"
        value = families.setdefault(family, {"event_count": 0, "duration_sum_us": 0.0})
        value["event_count"] += 1
        value["duration_sum_us"] += event["duration_us"]
    return {"native_marker": marker, "native_cpu_range_us": root_end-root_start,
            "instrumented_iteration_us": iteration_seconds*1e6,
            "native_gpu_event_count": len(assigned), "native_gpu_duration_sum_us": total,
            "native_gpu_interval_union_us": union, "unattributed_gpu_event_count": len(unknown),
            "unattributed_gpu_duration_sum_us": unknown_total,
            "gpu_attribution_duration_pct": total/(total+unknown_total)*100 if total+unknown_total else 0,
            "iteration_wall_minus_attributed_gpu_union_us": iteration_seconds*1e6-union,
            "families": families, "gpu_events": assigned, "unattributed_gpu_events": unknown,
            "scope": "Instrumented native diagnostic only. GPU families are name-based, not exact AIC-op mappings. Wall-minus-union is not pure CPU time or a correction to formal labels."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    path = Path(metadata["trace_path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["trace_sha256"]:
        raise ValueError("profile trace differs from the recorded sample")
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as source:
        trace = json.load(source)
    result = analyze(trace, metadata["native_marker"], float(metadata["measured_batch"]["iter_latency"]))
    result.update(metadata_path=str(args.metadata.resolve()), trace_path=str(path.resolve()), trace_sha256=digest)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
