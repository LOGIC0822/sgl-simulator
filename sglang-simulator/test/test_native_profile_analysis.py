"""Synthetic correlation/interval tests; no GPU trace accuracy claim."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("native_profile_analysis", Path(__file__).resolve().parents[1] / "tools/analyze_pr33824_native_profile_sample.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def event(name, cat, start, duration, correlation=None, **kwargs):
    value = {"ph": "X", "name": name, "cat": cat, "ts": start, "dur": duration,
             "pid": 10, "tid": 11, "args": {"correlation": correlation, "device": 0, "stream": 7}}
    value.update(kwargs)
    return value


def fixture():
    return {"traceEvents": [
        event("native", "user_annotation", 10, 10),
        event("cuLaunchKernel", "cuda_driver", 12, 1, 101),
        event("cudaLaunchKernel", "cuda_runtime", 15, 1, 102),
        event("cudaMemcpyAsync", "cuda_runtime", 16, 1, 103),
        # These asynchronous GPU events may extend past the CPU record range.
        event("ampere_bf16_gemm", "kernel", 18, 6, 101, pid=0, tid=7),
        event("_fwd_kernel_stage1", "kernel", 20, 6, 102, pid=0, tid=8),
        event("Memcpy DtoH", "gpu_memcpy", 19, 2, 103, pid=0, tid=7),
        event("unknown", "kernel", 30, 2, 999, pid=0, tid=7)]}


def test_gpu_launch_correlation_and_interval_union_do_not_double_count():
    value = analysis.analyze(fixture(), "native", .000030)
    assert value["native_cpu_range_us"] == 10
    assert value["native_gpu_duration_sum_us"] == 14
    assert value["native_gpu_interval_union_us"] == 8
    assert value["iteration_wall_minus_attributed_gpu_union_us"] == pytest.approx(22)
    assert value["unattributed_gpu_event_count"] == 1
    assert value["gpu_attribution_duration_pct"] == pytest.approx(87.5)
    assert value["families"]["gemm_gemv"]["duration_sum_us"] == 6
    assert value["families"]["triton_decode_attention"]["duration_sum_us"] == 6
    assert value["families"]["gpu_memcpy"]["duration_sum_us"] == 2


@pytest.mark.parametrize("kind", ["no_marker", "double_marker", "duplicate_gpu", "bad_time", "ambiguous", "two_devices"])
def test_analyzer_refuses_ambiguous_or_invalid_trace(kind):
    value = fixture()
    events = value["traceEvents"]
    if kind == "no_marker":
        events.pop(0)
    elif kind == "double_marker":
        events.append(events[0])
    elif kind == "duplicate_gpu":
        events.append(events[4])
    elif kind == "bad_time":
        events[4]["dur"] = float("nan")
    elif kind == "ambiguous":
        events.append(event("outside_launch", "cuda_driver", 1, 1, 101))
    else:
        events[5]["args"]["device"] = 1
    with pytest.raises(ValueError):
        analysis.analyze(value, "native", .000030)


def test_timestamp_overlap_without_launch_identity_is_not_attribution():
    value = fixture()
    for candidate in value["traceEvents"]:
        if candidate["cat"] in analysis.HOST_CATEGORIES:
            candidate["tid"] = 99
    with pytest.raises(ValueError, match="no GPU events reliably attributed"):
        analysis.analyze(value, "native", .000030)


def test_current_source_unified_prefill_and_lean_decode_names():
    value = fixture()
    value["traceEvents"][4]["name"] = "_fwd_kernel_unified"
    value["traceEvents"][5]["name"] = "_lean_attention_decode_kernel"
    result = analysis.analyze(value, "native", .000030)
    assert result["families"]["triton_prefill_attention"]["duration_sum_us"] == 6
    assert result["families"]["triton_decode_attention"]["duration_sum_us"] == 6
