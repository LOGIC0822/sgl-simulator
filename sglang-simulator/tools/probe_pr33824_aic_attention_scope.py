#!/usr/bin/env python3
"""Check AIC attention query scope on CPU; synthetic calls, NOT latency data."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import yaml
from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.attention import ContextAttention, GenerationAttention
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.perf_database import PerfDatabase


class ContractDatabase:
    """Synthetic attention results with the unchanged SDK analytic mem formula."""

    def __init__(self, system_spec: dict):
        self.system_spec = system_spec
        self._default_database_mode = common.DatabaseMode.SILICON
        self.mem_calls: list[int] = []

    def query_context_attention(self, *args, **kwargs) -> PerformanceResult:
        return PerformanceResult(0.01, source="synthetic_contract_only")

    def query_generation_attention(self, *args, **kwargs) -> PerformanceResult:
        return PerformanceResult(0.01, source="synthetic_contract_only")

    def query_mem_op(self, num_bytes: int) -> PerformanceResult:
        self.mem_calls.append(num_bytes)
        return PerfDatabase.query_mem_op(self, num_bytes)


def probe(system_spec: dict) -> dict:
    rows = []
    for batch_size, sequence_length in ((1, 128), (8, 1024)):
        for phase in ("prefill", "decode"):
            for enabled in (False, True):
                database = ContractDatabase(system_spec)
                if phase == "prefill":
                    operation = ContextAttention("contract", 1, 16, 8,
                                                 common.KVCacheQuantMode.bfloat16,
                                                 common.FMHAQuantMode.bfloat16,
                                                 head_size=128, use_qk_norm=enabled)
                    result = operation.query(database, batch_size=batch_size,
                                             s=sequence_length, prefix=0)
                else:
                    operation = GenerationAttention("contract", 1, 16, 8,
                                                    common.KVCacheQuantMode.bfloat16,
                                                    head_size=128, use_qk_norm=enabled)
                    result = operation.query(database, batch_size=batch_size,
                                             s=sequence_length, beam_width=1)
                rows.append({"phase": phase, "batch_size": batch_size,
                             "sequence_length": sequence_length, "use_qk_norm": enabled,
                             "synthetic_total_ms": float(result),
                             "analytic_mem_call_count": len(database.mem_calls),
                             "analytic_mem_call_bytes": database.mem_calls})
    differences = []
    for batch_size, sequence_length in ((1, 128), (8, 1024)):
        for phase in ("prefill", "decode"):
            values = [row for row in rows if row["phase"] == phase
                      and row["batch_size"] == batch_size
                      and row["sequence_length"] == sequence_length]
            differences.append({"phase": phase, "batch_size": batch_size,
                                "sequence_length": sequence_length,
                                "qk_norm_analytic_delta_ms_per_layer":
                                values[1]["synthetic_total_ms"] - values[0]["synthetic_total_ms"]})
    return {"kind": "CPU_SOURCE_CONTRACT_PROBE_NOT_BENCHMARK",
            "warning": "Synthetic attention result is 0.01 ms. No CUDA call, performance-table measurement, training label, or accuracy result is produced.",
            "aiconfigurator_version": importlib.metadata.version("aiconfigurator"),
            "attention_geometry": {"heads": 16, "kv_heads": 8, "head_dim": 128},
            "system_mem_parameters": {key: system_spec["gpu"][key] for key in
                                      ("mem_bw", "mem_bw_empirical_scaling_factor",
                                       "mem_empirical_constant_latency")},
            "calls": rows, "qk_norm_flag_differences": differences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-yaml", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()
    result = probe(yaml.safe_load(args.system_yaml.read_text()))
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    for row in result["qk_norm_flag_differences"]:
        print(f"{row['phase']} B={row['batch_size']} S={row['sequence_length']}: "
              f"analytic delta={row['qk_norm_analytic_delta_ms_per_layer']:.9f} ms/layer (NOT measured)")


if __name__ == "__main__":
    main()
