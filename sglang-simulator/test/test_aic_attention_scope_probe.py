"""Check analytic query scope; not a GPU operator accuracy benchmark."""
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "attention_scope_probe", ROOT / "tools/probe_pr33824_aic_attention_scope.py"
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_qk_norm_prefill_is_analytic_but_decode_has_no_extra_query():
    spec = yaml.safe_load((ROOT / "scripts/a100_pcie_local_aic010_system.yaml").read_text())
    result = probe.probe(spec)
    assert result["kind"] == "CPU_SOURCE_CONTRACT_PROBE_NOT_BENCHMARK"
    differences = result["qk_norm_flag_differences"]
    prefill = [row["qk_norm_analytic_delta_ms_per_layer"] for row in differences if row["phase"] == "prefill"]
    decode = [row["qk_norm_analytic_delta_ms_per_layer"] for row in differences if row["phase"] == "decode"]
    assert prefill[0] > 0 and prefill[0] == pytest.approx(prefill[1])
    assert decode == [0, 0]
    assert all(row["analytic_mem_call_count"] == 0 for row in result["calls"] if row["phase"] == "decode")
    assert {row["analytic_mem_call_count"] for row in result["calls"] if row["phase"] == "prefill"} == {3, 5}
