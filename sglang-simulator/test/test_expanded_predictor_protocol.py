"""Request budget and GPU capacity regression tests; no GPU is needed."""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpu_capacity", ROOT / "tools/wait_pr33824_gpu_capacity.py")
assert SPEC and SPEC.loader
capacity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capacity)


def test_capacity_budget():
    assert capacity.required_memory(81920, 0.2, 1024) == 17408
    assert capacity.required_memory(81920, 0.93, 1024) == 77210
    assert capacity.MODEL_MINIMUM_FREE_MIB["Qwen2.5-7B-Instruct"] == 57344
    assert capacity.MODEL_MINIMUM_FREE_MIB["Qwen3-8B"] == 57344


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan"), float("inf")])
def test_reject_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        capacity.required_memory(81920, fraction, 1024)


def test_expanded_equal_request_budget(tmp_path):
    checkpoint = tmp_path / "models/Qwen3-0.6B"
    checkpoint.mkdir(parents=True)
    stage = tmp_path / "stage/sgl_kernel/sm100"
    stage.mkdir(parents=True)
    (stage / "common_ops.abi3.so").touch()
    env = dict(os.environ, SGLANG_CHECKPOINT_ROOT=str(checkpoint.parent),
               SGLANG_KERNEL_COMPAT_STAGE=str(tmp_path / "stage"),
               SGLANG_SAME_DISTRIBUTION_TRAIN_CASES="latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf",
               SGLANG_SAME_DISTRIBUTION_HOLDOUT_CASES="latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf,latency_128x64_r1,saturation_1536x384_inf")
    result = subprocess.run(["bash", str(ROOT / "scripts/run_pr33824_data_tools_same_distribution_matrix.sh"),
                             "--models", "Qwen3-0.6B", "--dry-run"], env=env, text=True, capture_output=True, check=True)
    assert "Qwen3-0.6B\t0.20\t800\t1280" in result.stdout
    env["SGLANG_SAME_DISTRIBUTION_TRAIN_CASES"] = "saturation_1024x256_inf,saturation_1024x256_inf"
    bad = subprocess.run(["bash", str(ROOT / "scripts/run_pr33824_data_tools_same_distribution_matrix.sh"),
                          "--models", "Qwen3-0.6B", "--dry-run"], env=env, text=True, capture_output=True)
    assert bad.returncode != 0
    assert "duplicate workload" in bad.stderr
