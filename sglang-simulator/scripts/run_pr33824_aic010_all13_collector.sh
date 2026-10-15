#!/usr/bin/env bash
# Collect local A100 PCIe operator data used by the PR #33824 accuracy matrix.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONNOUSERSITE=1
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
aic_source=${AICONFIGURATOR_SOURCE_ROOT:-/home/ms-admin/workspace1/yiheng/ali/aiconfigurator-v0.10.0}
results_root=
phase=all
resume=0
models_csv=${SGLANG_BENCH_MODELS:-Qwen3-0.6B,Qwen2.5-3B-Instruct,llama-2-7b,Mistral-7B-Instruct-v0.2,Qwen2.5-7B-Instruct,Qwen3-8B,Qwen2.5-14B-Instruct,Qwen2.5-14B-Instruct-1M,Qwen3-14B,Meta-Llama-3.1-70B-Instruct-AWQ,Qwen3-30B-A3B,Qwen3-30B-A3B-Instruct-2507,Qwen3-32B}
attention_backend=${AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND:-triton}
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:-}
compat_kernel_enabled=0

usage() {
    cat <<'EOF'
Usage: run_pr33824_aic010_all13_collector.sh [--all|--dense|--moe] [options]

Options:
  --all                  Collect dense and Qwen3-30B-A3B MoE operators (default).
  --dense                Collect only the 13-model BF16 GEMM/attention shapes.
  --moe                  Collect only Qwen3-30B-A3B BF16 MoE shapes.
  --results-root PATH    A new result directory, or an existing one with --resume.
  --models CSV           Dense checkpoint directories to calibrate (default: historical all-13 set).
  --attention-backend NAME
                         Collector attention path: triton (default) or explicit torch_sdpa proxy.
  --resume               Continue checkpoints in --results-root.

Set SGLANG_KERNEL_COMPAT_STAGE to the isolated SM80 compatibility stage when
the paired real SGLang service uses that package. The stage remains first in
PYTHONPATH; collector_shims only supplies the spawned-worker adapter.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) phase=all; shift ;;
        --dense) phase=dense; shift ;;
        --moe) phase=moe; shift ;;
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --attention-backend) attention_backend=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -f "$aic_source/collector/collect.py" ]] || { echo "Missing AIConfigurator source: $aic_source" >&2; exit 1; }
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "A100 is not visible"'
[[ "$attention_backend" == triton || "$attention_backend" == torch_sdpa ]] || {
    echo "--attention-backend must be triton or torch_sdpa" >&2
    exit 2
}
if [[ -n "$compat_stage" ]]; then
    compat_stage=$(realpath "$compat_stage")
    [[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
        echo "Missing SM80 compatibility binary below: $compat_stage" >&2
        exit 1
    }
    compat_kernel_enabled=1
    unset SGLANG_FORCE_FUSED_OP_BACKEND
    unset SGLANG_WEAK_REF_COMPAT_LIBRARY
    torch_lib_dir=$("$python_bin" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
    export LD_LIBRARY_PATH="$torch_lib_dir:${LD_LIBRARY_PATH:-}"
fi

results_root=${results_root:-"$sim_root/test-results/pr33824_a100_pcie_aic010_all13_$(date +%Y%m%d_%H%M%S)"}
if [[ -e "$results_root" && "$resume" != 1 ]]; then
    echo "Refusing to overwrite existing result directory: $results_root" >&2
    exit 1
fi
if [[ "$resume" == 1 && ! -d "$results_root" ]]; then
    echo "--resume requires an existing result directory" >&2
    exit 1
fi
results_root=$(realpath -m "$results_root")

mkdir -p "$results_root"/{raw,checkpoints,hardware,environment,aic-systems/data/a100_pcie_local/sglang/0.5.10}
systems_root="$results_root/aic-systems"
data_dir="$systems_root/data/a100_pcie_local/sglang/0.5.10"
sglang_cache_dir=${SGLANG_CACHE_DIR:-"$results_root/cache"}
sglang_jit_cache_dir=${SGLANG_JIT_CACHE_DIR:-"$results_root/jit-cache"}
flashinfer_workspace_base=${SGLANG_FLASHINFER_WORKSPACE_BASE:-"$results_root/flashinfer-workspace"}
mkdir -p "$sglang_cache_dir" "$sglang_jit_cache_dir" "$flashinfer_workspace_base"
cp "$script_dir/a100_pcie_local_aic010_system.yaml" "$systems_root/a100_pcie_local.yaml"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader > "$results_root/hardware/gpu_identity.csv"
nvidia-smi -q > "$results_root/hardware/nvidia-smi-q.txt" 2>&1 || true

runtime_shim_dir="$script_dir/runtime_shims"
weak_ref_library=
pythonpath_prefix="$script_dir/collector_shims"
if [[ "$compat_kernel_enabled" == 1 ]]; then
    # sitecustomize remains second so Python-spawned AIC workers receive the
    # current-PR attention adapter; the compiled package stays first and wins
    # every sgl_kernel import.
    pythonpath_prefix="$compat_stage:$script_dir/collector_shims"
elif [[ "$attention_backend" == triton ]]; then
    weak_ref_build_dir="$results_root/environment/weak-ref-compat-build"
    MAX_JOBS=1 "$python_bin" "$script_dir/build_pr33824_weak_ref_compat.py" \
        --output-dir "$weak_ref_build_dir" > "$results_root/environment/weak-ref-compat-build.log" 2>&1
    weak_ref_library=$(find "$weak_ref_build_dir" -maxdepth 1 -type f -name 'sglang_weak_ref_compat*.so' -print -quit)
    [[ -n "$weak_ref_library" && -f "$weak_ref_library" ]] || {
        echo "Could not build the weak-reference compatibility library; see environment/weak-ref-compat-build.log" >&2
        exit 1
    }
    pythonpath_prefix="$runtime_shim_dir:$pythonpath_prefix"
fi

collector_env=(
    AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE=0.5.10
    AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS=1
    AIC_COLLECT_GEMM_TYPES=bfloat16
    AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND="$attention_backend"
    SGLANG_CACHE_DIR="$sglang_cache_dir"
    SGLANG_JIT_CACHE_DIR="$sglang_jit_cache_dir"
    FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base"
    PYTHONPATH="$pythonpath_prefix:$aic_source:$sglang_root/python:$sim_root/src${PYTHONPATH:+:$PYTHONPATH}"
)
if [[ "$compat_kernel_enabled" != 1 ]]; then
    collector_env+=(
        SGLANG_FORCE_FUSED_OP_BACKEND=torch
        SGLANG_WEAK_REF_COMPAT_LIBRARY="$weak_ref_library"
    )
fi

# Resolve imports in the exact collector environment and compare native
# binaries with the real-baseline snapshot before any operator sampling.
if [[ "$compat_kernel_enabled" == 1 ]]; then
    provenance_root="$results_root/provenance_before_wide_sweep"
    reference_hashes="$(dirname "$results_root")/provenance/formal_start_after_ninja_fix/sha256.json"
    provenance_args=(--sim-root "$sim_root" --aic-source "$aic_source" --kernel-stage "$compat_stage"
        --out-dir "$provenance_root" --require-stage-import)
    [[ ! -f "$reference_hashes" ]] || provenance_args+=(--reference-kernel-checksums "$reference_hashes")
    if [[ ! -e "$provenance_root" ]]; then
        env "${collector_env[@]}" "$python_bin" "$sim_root/tools/capture_pr33824_experiment_provenance.py" "${provenance_args[@]}"
    else
        # A resumed collector must not silently use a changed native library.
        env "${collector_env[@]}" "$python_bin" - "$provenance_root/sha256.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
hashes = json.loads(Path(sys.argv[1]).read_text())
native = {name: value for name, value in hashes.items() if name.startswith("/") and name.endswith(".so")}
assert native, "no recorded native kernel checksums"
for name, value in native.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == value, f"changed native kernel: {name}"
PY
    fi
    env "${collector_env[@]}" "$python_bin" - <<'PY' > "$results_root/environment/collector_runtime_versions.txt"
import torch, sgl_kernel
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_file={torch.__file__}")
print(f"sgl_kernel_file={sgl_kernel.__file__}")
print(f"sgl_kernel_runtime_version={sgl_kernel.__version__}")
PY
fi

run_dense() {
    local cases="$results_root/model-cases-dense-attention.yaml"
    "$python_bin" "$script_dir/generate_pr33824_aic_attention_cases.py" \
        --checkpoint-root "$checkpoint_root" --models "$models_csv" --output "$cases" > "$results_root/generate-dense-attention-cases.log"
    mkdir -p "$results_root/raw/dense" "$results_root/checkpoints/dense"
    local args=("$python_bin" "$script_dir/run_aiconfigurator_collector_compat.py" "$aic_source/collector/collect.py"
        --backend sglang --ops attention_context attention_generation --sm 80
        --model-cases "$cases" --checkpoint-dir "$results_root/checkpoints/dense" --keep-csv)
    [[ "$resume" == 1 ]] && args+=(--resume --resume-retry-failed)
    printf '%q ' env "${collector_env[@]}" "${args[@]}" > "$results_root/collector-dense.command.txt"
    printf '\n' >> "$results_root/collector-dense.command.txt"
    (cd "$results_root/raw/dense" && env "${collector_env[@]}" "${args[@]}") \
        2>&1 | tee "$results_root/collector-dense.log"
    cp "$results_root/raw/dense"/context_attention_perf.txt "$data_dir/"
    cp "$results_root/raw/dense"/generation_attention_perf.txt "$data_dir/"
    if [[ "$attention_backend" == triton ]]; then
        awk -F, 'NR > 1 && $5 != "triton" { exit 1 }' "$data_dir/context_attention_perf.txt" || {
            echo "Context attention collection did not record kernel_source=triton" >&2
            return 1
        }
        awk -F, 'NR > 1 && $5 != "triton" { exit 1 }' "$data_dir/generation_attention_perf.txt" || {
            echo "Decode attention collection did not record kernel_source=triton" >&2
            return 1
        }
    fi
}

run_exact_gemm_coverage() {
    # The AIC case YAML is a selector over a broad base sweep.  It cannot
    # express every dynamic runtime M value (notably the SGLang warm-up batch
    # of 6), nor Qwen3-0.6B's physical QKV width.  Measure the actual 13-model
    # GEMM keys after the broad dense pass and make this verified table the one
    # used by the Silicon database.
    local exact_dir="$results_root/raw/exact-target-gemm"
    mkdir -p "$exact_dir"
    local args=("$python_bin" "$script_dir/collect_pr33824_exact_gemm.py"
        --checkpoint-root "$checkpoint_root" --models "$models_csv" --output "$exact_dir/gemm_perf.txt" --seed-perf "$data_dir/gemm_perf.txt")
    printf '%q ' env "${collector_env[@]}" "${args[@]}" > "$results_root/collector-exact-gemm.command.txt"
    printf '\n' >> "$results_root/collector-exact-gemm.command.txt"
    env "${collector_env[@]}" "${args[@]}" 2>&1 | tee "$results_root/collector-exact-gemm.log"
    cp "$exact_dir/gemm_perf.txt" "$data_dir/gemm_perf.txt"
}

run_moe() {
    mkdir -p "$results_root/raw/moe" "$results_root/checkpoints/moe"
    local cases="$aic_source/collector/cases/models/Qwen3MoeForCausalLM_cases.yaml"
    local args=("$python_bin" "$script_dir/run_aiconfigurator_collector_compat.py" "$aic_source/collector/collect.py"
        --backend sglang --ops moe --sm 80 --model-path Qwen/Qwen3-30B-A3B
        --model-cases "$cases" --checkpoint-dir "$results_root/checkpoints/moe" --keep-csv)
    [[ "$resume" == 1 ]] && args+=(--resume --resume-retry-failed)
    local moe_collector_env=("${collector_env[@]}" AICONFIGURATOR_COLLECTOR_QWEN3_MOE_ONLY=1)
    printf '%q ' env "${moe_collector_env[@]}" "${args[@]}" > "$results_root/collector-moe.command.txt"
    printf '\n' >> "$results_root/collector-moe.command.txt"
    (cd "$results_root/raw/moe" && env "${moe_collector_env[@]}" "${args[@]}") \
        2>&1 | tee "$results_root/collector-moe.log"
    local expected_moe_rows=22  # header + 21 BF16 TP=EP=1 A100 measurements
    local actual_moe_rows
    actual_moe_rows=$(wc -l < "$results_root/raw/moe/moe_perf.txt")
    [[ "$actual_moe_rows" -eq "$expected_moe_rows" ]] || {
        echo "Expected $expected_moe_rows complete MoE rows, got $actual_moe_rows" >&2
        return 1
    }
    cp "$results_root/raw/moe"/*_perf.txt "$data_dir/"
}

if [[ "$phase" == all || "$phase" == dense ]]; then
    run_dense
    run_exact_gemm_coverage
fi
if [[ "$phase" == all || "$phase" == moe ]]; then run_moe; fi

"$python_bin" "$script_dir/write_pr33824_aic_config.py" \
    --output "$results_root/simulator-config.a100_pcie_local_aic010.json" \
    --database-path "$systems_root" --backend-version 0.5.10

printf 'aiconfigurator_version=0.10.0\ncollector_source=%s\ncollector_backend_version=0.5.10\ncollector_attention_backend=%s\nselected_dense_models=%s\ncompleted_phase=%s\n' \
    "$(git -C "$aic_source" rev-parse HEAD)" "$attention_backend" "$models_csv" "$phase" >> "$results_root/run_metadata.txt"
printf 'flashinfer_workspace_base=%s\n' "$flashinfer_workspace_base" >> "$results_root/run_metadata.txt"
printf 'sglang_cache_dir=%s\nsglang_jit_cache_dir=%s\n' "$sglang_cache_dir" "$sglang_jit_cache_dir" >> "$results_root/run_metadata.txt"
if [[ "$compat_kernel_enabled" == 1 ]]; then
    printf 'kernel_compat_stage=%s\nkernel_execution_path=isolated SM80 AOT package; SGLANG_FORCE_FUSED_OP_BACKEND=unset; collector_shims retained only for sitecustomize child-process adapter\n' \
        "$compat_stage" >> "$results_root/run_metadata.txt"
else
    printf 'kernel_compat_stage=disabled\nkernel_execution_path=historical Torch fallback\n' >> "$results_root/run_metadata.txt"
fi
echo "AIC database: $data_dir"
echo "Simulator config: $results_root/simulator-config.a100_pcie_local_aic010.json"
