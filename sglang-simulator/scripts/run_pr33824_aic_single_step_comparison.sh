#!/usr/bin/env bash
# Compare AIC Silicon static-forward estimates with real A100 one-batch steps.
#
# The real side deliberately keeps the same Triton + CUDA-Graph configuration
# as the completed dense-3 matrix.  AIC is queried separately because its
# Silicon model sums eager operator timings and cannot replay a CUDA Graph.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
aic_source=${AICONFIGURATOR_SOURCE_ROOT:-/home/ms-admin/workspace1/yiheng/ali/aiconfigurator-v0.10.0}
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
reference_root=${SGLANG_REFERENCE_MATRIX_ROOT:-$sim_root/test-results/pr33824_piecewise_a100_dense3_matrix_20260908_134523}
sim_config=${SGLANG_SIM_CONFIG_PATH:-$reference_root/simulator-config.json}
results_root=${SGLANG_RESULTS_ROOT:-$sim_root/test-results/pr33824_aic_vs_real_single_step_dense3_$(date +%Y%m%d_%H%M%S)}
models_csv=${SGLANG_BENCH_MODELS:-Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct,Qwen3-32B}
input_len=${SGLANG_SINGLE_STEP_INPUT_LEN:-1024}
output_len=${SGLANG_SINGLE_STEP_OUTPUT_LEN:-2}
repeats=${SGLANG_BENCH_REPEATS:-5}
context_length=${SGLANG_CONTEXT_LENGTH:-8192}
mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC:-0.80}
mem_fraction_static_overrides=${SGLANG_MEM_FRACTION_STATIC_OVERRIDES:-}
real_attention_backend=${SGLANG_REAL_ATTENTION_BACKEND:-triton}
real_cuda_graph_config=${SGLANG_REAL_CUDA_GRAPH_CONFIG:-'{"decode":{"backend":"full","max_bs":32},"prefill":{"backend":"tc_piecewise","max_bs":2048,"tc_compiler":"eager"}}'}
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:-}
compat_kernel_enabled=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_pr33824_aic_single_step_comparison.sh [options]

Run a low-level, graph-enabled A100 one-batch measurement and a direct AIC
Silicon query for the same three dense checkpoints.

Options:
  --results-root PATH  New immutable result directory.
  --models CSV         Checkpoint directory names below /mnt/data/models.
  --input-len N        Tokens in the single prefill step (default: 1024).
  --repeats N          Same-shape real measurements per model (default: 5).
  --mem-fraction-static FRACTION
                       Default real-server static GPU-memory fraction (default: 0.80).
  --mem-fraction-static-overrides CSV
                       Per-model real-server overrides, e.g. Qwen3-32B=0.96.
  --sim-config PATH    AIC Silicon configuration from the calibrated matrix.
  -h, --help           Show this help.

Set SGLANG_KERNEL_COMPAT_STAGE to the isolated SM80 AOT stage when the
paired collection and real one-batch measurement must use that package.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --input-len) input_len=${2:?missing value}; shift 2 ;;
        --repeats) repeats=${2:?missing value}; shift 2 ;;
        --mem-fraction-static) mem_fraction_static=${2:?missing value}; shift 2 ;;
        --mem-fraction-static-overrides) mem_fraction_static_overrides=${2:?missing value}; shift 2 ;;
        --sim-config) sim_config=${2:?missing value}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$input_len" =~ ^[2-9][0-9]*$|^[1-9][0-9]{2,}$ ]] || {
    echo "--input-len must be at least 2" >&2
    exit 2
}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be positive" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }

is_valid_mem_fraction() {
    "$python_bin" - "$1" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0 < value <= 1 else 1)
PY
}

if ! is_valid_mem_fraction "$mem_fraction_static"; then
    echo "--mem-fraction-static must be a finite number in (0, 1]" >&2
    exit 2
fi
if [[ -n "$mem_fraction_static_overrides" ]]; then
    IFS=',' read -r -a mem_fraction_override_entries <<< "$mem_fraction_static_overrides"
    for mem_fraction_override_entry in "${mem_fraction_override_entries[@]}"; do
        if [[ "$mem_fraction_override_entry" != *=* || -z "${mem_fraction_override_entry%%=*}" || -z "${mem_fraction_override_entry#*=}" ]]; then
            echo "--mem-fraction-static-overrides entries must be MODEL=FRACTION" >&2
            exit 2
        fi
        if ! is_valid_mem_fraction "${mem_fraction_override_entry#*=}"; then
            echo "Invalid per-model static GPU-memory fraction: $mem_fraction_override_entry" >&2
            exit 2
        fi
    done
fi
[[ -d "$aic_source" ]] || { echo "Missing AIConfigurator source: $aic_source" >&2; exit 1; }
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
[[ -f "$sim_config" ]] || { echo "Missing AIC simulator config: $sim_config" >&2; exit 1; }
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

results_root=$(realpath -m "$results_root")
if [[ -e "$results_root" ]]; then
    echo "Refusing to overwrite existing result directory: $results_root" >&2
    exit 1
fi
mkdir -p "$results_root"/{models,hardware,environment,analysis,cache,jit-cache,flashinfer-workspace}

lock_file="$results_root/.run.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another process owns $results_root" >&2
    exit 1
fi
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date -Is)" > "$results_root/launcher.pid"
exec > >(tee -a "$results_root/runner.log") 2>&1

runtime_shim_dir="$script_dir/runtime_shims"
collector_shim_dir="$script_dir/collector_shims"
real_pythonpath="$runtime_shim_dir:$collector_shim_dir:$sglang_root/python:$sim_root/src${PYTHONPATH:+:$PYTHONPATH}"
aic_pythonpath="$collector_shim_dir:$aic_source:$sglang_root/python:$sim_root/src${PYTHONPATH:+:$PYTHONPATH}"
sglang_cache_dir=${SGLANG_CACHE_DIR:-"$results_root/cache"}
sglang_jit_cache_dir=${SGLANG_JIT_CACHE_DIR:-"$results_root/jit-cache"}
flashinfer_workspace_base=${SGLANG_FLASHINFER_WORKSPACE_BASE:-"$results_root/flashinfer-workspace"}
mkdir -p "$sglang_cache_dir" "$sglang_jit_cache_dir" "$flashinfer_workspace_base"
if [[ "$compat_kernel_enabled" == 1 ]]; then
    # Keep the AOT package first.  The collector adapter stays second for
    # current-PR child-process compatibility, but must not shadow sgl_kernel.
    real_pythonpath="$compat_stage:$sglang_root/python:$sim_root/src"
    aic_pythonpath="$compat_stage:$collector_shim_dir:$aic_source:$sglang_root/python:$sim_root/src"
fi

weak_ref_library=
if [[ "$compat_kernel_enabled" != 1 ]]; then
    MAX_JOBS=1 "$python_bin" "$script_dir/build_pr33824_weak_ref_compat.py" \
        --output-dir "$results_root/environment/weak-ref-compat-build" \
        > "$results_root/environment/weak-ref-compat-build.log" 2>&1
    weak_ref_library=$(find "$results_root/environment/weak-ref-compat-build" -maxdepth 1 -type f -name 'sglang_weak_ref_compat*.so' -print -quit)
    [[ -n "$weak_ref_library" && -f "$weak_ref_library" ]] || {
        echo "Could not build weak_ref_tensor compatibility library" >&2
        exit 1
    }
fi

nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader \
    > "$results_root/hardware/gpu_identity.csv"
nvidia-smi -q > "$results_root/hardware/nvidia-smi-q.txt" 2>&1 || true

env PYTHONPATH="$real_pythonpath" SGLANG_CACHE_DIR="$sglang_cache_dir" \
    SGLANG_JIT_CACHE_DIR="$sglang_jit_cache_dir" FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base" \
    "$python_bin" - <<'PY' > "$results_root/package_versions.txt"
import importlib.metadata as md
import sys
import torch
import triton

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"triton={triton.__version__}")
try:
    import sgl_kernel

    print(f"sgl_kernel_file={sgl_kernel.__file__}")
    for operation in ("rmsnorm", "fused_add_rmsnorm", "silu_and_mul", "rotary_embedding", "weak_ref_tensor"):
        print(f"torch.ops.sgl_kernel.{operation}={hasattr(torch.ops.sgl_kernel, operation)}")
except Exception as exc:
    print(f"sgl_kernel_import_error={type(exc).__name__}: {exc}")
for name in ("sglang", "sglang-kernel", "aiconfigurator", "sglang-simulator"):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=not-installed")
PY

{
    printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
    printf 'source_path=%s\n' "$sglang_root"
    printf 'python=%s\nmodels=%s\n' "$python_bin" "$models_csv"
    printf 'simulator_config=%s\n' "$(realpath "$sim_config")"
    printf 'input_len=%s\noutput_len=%s\nrepeats=%s\n' "$input_len" "$output_len" "$repeats"
    printf 'context_length=%s\nmem_fraction_static_default=%s\nmem_fraction_static_overrides=%s\n' \
        "$context_length" "$mem_fraction_static" "${mem_fraction_static_overrides:-none}"
    printf 'real_attention_backend=%s\nreal_cuda_graph_config=%s\n' "$real_attention_backend" "$real_cuda_graph_config"
    printf 'flashinfer_workspace_base=%s\nsglang_cache_dir=%s\nsglang_jit_cache_dir=%s\n' \
        "$flashinfer_workspace_base" "$sglang_cache_dir" "$sglang_jit_cache_dir"
    printf 'real_measurement=upstream sglang.benchmark.one_batch; full checkpoint; batch=1; one prefill and first decode step; five same-shape repeats in one model process\n'
    printf 'aic_measurement=direct AIConfigurator SILICON static operator composition for identical model and shape; it does not replay CUDA Graph\n'
    printf 'aic_collector_graph_mode=eager attention operation; collector compatibility adapter explicitly disables CUDA Graph\n'
    printf 'attention_kernel_alignment=real one_batch and AIC collector use current-PR TritonAttnBackend; collector rows are kernel_source=triton\n'
    if [[ "$compat_kernel_enabled" == 1 ]]; then
        printf 'kernel_compat_stage=%s\nkernel_execution_path=isolated SM80 AOT package; real one-batch uses it with SGLANG_FORCE_FUSED_OP_BACKEND=unset; direct AIC query reads the paired Silicon table and does not execute CUDA kernels\n' "$compat_stage"
    else
        printf 'kernel_compat_stage=disabled\nkernel_execution_path=historical Torch fallback; installed cu129 wheel lacks SM80 support and fails active Torch ABI\n'
    fi
} > "$results_root/run_metadata.txt"

real_env=(
    env -u SGLANG_SIMULATOR_CONFIG_PATH -u SGLANG_SIMULATOR_OUTPUT_DIR -u SGLANG_USE_CPU_ENGINE
)
if [[ "$compat_kernel_enabled" == 1 ]]; then
    real_env+=(-u SGLANG_FORCE_FUSED_OP_BACKEND -u SGLANG_WEAK_REF_COMPAT_LIBRARY)
else
    real_env+=(SGLANG_WEAK_REF_COMPAT_LIBRARY="$weak_ref_library" SGLANG_FORCE_FUSED_OP_BACKEND=torch)
fi
real_env+=(
    PYTHONPATH="$real_pythonpath"
    SGLANG_CACHE_DIR="$sglang_cache_dir"
    SGLANG_JIT_CACHE_DIR="$sglang_jit_cache_dir"
    FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base"
)
aic_env=(env)
if [[ "$compat_kernel_enabled" == 1 ]]; then
    aic_env+=(-u SGLANG_FORCE_FUSED_OP_BACKEND -u SGLANG_WEAK_REF_COMPAT_LIBRARY)
fi
aic_env+=(
    PYTHONPATH="$aic_pythonpath" SGLANG_CACHE_DIR="$sglang_cache_dir"
    SGLANG_JIT_CACHE_DIR="$sglang_jit_cache_dir" FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base"
)

# A one-batch run still loads a full checkpoint and captures its CUDA graphs.
# Do not start it while an unrelated process has consumed the headroom needed
# by the largest dense checkpoint; the values remain overridable for a
# deliberately shared GPU.
wait_for_real_gpu_capacity() {
    local required_mib=${SGLANG_MIN_FREE_GPU_MIB:-76800}
    local poll_seconds=${SGLANG_GPU_CAPACITY_POLL_SECONDS:-600}
    local free_mib
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9' || true)
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= required_mib )); then
            printf '[gpu-capacity] free=%s MiB, required=%s MiB; starting one-batch real measurement\n' "$free_mib" "$required_mib"
            return 0
        fi
        printf '[gpu-capacity] free=%s MiB, required=%s MiB; waiting %s seconds before one-batch real measurement\n' \
            "${free_mib:-unknown}" "$required_mib" "$poll_seconds"
        sleep "$poll_seconds"
    done
}

resolve_model_mem_fraction() {
    local model=$1 entry name value resolved=$mem_fraction_static
    if [[ -n "$mem_fraction_static_overrides" ]]; then
        IFS=',' read -r -a mem_fraction_override_entries <<< "$mem_fraction_static_overrides"
        for entry in "${mem_fraction_override_entries[@]}"; do
            name=${entry%%=*}
            value=${entry#*=}
            if [[ "$name" == "$model" ]]; then
                resolved=$value
            fi
        done
    fi
    printf '%s\n' "$resolved"
}

IFS=',' read -r -a models <<< "$models_csv"
input_lengths=()
for _ in $(seq 1 "$repeats"); do
    input_lengths+=("$input_len")
done

for model in "${models[@]}"; do
    model_path="$checkpoint_root/$model"
    model_dir="$results_root/models/$model"
    model_mem_fraction_static=$(resolve_model_mem_fraction "$model")
    [[ -d "$model_path" ]] || { echo "Missing checkpoint: $model_path" >&2; exit 1; }
    mkdir -p "$model_dir/real"

    {
        printf 'model_path=%s\ncontext_length=%s\n' "$model_path" "$context_length"
        printf 'mem_fraction_static=%s\nattention_backend=%s\n' \
            "$model_mem_fraction_static" "$real_attention_backend"
        printf 'cuda_graph_config=%s\ncompat_kernel_enabled=%s\n' \
            "$real_cuda_graph_config" "$compat_kernel_enabled"
    } > "$model_dir/real/launch-settings.txt"

    if [[ "$compat_kernel_enabled" == 1 ]]; then
        "${real_env[@]}" "$python_bin" - <<'PY' > "$model_dir/real/kernel-identity.txt"
import sgl_kernel
import torch

print(f"sgl_kernel_file={sgl_kernel.__file__}")
for operation in ("rmsnorm", "fused_add_rmsnorm", "silu_and_mul", "rotary_embedding", "weak_ref_tensor"):
    print(f"torch.ops.sgl_kernel.{operation}={hasattr(torch.ops.sgl_kernel, operation)}")
PY
    fi

    printf '%q ' "${real_env[@]}" \
        "$python_bin" -m sglang.benchmark.one_batch \
        --model-path "$model_path" --device cuda --tp-size 1 --context-length "$context_length" \
        --mem-fraction-static "$model_mem_fraction_static" --max-running-requests 32 --chunked-prefill-size 2048 \
        --attention-backend "$real_attention_backend" --cuda-graph-config "$real_cuda_graph_config" \
        --batch-size 1 --input-len "${input_lengths[@]}" --output-len "$output_len" \
        --run-name "aic_single_step_$model" --result-filename "$model_dir/real/one_batch.jsonl" \
        > "$model_dir/real/command.txt"
    printf '\n' >> "$model_dir/real/command.txt"

    wait_for_real_gpu_capacity
    "${real_env[@]}" \
        "$python_bin" -m sglang.benchmark.one_batch \
        --model-path "$model_path" --device cuda --tp-size 1 --context-length "$context_length" \
        --mem-fraction-static "$model_mem_fraction_static" --max-running-requests 32 --chunked-prefill-size 2048 \
        --attention-backend "$real_attention_backend" --cuda-graph-config "$real_cuda_graph_config" \
        --batch-size 1 --input-len "${input_lengths[@]}" --output-len "$output_len" \
        --run-name "aic_single_step_$model" --result-filename "$model_dir/real/one_batch.jsonl" \
        > "$model_dir/real/one_batch.log" 2>&1

    rg -q 'Capture target prefill CUDA graph begin.*backend=tc_piecewise' \
        "$model_dir/real/one_batch.log" || {
        echo "Missing tc_piecewise prefill CUDA-Graph capture evidence for $model" >&2
        exit 1
    }
    rg -q 'Capture target decode CUDA graph begin.*backend=full' \
        "$model_dir/real/one_batch.log" || {
        echo "Missing full decode CUDA-Graph capture evidence for $model" >&2
        exit 1
    }

    "$python_bin" - "$model_dir/real/one_batch.jsonl" "$repeats" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
if len(rows) != expected:
    raise SystemExit(f"Expected {expected} result rows, found {len(rows)} in {path}")
for index, row in enumerate(rows, 1):
    for key in ("prefill_latency", "median_decode_latency"):
        if not isinstance(row.get(key), (float, int)) or row[key] <= 0:
            raise SystemExit(f"Invalid {key} in repeat {index}: {row.get(key)!r}")
PY
    printf 'ok\n' > "$model_dir/real/status.txt"

    printf '%q ' "${aic_env[@]}" \
        "$python_bin" "$script_dir/query_pr33824_aic_single_step.py" \
        --model-path "$model_path" --sim-config "$sim_config" --input-len "$input_len" \
        --batch-size 1 --output "$model_dir/aic_prediction.json" \
        > "$model_dir/aic_query.command.txt"
    printf '\n' >> "$model_dir/aic_query.command.txt"
    "${aic_env[@]}" \
        "$python_bin" "$script_dir/query_pr33824_aic_single_step.py" \
        --model-path "$model_path" --sim-config "$sim_config" --input-len "$input_len" \
        --batch-size 1 --output "$model_dir/aic_prediction.json" \
        > "$model_dir/aic_query.log" 2>&1
    "$python_bin" -m json.tool "$model_dir/aic_prediction.json" > /dev/null
    printf 'ok\n' > "$model_dir/aic_query.status.txt"
done

"$python_bin" "$script_dir/summarize_pr33824_aic_single_step.py" --results-root "$results_root" \
    > "$results_root/summary.log" 2>&1
printf 'status=complete\nmodels=%s\n' "${#models[@]}" > "$results_root/finalization.status"
echo "Completed: $results_root"
