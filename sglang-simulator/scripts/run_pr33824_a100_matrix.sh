#!/usr/bin/env bash
# Run an A100 accuracy matrix with the complete PR #33824 simulator.
# The real service runs CUDA kernels; the PR simulator deliberately runs its
# scheduler/cache runtime on CPU and uses the predictor selected in sim-config.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
shim_dir="$script_dir/collector_shims"
runtime_shim_dir="$script_dir/runtime_shims"
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
sim_config=${SGLANG_SIM_CONFIG_PATH:-$sim_root/test-results/pr33824_a100_pcie_aic010_all13_20260906_1755/simulator-config.a100_pcie_local_aic010.json}
results_root=${SGLANG_RESULTS_ROOT:-$sim_root/test-results/pr33824_a100_all13_fixed_trace_$(date +%Y%m%d_%H%M%S)}
repeats=${SGLANG_BENCH_REPEATS:-5}
prediction_repeat_policy=${SGLANG_PREDICTION_REPEAT_POLICY:-two-then-five}
skip_matrix_render=${SGLANG_SKIP_MATRIX_RENDER:-0}
real_attention_backend=${SGLANG_REAL_ATTENTION_BACKEND:-torch_native}
real_cuda_graph_config=${SGLANG_REAL_CUDA_GRAPH_CONFIG:-}
real_enable_overlap_schedule=${SGLANG_REAL_ENABLE_OVERLAP_SCHEDULE:-0}
# The static-pool fraction is deliberately an experiment input, not an
# inferred value.  A large checkpoint can require a higher value merely to
# leave a non-zero KV pool; record every override in the run metadata.
real_mem_fraction_static=${SGLANG_REAL_MEM_FRACTION_STATIC:-0.80}
models_csv=${SGLANG_BENCH_MODELS:-Qwen3-0.6B,Qwen2.5-3B-Instruct,llama-2-7b,Mistral-7B-Instruct-v0.2,Qwen2.5-7B-Instruct,Qwen3-8B,Qwen2.5-14B-Instruct,Qwen2.5-14B-Instruct-1M,Qwen3-14B,Meta-Llama-3.1-70B-Instruct-AWQ,Qwen3-30B-A3B,Qwen3-30B-A3B-Instruct-2507,Qwen3-32B}
real_port=${SGLANG_REAL_PORT:-33000}
simulator_port=${SGLANG_SIMULATOR_PORT:-33001}
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:-}
# Allows a per-model wrapper to preserve the original fixed-trace seed when
# reusing its real measurements instead of launching the entire model list.
model_index_offset=${SGLANG_MODEL_INDEX_OFFSET:-0}
compat_kernel_enabled=0
resume=0

usage() {
    cat <<'EOF'
Usage: bash tools/sglang-simulator/scripts/run_pr33824_a100_matrix.sh [options]

Runs the historical all-13-model accuracy design with the complete PR #33824
implementation: three fixed random-ids workloads and five matched repeats.

Options:
  --results-root PATH   New output root (default: timestamped test-results path).
  --models CSV          Checkpoint directory names below /mnt/data/models.
  --repeats N           Real-service repeats per model/workload (default: 5).
  --prediction-repeat-policy POLICY
                       two-then-five (default): run predictor twice and stop
                       when all compared metrics are identical, otherwise run N;
                       matched: run N predictor repetitions.
  --real-attention-backend NAME
                       Attention backend for the real CUDA server (default: torch_native).
  --real-cuda-graph-config JSON
                       Enable and record real CUDA graph settings.  Omit to retain
                       the historical disabled-graph control configuration.
  --real-enable-overlap-schedule
                       Leave real overlap scheduling enabled (default: disabled
                       for the historical control configuration).
  --real-mem-fraction-static FRACTION
                       Static GPU-memory fraction for real-server weights and
                       KV cache (default: 0.80; also SGLANG_REAL_MEM_FRACTION_STATIC).
  --sim-config PATH     Predictor configuration (AIC, ML, or replay).
  --resume              Continue a previously created result root.
  -h, --help            Show this message.

All raw client output, simulator server metrics, commands, logs, metadata, and
per-model progress are written below one immutable result root.

Set SGLANG_KERNEL_COMPAT_STAGE to an isolated SM80 compatibility stage to run
the real service without the historical Torch fallback. The paired AIC database
must be collected with the same environment before interpreting accuracy.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --repeats) repeats=${2:?missing value}; shift 2 ;;
        --prediction-repeat-policy) prediction_repeat_policy=${2:?missing value}; shift 2 ;;
        --real-attention-backend) real_attention_backend=${2:?missing value}; shift 2 ;;
        --real-cuda-graph-config) real_cuda_graph_config=${2:?missing value}; shift 2 ;;
        --real-enable-overlap-schedule) real_enable_overlap_schedule=1; shift ;;
        --real-mem-fraction-static) real_mem_fraction_static=${2:?missing value}; shift 2 ;;
        --sim-config) sim_config=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be positive" >&2; exit 2; }
[[ "$model_index_offset" =~ ^[0-9]+$ ]] || {
    echo "SGLANG_MODEL_INDEX_OFFSET must be a non-negative integer" >&2; exit 2;
}
[[ "$prediction_repeat_policy" == two-then-five || "$prediction_repeat_policy" == matched ]] || {
    echo "--prediction-repeat-policy must be two-then-five or matched" >&2
    exit 2
}
[[ "$real_enable_overlap_schedule" == 0 || "$real_enable_overlap_schedule" == 1 ]] || {
    echo "SGLANG_REAL_ENABLE_OVERLAP_SCHEDULE must be 0 or 1" >&2
    exit 2
}
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
if ! "$python_bin" - "$real_mem_fraction_static" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0 < value <= 1 else 1)
PY
then
    echo "--real-mem-fraction-static must be a finite number in (0, 1]" >&2
    exit 2
fi
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
[[ -f "$sim_config" ]] || { echo "Missing simulator config: $sim_config" >&2; exit 1; }
predictor_name=$(jq -r '.predictor.name // "unknown"' "$sim_config")
if [[ -n "$compat_stage" ]]; then
    compat_stage=$(realpath "$compat_stage")
    [[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
        echo "Missing SM80 compatibility binary below: $compat_stage" >&2
        exit 1
    }
    compat_kernel_enabled=1
    # Explicitly remove the old fallback policy. The stage itself contains
    # the compiled SM80 weak-reference operator used by CUDA graphs.
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
if [[ -e "$results_root" && "$resume" != 1 ]]; then
    echo "Refusing to overwrite existing output root: $results_root" >&2
    exit 1
fi
if [[ "$resume" == 1 && ! -d "$results_root" ]]; then
    echo "--resume requires an existing output root: $results_root" >&2
    exit 1
fi
initialize_run=0
if [[ "$resume" != 1 || ! -s "$results_root/run_metadata.txt" ]]; then
    initialize_run=1
fi

mkdir -p "$results_root" "$results_root/hardware"
flashinfer_workspace_base=${SGLANG_FLASHINFER_WORKSPACE_BASE:-"$results_root/flashinfer-workspace"}
mkdir -p "$flashinfer_workspace_base"
# A --resume invocation must remain the only writer for a result root.  The
# raw directories are intentionally immutable inputs to the final paired
# statistics, so two launchers would make a status.txt/metrics.json pair
# ambiguous even if both clients happen to finish successfully.
lock_file="$results_root/.matrix-run.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another matrix launcher already owns $results_root" >&2
    exit 1
fi
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date -Is)" > "$results_root/.matrix-run.owner"
# The launcher is usually detached for a multi-hour matrix.  Capture its live
# output inside the immutable result root so no experiment status is scattered
# elsewhere in the workspace.
exec > >(tee -a "$results_root/runner.log") 2>&1
# The installed ``sglang-kernel`` wheel is tagged ``+cu129`` (CUDA 12.9), but
# it contains no SM80 common-ops binary and fails against this host's active
# PyTorch CUDA ABI (Torch 2.9.1+cu126). The opt-in package must be first and
# must not inherit a same-name historical fallback shim.
if [[ "$compat_kernel_enabled" == 1 ]]; then
    export PYTHONPATH="$compat_stage:$sglang_root/python:$sim_root/src"
else
    export PYTHONPATH="$shim_dir:$sglang_root/python:$sim_root/src${PYTHONPATH:+:$PYTHONPATH}"
fi
export SGLANG_CACHE_DIR=${SGLANG_CACHE_DIR:-"$results_root/cache"}
export SGLANG_JIT_CACHE_DIR=${SGLANG_JIT_CACHE_DIR:-"$results_root/jit-cache"}
mkdir -p "$SGLANG_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR"
# The compatibility package imports FlashInfer's Python module, which creates
# a log/cache directory at import time even if AOT operators avoid its JIT.
# Keep it inside the immutable experiment root rather than under a user home.
export FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base"

weak_ref_library=
if [[ -n "$real_cuda_graph_config" && "$compat_kernel_enabled" != 1 ]]; then
    weak_ref_build_dir="$results_root/environment/weak-ref-compat-build"
    mkdir -p "$results_root/environment"
    MAX_JOBS=1 "$python_bin" "$script_dir/build_pr33824_weak_ref_compat.py" \
        --output-dir "$weak_ref_build_dir" > "$results_root/environment/weak-ref-compat-build.log" 2>&1
    weak_ref_library=$(find "$weak_ref_build_dir" -maxdepth 1 -type f -name 'sglang_weak_ref_compat*.so' -print -quit)
    [[ -n "$weak_ref_library" && -f "$weak_ref_library" ]] || {
        echo "Could not build the weak-reference CUDA-graph compatibility library" >&2
        exit 1
    }
fi

if [[ "$initialize_run" == 1 ]]; then
    cp "$sim_config" "$results_root/simulator-config.json"
    {
        printf 'pr_head=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
        printf 'pr_branch=%s\n' "$(git -C "$sglang_root" branch --show-current)"
        printf 'source_revision=%s\nsource_path=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)" "$sglang_root"
        printf 'source_version_identity=local PR #33824 source; local package metadata may be 0.0.0; PR compatibility floor is v0.5.16, not a v0.5.18 release\n'
        printf 'simulator_config_source=%s\npredictor_name=%s\n' "$(realpath "$sim_config")" "$predictor_name"
        printf 'python=%s\n' "$python_bin"
        printf 'flashinfer_workspace_base=%s\n' "$flashinfer_workspace_base"
        printf 'sglang_cache_dir=%s\nsglang_jit_cache_dir=%s\n' "$SGLANG_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR"
        printf 'models=%s\nrepeats=%s\nprediction_repeat_policy=%s\nmodel_index_offset=%s\n' "$models_csv" "$repeats" "$prediction_repeat_policy" "$model_index_offset"
        printf 'workloads=latency_256x128_r1;service_1024x256_r4;saturation_1024x256_inf\n'
        printf 'trace_policy=fixed: every real repeat of a model/workload uses the same seed; real repeat dispersion therefore measures runtime variance, not a changed request trace\n'
        printf 'real_attention_backend=%s\nreal_cuda_graph_config=%s\nreal_overlap_schedule_enabled=%s\nreal_mem_fraction_static_default=%s\n' "$real_attention_backend" "${real_cuda_graph_config:-disabled}" "$real_enable_overlap_schedule" "$real_mem_fraction_static"
        if [[ "$compat_kernel_enabled" == 1 ]]; then
            printf 'kernel_compat_stage=%s\nkernel_execution_path=isolated SM80 AOT package; SGLANG_FORCE_FUSED_OP_BACKEND=unset; no runtime/collector sgl_kernel shim\n' "$compat_stage"
            printf 'real_runtime=full checkpoint, CUDA A100 PCIe service; attention_backend=%s; cuda_graph=%s; overlap_scheduler_enabled=%s; compatibility package provides weak_ref_tensor\n' "$real_attention_backend" "${real_cuda_graph_config:-disabled}" "$real_enable_overlap_schedule"
        else
            printf 'kernel_compat_stage=disabled\nkernel_execution_path=historical Torch fallback\n'
            printf 'real_runtime=full checkpoint, CUDA A100 PCIe service; attention_backend=%s; cuda_graph=%s; overlap_scheduler_enabled=%s; the narrow local weak_ref_tensor ABI shim is used only when CUDA graph is enabled\n' "$real_attention_backend" "${real_cuda_graph_config:-disabled}" "$real_enable_overlap_schedule"
        fi
        case "$predictor_name" in
            aiconfigurator)
                printf 'simulator_runtime=complete PR #33824, CPU engine, OFFLINE logical time, AIC Silicon predictor for local A100 PCIe; it predicts operator timing but does not execute physical CUDA graph replay\n'
                ;;
            ml)
                printf 'simulator_runtime=complete PR #33824, CPU engine, OFFLINE logical time, MLTimePredictor loaded from the recorded sklearn/joblib bundle; it does not execute physical CUDA graph replay\n'
                ;;
            replay)
                printf 'simulator_runtime=complete PR #33824, CPU engine, OFFLINE logical time, replay predictor loaded from the recorded table; it does not execute physical CUDA graph replay\n'
                ;;
            *)
                printf 'simulator_runtime=complete PR #33824, CPU engine, OFFLINE logical time, predictor=%s; it does not execute physical CUDA graph replay\n' "$predictor_name"
                ;;
        esac
    } > "$results_root/run_metadata.txt"
    "$python_bin" - <<'PY' > "$results_root/package_versions.txt"
import importlib.metadata as md
import sys
print(f"python={sys.version.split()[0]}")
for name in ("torch", "transformers", "sglang", "sglang-simulator", "aiconfigurator", "numpy"):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=not-installed")
PY
    nvidia-smi -q > "$results_root/hardware/nvidia-smi-q.txt" 2>&1 || true
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader > "$results_root/hardware/gpu_identity.csv" 2>&1 || true
fi
if [[ "$resume" == 1 ]]; then
    resume_metadata="$results_root/resume-$(date +%Y%m%d_%H%M%S)-metadata.txt"
    {
        printf 'resumed_at=%s\nresumed_by_pid=%s\n' "$(date -Is)" "$$"
        printf 'models=%s\nrepeats=%s\nprediction_repeat_policy=%s\nmodel_index_offset=%s\n' "$models_csv" "$repeats" "$prediction_repeat_policy" "$model_index_offset"
        printf 'real_attention_backend=%s\nreal_cuda_graph_config=%s\nreal_overlap_schedule_enabled=%s\nreal_mem_fraction_static_default=%s\n' \
            "$real_attention_backend" "${real_cuda_graph_config:-disabled}" "$real_enable_overlap_schedule" "$real_mem_fraction_static"
        printf 'simulator_config_source=%s\nkernel_compat_stage=%s\n' "$(realpath "$sim_config")" "${compat_stage:-disabled}"
    } > "$resume_metadata"
fi

declare -a cases=(
    'latency_256x128_r1|8|256|128|1'
    'service_1024x256_r4|32|1024|256|4'
    'saturation_1024x256_inf|32|1024|256|inf'
)

server_pid=
gpu_monitor_pid=
active_server_log=
real_max_total_tokens=
# Real CUDA servers can need a little longer to release their scheduler and
# GPU allocation.  The CPU-only OFFLINE simulator is launched for exactly one
# isolated repeat, however, and has already emitted its metrics before this
# cleanup runs.  Avoid spending fifteen seconds on every simulated repeat.
server_shutdown_grace_seconds=15

cleanup_server() {
    if [[ -n "${gpu_monitor_pid:-}" ]]; then
        kill "$gpu_monitor_pid" 2>/dev/null || true
        wait "$gpu_monitor_pid" 2>/dev/null || true
        gpu_monitor_pid=
    fi
    if [[ -n "${server_pid:-}" ]]; then
        kill -- -"$server_pid" 2>/dev/null || true
        for _ in $(seq 1 "$server_shutdown_grace_seconds"); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -KILL -- -"$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}
trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT TERM

wait_for_server() {
    local port=$1
    for _ in $(seq 1 180); do
        if curl --silent --fail --max-time 2 "http://127.0.0.1:$port/v1/models" > /dev/null; then
            return 0
        fi
        if [[ -n "${server_pid:-}" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server exited before readiness; log: $active_server_log" >&2
            tail -120 "$active_server_log" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "Server readiness timeout; log: $active_server_log" >&2
    tail -120 "$active_server_log" >&2 || true
    return 1
}

# A failed model load is not a benchmark result.  The remaining checkpoints
# require most of this 80 GB A100, so do not compete with an unrelated CUDA
# job and then silently move on after an OOM.  Resume automatically once the
# requested capacity is available; the threshold and polling interval remain
# overrideable for a deliberately shared GPU.
wait_for_real_gpu_capacity() {
    local required_mib=${SGLANG_MIN_FREE_GPU_MIB:-76800}
    local poll_seconds=${SGLANG_GPU_CAPACITY_POLL_SECONDS:-600}
    local free_mib
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9' || true)
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= required_mib )); then
            printf '[gpu-capacity] free=%s MiB, required=%s MiB; starting real server\n' "$free_mib" "$required_mib" >&2
            return 0
        fi
        printf '[gpu-capacity] free=%s MiB, required=%s MiB; waiting %s seconds before real-server start\n' \
            "${free_mib:-unknown}" "$required_mib" "$poll_seconds" >&2
        sleep "$poll_seconds"
    done
}

is_real_complete() {
    [[ -s "$1/benchmark.jsonl" && -s "$1/status.txt" ]] && grep -qx ok "$1/status.txt"
}

is_sim_complete() {
    [[ -s "$1/benchmark.jsonl" && -s "$1/simulator-output/metrics.json" && -s "$1/status.txt" ]] && grep -qx ok "$1/status.txt"
}

model_real_complete() {
    local model_dir=$1 spec case_name prompts input_len output_len rate repeat
    for spec in "${cases[@]}"; do
        IFS='|' read -r case_name prompts input_len output_len rate <<< "$spec"
        for repeat in $(seq 1 "$repeats"); do
            is_real_complete "$model_dir/real/$case_name/repeat-$repeat" || return 1
        done
    done
}

start_real_server() {
    local model_path=$1 model_dir=$2 context_length=$3 mem_fraction=$4
    local -a model_override_args=()
    local -a quantization_args=()
    local -a cuda_graph_args=()
    local -a overlap_schedule_args=()
    local real_pythonpath="$shim_dir:$sglang_root/python:$sim_root/src${PYTHONPATH:+:$PYTHONPATH}"
    local fused_op_backend=torch
    if [[ "$compat_kernel_enabled" == 1 ]]; then
        real_pythonpath="$compat_stage:$sglang_root/python:$sim_root/src"
        fused_op_backend=compiled_sm80_aot
    fi
    if jq -e '.dual_chunk_attention_config != null' "$model_path/config.json" > /dev/null; then
        # The largest matrix request is 1,280 tokens, while this checkpoint's
        # dual-chunk transition begins at 262,144 tokens.  Disabling that
        # long-context-only configuration keeps the tested short-context model
        # semantics while allowing the host's supported A100 execution path.
        model_override_args=(--json-model-override-args '{"dual_chunk_attention_config": null}')
        printf '[model-override] disabling dual-chunk attention for short-context matrix; transition=262144 tokens\n' >&2
    fi
    if [[ "$(jq -r '.quantization_config.quant_method // empty' "$model_path/config.json")" == "awq" ]]; then
        # The installed cu129 sglang-kernel binary is not SM80/ABI compatible
        # with this CUDA-12.6 host.  Keep the original 4-bit AWQ checkpoint,
        # but select SGLang's non-Marlin AWQ path rather than failing during
        # Marlin weight repacking.
        quantization_args=(--quantization awq)
        printf '[quantization] forcing native AWQ path; preserving 4-bit checkpoint weights\n' >&2
    fi
    server_shutdown_grace_seconds=${SGLANG_REAL_SERVER_SHUTDOWN_GRACE_SECONDS:-15}
    if [[ -n "$real_cuda_graph_config" ]]; then
        cuda_graph_args=(--cuda-graph-config "$real_cuda_graph_config")
        if [[ "$compat_kernel_enabled" != 1 ]]; then
            real_pythonpath="$runtime_shim_dir:$real_pythonpath"
        fi
    else
        cuda_graph_args=(--disable-cuda-graph)
    fi
    if [[ "$real_enable_overlap_schedule" != 1 ]]; then
        overlap_schedule_args=(--disable-overlap-schedule)
    fi
    local -a server_env=(
        env -u SGLANG_SIMULATOR_CONFIG_PATH -u SGLANG_SIMULATOR_OUTPUT_DIR -u SGLANG_USE_CPU_ENGINE
    )
    if [[ "$compat_kernel_enabled" == 1 ]]; then
        server_env+=(-u SGLANG_FORCE_FUSED_OP_BACKEND -u SGLANG_WEAK_REF_COMPAT_LIBRARY)
    fi
    server_env+=(PYTHONPATH="$real_pythonpath")
    if [[ "$compat_kernel_enabled" != 1 ]]; then
        server_env+=(SGLANG_WEAK_REF_COMPAT_LIBRARY="$weak_ref_library" SGLANG_FORCE_FUSED_OP_BACKEND=torch)
    fi
    cleanup_server
    wait_for_real_gpu_capacity
    active_server_log="$model_dir/real/server.log"
    mkdir -p "$model_dir/real"
    {
        printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
        printf 'python=%s\nmodel_path=%s\ncontext_length=%s\n' \
            "$python_bin" "$model_path" "$context_length"
        printf 'mem_fraction_static=%s\nattention_backend=%s\n' \
            "$mem_fraction" "$real_attention_backend"
        printf 'cuda_graph_config=%s\noverlap_schedule_enabled=%s\n' \
            "${real_cuda_graph_config:-disabled}" "$real_enable_overlap_schedule"
        printf 'fused_op_backend=%s\n' "$fused_op_backend"
    } > "$model_dir/real/launch-settings.txt"
    {
        printf '%q ' "${server_env[@]}" setsid "$python_bin" -m sglang.launch_server \
            --model-path "$model_path" --host 127.0.0.1 --port "$real_port" --device cuda \
            --tp-size 1 --context-length "$context_length" --mem-fraction-static "$mem_fraction" \
            --max-running-requests 32 --chunked-prefill-size 2048 --attention-backend "$real_attention_backend"
        printf '%q ' "${model_override_args[@]}" "${quantization_args[@]}" \
            "${cuda_graph_args[@]}" "${overlap_schedule_args[@]}"
        printf '\n'
    } > "$model_dir/real/server.command.txt"
    "${server_env[@]}" setsid "$python_bin" -m sglang.launch_server \
        --model-path "$model_path" --host 127.0.0.1 --port "$real_port" --device cuda \
        --tp-size 1 --context-length "$context_length" --mem-fraction-static "$mem_fraction" \
        --max-running-requests 32 --chunked-prefill-size 2048 --attention-backend "$real_attention_backend" \
        "${model_override_args[@]}" \
        "${quantization_args[@]}" \
        "${cuda_graph_args[@]}" \
        "${overlap_schedule_args[@]}" \
        > "$active_server_log" 2>&1 &
    server_pid=$!
    wait_for_server "$real_port"
    # A simulator with a CPU dummy model would otherwise size its KV pool from
    # host RAM.  PR #33824 explicitly recommends reusing the real service's
    # effective token capacity for deployment-faithful cache/scheduler state.
    # Read the value emitted after the real service allocates its KV cache.
    real_max_total_tokens=
    for _ in $(seq 1 30); do
        real_max_total_tokens=$(grep -Eo 'max_total_num_tokens=[0-9]+' "$active_server_log" | tail -n 1 | cut -d= -f2 || true)
        [[ "$real_max_total_tokens" =~ ^[1-9][0-9]*$ ]] && break
        sleep 1
    done
    if [[ ! "$real_max_total_tokens" =~ ^[1-9][0-9]*$ ]]; then
        echo "Could not obtain real max_total_num_tokens from $active_server_log" >&2
        return 1
    fi
    printf '%s\n' "$real_max_total_tokens" > "$model_dir/real/max_total_tokens.txt"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 1 > "$model_dir/real/gpu-monitor.csv" 2>&1 &
    gpu_monitor_pid=$!
}

run_real_case() {
    local model_path=$1 run_dir=$2 prompts=$3 input_len=$4 output_len=$5 rate=$6 seed=$7
    mkdir -p "$run_dir"
    if is_real_complete "$run_dir"; then return 0; fi
    printf '%q ' "$python_bin" -m sglang.benchmark.serving --backend sglang --base-url "http://127.0.0.1:$real_port" --model "$model_path" --dataset-name random-ids --num-prompts "$prompts" --random-input-len "$input_len" --random-output-len "$output_len" --random-range-ratio 0 --request-rate "$rate" --seed "$seed" --warmup-requests 0 --flush-cache --disable-tqdm --ready-check-timeout-sec 0 --output-file "$run_dir/benchmark.jsonl" > "$run_dir/benchmark.command.txt"
    printf '\n' >> "$run_dir/benchmark.command.txt"
    if env -u SGLANG_SIMULATOR_CONFIG_PATH -u SGLANG_SIMULATOR_OUTPUT_DIR -u SGLANG_USE_CPU_ENGINE \
        "$python_bin" -m sglang.benchmark.serving --backend sglang --base-url "http://127.0.0.1:$real_port" \
        --model "$model_path" --dataset-name random-ids --num-prompts "$prompts" \
        --random-input-len "$input_len" --random-output-len "$output_len" --random-range-ratio 0 \
        --request-rate "$rate" --seed "$seed" --warmup-requests 0 --flush-cache --disable-tqdm \
        --ready-check-timeout-sec 0 --output-file "$run_dir/benchmark.jsonl" > "$run_dir/benchmark.log" 2>&1; then
        [[ -s "$run_dir/benchmark.jsonl" ]] && printf 'ok\n' > "$run_dir/status.txt" || printf 'missing-client-output\n' > "$run_dir/status.txt"
    else
        printf 'benchmark-failed\n' > "$run_dir/status.txt"
        return 1
    fi
    grep -qx ok "$run_dir/status.txt"
}

run_simulator_case() {
    local model_path=$1 run_dir=$2 prompts=$3 input_len=$4 output_len=$5 rate=$6 seed=$7 context_length=$8 max_total_tokens=$9
    local -a model_override_args=()
    local -a quantization_args=()
    if jq -e '.dual_chunk_attention_config != null' "$model_path/config.json" > /dev/null; then
        model_override_args=(--json-model-override-args '{"dual_chunk_attention_config": null}')
    fi
    if [[ "$(jq -r '.quantization_config.quant_method // empty' "$model_path/config.json")" == "awq" ]]; then
        quantization_args=(--quantization awq)
    fi
    server_shutdown_grace_seconds=${SGLANG_SIMULATOR_SERVER_SHUTDOWN_GRACE_SECONDS:-2}
    if is_sim_complete "$run_dir"; then return 0; fi
    # A failed server can leave a partial metrics directory.  Preserve it for
    # diagnosis, but do not let it make a later --resume attempt impossible.
    if [[ -e "$run_dir/simulator-output" ]]; then
        local stale_output="$run_dir/superseded-simulator-output-$(date +%Y%m%d_%H%M%S)"
        mv "$run_dir/simulator-output" "$stale_output"
        echo "Archived incomplete simulator output to $stale_output" >&2
    fi
    mkdir -p "$run_dir/simulator-output"
    cleanup_server
    active_server_log="$run_dir/server.log"
    SGLANG_USE_CPU_ENGINE=1 CUDA_VISIBLE_DEVICES='' SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE \
        SGLANG_SIMULATOR_CONFIG_PATH="$sim_config" SGLANG_SIMULATOR_OUTPUT_DIR="$run_dir/simulator-output" \
        setsid "$python_bin" -m sglang_simulator.simulation.sglang.launch_server \
        --model-path "$model_path" --host 127.0.0.1 --port "$simulator_port" --device cpu \
        --context-length "$context_length" --max-running-requests 32 --chunked-prefill-size 2048 \
        --max-total-tokens "$max_total_tokens" \
        "${model_override_args[@]}" \
        "${quantization_args[@]}" \
        --sim-config-path "$sim_config" > "$active_server_log" 2>&1 &
    server_pid=$!
    if ! wait_for_server "$simulator_port"; then
        printf 'server-start-failed\n' > "$run_dir/status.txt"
        cleanup_server
        return 1
    fi
    printf '%q ' "$python_bin" "$sglang_root/benchmark/simulator/bench_serving.py" --simulator-mode offline --backend sglang --base-url "http://127.0.0.1:$simulator_port" --model "$model_path" --dataset-name random-ids --num-prompts "$prompts" --random-input-len "$input_len" --random-output-len "$output_len" --random-range-ratio 0 --request-rate "$rate" --seed "$seed" --flush-cache --disable-tqdm --ready-check-timeout-sec 0 --output-file "$run_dir/benchmark.jsonl" > "$run_dir/benchmark.command.txt"
    printf '\n' >> "$run_dir/benchmark.command.txt"
    if SGLANG_SIMULATOR_OUTPUT_DIR="$run_dir/simulator-output" SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE \
        "$python_bin" "$sglang_root/benchmark/simulator/bench_serving.py" --simulator-mode offline \
        --backend sglang --base-url "http://127.0.0.1:$simulator_port" --model "$model_path" \
        --dataset-name random-ids --num-prompts "$prompts" --random-input-len "$input_len" \
        --random-output-len "$output_len" --random-range-ratio 0 --request-rate "$rate" --seed "$seed" \
        --flush-cache --disable-tqdm --ready-check-timeout-sec 0 --output-file "$run_dir/benchmark.jsonl" \
        > "$run_dir/benchmark.log" 2>&1 && [[ -s "$run_dir/simulator-output/metrics.json" ]] && [[ -s "$run_dir/benchmark.jsonl" ]]; then
        printf 'ok\n' > "$run_dir/status.txt"
    else
        printf 'benchmark-or-metrics-failed\n' > "$run_dir/status.txt"
        cleanup_server
        return 1
    fi
    cleanup_server
}

simulator_metrics_identical() {
    # The predictor is a deterministic logical-time model for a fixed trace.
    # Compare the reported prediction metrics, not collector wall-clock fields
    # such as ``time_cost`` / ``predictor_time_cost``: those measure the host
    # program's own execution and must not force needless repeats.
    local first=$1 second=$2
    "$python_bin" - "$first" "$second" <<'PY'
import json
import math
import sys

first = json.load(open(sys.argv[1], encoding="utf-8"))
second = json.load(open(sys.argv[2], encoding="utf-8"))
keys = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p99_e2e_latency_ms",
)
differences = []
for key in keys:
    left, right = first.get(key), second.get(key)
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        if not (math.isfinite(left) and math.isfinite(right) and left == right):
            differences.append(key)
    elif left != right:
        differences.append(key)
if differences:
    print("different_metrics=" + ",".join(differences))
    raise SystemExit(1)
print("identical_metrics=" + str(len(keys)))
PY
}

run_simulator_repeats() {
    local model=$1 model_path=$2 model_dir=$3 case_name=$4 prompts=$5 input_len=$6 output_len=$7 rate=$8 seed=$9 context_length=${10} max_total_tokens=${11}
    local target_repeats=$repeats repeat run_dir stability_file
    stability_file="$model_dir/simulator/$case_name/prediction-stability.txt"
    mkdir -p "$model_dir/simulator/$case_name"
    if [[ "$prediction_repeat_policy" == two-then-five && "$repeats" -gt 2 ]]; then
        target_repeats=2
    fi
    for repeat in $(seq 1 "$target_repeats"); do
        run_dir="$model_dir/simulator/$case_name/repeat-$repeat"
        echo "[simulator] $model $case_name repeat=$repeat"
        if run_simulator_case "$model_path" "$run_dir" "$prompts" "$input_len" "$output_len" "$rate" "$seed" "$context_length" "$max_total_tokens"; then
            printf '%s,%s,simulator,%s,%s,complete,seed=%s\n' "$(date --iso-8601=seconds)" "$model" "$case_name" "$repeat" "$seed" >> "$results_root/progress.csv"
        else
            printf '%s,%s,simulator,%s,%s,failed,seed=%s\n' "$(date --iso-8601=seconds)" "$model" "$case_name" "$repeat" "$seed" >> "$results_root/progress.csv"
            failures=1
        fi
    done
    if [[ "$prediction_repeat_policy" == two-then-five && "$repeats" -gt 2 ]] \
        && is_sim_complete "$model_dir/simulator/$case_name/repeat-1" \
        && is_sim_complete "$model_dir/simulator/$case_name/repeat-2"; then
        if simulator_metrics_identical \
            "$model_dir/simulator/$case_name/repeat-1/simulator-output/metrics.json" \
            "$model_dir/simulator/$case_name/repeat-2/simulator-output/metrics.json" \
            > "$stability_file"; then
            printf 'policy=two-then-five\nprediction_runs=2\nresult=identical-stop\n' >> "$stability_file"
            return
        fi
        printf 'policy=two-then-five\nprediction_runs=%s\nresult=different-continue\n' "$repeats" >> "$stability_file"
    else
        printf 'policy=%s\nprediction_runs=%s\nresult=insufficient-or-failed-initial-samples\n' \
            "$prediction_repeat_policy" "$repeats" > "$stability_file"
    fi
    for repeat in $(seq $((target_repeats + 1)) "$repeats"); do
        run_dir="$model_dir/simulator/$case_name/repeat-$repeat"
        echo "[simulator] $model $case_name repeat=$repeat"
        if run_simulator_case "$model_path" "$run_dir" "$prompts" "$input_len" "$output_len" "$rate" "$seed" "$context_length" "$max_total_tokens"; then
            printf '%s,%s,simulator,%s,%s,complete,seed=%s\n' "$(date --iso-8601=seconds)" "$model" "$case_name" "$repeat" "$seed" >> "$results_root/progress.csv"
        else
            printf '%s,%s,simulator,%s,%s,failed,seed=%s\n' "$(date --iso-8601=seconds)" "$model" "$case_name" "$repeat" "$seed" >> "$results_root/progress.csv"
            failures=1
        fi
    done
}

if [[ "$resume" != 1 || ! -s "$results_root/progress.csv" ]]; then
    printf 'timestamp,model,phase,case,repeat,status,detail\n' > "$results_root/progress.csv"
fi
IFS=',' read -r -a models <<< "$models_csv"
failures=0
model_index=$model_index_offset
for model in "${models[@]}"; do
    model=${model//[[:space:]]/}
    [[ -n "$model" ]] || continue
    model_path="$checkpoint_root/$model"
    model_dir="$results_root/results/$model"
    if [[ ! -f "$model_path/config.json" ]]; then
        printf '%s,%s,inventory,-,-,failed,missing-config\n' "$(date --iso-8601=seconds)" "$model" >> "$results_root/progress.csv"
        failures=1
        continue
    fi
    mkdir -p "$model_dir/cases"
    cp "$model_path/config.json" "$model_dir/model-config.json"
    context_length=$("$python_bin" - "$model_path/config.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
try:
    print(min(8192, int(data.get("max_position_embeddings", 8192))))
except (TypeError, ValueError):
    print(8192)
PY
)
    checkpoint_bytes=$(du -sb "$model_path" | awk '{print $1}')
    # CUDA Graph capture holds a graph pool in addition to weights and KV
    # cache.  The default leaves 20% outside the static pool; use the explicit
    # launch input for exceptionally large checkpoints and retain it per model.
    mem_fraction=$real_mem_fraction_static
    printf '%s\n' "$context_length" > "$model_dir/context-length.txt"
    printf '%s\n' "$checkpoint_bytes" > "$model_dir/checkpoint-bytes.txt"
    printf '%s\n' "$mem_fraction" > "$model_dir/real-mem-fraction-static.txt"
    case_index=0
    for spec in "${cases[@]}"; do
        IFS='|' read -r case_name prompts input_len output_len rate <<< "$spec"
        case_index=$((case_index + 1))
        printf '{"name":"%s","num_prompts":%s,"input_tokens":%s,"output_tokens":%s,"request_rate":"%s"}\n' "$case_name" "$prompts" "$input_len" "$output_len" "$rate" > "$model_dir/cases/$case_name.json"
    done

    if model_real_complete "$model_dir" && [[ -s "$model_dir/real/max_total_tokens.txt" ]]; then
        read -r real_max_total_tokens < "$model_dir/real/max_total_tokens.txt"
        if [[ ! "$real_max_total_tokens" =~ ^[1-9][0-9]*$ ]]; then
            echo "Invalid recorded real max_total_num_tokens for $model" >&2
            failures=1
            model_index=$((model_index + 1))
            continue
        fi
        echo "[real] $model reusing completed measurements; max_total_num_tokens=$real_max_total_tokens"
    elif start_real_server "$model_path" "$model_dir" "$context_length" "$mem_fraction"; then
        case_index=0
        for spec in "${cases[@]}"; do
            IFS='|' read -r case_name prompts input_len output_len rate <<< "$spec"
            case_index=$((case_index + 1))
            for repeat in $(seq 1 "$repeats"); do
                seed=$((600000 + model_index * 1000 + case_index * 10))
                run_dir="$model_dir/real/$case_name/repeat-$repeat"
                echo "[real] $model $case_name repeat=$repeat"
                if run_real_case "$model_path" "$run_dir" "$prompts" "$input_len" "$output_len" "$rate" "$seed"; then
                    printf '%s,%s,real,%s,%s,complete,seed=%s\n' "$(date --iso-8601=seconds)" "$model" "$case_name" "$repeat" "$seed" >> "$results_root/progress.csv"
                else
                    printf '%s,%s,real,%s,%s,failed,seed=%s\n' "$(date --iso-8601=seconds)" "$model" "$case_name" "$repeat" "$seed" >> "$results_root/progress.csv"
                    failures=1
                fi
            done
        done
    else
        printf '%s,%s,real,-,-,server-start-failed,-\n' "$(date --iso-8601=seconds)" "$model" >> "$results_root/progress.csv"
        failures=1
        cleanup_server
        model_index=$((model_index + 1))
        continue
    fi
    cleanup_server

    case_index=0
    for spec in "${cases[@]}"; do
        IFS='|' read -r case_name prompts input_len output_len rate <<< "$spec"
        case_index=$((case_index + 1))
        seed=$((600000 + model_index * 1000 + case_index * 10))
        run_simulator_repeats "$model" "$model_path" "$model_dir" "$case_name" \
            "$prompts" "$input_len" "$output_len" "$rate" "$seed" "$context_length" "$real_max_total_tokens"
    done
    model_index=$((model_index + 1))
done

summary_args=("$python_bin" "$script_dir/summarize_pr33824_a100_matrix.py" --results-root "$results_root")
if [[ "$prediction_repeat_policy" == two-then-five ]]; then
    summary_args+=(--prediction-aggregation mean-per-case)
fi
"${summary_args[@]}"
"$python_bin" "$script_dir/plot_pr33824_a100_matrix.py" --summary-root "$results_root/analysis"
if [[ "$skip_matrix_render" != 1 ]]; then
    "$python_bin" "$script_dir/render_pr33824_a100_report.py" --results-root "$results_root"
else
    echo "Skipping per-matrix narrative render (SGLANG_SKIP_MATRIX_RENDER=1)"
fi
if [[ "$failures" == 0 ]]; then
    # A successful --resume supersedes an earlier startup/sample failure in
    # the same immutable result root.  Keep exactly one terminal marker so
    # report readers do not mistake an already-recovered run for a failure.
    rm -f "$results_root/MATRIX_COMPLETED_WITH_FAILURES"
    touch "$results_root/MATRIX_COMPLETE"
else
    rm -f "$results_root/MATRIX_COMPLETE"
    touch "$results_root/MATRIX_COMPLETED_WITH_FAILURES"
fi
echo "Result root: $results_root"
exit "$failures"
