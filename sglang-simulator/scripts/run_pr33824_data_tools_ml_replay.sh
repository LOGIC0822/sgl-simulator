#!/usr/bin/env bash
# Replay collected raw requests through the CPU simulator using an ML bundle.
# The real collection and this replay deliberately share scheduler settings;
# only physical GPU execution is replaced by ML-predicted logical time.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
collection_root=${SGLANG_COLLECTION_ROOT:?Set SGLANG_COLLECTION_ROOT to a completed baseline collection root}
collection_root=$(realpath "$collection_root")
bundle=${SGLANG_ML_BUNDLE:-}
replay_sim_config=${SGLANG_REPLAY_SIM_CONFIG:-}
completion_marker=${SGLANG_REPLAY_COMPLETION_MARKER:-ML_REPLAY_COMPLETE}
predictor_label=${SGLANG_REPLAY_PREDICTOR_LABEL:-}
metadata_path="$collection_root/run_metadata.txt"
if [[ ${SGLANG_KERNEL_COMPAT_STAGE+x} ]]; then
    compat_stage=$SGLANG_KERNEL_COMPAT_STAGE
else
    compat_stage="$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4"
fi
port=${SGLANG_SIMULATOR_PORT:-30218}
requested_cases=${SGLANG_REPLAY_CASES:-}
results_root=${SGLANG_REPLAY_RESULTS_ROOT:-"$sim_root/test-results/pr33824_data_tools_ml_replay_$(basename "$collection_root")_$(date +%Y%m%d_%H%M%S)"}
cache_root=${SGLANG_REPLAY_CACHE_ROOT:-"$results_root/caches"}
config_writer="$sim_root/tools/write_pr33824_ml_replay_config.py"
aic_source=${AICONFIGURATOR_SOURCE_ROOT:-/home/ms-admin/workspace1/yiheng/ali/aiconfigurator-v0.10.0}

[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
[[ -f "$metadata_path" ]] || { echo "Missing collection metadata: $metadata_path" >&2; exit 1; }
[[ -z "$bundle" || -z "$replay_sim_config" ]] || {
    echo "Set exactly one of SGLANG_ML_BUNDLE or SGLANG_REPLAY_SIM_CONFIG" >&2; exit 2;
}
[[ -n "$bundle" || -n "$replay_sim_config" ]] || {
    echo "Set SGLANG_ML_BUNDLE or SGLANG_REPLAY_SIM_CONFIG" >&2; exit 2;
}
[[ "$completion_marker" =~ ^[A-Za-z0-9_]+$ ]] || {
    echo "SGLANG_REPLAY_COMPLETION_MARKER must contain only letters, digits, and underscores" >&2; exit 2;
}
if [[ -n "$bundle" ]]; then
    bundle=$(realpath "$bundle")
    [[ -f "$bundle" ]] || { echo "Missing ML bundle: $bundle" >&2; exit 1; }
    [[ -f "$config_writer" ]] || { echo "Missing config writer: $config_writer" >&2; exit 1; }
    predictor_label=${predictor_label:-ml_hgb}
else
    replay_sim_config=$(realpath "$replay_sim_config")
    [[ -f "$replay_sim_config" ]] || { echo "Missing replay simulator config: $replay_sim_config" >&2; exit 1; }
    [[ -d "$aic_source" ]] || { echo "Missing AIC source required by config replay: $aic_source" >&2; exit 1; }
    predictor_label=${predictor_label:-configured_predictor}
fi
if [[ -n "$compat_stage" ]]; then
    [[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
        echo "Missing SM80 compatibility binary below: $compat_stage" >&2
        exit 1
    }
    compat_stage=$(realpath "$compat_stage")
fi
[[ ! -e "$results_root" ]] || { echo "Refusing to overwrite: $results_root" >&2; exit 1; }

metadata_value() {
    local key=$1
    sed -n "s/^${key}=//p" "$metadata_path" | tail -n 1
}

model_path=${SGLANG_REPLAY_MODEL_PATH:-$(metadata_value model_path)}
context_length=${SGLANG_REPLAY_CONTEXT_LENGTH:-$(metadata_value context_length)}
max_running_requests=${SGLANG_REPLAY_MAX_RUNNING_REQUESTS:-$(metadata_value max_running_requests)}
chunked_prefill_size=${SGLANG_REPLAY_CHUNKED_PREFILL_SIZE:-$(metadata_value chunked_prefill_size)}
attention_backend=${SGLANG_REPLAY_ATTENTION_BACKEND:-$(metadata_value attention_backend)}
page_size=${SGLANG_REPLAY_PAGE_SIZE:-$(metadata_value page_size)}
[[ -d "$model_path" ]] || { echo "Missing replay model: $model_path" >&2; exit 1; }
[[ "$context_length" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid context_length: $context_length" >&2; exit 1; }
[[ "$max_running_requests" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid max_running_requests: $max_running_requests" >&2; exit 1; }
[[ "$chunked_prefill_size" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid chunked_prefill_size: $chunked_prefill_size" >&2; exit 1; }
[[ "$attention_backend" == triton || "$attention_backend" == dual_chunk_flash_attn ]] || {
    echo "Invalid attention_backend from collection metadata: $attention_backend" >&2
    exit 1
}
[[ -z "$page_size" || "$page_size" == auto || "$page_size" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid page_size from collection metadata: $page_size" >&2
    exit 1
}

cache_root=$(realpath -m "$cache_root")
mkdir -p "$results_root/cases" "$cache_root/sglang" "$cache_root/jit" \
    "$cache_root/flashinfer-workspace"
model_path=$(realpath "$model_path")
torch_lib_dir=$("$python_bin" - <<'PY'
import os
import torch

print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
server_pythonpath="$sim_root/src:$sglang_root/python${SGLANG_EXPERIMENT_EXTRA_PYTHONPATH:+:$SGLANG_EXPERIMENT_EXTRA_PYTHONPATH}"
client_pythonpath="$sglang_root/python:$sim_root/src${SGLANG_EXPERIMENT_EXTRA_PYTHONPATH:+:$SGLANG_EXPERIMENT_EXTRA_PYTHONPATH}"
[[ -z "$compat_stage" ]] || client_pythonpath="$compat_stage:$client_pythonpath"
if [[ -n "$replay_sim_config" ]]; then
    # The CPU logical simulator imports the AIC predictor at server start;
    # keep the checked-out AIC source explicit rather than relying on a global
    # editable install. The client still replays only trace timestamps.
    server_pythonpath="$aic_source/src:$aic_source:$server_pythonpath"
    client_pythonpath="$aic_source/src:$aic_source:$client_pythonpath"
fi
server_pid=
# The ordinary CPU simulator uses torch-native attention.  The 1M YaRN
# checkpoint declares dual-chunk attention, whose CUDA kernels cannot run in
# this CPU-only logical simulator.  Keep its physical page size but opt in to
# the simulator's narrow validation bypass; forward time still comes from the
# trained ML bundle rather than from any CPU attention implementation.
server_page_size_args=()
server_cpu_dual_chunk_env=()
if [[ "$attention_backend" == dual_chunk_flash_attn ]]; then
    server_cpu_dual_chunk_env=(
        "SGLANG_SIMULATOR_ALLOW_CPU_DUAL_CHUNK=1"
    )
    if [[ "$page_size" != auto && -n "$page_size" ]]; then
        server_page_size_args+=(--page-size "$page_size")
    fi
fi

cleanup_server() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        "$python_bin" "$sim_root/tools/terminate_pr33824_child_tree.py" --pid "$server_pid" || true
        kill "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=
}
trap cleanup_server EXIT INT TERM

wait_for_server() {
    local log_path=$1
    for _ in $(seq 1 120); do
        if curl --silent --fail --max-time 2 "http://127.0.0.1:$port/v1/models" > "$results_root/models.json"; then
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Simulator server exited before readiness; inspect $log_path" >&2
            tail -120 "$log_path" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "Timed out waiting for simulator server; inspect $log_path" >&2
    return 1
}

case_paths=()
if [[ -n "$requested_cases" ]]; then
    IFS=',' read -r -a requested_case_names <<< "$requested_cases"
    for case_name in "${requested_case_names[@]}"; do
        case_paths+=("$collection_root/cases/$case_name")
    done
else
    for case_path in "$collection_root"/cases/*; do
        [[ -d "$case_path" ]] && case_paths+=("$case_path")
    done
fi
(( ${#case_paths[@]} > 0 )) || { echo "No collection cases found" >&2; exit 1; }

if [[ -n "$bundle" ]]; then
    "$python_bin" "$config_writer" --output "$results_root/simulator-config.json" --bundle "$bundle" \
        > "$results_root/config_writer.log" 2>&1
    sim_config="$results_root/simulator-config.json"
else
    sim_config="$replay_sim_config"
fi

{
    printf 'created_at=%s\n' "$(date -Is)"
    printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
    printf 'collection_root=%s\nmodel_path=%s\npredictor_label=%s\n' \
        "$collection_root" "$model_path" "$predictor_label"
    if [[ -n "$bundle" ]]; then
        printf 'ml_bundle=%s\n' "$bundle"
    else
        printf 'replay_sim_config_source=%s\n' "$replay_sim_config"
    fi
    printf 'compat_stage=%s\n' "$compat_stage"
    printf 'cache_root=%s\n' "$cache_root"
    printf 'simulator_config=%s\ncompletion_marker=%s\n' "$sim_config" "$completion_marker"
    printf 'context_length=%s\nmax_running_requests=%s\nchunked_prefill_size=%s\n' \
        "$context_length" "$max_running_requests" "$chunked_prefill_size"
    printf 'attention_backend=%s\npage_size=%s\n' "$attention_backend" "${page_size:-auto}"
    printf 'scheduler_overlap=disabled\ncuda_graph=disabled\n'
    printf 'arrival_source=converted_raw_request_created_time\n'
} > "$results_root/run_metadata.txt"
if [[ -n "$bundle" ]]; then
    sha256sum "$bundle" "$sim_config" > "$results_root/artifact_sha256.txt"
else
    sha256sum "$sim_config" > "$results_root/artifact_sha256.txt"
fi

for source_case in "${case_paths[@]}"; do
    [[ -d "$source_case" ]] || { echo "Missing requested case: $source_case" >&2; exit 1; }
    case_name=$(basename "$source_case")
    trace_path="$source_case/autobench_trace.jsonl"
    [[ -s "$trace_path" ]] || { echo "Missing converted trace: $trace_path" >&2; exit 1; }
    max_total_tokens=${SGLANG_REPLAY_MAX_TOTAL_TOKENS:-}
    if [[ -z "$max_total_tokens" ]]; then
        max_total_tokens=$(rg -o 'max_total_num_tokens=[0-9]+' "$source_case.server.log" 2>/dev/null | tail -n 1 | cut -d= -f2 || true)
    fi
    [[ "$max_total_tokens" =~ ^[1-9][0-9]*$ ]] || {
        echo "Could not determine max_total_tokens for $case_name; set SGLANG_REPLAY_MAX_TOTAL_TOKENS" >&2
        exit 1
    }
    request_count=$(wc -l < "$trace_path")
    [[ "$request_count" =~ ^[1-9][0-9]*$ ]] || { echo "Empty trace: $trace_path" >&2; exit 1; }

    case_root="$results_root/cases/$case_name"
    mkdir -p "$case_root/simulator-output"
    server_log="$case_root/server.log"
    printf '%q ' env -u SGLANG_SIMULATOR_ATTENTION_BACKEND \
        SGLANG_USE_CPU_ENGINE=1 CUDA_VISIBLE_DEVICES='' \
        "${server_cpu_dual_chunk_env[@]}" \
        SGLANG_CACHE_DIR="$cache_root/sglang" SGLANG_JIT_CACHE_DIR="$cache_root/jit" \
        FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer-workspace" \
        SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE SGLANG_SIMULATOR_CONFIG_PATH="$sim_config" \
        SGLANG_SIMULATOR_OUTPUT_DIR="$case_root/simulator-output" PYTHONPATH="$server_pythonpath" \
        "$python_bin" -m sglang_simulator.simulation.sglang.launch_server \
        --model-path "$model_path" --host 127.0.0.1 --port "$port" --device cpu \
        --context-length "$context_length" --max-running-requests "$max_running_requests" \
        --chunked-prefill-size "$chunked_prefill_size" --max-total-tokens "$max_total_tokens" \
        "${server_page_size_args[@]}" \
        --disable-cuda-graph --disable-overlap-schedule --sim-config-path "$sim_config" \
        > "$case_root/server.command.txt"
    printf '\n' >> "$case_root/server.command.txt"

    env -u SGLANG_SIMULATOR_ATTENTION_BACKEND \
        SGLANG_USE_CPU_ENGINE=1 CUDA_VISIBLE_DEVICES='' \
        "${server_cpu_dual_chunk_env[@]}" \
        SGLANG_CACHE_DIR="$cache_root/sglang" SGLANG_JIT_CACHE_DIR="$cache_root/jit" \
        FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer-workspace" \
        SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE SGLANG_SIMULATOR_CONFIG_PATH="$sim_config" \
        SGLANG_SIMULATOR_OUTPUT_DIR="$case_root/simulator-output" PYTHONPATH="$server_pythonpath" \
        "$python_bin" -m sglang_simulator.simulation.sglang.launch_server \
        --model-path "$model_path" --host 127.0.0.1 --port "$port" --device cpu \
        --context-length "$context_length" --max-running-requests "$max_running_requests" \
        --chunked-prefill-size "$chunked_prefill_size" --max-total-tokens "$max_total_tokens" \
        "${server_page_size_args[@]}" \
        --disable-cuda-graph --disable-overlap-schedule --sim-config-path "$sim_config" \
        > "$server_log" 2>&1 &
    server_pid=$!
    wait_for_server "$server_log"

    printf '%q ' env SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE \
        SGLANG_CACHE_DIR="$cache_root/sglang" SGLANG_JIT_CACHE_DIR="$cache_root/jit" \
        FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer-workspace" \
        SGLANG_SIMULATOR_OUTPUT_DIR="$case_root/simulator-output" \
        PYTHONPATH="$client_pythonpath" LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$python_bin" "$sglang_root/benchmark/simulator/bench_serving.py" \
        --simulator-mode offline --backend sglang --base-url "http://127.0.0.1:$port" \
        --model "$model_path" --dataset-name autobench --dataset-path "$trace_path" \
        --num-prompts "$request_count" --use-trace-timestamps --flush-cache --disable-tqdm \
        --ready-check-timeout-sec 0 --output-file "$case_root/benchmark.jsonl" \
        > "$case_root/benchmark.command.txt"
    printf '\n' >> "$case_root/benchmark.command.txt"
    SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE SGLANG_SIMULATOR_OUTPUT_DIR="$case_root/simulator-output" \
        SGLANG_CACHE_DIR="$cache_root/sglang" SGLANG_JIT_CACHE_DIR="$cache_root/jit" \
        FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer-workspace" \
        PYTHONPATH="$client_pythonpath" LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$python_bin" "$sglang_root/benchmark/simulator/bench_serving.py" \
        --simulator-mode offline --backend sglang --base-url "http://127.0.0.1:$port" \
        --model "$model_path" --dataset-name autobench --dataset-path "$trace_path" \
        --num-prompts "$request_count" --use-trace-timestamps --flush-cache --disable-tqdm \
        --ready-check-timeout-sec 0 --output-file "$case_root/benchmark.jsonl" \
        > "$case_root/benchmark.log" 2>&1
    [[ -s "$case_root/simulator-output/metrics.json" ]] || {
        echo "Missing simulator metrics: $case_root" >&2
        exit 1
    }
    [[ -s "$case_root/benchmark.jsonl" ]] || { echo "Missing benchmark output: $case_root" >&2; exit 1; }
    "$python_bin" "$sim_root/tools/validate_pr33824_replay_result.py" \
        --real-case "$source_case" --replay-case "$case_root" \
        > "$case_root/replay_validation.log" 2>&1
    printf 'ok\n' > "$case_root/status.txt"
    cleanup_server
done

printf 'complete\n' > "$results_root/$completion_marker"
echo "${predictor_label} replay complete: $results_root"
