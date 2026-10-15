#!/usr/bin/env bash
# Collect device-synchronized ScheduleBatch rows from a real A100 SGLang server
# and train the upstream-style ML latency candidates from those rows.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
export PATH="$(dirname "$python_bin"):$PATH"
# Keep the experiment in hbf-cxl: user-site Torch can otherwise shadow conda.
export PYTHONNOUSERSITE=1
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
model_name=${SGLANG_COLLECT_MODEL:-Qwen3-0.6B}
model_path=${SGLANG_COLLECT_MODEL_PATH:-"$checkpoint_root/$model_name"}
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:-"$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4"}
results_root=${SGLANG_RESULTS_ROOT:-"$sim_root/test-results/pr33824_data_tools_baseline_${model_name}_$(date +%Y%m%d_%H%M%S)"}
port=${SGLANG_REAL_PORT:-30217}
context_length=${SGLANG_CONTEXT_LENGTH:-2048}
mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC:-0.30}
max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS:-32}
chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE:-512}
seed=${SGLANG_BENCH_SEED:-20260917}
preset=${SGLANG_COLLECTION_PRESET:-smoke}
train_after_collect=${SGLANG_TRAIN_AFTER_COLLECT:-1}
# The upstream trainer defaults to EXTEND (mode 1) only.  The simulator calls
# one predictor for both EXTEND and DECODE batches, so this runner trains both
# modes unless an experiment deliberately narrows the contract.
train_forward_modes=${SGLANG_TRAIN_FORWARD_MODES:-1,2}
trace_converter="$sim_root/scripts/convert_pr33824_baseline_trace.py"
warmup_requests=${SGLANG_WARMUP_REQUESTS:-4}
random_range_ratio=${SGLANG_RANDOM_RANGE_RATIO:-0}
requested_cases=${SGLANG_COLLECTION_CASES:-}
prompt_multiplier=${SGLANG_COLLECTION_PROMPT_MULTIPLIER:-1}
attention_backend=${SGLANG_ATTENTION_BACKEND:-triton}
moe_jit_align_shim=${SGLANG_MOE_JIT_ALIGN_SHIM:-}
runtime_shim_dir=${SGLANG_RUNTIME_SHIM_DIR:-}
fa2_compat_stage=${SGLANG_FA2_COMPAT_STAGE:-}
page_size=${SGLANG_PAGE_SIZE:-}

[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
ninja_path=$(command -v ninja || true)
[[ -n "$ninja_path" ]] || { echo "Missing Ninja binary in experiment PATH" >&2; exit 1; }
"$python_bin" - "$ninja_path" <<'PY'
import sys
from pathlib import Path
with Path(sys.argv[1]).open("rb") as stream:
    if stream.read(4) != b"\x7fELF":
        raise SystemExit(f"Refusing non-ELF Ninja wrapper: {sys.argv[1]}")
PY
[[ -d "$model_path" ]] || { echo "Missing model: $model_path" >&2; exit 1; }
[[ -f "$trace_converter" ]] || { echo "Missing trace converter: $trace_converter" >&2; exit 1; }
[[ "$warmup_requests" =~ ^[0-9]+$ ]] || { echo "SGLANG_WARMUP_REQUESTS must be a non-negative integer" >&2; exit 2; }
[[ "$prompt_multiplier" =~ ^[1-9][0-9]*$ ]] || {
    echo "SGLANG_COLLECTION_PROMPT_MULTIPLIER must be a positive integer" >&2
    exit 2
}
[[ "$random_range_ratio" =~ ^(0|0\.[0-9]+|1|1\.0+)$ ]] || {
    echo "SGLANG_RANDOM_RANGE_RATIO must be in [0, 1]" >&2
    exit 2
}
[[ "$attention_backend" == triton || "$attention_backend" == dual_chunk_flash_attn ]] || {
    echo "SGLANG_ATTENTION_BACKEND must be triton or dual_chunk_flash_attn" >&2
    exit 2
}
[[ -z "$page_size" || "$page_size" =~ ^[1-9][0-9]*$ ]] || {
    echo "SGLANG_PAGE_SIZE must be a positive integer when set" >&2
    exit 2
}
[[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
    echo "Missing SM80 compatibility binary below: $compat_stage" >&2
    exit 1
}
[[ ! -e "$results_root" ]] || { echo "Refusing to overwrite: $results_root" >&2; exit 1; }

case_specs=()
case "$preset" in
    smoke)
        # Two distinct, small cases establish collector/launcher correctness.
        case_specs=(
            'latency_256x16_r1|8|256|16|1'
            'service_512x32_r4|16|512|32|4'
        )
        ;;
    dense)
        # Mirrors the prior dense request shapes but uses non-overlapped,
        # device-synchronized collection required by the baseline collector.
        case_specs=(
            'latency_256x128_r1|16|256|128|1'
            'service_1024x256_r4|32|1024|256|4'
            'saturation_1024x256_inf|32|1024|256|inf'
        )
        ;;
    expanded)
        case_specs=(
            'latency_256x128_r1|16|256|128|1'
            'service_1024x256_r4|32|1024|256|4'
            'saturation_1024x256_inf|32|1024|256|inf'
            'latency_128x64_r1|16|128|64|1'
            'saturation_1536x384_inf|32|1536|384|inf'
        )
        ;;
    *)
        echo "Unknown SGLANG_COLLECTION_PRESET=$preset (use smoke or dense)" >&2
        exit 2
        ;;
esac

if [[ -n "$requested_cases" ]]; then
    IFS=',' read -r -a requested_case_names <<< "$requested_cases"
    selected_case_specs=()
    for requested_case_name in "${requested_case_names[@]}"; do
        found=0
        for case_spec in "${case_specs[@]}"; do
            IFS='|' read -r case_name _ <<< "$case_spec"
            if [[ "$case_name" == "$requested_case_name" ]]; then
                selected_case_specs+=("$case_spec")
                found=1
                break
            fi
        done
        if [[ "$found" != 1 ]]; then
            echo "Requested case is not defined by preset $preset: $requested_case_name" >&2
            exit 2
        fi
    done
    case_specs=("${selected_case_specs[@]}")
fi

mkdir -p "$results_root/hardware" "$results_root/caches"
compat_stage=$(realpath "$compat_stage")
model_path=$(realpath "$model_path")

torch_lib_dir=$("$python_bin" - <<'PY'
import os
import torch

print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
pythonpath_prefix="$compat_stage"
fa2_compat_enabled=0
if [[ -n "$fa2_compat_stage" ]]; then
    fa2_compat_stage=$(realpath "$fa2_compat_stage")
    [[ -f "$fa2_compat_stage/flash_attn/__init__.py" ]] || {
        echo "Missing local FlashAttention-2 package: $fa2_compat_stage" >&2
        exit 1
    }
    pythonpath_prefix="$fa2_compat_stage:$pythonpath_prefix"
    fa2_compat_enabled=1
    export SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN=1
else
    unset SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN
fi

moe_jit_align_enabled=0
if [[ -n "$moe_jit_align_shim" ]]; then
    moe_jit_align_shim=$(realpath "$moe_jit_align_shim")
    [[ -f "$moe_jit_align_shim/sitecustomize.py" ]] || {
        echo "Missing MoE JIT runtime shim sitecustomize.py: $moe_jit_align_shim" >&2
        exit 1
    }
    runtime_shim_dir=${runtime_shim_dir:-$moe_jit_align_shim}
    moe_jit_align_enabled=1
    export SGLANG_SIMULATOR_ENABLE_JIT_MOE_ALIGN=1
else
    unset SGLANG_SIMULATOR_ENABLE_JIT_MOE_ALIGN
fi
if [[ -n "$runtime_shim_dir" ]]; then
    runtime_shim_dir=$(realpath "$runtime_shim_dir")
    [[ -f "$runtime_shim_dir/sitecustomize.py" ]] || {
        echo "Missing runtime shim sitecustomize.py: $runtime_shim_dir" >&2
        exit 1
    }
    # The compiled SM80 package remains ahead of the shim.  The shim is here
    # solely for Python's startup hook and never shadows ``sgl_kernel``.
    pythonpath_prefix="$pythonpath_prefix:$runtime_shim_dir"
fi
export PYTHONPATH="$pythonpath_prefix:$sglang_root/python:$sim_root/src${SGLANG_EXPERIMENT_EXTRA_PYTHONPATH:+:$SGLANG_EXPERIMENT_EXTRA_PYTHONPATH}"
export LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.2}
unset SGLANG_FORCE_FUSED_OP_BACKEND
unset SGLANG_WEAK_REF_COMPAT_LIBRARY

server_page_size_args=()
if [[ -n "$page_size" ]]; then
    server_page_size_args=(--page-size "$page_size")
fi

server_pid=

cleanup_server() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        "$python_bin" "$sim_root/tools/terminate_pr33824_child_tree.py" --pid "$server_pid" || true
        # Keep the server as this script's direct child.  The SGLang parent
        # performs its own scheduler-child cleanup after SIGTERM; killing a
        # process group here can otherwise target the runner's own group when
        # setsid has forked a wrapper process.
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
    for _ in $(seq 1 240); do
        if curl --silent --fail --max-time 2 "http://127.0.0.1:$port/v1/models" > "$results_root/models.json"; then
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server exited before readiness; inspect $log_path" >&2
            tail -120 "$log_path" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "Timed out waiting for the collector server; inspect $log_path" >&2
    return 1
}

{
    printf 'created_at=%s\n' "$(date -Is)"
    printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
    printf 'source_path=%s\npython=%s\n' "$sglang_root" "$python_bin"
    printf 'ninja_path=%s\nninja_version=%s\n' "$ninja_path" "$("$ninja_path" --version)"
    printf 'model=%s\nmodel_path=%s\n' "$model_name" "$model_path"
    printf 'collector_source=https://github.com/littlefatfat/sglang/tree/simulator-data-tools-0914\n'
    printf 'collector_commit=9cfddcedc9d1e00885c28db75a45513eddbb6271\n'
    printf 'compat_stage=%s\n' "$compat_stage"
    printf 'kernel_policy=isolated_SM80_AOT; SGLANG_FORCE_FUSED_OP_BACKEND=unset\n'
    printf 'attention_backend=%s\n' "$attention_backend"
    if [[ "$moe_jit_align_enabled" == 1 ]]; then
        printf 'moe_align_policy=PR33824_JIT_source_shim; shim_path=%s\n' "$moe_jit_align_shim"
        printf 'moe_sum_reduce_policy=TP1_EP1_torch_sum_fallback_in_same_shim\n'
    else
        printf 'moe_align_policy=isolated_SM80_AOT_default\n'
    fi
    if [[ "$fa2_compat_enabled" == 1 ]]; then
        printf 'flash_attention_policy=local_FA2_fallback_for_dual_chunk; stage_path=%s\n' "$fa2_compat_stage"
        printf 'runtime_shim_path=%s\n' "$runtime_shim_dir"
    else
        printf 'flash_attention_policy=default_current_PR_path\n'
    fi
    printf 'disable_overlap_schedule=1\n'
    printf 'cuda_graph=disabled\n'
    printf 'context_length=%s\nmem_fraction_static=%s\n' "$context_length" "$mem_fraction_static"
    printf 'max_running_requests=%s\nchunked_prefill_size=%s\npage_size=%s\n' \
        "$max_running_requests" "$chunked_prefill_size" "${page_size:-auto}"
    printf 'preset=%s\nseed=%s\n' "$preset" "$seed"
    printf 'train_forward_modes=%s\n' "$train_forward_modes"
    printf 'warmup_requests=%s\nrandom_range_ratio=%s\n' "$warmup_requests" "$random_range_ratio"
    printf 'requested_cases=%s\nprompt_multiplier=%s\n' "$requested_cases" "$prompt_multiplier"
    printf 'prompt_input_policy=saturation_1536x384_inf: direct IDs via --tokenize-prompt; other cases: decoded text\n'
} > "$results_root/run_metadata.txt"

"$python_bin" - <<'PY' > "$results_root/package_versions.txt"
import importlib.metadata as md
import os
import sys
import torch
import sgl_kernel

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"sgl_kernel_file={sgl_kernel.__file__}")
print(f"sgl_kernel_runtime_version={sgl_kernel.__version__}")
print(f"torch_file={torch.__file__}")
print(f"PYTHONNOUSERSITE={os.environ.get('PYTHONNOUSERSITE', '0')}")
print(f"experiment_dependency_fallback={os.environ.get('SGLANG_EXPERIMENT_EXTRA_PYTHONPATH', 'none')}")
for operation in ("rmsnorm", "fused_add_rmsnorm", "silu_and_mul", "rotary_embedding", "weak_ref_tensor"):
    print(f"torch.ops.sgl_kernel.{operation}={hasattr(torch.ops.sgl_kernel, operation)}")
print(f"SGLANG_SIMULATOR_ENABLE_JIT_MOE_ALIGN={os.environ.get('SGLANG_SIMULATOR_ENABLE_JIT_MOE_ALIGN', '0')}")
print(f"SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN={os.environ.get('SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN', '0')}")
try:
    import flash_attn
    print(f"flash_attn_file={flash_attn.__file__}")
except ImportError:
    print("flash_attn_file=not-installed")
for operation in ("moe_align_block_size",):
    print(f"torch.ops.sgl_kernel.{operation}={hasattr(torch.ops.sgl_kernel, operation)}")
for name in ("sglang", "sgl-kernel", "sglang-kernel", "scikit-learn", "joblib", "transformers", "flashinfer-python", "triton", "numpy", "psutil", "httpx", "python-docx"):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=not-installed")
PY
nvidia-smi -q > "$results_root/hardware/nvidia-smi-before.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,memory.used,driver_version --format=csv,noheader \
    > "$results_root/hardware/gpu_identity.csv" 2>&1 || true
printf 'case\tprompts\tinput_len\toutput_len\trequest_rate\n' > "$results_root/case_manifest.tsv"

for case_spec in "${case_specs[@]}"; do
    IFS='|' read -r case_name prompts input_len output_len request_rate <<< "$case_spec"
    prompts=$((prompts * prompt_multiplier))
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" "$prompts" "$input_len" "$output_len" "$request_rate" \
        >> "$results_root/case_manifest.tsv"
    case_root="$results_root/cases/$case_name"
    case_input_args=()
    if [[ "$case_name" == saturation_1536x384_inf ]]; then
        # A decode/re-encode round trip can exceed the context bound even
        # when the nominal 1536+384 budget fits. Preserve actual ID bounds.
        case_input_args=(--tokenize-prompt)
    fi
    mkdir -p "$(dirname "$case_root")"
    [[ ! -e "$case_root" ]] || { echo "Unexpected existing case directory: $case_root" >&2; exit 1; }

    case_cache="$results_root/caches/$case_name"
    case_jit_cache=${SGLANG_BASELINE_SHARED_JIT_CACHE:-"$case_cache/jit-cache"}
    mkdir -p "$case_cache" "$case_cache/flashinfer-workspace"
    server_log="$case_root.server.log"
    if [[ "${SGLANG_GPU_CAPACITY_WAIT:-0}" == 1 ]]; then
        capacity_args=()
        if [[ -n "${SGLANG_GPU_MINIMUM_FREE_MIB:-}" ]]; then
            capacity_args+=(--minimum-free-mib "$SGLANG_GPU_MINIMUM_FREE_MIB")
        fi
        case "$model_name" in
            Qwen3-0.6B|Qwen2.5-3B-Instruct|llama-2-7b|Mistral-7B-Instruct-v0.2|Qwen2.5-7B-Instruct|Qwen3-8B|Qwen2.5-14B-Instruct|Qwen2.5-14B-Instruct-1M|Qwen3-14B|Meta-Llama-3.1-70B-Instruct-AWQ|Qwen3-30B-A3B|Qwen3-30B-A3B-Instruct-2507|Qwen3-32B)
                capacity_args+=(--model "$model_name") ;;
        esac
        "$python_bin" "$sim_root/tools/wait_pr33824_gpu_capacity.py" \
            --fraction "$mem_fraction_static" \
            --interval "${SGLANG_GPU_CAPACITY_CHECK_SECONDS:-600}" \
            "${capacity_args[@]}" \
            --log "$results_root/hardware/gpu_capacity.jsonl"
    fi
    printf '%q ' env -u SGLANG_SIMULATOR_CONFIG_PATH -u SGLANG_SIMULATOR_OUTPUT_DIR -u SGLANG_USE_CPU_ENGINE \
        SGLANG_CACHE_DIR="$case_cache/cache" SGLANG_JIT_CACHE_DIR="$case_jit_cache" \
        FLASHINFER_WORKSPACE_BASE="$case_cache/flashinfer-workspace" \
        PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$python_bin" -m sglang_simulator.baseline.launch_server \
        --collection-output-dir "$case_root" --model-path "$model_path" --host 127.0.0.1 --port "$port" \
        --device cuda --tp-size 1 --context-length "$context_length" --mem-fraction-static "$mem_fraction_static" \
        --max-running-requests "$max_running_requests" --chunked-prefill-size "$chunked_prefill_size" \
        "${server_page_size_args[@]}" \
        --attention-backend "$attention_backend" --disable-cuda-graph --disable-overlap-schedule \
        > "$case_root.server.command.txt"
    printf '\n' >> "$case_root.server.command.txt"

    env -u SGLANG_SIMULATOR_CONFIG_PATH -u SGLANG_SIMULATOR_OUTPUT_DIR -u SGLANG_USE_CPU_ENGINE \
        SGLANG_CACHE_DIR="$case_cache/cache" SGLANG_JIT_CACHE_DIR="$case_jit_cache" \
        FLASHINFER_WORKSPACE_BASE="$case_cache/flashinfer-workspace" \
        PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$python_bin" -m sglang_simulator.baseline.launch_server \
        --collection-output-dir "$case_root" --model-path "$model_path" --host 127.0.0.1 --port "$port" \
        --device cuda --tp-size 1 --context-length "$context_length" --mem-fraction-static "$mem_fraction_static" \
        --max-running-requests "$max_running_requests" --chunked-prefill-size "$chunked_prefill_size" \
        "${server_page_size_args[@]}" \
        --attention-backend "$attention_backend" --disable-cuda-graph --disable-overlap-schedule \
        > "$server_log" 2>&1 &
    server_pid=$!
    wait_for_server "$server_log"

    printf '%q ' "$python_bin" -m sglang.benchmark.serving --backend sglang \
        --base-url "http://127.0.0.1:$port" --model "$model_path" --dataset-name random-ids \
        --num-prompts "$prompts" --random-input-len "$input_len" --random-output-len "$output_len" \
        --random-range-ratio "$random_range_ratio" --request-rate "$request_rate" --seed "$seed" --warmup-requests "$warmup_requests" \
        "${case_input_args[@]}" \
        --flush-cache --disable-tqdm --ready-check-timeout-sec 0 --profile \
        --output-file "$case_root/benchmark.jsonl" > "$case_root.benchmark.command.txt"
    printf '\n' >> "$case_root.benchmark.command.txt"
    "$python_bin" -m sglang.benchmark.serving --backend sglang \
        --base-url "http://127.0.0.1:$port" --model "$model_path" --dataset-name random-ids \
        --num-prompts "$prompts" --random-input-len "$input_len" --random-output-len "$output_len" \
        --random-range-ratio "$random_range_ratio" --request-rate "$request_rate" --seed "$seed" --warmup-requests "$warmup_requests" \
        "${case_input_args[@]}" \
        --flush-cache --disable-tqdm --ready-check-timeout-sec 0 --profile \
        --output-file "$case_root/benchmark.jsonl" > "$case_root.benchmark.log" 2>&1

    [[ -s "$case_root/TP0.schedule_batch.jsonl" ]] || {
        echo "Collector produced no schedule batches: $case_root" >&2
        exit 1
    }
    [[ -s "$case_root/TP0.raw_request.jsonl" ]] || {
        echo "Collector produced no raw request trace: $case_root" >&2
        exit 1
    }
    [[ -s "$case_root/TP0.manifest.json" ]] || {
        echo "Collector produced no manifest: $case_root" >&2
        exit 1
    }
    "$python_bin" "$trace_converter" \
        --raw-request-jsonl "$case_root/TP0.raw_request.jsonl" \
        --output "$case_root/autobench_trace.jsonl" \
        --manifest-output "$case_root/autobench_trace.manifest.json" \
        > "$case_root/trace_conversion.log" 2>&1
    [[ -s "$case_root/autobench_trace.jsonl" ]] || {
        echo "Trace conversion produced no Autobench replay trace: $case_root" >&2
        exit 1
    }
    "$python_bin" - "$case_root" "$prompts" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected = int(sys.argv[2])
manifest = json.loads((root / "autobench_trace.manifest.json").read_text())
benchmark = json.loads((root / "benchmark.jsonl").read_text().splitlines()[-1])
if manifest["request_count"] != expected or int(benchmark["completed"]) != expected:
    raise RuntimeError(f"request count mismatch: expected={expected}, trace={manifest['request_count']}, completed={benchmark['completed']}")
PY
    printf 'ok\n' > "$case_root/status.txt"
    cleanup_server
done

if [[ "$train_after_collect" == 1 ]]; then
    training_root="$results_root/training"
    "$python_bin" "$sim_root/tools/train_latency_model.py" \
        --data-root "$results_root/cases" --out-dir "$training_root" \
        --model-prefix "${model_name//[^A-Za-z0-9._-]/_}_hgbmono" --loss both \
        --forward-modes "$train_forward_modes" \
        > "$results_root/train.log" 2>&1
    compgen -G "$training_root/*_l2.pkl" > /dev/null || {
        echo "Missing L2 model bundle" >&2
        exit 1
    }
    compgen -G "$training_root/*_p50.pkl" > /dev/null || {
        echo "Missing p50 model bundle" >&2
        exit 1
    }
fi

nvidia-smi -q > "$results_root/hardware/nvidia-smi-after.txt" 2>&1 || true
printf 'complete\n' > "$results_root/BASELINE_COLLECTION_COMPLETE"
echo "Baseline collection complete: $results_root"
