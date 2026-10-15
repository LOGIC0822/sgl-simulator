#!/usr/bin/env bash
# Profile the same graph-enabled single-step workload used for the AIC comparison.
#
# This is a diagnostic run, not a replacement for the five-repeat baseline.  It
# uses torch.profiler once per model to expose the CUDA kernels replayed by the
# graph and then compares unambiguous operator families with the AIC breakdown.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
baseline_root=${SGLANG_SINGLE_STEP_BASELINE_ROOT:-$sim_root/test-results/pr33824_aic_vs_real_single_step_dense3_20260908_155211}
results_root=${SGLANG_RESULTS_ROOT:-$sim_root/test-results/pr33824_aic_dtype_op_profile_dense3_$(date +%Y%m%d_%H%M%S)}
models_csv=${SGLANG_BENCH_MODELS:-Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct,Qwen3-32B}
input_len=${SGLANG_SINGLE_STEP_INPUT_LEN:-1024}
output_len=${SGLANG_SINGLE_STEP_OUTPUT_LEN:-2}
context_length=${SGLANG_CONTEXT_LENGTH:-8192}
mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC:-0.80}
attention_backend=${SGLANG_REAL_ATTENTION_BACKEND:-triton}
cuda_graph_config=${SGLANG_REAL_CUDA_GRAPH_CONFIG:-'{"decode":{"backend":"full","max_bs":32},"prefill":{"backend":"tc_piecewise","max_bs":2048,"tc_compiler":"eager"}}'}

usage() {
    cat <<'EOF'
Usage: bash scripts/run_pr33824_aic_dtype_op_profile.sh [options]

Profile graph-enabled A100 single steps for the three dense checkpoints and
compare trace-visible operator families with the completed AIC comparison.

Options:
  --results-root PATH   New, empty result directory.
  --baseline-root PATH  Completed AIC-vs-real single-step result directory.
  --models CSV          Checkpoint names below /mnt/data/models.
  --input-len N         Prefill length (default: 1024).
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --baseline-root) baseline_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --input-len) input_len=${2:?missing value}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$input_len" =~ ^[1-9][0-9]*$ ]] || { echo "--input-len must be positive" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
[[ -f "$baseline_root/finalization.status" ]] || { echo "Missing completed baseline: $baseline_root" >&2; exit 1; }
rg -qx 'status=complete' "$baseline_root/finalization.status" || { echo "Baseline is not complete" >&2; exit 1; }

results_root=$(realpath -m "$results_root")
baseline_root=$(realpath "$baseline_root")
if [[ -e "$results_root" ]]; then
    echo "Refusing to overwrite existing result directory: $results_root" >&2
    exit 1
fi
mkdir -p "$results_root"/{models,hardware,analysis}

weak_ref_library=$baseline_root/environment/weak-ref-compat-build/sglang_weak_ref_compat.so
[[ -f "$weak_ref_library" ]] || { echo "Missing weak-ref compatibility library: $weak_ref_library" >&2; exit 1; }

real_pythonpath="$sim_root/scripts/runtime_shims:$sim_root/scripts/collector_shims:$sglang_root/python:$sim_root/src"

{
    printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
    printf 'source_path=%s\npython=%s\n' "$sglang_root" "$python_bin"
    printf 'baseline_root=%s\nmodels=%s\n' "$baseline_root" "$models_csv"
    printf 'input_len=%s\noutput_len=%s\ncontext_length=%s\nmem_fraction_static=%s\n' "$input_len" "$output_len" "$context_length" "$mem_fraction_static"
    printf 'real_attention_backend=%s\nreal_cuda_graph_config=%s\n' "$attention_backend" "$cuda_graph_config"
    printf 'measurement=torch.profiler CPU+CUDA trace of one benchmark iteration after warmup; graph remains enabled; not a replacement for the five-repeat timing baseline\n'
    printf 'aic_prediction_source=%s\n' "$baseline_root/models/<model>/aic_prediction.json"
    printf 'real_dtype=auto from BF16 checkpoint; kv_cache_dtype=auto from model dtype; no explicit dtype override\n'
    printf 'real_kernel_alignment=triton attention, SGLANG_FORCE_FUSED_OP_BACKEND=torch\n'
} > "$results_root/run_metadata.txt"

nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader > "$results_root/hardware/gpu_identity.csv"
nvidia-smi -q > "$results_root/hardware/nvidia-smi-q.txt"

IFS=',' read -r -a models <<< "$models_csv"
for model in "${models[@]}"; do
    model_path=$checkpoint_root/$model
    model_dir=$results_root/models/$model
    [[ -d "$model_path" ]] || { echo "Missing checkpoint: $model_path" >&2; exit 1; }
    [[ -f "$baseline_root/models/$model/aic_prediction.json" ]] || { echo "Baseline AIC prediction missing for $model" >&2; exit 1; }
    mkdir -p "$model_dir/traces"

    command=(env -u SGLANG_SIMULATOR_CONFIG_PATH -u SGLANG_SIMULATOR_OUTPUT_DIR -u SGLANG_USE_CPU_ENGINE
        "PYTHONPATH=$real_pythonpath" "SGLANG_WEAK_REF_COMPAT_LIBRARY=$weak_ref_library"
        SGLANG_FORCE_FUSED_OP_BACKEND=torch SGLANG_CACHE_DIR=/tmp/sglang-pr33824-piecewise-cache
        "SGLANG_TORCH_PROFILER_DIR=$model_dir/traces"
        "$python_bin" -m sglang.benchmark.one_batch
        --model-path "$model_path" --device cuda --tp-size 1 --context-length "$context_length"
        --mem-fraction-static "$mem_fraction_static" --max-running-requests 32 --chunked-prefill-size 2048
        --attention-backend "$attention_backend" --cuda-graph-config "$cuda_graph_config"
        --batch-size 1 --input-len "$input_len" --output-len "$output_len"
        --run-name "aic_dtype_op_profile_$model" --result-filename "$model_dir/one_batch.jsonl"
        --profile --profile-record-shapes --profile-stage all --profile-start-step 0 --profile-steps 1
        --profile-prefix "aic_dtype_op_profile_$model")
    printf '%q ' "${command[@]}" > "$model_dir/command.txt"
    printf '\n' >> "$model_dir/command.txt"

    "${command[@]}" > "$model_dir/one_batch.log" 2>&1
    rg -q 'Capture target prefill CUDA graph begin.*backend=tc_piecewise' "$model_dir/one_batch.log" || {
        echo "Missing tc_piecewise prefill CUDA-Graph evidence for $model" >&2
        exit 1
    }
    rg -q 'Capture target decode CUDA graph begin.*backend=full' "$model_dir/one_batch.log" || {
        echo "Missing full decode CUDA-Graph evidence for $model" >&2
        exit 1
    }
    compgen -G "$model_dir/traces/*_prefill.trace.json.gz" > /dev/null || {
        echo "Missing prefill trace for $model" >&2
        exit 1
    }
    compgen -G "$model_dir/traces/*_decode.trace.json.gz" > /dev/null || {
        echo "Missing decode trace for $model" >&2
        exit 1
    }
    printf 'ok\n' > "$model_dir/status.txt"
done

PYTHONPATH="$real_pythonpath" SGLANG_WEAK_REF_COMPAT_LIBRARY="$weak_ref_library" \
    "$python_bin" "$script_dir/analyze_pr33824_aic_dtype_op_profile.py" \
    --results-root "$results_root" --baseline-root "$baseline_root" --models "$models_csv" \
    > "$results_root/analysis.log" 2>&1
printf 'status=complete\nmodels=%s\n' "${#models[@]}" > "$results_root/finalization.status"
echo "Completed: $results_root"
