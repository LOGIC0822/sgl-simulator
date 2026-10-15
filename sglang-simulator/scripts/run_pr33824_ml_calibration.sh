#!/usr/bin/env bash
# Collect native SGLang batch timings and train one MLTimePredictor bundle per
# dense checkpoint.  This script deliberately does not import or query AIC.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:-$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4}
results_root=${SGLANG_ML_CALIBRATION_ROOT:-$sim_root/test-results/pr33824_ml_native_calibration_dense3_$(date +%Y%m%d_%H%M%S)}
models_csv=${SGLANG_ML_MODELS:-Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct,Qwen3-32B}
poll_seconds=${SGLANG_ML_GPU_POLL_SECONDS:-600}
max_gpu_util=${SGLANG_ML_MAX_GPU_UTIL_PERCENT:-15}
mem_fraction_32b=${SGLANG_ML_32B_MEM_FRACTION_STATIC:-0.96}
resume=0

# These grids are intentionally independent from the serving benchmark trace.
# Training includes practical A100 sequence/batch ranges; the interleaved
# holdout grid is measured separately and never passed to estimator.fit().
train_batch_sizes=(1 2 4 8 16 32)
train_input_lens=(64 128 256 512 1024)
train_output_lens=(32 128 256)
holdout_batch_sizes=(3 6 12 24)
holdout_input_lens=(192 384 768 1280)
holdout_output_lens=(128)
cuda_graph_config='{"decode":{"backend":"full","max_bs":32},"prefill":{"backend":"tc_piecewise","max_bs":2048,"tc_compiler":"eager"}}'

usage() {
    cat <<'EOF'
Usage: bash tools/sglang-simulator/scripts/run_pr33824_ml_calibration.sh [options]

Collects native CUDA one_batch prefill/decode timings for 3B, 7B, and 32B
dense checkpoints, then writes per-model sklearn/joblib ML predictor bundles.
The holdout grid is measured but excluded from fitting.  No AIC database,
collector, or predictor participates in this run.

Options:
  --results-root PATH  New output root (default: timestamped test-results root)
  --models CSV         Checkpoint directory names below /mnt/data/models
  --poll-seconds N     GPU capacity check interval (default: 600)
  --max-gpu-util N     Start only when utilization is at most N percent (default: 15)
  --32b-mem-fraction-static F
                      Override the 32B calibration cache fraction (default: 0.96)
  --resume             Continue an incomplete output root
  -h, --help           Show this help

Environment overrides: SGLANG_BENCH_PYTHON, SGLANG_CHECKPOINT_ROOT,
SGLANG_KERNEL_COMPAT_STAGE, SGLANG_ML_CALIBRATION_ROOT, SGLANG_ML_MODELS,
SGLANG_ML_GPU_POLL_SECONDS, SGLANG_ML_MAX_GPU_UTIL_PERCENT, and
SGLANG_ML_32B_MEM_FRACTION_STATIC.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --poll-seconds) poll_seconds=${2:?missing value}; shift 2 ;;
        --max-gpu-util) max_gpu_util=${2:?missing value}; shift 2 ;;
        --32b-mem-fraction-static) mem_fraction_32b=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || { echo "--poll-seconds must be positive" >&2; exit 2; }
[[ "$max_gpu_util" =~ ^[0-9]{1,3}$ ]] && (( max_gpu_util <= 100 )) || {
    echo "--max-gpu-util must be an integer in [0, 100]" >&2; exit 2;
}
[[ "$mem_fraction_32b" =~ ^(0|0\.[0-9]+|1|1\.0+)$ ]] &&
    awk -v value="$mem_fraction_32b" 'BEGIN { exit !(value > 0 && value <= 1) }' || {
    echo "--32b-mem-fraction-static must be in (0, 1]" >&2; exit 2;
}
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
[[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
    echo "Missing isolated SM80 compatibility package: $compat_stage" >&2; exit 1;
}
if [[ -e "$results_root" && "$resume" != 1 ]]; then
    echo "Refusing to overwrite existing output root: $results_root" >&2
    exit 1
fi
if [[ "$resume" == 1 && ! -d "$results_root" ]]; then
    echo "--resume requires an existing output root: $results_root" >&2
    exit 1
fi

mkdir -p "$results_root" "$results_root/hardware" "$results_root/models"
lock_file="$results_root/.ml-calibration.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another ML calibration launcher already owns $results_root" >&2
    exit 1
fi
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date -Is)" > "$results_root/.ml-calibration.owner"
exec > >(tee -a "$results_root/runner.log") 2>&1
printf 'timestamp=%s\npid=%s\nresume=%s\npoll_seconds=%s\nmax_gpu_util_percent=%s\n32b_calibration_mem_fraction_static=%s\n\n' \
    "$(date -Is)" "$$" "$resume" "$poll_seconds" "$max_gpu_util" "$mem_fraction_32b" \
    >> "$results_root/launch_history.txt"

torch_lib_dir=$("$python_bin" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export LD_LIBRARY_PATH="$torch_lib_dir:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$compat_stage:$sglang_root/python:$sim_root/src${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_CACHE_DIR="$results_root/cache"
export SGLANG_JIT_CACHE_DIR="$results_root/jit-cache"
export FLASHINFER_WORKSPACE_BASE="$results_root/flashinfer-workspace"
mkdir -p "$SGLANG_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

if [[ "$resume" != 1 || ! -s "$results_root/run_metadata.txt" ]]; then
    {
        printf 'source_path=%s\nsource_revision=%s\n' "$sglang_root" "$(git -C "$sglang_root" rev-parse HEAD)"
        printf 'source_version_identity=local PR #33824 source; package metadata may be 0.0.0\n'
        printf 'python=%s\nmodels=%s\n' "$python_bin" "$models_csv"
        printf 'predictor=ml\naiconfigurator_used=false\n'
        printf 'target=per-model native SGLang one_batch CUDA-synchronized prefill/decode elapsed time\n'
        printf 'attention_backend=triton\ncuda_graph_config=%s\noverlap_schedule_enabled=1\n' "$cuda_graph_config"
        printf 'kernel_compat_stage=%s\nkernel_execution_path=isolated SM80 AOT package\n' "$compat_stage"
        printf 'train_batch_sizes=%s\ntrain_input_lens=%s\ntrain_output_lens=%s\n' \
            "${train_batch_sizes[*]}" "${train_input_lens[*]}" "${train_output_lens[*]}"
        printf 'holdout_batch_sizes=%s\nholdout_input_lens=%s\nholdout_output_lens=%s\n' \
            "${holdout_batch_sizes[*]}" "${holdout_input_lens[*]}" "${holdout_output_lens[*]}"
        printf 'gpu_capacity_policy=min-free-MiB-per-model and utilization<=%s; poll-seconds=%s\n' "$max_gpu_util" "$poll_seconds"
        printf '32b_calibration_mem_fraction_static=%s\n' "$mem_fraction_32b"
    } > "$results_root/run_metadata.txt"
    "$python_bin" - <<'PY' > "$results_root/package_versions.txt"
import importlib.metadata as md
import sys
print(f"python={sys.version.split()[0]}")
for name in ("torch", "transformers", "sglang", "sglang-simulator", "scikit-learn", "joblib", "numpy", "aiconfigurator"):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=not-installed")
PY
    nvidia-smi -q > "$results_root/hardware/nvidia-smi-q-at-start.txt" 2>&1 || true
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader \
        > "$results_root/hardware/gpu_identity.csv" 2>&1 || true
fi

required_free_mib() {
    case "$1" in
        Qwen3-32B) printf '76800\n' ;;
        Qwen2.5-7B-Instruct) printf '45000\n' ;;
        *) printf '35000\n' ;;
    esac
}

mem_fraction_for_model() {
    case "$1" in
        Qwen3-32B) printf '%s\n' "$mem_fraction_32b" ;;
        *) printf '0.80\n' ;;
    esac
}

wait_for_gpu() {
    local model=$1 required=$2 line free util
    while true; do
        line=$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
        free=$(awk -F, '{gsub(/[^0-9]/, "", $1); print $1}' <<< "$line")
        util=$(awk -F, '{gsub(/[^0-9]/, "", $2); print $2}' <<< "$line")
        if [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] && (( free >= required && util <= max_gpu_util )); then
            printf '[gpu-capacity] model=%s free=%s MiB required=%s MiB util=%s%%; starting collection\n' \
                "$model" "$free" "$required" "$util"
            return 0
        fi
        printf '[gpu-capacity] model=%s free=%s MiB required=%s MiB util=%s%% max=%s%%; waiting %s seconds\n' \
            "$model" "${free:-unknown}" "$required" "${util:-unknown}" "$max_gpu_util" "$poll_seconds"
        sleep "$poll_seconds"
    done
}

is_measurement_complete() {
    [[ -s "$1/result.jsonl" && -s "$1/benchmark.log" && -f "$1/status.txt" ]] && grep -qx ok "$1/status.txt"
}

archive_incomplete_measurement() {
    local output_dir=$1 model=$2 archive suffix=1
    [[ -d "$output_dir" ]] || return 0
    if ! find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | rg -q .; then
        return 0
    fi
    archive="${output_dir}.failed-$(date +%Y%m%d_%H%M%S)"
    while [[ -e "$archive" ]]; do
        archive="${output_dir}.failed-$(date +%Y%m%d_%H%M%S)-$suffix"
        suffix=$((suffix + 1))
    done
    mv "$output_dir" "$archive"
    printf '[collection] archived incomplete %s measurement at %s\n' "$model" "$archive"
}

run_one_batch_grid() {
    local model=$1 model_path=$2 mem_fraction=$3 split=$4 output_dir=$5
    shift 5
    local batch_sizes_text=$1 input_lens_text=$2 output_lens_text=$3
    local -a batch_sizes=() input_lens=() output_lens=()
    read -r -a batch_sizes <<< "$batch_sizes_text"
    read -r -a input_lens <<< "$input_lens_text"
    read -r -a output_lens <<< "$output_lens_text"
    if is_measurement_complete "$output_dir"; then
        echo "[collection] $model $split already complete"
        return 0
    fi
    archive_incomplete_measurement "$output_dir" "$model"
    mkdir -p "$output_dir"
    local -a model_override_args=() quantization_args=()
    if jq -e '.dual_chunk_attention_config != null' "$model_path/config.json" > /dev/null; then
        model_override_args=(--json-model-override-args '{"dual_chunk_attention_config": null}')
    fi
    if [[ "$(jq -r '.quantization_config.quant_method // empty' "$model_path/config.json")" == "awq" ]]; then
        quantization_args=(--quantization awq)
    fi
    local -a command=(
        env CUDA_VISIBLE_DEVICES=0
        SGLANG_CACHE_DIR="$SGLANG_CACHE_DIR/$model/$split"
        SGLANG_JIT_CACHE_DIR="$SGLANG_JIT_CACHE_DIR/$model/$split"
        FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE_BASE/$model/$split"
        "$python_bin" -m sglang.benchmark.one_batch
        --model-path "$model_path"
        --device cuda
        --tp-size 1
        --context-length 8192
        --mem-fraction-static "$mem_fraction"
        --max-running-requests 32
        --chunked-prefill-size 2048
        --attention-backend triton
        --cuda-graph-config "$cuda_graph_config"
        --batch-size "${batch_sizes[@]}"
        --input-len "${input_lens[@]}"
        --output-len "${output_lens[@]}"
        --log-decode-step 1
        --run-name "ml_native_${model}_${split}"
        --result-filename "$output_dir/result.jsonl"
    )
    command+=("${model_override_args[@]}" "${quantization_args[@]}")
    printf '%q ' "${command[@]}" > "$output_dir/command.txt"
    printf '\n' >> "$output_dir/command.txt"
    mkdir -p "$SGLANG_CACHE_DIR/$model/$split" "$SGLANG_JIT_CACHE_DIR/$model/$split" "$FLASHINFER_WORKSPACE_BASE/$model/$split"
    rm -f "$output_dir/result.jsonl"
    if "${command[@]}" > "$output_dir/benchmark.log" 2>&1; then
        if [[ -s "$output_dir/result.jsonl" ]] && ! rg -q '^skipping \(' "$output_dir/benchmark.log"; then
            printf 'ok\n' > "$output_dir/status.txt"
            return 0
        fi
        printf 'missing-or-skipped-shape\n' > "$output_dir/status.txt"
    else
        printf 'benchmark-failed\n' > "$output_dir/status.txt"
    fi
    return 1
}

run_grouped_one_batch_grid() {
    local model=$1 model_path=$2 mem_fraction=$3 split=$4 stage_dir=$5
    shift 5
    local -a group_specs=("$@")
    local group_spec group_name batch_text input_text output_text group_dir
    local -a batches=() inputs=() outputs=()
    local expected_rows=0 actual_rows=0

    if is_measurement_complete "$stage_dir"; then
        echo "[collection] $model $split already complete"
        return 0
    fi
    archive_incomplete_measurement "$stage_dir" "$model"
    mkdir -p "$stage_dir/groups"
    : > "$stage_dir/result.jsonl"
    : > "$stage_dir/benchmark.log"
    : > "$stage_dir/grid-manifest.txt"
    printf 'policy=32B trace-feasible static groups; each prefill group has batch_size*input_len<=2048\n' \
        >> "$stage_dir/grid-manifest.txt"

    for group_spec in "${group_specs[@]}"; do
        IFS='|' read -r group_name batch_text input_text output_text <<< "$group_spec"
        [[ -n "$group_name" && -n "$batch_text" && -n "$input_text" && -n "$output_text" ]] || {
            echo "Malformed grouped measurement spec: $group_spec" >&2
            printf 'malformed-group-spec\n' > "$stage_dir/status.txt"
            return 1
        }
        batches=()
        inputs=()
        outputs=()
        read -r -a batches <<< "$batch_text"
        read -r -a inputs <<< "$input_text"
        read -r -a outputs <<< "$output_text"
        group_dir="$stage_dir/groups/$group_name"
        if ! run_one_batch_grid "$model" "$model_path" "$mem_fraction" "${split}_${group_name}" "$group_dir" \
            "$batch_text" "$input_text" "$output_text"; then
            printf 'group-failed=%s\n' "$group_name" > "$stage_dir/status.txt"
            return 1
        fi
        expected_rows=$((expected_rows + ${#batches[@]} * ${#inputs[@]} * ${#outputs[@]}))
        printf 'group=%s\nbatches=%s\ninputs=%s\noutputs=%s\n' \
            "$group_name" "$batch_text" "$input_text" "$output_text" \
            >> "$stage_dir/grid-manifest.txt"
        cat "$group_dir/result.jsonl" >> "$stage_dir/result.jsonl"
        printf '\n# grouped measurement: %s\n' "$group_name" >> "$stage_dir/benchmark.log"
        cat "$group_dir/benchmark.log" >> "$stage_dir/benchmark.log"
    done

    actual_rows=$(wc -l < "$stage_dir/result.jsonl")
    printf '%s\n' "$expected_rows" > "$stage_dir/expected_result_rows.txt"
    if [[ "$actual_rows" != "$expected_rows" ]]; then
        printf 'shape-count-mismatch expected=%s actual=%s\n' "$expected_rows" "$actual_rows" > "$stage_dir/status.txt"
        return 1
    fi
    printf 'ok\n' > "$stage_dir/status.txt"
}

run_32b_trace_feasible_grid() {
    local model=$1 model_path=$2 mem_fraction=$3 split=$4 stage_dir=$5
    local -a group_specs=()
    case "$split" in
        calibration)
            # Existing 32B serving traces cap a prefill iteration at 2048
            # tokens, while decode reaches roughly batch 32 / past 1280.
            # The prior 32x1024 static prefill is therefore both absent from
            # the trace and too large for a single A100 80GB process.
            group_specs=(
                'b1_2|1 2|64 128 256 512 1024|32 128 256'
                'b4|4|64 128 256 512|32 128 256 768'
                'b8|8|64 128 256|32 128 256 1024'
                'b16|16|64 128|32 128 256 1152'
                'b32|32|64|32 128 256 1216'
            )
            ;;
        holdout)
            # These shapes are not part of the calibration groups.  The long
            # tails validate decode interpolation at the same past range.
            group_specs=(
                'b3|3|192 384|128 1088'
                'b6|6|192|128 1024'
                'b12|12|96|128 1152'
                'b24|24|64|128 1216'
            )
            ;;
        *)
            echo "Unsupported 32B grouped split: $split" >&2
            return 1
            ;;
    esac
    run_grouped_one_batch_grid "$model" "$model_path" "$mem_fraction" "$split" "$stage_dir" "${group_specs[@]}"
}

IFS=',' read -r -a models <<< "$models_csv"
failures=0
for model in "${models[@]}"; do
    model=${model//[[:space:]]/}
    [[ -n "$model" ]] || continue
    model_path="$checkpoint_root/$model"
    model_dir="$results_root/models/$model"
    if [[ ! -f "$model_path/config.json" ]]; then
        echo "[collection] missing config for $model: $model_path/config.json" >&2
        failures=1
        continue
    fi
    if [[ "$resume" == 1 && -f "$model_dir/CALIBRATION_COMPLETE" ]]; then
        echo "[collection] $model already trained"
        continue
    fi
    mkdir -p "$model_dir"
    cp "$model_path/config.json" "$model_dir/model-config.json"
    mem_fraction=$(mem_fraction_for_model "$model")
    required_mib=$(required_free_mib "$model")
    {
        printf 'model=%s\nmodel_path=%s\nmem_fraction_static=%s\nminimum_free_gpu_mib=%s\n' \
            "$model" "$model_path" "$mem_fraction" "$required_mib"
        printf 'attention_backend=triton\ncuda_graph_config=%s\noverlap_schedule_enabled=1\n' "$cuda_graph_config"
    } > "$model_dir/measurement-settings.txt"

    wait_for_gpu "$model" "$required_mib"
    if [[ "$model" == Qwen3-32B ]]; then
        calibration_runner=(run_32b_trace_feasible_grid "$model" "$model_path" "$mem_fraction" calibration "$model_dir/calibration")
    else
        calibration_runner=(run_one_batch_grid "$model" "$model_path" "$mem_fraction" calibration "$model_dir/calibration" \
            "${train_batch_sizes[*]}" "${train_input_lens[*]}" "${train_output_lens[*]}")
    fi
    if ! "${calibration_runner[@]}"; then
        echo "[collection] calibration grid failed for $model" >&2
        failures=1
        continue
    fi
    if [[ "$model" == Qwen3-32B ]]; then
        holdout_runner=(run_32b_trace_feasible_grid "$model" "$model_path" "$mem_fraction" holdout "$model_dir/holdout")
    else
        holdout_runner=(run_one_batch_grid "$model" "$model_path" "$mem_fraction" holdout "$model_dir/holdout" \
            "${holdout_batch_sizes[*]}" "${holdout_input_lens[*]}" "${holdout_output_lens[*]}")
    fi
    if ! "${holdout_runner[@]}"; then
        echo "[collection] holdout grid failed for $model" >&2
        failures=1
        continue
    fi

    if "$python_bin" "$script_dir/train_pr33824_ml_latency_model.py" \
        --model-name "$model" \
        --train-results "$model_dir/calibration/result.jsonl" \
        --train-log "$model_dir/calibration/benchmark.log" \
        --validation-results "$model_dir/holdout/result.jsonl" \
        --validation-log "$model_dir/holdout/benchmark.log" \
        --output-model "$model_dir/model/latency_model.pkl" \
        --output-dir "$model_dir/analysis" \
        --simulator-config "$model_dir/simulator-config.json" \
        > "$model_dir/training.log" 2>&1; then
        "$python_bin" - "$model_dir/model/latency_model.pkl" <<'PY'
import joblib
import sys
from sglang_simulator.time_predictor.ml import MLTimePredictor

bundle = joblib.load(sys.argv[1])
if bundle.get("features") != MLTimePredictor.FEATURE_NAMES:
    raise SystemExit("saved feature ABI does not match MLTimePredictor")
if not callable(getattr(bundle.get("model"), "predict", None)):
    raise SystemExit("saved ML model lacks predict()")
print("bundle_contract=ok")
PY
        printf 'ok\n' > "$model_dir/status.txt"
        touch "$model_dir/CALIBRATION_COMPLETE"
    else
        printf 'training-failed\n' > "$model_dir/status.txt"
        failures=1
    fi
done

nvidia-smi -q > "$results_root/hardware/nvidia-smi-q-at-end.txt" 2>&1 || true
if [[ "$failures" == 0 ]]; then
    rm -f "$results_root/CALIBRATION_COMPLETED_WITH_FAILURES"
    touch "$results_root/CALIBRATION_COMPLETE"
    echo "ML calibration complete: $results_root"
else
    touch "$results_root/CALIBRATION_COMPLETED_WITH_FAILURES"
    echo "ML calibration ended with failures: $results_root" >&2
fi
exit "$failures"
