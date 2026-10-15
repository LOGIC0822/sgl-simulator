#!/usr/bin/env bash
# Build one current-kernel AIC Silicon database, extend it with every operator
# shape observed in real training traces, add a regular no-holdout attention
# support grid, then build an isolated scoring DB for robust Silicon lookup.
# Holdout traces are intentionally never read by this script.

set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
aic_source=${AICONFIGURATOR_SOURCE_ROOT:-/home/ms-admin/workspace1/yiheng/ali/aiconfigurator-v0.10.0}
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
results_root=${SGLANG_COMPREHENSIVE_RESULTS_ROOT:-}
models_csv=${SGLANG_COMPREHENSIVE_MODELS:-Qwen3-0.6B,Qwen2.5-3B-Instruct,llama-2-7b,Mistral-7B-Instruct-v0.2,Qwen2.5-7B-Instruct,Qwen3-8B,Qwen2.5-14B-Instruct,Qwen2.5-14B-Instruct-1M,Qwen3-14B,Meta-Llama-3.1-70B-Instruct-AWQ,Qwen3-30B-A3B,Qwen3-30B-A3B-Instruct-2507,Qwen3-32B}
train_repeats=${SGLANG_COMPREHENSIVE_TRAIN_REPEATS:-5}
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:-"$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4"}
resume=0
export AICONFIGURATOR_COLLECTOR_TRITON_PAGE_SIZE=${AICONFIGURATOR_COLLECTOR_TRITON_PAGE_SIZE:-1}
sequence_grid_step=${SGLANG_AIC_SEQUENCE_GRID_STEP:-0}
if [[ "$sequence_grid_step" == 0 && "${SGLANG_COMPREHENSIVE_LEARNING_CURVES:-0}" == 1 ]]; then
    sequence_grid_step=64
fi

usage() {
    cat <<'EOF'
Usage: bash scripts/run_pr33824_trace_targeted_aic_calibration.sh [--results-root PATH] [--models CSV] [--resume]

First creates/resumes AIC's wide local A100 Silicon table. Then, for every
selected model, it derives additional attention and BF16 GEMM targets solely
from its real training traces and appends missing AIC points. Finally, it
measures a regular attention support grid from the declared scheduler bounds
and creates an isolated scoring database whose relevant context slices are
rectangular. It does not read holdout paths, so held-out scheduler latency
remains independent.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -d "$aic_source" && -f "$aic_source/collector/collect.py" ]] || { echo "Missing AIC source: $aic_source" >&2; exit 1; }
[[ -n "$results_root" ]] || { echo "Set SGLANG_COMPREHENSIVE_RESULTS_ROOT or pass --results-root" >&2; exit 2; }
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
[[ -d "$results_root/models" ]] || { echo "Missing comprehensive models root: $results_root/models" >&2; exit 1; }
[[ "$train_repeats" =~ ^[1-9][0-9]*$ ]] || { echo "train repeats must be positive" >&2; exit 2; }
[[ "$sequence_grid_step" =~ ^[0-9]+$ ]] || { echo "sequence grid step must be nonnegative" >&2; exit 2; }
compat_stage=$(realpath "$compat_stage")
[[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || { echo "Missing SM80 compatibility stage: $compat_stage" >&2; exit 1; }

results_root=$(realpath "$results_root")
aic_root="$results_root/aic_calibration"
data_dir="$aic_root/aic-systems/data/a100_pcie_local/sglang/0.5.10"
sglang_cache_dir="$aic_root/cache"
sglang_jit_cache_dir="$aic_root/jit-cache"
flashinfer_workspace_base="$aic_root/flashinfer-workspace"
support_plan="$aic_root/aic_interpolation_support_plan.json"
support_manifest="$aic_root/aic_interpolation_support_manifest.json"
support_marker="$aic_root/AIC_INTERPOLATION_SUPPORT_COMPLETE"
scoring_systems="$aic_root/aic-systems-regular-context"
scoring_config="$aic_root/simulator-config.a100_pcie_local_aic010_regular_context.json"
scoring_manifest="$aic_root/aic_regular_context_scoring_database_manifest.json"
scoring_marker="$aic_root/AIC_REGULAR_CONTEXT_SCORING_DB_COMPLETE"
status_path="$results_root/aic_calibration_status.tsv"
if [[ ! -f "$status_path" ]]; then
    printf 'timestamp\tmodel\tstage\tstate\tdetail\n' > "$status_path"
fi

record_status() {
    local model=$1 stage=$2 state=$3 detail=${4:-}
    detail=${detail//$'\t'/ }
    detail=${detail//$'\n'/ }
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$model" "$stage" "$state" "$detail" >> "$status_path"
}

is_complete() {
    [[ -f "$1/$2" ]] && grep -qx 'complete' "$1/$2"
}

wait_operator_capacity() {
    [[ "${SGLANG_GPU_CAPACITY_WAIT:-0}" == 1 ]] || return 0
    # Conservative headroom for synthetic GEMM/MoE/attention buffers;
    # not a claim that an isolated operator needs a complete checkpoint.
    "$python_bin" "$sim_root/tools/wait_pr33824_gpu_capacity.py" --fraction 0.7 \
        --minimum-free-mib "${SGLANG_AIC_MINIMUM_FREE_MIB:-57344}" \
        --interval "${SGLANG_GPU_CAPACITY_CHECK_SECONDS:-600}" --log "$1"
}

IFS=',' read -r -a models <<< "$models_csv"
filtered_models=()
for model in "${models[@]}"; do
    model=${model//[[:space:]]/}
    [[ -n "$model" ]] || continue
    [[ -d "$results_root/models/$model" ]] || { echo "Model has no comprehensive result directory: $model" >&2; exit 1; }
    for repeat in $(seq 1 "$train_repeats"); do
        collection="$results_root/models/$model/calibration/real/repeat-$repeat"
        is_complete "$collection" BASELINE_COLLECTION_COMPLETE || {
            echo "Incomplete real training collection: $collection" >&2; exit 1;
        }
    done
    filtered_models+=("$model")
done
(( ${#filtered_models[@]} > 0 )) || { echo "No selected models" >&2; exit 2; }

# The broad AIC attention/GEMM case generator is intentionally dense-only.
# Its separate --all MoE pass still measures the common Qwen3-MoE table, but
# supplying a MoE checkpoint to the dense YAML generator aborts before that
# pass begins.  The trace-targeted loop below remains responsible for every
# selected model, including MoE checkpoints.
wide_models=()
wide_moe_models=()
for model in "${filtered_models[@]}"; do
    if "$python_bin" -c 'import json, sys; config = json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(1 if (config.get("num_experts") or config.get("num_local_experts")) else 0)' "$checkpoint_root/$model/config.json"; then
        wide_models+=("$model")
    else
        wide_moe_models+=("$model")
    fi
done
(( ${#wide_models[@]} > 0 )) || {
    echo "No dense checkpoints are available for the wide AIC base sweep" >&2
    exit 1
}
wide_models_csv=$(IFS=,; echo "${wide_models[*]}")
wide_moe_models_csv=$(IFS=,; echo "${wide_moe_models[*]:-}")

lock_file="$results_root/.aic-calibration.lock"
exec 9>"$lock_file"
flock -n 9 || { echo "Another AIC calibration launcher owns $results_root" >&2; exit 1; }

if [[ ! -s "$aic_root/simulator-config.a100_pcie_local_aic010.json" ]]; then
    record_status matrix wide_aic started "dense base sweep plus common MoE table; trace-targeted stage covers all selected models"
    wait_operator_capacity "$aic_root/hardware/wide_gpu_capacity.jsonl"
    aic_args=(--all --results-root "$aic_root" --models "$wide_models_csv")
    [[ -d "$aic_root" ]] && aic_args+=(--resume)
    SGLANG_BENCH_PYTHON="$python_bin" \
        SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
        AICONFIGURATOR_SOURCE_ROOT="$aic_source" \
        bash "$script_dir/run_pr33824_aic010_all13_collector.sh" "${aic_args[@]}"
    {
        printf 'wide_base_model_policy=dense configurations only; generic Qwen3 MoE table collected by --all\n'
        printf 'wide_base_models=%s\n' "$wide_models_csv"
        printf 'wide_base_moe_models_excluded_from_dense_generator=%s\n' "$wide_moe_models_csv"
        printf 'trace_targeted_policy=all selected models, including MoE, are processed from training trace shapes below\n'
    } >> "$aic_root/run_metadata.txt"
    record_status matrix wide_aic complete ""
fi
[[ -d "$data_dir" ]] || { echo "AIC data directory is missing: $data_dir" >&2; exit 1; }
mkdir -p "$sglang_cache_dir" "$sglang_jit_cache_dir" "$flashinfer_workspace_base"

torch_lib_dir=$("$python_bin" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
collector_pythonpath="$compat_stage:$script_dir/collector_shims:$script_dir:$aic_source/src:$aic_source:$sglang_root/python:$sim_root/src${SGLANG_EXPERIMENT_EXTRA_PYTHONPATH:+:$SGLANG_EXPERIMENT_EXTRA_PYTHONPATH}"

for model in "${filtered_models[@]}"; do
    model_root="$results_root/models/$model"
    calibration_root="$model_root/aic_calibration"
    marker="$calibration_root/AIC_TRACE_CALIBRATION_COMPLETE"
    if is_complete "$calibration_root" AIC_TRACE_CALIBRATION_COMPLETE; then
        record_status "$model" trace_targets skipped "existing completion marker"
        continue
    fi
    [[ ! -e "$calibration_root" ]] || { echo "Incomplete AIC calibration directory: $calibration_root" >&2; exit 1; }
    mkdir -p "$calibration_root"
    plan_args=()
    for repeat in $(seq 1 "$train_repeats"); do
        plan_args+=(--data-root "$model_root/calibration/real/repeat-$repeat/cases")
    done
    record_status "$model" trace_plan started "training traces only"
    "$python_bin" "$sim_root/tools/generate_pr33824_trace_aic_plan.py" \
        "${plan_args[@]}" --model-path "$checkpoint_root/$model" \
        --output "$calibration_root/training_trace_aic_plan.json" \
        --summary-csv "$calibration_root/training_trace_attention_targets.csv" \
        > "$calibration_root/plan.log" 2>&1
    record_status "$model" trace_plan complete ""
    record_status "$model" trace_targets started "AIC triton attention and BF16 GEMM points"
    wait_operator_capacity "$calibration_root/hardware/gpu_capacity.jsonl"
    printf '%q ' env \
        AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND=triton \
        AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE=0.5.10 \
        AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS=1 \
        AIC_COLLECT_GEMM_TYPES=bfloat16 \
        SGLANG_CACHE_DIR="$sglang_cache_dir" SGLANG_JIT_CACHE_DIR="$sglang_jit_cache_dir" \
        FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base" \
        PYTHONPATH="$collector_pythonpath" LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$python_bin" "$sim_root/tools/collect_pr33824_trace_targeted_aic.py" \
        --plan "$calibration_root/training_trace_aic_plan.json" \
        --model-path "$checkpoint_root/$model" --data-dir "$data_dir" \
        --manifest "$calibration_root/aic_trace_collection_manifest.json" --resume \
        > "$calibration_root/collector.command.txt"
    printf '\n' >> "$calibration_root/collector.command.txt"
    env AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND=triton \
        AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE=0.5.10 \
        AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS=1 \
        AIC_COLLECT_GEMM_TYPES=bfloat16 \
        SGLANG_CACHE_DIR="$sglang_cache_dir" SGLANG_JIT_CACHE_DIR="$sglang_jit_cache_dir" \
        FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base" \
        PYTHONPATH="$collector_pythonpath" LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$python_bin" "$sim_root/tools/collect_pr33824_trace_targeted_aic.py" \
        --plan "$calibration_root/training_trace_aic_plan.json" \
        --model-path "$checkpoint_root/$model" --data-dir "$data_dir" \
        --manifest "$calibration_root/aic_trace_collection_manifest.json" --resume \
        > "$calibration_root/collector.log" 2>&1
    printf 'complete\n' > "$marker"
    record_status "$model" trace_targets complete ""
done

# Sparse trace-targeted points are exact for observed training shapes, but AIC
# Silicon's 3-D attention interpolation also needs regular support for unseen
# batch/sequence combinations.  This plan intentionally reads only the
# already-derived training plans plus the fixed scheduler limits below; it
# never opens any holdout trace path.
if ! is_complete "$aic_root" AIC_INTERPOLATION_SUPPORT_COMPLETE; then
    if [[ ! -f "$support_plan" ]]; then
        support_plan_args=()
        for model in "${filtered_models[@]}"; do
            calibration_root="$results_root/models/$model/aic_calibration"
            trace_plan="$calibration_root/training_trace_aic_plan.json"
            [[ -f "$trace_plan" ]] || { echo "Missing trace training plan: $trace_plan" >&2; exit 1; }
            support_plan_args+=(--model-path "$checkpoint_root/$model" --training-plan "$trace_plan")
        done
        record_status matrix interpolation_support_plan started "training plans plus context=2048,max_running_requests=32"
        "$python_bin" "$sim_root/tools/generate_pr33824_aic_interpolation_support_plan.py" \
            "${support_plan_args[@]}" --max-context-tokens 2048 --max-batch-size 32 \
            --sequence-grid-step "$sequence_grid_step" \
            --output "$support_plan" > "$aic_root/aic_interpolation_support_plan.log" 2>&1
        record_status matrix interpolation_support_plan complete "$(tail -n 1 "$aic_root/aic_interpolation_support_plan.log")"
    fi
    if [[ ! -s "$support_manifest" ]]; then
        record_status matrix interpolation_support_collect started "regular Triton grid; training-only support plan"
        wait_operator_capacity "$aic_root/hardware/support_gpu_capacity.jsonl"
        printf '%q ' env \
            AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND=triton \
            AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE=0.5.10 \
            AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS=1 \
            SGLANG_CACHE_DIR="$sglang_cache_dir" SGLANG_JIT_CACHE_DIR="$sglang_jit_cache_dir" \
            FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base" \
            PYTHONPATH="$collector_pythonpath" LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$python_bin" "$sim_root/tools/collect_pr33824_aic_interpolation_support.py" \
            --plan "$support_plan" --data-dir "$data_dir" --manifest "$support_manifest" --resume \
            > "$aic_root/aic_interpolation_support_collector.command.txt"
        printf '\n' >> "$aic_root/aic_interpolation_support_collector.command.txt"
        env AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND=triton \
            AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE=0.5.10 \
            AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS=1 \
            SGLANG_CACHE_DIR="$sglang_cache_dir" SGLANG_JIT_CACHE_DIR="$sglang_jit_cache_dir" \
            FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace_base" \
            PYTHONPATH="$collector_pythonpath" LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$python_bin" "$sim_root/tools/collect_pr33824_aic_interpolation_support.py" \
            --plan "$support_plan" --data-dir "$data_dir" --manifest "$support_manifest" --resume \
            > "$aic_root/aic_interpolation_support_collector.log" 2>&1
        record_status matrix interpolation_support_collect complete "$(tail -n 1 "$aic_root/aic_interpolation_support_collector.log")"
    else
        record_status matrix interpolation_support_collect resumed "existing complete manifest"
    fi
    [[ -s "$support_manifest" ]] || { echo "AIC interpolation support manifest is missing" >&2; exit 1; }
    printf 'complete\n' > "$support_marker"
    record_status matrix interpolation_support complete ""
fi

# The raw context table intentionally includes exact but sparse trace rows.
# AIC 0.10 chooses the nearest sequence key before interpolating batch size;
# therefore a sparse row can shadow the regular support grid for an unseen
# holdout shape.  Build a separate *scoring* DB whose selected context slices
# contain only the measured rectangular support grid. The raw calibration DB
# remains intact, and GEMM/generation/MoE tables are copied unchanged.
if ! is_complete "$aic_root" AIC_REGULAR_CONTEXT_SCORING_DB_COMPLETE; then
    if [[ ! -s "$scoring_manifest" ]]; then
        record_status matrix regular_context_scoring_db started "isolate complete AIC Silicon context support grid"
        "$python_bin" "$sim_root/tools/build_pr33824_aic_regular_context_scoring_db.py" \
            --source-systems "$aic_root/aic-systems" --support-plan "$support_plan" \
            --source-sim-config "$aic_root/simulator-config.a100_pcie_local_aic010.json" \
            --output-systems "$scoring_systems" --output-sim-config "$scoring_config" \
            --manifest "$scoring_manifest" > "$aic_root/aic_regular_context_scoring_database.log" 2>&1
        record_status matrix regular_context_scoring_db complete "$(tail -n 1 "$aic_root/aic_regular_context_scoring_database.log")"
    else
        record_status matrix regular_context_scoring_db resumed "existing scoring database manifest"
    fi
    [[ -d "$scoring_systems" && -s "$scoring_config" && -s "$scoring_manifest" ]] || {
        echo "AIC regular-context scoring database is incomplete" >&2; exit 1;
    }
    printf 'complete\n' > "$scoring_marker"
    record_status matrix regular_context_scoring_db_ready complete ""
fi

{
    printf 'created_or_resumed_at=%s\n' "$(date -Is)"
    printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
    printf 'models=%s\ntrain_repeats=%s\n' "$(IFS=,; echo "${filtered_models[*]}")" "$train_repeats"
    printf 'collector_backend_version=0.5.10\n'
    printf 'collector_triton_page_size=%s\n' "$AICONFIGURATOR_COLLECTOR_TRITON_PAGE_SIZE"
    printf 'input_policy=training ScheduleBatch shapes only; independent holdout trace paths were not read\n'
    printf 'operator_timing_policy=attention: 3 warmups,min 20 CUDA-event iterations; GEMM: 3 warmups,min 6 CUDA-event iterations (each iteration contains a cache-aware op list); adaptive run count and CUDA graph replay as defined by collector/helper.py\n'
    printf 'interpolation_support_policy=regular attention grid from training plans plus context_length=2048 and max_running_requests=32; holdout paths were not read\n'
    printf 'additional_sequence_grid_step=%s\n' "$sequence_grid_step"
    printf 'interpolation_support_plan=%s\ninterpolation_support_manifest=%s\n' \
        "$support_plan" "$support_manifest"
    printf 'scoring_database_policy=for relevant context-attention slices, retain only direct regular-grid measurements so sparse training keys cannot shadow Silicon batch interpolation; all other operator tables are copied unchanged\n'
    printf 'scoring_systems=%s\nscoring_simulator_config=%s\nscoring_database_manifest=%s\n' \
        "$scoring_systems" "$scoring_config" "$scoring_manifest"
    printf 'kernel_compat_stage=%s\naic_source_revision=%s\n' \
        "$compat_stage" "$(git -C "$aic_source" rev-parse HEAD)"
    printf 'sglang_cache_dir=%s\nsglang_jit_cache_dir=%s\nflashinfer_workspace_base=%s\n' \
        "$sglang_cache_dir" "$sglang_jit_cache_dir" "$flashinfer_workspace_base"
} > "$aic_root/trace_targeted_run_metadata.txt"
printf 'complete\n' > "$results_root/AIC_TRACE_CALIBRATION_COMPLETE"
record_status matrix complete ""
echo "AIC trace-targeted calibration complete: $aic_root"
