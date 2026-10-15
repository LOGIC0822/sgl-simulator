#!/usr/bin/env bash
# Complete standardized experiment: real SGLang collection -> per-model ML ->
# AIC Silicon calibration from training shapes -> matched AIC replay -> report.

set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
export PYTHONNOUSERSITE=1
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
aic_source=${AICONFIGURATOR_SOURCE_ROOT:-/home/ms-admin/workspace1/yiheng/ali/aiconfigurator-v0.10.0}
if [[ ${SGLANG_KERNEL_COMPAT_STAGE+x} ]]; then
    compat_stage=$SGLANG_KERNEL_COMPAT_STAGE
else
    compat_stage="$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4"
fi
fa2_compat_stage=${SGLANG_FA2_COMPAT_STAGE:-}
results_root=${SGLANG_COMPREHENSIVE_RESULTS_ROOT:-"$sim_root/test-results/pr33824_comprehensive_all13_predictors_$(date +%Y%m%d_%H%M%S)"}
models_csv=${SGLANG_COMPREHENSIVE_MODELS:-Qwen3-0.6B,Qwen2.5-3B-Instruct,llama-2-7b,Mistral-7B-Instruct-v0.2,Qwen2.5-7B-Instruct,Qwen3-8B,Qwen2.5-14B-Instruct,Qwen2.5-14B-Instruct-1M,Qwen3-14B,Meta-Llama-3.1-70B-Instruct-AWQ,Qwen3-30B-A3B,Qwen3-30B-A3B-Instruct-2507,Qwen3-32B}
train_repeats=${SGLANG_COMPREHENSIVE_TRAIN_REPEATS:-5}
holdout_repeats=${SGLANG_COMPREHENSIVE_HOLDOUT_REPEATS:-5}
train_cases=${SGLANG_SAME_DISTRIBUTION_TRAIN_CASES:-latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf}
holdout_cases=${SGLANG_SAME_DISTRIBUTION_HOLDOUT_CASES:-$train_cases,latency_128x64_r1,saturation_1536x384_inf}
prompt_multiplier=${SGLANG_COMPREHENSIVE_PROMPT_MULTIPLIER:-2}
real_port=${SGLANG_COMPREHENSIVE_REAL_PORT:-30227}
simulator_port=${SGLANG_COMPREHENSIVE_SIMULATOR_PORT:-30228}
aic_simulator_config=${SGLANG_AIC_SIMULATOR_CONFIG:-"$results_root/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"}
phase=all
resume=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_pr33824_comprehensive_predictor_matrix.sh [options]

Default `--phase all` runs the configured model comparison. With the default
three training and five holdout workloads, every selected model receives 800
training requests and 1280 independent holdout requests over five real runs.
The ML bundle and AIC Silicon
database are both derived strictly from training traces; batch-level and E2E
accuracy are then evaluated on the independent holdout traces.

Options:
  --results-root PATH      New immutable root, or an existing root with --resume.
  --models CSV             Model directories below /mnt/data/models.
  --train-repeats N        Training traces per model (default: 5).
  --holdout-repeats N      Holdout traces per model (default: 5).
  --prompt-multiplier N    32-request saturation multiplier (default: 2 = 64).
  --phase NAME             all, collect, real-ml, aic, predict, or report (default: all).
  --resume                 Continue completed stages without replacing artifacts.
  -h, --help               Show this help.

All raw requests, real server logs, kernel/version records, AIC tables,
predictor outputs, CSV summaries, and figures remain below test-results. The
final Markdown/DOCX copies are written under tools/sglang-simulator/docs/.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --train-repeats) train_repeats=${2:?missing value}; shift 2 ;;
        --holdout-repeats) holdout_repeats=${2:?missing value}; shift 2 ;;
        --prompt-multiplier) prompt_multiplier=${2:?missing value}; shift 2 ;;
        --phase) phase=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
export SGLANG_BASELINE_SHARED_JIT_CACHE=${SGLANG_BASELINE_SHARED_JIT_CACHE:-"$results_root/shared/jit-cache"}

[[ "$phase" == all || "$phase" == collect || "$phase" == real-ml || "$phase" == aic || "$phase" == predict || "$phase" == report ]] || {
    echo "--phase must be all, collect, real-ml, aic, predict, or report" >&2; exit 2;
}
[[ "$train_repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--train-repeats must be positive" >&2; exit 2; }
[[ "$holdout_repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--holdout-repeats must be positive" >&2; exit 2; }
[[ "$prompt_multiplier" =~ ^[1-9][0-9]*$ ]] || { echo "--prompt-multiplier must be positive" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
[[ -d "$aic_source" ]] || { echo "Missing AIC source: $aic_source" >&2; exit 1; }
if [[ -n "$compat_stage" ]]; then
    compat_stage=$(realpath "$compat_stage")
fi
if [[ "$phase" != predict && "$phase" != report ]]; then
    [[ -n "$compat_stage" && -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
        echo "Missing isolated SM80 kernel stage: ${compat_stage:-<unset>}" >&2; exit 1;
    }
fi

is_complete() {
    [[ -f "$1/$2" ]] && grep -qx 'complete' "$1/$2"
}

IFS=',' read -r -a requested_models <<< "$models_csv"
models=()
for model in "${requested_models[@]}"; do
    model=${model//[[:space:]]/}
    [[ -n "$model" ]] || continue
    [[ -d "$checkpoint_root/$model" ]] || { echo "Missing checkpoint: $checkpoint_root/$model" >&2; exit 1; }
    models+=("$model")
done
(( ${#models[@]} > 0 )) || { echo "No selected models" >&2; exit 2; }
models_csv=$(IFS=,; echo "${models[*]}")

run_real_ml() {
    local matrix_phase=${1:-all}
    local -a args=(--results-root "$results_root" --models "$models_csv" --train-repeats "$train_repeats" \
        --holdout-repeats "$holdout_repeats" --prompt-multiplier "$prompt_multiplier" --phase "$matrix_phase")
    [[ "$resume" == 1 && -d "$results_root" ]] && args+=(--resume)
    SGLANG_BENCH_PYTHON="$python_bin" \
        SGLANG_CHECKPOINT_ROOT="$checkpoint_root" \
        SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
        SGLANG_FA2_COMPAT_STAGE="$fa2_compat_stage" \
        SGLANG_SAME_DISTRIBUTION_RESULTS_ROOT="$results_root" \
        SGLANG_SAME_DISTRIBUTION_MODELS="$models_csv" \
        SGLANG_SAME_DISTRIBUTION_TRAIN_REPEATS="$train_repeats" \
        SGLANG_SAME_DISTRIBUTION_HOLDOUT_REPEATS="$holdout_repeats" \
        SGLANG_SAME_DISTRIBUTION_TRAIN_CASES="$train_cases" \
        SGLANG_SAME_DISTRIBUTION_HOLDOUT_CASES="$holdout_cases" \
        SGLANG_SAME_DISTRIBUTION_PROMPT_MULTIPLIER="$prompt_multiplier" \
        SGLANG_SAME_DISTRIBUTION_REAL_PORT="$real_port" \
        SGLANG_SAME_DISTRIBUTION_SIMULATOR_PORT="$simulator_port" \
        bash "$script_dir/run_pr33824_data_tools_same_distribution_matrix.sh" "${args[@]}"
}

require_data_collection() {
    is_complete "$results_root" BASELINE_DATA_ALL_MODELS_COMPLETE || \
        is_complete "$results_root" REAL_ML_ALL_MODELS_COMPLETE || {
        echo "Real training/holdout collection is incomplete: $results_root" >&2; exit 1;
    }
    for model in "${models[@]}"; do
        is_complete "$results_root/models/$model" BASELINE_DATA_COMPLETE || \
            is_complete "$results_root/models/$model" REAL_ML_COMPLETE || {
            echo "Model data collection is incomplete: $model" >&2; exit 1;
        }
    done
}

require_real_ml() {
    is_complete "$results_root" REAL_ML_ALL_MODELS_COMPLETE || {
        echo "Real/ML stage is incomplete: $results_root" >&2; exit 1;
    }
    for model in "${models[@]}"; do
        is_complete "$results_root/models/$model" REAL_ML_COMPLETE || {
            echo "Model real/ML stage is incomplete: $model" >&2; exit 1;
        }
    done
}

initialize_master_log() {
    mkdir -p "$results_root"
    master_status="$results_root/comprehensive_status.tsv"
    if [[ ! -f "$master_status" ]]; then
        printf 'timestamp\tmodel\tstage\tstate\tdetail\n' > "$master_status"
    fi
}

record_status() {
    local model=$1 stage=$2 state=$3 detail=${4:-}
    detail=${detail//$'\t'/ }
    detail=${detail//$'\n'/ }
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$model" "$stage" "$state" "$detail" >> "$master_status"
}

run_aic_calibration() {
    require_data_collection
    initialize_master_log
    # A pre-existing trace-target marker alone is not sufficient: sparse
    # training shapes need the regular no-holdout support grid before AIC
    # Silicon can interpolate independent batch/sequence combinations.
    if ! is_complete "$results_root" AIC_TRACE_CALIBRATION_COMPLETE || \
        ! is_complete "$results_root/aic_calibration" AIC_INTERPOLATION_SUPPORT_COMPLETE || \
        ! is_complete "$results_root/aic_calibration" AIC_REGULAR_CONTEXT_SCORING_DB_COMPLETE; then
        record_status matrix aic_calibration started "wide table, training-trace points, regular interpolation support, and isolated scoring DB"
        SGLANG_BENCH_PYTHON="$python_bin" \
            SGLANG_CHECKPOINT_ROOT="$checkpoint_root" \
            SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
            SGLANG_FA2_COMPAT_STAGE="$fa2_compat_stage" \
            AICONFIGURATOR_SOURCE_ROOT="$aic_source" \
            SGLANG_COMPREHENSIVE_RESULTS_ROOT="$results_root" \
            SGLANG_COMPREHENSIVE_MODELS="$models_csv" \
            SGLANG_COMPREHENSIVE_TRAIN_REPEATS="$train_repeats" \
            bash "$script_dir/run_pr33824_trace_targeted_aic_calibration.sh" \
            --results-root "$results_root" --models "$models_csv" --resume
        record_status matrix aic_calibration complete ""
    fi
    is_complete "$results_root" AIC_TRACE_CALIBRATION_COMPLETE || {
        echo "AIC trace-targeted calibration marker missing" >&2; exit 1;
    }
    is_complete "$results_root/aic_calibration" AIC_INTERPOLATION_SUPPORT_COMPLETE || {
        echo "AIC regular interpolation-support marker missing" >&2; exit 1;
    }
    is_complete "$results_root/aic_calibration" AIC_REGULAR_CONTEXT_SCORING_DB_COMPLETE || {
        echo "AIC regular-context scoring database marker missing" >&2; exit 1;
    }
}

archive_failed_aic_attempt() {
    local source=$1 label=$2 archive_root
    [[ -e "$source" ]] || return 0
    archive_root="$(dirname "$source")/aic_failure_attempts"
    mkdir -p "$archive_root"
    mv "$source" "$archive_root/${label}.$(date +%Y%m%d_%H%M%S)"
}

score_aic_model() {
    local model=$1 model_root="$results_root/models/$model"
    local score_root="$model_root/holdout/aic_teacher_forced"
    local e2e_root="$model_root/holdout/aic_e2e_comparison"
    local config="$aic_simulator_config"
    local torch_lib_dir aic_pythonpath repeat replay_root state
    if is_complete "$model_root" AIC_PREDICTOR_COMPLETE; then
        record_status "$model" aic_predictor skipped "existing completion marker"
        return 0
    fi
    [[ -f "$config" ]] || { echo "Missing AIC simulator config: $config" >&2; return 1; }
    # Keep the original sparse-grid failure as an artifact when a resumed run
    # starts after adding the regular support table.  A failed scorer creates
    # no summary, so this archive does not discard any valid output.
    if [[ "$resume" == 1 && ! -s "$score_root/summary.csv" ]]; then
        archive_failed_aic_attempt "$score_root" "aic_teacher_forced_pre_support_grid_output"
        archive_failed_aic_attempt "$model_root/holdout/aic_teacher_forced.log" "aic_teacher_forced_pre_support_grid"
    fi
    torch_lib_dir=$("$python_bin" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
    aic_pythonpath="$aic_source/src:$aic_source:$sglang_root/python:$sim_root/src${SGLANG_EXPERIMENT_EXTRA_PYTHONPATH:+:$SGLANG_EXPERIMENT_EXTRA_PYTHONPATH}"
    [[ -z "$compat_stage" ]] || aic_pythonpath="$compat_stage:$aic_pythonpath"
    if [[ ! -s "$score_root/summary.csv" ]]; then
        [[ ! -e "$score_root" ]] || { echo "Incomplete AIC score path: $score_root" >&2; return 1; }
        record_status "$model" aic_teacher_forced started "held-out device-synchronized ScheduleBatch rows"
        if ! SGLANG_CACHE_DIR="$results_root/aic_calibration/cache" \
            SGLANG_JIT_CACHE_DIR="$results_root/aic_calibration/jit-cache" \
            FLASHINFER_WORKSPACE_BASE="$results_root/aic_calibration/flashinfer-workspace" \
            PYTHONPATH="$aic_pythonpath" LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$python_bin" "$sim_root/tools/score_pr33824_aic_latency_model.py" \
            --data-root "$model_root/holdout/real" --model-path "$checkpoint_root/$model" \
            --sim-config "$config" --out-dir "$score_root" --forward-modes 1,2 \
            > "$model_root/holdout/aic_teacher_forced.log" 2>&1; then
            record_status "$model" aic_teacher_forced failed "see holdout/aic_teacher_forced.log"
            return 1
        fi
        record_status "$model" aic_teacher_forced complete ""
    fi
    if [[ "$resume" == 1 && ! -s "$e2e_root/summary.csv" ]]; then
        archive_failed_aic_attempt "$e2e_root" "aic_e2e_pre_support_grid_output"
    fi
    for repeat in $(seq 1 "$holdout_repeats"); do
        replay_root="$model_root/holdout/aic_simulator/repeat-$repeat"
        if is_complete "$replay_root" AIC_REPLAY_COMPLETE; then
            record_status "$model" "aic-replay-repeat-$repeat" skipped "existing completion marker"
            continue
        fi
        if [[ "$resume" == 1 ]]; then
            archive_failed_aic_attempt "$replay_root" "aic_replay_repeat_${repeat}_pre_support_grid_output"
        fi
        [[ ! -e "$replay_root" ]] || { echo "Incomplete AIC replay path: $replay_root" >&2; return 1; }
        record_status "$model" "aic-replay-repeat-$repeat" started "CPU logical replay of independent holdout trace"
        if ! SGLANG_COLLECTION_ROOT="$model_root/holdout/real/repeat-$repeat" \
            SGLANG_REPLAY_SIM_CONFIG="$config" \
            SGLANG_REPLAY_COMPLETION_MARKER=AIC_REPLAY_COMPLETE \
            SGLANG_REPLAY_PREDICTOR_LABEL=aic_silicon \
            SGLANG_REPLAY_CASES="$holdout_cases" \
            SGLANG_REPLAY_RESULTS_ROOT="$replay_root" \
            SGLANG_BENCH_PYTHON="$python_bin" \
            SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
            SGLANG_SIMULATOR_PORT="$simulator_port" \
            AICONFIGURATOR_SOURCE_ROOT="$aic_source" \
            bash "$script_dir/run_pr33824_data_tools_ml_replay.sh"; then
            record_status "$model" "aic-replay-repeat-$repeat" failed "see $replay_root"
            return 1
        fi
        record_status "$model" "aic-replay-repeat-$repeat" complete ""
    done
    if [[ ! -s "$e2e_root/summary.csv" ]]; then
        [[ ! -e "$e2e_root" ]] || { echo "Incomplete AIC E2E path: $e2e_root" >&2; return 1; }
        local -a compare_args=()
        for repeat in $(seq 1 "$holdout_repeats"); do
            compare_args+=(--real-root "$model_root/holdout/real/repeat-$repeat")
            compare_args+=(--replay-root "$model_root/holdout/aic_simulator/repeat-$repeat")
        done
        record_status "$model" aic_e2e started "paired independent holdout trace summaries"
        if ! PYTHONPATH="$sim_root/src:$sglang_root/python" "$python_bin" \
            "$sim_root/tools/summarize_pr33824_data_tools_replay.py" "${compare_args[@]}" \
            --out-dir "$e2e_root" --teacher-forced-summary "$score_root/summary.csv"; then
            record_status "$model" aic_e2e failed "summarizer returned non-zero"
            return 1
        fi
        record_status "$model" aic_e2e complete ""
    fi
    printf 'complete\n' > "$model_root/AIC_PREDICTOR_COMPLETE"
    record_status "$model" aic_predictor complete ""
}

run_aic_predictor() {
    run_aic_calibration
    initialize_master_log
    if [[ "$resume" == 1 && -f "$results_root/AIC_PREDICTOR_COMPLETED_WITH_FAILURES" ]] && \
        ! is_complete "$results_root" AIC_ALL_MODELS_COMPLETE; then
        archive_failed_aic_attempt "$results_root/AIC_PREDICTOR_COMPLETED_WITH_FAILURES" "AIC_PREDICTOR_COMPLETED_WITH_FAILURES_pre_support_grid"
        record_status matrix aic_predictor resumed "archived prior sparse-grid scoring failure"
    fi
    local failures=0
    for model in "${models[@]}"; do
        if ! score_aic_model "$model"; then
            failures=1
            record_status "$model" aic_predictor failed "see model-local logs"
        fi
    done
    if (( failures )); then
        printf 'completed-with-failures\n' > "$results_root/AIC_PREDICTOR_COMPLETED_WITH_FAILURES"
        return 1
    fi
    printf 'complete\n' > "$results_root/AIC_ALL_MODELS_COMPLETE"
    record_status matrix aic_predictor complete ""
}

render_report() {
    require_real_ml
    is_complete "$results_root" AIC_ALL_MODELS_COMPLETE || {
        echo "AIC predictor stage is incomplete: $results_root" >&2; exit 1;
    }
    initialize_master_log
    local analysis_root="$results_root/analysis"
    local report_path="$sim_root/docs/$(basename "$results_root")_complete_report.md"
    local docx_path="${report_path%.md}.docx"
    if is_complete "$results_root" FINAL_REPORT_COMPLETE; then
        record_status matrix final_report skipped "existing completion marker"
        return 0
    fi
    [[ ! -e "$analysis_root" ]] || { echo "Incomplete analysis directory: $analysis_root" >&2; return 1; }
    [[ ! -e "$report_path" && ! -e "$docx_path" ]] || { echo "Existing report would be overwritten: $report_path" >&2; return 1; }
    record_status matrix final_report started "CSV, figures, Markdown, and SimSun DOCX"
    mkdir -p "$results_root/matplotlib-cache"
    MPLCONFIGDIR="$results_root/matplotlib-cache" "$python_bin" "$sim_root/tools/summarize_pr33824_comprehensive_predictor_experiment.py" \
        --results-root "$results_root" --models "$models_csv" --out-dir "$analysis_root" --markdown "$report_path"
    if [[ "${SGLANG_COMPREHENSIVE_LEARNING_CURVES:-0}" == 1 ]]; then
        MPLCONFIGDIR="$results_root/matplotlib-cache" "$python_bin" "$sim_root/tools/append_pr33824_expanded_report.py" \
            --results-root "$results_root" --markdown "$report_path" --out-dir "$analysis_root/workloads"
    fi
    "$python_bin" "$sim_root/scripts/render_markdown_report_docx.py" "$report_path" --output "$docx_path"
    printf 'complete\n' > "$results_root/FINAL_REPORT_COMPLETE"
    record_status matrix final_report complete "markdown=$report_path docx=$docx_path"
}

run_learning_curves() {
    [[ "${SGLANG_COMPREHENSIVE_LEARNING_CURVES:-0}" == 1 ]] || return 0
    require_real_ml
    initialize_master_log
    local model target
    for model in "${models[@]}"; do
        target="$results_root/models/$model/learning_curve"
        if [[ -s "$target/learning_curve.csv" ]]; then continue; fi
        record_status "$model" learning_curve started "fixed holdout; nested training traces 1,3,5"
        OMP_NUM_THREADS=1 "$python_bin" "$sim_root/tools/run_pr33824_ml_learning_curve.py" \
            --sim-root "$sim_root" --model-root "$results_root/models/$model" \
            --python "$python_bin" --out-dir "$target" --train-counts 1,3,5
        record_status "$model" learning_curve complete ""
    done
}

prepare_local_aic_config() {
    local source_config="$results_root/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"
    local local_config="$results_root/local_cpu_prediction/simulator-config.aic.json"
    local database_path="$results_root/aic_calibration/aic-systems-regular-context"
    [[ -s "$source_config" && -d "$database_path" ]] || {
        echo "AIC calibration data is incomplete under $results_root/aic_calibration" >&2; exit 1;
    }
    if [[ -n "${SGLANG_AIC_SIMULATOR_CONFIG:-}" ]]; then
        aic_simulator_config=$(realpath "$SGLANG_AIC_SIMULATOR_CONFIG")
        [[ -s "$aic_simulator_config" ]] || { echo "Missing SGLANG_AIC_SIMULATOR_CONFIG" >&2; exit 1; }
        return
    fi
    mkdir -p "$(dirname "$local_config")"
    if [[ ! -e "$local_config" ]]; then
        "$python_bin" - "$source_config" "$local_config" "$database_path" <<'PY'
import json
import sys
from pathlib import Path

source, output, database = map(Path, sys.argv[1:])
config = json.loads(source.read_text(encoding="utf-8"))
config["predictor"]["database_path"] = str(database.resolve())
output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
    fi
    "$python_bin" - "$local_config" "$database_path" <<'PY'
import json
import sys
from pathlib import Path

config_path, database = map(Path, sys.argv[1:])
actual = Path(json.loads(config_path.read_text(encoding="utf-8"))["predictor"]["database_path"]).resolve()
expected = database.resolve()
if actual != expected:
    raise SystemExit(f"local AIC config points to {actual}, expected {expected}")
PY
    aic_simulator_config=$(realpath "$local_config")
    if [[ ! -e "$results_root/local_cpu_prediction/run_metadata.txt" ]]; then
        {
            printf 'created_at=%s\nphase=local_cpu_prediction\n' "$(date -Is)"
            printf 'python=%s\nmodel_root=%s\n' "$python_bin" "$checkpoint_root"
            printf 'training_cases=%s\nholdout_cases=%s\n' "$train_cases" "$holdout_cases"
            printf 'training_repeats=%s\nholdout_repeats=%s\nprompt_multiplier=%s\n' \
                "$train_repeats" "$holdout_repeats" "$prompt_multiplier"
            printf 'aic_simulator_config=%s\naic_database=%s\n' "$aic_simulator_config" "$database_path"
            printf 'cpu_only=1\ntp_size=1\n' 
        } > "$results_root/local_cpu_prediction/run_metadata.txt"
    fi
}

run_remote_collection() {
    run_real_ml collect
    require_data_collection
    run_aic_calibration
    printf 'complete\n' > "$results_root/A100_REMOTE_COLLECTION_COMPLETE"
}

if [[ "$phase" == all || "$phase" == real-ml ]]; then
    run_real_ml
    run_learning_curves
fi
if [[ "$phase" == real-ml ]]; then exit 0; fi
if [[ "$phase" == collect ]]; then
    run_remote_collection
    echo "Remote GPU collection complete: $results_root"
    exit 0
fi
if [[ "$phase" == predict ]]; then
    require_data_collection
    prepare_local_aic_config
    run_real_ml predict
    run_learning_curves
    run_aic_predictor
    render_report
    echo "Local CPU prediction complete: $results_root"
    exit 0
fi
if [[ "$phase" == all || "$phase" == aic ]]; then
    run_aic_predictor
fi
if [[ "$phase" == aic ]]; then exit 0; fi
render_report
echo "Comprehensive predictor experiment complete: $results_root"
