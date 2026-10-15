#!/usr/bin/env bash
# Run the three-model serving matrix with trained MLTimePredictor bundles.
# Native A100 baseline files are copied from a completed reference root; this
# script never relaunches the real server when those files validate.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:-$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4}
calibration_root=${SGLANG_ML_CALIBRATION_ROOT:-}
real_reference_root=${SGLANG_ML_REAL_REFERENCE_ROOT:-$sim_root/test-results/pr33824_a100_dense3_sm80compat_matrix_20260910_164000}
results_root=${SGLANG_ML_MATRIX_ROOT:-$sim_root/test-results/pr33824_ml_dense3_matrix_$(date +%Y%m%d_%H%M%S)}
models_csv=${SGLANG_ML_MODELS:-Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct,Qwen3-32B}
resume=0
dry_run=0

usage() {
    cat <<'EOF'
Usage: bash tools/sglang-simulator/scripts/run_pr33824_ml_matrix.sh [options]

Uses per-model joblib bundles produced by run_pr33824_ml_calibration.sh.  The
CPU simulator runs the fixed serving traces; it reuses, rather than reruns,
the completed five-repeat native A100 baseline below --real-reference-root.

Options:
  --calibration-root PATH     Completed native ML calibration root (required)
  --real-reference-root PATH  Completed A100 real-vs-simulator baseline root
  --results-root PATH         New output root
  --models CSV                Dense checkpoint names (default: 3B,7B,32B)
  --resume                    Continue an incomplete root
  --dry-run                   Validate inputs and write metadata only
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --calibration-root) calibration_root=${2:?missing value}; shift 2 ;;
        --real-reference-root) real_reference_root=${2:?missing value}; shift 2 ;;
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$calibration_root" ]] || { echo "--calibration-root is required" >&2; exit 2; }
calibration_root=$(realpath "$calibration_root")
real_reference_root=$(realpath "$real_reference_root")
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
[[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
    echo "Missing isolated SM80 compatibility package: $compat_stage" >&2; exit 1;
}
[[ -f "$calibration_root/CALIBRATION_COMPLETE" ]] || {
    echo "Calibration root is not complete: $calibration_root" >&2; exit 1;
}
[[ -f "$real_reference_root/MATRIX_COMPLETE" ]] || {
    echo "Real baseline root is not complete: $real_reference_root" >&2; exit 1;
}
if [[ -e "$results_root" && "$resume" != 1 ]]; then
    echo "Refusing to overwrite existing output root: $results_root" >&2
    exit 1
fi
if [[ "$resume" == 1 && ! -d "$results_root" ]]; then
    echo "--resume requires an existing output root: $results_root" >&2
    exit 1
fi

mkdir -p "$results_root" "$results_root/per-model"
lock_file="$results_root/.ml-matrix.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another ML matrix launcher already owns $results_root" >&2
    exit 1
fi
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date -Is)" > "$results_root/.ml-matrix.owner"
exec > >(tee -a "$results_root/runner.log") 2>&1

if [[ "$resume" != 1 || ! -s "$results_root/run_metadata.txt" ]]; then
    {
        printf 'predictor=ml\naiconfigurator_used=false\n'
        printf 'calibration_root=%s\nreal_reference_root=%s\n' "$calibration_root" "$real_reference_root"
        printf 'models=%s\npython=%s\ncheckpoint_root=%s\n' "$models_csv" "$python_bin" "$checkpoint_root"
        printf 'kernel_compat_stage=%s\n' "$compat_stage"
        printf 'real_baseline_policy=reuse completed five-repeat native A100 measurements; do not relaunch GPU server\n'
        printf 'prediction_policy=two deterministic ML simulator repeats, then mean-per-case against each of five real repeats\n'
    } > "$results_root/run_metadata.txt"
fi

reference_model_index() {
    local model=$1 model_list index=0 candidate
    model_list=$(awk -F= '$1 == "models" {print substr($0, index($0, "=") + 1); exit}' "$real_reference_root/run_metadata.txt")
    IFS=',' read -r -a candidates <<< "$model_list"
    for candidate in "${candidates[@]}"; do
        candidate=${candidate//[[:space:]]/}
        if [[ "$candidate" == "$model" ]]; then
            printf '%s\n' "$index"
            return 0
        fi
        index=$((index + 1))
    done
    echo "Model is absent from reference metadata: $model" >&2
    return 1
}

is_real_complete() {
    [[ -s "$1/benchmark.jsonl" && -f "$1/status.txt" ]] && grep -qx ok "$1/status.txt"
}

verify_reference_real() {
    local model=$1 case repeat
    local -a cases=(latency_256x128_r1 service_1024x256_r4 saturation_1024x256_inf)
    for case in "${cases[@]}"; do
        for repeat in 1 2 3 4 5; do
            is_real_complete "$real_reference_root/results/$model/real/$case/repeat-$repeat" || return 1
        done
    done
    [[ -s "$real_reference_root/results/$model/real/max_total_tokens.txt" ]]
}

mem_fraction_for_reference() {
    local path="$real_reference_root/results/$1/real/launch-settings.txt"
    awk -F= '$1 == "mem_fraction_static" {print $2; exit}' "$path"
}

attention_for_reference() {
    local path="$real_reference_root/results/$1/real/launch-settings.txt"
    awk -F= '$1 == "attention_backend" {print $2; exit}' "$path"
}

graph_for_reference() {
    local path="$real_reference_root/results/$1/real/launch-settings.txt"
    awk -F= '$1 == "cuda_graph_config" {print substr($0, index($0, "=") + 1); exit}' "$path"
}

overlap_for_reference() {
    local path="$real_reference_root/results/$1/real/launch-settings.txt"
    awk -F= '$1 == "overlap_schedule_enabled" {print $2; exit}' "$path"
}

IFS=',' read -r -a models <<< "$models_csv"
failures=0
for model in "${models[@]}"; do
    model=${model//[[:space:]]/}
    [[ -n "$model" ]] || continue
    model_path="$checkpoint_root/$model"
    calibration_model_root="$calibration_root/models/$model"
    model_root="$results_root/per-model/$model"
    config_path="$calibration_model_root/simulator-config.json"
    if [[ ! -f "$model_path/config.json" || ! -f "$calibration_model_root/CALIBRATION_COMPLETE" || ! -f "$config_path" ]]; then
        echo "Incomplete checkpoint/calibration inputs for $model" >&2
        failures=1
        continue
    fi
    if ! jq -e '.predictor.name == "ml"' "$config_path" > /dev/null; then
        echo "Calibration config is not an ML predictor for $model: $config_path" >&2
        failures=1
        continue
    fi
    if ! verify_reference_real "$model"; then
        echo "Reference root lacks all fifteen successful real samples for $model" >&2
        failures=1
        continue
    fi
    model_index=$(reference_model_index "$model") || { failures=1; continue; }
    mem_fraction=$(mem_fraction_for_reference "$model")
    attention_backend=$(attention_for_reference "$model")
    graph_config=$(graph_for_reference "$model")
    overlap_enabled=$(overlap_for_reference "$model")
    if [[ -z "$mem_fraction" || -z "$attention_backend" || -z "$graph_config" || -z "$overlap_enabled" ]]; then
        echo "Reference launch settings are incomplete for $model" >&2
        failures=1
        continue
    fi
    mkdir -p "$model_root/results/$model"
    if [[ ! -d "$model_root/results/$model/real" ]]; then
        cp -a "$real_reference_root/results/$model/real" "$model_root/results/$model/"
    fi
    if ! is_real_complete "$model_root/results/$model/real/latency_256x128_r1/repeat-1"; then
        echo "Copied real baseline did not validate for $model" >&2
        failures=1
        continue
    fi
    {
        printf 'model=%s\nmodel_index_from_reference=%s\n' "$model" "$model_index"
        printf 'calibration_config=%s\nmodel_bundle=%s\n' "$config_path" "$(jq -r '.predictor.database_path' "$config_path")"
        printf 'real_reference=%s\nmem_fraction_static=%s\nattention_backend=%s\n' "$real_reference_root" "$mem_fraction" "$attention_backend"
        printf 'cuda_graph_config=%s\noverlap_schedule_enabled=%s\n' "$graph_config" "$overlap_enabled"
    } > "$model_root/reference-and-config.txt"
    if [[ "$dry_run" == 1 ]]; then
        echo "[dry-run] validated $model"
        continue
    fi
    if [[ -f "$model_root/MATRIX_COMPLETE" ]]; then
        echo "[matrix] $model already complete"
        continue
    fi
    matrix_args=(
        --results-root "$model_root"
        --resume
        --models "$model"
        --repeats 5
        --prediction-repeat-policy two-then-five
        --sim-config "$config_path"
        --real-attention-backend "$attention_backend"
        --real-cuda-graph-config "$graph_config"
        --real-mem-fraction-static "$mem_fraction"
    )
    if [[ "$overlap_enabled" == 1 ]]; then
        matrix_args+=(--real-enable-overlap-schedule)
    fi
    if SGLANG_BENCH_PYTHON="$python_bin" \
        SGLANG_CHECKPOINT_ROOT="$checkpoint_root" \
        SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
        SGLANG_MODEL_INDEX_OFFSET="$model_index" \
        SGLANG_SKIP_MATRIX_RENDER=1 \
        bash "$script_dir/run_pr33824_a100_matrix.sh" "${matrix_args[@]}"; then
        [[ -f "$model_root/MATRIX_COMPLETE" ]] || { echo "Missing completion marker: $model_root" >&2; failures=1; }
    else
        echo "[matrix] ML simulator failed for $model" >&2
        failures=1
    fi
done

if [[ "$dry_run" == 1 ]]; then
    touch "$results_root/ML_MATRIX_DRY_RUN_COMPLETE"
    echo "ML matrix dry-run complete: $results_root"
    exit "$failures"
fi
if [[ "$failures" == 0 ]]; then
    "$python_bin" "$script_dir/summarize_pr33824_ml_experiment.py" \
        --experiment-root "$results_root" \
        --calibration-root "$calibration_root" \
        --real-reference-root "$real_reference_root" \
        --models "$models_csv" \
        --documentation-dir "$sim_root/docs" > "$results_root/summary.log" 2>&1
    touch "$results_root/ML_MATRIX_COMPLETE"
    echo "ML matrix complete: $results_root"
else
    touch "$results_root/ML_MATRIX_COMPLETED_WITH_FAILURES"
    echo "ML matrix ended with failures: $results_root" >&2
fi
exit "$failures"
