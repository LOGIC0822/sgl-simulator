#!/usr/bin/env bash
# Collect independent real saturation traces for both training and holdout,
# then train and evaluate each per-model HGB predictor. This is the data layer
# of the standardized all-model ML/AIC comparison; AIC consumes only the
# completed training traces in a later phase.

set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
export PYTHONNOUSERSITE=1
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
if [[ ${SGLANG_KERNEL_COMPAT_STAGE+x} ]]; then
    compat_stage=$SGLANG_KERNEL_COMPAT_STAGE
else
    compat_stage="$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4"
fi
fa2_compat_stage=${SGLANG_FA2_COMPAT_STAGE:-}
results_root=${SGLANG_SAME_DISTRIBUTION_RESULTS_ROOT:-"$sim_root/test-results/pr33824_comprehensive_all13_predictors_$(date +%Y%m%d_%H%M%S)"}
models_csv=${SGLANG_SAME_DISTRIBUTION_MODELS:-Qwen3-0.6B,Qwen2.5-3B-Instruct,llama-2-7b,Mistral-7B-Instruct-v0.2,Qwen2.5-7B-Instruct,Qwen3-8B,Qwen2.5-14B-Instruct,Qwen2.5-14B-Instruct-1M,Qwen3-14B,Meta-Llama-3.1-70B-Instruct-AWQ,Qwen3-30B-A3B,Qwen3-30B-A3B-Instruct-2507,Qwen3-32B}
train_repeats=${SGLANG_SAME_DISTRIBUTION_TRAIN_REPEATS:-5}
holdout_repeats=${SGLANG_SAME_DISTRIBUTION_HOLDOUT_REPEATS:-5}
train_seeds_csv=${SGLANG_SAME_DISTRIBUTION_TRAIN_SEEDS:-}
train_seed_base=${SGLANG_SAME_DISTRIBUTION_TRAIN_SEED_BASE:-20260918}
holdout_seeds_csv=${SGLANG_SAME_DISTRIBUTION_HOLDOUT_SEEDS:-}
holdout_seed_base=${SGLANG_SAME_DISTRIBUTION_HOLDOUT_SEED_BASE:-20261018}
prompt_multiplier=${SGLANG_SAME_DISTRIBUTION_PROMPT_MULTIPLIER:-2}
training_cases=${SGLANG_SAME_DISTRIBUTION_TRAIN_CASES:-saturation_1024x256_inf}
holdout_cases=${SGLANG_SAME_DISTRIBUTION_HOLDOUT_CASES:-$training_cases}
collection_preset=${SGLANG_SAME_DISTRIBUTION_COLLECTION_PRESET:-dense}
experiment_pythonpath="$sim_root/src:$sglang_root/python${SGLANG_EXPERIMENT_EXTRA_PYTHONPATH:+:$SGLANG_EXPERIMENT_EXTRA_PYTHONPATH}"
real_port=${SGLANG_SAME_DISTRIBUTION_REAL_PORT:-30227}
simulator_port=${SGLANG_SAME_DISTRIBUTION_SIMULATOR_PORT:-30228}
resume=0
dry_run=0
phase=all

usage() {
    cat <<'EOF'
Usage: bash tools/sglang-simulator/scripts/run_pr33824_data_tools_same_distribution_matrix.sh [options]

For each selected checkpoint, collect N new real training and M new real
holdout `saturation_1024x256_inf` traces, train an HGB bundle only on the
training rows, then score/replay the independent holdout traces. With the
defaults, every model has 5 x (32 x 2) = 320 training requests and the same
number of holdout requests.

Options:
  --phase NAME         collect, predict, or all (default: all).
  --results-root PATH  New immutable test-results root.
  --models CSV         Checkpoint directories below /mnt/data/models.
  --train-repeats N    Independent real training traces per model (default: 5).
  --holdout-repeats N  Independent real holdout traces per model (default: 5).
  --prompt-multiplier N
                       Multiplier applied to the 32-request saturation case
                       in every trace (default: 2, therefore 64 requests).
  --resume             Continue only stages with no completion marker.
  --dry-run            Print model, memory, and held-out-source plan.
  -h, --help           Show this help.

Environment overrides include SGLANG_SAME_DISTRIBUTION_TRAIN_SEEDS and
SGLANG_SAME_DISTRIBUTION_HOLDOUT_SEEDS (each a CSV with exactly its repeat
count of unique integer seeds), plus the normal hbf-cxl/kernel paths.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --train-repeats) train_repeats=${2:?missing value}; shift 2 ;;
        --holdout-repeats) holdout_repeats=${2:?missing value}; shift 2 ;;
        --prompt-multiplier) prompt_multiplier=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        --phase) phase=${2:?missing value}; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$train_repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--train-repeats must be positive" >&2; exit 2; }
[[ "$holdout_repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--holdout-repeats must be positive" >&2; exit 2; }
[[ "$prompt_multiplier" =~ ^[1-9][0-9]*$ ]] || { echo "--prompt-multiplier must be positive" >&2; exit 2; }
[[ "$phase" == all || "$phase" == collect || "$phase" == predict ]] || {
    echo "--phase must be all, collect, or predict" >&2; exit 2;
}
[[ "$train_seed_base" =~ ^[0-9]+$ ]] || { echo "train seed base must be an integer" >&2; exit 2; }
[[ "$holdout_seed_base" =~ ^[0-9]+$ ]] || { echo "holdout seed base must be an integer" >&2; exit 2; }
[[ "$real_port" =~ ^[1-9][0-9]*$ && "$simulator_port" =~ ^[1-9][0-9]*$ ]] || {
    echo "ports must be positive integers" >&2; exit 2;
}
[[ "$real_port" != "$simulator_port" ]] || { echo "real and simulator ports must differ" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
if [[ -n "$compat_stage" ]]; then compat_stage=$(realpath "$compat_stage"); fi
if [[ "$phase" != predict ]]; then
    [[ -n "$compat_stage" && -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
        echo "Missing isolated SM80 compatibility stage: ${compat_stage:-<unset>}" >&2; exit 1;
    }
fi

memory_fraction_for_model() {
    local pair name value
    local -a overrides
    IFS=',' read -r -a overrides <<< "${SGLANG_SAME_DISTRIBUTION_MEMORY_OVERRIDES:-}"
    for pair in "${overrides[@]}"; do
        name=${pair%%=*}
        value=${pair#*=}
        if [[ "$name" == "$1" ]]; then
            "$python_bin" -c 'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and 0 < value < 1; print(value)' "$value"
            return
        fi
    done
    case "$1" in
        Qwen3-0.6B) printf '0.20\n' ;;
        Qwen2.5-3B-Instruct) printf '0.30\n' ;;
        Mistral-7B-Instruct-v0.2|Qwen2.5-7B-Instruct) printf '0.35\n' ;;
        llama-2-7b) printf '0.72\n' ;;
        Qwen3-8B) printf '0.42\n' ;;
        Qwen2.5-14B-Instruct|Qwen2.5-14B-Instruct-1M|Qwen3-14B) printf '0.60\n' ;;
        Meta-Llama-3.1-70B-Instruct-AWQ) printf '0.80\n' ;;
        Qwen3-30B-A3B|Qwen3-30B-A3B-Instruct-2507) printf '0.90\n' ;;
        Qwen3-32B) printf '0.93\n' ;;
        *) echo "No memory plan for model: $1" >&2; return 1 ;;
    esac
}

attention_backend_for_model() {
    case "$1" in
        Qwen2.5-14B-Instruct-1M) printf 'dual_chunk_flash_attn\n' ;;
        *) printf 'triton\n' ;;
    esac
}

moe_jit_align_shim_for_model() {
    case "$1" in
        Qwen3-30B-A3B|Qwen3-30B-A3B-Instruct-2507) printf '%s\n' "$script_dir/collector_shims" ;;
        *) printf '\n' ;;
    esac
}

runtime_shim_for_model() {
    case "$1" in
        Qwen2.5-14B-Instruct-1M|Qwen3-30B-A3B|Qwen3-30B-A3B-Instruct-2507)
            printf '%s\n' "$script_dir/collector_shims" ;;
        *) printf '\n' ;;
    esac
}

fa2_stage_for_model() {
    case "$1" in
        Qwen2.5-14B-Instruct-1M)
            [[ -n "$fa2_compat_stage" && -f "$fa2_compat_stage/flash_attn/__init__.py" ]] || {
                echo "1M checkpoint requires SGLANG_FA2_COMPAT_STAGE" >&2; return 1;
            }
            printf '%s\n' "$fa2_compat_stage"
            ;;
        *) printf '\n' ;;
    esac
}

page_size_for_model() {
    case "$1" in
        Qwen2.5-14B-Instruct-1M) printf '256\n' ;;
        *) printf '\n' ;;
    esac
}

is_complete() {
    [[ -f "$1/$2" ]] && grep -qx 'complete' "$1/$2"
}

request_budget() {
    local names=$1 count=0 name
    local -a names_array
    declare -A seen=()
    IFS=',' read -r -a names_array <<< "$names"
    for name in "${names_array[@]}"; do
        [[ -n "$name" && -z "${seen[$name]:-}" ]] || { echo "Empty or duplicate workload: $name" >&2; return 1; }
        seen[$name]=1
        case "$name" in
            latency_256x128_r1|latency_128x64_r1) count=$((count + 16)) ;;
            service_1024x256_r4|saturation_1024x256_inf|saturation_1536x384_inf) count=$((count + 32)) ;;
            *) echo "Unsupported workload: $name" >&2; return 1 ;;
        esac
    done
    printf '%s\n' "$((count * prompt_multiplier))"
}
train_requests_per_repeat=$(request_budget "$training_cases")
holdout_requests_per_repeat=$(request_budget "$holdout_cases")

IFS=',' read -r -a models <<< "$models_csv"
filtered_models=()
for model in "${models[@]}"; do
    model=${model//[[:space:]]/}
    [[ -n "$model" ]] || continue
    [[ -d "$checkpoint_root/$model" ]] || { echo "Missing model directory: $checkpoint_root/$model" >&2; exit 1; }
    memory_fraction_for_model "$model" > /dev/null
    filtered_models+=("$model")
done
(( ${#filtered_models[@]} > 0 )) || { echo "No models selected" >&2; exit 2; }

train_seeds=()
if [[ -n "$train_seeds_csv" ]]; then
    IFS=',' read -r -a train_seeds <<< "$train_seeds_csv"
    (( ${#train_seeds[@]} == train_repeats )) || {
        echo "SGLANG_SAME_DISTRIBUTION_TRAIN_SEEDS must contain $train_repeats values" >&2; exit 2;
    }
else
    for index in $(seq 0 $((train_repeats - 1))); do
        train_seeds+=("$((train_seed_base + index))")
    done
fi
declare -A seen_seeds=()
for seed in "${train_seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid train seed: $seed" >&2; exit 2; }
    [[ -z "${seen_seeds[$seed]:-}" ]] || { echo "Duplicate train seed: $seed" >&2; exit 2; }
    seen_seeds[$seed]=1
done

holdout_seeds=()
if [[ -n "$holdout_seeds_csv" ]]; then
    IFS=',' read -r -a holdout_seeds <<< "$holdout_seeds_csv"
    (( ${#holdout_seeds[@]} == holdout_repeats )) || {
        echo "SGLANG_SAME_DISTRIBUTION_HOLDOUT_SEEDS must contain $holdout_repeats values" >&2; exit 2;
    }
else
    for index in $(seq 0 $((holdout_repeats - 1))); do
        holdout_seeds+=("$((holdout_seed_base + index))")
    done
fi
for seed in "${holdout_seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid holdout seed: $seed" >&2; exit 2; }
    [[ -z "${seen_seeds[$seed]:-}" ]] || { echo "Train and holdout seeds overlap: $seed" >&2; exit 2; }
    seen_seeds[$seed]=1
done

if [[ "$dry_run" == 1 ]]; then
    printf 'model\tmem_fraction_static\ttraining_requests\tholdout_requests\n'
    for model in "${filtered_models[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "$model" "$(memory_fraction_for_model "$model")" \
            "$((train_repeats * train_requests_per_repeat))" "$((holdout_repeats * holdout_requests_per_repeat))"
    done
    printf 'training_cases=%s\nholdout_cases=%s\nprompt_multiplier=%s\ntraining_seeds=%s\nholdout_seeds=%s\n' \
        "$training_cases" "$holdout_cases" \
        "$prompt_multiplier" "$(IFS=,; echo "${train_seeds[*]}")" "$(IFS=,; echo "${holdout_seeds[*]}")"
    exit 0
fi

if [[ -e "$results_root" && "$resume" != 1 ]]; then
    echo "Refusing to overwrite existing output root: $results_root" >&2; exit 1
fi
if [[ "$resume" == 1 && ! -d "$results_root" ]]; then
    echo "--resume requires an existing output root: $results_root" >&2; exit 1
fi

mkdir -p "$results_root/models" "$results_root/hardware"
lock_file="$results_root/.same-distribution.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another same-distribution launcher owns $results_root" >&2; exit 1
fi
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date -Is)" > "$results_root/.same-distribution.owner"
exec >> "$results_root/runner.log" 2>&1

status_path="$results_root/status.tsv"
if [[ ! -f "$status_path" ]]; then
    printf 'timestamp\tmodel\tstage\tstate\tdetail\n' > "$status_path"
fi

record_status() {
    local model=$1 stage=$2 state=$3 detail=${4:-}
    detail=${detail//$'\t'/ }
    detail=${detail//$'\n'/ }
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$model" "$stage" "$state" "$detail" >> "$status_path"
    printf '[%s] %s %s: %s %s\n' "$(date -Is)" "$model" "$stage" "$state" "$detail"
}

if [[ "$resume" != 1 || ! -s "$results_root/run_metadata.txt" ]]; then
    {
        printf 'created_at=%s\n' "$(date -Is)"
        printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
        printf 'source_path=%s\npython=%s\n' "$sglang_root" "$python_bin"
        printf 'models=%s\n' "$(IFS=,; echo "${filtered_models[*]}")"
        printf 'protocol=same_workload_independent_train_holdout\n'
        printf 'training_case=%s\ntraining_real_repeats=%s\ntraining_seeds=%s\n' \
            "$training_cases" "$train_repeats" "$(IFS=,; echo "${train_seeds[*]}")"
        printf 'training_fit_seed=%s\n' "$train_seed_base"
        printf 'heldout_case=%s\nheldout_real_repeats=%s\nheldout_seeds=%s\n' \
            "$holdout_cases" "$holdout_repeats" "$(IFS=,; echo "${holdout_seeds[*]}")"
        printf 'requests_per_training_repeat=%s\nrequests_per_holdout_repeat=%s\ntraining_requests_total=%s\nheldout_requests_total=%s\nprompt_multiplier=%s\n' \
            "$train_requests_per_repeat" "$holdout_requests_per_repeat" "$((train_repeats * train_requests_per_repeat))" \
            "$((holdout_repeats * holdout_requests_per_repeat))" "$prompt_multiplier"
        printf 'leakage_policy=training and holdout ScheduleBatch rows are separately collected with disjoint seeds; holdout rows are never used for ML fitting or AIC trace-target selection\n'
        printf 'phase=%s\nhardware_policy=one GPU collector at a time; collect phase leaves ML/AIC prediction for a later CPU phase\n' "$phase"
        printf 'kernel_policy=isolated_SM80_AOT; SGLANG_FORCE_FUSED_OP_BACKEND=unset\n'
        printf 'kernel_compat_stage=%s\nfa2_compat_stage=%s\n' "$compat_stage" "${fa2_compat_stage:-none}"
        printf 'collector_source=https://github.com/littlefatfat/sglang/tree/simulator-data-tools-0914\n'
        printf 'collector_commit=9cfddcedc9d1e00885c28db75a45513eddbb6271\n'
        printf 'scheduler_settings=attention=per-model (triton except 1M dual_chunk_flash_attn); 1M_page_size=256_for_FA2_paged_KV; overlap=disabled; cuda_graph=disabled; context_length=2048; max_running_requests=32; chunked_prefill_size=512; random_range_ratio=0\n'
    } > "$results_root/run_metadata.txt"
    printf 'model\tmem_fraction_static\tmodel_path\ttraining_requests\tholdout_requests\n' > "$results_root/model_manifest.tsv"
    for model in "${filtered_models[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\n' "$model" "$(memory_fraction_for_model "$model")" \
                "$checkpoint_root/$model" "$((train_repeats * train_requests_per_repeat))" \
                "$((holdout_repeats * holdout_requests_per_repeat))" >> "$results_root/model_manifest.tsv"
    done
    nvidia-smi -q > "$results_root/hardware/nvidia-smi-before.txt" 2>&1 || true
fi
if [[ ! -s "$results_root/model_manifest.tsv" ]]; then
    printf 'model\tmem_fraction_static\tmodel_path\ttraining_requests\tholdout_requests\n' > "$results_root/model_manifest.tsv"
    for model in "${filtered_models[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\n' "$model" "$(memory_fraction_for_model "$model")" \
            "$checkpoint_root/$model" "$((train_repeats * train_requests_per_repeat))" \
            "$((holdout_repeats * holdout_requests_per_repeat))" >> "$results_root/model_manifest.tsv"
    done
fi

if [[ "$phase" == predict ]] && ! is_complete "$results_root" BASELINE_DATA_ALL_MODELS_COMPLETE; then
    echo "Prediction phase requires completed remote data collection: $results_root" >&2
    exit 1
fi

require_clean_or_complete() {
    local path=$1 marker=$2
    if is_complete "$path" "$marker"; then
        return 0
    fi
    if [[ -e "$path" ]]; then
        echo "Incomplete stage already exists: $path" >&2
        return 1
    fi
    return 2
}

collect_real_repeat() {
    local model=$1 target=$2 mem_fraction=$3 seed=$4 role=$5 state
    local requested_cases="$training_cases"
    [[ "$role" != holdout ]] || requested_cases="$holdout_cases"
    local attention_backend moe_shim runtime_shim model_fa2 page_size
    local minimum_free=0
    case "$model" in
        Qwen3-0.6B) minimum_free=6144 ;;
        Qwen2.5-3B-Instruct) minimum_free=12288 ;;
        Mistral-7B-Instruct-v0.2|Qwen2.5-7B-Instruct) minimum_free=57344 ;;
        llama-2-7b) minimum_free=57344 ;;
        Qwen3-8B) minimum_free=57344 ;;
        Qwen2.5-14B-Instruct|Qwen2.5-14B-Instruct-1M|Qwen3-14B) minimum_free=57344 ;;
        Meta-Llama-3.1-70B-Instruct-AWQ) minimum_free=57344 ;;
        Qwen3-30B-A3B|Qwen3-30B-A3B-Instruct-2507|Qwen3-32B) minimum_free=73728 ;;
    esac
    attention_backend=$(attention_backend_for_model "$model")
    moe_shim=$(moe_jit_align_shim_for_model "$model")
    runtime_shim=$(runtime_shim_for_model "$model")
    page_size=$(page_size_for_model "$model")
    model_fa2=$(fa2_stage_for_model "$model") || return 1
    if require_clean_or_complete "$target" BASELINE_COLLECTION_COMPLETE; then
        record_status "$model" "$role-$(basename "$target")" skipped "existing completion marker"
        return 0
    else
        state=$?
    fi
    [[ "$state" == 2 ]] || return 1
    record_status "$model" "$role-$(basename "$target")" started \
        "cases=$requested_cases seed=$seed multiplier=$prompt_multiplier"
    if ! SGLANG_COLLECT_MODEL="$model" \
        SGLANG_COLLECT_MODEL_PATH="$checkpoint_root/$model" \
        SGLANG_RESULTS_ROOT="$target" \
        SGLANG_COLLECTION_PRESET="$collection_preset" \
        SGLANG_COLLECTION_CASES="$requested_cases" \
        SGLANG_TRAIN_AFTER_COLLECT=0 \
        SGLANG_MEM_FRACTION_STATIC="$mem_fraction" \
        SGLANG_GPU_MINIMUM_FREE_MIB="$minimum_free" \
        SGLANG_WARMUP_REQUESTS=4 \
        SGLANG_TRAIN_FORWARD_MODES=1,2 \
        SGLANG_COLLECTION_PROMPT_MULTIPLIER="$prompt_multiplier" \
        SGLANG_BENCH_SEED="$seed" \
        SGLANG_BENCH_PYTHON="$python_bin" \
        SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
        SGLANG_ATTENTION_BACKEND="$attention_backend" \
        SGLANG_MOE_JIT_ALIGN_SHIM="$moe_shim" \
        SGLANG_RUNTIME_SHIM_DIR="$runtime_shim" \
        SGLANG_FA2_COMPAT_STAGE="$model_fa2" \
        SGLANG_PAGE_SIZE="$page_size" \
        SGLANG_REAL_PORT="$real_port" \
        bash "$script_dir/run_pr33824_data_tools_baseline.sh"; then
        record_status "$model" "$role-$(basename "$target")" failed "collector returned non-zero; see $target"
        return 1
    fi
    is_complete "$target" BASELINE_COLLECTION_COMPLETE || {
        record_status "$model" "$role-$(basename "$target")" failed "completion marker missing"; return 1;
    }
    rg -qx "prompt_multiplier=$prompt_multiplier" "$target/run_metadata.txt" || {
        record_status "$model" "$role-$(basename "$target")" failed "prompt multiplier metadata mismatch"; return 1;
    }
    record_status "$model" "$role-$(basename "$target")" complete ""
}

train_bundle() {
    local model=$1 model_root=$2 training_root="$model_root/calibration/training"
    local safe_model=${model//[^A-Za-z0-9._-]/_}
    local -a command=("$python_bin" "$sim_root/tools/train_latency_model.py")
    local repeat
    if [[ -f "$training_root/${safe_model}_hgbmono_p50.pkl" && -f "$training_root/${safe_model}_hgbmono_l2.pkl" ]]; then
        record_status "$model" training skipped "existing model bundles"
        return 0
    fi
    [[ ! -e "$training_root" ]] || { echo "Incomplete training directory: $training_root" >&2; return 1; }
    mkdir -p "$training_root"
    for repeat in $(seq 1 "$train_repeats"); do
        command+=(--data-root "$model_root/calibration/real/repeat-$repeat/cases")
    done
    command+=(--out-dir "$training_root" --model-prefix "${safe_model}_hgbmono" --loss both --forward-modes "1,2" --seed "$train_seed_base" --test-fraction 0.2)
    printf '%q ' env "PYTHONPATH=$sim_root/src:$sglang_root/python" "${command[@]}" > "$model_root/calibration/train.command.txt"
    printf '\n' >> "$model_root/calibration/train.command.txt"
    record_status "$model" training started "HGB on $train_repeats independent saturation traces"
    if ! PYTHONPATH="$experiment_pythonpath" "${command[@]}" > "$model_root/calibration/train.log" 2>&1; then
        record_status "$model" training failed "trainer returned non-zero; see calibration/train.log"
        return 1
    fi
    [[ -f "$training_root/${safe_model}_hgbmono_p50.pkl" && -f "$training_root/${safe_model}_hgbmono_l2.pkl" ]] || {
        record_status "$model" training failed "bundle missing"; return 1;
    }
    record_status "$model" training complete ""
}

verify_holdout() {
    local model=$1 model_root=$2 real_root="$model_root/holdout/real" repeat
    for repeat in $(seq 1 "$holdout_repeats"); do
        is_complete "$real_root/repeat-$repeat" BASELINE_COLLECTION_COMPLETE || {
            echo "Held-out repeat-$repeat is incomplete: $real_root" >&2; return 1;
        }
        rg -Fxq "requested_cases=$holdout_cases" "$real_root/repeat-$repeat/run_metadata.txt" || {
            echo "Held-out repeat-$repeat workload mismatch" >&2; return 1;
        }
        rg -qx "prompt_multiplier=$prompt_multiplier" "$real_root/repeat-$repeat/run_metadata.txt" || {
            echo "Held-out repeat-$repeat has an unexpected request count multiplier" >&2; return 1;
        }
    done
}

score_holdout() {
    local model=$1 model_root=$2 bundle=$3 score_root="$model_root/holdout/ml_teacher_forced"
    if [[ -s "$score_root/summary.csv" ]]; then
        record_status "$model" teacher_forced skipped "existing summary.csv"
        return 0
    fi
    [[ ! -e "$score_root" ]] || { echo "Incomplete score directory: $score_root" >&2; return 1; }
    record_status "$model" teacher_forced started "independent real saturation Scheduler batches"
    if ! PYTHONPATH="$experiment_pythonpath" "$python_bin" \
        "$sim_root/tools/score_pr33824_latency_model.py" \
        --data-root "$model_root/holdout/real" --model-bundle "$bundle" --out-dir "$score_root" --forward-modes 1,2; then
        record_status "$model" teacher_forced failed "scoring command returned non-zero"
        return 1
    fi
    record_status "$model" teacher_forced complete ""
}

audit_distribution() {
    local model=$1 model_root=$2 output="$model_root/holdout/distribution_audit.csv"
    local -a args=()
    local repeat
    if [[ -s "$output" ]]; then
        record_status "$model" distribution_audit skipped "existing CSV"
        return 0
    fi
    for repeat in $(seq 1 "$train_repeats"); do
        args+=(--train-root "$model_root/calibration/real/repeat-$repeat/cases")
    done
    for repeat in $(seq 1 "$holdout_repeats"); do
        args+=(--test-root "$model_root/holdout/real/repeat-$repeat/cases")
    done
    record_status "$model" distribution_audit started "feature support between independent saturation traces"
    if ! "$python_bin" "$sim_root/tools/analyze_pr33824_ml_distribution.py" "${args[@]}" \
        --out "$output" --markdown "$model_root/holdout/distribution_audit.md"; then
        record_status "$model" distribution_audit failed "audit command returned non-zero"
        return 1
    fi
    record_status "$model" distribution_audit complete ""
}

replay_holdout_repeat() {
    local model=$1 real_root=$2 replay_root=$3 bundle=$4 state
    if require_clean_or_complete "$replay_root" ML_REPLAY_COMPLETE; then
        record_status "$model" "replay-$(basename "$replay_root")" skipped "existing completion marker"
        return 0
    else
        state=$?
    fi
    [[ "$state" == 2 ]] || return 1
    record_status "$model" "replay-$(basename "$replay_root")" started "CPU ML replay of held-out trace"
    if ! SGLANG_COLLECTION_ROOT="$real_root" \
        SGLANG_ML_BUNDLE="$bundle" \
        SGLANG_REPLAY_CASES="$holdout_cases" \
        SGLANG_REPLAY_RESULTS_ROOT="$replay_root" \
        SGLANG_REPLAY_MODEL_PATH="$checkpoint_root/$model" \
        SGLANG_BENCH_PYTHON="$python_bin" \
        SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
        SGLANG_SIMULATOR_PORT="$simulator_port" \
        bash "$script_dir/run_pr33824_data_tools_ml_replay.sh"; then
        record_status "$model" "replay-$(basename "$replay_root")" failed "replay launcher returned non-zero; see $replay_root"
        return 1
    fi
    is_complete "$replay_root" ML_REPLAY_COMPLETE || {
        record_status "$model" "replay-$(basename "$replay_root")" failed "completion marker missing"; return 1;
    }
    record_status "$model" "replay-$(basename "$replay_root")" complete ""
}

compare_e2e() {
    local model=$1 model_root=$2 e2e_root="$model_root/holdout/ml_e2e_comparison"
    local -a args=()
    local repeat
    if [[ -s "$e2e_root/summary.csv" ]]; then
        record_status "$model" e2e_comparison skipped "existing summary.csv"
        return 0
    fi
    [[ ! -e "$e2e_root" ]] || { echo "Incomplete E2E directory: $e2e_root" >&2; return 1; }
    for repeat in $(seq 1 "$holdout_repeats"); do
        args+=(--real-root "$model_root/holdout/real/repeat-$repeat")
        args+=(--replay-root "$model_root/holdout/ml_simulator/repeat-$repeat")
    done
    record_status "$model" e2e_comparison started "paired held-out real/replay summaries"
    if ! PYTHONPATH="$experiment_pythonpath" "$python_bin" \
        "$sim_root/tools/summarize_pr33824_data_tools_replay.py" "${args[@]}" \
        --out-dir "$e2e_root" --teacher-forced-summary "$model_root/holdout/ml_teacher_forced/summary.csv"; then
        record_status "$model" e2e_comparison failed "summarizer returned non-zero"
        return 1
    fi
    record_status "$model" e2e_comparison complete ""
}

run_model() {
    local model=$1 mem_fraction=$2 model_root="$results_root/models/$model"
    local safe_model=${model//[^A-Za-z0-9._-]/_}
    local bundle="$model_root/calibration/training/${safe_model}_hgbmono_p50.pkl"
    local repeat seed
    mkdir -p "$model_root"
    if is_complete "$model_root" REAL_ML_COMPLETE; then
        record_status "$model" model skipped "existing completion marker"
        return 0
    fi
    if [[ "$phase" != predict ]]; then
    {
        printf 'model=%s\nmodel_path=%s\nmem_fraction_static=%s\n' "$model" "$checkpoint_root/$model" "$mem_fraction"
        printf 'training_case=%s\ntraining_seeds=%s\ntraining_requests=%s\n' \
            "$training_cases" "$(IFS=,; echo "${train_seeds[*]}")" "$((train_repeats * train_requests_per_repeat))"
        printf 'heldout_case=%s\nheldout_seeds=%s\nheldout_requests=%s\n' \
            "$holdout_cases" "$(IFS=,; echo "${holdout_seeds[*]}")" "$((holdout_repeats * holdout_requests_per_repeat))"
        printf 'attention_backend=%s\nmoe_jit_align_shim=%s\nruntime_shim=%s\nfa2_compat_stage=%s\npage_size=%s\n' \
            "$(attention_backend_for_model "$model")" "$(moe_jit_align_shim_for_model "$model")" \
            "$(runtime_shim_for_model "$model")" "$(fa2_stage_for_model "$model")" "$(page_size_for_model "$model")"
    } > "$model_root/model_settings.txt"
    for repeat in $(seq 1 "$train_repeats"); do
        seed=${train_seeds[$((repeat - 1))]}
        collect_real_repeat "$model" "$model_root/calibration/real/repeat-$repeat" "$mem_fraction" "$seed" training || return 1
    done
    for repeat in $(seq 1 "$holdout_repeats"); do
        seed=${holdout_seeds[$((repeat - 1))]}
        collect_real_repeat "$model" "$model_root/holdout/real/repeat-$repeat" "$mem_fraction" "$seed" holdout || return 1
    done
    verify_holdout "$model" "$model_root" || return 1
        printf 'complete\n' > "$model_root/BASELINE_DATA_COMPLETE"
        record_status "$model" data_collection complete "training and independent holdout traces"
    else
        is_complete "$model_root" BASELINE_DATA_COMPLETE || {
            echo "Model data collection is incomplete: $model_root" >&2; return 1;
        }
        verify_holdout "$model" "$model_root" || return 1
    fi
    [[ "$phase" == collect ]] && return 0
    train_bundle "$model" "$model_root" || return 1
    score_holdout "$model" "$model_root" "$bundle" || return 1
    audit_distribution "$model" "$model_root" || return 1
    for repeat in $(seq 1 "$holdout_repeats"); do
        replay_holdout_repeat "$model" "$model_root/holdout/real/repeat-$repeat" \
            "$model_root/holdout/ml_simulator/repeat-$repeat" "$bundle" || return 1
    done
    compare_e2e "$model" "$model_root" || return 1
    printf 'complete\n' > "$model_root/REAL_ML_COMPLETE"
    record_status "$model" model complete ""
}

failures=0
for model in "${filtered_models[@]}"; do
    if ! run_model "$model" "$(memory_fraction_for_model "$model")"; then
        failures=1
        record_status "$model" model failed "see model-local logs"
    fi
done

if (( failures )); then
    if [[ "$phase" == collect ]]; then
        printf 'completed-with-failures\n' > "$results_root/BASELINE_DATA_COMPLETED_WITH_FAILURES"
    else
        printf 'completed-with-failures\n' > "$results_root/REAL_ML_COMPLETED_WITH_FAILURES"
    fi
    record_status matrix complete_with_failures "fix incomplete stages and resume this root"
    exit 1
fi

if [[ "$phase" == collect ]]; then
    printf 'complete\n' > "$results_root/BASELINE_DATA_ALL_MODELS_COMPLETE"
    record_status matrix data_collection_complete "real training and independent holdout traces"
    exit 0
fi

printf 'complete\n' > "$results_root/BASELINE_DATA_ALL_MODELS_COMPLETE"
printf 'complete\n' > "$results_root/REAL_ML_ALL_MODELS_COMPLETE"
record_status matrix real_ml_complete "AIC calibration, AIC replay, and the final joint report run in the comprehensive orchestrator"
