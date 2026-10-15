#!/usr/bin/env bash
# Run the native-collector ML experiment sequentially for every local model.
#
# Each checkpoint has its own calibration data, HGB bundle, five real held-out
# runs, five CPU replays, and paired summaries.  A failed checkpoint is kept in
# place with its logs and does not invalidate successfully completed models.

set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:-"$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4"}
results_root=${SGLANG_ALL_MODELS_RESULTS_ROOT:-"$sim_root/test-results/pr33824_data_tools_all13_$(date +%Y%m%d_%H%M%S)"}
models_csv=${SGLANG_ALL_MODELS:-Qwen3-0.6B,Qwen2.5-3B-Instruct,llama-2-7b,Mistral-7B-Instruct-v0.2,Qwen2.5-7B-Instruct,Qwen3-8B,Qwen2.5-14B-Instruct,Qwen2.5-14B-Instruct-1M,Qwen3-14B,Meta-Llama-3.1-70B-Instruct-AWQ,Qwen3-30B-A3B,Qwen3-30B-A3B-Instruct-2507,Qwen3-32B}
repeats=${SGLANG_ALL_MODELS_REPEATS:-5}
seed=${SGLANG_BENCH_SEED:-20260917}
resume=0
dry_run=0

usage() {
    cat <<'EOF'
Usage: bash tools/sglang-simulator/scripts/run_pr33824_data_tools_all_models_matrix.sh [options]

Collects real A100 ScheduleBatch rows, trains one model-local HGB predictor,
then runs five real saturation traces and five paired CPU replays per selected
checkpoint.  Runs are sequential: the GPU collector and CPU replay never
overlap.

Options:
  --results-root PATH  New immutable test-results root.
  --models CSV         Local checkpoint names below /mnt/data/models.
  --repeats N          Real/replay pairs per checkpoint (default: 5).
  --resume             Continue only stages whose completion marker is absent.
  --dry-run            Print the per-model memory plan without running.
  -h, --help           Show this help.

Incomplete stage directories are never overwritten.  Inspect their logs and
start a new root (or move the incomplete stage aside deliberately) before
retrying that stage.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --models) models_csv=${2:?missing value}; shift 2 ;;
        --repeats) repeats=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be a positive integer" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -d "$checkpoint_root" ]] || { echo "Missing checkpoint root: $checkpoint_root" >&2; exit 1; }
[[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
    echo "Missing isolated SM80 compatibility stage: $compat_stage" >&2
    exit 1
}

memory_fraction_for_model() {
    # Values reserve enough static pool for model weights, KV pages, and the
    # real server's workspaces at context_length=2048.  They are experiment
    # inputs, not a claim about a model's minimum production memory.
    case "$1" in
        Qwen3-0.6B) printf '0.20\n' ;;
        Qwen2.5-3B-Instruct) printf '0.30\n' ;;
        Mistral-7B-Instruct-v0.2|Qwen2.5-7B-Instruct) printf '0.35\n' ;;
        llama-2-7b) printf '0.72\n' ;; # 32 KV heads rather than GQA.
        Qwen3-8B) printf '0.42\n' ;;
        Qwen2.5-14B-Instruct|Qwen2.5-14B-Instruct-1M|Qwen3-14B) printf '0.60\n' ;;
        Meta-Llama-3.1-70B-Instruct-AWQ) printf '0.80\n' ;;
        Qwen3-30B-A3B|Qwen3-30B-A3B-Instruct-2507) printf '0.90\n' ;;
        Qwen3-32B) printf '0.93\n' ;;
        *) echo "No memory plan for model: $1" >&2; return 1 ;;
    esac
}

attention_backend_for_model() {
    # The 1M YaRN checkpoint enables SGLang dual-chunk attention internally;
    # forcing ordinary Triton is rejected by the runtime before collection.
    case "$1" in
        Qwen2.5-14B-Instruct-1M) printf 'dual_chunk_flash_attn\n' ;;
        *) printf 'triton\n' ;;
    esac
}

moe_jit_align_shim_for_model() {
    # The compatible SM80 AOT stage has no MoE align symbol.  Use the scoped
    # PR #33824 JIT source hook only for the two MoE checkpoints.
    case "$1" in
        Qwen3-30B-A3B|Qwen3-30B-A3B-Instruct-2507)
            printf '%s\n' "$script_dir/collector_shims"
            ;;
        *) printf '\n' ;;
    esac
}

runtime_shim_for_model() {
    # Both special cases use the same startup-hook directory: the MoE models
    # replace two omitted AOT helpers, and the 1M model redirects the FA3
    # wrapper to the isolated FA2 package.  Dense Triton runs have no shim.
    case "$1" in
        Qwen2.5-14B-Instruct-1M|Qwen3-30B-A3B|Qwen3-30B-A3B-Instruct-2507)
            printf '%s\n' "$script_dir/collector_shims"
            ;;
        *) printf '\n' ;;
    esac
}

fa2_compat_stage_for_model() {
    case "$1" in
        Qwen2.5-14B-Instruct-1M)
            [[ -n "${SGLANG_FA2_COMPAT_STAGE:-}" ]] || {
                echo "Qwen2.5-14B-Instruct-1M requires SGLANG_FA2_COMPAT_STAGE" >&2
                return 1
            }
            printf '%s\n' "$SGLANG_FA2_COMPAT_STAGE"
            ;;
        *) printf '\n' ;;
    esac
}

page_size_for_model() {
    # FA2 paged-KV expects physical page blocks that are multiples of 256.
    # The 1M dual-chunk compatibility path is therefore isolated from the
    # default automatic page size used by the ordinary Triton checkpoints.
    case "$1" in
        Qwen2.5-14B-Instruct-1M) printf '256\n' ;;
        *) printf '\n' ;;
    esac
}

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

if [[ "$dry_run" == 1 ]]; then
    printf 'model\tmodel_path\tmem_fraction_static\n'
    for model in "${filtered_models[@]}"; do
        printf '%s\t%s\t%s\n' "$model" "$checkpoint_root/$model" "$(memory_fraction_for_model "$model")"
    done
    exit 0
fi

if [[ -e "$results_root" && "$resume" != 1 ]]; then
    echo "Refusing to overwrite existing output root: $results_root" >&2
    exit 1
fi
if [[ "$resume" == 1 && ! -d "$results_root" ]]; then
    echo "--resume requires an existing output root: $results_root" >&2
    exit 1
fi

mkdir -p "$results_root/models" "$results_root/hardware"
lock_file="$results_root/.all-models.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another launcher already owns $results_root" >&2
    exit 1
fi
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date -Is)" > "$results_root/.all-models.owner"
exec >> "$results_root/runner.log" 2>&1

status_path="$results_root/status.tsv"
if [[ ! -f "$status_path" ]]; then
    printf 'timestamp\tmodel\tstage\tstate\tdetail\n' > "$status_path"
fi

record_status() {
    local model=$1 stage=$2 state=$3 detail=${4:-}
    detail=${detail//$'\t'/ }
    detail=${detail//$'\n'/ }
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$model" "$stage" "$state" "$detail" \
        >> "$status_path"
    printf '[%s] %s %s: %s %s\n' "$(date -Is)" "$model" "$stage" "$state" "$detail"
}

if [[ "$resume" != 1 || ! -s "$results_root/run_metadata.txt" ]]; then
    {
        printf 'created_at=%s\n' "$(date -Is)"
        printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
        printf 'source_path=%s\npython=%s\n' "$sglang_root" "$python_bin"
        printf 'models=%s\nrepeats=%s\nseed=%s\n' "$(IFS=,; echo "${filtered_models[*]}")" "$repeats" "$seed"
        printf 'hardware_policy=one A100 PCIe collector at a time; CPU replay begins only after each model real collection completes\n'
        printf 'kernel_policy=isolated_SM80_AOT; SGLANG_FORCE_FUSED_OP_BACKEND=unset\n'
        printf 'collector_source=https://github.com/littlefatfat/sglang/tree/simulator-data-tools-0914\n'
        printf 'collector_commit=9cfddcedc9d1e00885c28db75a45513eddbb6271\n'
        printf 'calibration_cases=latency_256x128_r1,service_1024x256_r4\n'
        printf 'holdout_case=saturation_1024x256_inf\n'
        printf 'scheduler_settings=attention=per-model (triton except 1M dual_chunk_flash_attn); 1M_page_size=256_for_FA2_paged_KV; overlap=disabled; cuda_graph=disabled; context_length=2048; max_running_requests=32; chunked_prefill_size=512\n'
    } > "$results_root/run_metadata.txt"
    printf 'model\tmem_fraction_static\tmodel_path\n' > "$results_root/model_manifest.tsv"
    for model in "${filtered_models[@]}"; do
        printf '%s\t%s\t%s\n' "$model" "$(memory_fraction_for_model "$model")" "$checkpoint_root/$model" \
            >> "$results_root/model_manifest.tsv"
    done
    nvidia-smi -q > "$results_root/hardware/nvidia-smi-before.txt" 2>&1 || true
fi

complete_marker() {
    [[ -f "$1/$2" ]] && grep -qx 'complete' "$1/$2"
}

require_clean_or_complete() {
    local path=$1 marker=$2
    if complete_marker "$path" "$marker"; then
        return 0
    fi
    if [[ -e "$path" ]]; then
        echo "Incomplete stage already exists: $path" >&2
        return 1
    fi
    return 2
}

run_baseline_stage() {
    local model=$1 target=$2 mem_fraction=$3 cases=$4 train_after=$5
    local state attention_backend moe_jit_align_shim runtime_shim fa2_compat_stage page_size
    attention_backend=$(attention_backend_for_model "$model")
    moe_jit_align_shim=$(moe_jit_align_shim_for_model "$model")
    runtime_shim=$(runtime_shim_for_model "$model")
    page_size=$(page_size_for_model "$model")
    if ! fa2_compat_stage=$(fa2_compat_stage_for_model "$model"); then
        record_status "$model" "$(basename "$target")" failed "required FA2 compatibility stage is unset"
        return 1
    fi
    if require_clean_or_complete "$target" BASELINE_COLLECTION_COMPLETE; then
        record_status "$model" "$(basename "$target")" skipped "existing completion marker"
        return 0
    else
        state=$?
    fi
    if [[ "$state" != 2 ]]; then
        return 1
    fi
    record_status "$model" "$(basename "$target")" started "cases=$cases mem_fraction_static=$mem_fraction"
    if ! SGLANG_COLLECT_MODEL="$model" \
        SGLANG_COLLECT_MODEL_PATH="$checkpoint_root/$model" \
        SGLANG_RESULTS_ROOT="$target" \
        SGLANG_COLLECTION_PRESET=dense \
        SGLANG_COLLECTION_CASES="$cases" \
        SGLANG_TRAIN_AFTER_COLLECT="$train_after" \
        SGLANG_MEM_FRACTION_STATIC="$mem_fraction" \
        SGLANG_WARMUP_REQUESTS=4 \
        SGLANG_TRAIN_FORWARD_MODES=1,2 \
        SGLANG_BENCH_SEED="$seed" \
        SGLANG_BENCH_PYTHON="$python_bin" \
        SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
        SGLANG_ATTENTION_BACKEND="$attention_backend" \
        SGLANG_MOE_JIT_ALIGN_SHIM="$moe_jit_align_shim" \
        SGLANG_RUNTIME_SHIM_DIR="$runtime_shim" \
        SGLANG_FA2_COMPAT_STAGE="$fa2_compat_stage" \
        SGLANG_PAGE_SIZE="$page_size" \
        bash "$script_dir/run_pr33824_data_tools_baseline.sh"; then
        record_status "$model" "$(basename "$target")" failed "baseline launcher returned non-zero; see $target"
        return 1
    fi
    complete_marker "$target" BASELINE_COLLECTION_COMPLETE || {
        record_status "$model" "$(basename "$target")" failed "completion marker missing"
        return 1
    }
    record_status "$model" "$(basename "$target")" complete ""
}

run_replay_stage() {
    local model=$1 real_root=$2 replay_root=$3 bundle=$4
    local state
    if require_clean_or_complete "$replay_root" ML_REPLAY_COMPLETE; then
        record_status "$model" "$(basename "$replay_root")" skipped "existing completion marker"
        return 0
    else
        state=$?
    fi
    if [[ "$state" != 2 ]]; then
        return 1
    fi
    record_status "$model" "$(basename "$replay_root")" started "CPU replay"
    if ! SGLANG_COLLECTION_ROOT="$real_root" \
        SGLANG_ML_BUNDLE="$bundle" \
        SGLANG_REPLAY_CASES=saturation_1024x256_inf \
        SGLANG_REPLAY_RESULTS_ROOT="$replay_root" \
        SGLANG_BENCH_PYTHON="$python_bin" \
        SGLANG_KERNEL_COMPAT_STAGE="$compat_stage" \
        bash "$script_dir/run_pr33824_data_tools_ml_replay.sh"; then
        record_status "$model" "$(basename "$replay_root")" failed "replay launcher returned non-zero; see $replay_root"
        return 1
    fi
    complete_marker "$replay_root" ML_REPLAY_COMPLETE || {
        record_status "$model" "$(basename "$replay_root")" failed "completion marker missing"
        return 1
    }
    record_status "$model" "$(basename "$replay_root")" complete ""
}

run_model() {
    local model=$1 mem_fraction=$2
    local model_root="$results_root/models/$model"
    local calibration="$model_root/calibration"
    local real_root="$model_root/holdout/real"
    local simulator_root="$model_root/holdout/simulator"
    local score_root="$model_root/holdout/teacher_forced"
    local e2e_root="$model_root/holdout/e2e_comparison"
    local safe_model=${model//[^A-Za-z0-9._-]/_}
    local bundle
    local repeat real_repeat replay_repeat state
    local -a e2e_args=()

    mkdir -p "$model_root"
    if complete_marker "$model_root" MODEL_COMPLETE; then
        record_status "$model" model skipped "existing completion marker"
        return 0
    fi
    {
        printf 'model=%s\nmodel_path=%s\nmem_fraction_static=%s\n' "$model" "$checkpoint_root/$model" "$mem_fraction"
        printf 'attention_backend=%s\nmoe_jit_align_shim=%s\n' \
            "$(attention_backend_for_model "$model")" "$(moe_jit_align_shim_for_model "$model")"
        printf 'runtime_shim=%s\nfa2_compat_stage=%s\npage_size=%s\n' \
            "$(runtime_shim_for_model "$model")" "$(fa2_compat_stage_for_model "$model")" \
            "$(page_size_for_model "$model")"
        printf 'calibration_cases=latency_256x128_r1,service_1024x256_r4\nholdout_case=saturation_1024x256_inf\n'
    } > "$model_root/model_settings.txt"

    run_baseline_stage "$model" "$calibration" "$mem_fraction" \
        latency_256x128_r1,service_1024x256_r4 1 || return 1
    bundle="$calibration/training/${safe_model}_hgbmono_p50.pkl"
    [[ -f "$bundle" ]] || { record_status "$model" training failed "missing p50 bundle: $bundle"; return 1; }

    for repeat in $(seq 1 "$repeats"); do
        real_repeat="$real_root/repeat-$repeat"
        run_baseline_stage "$model" "$real_repeat" "$mem_fraction" saturation_1024x256_inf 0 || return 1
    done

    if [[ -e "$score_root" ]]; then
        [[ -s "$score_root/summary.csv" ]] || {
            record_status "$model" teacher_forced failed "incomplete scoring directory: $score_root"
            return 1
        }
        record_status "$model" teacher_forced skipped "existing summary.csv"
    else
        record_status "$model" teacher_forced started "real Scheduler batches"
        if ! PYTHONPATH="$sim_root/src:$sglang_root/python" "$python_bin" \
            "$sim_root/tools/score_pr33824_latency_model.py" \
            --data-root "$real_root" --model-bundle "$bundle" --out-dir "$score_root" \
            --forward-modes 1,2; then
            record_status "$model" teacher_forced failed "scoring command returned non-zero"
            return 1
        fi
        record_status "$model" teacher_forced complete ""
    fi

    for repeat in $(seq 1 "$repeats"); do
        real_repeat="$real_root/repeat-$repeat"
        replay_repeat="$simulator_root/repeat-$repeat"
        run_replay_stage "$model" "$real_repeat" "$replay_repeat" "$bundle" || return 1
        e2e_args+=(--real-root "$real_repeat" --replay-root "$replay_repeat")
    done

    if [[ -e "$e2e_root" ]]; then
        [[ -s "$e2e_root/summary.csv" ]] || {
            record_status "$model" e2e_comparison failed "incomplete comparison directory: $e2e_root"
            return 1
        }
        record_status "$model" e2e_comparison skipped "existing summary.csv"
    else
        record_status "$model" e2e_comparison started "paired real/replay summaries"
        if ! PYTHONPATH="$sim_root/src:$sglang_root/python" "$python_bin" \
            "$sim_root/tools/summarize_pr33824_data_tools_replay.py" \
            "${e2e_args[@]}" --out-dir "$e2e_root" \
            --teacher-forced-summary "$score_root/summary.csv"; then
            record_status "$model" e2e_comparison failed "summarizer returned non-zero"
            return 1
        fi
        record_status "$model" e2e_comparison complete ""
    fi
    printf 'complete\n' > "$model_root/MODEL_COMPLETE"
    record_status "$model" model complete ""
}

failures=0
for model in "${filtered_models[@]}"; do
    mem_fraction=$(memory_fraction_for_model "$model")
    if ! run_model "$model" "$mem_fraction"; then
        failures=1
        record_status "$model" model failed "see model_settings.txt and stage logs"
    fi
done

nvidia-smi -q > "$results_root/hardware/nvidia-smi-after.txt" 2>&1 || true
if [[ "$failures" != 0 ]]; then
    printf 'completed-with-failures\n' > "$results_root/ALL_MODELS_COMPLETED_WITH_FAILURES"
    record_status matrix matrix completed-with-failures "successful model directories retain MODEL_COMPLETE"
    exit 1
fi

aggregate="$results_root/aggregate"
report="$sim_root/docs/$(basename "$results_root")_report.md"
docx="${report%.md}.docx"
if [[ ! -f "$results_root/ALL_MODELS_COMPLETE" ]]; then
    record_status matrix aggregate started "generating CSV, Markdown, and DOCX"
    PYTHONPATH="$sim_root/src:$sglang_root/python" "$python_bin" \
        "$sim_root/tools/summarize_pr33824_data_tools_experiment.py" \
        --models-root "$results_root/models" --models "$(IFS=,; echo "${filtered_models[*]}")" \
        --out-dir "$aggregate" --markdown "$report" --repeats "$repeats"
    "$python_bin" "$script_dir/render_markdown_report_docx.py" "$report" --output "$docx"
    printf 'complete\n' > "$results_root/ALL_MODELS_COMPLETE"
    record_status matrix aggregate complete "report=$report docx=$docx"
fi
echo "All-model matrix complete: $results_root"
