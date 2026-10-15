#!/usr/bin/env bash
# One-command controller: deploy a source snapshot, collect on a remote A100,
# fetch raw evidence, then train and replay predictors on the local CPU.

set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
config_file=
phase=all
resume=0
dry_run=0
results_override=
run_id_override=

usage() {
    cat <<'EOF'
Usage: bash scripts/run_pr33824_remote_a100_experiment.sh --config FILE [options]

Options:
  --phase all|collect|predict  Run both halves, remote collection only, or local prediction only.
  --run-id ID                  Use a stable ID (reuse it with --resume).
  --results-root PATH          Existing/new local result root (useful for --phase predict).
  --resume                     Resume the same RUN_ID and result root.
  --dry-run                    Print the selected actions without SSH or GPU work.
  -h, --help                   Show this help.

`all` deploys the current source snapshot, collects real SGLang traces and AIC
Silicon data remotely, copies data and sources back, then runs ML/AIC prediction
and CPU simulator replay locally.  The remote Python environment and local CPU
environment must already be prepared; this script validates them before use.
The current collector supports TP=1 only.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config_file=${2:?missing value}; shift 2 ;;
        --phase) phase=${2:?missing value}; shift 2 ;;
        --run-id) run_id_override=${2:?missing value}; shift 2 ;;
        --results-root) results_override=${2:?missing value}; shift 2 ;;
        --resume) resume=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$config_file" && -f "$config_file" ]] || { usage >&2; exit 2; }
[[ "$phase" == all || "$phase" == collect || "$phase" == predict ]] || {
    echo "--phase must be all, collect, or predict" >&2; exit 2;
}
# The config is a shell assignment file under the user's control. Never put
# passwords, private keys, or tokens in it; SSH tries the existing keys and, if
# none are installed on the remote, falls back to an interactive password
# prompt over a single multiplexed connection.
# shellcheck source=/dev/null
source "$config_file"

run_id=${run_id_override:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)}}
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "RUN_ID may contain only letters, digits, dot, underscore, and dash" >&2; exit 2; }
models=${MODELS:-Qwen3-0.6B,Qwen2.5-3B-Instruct,llama-2-7b,Mistral-7B-Instruct-v0.2,Qwen2.5-7B-Instruct,Qwen3-8B,Qwen2.5-14B-Instruct,Qwen2.5-14B-Instruct-1M,Qwen3-14B,Meta-Llama-3.1-70B-Instruct-AWQ,Qwen3-30B-A3B,Qwen3-30B-A3B-Instruct-2507,Qwen3-32B}
IFS=',' read -r -a requested_models <<< "$models"
remote_port=${REMOTE_PORT:-22}
remote_gpu_id=${REMOTE_GPU_ID:-0}
train_repeats=${TRAIN_REPEATS:-5}
holdout_repeats=${HOLDOUT_REPEATS:-5}
prompt_multiplier=${PROMPT_MULTIPLIER:-2}
train_cases=${TRAIN_CASES:-latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf}
holdout_cases=${HOLDOUT_CASES:-latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf,latency_128x64_r1,saturation_1536x384_inf}
local_results_root=${results_override:-${LOCAL_RESULTS_ROOT:-"$sim_root/test-results/a100_remote_$run_id"}}
remote_run_root=
remote_source_root=
remote_aic_root=
remote_kernel_root=
remote_results_root=
local_source_root="$local_results_root/deployment/sglang-pr33824-merged"
local_aic_root="$local_results_root/deployment/aiconfigurator"

require_local_prediction_config() {
    : "${LOCAL_PYTHON:?Set LOCAL_PYTHON in the config}"
    : "${LOCAL_MODEL_ROOT:?Set LOCAL_MODEL_ROOT in the config}"
    [[ -x "$LOCAL_PYTHON" ]] || { echo "Missing LOCAL_PYTHON: $LOCAL_PYTHON" >&2; exit 1; }
    [[ -d "$LOCAL_MODEL_ROOT" ]] || { echo "Missing LOCAL_MODEL_ROOT: $LOCAL_MODEL_ROOT" >&2; exit 1; }
}

ssh_opts=(-p "$remote_port" -o "ConnectTimeout=10" -o "ServerAliveInterval=60" -o "ServerAliveCountMax=30")
remote_run() {
    local remote_command
    printf -v remote_command '%q ' "$@"
    # The command is deliberately shell-escaped above, then interpreted remotely.
    # shellcheck disable=SC2029
    ssh "${ssh_opts[@]}" "$REMOTE_HOST" "$remote_command"
}

local_prediction_preflight() {
    local python_path=$1 source_tree=$2 aic_path=$3 metadata_path=$4
    [[ "$metadata_path" == - ]] || mkdir -p "$(dirname "$metadata_path")"
    CUDA_VISIBLE_DEVICES='' PYTHONPATH="$source_tree/tools/sglang-simulator/src:$source_tree/python:$aic_path/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$python_path" - "$metadata_path" <<'PY'
import importlib.metadata as metadata
import json
import pathlib
import sys
import torch
import numpy
import sklearn
import joblib
import matplotlib
import docx
import aiconfigurator

if sys.version_info < (3, 10):
    raise SystemExit(f"Local CPU Python 3.10+ required, found {sys.version}")
if torch.cuda.is_available():
    raise SystemExit("Local prediction preflight unexpectedly sees a CUDA device")
if metadata.version("aiconfigurator") != "0.10.0":
    raise SystemExit(f"Local AIC must be 0.10.0, found {metadata.version('aiconfigurator')}")
versions = {name: metadata.version(name) for name in (
    "torch", "numpy", "scikit-learn", "joblib", "matplotlib", "python-docx", "aiconfigurator"
)}
versions.update({"python": sys.version, "python_executable": sys.executable,
                 "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                 "aiconfigurator_file": aiconfigurator.__file__})
if sys.argv[1] != "-":
    pathlib.Path(sys.argv[1]).write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
print(json.dumps(versions, indent=2))
PY
}

verify_local_model_directories() {
    local model
    for model in "${requested_models[@]}"; do
        model=${model//[[:space:]]/}
        [[ -n "$model" ]] || continue
        [[ -d "$LOCAL_MODEL_ROOT/$model" ]] || {
            echo "Missing local model config directory: $LOCAL_MODEL_ROOT/$model" >&2; exit 1;
        }
    done
}

sync_back() {
    local remote_shell="ssh -p $remote_port -o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=30 -o ControlMaster=auto -o ControlPath=$ssh_control_path -o ControlPersist=300"
    mkdir -p "$local_results_root/deployment"
    rsync -a --partial --info=stats2 -e "$remote_shell" \
        "$REMOTE_HOST:$remote_source_root/" "$local_source_root/"
    rsync -a --partial --info=stats2 -e "$remote_shell" \
        "$REMOTE_HOST:$remote_aic_root/" "$local_aic_root/"
    rsync -a --partial --info=stats2 -e "$remote_shell" \
        "$REMOTE_HOST:$remote_results_root/" "$local_results_root/"
}

run_local_prediction() {
    require_local_prediction_config
    verify_local_model_directories
    [[ -f "$local_results_root/A100_REMOTE_COLLECTION_COMPLETE" ]] || {
        echo "No completed remote collection in $local_results_root" >&2; exit 1;
    }
    [[ -d "$local_source_root/tools/sglang-simulator" && -d "$local_aic_root/src/aiconfigurator" ]] || {
        echo "Returned source snapshot is missing under $local_results_root/deployment" >&2; exit 1;
    }
    local_prediction_preflight "$LOCAL_PYTHON" "$local_source_root" "$local_aic_root" \
        "$local_results_root/local_cpu_prediction/local_package_versions.json"
    local predictor="$local_source_root/tools/sglang-simulator/scripts/run_pr33824_comprehensive_predictor_matrix.sh"
    local -a args=(--phase predict --results-root "$local_results_root" --models "$models"
        --train-repeats "$train_repeats" --holdout-repeats "$holdout_repeats"
        --prompt-multiplier "$prompt_multiplier" --resume)
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
        PYTHONPATH="$local_source_root/tools/sglang-simulator/src:$local_source_root/python:$local_aic_root/src${PYTHONPATH:+:$PYTHONPATH}" \
        SGLANG_BENCH_PYTHON="$LOCAL_PYTHON" \
        SGLANG_CHECKPOINT_ROOT="$LOCAL_MODEL_ROOT" \
        AICONFIGURATOR_SOURCE_ROOT="$local_aic_root" \
        SGLANG_KERNEL_COMPAT_STAGE= \
        SGLANG_COMPREHENSIVE_RESULTS_ROOT="$local_results_root" \
        SGLANG_COMPREHENSIVE_MODELS="$models" \
        SGLANG_COMPREHENSIVE_TRAIN_REPEATS="$train_repeats" \
        SGLANG_COMPREHENSIVE_HOLDOUT_REPEATS="$holdout_repeats" \
        SGLANG_COMPREHENSIVE_PROMPT_MULTIPLIER="$prompt_multiplier" \
        SGLANG_SAME_DISTRIBUTION_TRAIN_CASES="$train_cases" \
        SGLANG_SAME_DISTRIBUTION_HOLDOUT_CASES="$holdout_cases" \
        bash "$predictor" "${args[@]}"
}

if [[ "$phase" == predict ]]; then
    run_local_prediction
    exit 0
fi

: "${REMOTE_HOST:?Set REMOTE_HOST=user@host in the config}"
: "${REMOTE_ROOT:?Set REMOTE_ROOT to a writable remote directory}"
: "${REMOTE_PYTHON:?Set REMOTE_PYTHON to the prepared GPU Python}"
: "${REMOTE_MODEL_ROOT:?Set REMOTE_MODEL_ROOT to the remote checkpoint directory}"
: "${AICONFIGURATOR_SOURCE_ROOT:?Set AICONFIGURATOR_SOURCE_ROOT to the local AIC checkout}"
: "${KERNEL_COMPAT_STAGE:?Set KERNEL_COMPAT_STAGE to the tested SM80 kernel stage}"
[[ "$remote_port" =~ ^[0-9]+$ && "$remote_gpu_id" =~ ^[0-9]+$ ]] || { echo "Port and GPU ID must be integers" >&2; exit 2; }
[[ "$REMOTE_ROOT" == /* && "$REMOTE_MODEL_ROOT" == /* ]] || { echo "Remote paths must be absolute" >&2; exit 2; }
[[ "$REMOTE_ROOT$REMOTE_MODEL_ROOT" != *[[:space:]]* ]] || { echo "Remote paths cannot contain whitespace" >&2; exit 2; }
: "${REMOTE_PYTHON:?Set REMOTE_PYTHON to an absolute Python path on the remote host}"
[[ "$REMOTE_PYTHON" == /* && "$REMOTE_PYTHON" != *[[:space:]]* ]] || {
    echo "REMOTE_PYTHON must be an absolute path without whitespace" >&2; exit 2;
}
[[ -d "$AICONFIGURATOR_SOURCE_ROOT" && -d "$AICONFIGURATOR_SOURCE_ROOT/src/aiconfigurator" ]] || {
    echo "Missing local AIC source checkout: $AICONFIGURATOR_SOURCE_ROOT" >&2; exit 1;
}
[[ -d "$KERNEL_COMPAT_STAGE/sgl_kernel" ]] || { echo "Missing kernel stage: $KERNEL_COMPAT_STAGE" >&2; exit 1; }

# The remote accepts only publickey+password and no local key is installed on
# it, so BatchMode (which forbids password prompts) cannot be used. Multiplex
# every ssh/rsync call over one master connection: the password is prompted
# exactly once and the socket is closed when the script exits.
ssh_control_path="/tmp/sglang-sim-a100-ssh-$(printf '%s:%s' "$REMOTE_HOST" "$remote_port" | md5sum | cut -c1-8)"
ssh_opts+=(-o "ControlMaster=auto" -o "ControlPath=$ssh_control_path" -o "ControlPersist=300")
cleanup_ssh_control() {
    ssh -O exit -o "ControlPath=$ssh_control_path" "${REMOTE_HOST:-}" >/dev/null 2>&1 || true
}
trap cleanup_ssh_control EXIT

remote_run_root="$REMOTE_ROOT/$run_id"
remote_source_root="$remote_run_root/source/sglang-pr33824-merged"
remote_aic_root="$remote_run_root/source/aiconfigurator"
remote_kernel_root="$remote_run_root/source/kernel-stage"
remote_results_root="$remote_run_root/results"

if [[ "$dry_run" == 1 ]]; then
    printf 'phase=%s\nrun_id=%s\nmodels=%s\n' "$phase" "$run_id" "$models"
    printf 'remote=%s:%s\nremote_results=%s\nlocal_results=%s\n' \
        "$REMOTE_HOST" "$remote_port" "$remote_results_root" "$local_results_root"
    printf 'remote_gpu=CUDA_VISIBLE_DEVICES=%s; tp_size=1\n' "$remote_gpu_id"
    printf 'remote_python=%s\nlocal_python=%s\n' "$REMOTE_PYTHON" "${LOCAL_PYTHON:-<set for predict phase>}"
    printf 'training=%s repeats x %s prompts per repeat\n' "$train_cases" "$train_repeats"
    printf 'holdout=%s repeats x %s prompts per repeat\n' "$holdout_cases" "$holdout_repeats"
    printf 'would_deploy_source_collect_sync_and_run_local_cpu_prediction=%s\n' "$([[ "$phase" == all ]] && echo yes || echo no)"
    exit 0
fi

if [[ "$phase" == all ]]; then
    require_local_prediction_config
    verify_local_model_directories
fi
[[ ! -e "$local_results_root" || "$resume" == 1 ]] || {
    echo "Refusing to overwrite local results root: $local_results_root" >&2; exit 1;
}
if [[ "$phase" == all ]]; then
    local_prediction_preflight "$LOCAL_PYTHON" "$sglang_root" "$AICONFIGURATOR_SOURCE_ROOT" -
fi
if [[ "$resume" != 1 ]]; then
    ssh "${ssh_opts[@]}" "$REMOTE_HOST" true
    remote_run mkdir -p "$REMOTE_ROOT"
    remote_run test '!' -e "$remote_run_root" || {
        echo "Remote run root already exists; use --resume with the same RUN_ID" >&2; exit 1;
    }
    remote_run mkdir -p "$remote_source_root" "$remote_aic_root" "$remote_kernel_root" "$remote_results_root"
    rsync_transport="ssh -p $remote_port -o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=30 -o ControlMaster=auto -o ControlPath=$ssh_control_path -o ControlPersist=300"
    # 1. 推送 sglang python/sglang
    remote_run mkdir -p "$remote_source_root/python/sglang"
    rsync -a --partial --info=stats2 --exclude='test-results' --exclude='.git' --exclude='__pycache__' \
    -e "$rsync_transport" "$sglang_root/python/sglang/" "$REMOTE_HOST:$remote_source_root/python/sglang/"

    # 2. 推送 simulator（修正：用 remote_source_root 拼接路径，删掉不存在的 remote_sim_root）
    remote_run mkdir -p "$remote_source_root/tools/sglang-simulator"
    rsync -a --partial --info=stats2 --exclude='test-results' --exclude='.git' --exclude='__pycache__' \
    -e "$rsync_transport" "$sim_root/" "$REMOTE_HOST:$remote_source_root/tools/sglang-simulator/"

    # 3. 推送 kernel-stage（保留 -L，解软链接！）
    remote_run mkdir -p "$remote_kernel_root"
    rsync -aL --partial --info=stats2 \
    -e "$rsync_transport" "$KERNEL_COMPAT_STAGE/" "$REMOTE_HOST:$remote_kernel_root/"
    rsync -aL --partial --info=stats2 -e "$rsync_transport" \
        "$KERNEL_COMPAT_STAGE/" "$REMOTE_HOST:$remote_kernel_root/"

    for snapshot in "$remote_source_root" "$remote_aic_root"; do
        remote_run git -C "$snapshot" init --quiet
        remote_run git -c user.name=ExperimentSnapshot -c user.email=experiment-snapshot@localhost \
            -C "$snapshot" add -A
        remote_run git -c user.name=ExperimentSnapshot -c user.email=experiment-snapshot@localhost \
            -C "$snapshot" commit --quiet -m "Portable experiment source snapshot $run_id"
    done
else
    remote_run test -d "$remote_source_root"
    remote_run test -d "$remote_aic_root"
    remote_run test -d "$remote_kernel_root"
    remote_run test -d "$remote_results_root"
fi

remote_collect="$remote_source_root/tools/sglang-simulator/scripts/run_pr33824_remote_a100_collect.sh"
remote_args=(bash "$remote_collect" --source-root "$remote_source_root" --aic-source "$remote_aic_root"
    --kernel-stage "$remote_kernel_root" --python "$REMOTE_PYTHON" --model-root "$REMOTE_MODEL_ROOT"
    --results-root "$remote_results_root" --gpu-id "$remote_gpu_id" --models "$models"
    --train-repeats "$train_repeats" --holdout-repeats "$holdout_repeats"
    --prompt-multiplier "$prompt_multiplier" --train-cases "$train_cases" --holdout-cases "$holdout_cases")
remote_run "${remote_args[@]}"

sync_back
[[ -f "$local_results_root/A100_REMOTE_COLLECTION_COMPLETE" ]] || {
    echo "Remote collection marker was not transferred" >&2; exit 1;
}
if [[ "$phase" == all ]]; then
    run_local_prediction
fi

printf 'Experiment data: %s\n' "$local_results_root"
printf 'Status files: %s/comprehensive_status.tsv and %s/status.tsv\n' "$local_results_root" "$local_results_root"
if [[ "$phase" == all ]]; then
    printf 'Local prediction report: %s/deployment/sglang-pr33824-merged/tools/sglang-simulator/docs/%s_complete_report.md\n' \
        "$local_results_root" "$(basename "$local_results_root")"
    printf 'Local prediction report (DOCX): %s/deployment/sglang-pr33824-merged/tools/sglang-simulator/docs/%s_complete_report.docx\n' \
        "$local_results_root" "$(basename "$local_results_root")"
fi
