#!/usr/bin/env bash
# Equal per-model request budgets with separate same/different-workload tests.
set -Eeuo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
export PYTHONNOUSERSITE=1
unset SGLANG_BASELINE_DEBUG_IO
export PATH=/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin:$PATH
# Some optional SGLang adapters remain in user site. Place conda BEFORE that
# fallback explicitly, so user-site CPU Torch cannot shadow the CUDA runtime.
export SGLANG_EXPERIMENT_EXTRA_PYTHONPATH=/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/lib/python3.10/site-packages:/home/ms-admin/.local/lib/python3.10/site-packages
export PYTHONPATH="$sim_root/src:$sim_root/../../python:$SGLANG_EXPERIMENT_EXTRA_PYTHONPATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export SGLANG_CHECKPOINT_ROOT=${SGLANG_CHECKPOINT_ROOT:-/home/ms-admin/data/models}
export SGLANG_FA2_COMPAT_STAGE=${SGLANG_FA2_COMPAT_STAGE:-"$sim_root/test-results/pr33824_sm80_fa2_compat_v6_20260918_1305/stage"}
export SGLANG_SAME_DISTRIBUTION_COLLECTION_PRESET=expanded
export SGLANG_SAME_DISTRIBUTION_TRAIN_CASES=latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf
export SGLANG_SAME_DISTRIBUTION_HOLDOUT_CASES="$SGLANG_SAME_DISTRIBUTION_TRAIN_CASES,latency_128x64_r1,saturation_1536x384_inf"
export SGLANG_SAME_DISTRIBUTION_TRAIN_SEED_BASE=202609280
export SGLANG_SAME_DISTRIBUTION_HOLDOUT_SEED_BASE=202609380
export SGLANG_COMPREHENSIVE_TRAIN_REPEATS=5
export SGLANG_COMPREHENSIVE_HOLDOUT_REPEATS=5
export SGLANG_COMPREHENSIVE_PROMPT_MULTIPLIER=2
export SGLANG_COMPREHENSIVE_LEARNING_CURVES=1
export SGLANG_SAME_DISTRIBUTION_MEMORY_OVERRIDES=${SGLANG_SAME_DISTRIBUTION_MEMORY_OVERRIDES:-Qwen3-0.6B=0.65,Qwen2.5-3B-Instruct=0.80}
export SGLANG_GPU_CAPACITY_WAIT=1
export SGLANG_GPU_CAPACITY_CHECK_SECONDS=${SGLANG_GPU_CAPACITY_CHECK_SECONDS:-600}
export SGLANG_COMPREHENSIVE_RESULTS_ROOT=${SGLANG_COMPREHENSIVE_RESULTS_ROOT:-"$sim_root/test-results/pr33824_expanded_all13_predictors_$(date +%Y%m%d_%H%M%S)"}
exec bash "$script_dir/run_pr33824_comprehensive_predictor_matrix.sh" "$@"
