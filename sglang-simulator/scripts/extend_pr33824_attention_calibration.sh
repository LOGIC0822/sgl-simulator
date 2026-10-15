#!/usr/bin/env bash
# Add measured AIC attention points required for valid Silicon interpolation.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
aic_source=${AICONFIGURATOR_SOURCE_ROOT:-/home/ms-admin/workspace1/yiheng/ali/aiconfigurator-v0.10.0}
calibration_root=

usage() {
    echo "Usage: $0 --calibration-root PATH" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --calibration-root) calibration_root=${2:?missing value}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

[[ -n "$calibration_root" && -d "$calibration_root" ]] || { usage; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python" >&2; exit 1; }
[[ -f "$aic_source/collector/collect.py" ]] || { echo "Missing AIConfigurator source" >&2; exit 1; }
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "A100 is not visible"'

data_dir="$calibration_root/aic-systems/data/a100_pcie_local/sglang/0.5.10"
[[ -s "$data_dir/context_attention_perf.txt" && -s "$data_dir/generation_attention_perf.txt" ]] || {
    echo "Missing existing attention tables under $data_dir" >&2
    exit 1
}
stamp=$(date +%Y%m%d_%H%M%S)
raw_dir="$calibration_root/raw/attention-interpolation-coverage/attention_${stamp}"
checkpoint_dir="$calibration_root/checkpoints/attention-interpolation-coverage"
backup_dir="$calibration_root/aic-systems/backups/attention-interpolation-coverage_${stamp}"
mkdir -p "$raw_dir" "$checkpoint_dir" "$backup_dir"

collector_env=(
    AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE=0.5.10
    AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS=1
    AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND=torch_sdpa
    SGLANG_CACHE_DIR=/mnt/data/sglang-pr33824-cache
    PYTHONPATH="$script_dir/collector_shims:$aic_source:$sglang_root/python:$sim_root/src${PYTHONPATH:+:$PYTHONPATH}"
)
collector_args=("$python_bin" "$script_dir/run_aiconfigurator_collector_compat.py" "$aic_source/collector/collect.py"
    --backend sglang --ops attention_context attention_generation --sm 80
    --model-cases "$script_dir/a100_pcie_attention_interpolation_coverage_cases.yaml"
    --checkpoint-dir "$checkpoint_dir" --keep-csv)
printf '%q ' env "${collector_env[@]}" "${collector_args[@]}" > "$raw_dir/collector.command.txt"
printf '\n' >> "$raw_dir/collector.command.txt"
(cd "$raw_dir" && env "${collector_env[@]}" "${collector_args[@]}") 2>&1 | tee "$raw_dir/collector.log"

merge_attention_tables() {
    local target_dir=$1 backup_root=$2 report_suffix=$3
    for table in context_attention_perf.txt generation_attention_perf.txt; do
        "$python_bin" "$script_dir/merge_aiconfigurator_perf_tables.py" \
            --existing "$target_dir/$table" \
            --supplement "$raw_dir/$table" \
            --backup "$backup_root/$table" \
            --report "$raw_dir/${table%.txt}.${report_suffix}.merge.json"
    done
}

merge_attention_tables "$data_dir" "$backup_dir" primary

# The exact-GEMM database is a copy made for the matrix after direct GEMM
# collection.  Attention updates must reach it as well: the simulator config
# points at this copy, not at the source database above.
exact_data_dir="$calibration_root/aic-systems-exact-gemm/data/a100_pcie_local/sglang/0.5.10"
if [[ -d "$exact_data_dir" ]]; then
    [[ -s "$exact_data_dir/context_attention_perf.txt" && -s "$exact_data_dir/generation_attention_perf.txt" ]] || {
        echo "Incomplete exact-GEMM attention database: $exact_data_dir" >&2
        exit 1
    }
    merge_attention_tables "$exact_data_dir" "$calibration_root/aic-systems-exact-gemm/backups/attention-interpolation-coverage_${stamp}" exact_gemm
fi
printf 'raw_dir=%s\nbackup_dir=%s\nexact_data_dir=%s\n' "$raw_dir" "$backup_dir" "$exact_data_dir" > "$raw_dir/extension_metadata.txt"
echo "Supplemental attention calibration completed: $raw_dir"
