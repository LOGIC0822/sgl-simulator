#!/usr/bin/env bash
# Refresh the readable status page while a detached PR #33824 matrix is live.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
results_root=
interval_seconds=600

usage() {
    echo "Usage: $0 --results-root PATH [--interval-seconds N]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --interval-seconds) interval_seconds=${2:?missing value}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

[[ -n "$results_root" && -d "$results_root" ]] || { usage; exit 2; }
[[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }

while :; do
    "$python_bin" "$script_dir/write_pr33824_matrix_status.py" --results-root "$results_root"
    if [[ -e "$results_root/MATRIX_COMPLETE" || -e "$results_root/MATRIX_COMPLETED_WITH_FAILURES" ]]; then
        "$python_bin" "$script_dir/summarize_pr33824_a100_matrix.py" --results-root "$results_root"
        "$python_bin" "$script_dir/plot_pr33824_a100_matrix.py" --summary-root "$results_root/analysis"
        "$python_bin" "$script_dir/render_pr33824_a100_report.py" --results-root "$results_root"
        "$python_bin" "$script_dir/verify_pr33824_a100_matrix.py" --results-root "$results_root"
        exit 0
    fi
    sleep "$interval_seconds"
done
