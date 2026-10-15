#!/usr/bin/env bash
# Wait for one owned matrix launcher, validate its raw samples, then render the handoff.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 RESULTS_ROOT MATRIX_PID" >&2
    exit 2
fi

results_root=$(realpath -m "$1")
matrix_pid=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
status_file="$results_root/finalization.status"

[[ "$matrix_pid" =~ ^[1-9][0-9]*$ ]] || { echo "MATRIX_PID must be numeric" >&2; exit 2; }
[[ -d "$results_root/results" ]] || { echo "Missing results: $results_root/results" >&2; exit 2; }

while kill -0 "$matrix_pid" 2>/dev/null; do
    sleep 30
done

real_ok=$(find "$results_root/results" -path '*/real/*/repeat-*/status.txt' -exec grep -lx ok {} \; | wc -l)
simulator_ok=$(find "$results_root/results" -path '*/simulator/*/repeat-*/status.txt' -exec grep -lx ok {} \; | wc -l)
if (( real_ok != 45 || simulator_ok < 18 )); then
    printf 'status=incomplete\nreal_ok=%s\nsimulator_ok=%s\n' "$real_ok" "$simulator_ok" > "$status_file"
    exit 1
fi

"$python_bin" "$script_dir/summarize_pr33824_a100_matrix.py" \
    --results-root "$results_root" --prediction-aggregation mean-per-case
"$python_bin" "$script_dir/render_pr33824_a100_report.py" --results-root "$results_root"
printf 'status=complete\nreal_ok=%s\nsimulator_ok=%s\n' "$real_ok" "$simulator_ok" > "$status_file"
