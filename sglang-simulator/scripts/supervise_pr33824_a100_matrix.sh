#!/usr/bin/env bash
# Keep one resumable PR #33824 experiment launcher alive without concurrent writes.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results_root=
sim_config=
poll_seconds=120
max_restarts=3

usage() {
    echo "Usage: $0 --results-root PATH --sim-config PATH [--poll-seconds N] [--max-restarts N]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root=${2:?missing value}; shift 2 ;;
        --sim-config) sim_config=${2:?missing value}; shift 2 ;;
        --poll-seconds) poll_seconds=${2:?missing value}; shift 2 ;;
        --max-restarts) max_restarts=${2:?missing value}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

[[ -n "$results_root" && -d "$results_root" && -f "$sim_config" ]] || { usage; exit 2; }
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ && "$max_restarts" =~ ^[0-9]+$ ]] || { usage; exit 2; }

launcher="$script_dir/run_pr33824_a100_matrix.sh"
restarts=0

launcher_is_live() {
    ps -eo args= | grep -F "$launcher --resume --results-root $results_root" | grep -v grep > /dev/null
}

while :; do
    if [[ -e "$results_root/MATRIX_COMPLETE" || -e "$results_root/MATRIX_COMPLETED_WITH_FAILURES" ]]; then
        exit 0
    fi
    if launcher_is_live; then
        sleep "$poll_seconds"
        continue
    fi
    if (( restarts >= max_restarts )); then
        printf 'restart limit reached (%s); inspect runner.log and status.md\n' "$max_restarts" \
            > "$results_root/SUPERVISOR_RETRY_LIMIT_REACHED"
        exit 1
    fi

    restarts=$((restarts + 1))
    printf '%s supervisor restart %s/%s\n' "$(date -Is)" "$restarts" "$max_restarts" \
        >> "$results_root/runner.log"
    setsid bash "$launcher" --resume --results-root "$results_root" --sim-config "$sim_config" \
        >> "$results_root/runner.log" 2>&1 &
    printf 'supervisor_runner_pid=%s restart=%s at=%s\n' "$!" "$restarts" "$(date -Is)" \
        >> "$results_root/launcher_processes.txt"
    sleep 10
done
