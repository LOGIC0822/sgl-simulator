#!/usr/bin/env bash
# Run one CPU-simulator replay for each completed real-GPU repeat.  An existing
# incomplete output is never reused: a numbered retry directory is selected and
# recorded in replay_mapping.tsv instead.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
real_root=${SGLANG_REAL_REPEATS_ROOT:?Set SGLANG_REAL_REPEATS_ROOT to the directory containing repeat-N}
bundle=${SGLANG_ML_BUNDLE:?Set SGLANG_ML_BUNDLE to a trained ML bundle}
replay_root=${SGLANG_REPLAY_REPEATS_ROOT:?Set SGLANG_REPLAY_REPEATS_ROOT for CPU replay outputs}
repeat_count=${SGLANG_REPLAY_REPEAT_COUNT:-5}

[[ "$repeat_count" =~ ^[1-9][0-9]*$ ]] || {
    echo "SGLANG_REPLAY_REPEAT_COUNT must be a positive integer" >&2
    exit 2
}
[[ -d "$real_root" ]] || { echo "Missing real repeat root: $real_root" >&2; exit 2; }
[[ -f "$bundle" ]] || { echo "Missing ML bundle: $bundle" >&2; exit 2; }
mkdir -p "$replay_root"

mapping="$replay_root/replay_mapping.tsv"
if [[ ! -e "$mapping" ]]; then
    printf 'repeat\treal_root\treplay_root\tstatus\n' > "$mapping"
fi

complete_marker() {
    [[ -f "$1/ML_REPLAY_COMPLETE" ]] && grep -qx 'complete' "$1/ML_REPLAY_COMPLETE"
}

for repeat in $(seq 1 "$repeat_count"); do
    source_root="$real_root/repeat-$repeat"
    [[ -f "$source_root/run_metadata.txt" ]] || {
        echo "Missing completed real repeat metadata: $source_root" >&2
        exit 2
    }

    destination="$replay_root/repeat-$repeat"
    if complete_marker "$destination"; then
        printf '%s\t%s\t%s\tcomplete-existing\n' "$repeat" "$source_root" "$destination" >> "$mapping"
        continue
    fi
    if [[ -e "$destination" ]]; then
        retry=1
        while [[ -e "$destination-retry-$retry" ]]; do
            if complete_marker "$destination-retry-$retry"; then
                destination="$destination-retry-$retry"
                break
            fi
            retry=$((retry + 1))
        done
        if ! complete_marker "$destination"; then
            destination="$destination-retry-$retry"
        fi
    fi

    SGLANG_COLLECTION_ROOT="$source_root" \
    SGLANG_ML_BUNDLE="$bundle" \
    SGLANG_REPLAY_CASES="${SGLANG_REPLAY_CASES:-saturation_1024x256_inf}" \
    SGLANG_REPLAY_RESULTS_ROOT="$destination" \
    bash "$script_dir/run_pr33824_data_tools_ml_replay.sh"
    complete_marker "$destination" || {
        echo "Replay did not write its completion marker: $destination" >&2
        exit 1
    }
    printf '%s\t%s\t%s\tcomplete\n' "$repeat" "$source_root" "$destination" >> "$mapping"
done

printf 'complete\n' > "$replay_root/REPLAY_REPEATS_COMPLETE"
echo "Replay repeats complete: $replay_root"
