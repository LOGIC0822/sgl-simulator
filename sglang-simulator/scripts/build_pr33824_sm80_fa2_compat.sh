#!/usr/bin/env bash
# Build an isolated FlashAttention-2 fallback for A100 dual-chunk experiments.
# It does not modify the hbf-cxl environment or the installed sglang-kernel.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
reference_root=${SGLANG_SM80_COMPAT_REFERENCE_ROOT:-"$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915"}
source_root=${SGLANG_FA2_SOURCE_ROOT:-"$reference_root/cmake/_deps/repo-flash-attention-src"}
cutlass_root=${SGLANG_FA2_CUTLASS_ROOT:-"$reference_root/cmake/_deps/repo-cutlass-src"}
results_root=${SGLANG_FA2_RESULTS_ROOT:-"$sim_root/test-results/pr33824_sm80_fa2_compat_$(date +%Y%m%d_%H%M%S)"}
stage="$results_root/stage"
build_source="$results_root/source"

[[ -x "$python_bin" ]] || { echo "Missing hbf-cxl Python: $python_bin" >&2; exit 1; }
[[ -f "$source_root/setup.py" ]] || { echo "Missing vendored FlashAttention source: $source_root" >&2; exit 1; }
[[ -f "$cutlass_root/include/cutlass/cutlass.h" ]] || { echo "Missing vendored Cutlass source: $cutlass_root" >&2; exit 1; }
[[ ! -e "$results_root" ]] || { echo "Refusing to overwrite: $results_root" >&2; exit 1; }

mkdir -p "$results_root"
source_root=$(realpath "$source_root")
cutlass_root=$(realpath "$cutlass_root")
results_root=$(realpath "$results_root")

# The CMake fetch artifact deliberately contains an empty git-submodule
# placeholder at csrc/cutlass.  Copy both inputs into this result directory so
# pip's source build is self-contained and never mutates the historical
# compatibility tree.
cp -a "$source_root" "$build_source"
cp -a "$cutlass_root/." "$build_source/csrc/cutlass/"
patch -d "$build_source" -p1 --batch < "$script_dir/patches/flash_attn_minimal_sm80.patch"

{
    printf 'created_at=%s\n' "$(date -Is)"
    printf 'python=%s\nsource_root=%s\ncutlass_root=%s\nbuild_source=%s\nstage=%s\n' \
        "$python_bin" "$source_root" "$cutlass_root" "$build_source" "$stage"
    printf 'cuda_home=/usr/local/cuda-12.2\narch=sm80\n'
    printf 'scope=BF16 hdim128 forward-only FA2 dispatch; local fallback for Qwen2.5-14B-Instruct-1M dual_chunk_flash_attn\n'
    printf 'source_patch=%s\n' "$script_dir/patches/flash_attn_minimal_sm80.patch"
    printf 'does_not_modify=hbf-cxl site-packages or installed sglang-kernel\n'
} > "$results_root/run_metadata.txt"

printf '%q ' env \
    CUDA_HOME=/usr/local/cuda-12.2 TORCH_CUDA_ARCH_LIST=8.0 FLASH_ATTN_CUDA_ARCHS=80 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASHATTENTION_MINIMAL_SM80=TRUE \
    MAX_JOBS=1 NVCC_THREADS=1 PIP_NO_INDEX=1 \
    "$python_bin" -m pip install --no-deps --no-build-isolation --target "$stage" "$build_source" \
    > "$results_root/build.command.txt"
printf '\n' >> "$results_root/build.command.txt"

env CUDA_HOME=/usr/local/cuda-12.2 TORCH_CUDA_ARCH_LIST=8.0 FLASH_ATTN_CUDA_ARCHS=80 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASHATTENTION_MINIMAL_SM80=TRUE \
    MAX_JOBS=1 NVCC_THREADS=1 PIP_NO_INDEX=1 \
    "$python_bin" -m pip install --no-deps --no-build-isolation --target "$stage" "$build_source" \
    > "$results_root/build.log" 2>&1

PYTHONPATH="$stage" "$python_bin" - <<'PY' > "$results_root/validation.txt"
import flash_attn
import flash_attn_2_cuda
import torch

print(f"flash_attn={flash_attn.__file__}")
print(f"flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
PY
printf 'complete\n' > "$results_root/FA2_COMPAT_COMPLETE"
echo "FA2 compatibility stage: $stage"
