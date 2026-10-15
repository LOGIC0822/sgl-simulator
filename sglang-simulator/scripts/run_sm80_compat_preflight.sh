#!/usr/bin/env bash
# Run one real A100/SM80 SGLang preflight using the isolated compatibility
# kernel.  This is a functional check, not an accuracy-matrix launcher.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
sglang_root=$(cd "$sim_root/../.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
checkpoint_root=${SGLANG_CHECKPOINT_ROOT:-/mnt/data/models}
model_name=${SGLANG_COMPAT_MODEL:-Qwen2.5-3B-Instruct}
model_path="$checkpoint_root/$model_name"
compat_stage=${SGLANG_KERNEL_COMPAT_STAGE:?Set SGLANG_KERNEL_COMPAT_STAGE to the minimal-stage-v4 directory}
results_root=${SGLANG_RESULTS_ROOT:-"$sim_root/test-results/sm80_compat_${model_name}_$(date +%Y%m%d_%H%M%S)"}
port=${SGLANG_REAL_PORT:-34003}
mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC:-0.30}

[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
[[ -d "$model_path" ]] || { echo "Missing model: $model_path" >&2; exit 1; }
[[ -f "$compat_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
    echo "Missing SM80 compatibility binary below: $compat_stage" >&2
    exit 1
}
[[ ! -e "$results_root" ]] || { echo "Refusing to overwrite: $results_root" >&2; exit 1; }

mkdir -p "$results_root/hardware" "$results_root/cache" "$results_root/jit-cache"
server_pid=

cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -- -"$server_pid" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- -"$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Do not append inherited PYTHONPATH: a historical collector/runtime shim has
# a same-name sgl_kernel package and would shadow this compatibility build.
export PYTHONPATH="$compat_stage:$sglang_root/python:$sim_root/src"
export LD_LIBRARY_PATH="$(dirname "$($python_bin - <<'PY'
import os
import torch
print(torch.__file__)
PY
)")/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.2}
export SGLANG_CACHE_DIR="$results_root/cache"
export SGLANG_JIT_CACHE_DIR="$results_root/jit-cache"
# FlashInfer creates its JIT logger during import, even when the selected
# dense-Qwen path uses the isolated AOT operators below.  Keep that mutable
# workspace alongside this run instead of relying on a user-home cache.
export FLASHINFER_WORKSPACE_BASE=${SGLANG_FLASHINFER_WORKSPACE_BASE:-"$results_root/flashinfer-workspace"}
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
unset SGLANG_FORCE_FUSED_OP_BACKEND
unset SGLANG_WEAK_REF_COMPAT_LIBRARY

{
    printf 'created_at=%s\n' "$(date -Is)"
    printf 'source_revision=%s\n' "$(git -C "$sglang_root" rev-parse HEAD)"
    printf 'source_path=%s\npython=%s\n' "$sglang_root" "$python_bin"
    printf 'model=%s\nmodel_path=%s\n' "$model_name" "$model_path"
    printf 'compat_stage=%s\n' "$(realpath "$compat_stage")"
    printf 'kernel_policy=isolated_AOT_SM80; SGLANG_FORCE_FUSED_OP_BACKEND=unset\n'
    printf 'flashinfer_workspace_base=%s\n' "$FLASHINFER_WORKSPACE_BASE"
    printf 'attention_backend=triton\n'
    printf 'mem_fraction_static=%s\n' "$mem_fraction_static"
    printf 'cuda_graph_config={"decode":{"backend":"full","max_bs":4},"prefill":{"backend":"tc_piecewise","max_bs":128,"tc_compiler":"eager"}}\n'
    printf 'workload=random-ids; prompts=4; input_len=256; output_len=32; request_rate=1; seed=20260908\n'
} > "$results_root/run_metadata.txt"

"$python_bin" - <<'PY' > "$results_root/package_versions.txt"
import importlib.metadata as md
import sys
import torch
import sgl_kernel

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"sgl_kernel_file={sgl_kernel.__file__}")
for operation in ("rmsnorm", "fused_add_rmsnorm", "silu_and_mul", "rotary_embedding", "weak_ref_tensor"):
    print(f"torch.ops.sgl_kernel.{operation}={hasattr(torch.ops.sgl_kernel, operation)}")
for name in ("sglang", "sglang-kernel", "aiconfigurator"):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=not-installed")
PY
nvidia-smi -q > "$results_root/hardware/nvidia-smi-before.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,memory.used,driver_version --format=csv,noheader \
    > "$results_root/hardware/gpu_identity.csv" 2>&1 || true

graph_config='{"decode":{"backend":"full","max_bs":4},"prefill":{"backend":"tc_piecewise","max_bs":128,"tc_compiler":"eager"}}'
printf '%q ' "$python_bin" -m sglang.launch_server \
    --model-path "$model_path" --host 127.0.0.1 --port "$port" --device cuda --tp-size 1 \
    --context-length 2048 --mem-fraction-static "$mem_fraction_static" --max-running-requests 4 \
    --chunked-prefill-size 128 --attention-backend triton --cuda-graph-config "$graph_config" \
    > "$results_root/server.command.txt"
printf '\n' >> "$results_root/server.command.txt"

setsid "$python_bin" -m sglang.launch_server \
    --model-path "$model_path" --host 127.0.0.1 --port "$port" --device cuda --tp-size 1 \
    --context-length 2048 --mem-fraction-static "$mem_fraction_static" --max-running-requests 4 \
    --chunked-prefill-size 128 --attention-backend triton --cuda-graph-config "$graph_config" \
    > "$results_root/server.log" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$port/v1/models" > "$results_root/models.json"; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Server exited before readiness; inspect $results_root/server.log" >&2
        tail -120 "$results_root/server.log" >&2 || true
        exit 1
    fi
    sleep 1
done
[[ -s "$results_root/models.json" ]] || { echo "Server readiness timeout" >&2; exit 1; }

printf '%q ' "$python_bin" -m sglang.benchmark.serving --backend sglang --base-url "http://127.0.0.1:$port" \
    --model "$model_path" --dataset-name random-ids --num-prompts 4 --random-input-len 256 \
    --random-output-len 32 --random-range-ratio 0 --request-rate 1 --seed 20260908 --warmup-requests 0 \
    --flush-cache --disable-tqdm --ready-check-timeout-sec 0 --output-file "$results_root/benchmark.jsonl" \
    > "$results_root/benchmark.command.txt"
printf '\n' >> "$results_root/benchmark.command.txt"
"$python_bin" -m sglang.benchmark.serving --backend sglang --base-url "http://127.0.0.1:$port" \
    --model "$model_path" --dataset-name random-ids --num-prompts 4 --random-input-len 256 \
    --random-output-len 32 --random-range-ratio 0 --request-rate 1 --seed 20260908 --warmup-requests 0 \
    --flush-cache --disable-tqdm --ready-check-timeout-sec 0 --output-file "$results_root/benchmark.jsonl" \
    > "$results_root/benchmark.log" 2>&1

nvidia-smi -q > "$results_root/hardware/nvidia-smi-after.txt" 2>&1 || true
printf 'ok\n' > "$results_root/status.txt"
echo "Preflight complete: $results_root"
