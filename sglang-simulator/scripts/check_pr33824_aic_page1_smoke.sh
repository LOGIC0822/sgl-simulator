#!/usr/bin/env bash
# Two real CUDA operator calls, separate from formal timing tables.
set -Eeuo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sim_root=$(cd "$script_dir/.." && pwd)
python_bin=${SGLANG_BENCH_PYTHON:-/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python}
output=${1:?Specify a new preflight output directory}
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
mkdir -p "$output"
output=$(realpath "$output")
stage="$sim_root/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4"
aic_source=/home/ms-admin/workspace1/yiheng/ali/aiconfigurator-v0.10.0
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$stage:$script_dir/collector_shims:$script_dir:$aic_source:$sim_root/../../python:$sim_root/src:/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/lib/python3.10/site-packages:/home/ms-admin/.local/lib/python3.10/site-packages"
export AICONFIGURATOR_COLLECTOR_TRITON_PAGE_SIZE=1
export AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND=triton
export AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE=0.5.10
export AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS=1
export SGLANG_JIT_CACHE_DIR="$output/jit-cache"
export SGLANG_CACHE_DIR="$output/cache"
export FLASHINFER_WORKSPACE_BASE="$output/flashinfer-workspace"
export TRITON_CACHE_DIR="$output/triton-cache"
unset SGLANG_FORCE_FUSED_OP_BACKEND
torch_lib=$("$python_bin" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$python_bin" - "$output" <<'PY' > "$output/smoke.log" 2>&1
import csv, json, sys, time
from pathlib import Path
import torch, sgl_kernel
from run_aiconfigurator_collector_compat import install_pr33824_triton_attention_compat
install_pr33824_triton_attention_compat()
from collector.sglang import collect_attn
assert collect_attn._PR33824_TRITON_PAGE_SIZE == 1
root=Path(sys.argv[1])
identity={'purpose':'CUDA compatibility check, not accuracy results or formal calibration',
          'started_wall_time':time.time(),'torch':torch.__version__,
          'kernel_runtime':sgl_kernel.__version__,'kernel_file':sgl_kernel.__file__,
          'page_size':1,'kv_splits':8,'dtype':'bfloat16','batch':1,'sequence':128,'heads':16,'kv_heads':8,'head_dim':128}
print(json.dumps(identity),flush=True)
for context in (True,False):
    path=root/('context_attention_perf.txt' if context else 'generation_attention_perf.txt')
    collect_attn.run_attention_torch(1,128,16,8,128,False,False,context,perf_filename=str(path),device='cuda:0')
    rows=list(csv.DictReader(path.open()))
    assert len(rows)==1 and rows[0]['kernel_source']=='triton' and float(rows[0]['latency'])>0
    print('verified',path.name,rows[0],flush=True)
identity['finished_wall_time']=time.time()
(root/'runtime_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
(root/'CUDA_PAGE1_SMOKE_COMPLETE').write_text('complete\n')
PY
echo "CUDA page=1 smoke completed: $output"
