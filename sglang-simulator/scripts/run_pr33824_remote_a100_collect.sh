#!/usr/bin/env bash
# Remote half of the portable A100 workflow: verify the environment and collect
# real SGLang traces plus AIC Silicon tables. Prediction is performed locally.

set -Eeuo pipefail

source_root=
aic_source=
kernel_stage=
python_bin=
model_root=
results_root=
gpu_id=0
models=
train_repeats=5
holdout_repeats=5
prompt_multiplier=2
train_cases=latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf
holdout_cases=latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf,latency_128x64_r1,saturation_1536x384_inf

usage() {
    cat <<'EOF'
Usage: run_pr33824_remote_a100_collect.sh --source-root PATH --aic-source PATH
       --kernel-stage PATH --python PATH --model-root PATH --results-root PATH
       --models CSV [--gpu-id N]

Checks the selected A100 environment, captures hardware/package provenance,
then runs comprehensive predictor phase `collect` (real traces and AIC data).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-root) source_root=${2:?}; shift 2 ;;
        --aic-source) aic_source=${2:?}; shift 2 ;;
        --kernel-stage) kernel_stage=${2:?}; shift 2 ;;
        --python) python_bin=${2:?}; shift 2 ;;
        --model-root) model_root=${2:?}; shift 2 ;;
        --results-root) results_root=${2:?}; shift 2 ;;
        --gpu-id) gpu_id=${2:?}; shift 2 ;;
        --models) models=${2:?}; shift 2 ;;
        --train-repeats) train_repeats=${2:?}; shift 2 ;;
        --holdout-repeats) holdout_repeats=${2:?}; shift 2 ;;
        --prompt-multiplier) prompt_multiplier=${2:?}; shift 2 ;;
        --train-cases) train_cases=${2:?}; shift 2 ;;
        --holdout-cases) holdout_cases=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in "$source_root" "$aic_source" "$kernel_stage" "$python_bin" "$model_root" "$results_root" "$models"; do
    [[ -n "$required" ]] || { usage >&2; exit 2; }
done
[[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "--gpu-id must be a physical GPU index" >&2; exit 2; }
[[ "$train_repeats" =~ ^[1-9][0-9]*$ && "$holdout_repeats" =~ ^[1-9][0-9]*$ && "$prompt_multiplier" =~ ^[1-9][0-9]*$ ]] || {
    echo "repeat counts and prompt multiplier must be positive integers" >&2; exit 2;
}
[[ -d "$source_root/tools/sglang-simulator" && -d "$source_root/python/sglang" ]] || {
    echo "Incomplete SGLang source bundle: $source_root" >&2; exit 1;
}
[[ -d "$aic_source/src/aiconfigurator" && -f "$aic_source/collector/collect.py" ]] || {
    echo "Incomplete AIConfigurator source bundle: $aic_source" >&2; exit 1;
}
[[ -f "$kernel_stage/sgl_kernel/sm100/common_ops.abi3.so" ]] || {
    echo "Missing SM80 compatibility library: $kernel_stage" >&2; exit 1;
}
[[ -x "$python_bin" && -d "$model_root" ]] || { echo "Python or model root is missing" >&2; exit 1; }

sim_root="$source_root/tools/sglang-simulator"
export PYTHONNOUSERSITE=1
python_bin_dir=$(dirname "$python_bin")
export PATH="$python_bin_dir:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONPATH="$kernel_stage:$aic_source/src:$sim_root/src:$source_root/python:$sim_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
torch_lib_dir=$($python_bin -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$results_root/hardware" "$results_root/environment"
nvidia-smi -L > "$results_root/hardware/nvidia-smi-L.txt"
nvidia-smi -q > "$results_root/hardware/nvidia-smi-q.txt"
nvidia-smi topo -m > "$results_root/hardware/nvidia-smi-topo.txt"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader \
    > "$results_root/hardware/gpu_identity.csv"

"$python_bin" - "$results_root/environment" "$kernel_stage" "$aic_source" "$source_root" <<'PY'
import importlib.metadata as metadata
import json
import os
import pathlib
import sys
import torch
import sglang

out = pathlib.Path(sys.argv[1])
kernel_stage = pathlib.Path(sys.argv[2]).resolve()
aic_source = pathlib.Path(sys.argv[3]).resolve()
sglang_source = pathlib.Path(sys.argv[4]).resolve() / "python"
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one CUDA-visible GPU; check CUDA_VISIBLE_DEVICES")
name = torch.cuda.get_device_name(0)
if "A100" not in name or "PCIe" not in name or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
    raise SystemExit(f"This runner's AIC system profile is A100 PCIe 80GB; found {name}")
import sgl_kernel
import aiconfigurator
if not pathlib.Path(sgl_kernel.__file__).resolve().is_relative_to(kernel_stage):
    raise SystemExit(f"sgl_kernel loaded from unexpected path: {sgl_kernel.__file__}")
if not pathlib.Path(aiconfigurator.__file__).resolve().is_relative_to(aic_source / "src"):
    raise SystemExit(f"aiconfigurator loaded from unexpected path: {aiconfigurator.__file__}")
if not pathlib.Path(sglang.__file__).resolve().is_relative_to(sglang_source):
    raise SystemExit(f"sglang loaded from unexpected path: {sglang.__file__}")
versions = {}
for package in ("torch", "sglang", "sglang-kernel", "sgl-kernel", "triton", "flashinfer-python",
                "transformers", "aiconfigurator", "scikit-learn", "numpy", "joblib"):
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
versions.update({"python": sys.version, "python_executable": sys.executable,
                 "torch_cuda": torch.version.cuda, "gpu": name,
                 "visible_gpu_count": torch.cuda.device_count(),
                 "sgl_kernel_file": sgl_kernel.__file__,
                 "aiconfigurator_file": aiconfigurator.__file__,
                 "nccl": torch.cuda.nccl.version(),
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
(out / "remote_package_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
expected = {"torch": "2.9.1+cu126", "triton": "3.5.1", "flashinfer-python": "0.6.18",
            "transformers": "5.12.1", "aiconfigurator": "0.10.0",
            "sglang": "0.0.0", "sglang-kernel": "0.4.6.post1+cu129",
            "numpy": "1.26.4", "scikit-learn": "1.6.1", "joblib": "1.4.2"}
actual = {key: versions.get(key) for key in expected}
if actual != expected or sys.version_info[:3] != (3, 10, 21) or torch.version.cuda != "12.6":
    raise SystemExit(f"Remote environment differs from the tested stack. expected={expected}, actual={actual}, python={sys.version}, torch_cuda={torch.version.cuda}")
kernel_libraries = sorted(kernel_stage.glob("sgl_kernel/**/*.so"))
if not kernel_libraries:
    raise SystemExit(f"No compiled sgl_kernel libraries found below {kernel_stage}")
with (out / "sgl_kernel_sha256.txt").open("w", encoding="utf-8") as digest_file:
    import hashlib
    for library in kernel_libraries:
        digest_file.write(f"{hashlib.sha256(library.read_bytes()).hexdigest()}  {library.relative_to(kernel_stage)}\n")
print(json.dumps(versions, indent=2))
PY

source_revision=$(git -C "$source_root" rev-parse HEAD)
aic_revision=$(git -C "$aic_source" rev-parse HEAD)
{
    printf 'source_snapshot_revision=%s\n' "$source_revision"
    printf 'aic_snapshot_revision=%s\n' "$aic_revision"
    printf 'gpu_id=%s\nmodels=%s\n' "$gpu_id" "$models"
    printf 'python=%s\nmodel_root=%s\nresults_root=%s\n' "$python_bin" "$model_root" "$results_root"
    printf 'tp_size=1\nphase=remote_collection\n'
} > "$results_root/remote_source_identity.txt"

args=(--phase collect --results-root "$results_root" --models "$models"
      --train-repeats "$train_repeats" --holdout-repeats "$holdout_repeats"
      --prompt-multiplier "$prompt_multiplier" --resume)
SGLANG_BENCH_PYTHON="$python_bin" \
SGLANG_CHECKPOINT_ROOT="$model_root" \
AICONFIGURATOR_SOURCE_ROOT="$aic_source" \
SGLANG_KERNEL_COMPAT_STAGE="$kernel_stage" \
SGLANG_COMPREHENSIVE_RESULTS_ROOT="$results_root" \
SGLANG_COMPREHENSIVE_MODELS="$models" \
SGLANG_COMPREHENSIVE_TRAIN_REPEATS="$train_repeats" \
SGLANG_COMPREHENSIVE_HOLDOUT_REPEATS="$holdout_repeats" \
SGLANG_COMPREHENSIVE_PROMPT_MULTIPLIER="$prompt_multiplier" \
SGLANG_SAME_DISTRIBUTION_TRAIN_CASES="$train_cases" \
SGLANG_SAME_DISTRIBUTION_HOLDOUT_CASES="$holdout_cases" \
SGLANG_GPU_CAPACITY_WAIT="${SGLANG_GPU_CAPACITY_WAIT:-0}" \
bash "$sim_root/scripts/run_pr33824_comprehensive_predictor_matrix.sh" "${args[@]}"

[[ -f "$results_root/A100_REMOTE_COLLECTION_COMPLETE" ]] || {
    echo "Collection did not write A100_REMOTE_COLLECTION_COMPLETE" >&2; exit 1;
}
