#!/usr/bin/env python3
"""Wait for one experiment's memory budget without touching other processes."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path


MODEL_MINIMUM_FREE_MIB = {
    "Qwen3-0.6B": 6144,
    "Qwen2.5-3B-Instruct": 12288,
    "llama-2-7b": 57344,
    "Mistral-7B-Instruct-v0.2": 57344,
    "Qwen2.5-7B-Instruct": 57344,
    "Qwen3-8B": 57344,
    "Qwen2.5-14B-Instruct": 57344,
    "Qwen2.5-14B-Instruct-1M": 57344,
    "Qwen3-14B": 57344,
    "Meta-Llama-3.1-70B-Instruct-AWQ": 57344,
    "Qwen3-30B-A3B": 73728,
    "Qwen3-30B-A3B-Instruct-2507": 73728,
    "Qwen3-32B": 73728,
}


def required_memory(total_mib: int, fraction: float, margin_mib: int) -> int:
    if total_mib <= 0 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("positive GPU memory and a finite fraction in (0, 1) required")
    if margin_mib < 0:
        raise ValueError("margin must be nonnegative")
    required = math.ceil(total_mib * fraction) + margin_mib
    if required > total_mib:
        raise ValueError("memory fraction plus margin exceeds total GPU memory")
    return required


def snapshot(index: int) -> dict:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={index}",
         "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    uuid, total, free, utilization = [value.strip() for value in output.split(",")]
    processes = subprocess.check_output(
        ["nvidia-smi", f"--id={index}",
         "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    return {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "uuid": uuid, "total_mib": int(total), "free_mib": int(free),
            "utilization_pct": int(utilization), "other_processes": processes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--margin-mib", type=int, default=1024)
    parser.add_argument("--minimum-free-mib", type=int,
                        help="Model-specific free-memory floor; SGLang fractions apply to pre-load free memory, not card capacity")
    parser.add_argument("--model", choices=sorted(MODEL_MINIMUM_FREE_MIB),
                        help="Enforce the matrix model's safe free-memory floor")
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.interval <= 0 or args.index < 0:
        parser.error("interval must be positive and GPU index nonnegative")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        observed = snapshot(args.index)
        required = required_memory(observed["total_mib"], args.fraction, args.margin_mib)
        if args.minimum_free_mib is not None:
            if not 0 < args.minimum_free_mib <= observed["total_mib"]:
                raise ValueError("minimum free memory must be within GPU capacity")
            required = args.minimum_free_mib
        if args.model is not None:
            # In particular 7B @0.35 cannot load weights from only 24 GiB
            # free: 0.35 applies to FREE memory, not the full 80 GiB card.
            required = max(required if args.minimum_free_mib is not None else 0,
                           MODEL_MINIMUM_FREE_MIB[args.model])
        observed.update(required_mib=required, state="ready" if observed["free_mib"] >= required else "waiting")
        with args.log.open("a", encoding="utf-8") as output:
            output.write(json.dumps(observed, ensure_ascii=False) + "\n")
        print(json.dumps(observed, ensure_ascii=False), flush=True)
        if observed["state"] == "ready":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
