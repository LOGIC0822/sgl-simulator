#!/usr/bin/env python3
"""Record GPU telemetry and other resident processes; never control them."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil


def own_collector(pid: int, root: Path) -> bool:
    try:
        process = psutil.Process(pid)
        for parent in [process, *process.parents()]:
            command = " ".join(parent.cmdline())
            if "-m sglang_simulator.baseline.launch_server" in command and str(root) in command:
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return False


def query(kind: str, fields: list[str]) -> list[dict[str, str]]:
    output = subprocess.check_output(["nvidia-smi", "--id=0", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"], text=True, timeout=10)
    return [{key: value.strip() for key, value in zip(fields, row, strict=True)}
            for row in csv.reader(output.splitlines()) if row]


def sample(root: Path) -> dict:
    value = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "wall_time": time.time()}
    try:
        value["gpu"] = query("gpu", ["uuid", "memory.free", "utilization.gpu", "power.draw", "clocks.sm", "clocks.mem", "temperature.gpu"])[0]
        processes = query("compute-apps", ["pid", "process_name", "used_gpu_memory"])
        for process in processes:
            process["own_real_collector"] = own_collector(int(process["pid"]), root)
        value["processes"] = processes
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        value["error"] = f"{type(error).__name__}: {error}"
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be positive")
    root = args.results_root.absolute()
    log = root / "hardware/gpu_telemetry.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with log.open("a") as output:
            output.write(json.dumps(sample(root), ensure_ascii=False) + "\n")
        if args.once or (root / "FINAL_REPORT_COMPLETE").is_file():
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
