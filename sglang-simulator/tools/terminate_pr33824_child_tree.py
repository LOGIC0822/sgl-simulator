#!/usr/bin/env python3
"""Terminate only a verified benchmark server and its captured descendants."""
from __future__ import annotations
import argparse
import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    try:
        parent = psutil.Process(args.pid)
        command = " ".join(parent.cmdline())
    except psutil.NoSuchProcess:
        return
    allowed = ("-m sglang_simulator.baseline.launch_server", "-m sglang_simulator.simulation.sglang.launch_server")
    if not any(name in command for name in allowed):
        raise RuntimeError(f"refusing to terminate an unrelated process: {args.pid}")
    processes = [parent, *parent.children(recursive=True)]
    # psutil checks create_time when signalling, protecting against PID reuse.
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


if __name__ == "__main__":
    main()
