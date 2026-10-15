"""Explicit profiling server entry; ordinary baseline launcher is unchanged."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from sglang_simulator.baseline.scheduler_profile import PROFILE_DIR, SKIP_ENV, STEPS_ENV, run_profiled_collection_scheduler_process


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Other arguments are forwarded to the real baseline launcher. Explicit CUDA, TP1, graph-off and overlap-off flags are required.")
    parser.add_argument("--native-profile-dir", type=Path, required=True)
    parser.add_argument("--native-profile-steps-per-phase", type=int, default=3)
    parser.add_argument("--native-profile-skip-per-phase", type=int, default=2)
    options, server_args = parser.parse_known_args(argv)
    if options.native_profile_steps_per_phase < 1 or options.native_profile_skip_per_phase < 0:
        parser.error("profile steps must be positive and skip nonnegative")
    for flag in ("--disable-cuda-graph", "--disable-overlap-schedule"):
        if server_args.count(flag) != 1:
            parser.error(f"profiling requires explicit {flag}")
    for flag, value in (("--device", "cuda"), ("--tp-size", "1")):
        if server_args.count(flag) != 1:
            parser.error(f"profiling requires explicit {flag} {value}")
        index = server_args.index(flag)
        if index+1 >= len(server_args) or server_args[index+1] != value:
            parser.error(f"profiling requires {flag} {value}")
    if "--cuda-graph-config" in server_args:
        parser.error("do not override the graph-off diagnostic with a graph config")
    directory = options.native_profile_dir.resolve()
    if directory.exists():
        parser.error("refusing to overwrite an existing profile directory")
    # SGLang imports/frontend hook installation remain in the ordinary launcher.
    # The substituted callback is picklable by its new module in spawn children.
    from sglang_simulator.baseline import launch_server

    os.environ[PROFILE_DIR] = str(directory)
    os.environ[STEPS_ENV] = str(options.native_profile_steps_per_phase)
    os.environ[SKIP_ENV] = str(options.native_profile_skip_per_phase)
    launch_server.run_collection_scheduler_process = run_profiled_collection_scheduler_process
    launch_server.main(server_args)


if __name__ == "__main__":
    main()
