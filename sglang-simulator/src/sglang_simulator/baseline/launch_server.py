"""Launch a real SGLang server instrumented for simulator baseline collection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sglang_simulator.baseline.collector import (
    COLLECTION_OUTPUT_DIR_ENV,
    install_frontend_collection_hooks,
    run_collection_scheduler_process,
    validate_frontend_collection_hooks,
)

# This must run before importing SGLang's HTTP server and TokenizerManager.
install_frontend_collection_hooks()


def _prepare_output_dir(parser: argparse.ArgumentParser, value: str) -> Path:
    output_dir = Path(value).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"baseline collection output directory must be empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ[COLLECTION_OUTPUT_DIR_ENV] = str(output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    # Defer SGLang imports until after install_frontend_collection_hooks().
    from sglang.srt.entrypoints.http_server import launch_server
    # Force the class definition in this parent process before checking its
    # process-local hook. The Scheduler class is defined in a spawned child.
    from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: F401
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    validate_frontend_collection_hooks()

    parser = argparse.ArgumentParser()
    collection = parser.add_argument_group("baseline collection")
    collection.add_argument(
        "--collection-output-dir",
        default=os.getenv(COLLECTION_OUTPUT_DIR_ENV),
        required=os.getenv(COLLECTION_OUTPUT_DIR_ENV) is None,
        help=(
            "Empty directory for TP*.request.jsonl, raw_request.jsonl, "
            "schedule_batch.jsonl, and manifest.json"
        ),
    )
    ServerArgs.add_cli_args(parser.add_argument_group("sglang"))

    raw_args = parser.parse_args(argv)
    output_dir = _prepare_output_dir(parser, raw_args.collection_output_dir)
    server_args = ServerArgs.from_cli_args(raw_args)
    if not getattr(server_args, "disable_overlap_schedule", False):
        parser.error(
            "--disable-overlap-schedule is required so synchronized batch "
            "latencies match ScheduleBatch execution boundaries"
        )

    print(f"SGLang Simulator baseline collection directory: {output_dir}")
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_collection_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main(sys.argv[1:])
