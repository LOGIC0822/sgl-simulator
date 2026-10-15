#!/usr/bin/env python3
"""Convert collected real-SGLang requests into an Autobench replay trace.

The baseline collector retains the token IDs and completed output lengths that
reached the real scheduler.  This converter keeps those values, normalizes the
first arrival to zero, and writes the public ``autobench`` JSONL contract used
by ``benchmark/simulator/bench_serving.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield line_number, record


def _as_token_ids(value: Any, *, field: str, path: Path, line_number: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty token-ID list at {path}:{line_number}")
    if not all(isinstance(token, int) and not isinstance(token, bool) for token in value):
        raise ValueError(f"{field} must contain only integer token IDs at {path}:{line_number}")
    return value


def _timestamp(record: dict[str, Any], *, path: Path, line_number: int) -> float:
    for key in ("created_time", "client_created_time", "server_created_time", "queue_start"):
        value = record.get(key)
        if value is None:
            continue
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(timestamp) and timestamp >= 0:
            return timestamp
    raise ValueError(
        f"record has no finite non-negative arrival time at {path}:{line_number}"
    )


def convert_raw_requests(raw_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build normalized replay rows and an audit manifest from raw collector rows."""
    converted: list[tuple[float, int, dict[str, Any]]] = []
    for line_number, record in _load_jsonl(raw_path):
        input_ids = _as_token_ids(
            record.get("input_ids"),
            field="input_ids",
            path=raw_path,
            line_number=line_number,
        )
        output_ids = _as_token_ids(
            record.get("output_ids"),
            field="output_ids",
            path=raw_path,
            line_number=line_number,
        )
        timestamp = _timestamp(record, path=raw_path, line_number=line_number)
        rid = str(record.get("rid", ""))
        converted.append(
            (
                timestamp,
                line_number,
                {
                    "prompt": input_ids,
                    "prompt_len": len(input_ids),
                    "output_len": len(output_ids),
                    "metadata": {"source_rid": rid},
                },
            )
        )

    if not converted:
        raise ValueError(f"no completed raw requests found in {raw_path}")

    converted.sort(key=lambda item: (item[0], item[1]))
    origin = converted[0][0]
    rows: list[dict[str, Any]] = []
    for timestamp, _, row in converted:
        row["timestamp"] = (timestamp - origin) * 1000.0
        rows.append(row)

    manifest = {
        "schema_version": 1,
        "source_raw_request_jsonl": str(raw_path.resolve()),
        "request_count": len(rows),
        "arrival_time_source": "created_time|client_created_time|server_created_time|queue_start",
        "timestamp_unit": "milliseconds_since_first_collected_arrival",
        "prompt_source": "raw_request.input_ids",
        "output_length_source": "len(raw_request.output_ids)",
        "input_tokens": sum(row["prompt_len"] for row in rows),
        "output_tokens": sum(row["output_len"] for row in rows),
    }
    return rows, manifest


def write_exclusive_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-request-jsonl", required=True)
    parser.add_argument("--output", required=True, help="New Autobench JSONL path")
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="New conversion manifest path (default: <output>.manifest.json)",
    )
    args = parser.parse_args(argv)
    args.raw_request_jsonl = Path(args.raw_request_jsonl).expanduser().resolve()
    args.output = Path(args.output).expanduser().resolve()
    args.manifest_output = Path(
        args.manifest_output
        if args.manifest_output is not None
        else f"{args.output}.manifest.json"
    ).expanduser().resolve()
    if not args.raw_request_jsonl.is_file():
        parser.error(f"raw request file does not exist: {args.raw_request_jsonl}")
    if args.output.exists():
        parser.error(f"refusing to overwrite trace: {args.output}")
    if args.manifest_output.exists():
        parser.error(f"refusing to overwrite manifest: {args.manifest_output}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows, manifest = convert_raw_requests(args.raw_request_jsonl)
    write_exclusive_jsonl(args.output, rows)
    write_exclusive_json(args.manifest_output, manifest)
    print(
        f"converted {manifest['request_count']} requests "
        f"({manifest['input_tokens']} input tokens, {manifest['output_tokens']} output tokens) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
