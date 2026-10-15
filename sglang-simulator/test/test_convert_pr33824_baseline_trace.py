"""Contracts for converting collected requests into simulator traces."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "convert_pr33824_baseline_trace.py"
)
SPEC = importlib.util.spec_from_file_location("baseline_trace_converter", SCRIPT)
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def test_convert_preserves_tokens_output_lengths_and_relative_arrivals(tmp_path):
    raw = tmp_path / "TP0.raw_request.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "rid": "later",
                        "created_time": 12.75,
                        "input_ids": [9, 8],
                        "output_ids": [7],
                    }
                ),
                json.dumps(
                    {
                        "rid": "earlier",
                        "client_created_time": 12.25,
                        "input_ids": [1, 2, 3],
                        "output_ids": [4, 5],
                    }
                ),
            ]
        )
        + "\n"
    )

    rows, manifest = converter.convert_raw_requests(raw)

    assert [row["metadata"]["source_rid"] for row in rows] == ["earlier", "later"]
    assert rows[0]["prompt"] == [1, 2, 3]
    assert rows[0]["output_len"] == 2
    assert rows[0]["timestamp"] == 0.0
    assert rows[1]["timestamp"] == 500.0
    assert manifest["input_tokens"] == 5
    assert manifest["output_tokens"] == 3


def test_convert_rejects_unfinished_requests(tmp_path):
    raw = tmp_path / "TP0.raw_request.jsonl"
    raw.write_text(
        json.dumps(
            {
                "rid": "unfinished",
                "created_time": 1.0,
                "input_ids": [1],
                "output_ids": [],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="output_ids must be a non-empty"):
        converter.convert_raw_requests(raw)
