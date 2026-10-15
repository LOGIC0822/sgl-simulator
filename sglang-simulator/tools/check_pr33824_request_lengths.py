#!/usr/bin/env python3
"""Check text re-tokenization against context bounds without launching a GPU."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from sglang.benchmark.datasets.random import sample_random_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    metadata = dict(line.split("=", 1) for line in (args.results_root / "run_metadata.txt").read_text().splitlines() if "=" in line)
    rows = []
    for model in metadata["models"].split(","):
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_root / model, local_files_only=True)
        for seed in map(int, metadata["heldout_seeds"].split(",")):
            random.seed(seed)
            np.random.seed(seed)
            requests = sample_random_requests(input_len=1536, output_len=384, num_prompts=64,
                                             range_ratio=0, tokenizer=tokenizer, dataset_path="",
                                             random_sample=False, return_text=True)
            lengths = [len(tokenizer.encode(request.prompt)) for request in requests]
            totals = [length + int(request.output_len) for length, request in zip(lengths, requests)]
            row = {"model": model, "case": "saturation_1536x384_inf", "seed": seed, "requests": len(requests),
                   "max_nominal_input_tokens": max(int(request.prompt_len) for request in requests),
                   "max_encoded_input_tokens": max(lengths), "max_encoded_input_plus_output": max(totals),
                   "input_over_context_2048": sum(value >= 2048 for value in lengths),
                   "input_plus_output_over_context_2048": sum(value > 2048 for value in totals)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        del tokenizer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Tokenizer-only bounds check; actual served input IDs remain authoritative", flush=True)


if __name__ == "__main__":
    main()
