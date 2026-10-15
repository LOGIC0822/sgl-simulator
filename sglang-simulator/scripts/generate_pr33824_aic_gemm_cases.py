#!/usr/bin/env python3
"""Generate exact BF16 linear shapes for the local 13-model AIC calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = (
    "Qwen3-0.6B",
    "Qwen2.5-3B-Instruct",
    "llama-2-7b",
    "Mistral-7B-Instruct-v0.2",
    "Qwen2.5-7B-Instruct",
    "Qwen3-8B",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-14B-Instruct-1M",
    "Qwen3-14B",
    "Meta-Llama-3.1-70B-Instruct-AWQ",
    "Qwen3-30B-A3B",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-32B",
)

# `6` is the SGLang server warm-up batch.  The other values cover all matrix
# prefill chunks and decode batch sizes, so Silicon interpolation never has to
# invent an unmeasured warm-up or serving shape.
M_VALUES = (1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 1280, 2048, 4096, 8192)


def gemm_shapes(config: dict[str, object]) -> set[tuple[int, int]]:
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    kv_heads = int(config.get("num_key_value_heads") or heads)
    vocab = int(config["vocab_size"])
    q_size = heads * head_dim
    shapes = {
        # QKV, attention output, and LM head.  AIC's GEMM convention is M,N,K.
        # Do not assume `num_attention_heads * head_dim == hidden_size`: Qwen3
        # 0.6B advertises a 128-wide head dimension, so its physical QKV
        # projection is 4096x1024 even though hidden_size is only 1024.
        (q_size + 2 * kv_heads * head_dim, hidden),
        (hidden, q_size),
        (vocab, hidden),
    }
    if config.get("num_experts"):
        intermediate = int(config["moe_intermediate_size"])
    else:
        intermediate = int(config["intermediate_size"])
    # Gate/up projection followed by down projection.
    shapes.add((2 * intermediate, hidden))
    shapes.add((hidden, intermediate))
    return shapes


def parse_models(value: str | None) -> tuple[str, ...]:
    """Keep a short calibration run explicit and reject accidental typos."""
    if not value:
        return MODELS
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown calibration models: {', '.join(unknown)}")
    if not selected:
        raise ValueError("--models must name at least one model")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/mnt/data/models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        help="comma-separated checkpoint directory names; defaults to the historical all-13 set",
    )
    args = parser.parse_args()
    models = parse_models(args.models)

    shapes: set[tuple[int, int]] = set()
    absent: list[str] = []
    for model in models:
        config_path = args.checkpoint_root / model / "config.json"
        if not config_path.is_file():
            absent.append(model)
            continue
        shapes.update(gemm_shapes(json.loads(config_path.read_text(encoding="utf-8"))))
    if absent:
        raise SystemExit(f"missing model configurations: {', '.join(absent)}")

    rules = []
    batch_values = ", ".join(map(str, M_VALUES))
    for n, k in sorted(shapes):
        rules.append(
            "      - fields: [gemm_type, batch_size, N, K]\n"
            # The collector treats scalar match values as unconstrained.  Keep
            # singleton dimensions as YAML lists so this plan filters the base
            # cases down to exactly the requested N,K pairs.
            f"        match: {{gemm_type: bfloat16, batch_size: [{batch_values}], N: [{n}], K: [{k}]}}"
        )
    document = (
        "schema_version: 1\n"
        "architecture: A100PCIeAll13ExactGemmCalibration\n"
        "description: >\n"
        "  Exact BF16 linear operator shapes derived from the 13 local checkpoint\n"
        "  configurations, including QKV, attention output, MLP, MoE expert MLP,\n"
        "  and LM head. Values are measured on the local A100 PCIe.\n\n"
        "all_frameworks_op_cases:\n"
        "  gemm:\n"
        "    rules:\n"
        + "\n".join(rules)
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"models={len(models)} unique_nk={len(shapes)} cases={len(shapes) * len(M_VALUES)}")


if __name__ == "__main__":
    main()
