#!/usr/bin/env python3
"""Generate dense-attention collection cases from local checkpoints.

The request lengths are deliberately wider than the serving matrix so the
Silicon table has nearby prefill/decode keys.  Each model topology comes from
a selected checkpoint's ``config.json``.  The collector also emits a small
set of explicitly labelled, same-KV-head support shapes.  AIConfigurator's
silicon lookup interpolates across ``(num_heads, total_sequence, batch)``;
without these support shapes a three-model matrix leaves each KV-head slice
with a degenerate num-head axis and cannot answer non-exact serving shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_pr33824_aic_gemm_cases import parse_models


CONTEXT_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
CONTEXT_LENGTHS = (1, 16, 32, 64, 128, 256, 512, 1024)
DECODE_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
DECODE_HISTORY_LENGTHS = (1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047)


def topology(config: dict[str, object]) -> tuple[int, int, int]:
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    return heads, kv_heads, head_dim


def interpolation_support_topologies(
    model_topologies: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """Add measured, same-KV-head shapes needed by AIC's 3-D lookup.

    These are *calibration support points*, not claims about a checkpoint
    architecture.  Keeping the KV head count and head dimension unchanged
    makes them part of the same AIC data slice as the real topology while
    varying only the lookup's num-head axis.  The real model shape is always
    retained exactly.
    """
    support: set[tuple[int, int, int]] = set(model_topologies)
    for heads, kv_heads, head_dim in model_topologies:
        # Ratios 2/4/8/16 give values on both sides of common GQA shapes
        # (for example Qwen2.5-3B: 8, 16, 32 with 2 KV heads).
        for ratio in (2, 4, 8, 16):
            support.add((kv_heads * ratio, kv_heads, head_dim))
        support.add((heads, kv_heads, head_dim))
    return support


def values(items: tuple[int, ...]) -> str:
    return "[" + ", ".join(map(str, items)) + "]"


def rule(*, phase: str, heads: int, kv_heads: int, head_dim: int) -> str:
    if phase == "context":
        batch_sizes = values(CONTEXT_BATCH_SIZES)
        lengths = values(CONTEXT_LENGTHS)
        context = "true"
        fp8_context = "false"
    else:
        batch_sizes = values(DECODE_BATCH_SIZES)
        lengths = values(DECODE_HISTORY_LENGTHS)
        context = "false"
        fp8_context = "false"
    return (
        "      - fields: [batch_size, input_len, num_heads, num_key_value_heads, head_dim, "
        "use_fp8_kv_cache, use_fp8_context_fmha, is_context_phase, window_size]\n"
        "        match: {"
        f"batch_size: {batch_sizes}, input_len: {lengths}, num_heads: {heads}, "
        f"num_key_value_heads: {kv_heads}, head_dim: {head_dim}, "
        "use_fp8_kv_cache: false, "
        f"use_fp8_context_fmha: {fp8_context}, is_context_phase: {context}, window_size: 0}}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/mnt/data/models"))
    parser.add_argument("--models", required=True, help="comma-separated checkpoint directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = parse_models(args.models)
    model_topologies: set[tuple[int, int, int]] = set()
    for model in models:
        config_path = args.checkpoint_root / model / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing model config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("num_experts") or config.get("num_local_experts"):
            raise ValueError(f"{model} is MoE; this dense-only generator cannot calibrate it")
        model_topologies.add(topology(config))

    topologies = interpolation_support_topologies(model_topologies)

    sections = []
    for phase in ("context", "generation"):
        rules = "\n".join(
            rule(phase=phase, heads=heads, kv_heads=kv_heads, head_dim=head_dim)
            for heads, kv_heads, head_dim in sorted(topologies)
        )
        sections.append(f"  attention_{phase}:\n    rules:\n{rules}")
    document = (
        "schema_version: 1\n"
        "architecture: A100PCIeSelectedDenseAttentionCalibration\n"
        "description: >\n"
        "  BF16 dense-attention model topologies read from the selected local checkpoint\n"
        "  configurations, plus same-KV-head interpolation support shapes. All are\n"
        "  measured on the local A100 PCIe with current-PR Triton attention in eager\n"
        "  operation-collection mode.\n\n"
        "all_frameworks_op_cases:\n"
        + "\n\n".join(sections)
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(
        f"models={len(models)} model_topologies={len(model_topologies)} "
        f"collected_topologies={len(topologies)} "
        f"context_cases={len(topologies) * len(CONTEXT_BATCH_SIZES) * len(CONTEXT_LENGTHS)} "
        f"decode_cases={len(topologies) * len(DECODE_BATCH_SIZES) * len(DECODE_HISTORY_LENGTHS)}"
    )


if __name__ == "__main__":
    main()
