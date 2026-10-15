"""Shared feature contract for ML latency prediction and offline training.

The ordered feature list is an on-disk ABI: a training bundle must carry this
exact sequence before :class:`MLTimePredictor` will load it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


ML_FEATURE_NAMES = (
    "batch_size",
    "sum_extend",
    "max_extend",
    "min_extend",
    "sum_past",
    "max_past",
    "min_past",
    "sum_extend_x_past",
    "sum_extend_squared",
    "sum_past_squared",
    "sum_attn_flops",
    "sum_extend_x_max_past",
    "log1p_sum_past",
    "log1p_sum_attn_flops",
    "batch_size_x_sum_extend",
    "max_past_minus_min_past",
    "is_decode",
    "is_prefill",
)


def extract_ml_features(
    extend_lengths: Sequence[int], past_kv_lengths: Sequence[int]
) -> list[float]:
    """Build the ordered 18-feature vector consumed by ``MLTimePredictor``."""
    if not extend_lengths:
        raise ValueError("cannot extract ML features from an empty batch")
    if len(extend_lengths) != len(past_kv_lengths):
        raise ValueError("extend and past-KV length vectors must have equal size")
    if any(value < 0 for value in (*extend_lengths, *past_kv_lengths)):
        raise ValueError("token lengths must be non-negative")

    batch_size = len(extend_lengths)
    sum_extend = sum(extend_lengths)
    sum_past = sum(past_kv_lengths)
    sum_extend_x_past = sum(
        extend * past for extend, past in zip(extend_lengths, past_kv_lengths)
    )
    sum_extend_squared = sum(value * value for value in extend_lengths)
    sum_past_squared = sum(value * value for value in past_kv_lengths)
    sum_attn_flops = sum(
        extend * (past + extend / 2)
        for extend, past in zip(extend_lengths, past_kv_lengths)
    )
    max_extend = max(extend_lengths)
    min_extend = min(extend_lengths)
    max_past = max(past_kv_lengths)
    min_past = min(past_kv_lengths)

    return [
        float(batch_size),
        float(sum_extend),
        float(max_extend),
        float(min_extend),
        float(sum_past),
        float(max_past),
        float(min_past),
        float(sum_extend_x_past),
        float(sum_extend_squared),
        float(sum_past_squared),
        float(sum_attn_flops),
        float(sum_extend * max_past),
        float(math.log1p(sum_past)),
        float(math.log1p(sum_attn_flops)),
        float(batch_size * sum_extend),
        float(max_past - min_past),
        float(int(all(value == 1 for value in extend_lengths))),
        float(int(any(value > 1 for value in extend_lengths))),
    ]
