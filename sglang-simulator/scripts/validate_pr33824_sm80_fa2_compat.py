#!/usr/bin/env python3
"""Exercise the narrow local FA2 fallback used by the 1M dual-chunk run.

This is deliberately not a general FlashAttention test.  It checks the two
calls the compatibility path needs on an A100: BF16 varlen attention and
BF16 paged-KV attention with a 256-token physical page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.expanduser().resolve()
    if not (stage / "flash_attn" / "__init__.py").is_file():
        parser.error(f"not a FlashAttention stage: {stage}")
    sys.path.insert(0, str(stage))

    import flash_attn
    import flash_attn_2_cuda
    import torch
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this validation")
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError(
            f"the local compatibility package targets SM80, got {torch.cuda.get_device_capability()}"
        )

    torch.manual_seed(20260918)
    device = "cuda"
    dtype = torch.bfloat16
    q_heads, kv_heads, head_dim = 40, 8, 128

    q = torch.randn((16, q_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((16, kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((16, kv_heads, head_dim), device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, 16], device=device, dtype=torch.int32)
    varlen_out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        16,
        16,
        causal=True,
    )
    torch.cuda.synchronize()
    if not torch.isfinite(varlen_out).all():
        raise RuntimeError("varlen FA2 output contains a non-finite value")

    page_size = 256
    decode_q = torch.randn((1, 1, q_heads, head_dim), device=device, dtype=dtype)
    k_cache = torch.randn((1, page_size, kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.randn((1, page_size, kv_heads, head_dim), device=device, dtype=dtype)
    cache_seqlens = torch.tensor([16], device=device, dtype=torch.int32)
    block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
    decode_out, decode_lse = flash_attn_with_kvcache(
        decode_q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=False,
        return_softmax_lse=True,
    )
    torch.cuda.synchronize()
    if not torch.isfinite(decode_out).all() or not torch.isfinite(decode_lse).all():
        raise RuntimeError("paged-KV FA2 output contains a non-finite value")

    print(f"flash_attn={flash_attn.__file__}")
    print(f"flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
    print(f"torch={torch.__version__}")
    print(f"cuda_capability={torch.cuda.get_device_capability()}")
    print(f"varlen_shape={tuple(varlen_out.shape)}")
    print(f"paged_kv_shape={tuple(decode_out.shape)}")
    print("FA2_FORWARD_VALIDATION_COMPLETE")


if __name__ == "__main__":
    main()
