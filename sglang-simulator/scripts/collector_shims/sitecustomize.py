"""Startup hook for the CUDA-12 AIC collector compatibility path.

The collector launches each case with Python's ``spawn`` start method.  A
module registered by the parent process is not inherited, so this hook creates
the import-only FP8 placeholder in every child interpreter before AIC imports
its collector module.  It has no effect unless explicitly enabled.
"""

from __future__ import annotations

import os
import sys
import types


if os.environ.get("AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS") == "1":
    # AIC 0.10's SGLang MoE collector initializes the historical module-global
    # ``_global_server_args`` before importing fused-MoE kernels.  PR #33824
    # replaced that private field with ``get_global_server_args()``.  Preserve
    # the collector's mock-only initialization contract in this isolated
    # collector process; the normal SGLang server never imports this hook.
    import sglang.srt.server_args as server_args

    if not hasattr(server_args, "_global_server_args"):
        server_args._global_server_args = None

        def get_collector_server_args() -> object:
            value = server_args._global_server_args
            if value is None:
                raise ValueError("Global server args is not set yet!")
            return value

        server_args.get_global_server_args = get_collector_server_args

    # The same PR moves execution options into a published runtime-context
    # namespace.  AIC invokes kernels directly (there is no SGLang scheduler
    # process to publish that namespace), so provide the defaults read by the
    # BF16 fused-MoE path.  This is deliberately collector-only and mirrors
    # the old collector's ``MagicMock`` server-args setup.
    from types import SimpleNamespace

    import sglang.srt.runtime_context as runtime_context

    runtime_context._CONTEXT._config_bags = {
        "exec": SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
            moe=SimpleNamespace(
                deepep_dispatcher_output_dtype="auto",
                disable_shared_experts_fusion=False,
                enable_eplb=False,
                enable_fused_moe_sum_all_reduce=False,
                enable_waterfill=False,
                ep_join_mode="none",
                ep_num_redundant_experts=0,
                flashinfer_mxfp4_moe_precision="default",
                init_expert_location="trivial",
                kt_weight_path=None,
                moe_runner_backend="auto",
            ),
            overlap=SimpleNamespace(),
        )
    }

    # The installed ``sglang-kernel=0.4.6.post1+cu129`` wheel provides SM90/
    # SM100 common-ops binaries, not an SM80 A100 binary, and does not load
    # against the active Torch 2.9.1+cu126 CUDA ABI.  PR #33824 ships the same
    # align kernel as a JIT source fallback; use that exact source instead of
    # the unavailable binary extension.  The experiment has TP=EP=1, so
    # invalid expert IDs never occur in its balanced routing workload.
    import importlib

    from sglang.kernels.ops.moe.moe_align import moe_align_block_size as jit_moe_align_block_size

    moe_align_module = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"
    )

    def collector_jit_moe_align(
        topk_ids: object,
        num_experts: int,
        block_size: int,
        sorted_token_ids: object,
        expert_ids: object,
        num_tokens_post_pad: object,
        cumsum_buffer: object,
        pad_sorted_token_ids: bool = False,
        ignore_invalid_expert: bool = False,
    ) -> None:
        if ignore_invalid_expert:
            raise RuntimeError("The A100 TP=EP=1 MoE collector does not support invalid expert IDs")
        jit_moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            pad_sorted_token_ids,
        )

    moe_align_module.sgl_moe_align_block_size = collector_jit_moe_align

    # At larger batches the upstream fused-MoE path also uses the binary
    # ``moe_sum_reduce`` helper.  Its semantics are a sum across the top-k
    # axis followed by the routed scaling factor, which has a direct Torch
    # equivalent for this BF16 collector.  This keeps the measured expert
    # GEMMs and PR-provided alignment kernel on the real A100 while avoiding
    # the incompatible cu129 helper wheel.
    fused_moe_module = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    )

    def collector_torch_moe_sum_reduce(
        intermediate: object,
        output: object,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        import torch

        torch.sum(intermediate, dim=1, out=output)
        output.mul_(routed_scaling_factor)

    fused_moe_module.moe_sum_reduce = collector_torch_moe_sum_reduce

    # The AIC multiprocessing worker may publish a partial mock ServerArgs
    # after this startup hook has run.  That replaces ``_config_bags`` with an
    # ``exec`` bag that has no ``moe`` child, even though fused_moe only needs
    # the single default below.  Bind the collector-local accessor directly
    # into the already imported fused_moe module.  It is deliberately narrower
    # than changing RuntimeContext globally, and is active only in the AIC
    # compatibility process.
    collector_exec_config = runtime_context._CONTEXT._config_bags["exec"]

    def collector_get_exec() -> object:
        return collector_exec_config

    fused_moe_module.get_exec = collector_get_exec

    # This matrix runs only a single A100 with Qwen3-30B-A3B at TP=EP=1.
    # AIC's stock MoE plan expands every multi-GPU topology and three routing
    # distributions (1,863 cases), even though those rows can never be looked
    # up by this experiment.  Supply the exact single-GPU BF16 grid through an
    # opt-in hook so the measured table stays relevant and collection remains
    # practical.  Include all token counts the scheduler can emit (up to its
    # 2,048-token prefill chunk); smaller points are needed for decode batches.
    if os.environ.get("AICONFIGURATOR_COLLECTOR_QWEN3_MOE_ONLY") == "1":
        import collector.case_generator as case_generator

        def get_qwen3_single_gpu_moe_cases() -> list[object]:
            return [
                case_generator.MoeCommonTestCase(
                    num_tokens_list=[
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        48,
                        64,
                        80,
                        96,
                        128,
                        160,
                        192,
                        256,
                        320,
                        384,
                        512,
                        768,
                        1024,
                        1536,
                        2048,
                    ],
                    hidden_size=2048,
                    inter_size=768,
                    topk=8,
                    num_experts=128,
                    tp=1,
                    ep=1,
                    model_name="Qwen/Qwen3-30B-A3B",
                    token_expert_distribution="balanced",
                    power_law_alpha=0.0,
                    architecture="Qwen3MoeForCausalLM",
                )
            ]

        case_generator.get_common_moe_test_cases = get_qwen3_single_gpu_moe_cases
        # AIC's collector imports this module without its package prefix.
        # Register the same patched module under that historical name.
        sys.modules["case_generator"] = case_generator

    module_name = "sglang.srt.layers.quantization.fp8_kernel"
    if module_name not in sys.modules:
        fp8_kernel = types.ModuleType(module_name)

        def unavailable(*args: object, **kwargs: object) -> None:
            raise RuntimeError("FP8 collection is disabled in the CUDA-12 compatibility path")

        fp8_kernel.sglang_per_token_group_quant_fp8 = unavailable
        sys.modules[module_name] = fp8_kernel

    if os.environ.get("AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND") == "triton":
        # AIC uses the ``spawn`` multiprocessing method.  The parent wrapper's
        # monkey patches are therefore not inherited by attention workers;
        # install the same process-local adapter during interpreter startup.
        # This path is only active for the explicitly selected current-PR
        # Triton collector and never affects a normal SGLang server.
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from run_aiconfigurator_collector_compat import (
            install_pr33824_triton_attention_compat,
        )

        install_pr33824_triton_attention_compat()


if os.environ.get("SGLANG_SIMULATOR_ENABLE_JIT_MOE_ALIGN") == "1":
    # The isolated SM80 compatibility package intentionally contains only the
    # common AOT operators needed by the dense path.  Its ``sgl_kernel``
    # extension therefore has no ``moe_align_block_size`` entry point.  Do not
    # enable the experimental LoRA switch merely to choose the JIT source: it
    # changes the TopKOutput carrier and is incompatible with the ordinary
    # fused-MoE runner in this PR.  Instead, replace this one imported helper
    # with PR #33824's signature-compatible JIT implementation before SGLang
    # constructs worker processes.  The hook is opt-in and has no effect on
    # dense servers or AIC collection unless their launcher sets this variable.
    import importlib

    from sglang.kernels.ops.moe.moe_align import moe_align_block_size as jit_moe_align_block_size

    moe_align_module = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"
    )

    def runtime_jit_moe_align(
        topk_ids: object,
        num_experts: int,
        block_size: int,
        sorted_token_ids: object,
        expert_ids: object,
        num_tokens_post_pad: object,
        cumsum_buffer: object,
        pad_sorted_token_ids: bool = False,
        ignore_invalid_expert: bool = False,
    ) -> None:
        if ignore_invalid_expert:
            raise RuntimeError(
                "The isolated A100 MoE runtime shim does not support invalid expert IDs"
            )
        jit_moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            pad_sorted_token_ids,
        )

    moe_align_module.sgl_moe_align_block_size = runtime_jit_moe_align

    # The same intentionally small AOT package also omits ``moe_sum_reduce``.
    # In this single-GPU, unquantized BF16 path its contract is a sum over the
    # routed top-k dimension followed by the scalar routing factor.  Keep the
    # fallback adjacent to the missing alignment replacement instead of
    # enabling a broad torch backend for every fused operation.
    fused_moe_module = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    )

    def runtime_torch_moe_sum_reduce(
        intermediate: object,
        output: object,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        import torch

        torch.sum(intermediate, dim=1, out=output)
        output.mul_(routed_scaling_factor)

    fused_moe_module.moe_sum_reduce = runtime_torch_moe_sum_reduce


if os.environ.get("SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN") == "1":
    # Qwen2.5-14B-Instruct-1M activates dual-chunk attention.  The compact
    # SM80 AOT package used by the other experiments omits FA3, while this
    # A100-compatible local package provides the narrower FA2 implementation
    # required by SGLang's documented fallback branch.  Override only the
    # capability predicate; this preserves the dual-chunk model configuration
    # and routes its existing wrapper into ``flash_attn`` rather than claiming
    # that a missing FA3 binary is available.
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN requires a local flash_attn stage"
        ) from exc

    import importlib

    fa3_module = importlib.import_module(
        "sglang.kernels.ops.attention.flash_attention_v3"
    )
    fa3_module._is_fa3_supported = lambda device=None: False

    # The generic varlen wrapper already has an FA2 fallback.  Its paged-KV
    # sibling is FA3-only in this PR, although the local FA2 package exposes
    # the equivalent API.  The dual-chunk decoder uses only the common BF16
    # paged-KV subset below; reject every FA3-only feature rather than silently
    # dropping it.  The launcher selects page_size=256 for this path because
    # FA2's paged-KV interface requires page blocks that are multiples of 256.
    from flash_attn import flash_attn_with_kvcache as fa2_flash_attn_with_kvcache

    def runtime_fa2_flash_attn_with_kvcache(
        q: object,
        k_cache: object,
        v_cache: object,
        k: object = None,
        v: object = None,
        qv: object = None,
        rotary_cos: object = None,
        rotary_sin: object = None,
        cache_seqlens: object = None,
        cache_batch_idx: object = None,
        cache_leftpad: object = None,
        page_table: object = None,
        cu_seqlens_q: object = None,
        cu_seqlens_k_new: object = None,
        max_seqlen_q: object = None,
        rotary_seqlens: object = None,
        q_descale: object = None,
        k_descale: object = None,
        v_descale: object = None,
        softmax_scale: object = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        attention_chunk: object = None,
        softcap: float = 0.0,
        rotary_interleaved: bool = True,
        scheduler_metadata: object = None,
        num_splits: int = 0,
        pack_gqa: object = None,
        only_qv: bool = False,
        sm_margin: int = 0,
        return_softmax_lse: bool = False,
        sinks: object = None,
        out: object = None,
    ) -> object:
        unsupported = {
            "qv": qv,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k_new": cu_seqlens_k_new,
            "max_seqlen_q": max_seqlen_q,
            "rotary_seqlens": rotary_seqlens,
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
            "attention_chunk": attention_chunk,
            "scheduler_metadata": scheduler_metadata,
            "pack_gqa": pack_gqa,
            "sinks": sinks,
            "out": out,
        }
        if only_qv or sm_margin != 0 or any(value is not None for value in unsupported.values()):
            active = [name for name, value in unsupported.items() if value is not None]
            if only_qv:
                active.append("only_qv")
            if sm_margin != 0:
                active.append("sm_margin")
            raise RuntimeError(
                "The local FA2 compatibility path does not support: " + ", ".join(active)
            )
        return fa2_flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad,
            block_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            rotary_interleaved=rotary_interleaved,
            num_splits=num_splits,
            return_softmax_lse=return_softmax_lse,
        )

    fa3_module.flash_attn_with_kvcache = runtime_fa2_flash_attn_with_kvcache
