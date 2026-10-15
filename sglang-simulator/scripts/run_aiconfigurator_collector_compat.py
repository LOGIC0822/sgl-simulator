#!/usr/bin/env python3
"""Run AIConfigurator's collector with the local SGLang source tree.

AIConfigurator 0.10 validates the installed SGLang distribution version before
it resolves collector modules.  This wrapper changes only that metadata lookup:
the actual imported SGLang code remains the checked-out PR #33824 source.
"""

from __future__ import annotations

import importlib.metadata
import functools
import os
import runpy
import sys
import types
from pathlib import Path
from typing import Any, Callable


def triton_collection_page_size() -> int:
    value = int(os.environ.get("AICONFIGURATOR_COLLECTOR_TRITON_PAGE_SIZE", "64"))
    if value <= 0:
        raise ValueError("collector Triton page size must be positive")
    return value


def with_fixed_page_size(function: Callable[..., Any], page_size: int) -> Callable[..., Any]:
    @functools.wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        if "page_size" in kwargs and kwargs["page_size"] != page_size:
            raise ValueError("attention call page size differs from declared collection policy")
        kwargs["page_size"] = page_size
        return function(*args, **kwargs)
    return call


def install_pr33824_triton_attention_compat() -> None:
    """Adapt AIC 0.10's isolated attention mock to the current PR API.

    This is deliberately a process-local adapter: it neither edits the AIC
    checkout nor changes SGLang production code.  It is only used when the
    collector is explicitly asked for ``triton`` and keeps CUDA graphs off,
    since the collector measures one eager attention operation at a time.
    """
    import collector.sglang.collect_attn as collect_attn
    if getattr(collect_attn, "_PR33824_TRITON_COMPAT_INSTALLED", False):
        return

    import torch
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
    from sglang.srt.runtime_context import _CONTEXT, _ConfigBag
    page_size = triton_collection_page_size()

    def bag(name: str, **fields: object) -> _ConfigBag:
        value = _ConfigBag(name)
        for field, field_value in fields.items():
            value._set(field, field_value)
        return value

    # The full server publishes these namespaces.  AIC deliberately does not
    # construct ServerArgs, so publish the small read-only subset used by the
    # Triton backend.  All graph phases are disabled for this eager collector.
    spec = bag(
        "spec",
        speculative_eagle_topk=0,
        speculative_num_draft_tokens=None,
        speculative_num_steps=0,
        speculative_algorithm=None,
    )
    kernel = bag("exec.kernel", triton_attention_num_kv_splits=8)
    deterministic = bag(
        "exec.deterministic", enable_deterministic_inference=False
    )
    graph = bag(
        "exec.graph",
        cuda_graph_config=CudaGraphConfig.from_dict(
            {
                "decode": {"backend": "disabled"},
                "prefill": {"backend": "disabled"},
            }
        ),
    )
    execution = bag("exec")
    execution._set_sub("kernel", kernel)
    execution._set_sub("deterministic", deterministic)
    execution._set_sub("graph", graph)
    schedule = bag("schedule", chunked_prefill_size=-1, page_size=page_size)
    _CONTEXT._config_bags = {
        **(_CONTEXT._config_bags or {}),
        "spec": spec,
        "exec": execution,
        "schedule": schedule,
    }
    _CONTEXT._publish_role = None

    class ParallelContext:
        attn_tp_size = 1
        attn_dcp_size = 1
        attn_dcp_rank = 0
        dcp_group = None

    # AIC's old mock used the former ForwardBatch constructor and a bare
    # req_to_token mapping.  Current SGLang requires a translator even for
    # the unchanged (non-unified) ID space.  This implementation preserves
    # those physical IDs and uses eager tensor copies; its own cost is outside
    # the timed attention function, just as AIC's original mock setup was.
    class IdentityKVIndexTranslator:
        is_translating = False

        def __init__(self, runner: object) -> None:
            self.runner = runner

        def fill_packed_read_stream(
            self,
            *,
            req_pool_indices: torch.Tensor,
            seq_lens: torch.Tensor,
            indptr: torch.Tensor,
            total_tokens: int,
            out: torch.Tensor,
            kv_start_idx: torch.Tensor | None = None,
            sliding_window: bool = False,
        ) -> bool:
            if sliding_window:
                raise RuntimeError("AIC Triton collector does not collect sliding-window attention")
            pool = self.runner.req_to_token_pool
            if pool is None:
                raise RuntimeError("AIC Triton collector has no req_to_token pool")
            batch_size = int(seq_lens.numel())
            if req_pool_indices.numel() < batch_size:
                raise RuntimeError("request-index buffer is shorter than sequence-length buffer")
            if total_tokens > out.numel():
                raise RuntimeError("KV output buffer is shorter than the requested token stream")
            for index in range(batch_size):
                start = int(indptr[index].item())
                end = int(indptr[index + 1].item())
                length = end - start
                source_start = int(kv_start_idx[index].item()) if kv_start_idx is not None else 0
                request = int(req_pool_indices[index].item())
                out[start:end].copy_(
                    pool.req_to_token[request, source_start : source_start + length],
                    non_blocking=False,
                )
            return False

        def sliding_window_write_loc_for(
            self, out_cache_loc: torch.Tensor | None
        ) -> torch.Tensor | None:
            return out_cache_loc

    original_config_init = collect_attn.MockModelConfig.__init__

    def config_init(self: object, *args: object, **kwargs: object) -> None:
        original_config_init(self, *args, **kwargs)
        self.v_head_dim = self.head_dim
        self.swa_v_head_dim = self.head_dim
        self.linear_attn_registry_result = None
        self.hf_config.get_text_config = lambda: self.hf_config

    collect_attn.MockModelConfig.__init__ = config_init
    collect_attn.MockModelConfig.get_max_num_attention_heads = (
        lambda self: self.num_attention_heads
    )
    collect_attn.MockModelConfig.get_num_kv_heads = (
        lambda self, tp_size, dcp_size=1: self.num_key_value_heads // tp_size
    )

    original_server_args_init = collect_attn.MockServerArgs.__init__

    def server_args_init(self: object, *args: object, **kwargs: object) -> None:
        original_server_args_init(self, *args, **kwargs)
        self.enable_lean_attention = None

    collect_attn.MockServerArgs.__init__ = server_args_init
    original_runner_init = collect_attn.MockModelRunner.__init__

    def runner_init(self: object, *args: object, **kwargs: object) -> None:
        original_runner_init(self, *args, **kwargs)
        self.kv_index_translator = IdentityKVIndexTranslator(self)

    collect_attn.MockModelRunner.__init__ = runner_init
    original_forward_batch = collect_attn.ForwardBatch

    def forward_batch_compat(*args: object, **kwargs: object) -> object:
        attn_backend = kwargs.pop("attn_backend", None)
        value = original_forward_batch(*args, **kwargs)
        value.attn_backend = attn_backend
        return value

    collect_attn.ForwardBatch = forward_batch_compat
    original_radix_forward = collect_attn.RadixAttention.forward

    def radix_forward_compat(self: object, *args: object, **kwargs: object) -> object:
        # PR #33824 publishes the active attention backend through a scoped
        # ForwardContext.  The older AIC harness calls one layer directly, so
        # recreate exactly that small runner scope around the layer invocation.
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        forward_batch = args[3] if len(args) > 3 else kwargs["forward_batch"]
        # The former collector used zero as "no sliding window".  Current
        # SGLang reserves non-negative values for an active window, while -1
        # denotes ordinary full attention.
        if self.sliding_window_size == 0:
            self.sliding_window_size = -1
        with forward_context(ForwardContext(attn_backend=forward_batch.attn_backend)):
            return original_radix_forward(self, *args, **kwargs)

    collect_attn.RadixAttention.forward = radix_forward_compat
    # AIC normally selects FlashAttention on SM80.  Only its isolated
    # collector is redirected; the hardware target remains the actual SM80
    # specified on the command line and stored in the output table.
    collect_attn.get_sm_version = lambda: 110
    import sglang.srt.layers.attention.triton_backend as triton_backend

    triton_backend.get_parallel = lambda: ParallelContext()
    collect_attn.run_attention_torch = with_fixed_page_size(collect_attn.run_attention_torch, page_size)
    collect_attn._PR33824_TRITON_PAGE_SIZE = page_size
    collect_attn._PR33824_TRITON_COMPAT_INSTALLED = True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_aiconfigurator_collector_compat.py COLLECT_PY [args...]")

    collector_path = Path(sys.argv[1]).resolve()
    if not collector_path.is_file():
        raise SystemExit(f"collector file does not exist: {collector_path}")

    override = os.environ.get("AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE")
    if not override:
        raise SystemExit("AICONFIGURATOR_COLLECTOR_SGLANG_VERSION_OVERRIDE must be set")

    real_version = importlib.metadata.version

    def version(distribution_name: str) -> str:
        if distribution_name == "sglang":
            return override
        return real_version(distribution_name)

    if os.environ.get("AICONFIGURATOR_COLLECTOR_ENABLE_CUDA12_SHIMS") == "1":
        # AIC 0.10 imports this post-PR helper even for its BF16-only path.
        # The collector's GEMM loop does not call it when AIC_COLLECT_GEMM_TYPES
        # is bfloat16, so an import-compatible placeholder preserves the actual
        # Torch BF16 and FlashInfer attention measurements.
        fp8_kernel = types.ModuleType("sglang.srt.layers.quantization.fp8_kernel")

        def unavailable(*args: object, **kwargs: object) -> None:
            raise RuntimeError("FP8 collection is disabled in the CUDA-12 compatibility path")

        fp8_kernel.sglang_per_token_group_quant_fp8 = unavailable
        sys.modules[fp8_kernel.__name__] = fp8_kernel

    sys.path.insert(0, str(collector_path.parent.parent))
    sys.path.insert(0, str(collector_path.parent))
    importlib.metadata.version = version
    if os.environ.get("AICONFIGURATOR_COLLECTOR_ATTENTION_BACKEND") == "triton":
        install_pr33824_triton_attention_compat()
    sys.argv = [str(collector_path), *sys.argv[2:]]
    runpy.run_path(str(collector_path), run_name="__main__")


if __name__ == "__main__":
    main()
