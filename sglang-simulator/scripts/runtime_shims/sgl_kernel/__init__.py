"""Narrow runtime shim for the PR #33824 A100 graph-validation matrix.

The installed cu129 sglang-kernel wheel has no SM80 common-ops binary and
cannot load with the active hbf-cxl PyTorch CUDA ABI.  This module exposes only
the locally built weak-reference op needed by piecewise CUDA-graph capture.
All other symbols fail loudly, so an unsupported optimized or HiCache path
cannot silently alter a benchmark.
"""

from __future__ import annotations

import os
from typing import Any

import torch


_library_path = os.environ.get("SGLANG_WEAK_REF_COMPAT_LIBRARY")
if not _library_path:
    raise RuntimeError(
        "SGLANG_WEAK_REF_COMPAT_LIBRARY is required for the A100 "
        "piecewise-CUDA-graph runtime shim."
    )
if not os.path.isfile(_library_path):
    raise RuntimeError(f"Missing weak-reference compatibility library: {_library_path}")
torch.ops.load_library(_library_path)


def weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops.sglang_weak_ref_compat.weak_ref_tensor(tensor)


def _unavailable(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "This A100 graph-validation shim implements only weak_ref_tensor; "
        "the requested sglang-kernel operation requires a matching native build."
    )


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return _unavailable
