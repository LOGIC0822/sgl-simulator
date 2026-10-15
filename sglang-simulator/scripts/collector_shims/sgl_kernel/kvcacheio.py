"""Import-compatible fallback for optional HiCache transfer kernels.

The PR #33824 scheduler imports these names on CUDA hosts even when its normal
single-GPU serving path does not use host/device KV transfers.  The installed
cu129 wheel has no compatible SM80 common-ops library for this host, so keep
import compatibility while failing explicitly if an optional HiCache path is
actually selected.
"""

from __future__ import annotations

from typing import Any


def _unavailable(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "HiCache KV-transfer kernels require an SM80- and Torch-ABI-matched "
        "sglang-kernel build. This A100 validation uses the normal GPU-resident "
        "KV-cache path."
    )


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return _unavailable
