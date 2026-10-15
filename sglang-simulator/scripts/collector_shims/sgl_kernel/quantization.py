"""Import-only quantization symbols for the BF16 collector compatibility shim."""

from __future__ import annotations

from typing import Any


def _unavailable(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "The incompatible sglang-kernel quantization path is unavailable in this "
        "BF16-only collection."
    )


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return _unavailable
