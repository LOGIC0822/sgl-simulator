"""Fail-closed optional quantization symbols for the BF16 dense validation."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Quantized execution requires a complete ABI-matched sglang-kernel build.")

    return unavailable
