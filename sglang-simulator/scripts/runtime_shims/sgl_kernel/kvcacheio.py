"""Fail-closed optional HiCache symbols for the graph-validation shim."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("HiCache transfer requires a complete ABI-matched sglang-kernel build.")

    return unavailable
