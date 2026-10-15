"""Import-only fallback for AIConfigurator BF16 collection on this A100 host."""

from __future__ import annotations

from typing import Any


def _unavailable(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "An unavailable sglang-kernel symbol was invoked during the A100 "
        "AIConfigurator collection. Restrict this path to BF16 Torch/Triton "
        "operators or install an SM80- and Torch-ABI-matched kernel wheel."
    )


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return _unavailable
