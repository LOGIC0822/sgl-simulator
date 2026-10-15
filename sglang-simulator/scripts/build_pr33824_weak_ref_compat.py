#!/usr/bin/env python3
"""Build the minimal CUDA-graph weak-reference extension for this PyTorch ABI.

The output directory is supplied by the experiment launcher and retained with
the run metadata.  The extension has no CUDA source: it uses PyTorch's current
C++ ABI and registers one CUDA dispatch operator that creates a non-owning
Tensor view for SGLang piecewise CUDA-graph capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Persistent build directory below the immutable experiment root.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(__file__).resolve().parents[1] / "native" / "sglang_weak_ref_compat.cpp"
    if not source.is_file():
        raise FileNotFoundError(f"Missing compatibility source: {source}")

    existing = sorted(output_dir.glob("sglang_weak_ref_compat*.so"))
    if existing:
        print(existing[0])
        return

    # is_python_module=False calls torch.ops.load_library after a successful
    # build, so no Python module init symbol is needed.
    load(
        name="sglang_weak_ref_compat",
        sources=[str(source)],
        build_directory=str(output_dir),
        extra_cflags=["-O2"],
        with_cuda=False,
        is_python_module=False,
        verbose=True,
    )
    built = sorted(output_dir.glob("sglang_weak_ref_compat*.so"))
    if len(built) != 1:
        raise RuntimeError(f"Expected one compatibility library in {output_dir}, found {built}")
    print(built[0])


if __name__ == "__main__":
    main()
