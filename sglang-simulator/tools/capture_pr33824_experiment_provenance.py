#!/usr/bin/env python3
"""Archive source identity and environment without initializing a GPU worker."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(source.read()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def archive_aic_sources(root: Path, out: Path) -> dict:
    """Preserve collector/SDK code, including dirty and untracked source files."""
    root = root.resolve()
    suffixes = {".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".md",
                ".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp"}
    files = set()
    for folder in (root / "collector", root / "src/aiconfigurator"):
        for path in folder.rglob("*"):
            relative = path.relative_to(root)
            if (path.is_file() and path.suffix in suffixes
                    and "__pycache__" not in relative.parts
                    and relative.parts[:4] != ("src", "aiconfigurator", "systems", "data")):
                files.add(path)
    for name in ("pyproject.toml", "README.md"):
        if (root / name).is_file():
            files.add(root / name)
    status = git(root, "status", "--short")
    (out / "aic_git_status.txt").write_text(status, encoding="utf-8")
    (out / "aic_tracked_changes.patch").write_text(git(root, "diff", "HEAD"), encoding="utf-8")
    checksums = {str(path.relative_to(root)): digest(path) for path in sorted(files)}
    (out / "aic_source_sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")
    with tarfile.open(out / "aic_collector_and_sdk_sources.tar.gz", "w:gz", dereference=True) as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(root)))
    return {"source_files": len(files), "dirty": bool(status.strip()),
            "status_file": "aic_git_status.txt", "patch_file": "aic_tracked_changes.patch",
            "sha256_file": "aic_source_sha256.json",
            "archive_file": "aic_collector_and_sdk_sources.tar.gz",
            "scope": "collector, SDK and configuration sources; excludes systems/data performance tables"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--aic-source", type=Path, required=True)
    parser.add_argument("--kernel-stage", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-kernel-checksums", type=Path)
    parser.add_argument("--require-stage-import", action="store_true")
    args = parser.parse_args()
    sim = args.sim_root.resolve()
    source = sim.parents[1]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    identity = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": sys.version,
        "source_root": str(source),
        "source_revision": git(source, "rev-parse", "HEAD").strip(),
        "aic_root": str(args.aic_source.resolve()),
        "aic_revision": git(args.aic_source, "rev-parse", "HEAD").strip(),
        "kernel_stage": str(args.kernel_stage.resolve()),
        "ninja_path": shutil.which("ninja"),
        "pythonpath": sys.path,
        "environment": {key: ("<redacted>" if key.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")) else value) for key, value in os.environ.items()
                        if key.startswith("SGLANG_") or key in ("PYTHONNOUSERSITE", "PYTHONPATH", "OMP_NUM_THREADS", "CUDA_HOME", "CUDA_VISIBLE_DEVICES")},
        "distributions": {},
        "module_origins": {},
    }
    for name in ("torch", "sglang", "sgl-kernel", "sglang-kernel", "aiconfigurator", "transformers", "triton", "numpy", "scikit-learn", "flashinfer-python", "python-docx", "ninja"):
        try:
            identity["distributions"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            identity["distributions"][name] = None
    for name in ("torch", "sglang", "sgl_kernel", "aiconfigurator", "sklearn", "gguf"):
        spec = importlib.util.find_spec(name)
        identity["module_origins"][name] = spec.origin if spec else None
    if args.require_stage_import:
        origin = identity["module_origins"]["sgl_kernel"]
        if not origin or not Path(origin).resolve().is_relative_to(args.kernel_stage.resolve()):
            raise RuntimeError(f"sgl_kernel import does not use declared compatibility stage: {origin}")
    aic_origin = identity["module_origins"]["aiconfigurator"]
    if aic_origin:
        package = Path(aic_origin).parent
        reference = args.aic_source / "src/aiconfigurator"
        checked = list(reference.rglob("*.py"))
        mismatches = [str(path.relative_to(reference)) for path in checked
                      if not (package / path.relative_to(reference)).is_file()
                      or digest(path) != digest(package / path.relative_to(reference))]
        identity["aic_import_source_match"] = {"checked_python_files": len(checked), "mismatches": mismatches}
        if mismatches:
            raise RuntimeError(f"imported AIC differs from declared checkout: {mismatches[:5]}")
    identity["aic_source_snapshot"] = archive_aic_sources(args.aic_source, out)
    (out / "identity.json").write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n")
    (out / "tracked_changes.patch").write_text(git(source, "diff", "HEAD", "--", "python/sglang", "tools/sglang-simulator"))
    (out / "git_status.txt").write_text(git(source, "status", "--short"))
    files = set()
    for folder in (sim / "src", sim / "scripts", sim / "tools", sim / "test"):
        files.update(path for path in folder.rglob("*") if path.is_file() and path.suffix in (".py", ".sh") and "__pycache__" not in path.parts)
    for relative in git(source, "diff", "--name-only", "HEAD", "--", "python/sglang").splitlines():
        path = source / relative
        if path.is_file():
            files.add(path)
    checksums = {str(path.relative_to(source)): digest(path) for path in sorted(files)}
    with tarfile.open(out / "experiment_sources.tar.gz", "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(source)))
    for path in sorted(args.kernel_stage.rglob("*.so")):
        checksums[str(path.resolve())] = digest(path)
    if args.reference_kernel_checksums:
        reference = json.loads(args.reference_kernel_checksums.read_text())
        native = {path: value for path, value in reference.items() if path.startswith("/") and path.endswith(".so")}
        if not native:
            raise RuntimeError("reference contains no native kernel checksums")
        if any(checksums.get(path) != value for path, value in native.items()):
            raise RuntimeError("native kernel differs from the real-baseline provenance; recollect before comparing")
        identity["kernel_matches_reference"] = {"reference": str(args.reference_kernel_checksums), "native_files": len(native)}
        (out / "identity.json").write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n")
    (out / "sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(f"Provenance archived: {out}; source files={len(files)}")


if __name__ == "__main__":
    main()
