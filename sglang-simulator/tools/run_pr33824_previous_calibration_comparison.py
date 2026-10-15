#!/usr/bin/env python3
"""Score historical ML/AIC calibration on the same new real holdout batches.

This read-only comparison never recollects from the old experiment, refits a
historical bundle, or starts a GPU server. It waits for the new real collection
to finish so CPU scoring cannot compete with the real scheduler measurement.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def metadata(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def complete(path: Path) -> bool:
    return path.is_file() and path.read_text().strip() == "complete"


def validate(root: Path, previous: Path) -> list[str]:
    current, old = metadata(root / "run_metadata.txt"), metadata(previous / "run_metadata.txt")
    if current["source_revision"] != old["source_revision"]:
        raise ValueError("historical source revision differs; not a calibration-only comparison")
    if Path(current["kernel_compat_stage"]).resolve() != Path(old["kernel_compat_stage"]).resolve():
        raise ValueError("historical kernel stage differs")
    if not complete(previous / "FINAL_REPORT_COMPLETE"):
        raise ValueError("historical reference experiment is incomplete")
    models = current["models"].split(",")
    if not set(models) <= set(old["models"].split(",")):
        raise ValueError("historical reference does not contain all selected models")
    if set(current["heldout_seeds"].split(",")) & set(old["training_seeds"].split(",")):
        raise ValueError("new holdout seeds overlap historical training")
    config = previous / "aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root, previous = args.results_root.resolve(), args.previous_root.resolve()
    if args.interval <= 0:
        parser.error("interval must be positive")
    models = validate(root, previous)
    if args.validate_only:
        print(f"Compatible historical calibration reference: {len(models)} models")
        return
    target = root / "previous_calibration"
    target.mkdir(exist_ok=True)
    (target / "owner.json").write_text(json.dumps({"pid": os.getpid(), "created_at": datetime.now().astimezone().isoformat()}, indent=2))
    old_meta = metadata(previous / "run_metadata.txt")
    (target / "reference.json").write_text(json.dumps({
        "previous_root": str(previous), "source_revision": old_meta["source_revision"],
        "kernel_stage": old_meta["kernel_compat_stage"],
        "training_requests_per_model": int(old_meta["training_requests_total"]),
        "training_cases": old_meta["training_case"],
        "evaluation": "same new holdout ScheduleBatch rows; teacher-forced only",
        "limitation": "training workload coverage and request count both change; not a request-count-only comparison",
    }, indent=2, ensure_ascii=False))
    while not complete(root / "REAL_ML_ALL_MODELS_COMPLETE"):
        print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} waiting for all real/ML collection", flush=True)
        time.sleep(args.interval)
    sim = Path(__file__).resolve().parents[1]
    manifest = root / "model_manifest.tsv"
    with manifest.open() as source:
        model_paths = {row["model"]: row["model_path"] for row in csv.DictReader(source, delimiter="\t")}
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONNOUSERSITE="1")
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec and torch_spec.origin:
        env["LD_LIBRARY_PATH"] = str(Path(torch_spec.origin).parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = ":".join([
        old_meta["kernel_compat_stage"], str(sim / "src"), str(sim.parents[1] / "python"),
        str(Path('/home/ms-admin/workspace1/yiheng/ali/aiconfigurator-v0.10.0/src')),
        env.get("SGLANG_EXPERIMENT_EXTRA_PYTHONPATH", ""),
    ])
    config = previous / "aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"
    for model in models:
        for method in ("ml", "aic"):
            output = target / "models" / model / f"{method}_teacher_forced"
            marker = output / "SCORING_COMPLETE"
            if complete(marker):
                continue
            if output.exists():
                raise RuntimeError(f"incomplete scoring attempt requires an explicit archive before resume: {output}")
            command = [sys.executable, str(sim / "tools" / f"score_pr33824_{'aic_latency' if method == 'aic' else 'latency'}_model.py"),
                       "--data-root", str(root / "models" / model / "holdout/real"), "--out-dir", str(output), "--forward-modes", "1,2"]
            if method == "ml":
                bundle = previous / "models" / model / "calibration/training" / f"{model}_hgbmono_p50.pkl"
                if not bundle.is_file():
                    raise FileNotFoundError(bundle)
                command.extend(["--model-bundle", str(bundle)])
            else:
                command.extend(["--model-path", model_paths[model], "--sim-config", str(config)])
            output.parent.mkdir(parents=True, exist_ok=True)
            (output.parent / f"{method}.command.json").write_text(json.dumps(command, indent=2))
            print(f"{model} historical {method} scoring started", flush=True)
            with (output.parent / f"{method}.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            marker.write_text("complete\n")
            print(f"{model} historical {method} scoring complete", flush=True)
    (target / "PREVIOUS_CALIBRATION_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--results-root" in sys.argv:
            failure_root = Path(sys.argv[sys.argv.index("--results-root") + 1]) / "previous_calibration"
            if failure_root.is_dir():
                (failure_root / "COMPARISON_FAILED").write_text(f"{type(error).__name__}: {error}\n")
        raise
