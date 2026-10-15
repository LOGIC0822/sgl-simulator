#!/usr/bin/env python3
"""Audit AIC memoization against actual query arguments without querying AIC.

Selected functions are executed from the trusted repository's current AST.
A backend/config spy captures arguments, not latency. No GPU, fitting, SDK
import, table interpolation, or change to the formal scorer is involved.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeCapture:
    def __init__(self, **kwargs):
        self.arguments = {"prefix": 0, "seq_imbalance_correction_scale": 1.0, **kwargs}


class BackendCapture:
    def _run_static_breakdown(self, model, database, runtime, mode):
        self.signature = (mode, tuple(sorted(runtime.arguments.items())))
        return {}, {}, {}


def source_capture(sim: Path, aic_root: Path) -> tuple[dict, object, dict]:
    paths = {
        "scorer": sim / "tools/score_pr33824_aic_latency_model.py",
        "predictor": sim / "src/sglang_simulator/time_predictor/aiconfigurator.py",
        "sdk_config": aic_root / "src/aiconfigurator/sdk/config.py",
    }
    trees = {name: ast.parse(path.read_text()) for name, path in paths.items()}
    runtime = next(node for node in trees["sdk_config"].body
                   if isinstance(node, ast.ClassDef) and node.name == "RuntimeConfig")
    defaults = {node.target.id: ast.literal_eval(node.value) for node in runtime.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id in ("prefix", "seq_imbalance_correction_scale")}
    if defaults != {"prefix": 0, "seq_imbalance_correction_scale": 1.0}:
        raise ValueError("SDK config defaults changed; reassess argument capture")
    selected = [node for node in trees["scorer"].body
                if isinstance(node, ast.FunctionDef)
                and node.name in ("schedule_files", "read_rows", "cache_key")]
    predictor = next(node for node in trees["predictor"].body
                     if isinstance(node, ast.ClassDef) and node.name == "AIConfiguratorTimePredictor")
    selected += [node for node in predictor.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("ctx_attn_flops_ratio_with_avg", "predict_infer_latency_dict")]
    if len(selected) != 5:
        raise ValueError("query/scorer methods changed; reassess audit")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"Path": Path, "json": json, "math": math, "np": np,
                 "RuntimeConfig": RuntimeCapture}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[])),
                 str(paths["predictor"]), "exec"), namespace)
    backend = BackendCapture()
    owner = SimpleNamespace(enable_oom_check=False, _session=SimpleNamespace(
        _backend=backend, _model=None, _database=None))
    owner.ctx_attn_flops_ratio_with_avg = MethodType(namespace["ctx_attn_flops_ratio_with_avg"], owner)

    def signature(row: dict) -> tuple:
        requests = [SimpleNamespace(extend_length=extend, past_kv_length=past)
                    for extend, past in zip(row["extend"], row["past"])]
        if len(requests) != row["batch_size"] or not requests:
            raise ValueError("invalid batch length for query capture")
        backend.signature = None
        batch = SimpleNamespace(reqs=requests, batch_size=len(requests),
                                is_decode=lambda: row["forward_mode"] == 2)
        namespace["predict_infer_latency_dict"](owner, batch)
        if backend.signature is None:
            raise ValueError("query method did not invoke the captured backend")
        return backend.signature

    provenance = {name: {"path": str(path.resolve()), "sha256": digest(path)}
                  for name, path in paths.items()}
    provenance["captured_methods"] = [node.name for node in selected]
    provenance["runtime_defaults"] = defaults
    return namespace, signature, provenance


def audit_rows(rows: list[dict], key_function, signature_function) -> tuple[dict, list[dict]]:
    cache, examples, exact_keys, material_keys = {}, [], set(), set()
    counts = {"rows": len(rows), "prefill_rows": 0, "decode_rows": 0, "queries": 0,
              "reused_rows": 0, "exact_alias_rows": 0, "material_alias_rows": 0,
              "prefill_material_alias_rows": 0, "decode_material_alias_rows": 0}
    for row in rows:
        phase = "decode" if row["forward_mode"] == 2 else "prefill"
        counts[f"{phase}_rows"] += 1
        key, signature = key_function(row), signature_function(row)
        if key not in cache:
            cache[key] = signature, row
            counts["queries"] += 1
            continue
        counts["reused_rows"] += 1
        previous, first = cache[key]
        if previous == signature:
            continue
        counts["exact_alias_rows"] += 1
        exact_keys.add(key)
        old, new = dict(previous[1]), dict(signature[1])
        material = (previous[0] != signature[0] or old.keys() != new.keys()
                    or any(not math.isclose(float(old[name]), float(new[name]),
                                           rel_tol=1e-12, abs_tol=1e-12) for name in old))
        if not material:
            continue
        material_keys.add(key)
        counts["material_alias_rows"] += 1
        counts[f"{phase}_material_alias_rows"] += 1
        if len(examples) < 20:
            examples.append({"cache_key": key, "first_query": previous, "later_query": signature,
                             "first_row": first, "later_row": row})
    counts.update(exact_alias_keys=len(exact_keys), material_alias_keys=len(material_keys))
    return counts, examples


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, models: list[str], sim: Path, aic_root: Path, output: Path) -> list[dict]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    namespace, signature, provenance = source_capture(sim, aic_root)
    sources = {item["path"]: item["sha256"] for item in provenance.values()
               if isinstance(item, dict) and "sha256" in item}
    summaries, examples = [], []
    for model in models:
        model_root = root / "models" / model
        marker = model_root / "REAL_ML_COMPLETE"
        if marker.read_text().strip() != "complete":
            raise ValueError(f"model not complete: {model}")
        sources[str(marker.resolve())] = digest(marker)
        for role in ("calibration", "holdout"):
            directory = model_root / role / "real"
            groups = sorted(directory.glob("repeat-*"))
            if not groups:
                raise ValueError(f"no collected groups: {directory}")
            files = namespace["schedule_files"]([str(directory)], "TP0*.schedule_batch.jsonl")
            for group in groups:
                complete = group / "BASELINE_COLLECTION_COMPLETE"
                if complete.read_text().strip() != "complete":
                    raise ValueError(f"incomplete group: {group}")
                sources[str(complete.resolve())] = digest(complete)
            for path in files:
                status = path.parent / "status.txt"
                if status.read_text().strip() != "ok":
                    raise ValueError(f"incomplete case: {path.parent}")
                sources[str(status.resolve())] = digest(status)
                sources[str(path.resolve())] = digest(path)
            rows = namespace["read_rows"]([str(directory)], "TP0*.schedule_batch.jsonl", {1, 2})
            counts, selected = audit_rows(rows, namespace["cache_key"], signature)
            summaries.append({"model": model, "role": role, "files": len(files), **counts})
            examples += [{"model": model, "role": role, **item} for item in selected]
    for path, before in sources.items():
        if digest(Path(path)) != before:
            raise ValueError(f"source changed during audit: {path}")
    output.mkdir(parents=True)
    write_csv(output / "cache_contract_counts.csv", summaries)
    write_csv(output / "source_sha256.csv", [{"path": path, "sha256": value}
                                             for path, value in sorted(sources.items())])
    (output / "alias_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
                "results_root": str(root.resolve()), "models": models, "source_contract": provenance,
                "source_files_verified_unchanged": len(sources), "source_unchanged": True,
                "gpu_fit_or_database_query_invoked": False,
                "limitation": "Argument alias counts are not measured latency-error contributions."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--aic-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models or len(set(models)) != len(models):
        parser.error("--models must contain unique nonempty model names")
    rows = run(args.results_root.resolve(), models, Path(__file__).resolve().parents[1],
               args.aic_root.resolve(), args.out_dir.resolve())
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
