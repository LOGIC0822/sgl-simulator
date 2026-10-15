#!/usr/bin/env python3
"""Compare frozen AIC estimates with separately instrumented native samples.

Native GPU interval union and kernel-family duration sums remain separate.
Three predeclared KV interpretations are queried, without test-based tuning.
Physical-before inputs follow AIC's generation s=isl+1 and context s+prefix
contract; no operator-family one-to-one correspondence is asserted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics


SIM = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_inputs(metadata: dict, measured: dict) -> dict[str, list[int]]:
    if measured != metadata["measured_batch"]:
        raise ValueError("profile metadata is not the exact exported native timing row")
    if metadata["forward_mode"] != measured["forward_mode"]:
        raise ValueError("profile phase differs from measured native phase")
    if metadata["model_dtype"] != "torch.bfloat16" or metadata["kv_cache_dtype"] != "torch.bfloat16":
        raise ValueError("dense profiling comparison requires actual BF16 model and KV")
    physical = metadata["physical_prepared_seq_lens_cpu"]
    saved, observed = metadata["requests"], measured["request_infos"]
    if len(physical) != len(saved) or len(saved) != len(observed) or not saved:
        raise ValueError("profile sequence/request counts differ")
    keys = ("extend_input_len", "prefix_indices_len", "output_ids_len")
    for before, after in zip(saved, observed):
        if str(before["rid"]) != str(after["rid"]) or any(before[key] != after[key] for key in keys):
            raise ValueError("profile versus collector request identity/length differs")
    for item in saved:
        if any(isinstance(item[key], bool) or not isinstance(item[key], int) or item[key] < 0 for key in keys):
            raise ValueError("invalid native request token lengths")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in physical):
        raise ValueError("invalid physical sequence lengths")
    extend = [item["extend_input_len"] for item in saved]
    if any(value < 1 for value in extend) or any(total < new for total, new in zip(physical, extend)):
        raise ValueError("invalid physical prepared versus new-token lengths")
    return {"extend": extend,
            "prefix": [item["prefix_indices_len"] for item in saved],
            "runtime": [item["prefix_indices_len"]+item["output_ids_len"] for item in saved],
            "physical_before": [total-new for total, new in zip(physical, extend)]}


def predict_variants(predictor, scorer, mode: int, inputs: dict) -> dict:
    from sglang_simulator.time_predictor.base import ScheduleRequest

    if mode not in (1, 2) or (mode == 2 and any(value != 1 for value in inputs["extend"])):
        raise ValueError("invalid native phase versus extension lengths")
    result = {}
    for interpretation in ("prefix", "runtime", "physical_before"):
        batch = scorer.ObservedScheduleBatch(mode, [ScheduleRequest(extend_length=new, past_kv_length=old)
            for new, old in zip(inputs["extend"], inputs[interpretation])])
        operators = {key: float(value) for key, value in predictor.predict_infer_latency_dict(batch).items()}
        if not operators or any(not math.isfinite(value) or value < 0 for value in operators.values()):
            raise ValueError("invalid AIC operator prediction")
        predicted = float(predictor.predict_infer_time(batch))*1000
        if not math.isclose(sum(operators.values()), predicted, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError("AIC total contains a scale/floor not represented by operator sums")
        result[interpretation] = {"total_ms": predicted, "operators_ms": operators}
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no native profile comparisons")
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("AIC scoring is CPU-only; set CUDA_VISIBLE_DEVICES=''")
    root, profile = args.results_root.resolve(), args.profile_dir.resolve()
    load_tool("run_pr33824_matched_native_profiles").require_complete(root)
    if args.out_dir.exists():
        parser.error("refusing to overwrite profile scoring output")
    manifest = json.loads((profile / "collection_manifest.json").read_text())
    if manifest["formal_root"] != str(root) or len(manifest["configs"]) != 15:
        raise ValueError("require original matrix identity and full 3 x 5 profile coverage")
    kernel = manifest["kernel_sha256"]
    runtime_sources = manifest["runtime_source_sha256"]
    if not kernel or not runtime_sources or any(digest(Path(path)) != value for path, value in {**kernel, **runtime_sources}.items()):
        raise ValueError("profile/scoring kernel differs")
    protocol = dict(line.split("=", 1) for line in (root / "run_metadata.txt").read_text().splitlines() if "=" in line)
    kernel_spec = importlib.util.find_spec("sgl_kernel")
    stage = Path(protocol["kernel_compat_stage"]).resolve()
    if not kernel_spec or not kernel_spec.origin or not Path(kernel_spec.origin).resolve().is_relative_to(stage):
        raise ValueError("AIC scoring import does not use the profiled kernel stage")
    audit = load_tool("audit_pr33824_aic_kv_inputs")
    config = root / "aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json"
    database = audit.database_snapshot(config)
    scorer, analyzer = load_tool("score_pr33824_aic_latency_model"), load_tool("analyze_pr33824_native_profile_sample")
    with (root / "model_manifest.tsv").open() as source:
        paths = {row["model"]: Path(row["model_path"]) for row in csv.DictReader(source, delimiter="\t")}
    args.out_dir.mkdir(parents=True)
    rows, sources, predictors = [], dict(database), {}
    models = load_tool("run_pr33824_matched_native_profiles").MODELS
    cases = protocol["heldout_case"].split(",")
    configurations = manifest["configs"]
    if {(entry["model"], entry["case"]) for entry in configurations} != {(model, case) for model in models for case in cases}:
        raise ValueError("profile configs contain duplicate, missing or unexpected cases")
    for entry in configurations:
        model, case = entry["model"], entry["case"]
        folder = profile / model / case
        if (folder / "PROFILE_COLLECTION_COMPLETE").read_text().strip() != "complete":
            raise ValueError("incomplete native profile case")
        runner = load_tool("run_pr33824_matched_native_profiles")
        runner.verify_collection(folder, int(runner.one_value(entry["client"], "--num-prompts")))
        for source_path in (folder / "benchmark.jsonl", folder / "collection/TP0.manifest.json", folder / "collection/TP0.raw_request.jsonl"):
            sources[str(source_path)] = digest(source_path)
        checkpoint_config = paths[model] / "config.json"
        sources[str(checkpoint_config)] = digest(checkpoint_config)
        if model not in predictors:
            predictors[model] = scorer.load_predictor(config, paths[model])
        predictor = predictors[model]
        schedule = folder / "collection/TP0.schedule_batch.jsonl"
        sources[str(schedule)] = digest(schedule)
        measured = [json.loads(line) for line in schedule.read_text().splitlines()]
        records = sorted((folder / "profiles").glob("*.metadata.json"))
        steps = manifest["steps_per_phase"]
        if len(records) != 2*steps or any(sum(json.loads(path.read_text())["forward_mode"] == mode for path in records) != steps for mode in (1, 2)):
            raise ValueError("insufficient phase sample coverage")
        seen_lines = set()
        for path in records:
            sources[str(path)] = digest(path)
            metadata = json.loads(path.read_text())
            if Path(metadata["python_executable"]).resolve() != Path(protocol["python"]).resolve():
                raise ValueError("profile worker executable differs from formal environment")
            for name, expected in (("sglang", SIM.parents[1] / "python/sglang"), ("sgl_kernel", stage / "sgl_kernel")):
                origin = metadata["module_origins"].get(name)
                if not origin or not Path(origin).resolve().is_relative_to(expected):
                    raise ValueError(f"native worker {name} imported from a different source")
            index = metadata["schedule_line"]-1
            if index < 0 or index >= len(measured) or index in seen_lines:
                raise ValueError("profile sample outside exported native row coverage")
            seen_lines.add(index)
            inputs = sample_inputs(metadata, measured[index])
            trace_path = Path(metadata["trace_path"])
            if not trace_path.resolve().is_relative_to(folder.resolve()) or digest(trace_path) != metadata["trace_sha256"]:
                raise ValueError("native trace differs or is outside its diagnostic directory")
            sources[str(trace_path)] = digest(trace_path)
            with (gzip.open(trace_path, "rt") if trace_path.suffix == ".gz" else trace_path.open()) as source:
                trace = json.load(source)
            analysis = analyzer.analyze(trace, metadata["native_marker"], float(measured[index]["iter_latency"]))
            predictions = predict_variants(predictor, scorer, metadata["forward_mode"], inputs)
            analysis.update(input_interpretations=inputs, aic_predictions=predictions, metadata_path=str(path))
            (args.out_dir / f"{model}.{case}.{path.stem}.analysis.json").write_text(json.dumps(analysis, indent=2))
            gpu = analysis["native_gpu_interval_union_us"]/1000
            wall = analysis["instrumented_iteration_us"]/1000
            row = {"model": model, "case": case, "phase": {1: "prefill", 2: "decode"}[metadata["forward_mode"]],
                   "metadata_path": str(path), "batch_size": len(inputs["extend"]), "sum_extend": sum(inputs["extend"]),
                   "instrumented_wall_ms": wall, "native_gpu_union_ms": gpu,
                   "native_gpu_duration_sum_ms": analysis["native_gpu_duration_sum_us"]/1000,
                   "wall_minus_gpu_union_ms": wall-gpu,
                   "gpu_attribution_duration_pct": analysis["gpu_attribution_duration_pct"],
                   "unattributed_gpu_event_count": analysis["unattributed_gpu_event_count"],
                   "runtime_vs_physical_before_mismatch_requests": sum(a != b for a, b in zip(inputs["runtime"], inputs["physical_before"]))}
            for interpretation, value in predictions.items():
                predicted = value["total_ms"]
                row.update({f"{interpretation}_aic_ms": predicted,
                            f"{interpretation}_ape_vs_wall_pct": abs(predicted-wall)/wall*100,
                            f"{interpretation}_ape_vs_gpu_union_pct": abs(predicted-gpu)/gpu*100})
            operators = predictions["physical_before"]["operators_ms"]
            row["physical_aic_gemm_sum_ms"] = sum(value for name, value in operators.items() if name.endswith("_gemm"))
            row["physical_aic_attention_sum_ms"] = sum(value for name, value in operators.items() if name in ("context_attention", "generation_attention"))
            row["physical_aic_other_sum_ms"] = sum(operators.values())-row["physical_aic_gemm_sum_ms"]-row["physical_aic_attention_sum_ms"]
            for family in ("gemm_gemv", "triton_prefill_attention", "triton_decode_attention", "other_kernel", "gpu_memcpy", "gpu_memset"):
                row[f"{family}_duration_sum_ms"] = analysis["families"].get(family, {}).get("duration_sum_us", 0)/1000
            rows.append(row)
    if audit.database_snapshot(config) != database or any(digest(Path(path)) != value for path, value in {**sources, **kernel, **runtime_sources}.items()):
        raise ValueError("frozen DB, checkpoint, native source or profile records changed during scoring")
    write_csv(args.out_dir / "per_sample.csv", rows)
    summaries = []
    for model in models:
        for phase in ("prefill", "decode"):
            selected = [row for row in rows if row["model"] == model and row["phase"] == phase]
            summary = {"model": model, "phase": phase, "n": len(selected)}
            for key in rows[0]:
                if key not in ("model", "case", "phase", "metadata_path"):
                    summary[key] = statistics.mean(row[key] for row in selected)
            summaries.append(summary)
    write_csv(args.out_dir / "summary.csv", summaries)
    (args.out_dir / "source_manifest.json").write_text(json.dumps({"source_sha256": {**sources, **kernel, **runtime_sources},
        "scope": "Instrumented diagnostic, first three samples per phase after two skipped steps in each of five workloads. GPU family sums are heuristic, not one-to-one AIC operators; wall-minus-union is not pure CPU time. No formal baselines, predictor weights or tables were changed."}, indent=2))


if __name__ == "__main__":
    main()
