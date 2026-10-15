"""Verify all completed second-MoE native/ML endpoint pairs, without rerunning."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import sys
import time
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve()
    root = script.parent.parent.parent
    sim = Path.cwd().resolve()
    model = "Qwen3-30B-A3B-Instruct-2507"
    mr = root / "models" / model
    sha = {}
    expected_bundle = "5b13bad77cae8f3421c988eeeeba45ecbaadbffbbf6bf1d23f8dabd26a7620d8"
    keys = ("PYTHONNOUSERSITE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH", "MPLCONFIGDIR")
    record = {
        "python_argv": [sys.executable, *sys.argv], "cwd": str(sim),
        "launch_command": ["env", *[f"{key}={os.environ.get(key, '')}" for key in keys],
                           "nice", "-n", "15", sys.executable, str(script)],
        "environment": {key: os.environ.get(key) for key in keys},
        "started_unix": time.time(), "exit_code": 1,
        "scope": "All 25 existing endpoint pairs; no fit/predict/GPU/AIC/new replay",
    }

    def preserve(path):
        path = path.resolve()
        sha[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(path):
        return json.loads(preserve(path).read_text())

    def metadata(path):
        return dict(line.split("=", 1) for line in preserve(path).read_text().splitlines() if "=" in line)

    def argument(command, name):
        positions = [i for i, value in enumerate(command) if value == name]
        assert len(positions) == 1, name
        return command[positions[0] + 1]

    def module(name):
        path = preserve(sim / "tools" / f"{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    try:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        from sklearn.ensemble import HistGradientBoostingRegressor
        def forbidden(*args, **kwargs):
            raise AssertionError("Fit and prediction invocation forbidden in endpoint audit")
        HistGradientBoostingRegressor.fit = forbidden
        HistGradientBoostingRegressor.predict = forbidden
        preserve(script)
        pair = module("summarize_pr33824_data_tools_replay")
        report = module("append_pr33824_expanded_report")
        assert preserve(mr / "REAL_ML_COMPLETE").read_text().strip() == "complete"
        bundle = preserve(mr / "calibration/training" / f"{model}_hgbmono_p50.pkl")
        assert sha[str(bundle)] == expected_bundle
        held_record = read_json(root / "verification/moe30_instruct_full_independent_saved_score_integrity_20260929_2354/manifest.json")
        assert held_record["exit_code"] == 0
        held = held_record["result"]
        assert held["test_requests"] == 1280
        assert held["test_timing_rows"] == 12188 and held["frozen_p50_sha256"] == expected_bundle
        preserve(root / "run_metadata.txt")
        preserve(mr / "model_settings.txt")
        counts = {"latency_128x64_r1": 32, "latency_256x128_r1": 32,
                  "service_1024x256_r4": 64, "saturation_1024x256_inf": 64,
                  "saturation_1536x384_inf": 64}
        rows, capacities, versions = [], [], []
        for repeat in range(1, 6):
            real = mr / "holdout/real" / f"repeat-{repeat}"
            replay = mr / "holdout/ml_simulator" / f"repeat-{repeat}"
            for path in (real / "BASELINE_COLLECTION_COMPLETE", replay / "ML_REPLAY_COMPLETE"):
                assert preserve(path).read_text().strip() == "complete"
            versions.append({key: metadata(real / "package_versions.txt")[key] for key in (
                "python", "torch", "torch_cuda", "sglang", "sglang-kernel",
                "sgl_kernel_runtime_version", "sgl_kernel_file", "triton")})
            meta = metadata(replay / "run_metadata.txt")
            assert Path(meta["collection_root"]).resolve() == real
            assert Path(meta["ml_bundle"]).resolve() == bundle
            assert meta["source_revision"] == "59799a368793b9f795baf59b067233a65ad8e38e"
            assert Path(meta["compat_stage"]).name == "minimal-stage-v4"
            assert meta["cuda_graph"] == meta["scheduler_overlap"] == "disabled"
            config = read_json(replay / "simulator-config.json")
            assert config["predictor"]["name"] == "ml"
            assert config["predictor"]["latency_scale"] == 1.0
            assert Path(config["predictor"]["database_path"]).resolve() == bundle
            assert all(config["scheduler"][key] == 1 for key in ("tp_size", "ep_size", "dp_size"))
            for parent in (real, replay):
                assert {p.name for p in (parent / "cases").iterdir() if p.is_dir()} == set(counts)
            for case, count in counts.items():
                rc, pc = real / "cases" / case, replay / "cases" / case
                for path in (rc / "status.txt", pc / "status.txt"):
                    assert preserve(path).read_text().strip() == "ok"
                native = pair.final_jsonl_object(preserve(rc / "benchmark.jsonl"))
                trace = read_json(rc / "autobench_trace.manifest.json")
                metric = read_json(pc / "simulator-output/metrics.json")
                for path in (rc / "TP0.raw_request.jsonl", rc / "TP0.schedule_batch.jsonl",
                             rc / "autobench_trace.jsonl", pc / "simulator-output/iteration.jsonl",
                             pc / "simulator-output/request.jsonl", pc / "replay_validation.log"):
                    preserve(path)
                native_cmd = shlex.split(preserve(real / "cases" / f"{case}.server.command.txt").read_text())
                replay_cmd = shlex.split(preserve(pc / "server.command.txt").read_text())
                real_client = shlex.split(preserve(real / "cases" / f"{case}.benchmark.command.txt").read_text())
                replay_client = shlex.split(preserve(pc / "benchmark.command.txt").read_text())
                assert argument(native_cmd, "--device") == "cuda"
                assert argument(replay_cmd, "--device") == "cpu"
                assert argument(native_cmd, "--tp-size") == "1"
                assert argument(native_cmd, "--mem-fraction-static") == "0.90"
                assert argument(native_cmd, "--attention-backend") == "triton"
                for command in (native_cmd, replay_cmd):
                    assert Path(argument(command, "--model-path")).resolve() == Path("/home/ms-admin/data/models") / model
                    for key, value in (("--context-length", "2048"), ("--max-running-requests", "32"),
                                       ("--chunked-prefill-size", "512")):
                        assert argument(command, key) == value
                    assert "--disable-cuda-graph" in command and "--disable-overlap-schedule" in command
                assert argument(real_client, "--dataset-name") == "random-ids"
                assert int(argument(real_client, "--seed")) == 202609379 + repeat
                assert argument(real_client, "--random-range-ratio") == "0"
                assert ("--tokenize-prompt" in real_client) == (case == "saturation_1536x384_inf")
                assert argument(replay_client, "--dataset-name") == "autobench"
                assert Path(argument(replay_client, "--dataset-path")).resolve() == rc / "autobench_trace.jsonl"
                assert "--use-trace-timestamps" in replay_client
                assert int(argument(real_client, "--num-prompts")) == int(argument(replay_client, "--num-prompts")) == count
                caps = re.findall(r"max_total_num_tokens=(\d+)", preserve(real / "cases" / f"{case}.server.log").read_text())
                assert caps and int(caps[-1]) == int(argument(replay_cmd, "--max-total-tokens"))
                capacities.append({"repeat": repeat, "case": case, "native_tokens": int(caps[-1]),
                                   "replay_tokens": int(argument(replay_cmd, "--max-total-tokens"))})
                assert native["completed"] == trace["request_count"] == metric["completed"] == metric["num_requests"] == count
                assert metric["total_input"] == trace["input_tokens"]
                assert metric["total_output"] == trace["output_tokens"] == native["total_output_tokens"]
                info = native["server_info"]
                assert info["device"] == "cuda" and Path(info["model_path"]).name == model
                assert info["context_length"] == 2048 and info["disable_cuda_graph"] and info["disable_overlap_schedule"]
            rows += pair.pair_roots(real, replay, f"repeat-{repeat}")
        assert all(version == versions[0] for version in versions)
        assert versions[0]["torch"] == "2.9.1+cu126" and versions[0]["sgl_kernel_runtime_version"] == "0.4.6.post1"
        saved = report.read_csv(preserve(mr / "holdout/ml_e2e_comparison/comparison.csv"))
        key = lambda row: (row["pair"], row["case"], row["metric"])
        expected = {(f"repeat-{r}", c, m) for r in range(1, 6) for c in counts for m in pair.METRICS}
        lookup = {key(row): row for row in saved}
        assert len(saved) == len(rows) == len(lookup) == len(expected) == 100
        assert set(lookup) == {key(row) for row in rows} == expected
        for row in rows:
            previous = lookup[key(row)]
            for field, value in row.items():
                if field in ("pair", "case", "metric", "throughput_token_source"):
                    assert previous[field] == value
                elif field in ("real_root", "replay_root"):
                    assert Path(previous[field]).resolve() == Path(value).resolve()
                else:
                    assert math.isclose(float(previous[field]), float(value), rel_tol=1e-12, abs_tol=1e-12)
        summary = pair.summarize(rows)
        stored = report.read_csv(preserve(mr / "holdout/ml_e2e_comparison/summary.csv"))
        lookup = {(row["group"], row["value"]): row for row in stored}
        assert len(stored) == len(lookup) == len(summary) == 24
        for row in summary:
            previous = lookup[row["group"], row["value"]]
            assert row["n"] == (25 if row["group"] == "metric" else 5)
            for field in row.keys() - {"group", "value"}:
                assert math.isclose(float(previous[field]), float(row[field]), rel_tol=1e-12, abs_tol=1e-12)
        output = script.parent / "results"
        output.mkdir(exist_ok=False)
        report.write_csv(output / "verified_e2e_comparison.csv", rows)
        report.write_csv(output / "verified_e2e_summary.csv", summary)
        report.write_csv(output / "verified_kv_capacity.csv", capacities)
        lines = report.replay_overhead_details(root, [model], output, methods=("ML",))
        (output / "replay_overhead_sections.md").write_text("\n".join(lines), encoding="utf-8")
        components = report.read_csv(output / "replay_wall_time_components.csv")
        assert len(components) == 25
        for path, value in sha.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == value, path
        report.write_csv(output / "source_sha256.csv", [{"path": p, "sha256": v} for p, v in sorted(sha.items())])
        result = {"model": model, "paired_cases": 25, "paired_metric_rows": 100,
                  "independent_requests": 1280, "native_step_rows": 12188,
                  "bundle_sha256": expected_bundle, "source_files_unchanged": len(sha),
                  "native_versions": versions[0], "matched_capacity_cases": len(capacities),
                  "aggregate_metrics": [row for row in summary if row["group"] == "metric"],
                  "simulator_iterations": sum(int(row["iterations"]) for row in components),
                  "fit_prediction_gpu_aic_or_new_replay_invoked": False,
                  "scope": "This model's full endpoints; not all13/fresh AIC completion or native CPU profiling"}
        with (output / "manifest.json").open("x", encoding="utf-8") as destination:
            json.dump(result, destination, ensure_ascii=False, indent=2)
        record["exit_code"] = 0
        print(json.dumps(result, ensure_ascii=False))
    finally:
        record["finished_unix"] = time.time()
        with (script.parent / "execution.command.json").open("x", encoding="utf-8") as destination:
            json.dump(record, destination, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
