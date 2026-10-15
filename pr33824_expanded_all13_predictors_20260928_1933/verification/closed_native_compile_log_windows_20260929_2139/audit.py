"""Classify recorded slow Triton warnings relative to formal collection logs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve()
    root = script.parent.parent.parent
    sim = Path.cwd().resolve()
    files = {}
    record = {"python_argv": [sys.executable, *sys.argv], "cwd": str(sim),
              "started_unix": time.time(), "exit_code": 1,
              "environment": {key: os.environ.get(key) for key in (
                  "CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "OMP_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "scope": "Closed-case log timestamps only; no GPU/profile/predictions or latency filtering"}

    def preserved_text(path: Path) -> str:
        data = path.read_bytes()
        files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")

    try:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        preserved_text(script)
        for path in (sim / "src/sglang_simulator/baseline/collector.py",
                     sim / "scripts/run_pr33824_data_tools_baseline.sh",
                     sim / "../../python/sglang/srt/utils/triton_load_watch.py",
                     sim / "../../python/sglang/srt/environ.py"):
            preserved_text(path.resolve())
        pattern = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*$")
        compile_pattern = re.compile(r"Triton kernel '([^']+)' took ([0-9.]+) s to compile after serving started")
        models = sorted(path.parent.name for path in (root / "models").glob("*/REAL_ML_COMPLETE"))
        assert len(models) == 11
        cases, warnings = [], []
        for model in models:
            mr = root / "models" / model
            assert preserved_text(mr / "REAL_ML_COMPLETE").strip() == "complete"
            for role, expected_cases in (("calibration", 3), ("holdout", 5)):
                repeats = sorted((mr / role / "real").glob("repeat-*"))
                assert len(repeats) == 5
                for repeat in repeats:
                    assert preserved_text(repeat / "BASELINE_COLLECTION_COMPLETE").strip() == "complete"
                    paths = sorted((repeat / "cases").glob("*/status.txt"))
                    assert len(paths) == expected_cases
                    for status in paths:
                        assert preserved_text(status).strip() == "ok"
                        case = status.parent.name
                        log = repeat / "cases" / f"{case}.server.log"
                        lines = preserved_text(log).splitlines()
                        starts, stops, events = [], [], []
                        for line_number, line in enumerate(lines, 1):
                            match = pattern.match(line)
                            if match is None:
                                continue
                            stamp = datetime.fromisoformat(match[1])
                            if '"POST /start_profile HTTP/1.1" 200 OK' in line:
                                starts.append((line_number, stamp))
                            if '"POST /stop_profile HTTP/1.1" 200 OK' in line:
                                stops.append((line_number, stamp))
                            compile_match = compile_pattern.search(line)
                            if compile_match:
                                events.append((line_number, stamp, compile_match[1], float(compile_match[2])))
                        assert len(starts) == len(stops) == 1, str(log)
                        assert starts[0][0] < stops[0][0] and starts[0][1] <= stops[0][1]
                        counts = Counter()
                        for line_number, stamp, kernel, seconds in events:
                            if line_number < starts[0][0] and stamp < starts[0][1]:
                                window = "before_start_profile_log"
                            elif line_number > stops[0][0] and stamp > stops[0][1]:
                                window = "after_stop_profile_log"
                            elif starts[0][0] < line_number < stops[0][0] and starts[0][1] < stamp < stops[0][1]:
                                window = "between_profile_logs"
                            else:
                                window = "boundary_second_ambiguous"
                            counts[window] += 1
                            warnings.append({"model": model, "role": role, "repeat": repeat.name,
                                             "case": case, "log": str(log), "line": line_number,
                                             "timestamp": stamp.isoformat(), "kernel": kernel,
                                             "reported_compile_seconds": seconds, "window": window,
                                             "seconds_relative_to_start_log": (stamp - starts[0][1]).total_seconds()})
                        cases.append({"model": model, "role": role, "repeat": repeat.name,
                                      "case": case, "start_profile_log": starts[0][1].isoformat(),
                                      "stop_profile_log": stops[0][1].isoformat(), "warnings": len(events),
                                      **{key: counts[key] for key in (
                                          "before_start_profile_log", "between_profile_logs",
                                          "after_stop_profile_log", "boundary_second_ambiguous")}})
        assert len(cases) == 440
        for path, expected in files.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
        output = script.parent / "results"
        output.mkdir(exist_ok=False)
        for filename, values in (("case_log_windows.csv", cases), ("recorded_compile_warnings.csv", warnings),
                                 ("source_sha256.csv", [{"path": p, "sha256": v} for p, v in sorted(files.items())])):
            with (output / filename).open("x", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
        result = {"models": models, "closed_native_cases": len(cases),
                  "case_roles": dict(Counter(row["role"] for row in cases)),
                  "recorded_warnings": len(warnings), "warning_windows": dict(Counter(row["window"] for row in warnings)),
                  "per_model_warning_windows": {model: dict(Counter(row["window"] for row in warnings if row["model"] == model)) for model in models},
                  "sources_verified_unchanged": len(files),
                  "interpretation": "Recorded slow uncached-compilation warnings only. Serving-start flag also covers request warmup. HTTP profile logs are not kernel timestamps; same-second boundaries are ambiguous. Absence of warnings does not exclude sub-threshold work, cache-hit loading, other JIT or runtime overhead. No individual residual is causally attributed.",
                  "gpu_profile_predict_fit_or_latency_filter_invoked": False}
        with (output / "manifest.json").open("x", encoding="utf-8") as destination:
            json.dump(result, destination, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False))
        record["exit_code"] = 0
    finally:
        record["finished_unix"] = time.time()
        with (script.parent / "execution.command.json").open("x", encoding="utf-8") as destination:
            json.dump(record, destination, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
