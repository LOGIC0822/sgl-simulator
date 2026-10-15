"""Compare saved ML errors by exact training-feature support, without fitting.

Run from the simulator directory with PYTHONPATH=src:../../python. The script
reads only completed native schedules and already saved ML score CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from sglang_simulator.time_predictor.ml_features import extract_ml_features


def scored_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
    if not records:
        raise ValueError(f"empty score CSV: {path}")
    by_schedule = defaultdict(dict)
    for record in records:
        schedule = Path(record["schedule_path"]).resolve(strict=True)
        line_number = int(record["line_number"])
        if line_number in by_schedule[schedule]:
            raise ValueError(f"duplicate scored row: {schedule}:{line_number}")
        by_schedule[schedule][line_number] = record
    for schedule, selected in by_schedule.items():
        seen = set()
        with schedule.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line_number not in selected:
                    continue
                record = selected[line_number]
                raw = json.loads(line)
                requests = raw["request_infos"]
                extend = [int(item["extend_input_len"]) for item in requests]
                past = [int(item["prefix_indices_len"]) for item in requests]
                actual = float(record["actual_seconds"])
                predicted = float(record["predicted_seconds"])
                if (int(record["forward_mode"]) != raw["forward_mode"]
                        or int(record["batch_size"]) != len(requests)
                        or int(record["sum_extend"]) != sum(extend)
                        or int(record["sum_past"]) != sum(past)
                        or not math.isclose(actual, float(raw["iter_latency"]), rel_tol=0, abs_tol=1e-12)
                        or not math.isclose(float(record["absolute_percentage_error_pct"]),
                                            100 * abs(predicted - actual) / actual,
                                            rel_tol=0, abs_tol=1e-8)):
                    raise ValueError(f"saved score does not match native row: {schedule}:{line_number}")
                seen.add(line_number)
                yield {
                    "case": record["case"],
                    "phase": "prefill" if raw["forward_mode"] == 1 else "decode",
                    "feature": tuple(extract_ml_features(extend, past)),
                    "ape_pct": float(record["absolute_percentage_error_pct"]),
                }
        if seen != set(selected):
            raise ValueError(f"missing scored source lines: {schedule}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    models = args.models or [item.name for item in sorted((root / "models").iterdir())
                             if (item / "holdout/ml_teacher_forced/per_batch.csv").is_file()]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in models:
        model_root = root / "models" / model
        fit_csv = (root / "verification/all13_frozen_fit_preflight_20260930_0221"
                   / "ml_fit_input_readback" / model / "per_batch.csv")
        heldout_csv = model_root / "holdout/ml_teacher_forced/per_batch.csv"
        if not fit_csv.is_file() or not heldout_csv.is_file():
            raise FileNotFoundError(f"missing completed input or heldout score: {model}")
        fit = list(scored_rows(fit_csv))
        supported = {row["feature"] for row in fit}
        trained_cases = {row["case"] for row in fit}
        test = list(scored_rows(heldout_csv))
        fit_ape_by_feature = defaultdict(list)
        for row in fit:
            fit_ape_by_feature[row["feature"]].append(row["ape_pct"])
        fit_ape_mean = {feature: sum(values) / len(values)
                        for feature, values in fit_ape_by_feature.items()}
        reweighted = [{**row, "ape_pct": fit_ape_mean[row["feature"]]}
                      for row in test if row["feature"] in supported]
        for domain, data in (("fit_input", fit), ("heldout", test),
                             ("heldout_seen_workload", [row for row in test if row["case"] in trained_cases]),
                             ("heldout_unseen_workload", [row for row in test if row["case"] not in trained_cases]),
                             ("fit_reweighted_to_heldout_exact", reweighted),
                             ("fit_reweighted_to_seen_exact", [row for row in reweighted if row["case"] in trained_cases]),
                             ("fit_reweighted_to_unseen_exact", [row for row in reweighted if row["case"] not in trained_cases])):
            if not data:
                continue
            for phase in ("all", "prefill", "decode"):
                selected = [row for row in data if phase == "all" or row["phase"] == phase]
                for support in ("all", "exact_training_feature", "other_feature"):
                    subset = [row for row in selected if support == "all" or
                              ((row["feature"] in supported) == (support == "exact_training_feature"))]
                    if not subset:
                        continue
                    summaries.append({
                        "model": model, "domain": domain, "phase": phase,
                        "feature_support": support, "n": len(subset),
                        "share_of_phase_pct": 100 * len(subset) / len(selected),
                        "mape_pct": sum(row["ape_pct"] for row in subset) / len(subset),
                    })
        print(f"{model}: fit={len(fit)}, heldout={len(test)}, exact={sum(row['feature'] in supported for row in test)}", flush=True)
    with output.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
