"""Read-only RID-overlap probe using the archived upstream row splitter."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path("test-results/pr33824_expanded_all13_predictors_20260928_1933").resolve()
source = root / "verification/upstream_row_split_identity_20260929_0236/upstream_train_latency_model.py"
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha == "2e64cfe2900b4bbe8c5932a39135b5d1b19c304c45bdfa66b8742ba3d3bb9721"
spec = importlib.util.spec_from_file_location("upstream_train_latency_model_frozen", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
calibration = root / "models/Qwen3-32B/calibration"
data_roots = [str(calibration / "real" / f"repeat-{repeat}" / "cases") for repeat in range(1, 6)]
source_paths = module.iter_schedule_files(data_roots, "TP0*.schedule_batch.jsonl")
assert len(source_paths) == 15
source_before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
rid_lookup = {}
for path in source_paths:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            rids = {info["rid"] for info in item.get("request_infos", [])}
            assert rids and len(rids) == len(item["request_infos"])
            rid_lookup[(str(path), line_number)] = rids


def inspect(modes, seed, fraction):
    rows, cases = module.collect_rows(data_roots, "TP0*.schedule_batch.jsonl", set(modes), 30.0)
    assert len(cases) == 15 and sum(int(row["rows"]) for row in cases) == len(rows)
    train, test = module.build_case_stratified_split(rows, seed=seed, test_fraction=fraction)
    row_id = lambda row: (row["source_schedule"], row["line_number"])
    train_rows = {row_id(row) for row in train}
    test_rows = {row_id(row) for row in test}
    assert len(train_rows) == len(train) and len(test_rows) == len(test)
    assert not (train_rows & test_rows)
    train_rids = set().union(*(rid_lookup[key] for key in train_rows))
    test_rids = set().union(*(rid_lookup[key] for key in test_rows))
    all_rids = train_rids | test_rids
    assert len(all_rids) == 800
    shared_rids = train_rids & test_rids
    return {
        "forward_modes": list(modes),
        "seed": seed,
        "test_fraction": fraction,
        "rows": len(rows),
        "train_rows": len(train),
        "internal_test_rows": len(test),
        "unique_request_ids": len(all_rids),
        "train_unique_request_ids": len(train_rids),
        "internal_test_unique_request_ids": len(test_rids),
        "shared_request_ids": len(shared_rids),
        "shared_pct_of_all_requests": 100 * len(shared_rids) / len(all_rids),
        "internal_test_request_overlap_pct": 100 * len(shared_rids) / len(test_rids),
        "identical_batch_rows_in_both_splits": len(train_rows & test_rows),
    }


results = [
    inspect((1,), 42, 0.5),
    inspect((1, 2), 42, 0.5),
    inspect((1, 2), 202609280, 0.2),
]
saved_summary = list(csv.DictReader((calibration / "training/eval/split_summary.csv").open(encoding="utf-8")))
saved_p50 = next(row for row in saved_summary if row["loss"] == "p50")
assert results[-1]["train_rows"] == int(saved_p50["train_rows"]) == 6344
assert results[-1]["internal_test_rows"] == int(saved_p50["test_rows"]) == 1586
source_after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
assert source_after == source_before
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
print(json.dumps({
    "model": "Qwen3-32B",
    "source_commit": "9cfddcedc9d1e00885c28db75a45513eddbb6271",
    "source_sha256": source_sha,
    "results": results,
    "saved_internal_p50_mape_pct": float(saved_p50["MAPE_pct"]),
    "source_schedule_files_unchanged": len(source_before),
    "scope": "Pinned upstream row split on our Qwen3-32B closed training traces; no fit/predict, no upstream-author accuracy claim, no independent holdout score.",
}, sort_keys=True))
