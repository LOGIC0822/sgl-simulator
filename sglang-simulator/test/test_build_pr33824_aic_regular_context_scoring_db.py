import csv
import json
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from build_pr33824_aic_regular_context_scoring_db import CONTEXT_RELATIVE_PATH, main


HEADER = [
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "batch_size",
    "isl",
    "num_heads",
    "num_key_value_heads",
    "head_dim",
    "beam_width",
    "attn_dtype",
    "kv_cache_dtype",
    "step",
    "window_size",
    "latency",
]


def row(batch: int, isl: int, heads: int = 8, kv_heads: int = 2) -> dict[str, str]:
    return {
        "framework": "SGLang",
        "version": "0.0.0",
        "device": "A100",
        "op_name": "context_attention",
        "kernel_source": "triton",
        "batch_size": str(batch),
        "isl": str(isl),
        "num_heads": str(heads),
        "num_key_value_heads": str(kv_heads),
        "head_dim": "128",
        "beam_width": "1",
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "step": "0",
        "window_size": "0",
        "latency": "0.1",
    }


def test_builder_keeps_only_rectangular_context_rows_for_selected_slices(tmp_path: Path) -> None:
    source = tmp_path / "source-systems"
    table = source / CONTEXT_RELATIVE_PATH
    table.parent.mkdir(parents=True)
    rows = [row(batch, isl) for batch in (1, 2) for isl in (16, 32)]
    rows.extend([row(1, 17), row(1, 16, heads=12, kv_heads=4)])
    with table.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    source_config = tmp_path / "source-config.json"
    source_config.write_text(json.dumps({"predictor": {"name": "aiconfigurator"}}), encoding="utf-8")
    plan = tmp_path / "support.json"
    plan.write_text(
        json.dumps(
            {
                "purpose": "AIC Silicon regular interpolation-support attention grid",
                "selection_policy": {"holdout_policy": "holdout rows are not read"},
                "grid": {"batch_sizes": [1, 2], "context_input_lengths": [16, 32]},
                "topology_slices": [
                    {
                        "head_dim": 128,
                        "head_configs": [{"num_heads": 8, "num_key_value_heads": 2}],
                    }
                ],
                "expected_attention_targets": {"context": 4},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "regular-systems"
    output_config = tmp_path / "regular-config.json"
    manifest = tmp_path / "manifest.json"

    main(
        [
            "--source-systems",
            str(source),
            "--support-plan",
            str(plan),
            "--source-sim-config",
            str(source_config),
            "--output-systems",
            str(output),
            "--output-sim-config",
            str(output_config),
            "--manifest",
            str(manifest),
        ]
    )

    with (output / CONTEXT_RELATIVE_PATH).open(newline="", encoding="utf-8") as source_file:
        result = list(csv.DictReader(source_file))
    assert {(item["batch_size"], item["isl"]) for item in result if item["num_heads"] == "8"} == {
        ("1", "16"),
        ("1", "32"),
        ("2", "16"),
        ("2", "32"),
    }
    assert any(item["num_heads"] == "12" for item in result)
    assert json.loads(output_config.read_text(encoding="utf-8"))["predictor"]["database_path"] == str(output.resolve())
    assert json.loads(manifest.read_text(encoding="utf-8"))["row_counts"]["removed_sparse_or_incompatible_rows"] == 1
