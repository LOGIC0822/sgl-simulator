import json
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from generate_pr33824_trace_aic_plan import main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_plan_uses_training_rows_and_aic_static_shape_reduction(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 128,
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "training"
    write_jsonl(
        root / "case" / "TP0.schedule_batch.jsonl",
        [
            {
                "forward_mode": 1,
                "iter_latency": 0.01,
                "request_infos": [
                    {"extend_input_len": 3, "prefix_indices_len": 2},
                    {"extend_input_len": 5, "prefix_indices_len": 4},
                ],
            },
            {
                "forward_mode": 2,
                "iter_latency": 0.01,
                "request_infos": [
                    {"extend_input_len": 1, "prefix_indices_len": 7},
                    {"extend_input_len": 1, "prefix_indices_len": 9},
                ],
            },
        ],
    )
    output = tmp_path / "plan.json"

    main(["--data-root", str(root), "--model-path", str(model), "--output", str(output)])

    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["schedule_rows"] == {
        "forward_mode_1": 1,
        "forward_mode_2": 1,
        "total": 2,
        "skipped_zero_length_aic_shape": {},
    }
    assert plan["attention_targets"] == [
        {"phase": "context", "batch_size": 2, "input_len": 7, "occurrences": 1},
        {"phase": "generation", "batch_size": 2, "input_len": 8, "occurrences": 1},
    ]
    assert plan["gemm_m_values"] == [2, 8]


def test_plan_can_cap_gemm_m_values_by_observed_frequency(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 128,
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "training"
    write_jsonl(
        root / "case" / "TP0.schedule_batch.jsonl",
        [
            {
                "forward_mode": 2,
                "request_infos": [
                    {"extend_input_len": 1, "prefix_indices_len": 4},
                    {"extend_input_len": 1, "prefix_indices_len": 4},
                ],
            },
            {
                "forward_mode": 2,
                "request_infos": [
                    {"extend_input_len": 1, "prefix_indices_len": 5},
                    {"extend_input_len": 1, "prefix_indices_len": 5},
                ],
            },
            {
                "forward_mode": 1,
                "request_infos": [
                    {"extend_input_len": 3, "prefix_indices_len": 2},
                    {"extend_input_len": 3, "prefix_indices_len": 2},
                ],
            },
        ],
    )
    output = tmp_path / "capped-plan.json"

    main(
        [
            "--data-root",
            str(root),
            "--model-path",
            str(model),
            "--output",
            str(output),
            "--max-gemm-m-values",
            "1",
        ]
    )

    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["gemm_m_values"] == [2]
    assert plan["gemm_m_values_observed"] == 2
    assert plan["gemm_m_values_omitted_by_cap"] == 1
