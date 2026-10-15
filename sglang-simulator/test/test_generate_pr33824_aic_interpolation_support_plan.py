import json
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from generate_pr33824_aic_interpolation_support_plan import main, sequence_grid
from collect_pr33824_aic_interpolation_support import checked_plan, iter_targets


def write_model(path: Path, heads: int, kv_heads: int) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": heads * 128,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "head_dim": 128,
            }
        ),
        encoding="utf-8",
    )


def write_training_plan(path: Path, model: Path, heads: int, kv_heads: int) -> None:
    path.write_text(
        json.dumps(
            {
                "purpose": "AIC Silicon calibration targets derived from training traces only",
                "model_path": str(model.resolve()),
                "topology": {
                    "num_attention_heads": heads,
                    "num_key_value_heads": kv_heads,
                    "head_dim": 128,
                },
                "input_data_roots": [str((model.parent / "training").resolve())],
                "schedule_rows": {"total": 12},
                "attention_targets": [
                    {"phase": "context", "batch_size": 3, "input_len": 128},
                    {"phase": "generation", "batch_size": 3, "input_len": 127},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_regular_grid_uses_only_training_plans_and_adds_second_mha_anchor(tmp_path: Path) -> None:
    mha = tmp_path / "mha"
    xqa = tmp_path / "xqa"
    write_model(mha, 32, 32)
    write_model(xqa, 16, 2)
    mha_plan = tmp_path / "mha-training-plan.json"
    xqa_plan = tmp_path / "xqa-training-plan.json"
    write_training_plan(mha_plan, mha, 32, 32)
    write_training_plan(xqa_plan, xqa, 16, 2)
    output = tmp_path / "support-plan.json"

    main(
        [
            "--model-path",
            str(mha),
            "--training-plan",
            str(mha_plan),
            "--model-path",
            str(xqa),
            "--training-plan",
            str(xqa_plan),
            "--max-context-tokens",
            "128",
            "--max-batch-size",
            "4",
            "--output",
            str(output),
        ]
    )

    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["selection_policy"]["source"] == "training trace plans plus declared scheduler limits only"
    assert plan["grid"]["batch_sizes"] == [1, 2, 3, 4]
    assert plan["grid"]["context_input_lengths"][-1] == 128
    assert plan["grid"]["generation_input_lengths"][-1] == 256
    assert plan["expected_attention_targets"]["total"] == 4 * 4 * (8 + 9)
    mha_slice = next(item for item in plan["topology_slices"] if item["aic_num_key_value_heads_lookup"] == 0)
    assert [item["num_heads"] for item in mha_slice["head_configs"]] == [16, 32]
    assert any(item["origin"] == "synthetic_interpolation_neighbor" for item in mha_slice["head_configs"])
    batch_sizes, lengths, slices = checked_plan(plan)
    assert len(list(iter_targets(batch_sizes, lengths, slices))) == plan["expected_attention_targets"]["total"]


def test_denser_sequence_grid_preserves_old_support_without_holdout():
    old = sequence_grid(2048)
    dense = sequence_grid(2048, 64)
    assert set(old).issubset(dense)
    assert len(old) == 21 and len(dense) == 38
    assert all(value in dense for value in range(64, 2049, 64))
    assert sequence_grid(2304, 64)[-1] == 2304
