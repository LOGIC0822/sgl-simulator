import json

import pytest
from aiconfigurator.sdk import common as aic_common
from aiconfigurator.sdk.models import helpers as aic_model_helpers

from sglang_simulator.time_predictor.aiconfigurator import (
    _aic_awq_bf16_compat,
    _register_aic_architecture_aliases,
)


def test_registers_mistral_as_llama_when_aic_lacks_the_alias(monkeypatch):
    monkeypatch.delitem(
        aic_common.ARCHITECTURE_TO_MODEL_FAMILY,
        "MistralForCausalLM",
        raising=False,
    )

    _register_aic_architecture_aliases()

    assert (
        aic_common.ARCHITECTURE_TO_MODEL_FAMILY["MistralForCausalLM"] == "LLAMA"
    )


def test_awq_compat_suppresses_only_aic_unsupported_quant_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "awq"}}),
        encoding="utf-8",
    )
    raw_config = {"quant_algo": "awq"}

    with pytest.raises(ValueError, match="Unsupported quant algorithm"):
        aic_model_helpers._infer_quant_modes_from_raw_config(raw_config)

    with _aic_awq_bf16_compat(str(tmp_path)):
        assert aic_model_helpers._infer_quant_modes_from_raw_config(raw_config) == {}

    with pytest.raises(ValueError, match="Unsupported quant algorithm"):
        aic_model_helpers._infer_quant_modes_from_raw_config(raw_config)
