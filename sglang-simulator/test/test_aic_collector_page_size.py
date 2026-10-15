"""CPU checks for explicit AIC Triton page-size alignment."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("aic_collector_wrapper", Path(__file__).resolve().parents[1] / "scripts/run_aiconfigurator_collector_compat.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_explicit_page_size_and_legacy_default(monkeypatch):
    monkeypatch.delenv("AICONFIGURATOR_COLLECTOR_TRITON_PAGE_SIZE", raising=False)
    assert wrapper.triton_collection_page_size() == 64
    monkeypatch.setenv("AICONFIGURATOR_COLLECTOR_TRITON_PAGE_SIZE", "1")
    assert wrapper.triton_collection_page_size() == 1
    monkeypatch.setenv("AICONFIGURATOR_COLLECTOR_TRITON_PAGE_SIZE", "0")
    with pytest.raises(ValueError, match="positive"):
        wrapper.triton_collection_page_size()


def test_attention_call_uses_declared_page_size():
    def operation(batch_size, *, page_size=64):
        return batch_size, page_size
    aligned = wrapper.with_fixed_page_size(operation, 1)
    assert aligned(8) == (8, 1)
    assert aligned(8, page_size=1) == (8, 1)
    with pytest.raises(ValueError, match="differs"):
        aligned(8, page_size=64)
