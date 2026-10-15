"""Query-argument capture only: tests do not import/query the AIC backend."""
import importlib.util
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aic_cache_audit", SIM / "tools/audit_pr33824_aic_cache_contract.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
@pytest.fixture
def captured(tmp_path):
    # Only SDK defaults are needed. No local AIC installation is required.
    config = tmp_path / "src/aiconfigurator/sdk/config.py"
    config.parent.mkdir(parents=True)
    config.write_text("class RuntimeConfig:\n    prefix: int = 0\n"
                      "    seq_imbalance_correction_scale: float = 1.0\n")
    return audit.source_capture(SIM, tmp_path)


def row(extend, past, mode=1):
    return {"forward_mode": mode, "batch_size": len(extend), "extend": extend, "past": past}


def test_actual_methods_expose_synthetic_prefill_cache_alias(captured):
    namespace, signature, provenance = captured
    first, later = row([1, 1, 1, 5], [0] * 4), row([1, 3, 3, 3], [0] * 4)
    assert namespace["cache_key"](first) == namespace["cache_key"](later)
    assert dict(signature(first)[1])["seq_imbalance_correction_scale"] == 1.75
    assert dict(signature(later)[1])["seq_imbalance_correction_scale"] == 1.12
    counts, examples = audit.audit_rows([first, later], namespace["cache_key"], signature)
    assert counts["queries"] == 1
    assert counts["material_alias_rows"] == 1
    assert counts["prefill_material_alias_rows"] == 1
    assert len(examples) == 1
    assert len(provenance["captured_methods"]) == 5


def test_identical_queries_do_not_report_aliases(captured):
    namespace, signature, _ = captured
    first = row([10, 20], [3, 7])
    counts, examples = audit.audit_rows([first] * 3, namespace["cache_key"], signature)
    assert counts["reused_rows"] == 2
    assert counts["exact_alias_rows"] == 0
    assert examples == []


def test_decode_mean_aggregation_is_actual_query_not_request_identity(captured):
    namespace, signature, _ = captured
    rows = [row([1, 1], [2, 4], 2), row([1, 1], [3, 3], 2)]
    counts, examples = audit.audit_rows(rows, namespace["cache_key"], signature)
    assert signature(rows[0]) == signature(rows[1])
    assert counts["decode_rows"] == 2
    assert counts["reused_rows"] == 1
    assert counts["material_alias_rows"] == 0
    assert examples == []


def test_one_token_native_prefill_is_not_inferred_as_decode(captured):
    _, signature, _ = captured
    assert signature(row([1], [0]))[0] == "static_ctx"
    assert signature(row([1], [0], 2))[0] == "static_gen"


def test_capture_uses_sum_of_fractional_means(captured):
    namespace, signature, _ = captured
    value = row([1, 2], [0, 1])
    assert namespace["cache_key"](value)[2] == 1
    assert dict(signature(value)[1])["isl"] == 2


def test_capture_rejects_bad_batch_count(captured):
    _, signature, _ = captured
    with pytest.raises(ValueError, match="batch length"):
        signature({**row([1], [0]), "batch_size": 2})


def test_tiny_rounding_alias_is_counted_separately():
    rows = [{"forward_mode": 1, "x": value} for value in (1.0, 1.0 + 1e-14)]
    counts, _ = audit.audit_rows(rows, lambda _: (1,),
                                lambda value: ("static_ctx", (("scale", value["x"]),)))
    assert counts["exact_alias_rows"] == 1
    assert counts["material_alias_rows"] == 0


def test_refuse_to_overwrite_output_before_loading_source(tmp_path):
    with pytest.raises(FileExistsError, match="overwrite"):
        audit.run(tmp_path, ["unused"], tmp_path, tmp_path, tmp_path)
