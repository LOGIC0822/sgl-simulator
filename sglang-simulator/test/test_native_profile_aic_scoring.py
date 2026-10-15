"""Physical sequence/query contracts on synthetic samples; no GPU/AIC DB."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("profile_scoring", Path(__file__).resolve().parents[1] / "tools/score_pr33824_native_profile_aic.py")
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def fixture():
    requests = [{"rid": "r1", "extend_input_len": 1, "prefix_indices_len": 10, "output_ids_len": 4},
                {"rid": "r2", "extend_input_len": 1, "prefix_indices_len": 11, "output_ids_len": 6}]
    measured = {"forward_mode": 2, "request_infos": copy.deepcopy(requests), "iter_latency": .01}
    return {"requests": requests, "physical_prepared_seq_lens_cpu": [15, 18], "measured_batch": measured,
            "forward_mode": 2, "model_dtype": "torch.bfloat16", "kv_cache_dtype": "torch.bfloat16"}, measured


def test_physical_query_lengths_remove_current_extension_not_outputs():
    metadata, measured = fixture()
    inputs = score.sample_inputs(metadata, measured)
    assert inputs == {"extend": [1, 1], "prefix": [10, 11], "runtime": [14, 17], "physical_before": [14, 17]}
    metadata["physical_prepared_seq_lens_cpu"] = [16, 19]
    inputs = score.sample_inputs(metadata, measured)
    assert inputs["runtime"] != inputs["physical_before"]  # Preserve, do not silently force equality.


@pytest.mark.parametrize("kind", ["measured", "rid", "dtype", "shape", "bool", "too_short", "phase"])
def test_sample_refuses_identity_or_length_mismatch(kind):
    metadata, measured = fixture()
    if kind == "measured":
        measured = {**measured, "iter_latency": .02}
    elif kind == "rid":
        metadata["requests"][0]["rid"] = "different"
    elif kind == "dtype":
        metadata["kv_cache_dtype"] = "torch.float16"
    elif kind == "shape":
        metadata["physical_prepared_seq_lens_cpu"] = [15]
    elif kind == "bool":
        metadata["physical_prepared_seq_lens_cpu"][0] = True
    elif kind == "phase":
        metadata["forward_mode"] = 1
    else:
        metadata["physical_prepared_seq_lens_cpu"][0] = 0
    with pytest.raises(ValueError):
        score.sample_inputs(metadata, measured)


def test_frozen_aic_variants_share_actual_phase_and_do_not_tune():
    metadata, measured = fixture()
    calls = []
    class Batch:
        def __init__(self, mode, reqs):
            self.mode, self.reqs = mode, reqs
    class Predictor:
        def predict_infer_latency_dict(self, batch):
            calls.append((batch.mode, [req.past_kv_length for req in batch.reqs]))
            return {"generation_qkv_gemm": 3, "generation_attention": 1}
        def predict_infer_time(self, batch):
            return .004
    result = score.predict_variants(Predictor(), SimpleNamespace(ObservedScheduleBatch=Batch), 2,
                                    score.sample_inputs(metadata, measured))
    assert list(result) == ["prefix", "runtime", "physical_before"]
    assert calls == [(2, [10, 11]), (2, [14, 17]), (2, [14, 17])]
    assert all(value["total_ms"] == 4 for value in result.values())


def test_scaled_aic_sum_is_not_mislabeled_operator_total():
    metadata, measured = fixture()
    class Batch:
        def __init__(self, mode, reqs):
            pass
    predictor = SimpleNamespace(predict_infer_latency_dict=lambda batch: {"gemm": 3}, predict_infer_time=lambda batch: .006)
    with pytest.raises(ValueError, match="scale/floor"):
        score.predict_variants(predictor, SimpleNamespace(ObservedScheduleBatch=Batch), 2, score.sample_inputs(metadata, measured))


@pytest.mark.parametrize("generated", [1, 4, 64])
def test_ordinary_decode_counts_the_current_token_once(generated):
    """Synthetic source-contract case, not a measured GPU/AIC latency result."""
    metadata, _ = fixture()
    for request, input_length in zip(metadata["requests"], (10, 11)):
        request.update(input_length=input_length, output_ids_len=generated)
    measured = {"forward_mode": 2, "request_infos": copy.deepcopy(metadata["requests"]),
                "iter_latency": .01}
    metadata.update(measured_batch=measured,
                    physical_prepared_seq_lens_cpu=[10+generated, 11+generated])
    inputs = score.sample_inputs(metadata, measured)
    assert inputs["runtime"] == metadata["physical_prepared_seq_lens_cpu"]
    assert inputs["physical_before"] == [9+generated, 10+generated]

    class Batch:
        def __init__(self, mode, reqs):
            self.mode, self.reqs = mode, reqs

    class ContractPredictor:
        # This artificial value exposes s=isl+1; it is not benchmark data.
        def predict_infer_latency_dict(self, batch):
            assert batch.mode == 2
            query_s = sum(req.past_kv_length for req in batch.reqs)//len(batch.reqs)+1
            return {"synthetic_sequence_contract": float(query_s)}

        def predict_infer_time(self, batch):
            return sum(self.predict_infer_latency_dict(batch).values())/1000

    predictions = score.predict_variants(ContractPredictor(),
        SimpleNamespace(ObservedScheduleBatch=Batch), 2, inputs)
    expected_s = sum(metadata["physical_prepared_seq_lens_cpu"])//2
    assert predictions["physical_before"]["total_ms"] == expected_s
    assert predictions["runtime"]["total_ms"] == expected_s+1
    assert predictions["prefix"]["total_ms"] == expected_s-(generated-1)


def test_prefill_physical_history_preserves_native_phase_and_cached_prefix():
    """A one-token prefill remains prefill; only prepared-minus-extend is used."""
    requests = [
        {"rid": "prefill-cached", "extend_input_len": 1,
         "prefix_indices_len": 10, "output_ids_len": 0},
        {"rid": "prefill-uncached", "extend_input_len": 15,
         "prefix_indices_len": 0, "output_ids_len": 0},
    ]
    measured = {"forward_mode": 1, "request_infos": copy.deepcopy(requests),
                "iter_latency": .01}
    metadata = {"forward_mode": 1, "requests": requests,
                "physical_prepared_seq_lens_cpu": [11, 15],
                "model_dtype": "torch.bfloat16", "kv_cache_dtype": "torch.bfloat16",
                "measured_batch": measured}
    inputs = score.sample_inputs(metadata, measured)
    assert inputs["extend"] == [1, 15]
    assert inputs["prefix"] == inputs["runtime"] == inputs["physical_before"] == [10, 0]
    calls = []

    class Batch:
        def __init__(self, mode, reqs):
            self.mode, self.reqs = mode, reqs

    class ContractPredictor:
        def predict_infer_latency_dict(self, batch):
            calls.append(batch.mode)
            return {"synthetic_prefill_contract": 1.0}

        def predict_infer_time(self, batch):
            return .001

    score.predict_variants(ContractPredictor(), SimpleNamespace(ObservedScheduleBatch=Batch), 1, inputs)
    assert calls == [1, 1, 1]
