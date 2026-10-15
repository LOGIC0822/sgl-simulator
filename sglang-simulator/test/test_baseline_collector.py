"""CPU-only contracts for the real-server baseline collector."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sglang_simulator.baseline import collector


@pytest.fixture(autouse=True)
def reset_collection_state():
    collector.COLLECTION_STATE.active = False
    collector.COLLECTION_STATE.requests.clear()
    collector.COLLECTION_STATE.schedule_batches.clear()
    yield
    collector.COLLECTION_STATE.active = False
    collector.COLLECTION_STATE.requests.clear()
    collector.COLLECTION_STATE.schedule_batches.clear()


def test_collection_state_exports_ranked_contract(tmp_path, monkeypatch):
    monkeypatch.setenv(collector.COLLECTION_OUTPUT_DIR_ENV, str(tmp_path))
    state = collector.COLLECTION_STATE
    state.start()
    state.requests["r0"] = collector.RequestRecord(
        rid="r0",
        created_time=12.5,
        input_length=8,
        output_length=2,
        before_adder_device_hit_len=4,
        final_host_hit_len=2,
        input_ids=[1, 2],
        output_ids=[3, 4],
    )
    state.schedule_batches.append(
        {
            "forward_mode": 1,
            "request_infos": [
                {
                    "rid": "r0",
                    "extend_input_len": 4,
                    "prefix_indices_len": 4,
                    "output_ids_len": 0,
                }
            ],
            "iter_latency": 0.01,
        }
    )
    scheduler = SimpleNamespace(
        page_size=1,
        ps=SimpleNamespace(
            tp_rank=0,
            dp_size=2,
            dp_rank=1,
            pp_size=1,
            moe_ep_size=1,
        ),
    )

    paths = state.stop_and_export(scheduler)

    assert [path.name for path in paths] == [
        "TP0-DP1.request.jsonl",
        "TP0-DP1.raw_request.jsonl",
        "TP0-DP1.schedule_batch.jsonl",
        "TP0-DP1.manifest.json",
    ]
    assert json.loads(paths[0].read_text()) == {
        "rid": "r0",
        "timestamp": 12500.0,
        "input_length": 8,
        "output_length": 2,
        "device_cache_hit_length": 4,
        "host_cache_hit_length": 2,
        "disk_cache_hit_length": 0,
    }
    manifest = json.loads(paths[-1].read_text())
    assert manifest["schema_version"] == 1
    assert manifest["request_count"] == 1
    assert manifest["schedule_batch_count"] == 1


def test_export_refuses_to_overwrite_existing_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv(collector.COLLECTION_OUTPUT_DIR_ENV, str(tmp_path))
    (tmp_path / "TP0.request.jsonl").write_text("old\n")
    collector.COLLECTION_STATE.start()

    with pytest.raises(FileExistsError):
        collector.COLLECTION_STATE.stop_and_export(
            SimpleNamespace(page_size=1, tp_rank=0)
        )


def test_scheduler_hook_collects_only_between_profile_requests(tmp_path, monkeypatch):
    monkeypatch.setenv(collector.COLLECTION_OUTPUT_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(collector, "_synchronize_device", lambda: None)
    monkeypatch.setattr(
        collector,
        "_profile_output",
        lambda message: SimpleNamespace(success=True, message=message),
    )

    class ForwardMode:
        def is_decode(self):
            return False

        def __int__(self):
            return 1

    request = SimpleNamespace(
        rid="request-1",
        origin_input_ids=[1, 2, 3],
        output_ids=[4],
        prefix_indices=[0],
        extend_range=SimpleNamespace(length=2),
        sampling_params=SimpleNamespace(
            max_new_tokens=1,
            custom_params={"server_created_time": 10.0},
        ),
        finished=lambda: True,
    )
    batch = SimpleNamespace(
        reqs=[request],
        forward_mode=ForwardMode(),
        is_empty=lambda: False,
    )

    class Scheduler:
        page_size = 1
        ps = SimpleNamespace(
            tp_rank=0,
            dp_size=1,
            pp_size=1,
            moe_ep_size=1,
        )

        def process_input_requests(self, recv_reqs):
            return recv_reqs

        def get_new_batch_prefill(self):
            return SimpleNamespace(batch_to_run=batch)

        def run_batch(self, value):
            return "ran"

        def process_batch_result(self, value):
            return "processed"

        def init_profiler(self):
            self.profiler_manager = SimpleNamespace(_profile=lambda req: None)

    monkeypatch.setattr(
        collector, "_schedule_batch_from_args", lambda *args, **kwargs: batch
    )
    collector.SchedulerCollectionHook.hook(Scheduler)
    scheduler = Scheduler()
    scheduler.init_profiler()

    scheduler.process_input_requests([request])
    assert not collector.COLLECTION_STATE.requests

    start = SimpleNamespace(req_type=SimpleNamespace(name="START_PROFILE"))
    stop = SimpleNamespace(req_type=SimpleNamespace(name="STOP_PROFILE"))
    assert scheduler.profiler_manager._profile(start).success
    scheduler.process_input_requests([request])
    scheduler.get_new_batch_prefill()
    assert scheduler.run_batch(batch) == "ran"
    assert scheduler.process_batch_result(batch) == "processed"
    assert scheduler.profiler_manager._profile(stop).success

    schedule = json.loads((tmp_path / "TP0.schedule_batch.jsonl").read_text())
    assert schedule["forward_mode"] == 1
    assert schedule["request_infos"] == [
        {
            "rid": "request-1",
            "extend_input_len": 2,
            "prefix_indices_len": 1,
            "output_ids_len": 1,
        }
    ]
    raw = json.loads((tmp_path / "TP0.raw_request.jsonl").read_text())
    assert raw["input_ids"] == [1, 2, 3]
    assert raw["output_ids"] == [4]
    assert not collector.COLLECTION_STATE.active


def test_tokenizer_hook_adds_server_created_time():
    class TokenizerManager:
        def _send_one_request(self, tokenized_obj):
            return tokenized_obj

    collector.TokenizerManagerCollectionHook.hook(TokenizerManager)
    tokenized = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params=None),
        time_stats=SimpleNamespace(created_time=123.0),
    )

    returned = TokenizerManager()._send_one_request(tokenized)

    assert returned.sampling_params.custom_params == {"server_created_time": 123.0}


def test_process_local_validation_does_not_require_frontend_in_scheduler(
    monkeypatch,
):
    """The scheduler child does not import TokenizerManager under spawn."""
    matched = {collector.SchedulerCollectionHook}
    monkeypatch.setattr(
        collector,
        "is_class_hook_matched",
        lambda hook: hook in matched,
    )

    collector.validate_scheduler_collection_hooks()
    with pytest.raises(RuntimeError, match="TokenizerManagerCollectionHook"):
        collector.validate_frontend_collection_hooks()
