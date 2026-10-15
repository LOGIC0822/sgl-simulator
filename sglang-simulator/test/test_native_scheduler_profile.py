"""Bounded profiling wrapper tests with a fake profiler; no GPU server."""
import contextlib
import json
from types import SimpleNamespace

import pytest

from sglang_simulator.baseline import collector, scheduler_profile as profile


@pytest.fixture(autouse=True)
def state():
    collector.COLLECTION_STATE.active = False
    collector.COLLECTION_STATE.schedule_batches.clear()
    collector.COLLECTION_STATE.requests.clear()
    yield
    collector.COLLECTION_STATE.active = False
    collector.COLLECTION_STATE.schedule_batches.clear()
    collector.COLLECTION_STATE.requests.clear()


def fixture(tmp_path, monkeypatch, steps=1, skip=1):
    timeline = []
    class Mode:
        def __int__(self):
            return 2
        def is_decode(self):
            return True
    sequence = SimpleNamespace(device=SimpleNamespace(type="cpu"), tolist=lambda: [24])
    req = SimpleNamespace(rid="r", origin_input_ids=[1]*20, output_ids=[2]*4,
                          prefix_indices=[0]*20, extend_input_len=1)
    batch = SimpleNamespace(forward_mode=Mode(), reqs=[req], seq_lens_cpu=sequence)
    class Scheduler:
        ps = SimpleNamespace(tp_rank=0)
        tp_worker = SimpleNamespace(model_runner=SimpleNamespace(dtype="torch.bfloat16", kv_cache_dtype="torch.bfloat16"))
        def process_input_requests(self, reqs):
            return reqs
        def get_new_batch_prefill(self):
            return None
        def run_batch(self, value):
            timeline.append("native")
            return "result"
        def process_batch_result(self, value):
            return "processed"
        def init_profiler(self):
            self.profiler_manager = SimpleNamespace(_profile=lambda value: None)
    class FakeProfiler:
        def start(self):
            timeline.append("profiler_start")
        def stop(self):
            timeline.append("profiler_stop")
            assert collector.COLLECTION_STATE.schedule_batches
        def export_chrome_trace(self, path):
            timeline.append("export")
            assert timeline[-2] == "profiler_stop"
            with open(path, "x") as output:
                json.dump({"traceEvents": []}, output)
    @contextlib.contextmanager
    def scope(name):
        timeline.append("range_start:"+name)
        yield
        timeline.append("range_end:"+name)
    monkeypatch.setattr(collector, "_synchronize_device", lambda: timeline.append("synchronize"))
    monkeypatch.setattr(collector, "_schedule_batch_from_args", lambda *args, **kwargs: batch)
    controller = profile.ProfileController(tmp_path / "traces", steps=steps, skip=skip,
        profiler_factory=FakeProfiler, range_factory=scope, synchronize=lambda: timeline.append("drain"))
    profile.install_on_scheduler(Scheduler, collector.SchedulerCollectionHook.hook, lambda: controller)
    return Scheduler(), batch, controller, timeline


def test_profile_only_selected_active_steps_and_exports_outside_label(tmp_path, monkeypatch):
    scheduler, batch, controller, timeline = fixture(tmp_path, monkeypatch)
    assert scheduler.run_batch(batch) == "result"
    assert timeline == ["native"]
    collector.COLLECTION_STATE.start()
    timeline.clear()
    scheduler.run_batch(batch)  # Skip first active decode iteration.
    assert "profiler_start" not in timeline
    timeline.clear()
    scheduler.run_batch(batch)
    assert timeline.index("drain") < timeline.index("profiler_start") < timeline.index("native")
    assert timeline.index("native") < timeline.index("profiler_stop") < timeline.index("export")
    markers = [value for value in timeline if value.startswith("range_start:")]
    assert len(markers) == 1 and "mode2/index1" in markers[0]
    paths = list((tmp_path / "traces").glob("*.metadata.json"))
    assert len(paths) == 1
    value = json.loads(paths[0].read_text())
    assert value["schedule_line"] == 2
    assert value["physical_prepared_seq_lens_cpu"] == [24]
    assert value["requests"][0]["prefix_indices_len"] == 20
    assert value["requests"][0]["output_ids_len"] == 4
    assert value["measured_batch"]["forward_mode"] == 2
    assert value["model_dtype"] == value["kv_cache_dtype"] == "torch.bfloat16"
    timeline.clear()
    scheduler.run_batch(batch)  # Budget exhausted.
    assert "profiler_start" not in timeline and controller.marker is None


def test_profile_failure_clears_marker_and_stops_profiler(tmp_path, monkeypatch):
    scheduler, batch, controller, _ = fixture(tmp_path, monkeypatch, skip=0)
    collector.COLLECTION_STATE.start()
    stopped = []
    class FailureProfiler:
        def start(self):
            pass
        def stop(self):
            stopped.append(True)
    controller.profiler_factory = FailureProfiler
    def fail(*args, **kwargs):
        raise RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        controller.collected_call(fail, scheduler, batch)
    assert controller.marker is None and stopped == [True]


@pytest.mark.parametrize("kind", ["cuda_seq", "shape", "rank"])
def test_profile_refuses_ambiguous_native_metadata(tmp_path, monkeypatch, kind):
    scheduler, batch, _, timeline = fixture(tmp_path, monkeypatch, skip=0)
    collector.COLLECTION_STATE.start()
    if kind == "cuda_seq":
        batch.seq_lens_cpu.device.type = "cuda"
    elif kind == "shape":
        batch.seq_lens_cpu.tolist = lambda: [24, 25]
    else:
        scheduler.ps = SimpleNamespace(tp_rank=1)
    with pytest.raises(ValueError):
        scheduler.run_batch(batch)
    assert "profiler_start" not in timeline


def test_native_profiler_has_no_silent_cpu_fallback(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        profile.make_profiler()


def test_profiled_launcher_forwards_native_flags_to_isolated_callback(tmp_path, monkeypatch):
    from sglang_simulator import baseline
    from sglang_simulator.baseline import profiled_launcher
    observed = []
    fake = SimpleNamespace(main=lambda args: observed.append(args))
    monkeypatch.setattr(baseline, "launch_server", fake, raising=False)
    for key in (profile.PROFILE_DIR, profile.STEPS_ENV, profile.SKIP_ENV):
        monkeypatch.setenv(key, "previous")
    server = ["--device", "cuda", "--tp-size", "1", "--disable-cuda-graph",
              "--disable-overlap-schedule", "--model-path", "/fixture/model"]
    profiled_launcher.main(["--native-profile-dir", str(tmp_path / "new"),
                           "--native-profile-steps-per-phase", "2", *server])
    assert observed == [server]
    assert fake.run_collection_scheduler_process is profile.run_profiled_collection_scheduler_process


@pytest.mark.parametrize("flags", [[], ["--device", "cpu"], ["--tp-size", "2"]])
def test_profiled_launcher_rejects_unmatched_execution_settings(tmp_path, flags):
    from sglang_simulator.baseline import profiled_launcher
    with pytest.raises(SystemExit) as exc:
        profiled_launcher.main(["--native-profile-dir", str(tmp_path / "new"), *flags])
    assert exc.value.code == 2
