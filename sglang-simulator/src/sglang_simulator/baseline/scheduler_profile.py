"""Opt-in, bounded native Scheduler.run_batch profiling in a separate worker.

Ordinary baseline collection never imports/installs this module. Profiler
start/stop/export and metadata capture stay outside the collector's iteration
timer. record_function and CUPTI still perturb the diagnostic; its labels must
not replace unprofiled formal baselines.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sglang_simulator.baseline import collector


PROFILE_DIR = "SGLANG_NATIVE_PROFILE_DIR"
STEPS_ENV = "SGLANG_NATIVE_PROFILE_STEPS_PER_PHASE"
SKIP_ENV = "SGLANG_NATIVE_PROFILE_SKIP_PER_PHASE"
MARKER_PREFIX = "sglang_native_run_batch/"


def make_profiler() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("native scheduler profiling requires CUDA; no CPU fallback")
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
        profile_memory=False, with_stack=False, with_flops=False)


def record_range(name: str) -> Any:
    from torch.profiler import record_function

    return record_function(name)


def snapshot(scheduler: Any, batch: Any) -> dict:
    sequence = batch.seq_lens_cpu
    if sequence is None or sequence.device.type != "cpu":
        raise ValueError("require actual CPU seq_lens; do not introduce a GPU-copy probe")
    lengths = sequence.tolist()
    if len(lengths) != len(batch.reqs) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in lengths):
        raise ValueError("native sequence lengths do not match request rows")
    if getattr(getattr(scheduler, "ps", scheduler), "tp_rank", 0) != 0:
        raise ValueError("this profiling diagnostic requires single-rank TP0")
    runner = scheduler.tp_worker.model_runner
    modules = {name: getattr(sys.modules.get(name), "__file__", None)
               for name in ("sglang", "sgl_kernel", "torch")}
    return {"physical_prepared_seq_lens_cpu": lengths,
            "worker_pid": os.getpid(), "python_executable": sys.executable,
            "module_origins": modules,
            "model_dtype": str(runner.dtype), "kv_cache_dtype": str(runner.kv_cache_dtype),
            "requests": [{"rid": str(req.rid), "input_length": len(req.origin_input_ids),
                          "extend_input_len": collector._extend_input_length(batch, req),
                          "prefix_indices_len": len(req.prefix_indices),
                          "output_ids_len": len(req.output_ids)} for req in batch.reqs]}


class ProfileController:
    def __init__(self, directory: Path, steps: int = 3, skip: int = 2,
                 profiler_factory=make_profiler, range_factory=record_range,
                 synchronize=collector._synchronize_device) -> None:
        if steps < 1 or skip < 0:
            raise ValueError("profile steps must be positive and skip nonnegative")
        self.directory, self.steps, self.skip = directory, steps, skip
        self.profiler_factory, self.range_factory, self.synchronize = profiler_factory, range_factory, synchronize
        self.seen = {1: 0, 2: 0}
        self.marker = None

    def native_call(self, original: Callable, scheduler: Any, *args, **kwargs) -> Any:
        if self.marker is None:
            return original(scheduler, *args, **kwargs)
        with self.range_factory(self.marker):
            return original(scheduler, *args, **kwargs)

    def collected_call(self, original: Callable, scheduler: Any, *args, **kwargs) -> Any:
        if not collector.COLLECTION_STATE.active:
            return original(scheduler, *args, **kwargs)
        batch = collector._schedule_batch_from_args(*args, **kwargs)
        if batch is None:
            raise ValueError("cannot identify the native ScheduleBatch")
        mode = int(batch.forward_mode)
        if mode not in self.seen:
            return original(scheduler, *args, **kwargs)
        occurrence = self.seen[mode]
        self.seen[mode] += 1
        if not self.skip <= occurrence < self.skip+self.steps:
            return original(scheduler, *args, **kwargs)
        index = len(collector.COLLECTION_STATE.schedule_batches)
        metadata = snapshot(scheduler, batch)
        marker = f"{MARKER_PREFIX}mode{mode}/index{index}"
        self.directory.mkdir(parents=True, exist_ok=True)
        trace = self.directory / f"mode{mode}_index{index}.trace.json"
        record = trace.with_suffix(".metadata.json")
        if trace.exists() or record.exists():
            raise FileExistsError(f"refusing to overwrite a profile sample: {trace}")
        # Drain preparation/earlier work before starting the profiler. The
        # unchanged baseline wrapper synchronizes again around original run.
        self.synchronize()
        profiler = self.profiler_factory()
        profiler.start()
        self.marker = marker
        try:
            result = original(scheduler, *args, **kwargs)
        finally:
            self.marker = None
            profiler.stop()
        if len(collector.COLLECTION_STATE.schedule_batches) != index+1:
            raise ValueError("profile sample did not produce exactly one native timing row")
        measured = collector.COLLECTION_STATE.schedule_batches[index]
        if int(measured["forward_mode"]) != mode:
            raise ValueError("profile versus measured native phase differs")
        latency = float(measured["iter_latency"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("invalid diagnostic native iteration label")
        profiler.export_chrome_trace(str(trace))
        metadata.update(native_marker=marker, forward_mode=mode, schedule_line=index+1,
                        profile_phase_occurrence=occurrence, measured_batch=dict(measured),
                        trace_path=str(trace.resolve()), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                        scope="Instrumented diagnostic, not formal accuracy baseline. Profiler setup/export outside iteration timer; record_function/CUPTI overhead remains.")
        with record.open("x", encoding="utf-8") as output:
            json.dump(metadata, output, indent=2)
        return result


def install_on_scheduler(target: type, baseline_hook: Callable[[type], None],
                         controller_factory: Callable[[], ProfileController] | None = None) -> None:
    """Wrap the native call inside the unchanged baseline timing wrapper."""
    original = target.run_batch
    if controller_factory is None:
        def controller_factory():
            return ProfileController(Path(os.environ[PROFILE_DIR]),
                steps=int(os.environ.get(STEPS_ENV, "3")), skip=int(os.environ.get(SKIP_ENV, "2")))
    def controller(scheduler):
        if not hasattr(scheduler, "_native_profile_controller"):
            scheduler._native_profile_controller = controller_factory()
        return scheduler._native_profile_controller
    def native(self, *args, **kwargs):
        return controller(self).native_call(original, self, *args, **kwargs)
    target.run_batch = native
    baseline_hook(target)
    collected = target.run_batch
    def wrapped(self, *args, **kwargs):
        return controller(self).collected_call(collected, self, *args, **kwargs)
    target.run_batch = wrapped


def run_profiled_collection_scheduler_process(*args, **kwargs) -> Any:
    """Change only this freshly spawned worker's hook implementation."""
    original_hook = collector.SchedulerCollectionHook.hook.__func__
    def hook(cls, target):
        install_on_scheduler(target, lambda value: original_hook(cls, value))
    collector.SchedulerCollectionHook.hook = classmethod(hook)
    return collector.run_collection_scheduler_process(*args, **kwargs)
