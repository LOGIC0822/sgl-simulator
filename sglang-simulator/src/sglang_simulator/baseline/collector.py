"""Runtime hooks for collecting request and schedule-batch baselines.

The collector records only while a client has issued ``/start_profile`` and
``/stop_profile``.  Each retained ``ScheduleBatch`` is synchronized before and
after ``Scheduler.run_batch``, so ``iter_latency`` is a device-complete forward
time in seconds rather than CPU enqueue time.

The scheduler is created in a spawned process.  Keep SGLang imports inside
``run_collection_scheduler_process`` so class hooks are installed before that
child imports the scheduler classes.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sglang_simulator.hook import BaseHook, install_class_hooks
from sglang_simulator.hook.class_hook_entry import is_class_hook_matched
from sglang_simulator.hook.utils import get_obj_from_args


COLLECTION_OUTPUT_DIR_ENV = "SGLANG_SIMULATOR_COLLECTION_DIR"


@dataclass
class RequestRecord:
    """Raw request state used for normalized trace and audit outputs."""

    rid: str = ""
    created_time: float | None = None
    client_created_time: float | None = None
    server_created_time: float | None = None
    queue_start: float = 0.0
    queue_end: float = 0.0
    output_length: int = 0
    input_length: int = 0
    recv_device_hit_len: int = 0
    before_adder_device_hit_len: int = 0
    final_device_hit_len: int = 0
    recv_host_hit_len: int = 0
    final_host_hit_len: int = 0
    recv_disk_hit_len: int = 0
    final_disk_hit_len: int = 0
    last_event_time: float = 0.0
    gen_token_latencies: list[float] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)


@dataclass
class CollectionState:
    """Per-scheduler-process in-memory collection state."""

    active: bool = False
    requests: dict[str, RequestRecord] = field(default_factory=dict)
    schedule_batches: list[dict[str, Any]] = field(default_factory=list)

    def start(self) -> None:
        if self.active:
            raise RuntimeError("baseline collection is already active")
        self.requests.clear()
        self.schedule_batches.clear()
        self.active = True

    def get_request(self, rid: str) -> RequestRecord:
        record = self.requests.get(rid)
        if record is None:
            record = RequestRecord(rid=rid)
            self.requests[rid] = record
        return record

    def stop_and_export(self, scheduler: Any) -> list[Path]:
        if not self.active:
            raise RuntimeError("baseline collection is not active")
        output_dir = os.getenv(COLLECTION_OUTPUT_DIR_ENV)
        if not output_dir:
            raise RuntimeError(
                f"{COLLECTION_OUTPUT_DIR_ENV} is required for baseline collection"
            )

        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        prefix = _rank_prefix(scheduler)
        page_size = int(getattr(scheduler, "page_size", 1))
        normalized_requests = [
            _normalized_request(record, page_size) for record in self.requests.values()
        ]
        raw_requests = [asdict(record) for record in self.requests.values()]

        paths = [
            _write_jsonl_exclusive(
                root / f"{prefix}.request.jsonl", normalized_requests
            ),
            _write_jsonl_exclusive(root / f"{prefix}.raw_request.jsonl", raw_requests),
            _write_jsonl_exclusive(
                root / f"{prefix}.schedule_batch.jsonl", self.schedule_batches
            ),
        ]
        manifest_path = root / f"{prefix}.manifest.json"
        _write_json_exclusive(
            manifest_path,
            {
                "schema_version": 1,
                "rank_prefix": prefix,
                "request_count": len(raw_requests),
                "schedule_batch_count": len(self.schedule_batches),
                "batch_latency_unit": "seconds",
                "batch_timing": "device_synchronized",
                "request_file": paths[0].name,
                "raw_request_file": paths[1].name,
                "schedule_batch_file": paths[2].name,
            },
        )
        paths.append(manifest_path)

        self.active = False
        self.requests.clear()
        self.schedule_batches.clear()
        return paths


COLLECTION_STATE = CollectionState()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return list(value)
    except TypeError as exc:
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        ) from exc


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, default=_json_default) + "\n")
    return path


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def _parallel_value(scheduler: Any, name: str, default: int) -> int:
    value = getattr(scheduler, name, None)
    parallel_state = getattr(scheduler, "ps", None)
    if value is None and parallel_state is not None:
        value = getattr(parallel_state, name, None)
    return default if value is None else int(value)


def _rank_prefix(scheduler: Any) -> str:
    prefix = f"TP{_parallel_value(scheduler, 'tp_rank', 0)}"
    for size_name, rank_name, label in (
        ("dp_size", "dp_rank", "DP"),
        ("pp_size", "pp_rank", "PP"),
        ("moe_ep_size", "moe_ep_rank", "EP"),
    ):
        if _parallel_value(scheduler, size_name, 1) > 1:
            prefix += f"-{label}{_parallel_value(scheduler, rank_name, 0)}"
    return prefix


def _normalized_request(record: RequestRecord, page_size: int) -> dict[str, Any]:
    host_hit = max(record.final_host_hit_len - record.final_disk_hit_len, 0)
    disk_hit = record.recv_disk_hit_len
    device_hit = record.before_adder_device_hit_len

    # A final partial cache page cannot be reused as a complete prefix.
    if device_hit + host_hit + disk_hit >= record.input_length:
        disk_hit -= min(page_size, disk_hit)

    timestamp = record.created_time or record.queue_start
    return {
        "rid": record.rid,
        "timestamp": timestamp * 1000,
        "input_length": record.input_length,
        "output_length": record.output_length,
        "device_cache_hit_length": device_hit,
        "host_cache_hit_length": host_hit,
        "disk_cache_hit_length": disk_hit,
    }


def _sampling_custom_params(sampling_params: Any) -> dict[str, Any]:
    if isinstance(sampling_params, dict):
        custom_params = sampling_params.get("custom_params")
        if custom_params is None:
            custom_params = {}
            sampling_params["custom_params"] = custom_params
        return custom_params

    custom_params = getattr(sampling_params, "custom_params", None)
    if custom_params is None:
        custom_params = {}
        setattr(sampling_params, "custom_params", custom_params)
    return custom_params


def _max_new_tokens(sampling_params: Any) -> int:
    if isinstance(sampling_params, dict):
        return int(sampling_params.get("max_new_tokens", 0))
    return int(getattr(sampling_params, "max_new_tokens", 0))


def _iter_received_requests(recv_reqs: Iterable[Any]) -> Iterable[Any]:
    for recv_req in recv_reqs:
        if recv_req.__class__.__name__.startswith("BatchTokenized"):
            yield from recv_req
        else:
            yield recv_req


def _record_received_requests(recv_reqs: Iterable[Any]) -> None:
    if not COLLECTION_STATE.active:
        return

    recv_time = time.time()
    for req in _iter_received_requests(recv_reqs):
        rid = getattr(req, "rid", None)
        if not rid or req.__class__.__name__ == "AbortReq":
            continue
        record = COLLECTION_STATE.get_request(str(rid))
        record.queue_start = recv_time
        record.last_event_time = recv_time
        custom_params = _sampling_custom_params(getattr(req, "sampling_params", {}))
        record.server_created_time = custom_params.get("server_created_time")
        record.client_created_time = custom_params.get("client_created_time")
        record.created_time = (
            record.client_created_time
            or record.server_created_time
            or record.queue_start
        )


def _schedule_batch_from_args(*args: Any, **kwargs: Any) -> Any:
    return get_obj_from_args(
        "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
    )


def _extend_input_length(batch: Any, req: Any) -> int:
    if batch.forward_mode.is_decode():
        return 1
    extend_range = getattr(req, "extend_range", None)
    if extend_range is not None:
        return int(extend_range.length)
    return int(getattr(req, "extend_input_len"))


def _synchronize_device() -> None:
    # Import after hook installation so this module does not eagerly import SGLang.
    from sglang.srt.platforms import current_platform

    current_platform.synchronize()


def _profile_request_is_start(recv_req: Any) -> bool:
    req_type = getattr(recv_req, "req_type", None)
    return getattr(req_type, "name", str(req_type)) == "START_PROFILE"


def _profile_output(message: str) -> Any:
    from sglang.srt.managers.io_struct import ProfileReqOutput

    return ProfileReqOutput(success=True, message=message)


def _handle_profile_request(scheduler: Any, recv_req: Any) -> Any:
    if _profile_request_is_start(recv_req):
        COLLECTION_STATE.start()
        return _profile_output("Baseline collection started")

    paths = COLLECTION_STATE.stop_and_export(scheduler)
    return _profile_output(
        "Baseline collection exported: " + ", ".join(path.name for path in paths)
    )


class TokenizerManagerCollectionHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target: Any) -> None:
        original = target._send_one_request

        def wrapped(self: Any, tokenized_obj: Any, *args: Any, **kwargs: Any) -> Any:
            if os.environ.get("SGLANG_BASELINE_DEBUG_IO") == "1":
                print(f"[baseline-debug] frontend send rid={getattr(tokenized_obj, 'rid', None)}", flush=True)
            sampling_params = getattr(tokenized_obj, "sampling_params", {})
            custom_params = _sampling_custom_params(sampling_params)
            time_stats = getattr(tokenized_obj, "time_stats", None)
            created_time = getattr(time_stats, "created_time", None)
            if created_time is not None:
                custom_params["server_created_time"] = created_time
            return original(self, tokenized_obj, *args, **kwargs)

        target._send_one_request = wrapped
        if os.environ.get("SGLANG_BASELINE_DEBUG_IO") == "1":
            original_loop_start = target.auto_create_handle_loop

            async def inspect_tasks() -> None:
                while True:
                    await asyncio.sleep(20)
                    for task in asyncio.all_tasks():
                        frames = []
                        coroutine = task.get_coro()
                        for _ in range(20):
                            frame = getattr(coroutine, "cr_frame", None) or getattr(coroutine, "gi_frame", None)
                            if frame is not None:
                                frames.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
                            coroutine = getattr(coroutine, "cr_await", None) or getattr(coroutine, "gi_yieldfrom", None)
                            if coroutine is None:
                                break
                        print(f"[baseline-debug] task {task.get_name()}: {' -> '.join(frames)}", flush=True)

            def start_loop(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_loop_start(self, *args, **kwargs)
                if not getattr(self, "_baseline_debug_task", None):
                    self._baseline_debug_task = asyncio.create_task(inspect_tasks())
                return result

            target.auto_create_handle_loop = start_loop


class PrefillAdderCollectionHook(BaseHook):
    HOOK_CLASS_NAME = "PrefillAdder"
    HOOK_MODULE_NAME = "sglang.srt.managers.schedule_policy"
    REQUIRED = False

    @classmethod
    def hook(cls, target: Any) -> None:
        original = target.add_one_req

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            req = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.Req", *args, **kwargs
            )
            if COLLECTION_STATE.active and req is not None:
                record = COLLECTION_STATE.get_request(str(req.rid))
                record.before_adder_device_hit_len = len(req.prefix_indices)
                record.final_host_hit_len = int(getattr(req, "host_hit_length", 0))
            return original(self, *args, **kwargs)

        target.add_one_req = wrapped


class SchedulerCollectionHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    @classmethod
    def hook(cls, target: Any) -> None:
        original_process_input = getattr(target, "process_input_requests", None)
        original_recv = getattr(target, "recv_requests", None)
        if original_process_input is None and original_recv is None:
            raise AttributeError(
                "Scheduler exposes neither process_input_requests nor recv_requests"
            )

        original_get_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_result = target.process_batch_result

        if original_process_input is not None:

            def wrapped_process_input(
                self: Any, recv_reqs: Iterable[Any], *args: Any, **kwargs: Any
            ) -> Any:
                if recv_reqs and os.environ.get("SGLANG_BASELINE_DEBUG_IO") == "1":
                    print(f"[baseline-debug] scheduler received {[type(req).__name__ for req in recv_reqs]}", flush=True)
                _record_received_requests(recv_reqs)
                return original_process_input(self, recv_reqs, *args, **kwargs)

            target.process_input_requests = wrapped_process_input
        else:

            def wrapped_recv(self: Any, *args: Any, **kwargs: Any) -> Any:
                recv_reqs = original_recv(self, *args, **kwargs)
                _record_received_requests(recv_reqs)
                return recv_reqs

            target.recv_requests = wrapped_recv

        def wrapped_get_prefill(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_get_prefill(self, *args, **kwargs)
            scheduled_batch = getattr(result, "batch_to_run", result)
            if (
                COLLECTION_STATE.active
                and scheduled_batch is not None
                and not scheduled_batch.is_empty()
            ):
                scheduled_time = time.time()
                for req in scheduled_batch.reqs:
                    record = COLLECTION_STATE.get_request(str(req.rid))
                    if record.queue_end == 0:
                        record.queue_end = scheduled_time
                        record.input_length = len(req.origin_input_ids)
                        record.output_length = _max_new_tokens(req.sampling_params)
                        record.final_device_hit_len = len(req.prefix_indices)
            return result

        def wrapped_run_batch(self: Any, *args: Any, **kwargs: Any) -> Any:
            if os.environ.get("SGLANG_BASELINE_DEBUG_IO") == "1":
                print("[baseline-debug] scheduler run_batch", flush=True)
            if not COLLECTION_STATE.active:
                return original_run_batch(self, *args, **kwargs)

            batch = _schedule_batch_from_args(*args, **kwargs)
            _synchronize_device()
            start = time.perf_counter()
            result = original_run_batch(self, *args, **kwargs)
            _synchronize_device()
            end = time.perf_counter()

            if batch is not None:
                request_infos = [
                    {
                        "rid": str(req.rid),
                        "extend_input_len": _extend_input_length(batch, req),
                        "prefix_indices_len": len(req.prefix_indices),
                        "output_ids_len": len(req.output_ids),
                    }
                    for req in batch.reqs
                ]
                COLLECTION_STATE.schedule_batches.append(
                    {
                        "start_timestamp": start,
                        "end_timestamp": end,
                        "forward_mode": int(batch.forward_mode),
                        "request_infos": request_infos,
                        "iter_latency": end - start,
                    }
                )
            return result

        def wrapped_process_result(self: Any, *args: Any, **kwargs: Any) -> Any:
            batch = _schedule_batch_from_args(*args, **kwargs)
            result = original_process_result(self, *args, **kwargs)
            if not COLLECTION_STATE.active or batch is None or batch.reqs is None:
                return result

            event_time = time.time()
            for req in batch.reqs:
                record = COLLECTION_STATE.get_request(str(req.rid))
                if len(req.output_ids) != 0 and record.last_event_time:
                    record.gen_token_latencies.append(
                        max(event_time - record.last_event_time, 0.0)
                    )
                    record.last_event_time = event_time
                if req.finished():
                    record.input_ids = list(req.origin_input_ids)
                    record.output_ids = list(req.output_ids)
                    record.output_length = len(record.output_ids)
            return result

        target.get_new_batch_prefill = wrapped_get_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_result

        original_init_profiler = getattr(target, "init_profiler", None)
        if original_init_profiler is not None:

            def wrapped_init_profiler(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_init_profiler(self, *args, **kwargs)
                self.profiler_manager._profile = lambda recv_req: _handle_profile_request(
                    self, recv_req
                )
                return result

            target.init_profiler = wrapped_init_profiler
        elif hasattr(target, "profile"):

            def wrapped_profile(
                self: Any, recv_req: Any, *args: Any, **kwargs: Any
            ) -> Any:
                return _handle_profile_request(self, recv_req)

            target.profile = wrapped_profile
        else:
            raise AttributeError("Scheduler exposes neither init_profiler nor profile")


class HiCacheCollectionHook(BaseHook):
    HOOK_CLASS_NAME = "HiCacheController"
    HOOK_MODULE_NAME = "sglang.srt.managers.cache_controller"
    REQUIRED = False

    @classmethod
    def hook(cls, target: Any) -> None:
        original_terminate = getattr(target, "terminate_prefetch", None)
        original_query = getattr(target, "_storage_hit_query", None)

        if original_terminate is not None:

            def wrapped_terminate(self: Any, operation: Any) -> Any:
                result = original_terminate(self, operation)
                request_id = getattr(operation, "request_id", None)
                if COLLECTION_STATE.active and request_id and result:
                    COLLECTION_STATE.get_request(
                        str(request_id)
                    ).final_disk_hit_len = int(result[0])
                return result

            target.terminate_prefetch = wrapped_terminate

        if original_query is not None:

            def wrapped_query(self: Any, operation: Any) -> Any:
                result = original_query(self, operation)
                request_id = getattr(operation, "request_id", None)
                if COLLECTION_STATE.active and request_id and result:
                    COLLECTION_STATE.get_request(
                        str(request_id)
                    ).recv_disk_hit_len = int(result[1])
                return result

            target._storage_hit_query = wrapped_query


_FRONTEND_HOOKS_INSTALLED = False
_SCHEDULER_HOOKS_INSTALLED = False


def install_frontend_collection_hooks() -> None:
    global _FRONTEND_HOOKS_INSTALLED
    if not _FRONTEND_HOOKS_INSTALLED:
        install_class_hooks(TokenizerManagerCollectionHook)
        _FRONTEND_HOOKS_INSTALLED = True


def install_scheduler_collection_hooks() -> None:
    global _SCHEDULER_HOOKS_INSTALLED
    if not _SCHEDULER_HOOKS_INSTALLED:
        install_class_hooks(
            [
                SchedulerCollectionHook,
                PrefillAdderCollectionHook,
                HiCacheCollectionHook,
            ]
        )
        _SCHEDULER_HOOKS_INSTALLED = True


def validate_frontend_collection_hooks() -> None:
    """Ensure the parent process instrumented its TokenizerManager class.

    The HTTP/tokenizer process and the scheduler process load different SGLang
    classes.  Checking every registered class hook from either process gives a
    false failure under ``spawn``: the scheduler never imports
    ``TokenizerManager``.  Validate each process-local required hook instead.
    """
    if not is_class_hook_matched(TokenizerManagerCollectionHook):
        raise RuntimeError(
            "TokenizerManagerCollectionHook did not match the imported SGLang "
            "TokenizerManager. The collector must be adapted to this SGLang "
            "revision."
        )


def validate_scheduler_collection_hooks() -> None:
    """Ensure the spawned scheduler process instrumented Scheduler."""
    if not is_class_hook_matched(SchedulerCollectionHook):
        raise RuntimeError(
            "SchedulerCollectionHook did not match the imported SGLang "
            "Scheduler. The collector must be adapted to this SGLang revision."
        )


def run_collection_scheduler_process(*args: Any, **kwargs: Any) -> Any:
    """Install hooks before importing and running the spawned scheduler."""
    install_scheduler_collection_hooks()
    from sglang.srt.managers.scheduler import run_scheduler_process

    validate_scheduler_collection_hooks()
    return run_scheduler_process(*args, **kwargs)
