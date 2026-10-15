"""Early compatibility checks for the SGLang surfaces used by the simulator."""

import inspect
import os
from importlib import metadata

SIMULATOR_SERVER_ARG_OVERRIDES = {
    # The simulator models the target deployment topology separately through
    # sim_config.scheduler. Keep the SGLang runtime single-process so host-side
    # simulator work is not multiplied by the modeled parallel world size.
    "tp_size": 1,
    "ep_size": 1,
    "dp_size": 1,
    "pp_size": 1,
    "attn_cp_size": 1,
    "dcp_size": 1,
    "disable_overlap_schedule": True,
    "disable_cuda_graph": True,
    "attention_backend": "torch_native",
    "prefill_attention_backend": "torch_native",
    "decode_attention_backend": "torch_native",
}


class SGLangCompatibilityError(RuntimeError):
    pass


def _sglang_version() -> str:
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return "source-checkout"


def _require_parameters(function, required: set[str], surface: str) -> None:
    parameters = set(inspect.signature(function).parameters)
    missing = required - parameters
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang {_sglang_version()} is missing {surface} parameters: "
            f"{', '.join(sorted(missing))}. The simulator must be adapted to "
            "this SGLang revision before it can run."
        )


def apply_simulator_server_args(target) -> None:
    """Apply simulator-owned values before constructing the final ServerArgs."""
    # Most CPU simulator runs use the no-kernel torch-native backend.  Keep
    # this optional override for callers that intentionally own another
    # backend contract.  The data-tools dual-chunk replay instead uses the
    # narrower validation bypass below and retains torch-native allocation.
    overrides = dict(SIMULATOR_SERVER_ARG_OVERRIDES)
    attention_backend = os.getenv("SGLANG_SIMULATOR_ATTENTION_BACKEND")
    if attention_backend:
        overrides["attention_backend"] = attention_backend
        overrides["prefill_attention_backend"] = attention_backend
        overrides["decode_attention_backend"] = attention_backend
    if isinstance(target, dict):
        target.update(overrides)
        return

    for name, value in overrides.items():
        setattr(target, name, value)


def install_cpu_dual_chunk_validation_bypass() -> None:
    """Allow the CPU simulator to use torch-native scheduling for dual-chunk models.

    The simulator never executes the model's attention kernel: its forward
    duration comes from the configured predictor.  SGLang nevertheless
    validates a checkpoint's ``dual_chunk_attention_config`` before scheduler
    hooks take over and rejects the CPU engine's required ``torch_native``
    backend.  This opt-in patch suppresses only that contradiction.  It does
    not enable dual-chunk CUDA execution, alter a GPU server, or affect a
    checkpoint without dual-chunk configuration.
    """
    if os.getenv("SGLANG_SIMULATOR_ALLOW_CPU_DUAL_CHUNK") != "1":
        return

    from sglang.srt.arg_groups import attention_hook, overrides

    original = overrides._attention_backend_dual_chunk
    if getattr(original, "_sglang_simulator_cpu_bypass", False):
        return

    def cpu_compatible_dual_chunk(view):
        model_config = overrides.model_config_of(view)
        dual_chunk_config = getattr(
            getattr(model_config, "hf_config", None),
            "dual_chunk_attention_config",
            None,
        )
        if dual_chunk_config is not None and view.attention_backend == "torch_native":
            return {}
        return original(view)

    cpu_compatible_dual_chunk._sglang_simulator_cpu_bypass = True
    overrides._attention_backend_dual_chunk = cpu_compatible_dual_chunk
    # ``attention_hook`` imports this function directly, so patch both module
    # references before ServerArgs resolves the backend policy.
    attention_hook._attention_backend_dual_chunk = cpu_compatible_dual_chunk


def validate_simulator_server_args(server_args) -> None:
    """Fail early if a process bypassed a simulator-owned launch entry point."""
    expected = dict(SIMULATOR_SERVER_ARG_OVERRIDES)
    # Keep this validation in lockstep with ``apply_simulator_server_args``.
    # The CPU simulator normally declares torch_native, whereas a dual-chunk
    # checkpoint must retain its model-required backend declaration through
    # SGLang's argument-resolution phase even though no CUDA attention kernel
    # is executed by the simulator hooks.
    attention_backend = os.getenv("SGLANG_SIMULATOR_ATTENTION_BACKEND")
    if attention_backend:
        expected["attention_backend"] = attention_backend
        expected["prefill_attention_backend"] = attention_backend
        expected["decode_attention_backend"] = attention_backend
    mismatches = [
        f"{name}={getattr(server_args, name, None)!r} (expected {expected!r})"
        for name, expected_value in expected.items()
        if getattr(server_args, name, None) != expected_value
    ]
    if mismatches:
        raise SGLangCompatibilityError(
            "SGLang Simulator server arguments were not prepared by a supported "
            f"entry point: {', '.join(mismatches)}"
        )


def validate_launch_runtime() -> None:
    from sglang.srt.entrypoints.http_server import launch_server

    _require_parameters(
        launch_server,
        {"server_args", "run_scheduler_process_func", "run_detokenizer_process_func"},
        "launch_server",
    )


def validate_benchmark_runtime() -> None:
    from sglang.benchmark import serving

    missing = [
        name
        for name in (
            "BenchmarkMetrics",
            "calculate_metrics",
            "cli_main",
            "get_request",
            "run_benchmark",
        )
        if not hasattr(serving, name)
    ]
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang {_sglang_version()} is missing benchmark surfaces: "
            f"{', '.join(missing)}. The simulator must be adapted to this "
            "SGLang revision before it can run."
        )
