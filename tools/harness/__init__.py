"""Kernel validation and benchmarking harness with lazy runtime imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ValidationResult",
    "validate_kernel_output",
    "generate_test_inputs",
    "BenchResult",
    "bench_kernel",
    "get_pygpubench_bridge",
]


if TYPE_CHECKING:
    from .bench import BenchResult, bench_kernel
    from .validate import ValidationResult, generate_test_inputs, validate_kernel_output


def _import_validate():
    try:
        from . import validate
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError(
                "tools.harness validation helpers require torch. "
                "Install the CUDA/PyTorch runtime before importing "
                "ValidationResult or validate_kernel_output."
            ) from exc
        raise
    return validate


def _import_bench():
    try:
        from . import bench
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError(
                "tools.harness benchmarking helpers require torch. "
                "Install the CUDA/PyTorch runtime before importing "
                "BenchResult or bench_kernel."
            ) from exc
        raise
    return bench


def __getattr__(name: str):
    if name in {"ValidationResult", "validate_kernel_output", "generate_test_inputs"}:
        validate = _import_validate()
        return getattr(validate, name)
    if name in {"BenchResult", "bench_kernel"}:
        bench = _import_bench()
        return getattr(bench, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_pygpubench_bridge():
    """Import the pygpubench bridge module (requires pygpubench installed)."""
    from . import pygpubench_bridge

    return pygpubench_bridge
