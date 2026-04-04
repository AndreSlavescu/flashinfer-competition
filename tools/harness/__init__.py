"""Kernel validation and benchmarking harness with integrity defenses."""

from .validate import ValidationResult, validate_kernel_output, generate_test_inputs
from .bench import BenchResult, bench_kernel

__all__ = [
    "ValidationResult",
    "validate_kernel_output",
    "generate_test_inputs",
    "BenchResult",
    "bench_kernel",
]

# pygpubench bridge is imported lazily (requires pygpubench to be installed)
def get_pygpubench_bridge():
    """Import the pygpubench bridge module (requires pygpubench installed)."""
    from . import pygpubench_bridge
    return pygpubench_bridge
