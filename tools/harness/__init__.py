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
