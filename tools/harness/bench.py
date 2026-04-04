"""
Validated kernel benchmarking with integrity defenses.

Timing is performed with CUDA events, L2 cache flushing between runs,
and all validation defenses active. No kernel can game this harness
through stream injection, thread injection, lazy evaluation, precision
downgrading, or monkey-patching.
"""

import statistics
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from .validate import (
    ValidationResult,
    _ORIGINAL_CUDA_EVENT,
    _ORIGINAL_CUDA_SYNCHRONIZE,
    _capture_stream_state,
    _capture_thread_count,
    _check_streams,
    _check_threads,
    _verify_no_monkey_patches,
    _verify_precision,
    _verify_tensor_materialized,
)


@dataclass
class BenchResult:
    """Result of a validated benchmark run."""
    median_ms: float
    mean_ms: float
    min_ms: float
    p99_ms: float
    std_ms: float
    num_runs: int
    correctness_passed: bool
    validation_result: Optional[ValidationResult] = None
    timings_ms: Optional[list] = None

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.num_runs} runs",
            f"  Median: {self.median_ms:.3f} ms",
            f"  Mean:   {self.mean_ms:.3f} ms",
            f"  Min:    {self.min_ms:.3f} ms",
            f"  P99:    {self.p99_ms:.3f} ms",
            f"  Std:    {self.std_ms:.3f} ms",
            f"  Correct: {self.correctness_passed}",
        ]
        return "\n".join(lines)


def bench_kernel(
    kernel_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    ref_output: torch.Tensor,
    ref_lse: torch.Tensor,
    num_warmups: int = 10,
    num_runs: int = 100,
    flush_l2: bool = True,
) -> BenchResult:
    """
    Benchmark a kernel with all integrity defenses active.

    Args:
        kernel_fn: Callable that returns (output, lse).
        ref_output: Reference output for correctness validation.
        ref_lse: Reference LSE for correctness validation.
        num_warmups: Number of warmup iterations (not timed).
        num_runs: Number of timed iterations.
        flush_l2: Whether to flush L2 cache between runs.

    Returns:
        BenchResult with timing statistics and correctness.
    """
    # --- Pre-checks ---
    mp_ok, mp_msg = _verify_no_monkey_patches()
    if not mp_ok:
        raise RuntimeError(f"Integrity check failed: {mp_msg}")

    before_stream = _capture_stream_state()
    before_threads = _capture_thread_count()

    # L2 flush buffer (8 GB, matches bench_kineto pattern)
    l2_flush_buf = None
    if flush_l2:
        l2_flush_buf = torch.empty(int(8e9 // 4), dtype=torch.int32, device="cuda")

    # --- Warmup ---
    _ORIGINAL_CUDA_SYNCHRONIZE()
    for _ in range(num_warmups):
        kernel_fn()
    _ORIGINAL_CUDA_SYNCHRONIZE()

    # --- Timed runs ---
    start_events = [_ORIGINAL_CUDA_EVENT(enable_timing=True) for _ in range(num_runs)]
    end_events = [_ORIGINAL_CUDA_EVENT(enable_timing=True) for _ in range(num_runs)]

    last_output = None
    last_lse = None

    for i in range(num_runs):
        if flush_l2 and l2_flush_buf is not None:
            l2_flush_buf.zero_()

        start_events[i].record()
        output, lse = kernel_fn()
        end_events[i].record()

        # Keep last run's output for correctness check
        if i == num_runs - 1:
            last_output = output
            last_lse = lse

    # Force all work to complete (using cached reference)
    _ORIGINAL_CUDA_SYNCHRONIZE()

    # --- Collect timings ---
    timings_ms = [
        start_events[i].elapsed_time(end_events[i])
        for i in range(num_runs)
    ]

    # --- Post-checks ---
    stream_ok, stream_msg = _check_streams(before_stream)
    if not stream_ok:
        raise RuntimeError(f"Integrity check failed: {stream_msg}")

    thread_ok, thread_msg = _check_threads(before_threads)
    if not thread_ok:
        raise RuntimeError(f"Integrity check failed: {thread_msg}")

    # --- Correctness on last output ---
    correctness_passed = False
    validation_result = None

    if last_output is not None and last_lse is not None:
        # Verify materialization
        out_ok, _ = _verify_tensor_materialized(last_output, "output")
        lse_ok, _ = _verify_tensor_materialized(last_lse, "lse")

        if out_ok and lse_ok:
            (output_ok, lse_result_ok, output_cos, lse_cos,
             output_abs, lse_abs, prec_msg) = _verify_precision(
                last_output, last_lse, ref_output, ref_lse,
            )
            correctness_passed = output_ok and lse_result_ok
            validation_result = ValidationResult(
                passed=correctness_passed,
                output_correct=output_ok,
                lse_correct=lse_result_ok,
                output_cos_diff=output_cos,
                lse_cos_diff=lse_cos,
                output_max_abs_err=output_abs,
                lse_max_abs_err=lse_abs,
                defense_results={
                    "monkey_patch": mp_ok,
                    "stream_injection": stream_ok,
                    "thread_injection": thread_ok,
                    "output_materialized": out_ok,
                    "lse_materialized": lse_ok,
                    "precision": correctness_passed,
                },
            )

    # --- Statistics ---
    sorted_timings = sorted(timings_ms)
    p99_idx = max(0, int(0.99 * len(sorted_timings)) - 1)

    return BenchResult(
        median_ms=statistics.median(timings_ms),
        mean_ms=statistics.mean(timings_ms),
        min_ms=min(timings_ms),
        p99_ms=sorted_timings[p99_idx],
        std_ms=statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        num_runs=num_runs,
        correctness_passed=correctness_passed,
        validation_result=validation_result,
        timings_ms=timings_ms,
    )
