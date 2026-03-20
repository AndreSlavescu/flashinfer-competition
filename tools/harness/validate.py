"""
Kernel output validation with built-in integrity defenses.

All defenses are transparent and non-optional. The harness validates that:
1. Timing functions haven't been monkey-patched
2. No hidden CUDA streams were created
3. No background threads were spawned
4. Output tensors are fully materialized (not lazy/deferred)
5. Output precision matches spec (BF16 output, float32 LSE)
6. Outputs match the reference kernel within tight tolerances
"""

import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Module-level cached references (captured at import time, before any kernel
# code has a chance to monkey-patch them)
# ---------------------------------------------------------------------------
_ORIGINAL_CUDA_EVENT = torch.cuda.Event
_ORIGINAL_CUDA_SYNCHRONIZE = torch.cuda.synchronize
_ORIGINAL_PERF_COUNTER = time.perf_counter
_ORIGINAL_THREAD_COUNT = threading.active_count
_ORIGINAL_CURRENT_STREAM = torch.cuda.current_stream

# ---------------------------------------------------------------------------
# Import correctness utilities from existing kernelkit
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "tests"))

from kernelkit.compare import check_is_allclose, get_cos_diff  # noqa: E402


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Result of a full validation run including all defense checks."""
    passed: bool
    output_correct: bool
    lse_correct: bool
    output_cos_diff: float
    lse_cos_diff: float
    output_max_abs_err: float
    lse_max_abs_err: float
    defense_results: Dict[str, bool] = field(default_factory=dict)
    error_message: Optional[str] = None

    def summary(self) -> str:
        lines = [f"Validation {'PASSED' if self.passed else 'FAILED'}"]
        lines.append(f"  Output correct: {self.output_correct} (cos_diff={self.output_cos_diff:.2e}, max_abs={self.output_max_abs_err:.2e})")
        lines.append(f"  LSE correct:    {self.lse_correct} (cos_diff={self.lse_cos_diff:.2e}, max_abs={self.lse_max_abs_err:.2e})")
        for name, ok in self.defense_results.items():
            lines.append(f"  Defense [{name}]: {'OK' if ok else 'FAILED'}")
        if self.error_message:
            lines.append(f"  Error: {self.error_message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Defense functions
# ---------------------------------------------------------------------------
def _verify_no_monkey_patches() -> Tuple[bool, str]:
    """Verify that critical timing/sync functions haven't been replaced."""
    checks = [
        (torch.cuda.Event, _ORIGINAL_CUDA_EVENT, "torch.cuda.Event"),
        (torch.cuda.synchronize, _ORIGINAL_CUDA_SYNCHRONIZE, "torch.cuda.synchronize"),
        (time.perf_counter, _ORIGINAL_PERF_COUNTER, "time.perf_counter"),
    ]
    for current, original, name in checks:
        if current is not original:
            return False, f"{name} has been monkey-patched"
    return True, ""


def _capture_stream_state() -> torch.cuda.Stream:
    """Capture the current CUDA stream for later comparison."""
    return _ORIGINAL_CURRENT_STREAM()


def _check_streams(before_stream: torch.cuda.Stream) -> Tuple[bool, str]:
    """Verify no hidden streams were created and all work is complete."""
    # Force synchronization using the cached reference to catch hidden stream work
    _ORIGINAL_CUDA_SYNCHRONIZE()
    current = _ORIGINAL_CURRENT_STREAM()
    if current != before_stream:
        return False, f"CUDA stream changed: {before_stream} -> {current}"
    return True, ""


def _capture_thread_count() -> int:
    """Capture current thread count."""
    return _ORIGINAL_THREAD_COUNT()


def _check_threads(before_count: int) -> Tuple[bool, str]:
    """Verify no background threads were spawned during kernel execution."""
    after_count = _ORIGINAL_THREAD_COUNT()
    if after_count > before_count:
        return False, f"Thread injection detected: {before_count} -> {after_count} threads"
    return True, ""


def _verify_tensor_materialized(tensor: torch.Tensor, name: str) -> Tuple[bool, str]:
    """Verify a tensor is fully materialized, not a lazy/deferred wrapper."""
    # Check it's a base torch.Tensor, not a subclass
    if type(tensor) is not torch.Tensor:
        return False, f"{name}: not a base torch.Tensor (got {type(tensor).__name__})"
    # Check it's on CUDA
    if not tensor.is_cuda:
        return False, f"{name}: not on CUDA (device={tensor.device})"
    # Check storage is allocated
    if tensor.storage().size() == 0:
        return False, f"{name}: empty storage (size=0)"
    # Check data pointer is valid
    if tensor.data_ptr() == 0:
        return False, f"{name}: null data pointer"
    return True, ""


def _verify_precision(
    output: torch.Tensor,
    lse: torch.Tensor,
    ref_output: torch.Tensor,
    ref_lse: torch.Tensor,
) -> Tuple[bool, bool, float, float, float, float, str]:
    """
    Verify output dtypes match spec and outputs match reference.

    Returns: (output_ok, lse_ok, output_cos_diff, lse_cos_diff,
              output_max_abs, lse_max_abs, error_msg)
    """
    errors = []

    # Dtype checks
    if output.dtype != torch.bfloat16:
        errors.append(f"Output dtype: expected bfloat16, got {output.dtype}")
    if lse.dtype != torch.float32:
        errors.append(f"LSE dtype: expected float32, got {lse.dtype}")

    # Shape checks
    if output.shape != ref_output.shape:
        errors.append(f"Output shape: expected {ref_output.shape}, got {output.shape}")
    if lse.shape != ref_lse.shape:
        errors.append(f"LSE shape: expected {ref_lse.shape}, got {lse.shape}")

    if errors:
        return False, False, float("inf"), float("inf"), float("inf"), float("inf"), "; ".join(errors)

    # Numerical comparison using existing kernelkit utilities
    output_f = output.float()
    ref_output_f = ref_output.float()
    lse_f = lse.float()
    ref_lse_f = ref_lse.float()

    output_cos_diff = abs(get_cos_diff(output_f, ref_output_f))
    lse_cos_diff = abs(get_cos_diff(lse_f, ref_lse_f))

    output_max_abs = (output_f - ref_output_f).abs().max().item()
    lse_max_abs = (lse_f - ref_lse_f).abs().max().item()

    # Tight tolerances for competition
    output_ok = check_is_allclose(
        "output", output, ref_output,
        abs_tol=1e-2, rel_tol=1e-2, cos_diff_tol=1e-6, quiet=True,
    )
    lse_ok = check_is_allclose(
        "lse", lse, ref_lse,
        abs_tol=1e-3, rel_tol=1e-2, cos_diff_tol=1e-6, quiet=True,
    )

    error_msg = ""
    if not output_ok:
        error_msg += f"Output mismatch: cos_diff={output_cos_diff:.2e}, max_abs={output_max_abs:.2e}. "
    if not lse_ok:
        error_msg += f"LSE mismatch: cos_diff={lse_cos_diff:.2e}, max_abs={lse_max_abs:.2e}."

    return output_ok, lse_ok, output_cos_diff, lse_cos_diff, output_max_abs, lse_max_abs, error_msg


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------
def generate_test_inputs(
    num_tokens: int = 1,
    num_pages: int = 8462,
    page_size: int = 64,
    topk: int = 2048,
    h_q: int = 16,
    ckv_dim: int = 512,
    kpe_dim: int = 64,
    sm_scale: float = 0.135,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Generate competition-scale test inputs for the DSA decode kernel."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    total_kv_tokens = num_pages * page_size

    q_nope = torch.randn(num_tokens, h_q, ckv_dim, dtype=torch.bfloat16, device=device) / 10.0
    q_pe = torch.randn(num_tokens, h_q, kpe_dim, dtype=torch.bfloat16, device=device) / 10.0
    ckv_cache = torch.randn(num_pages, page_size, ckv_dim, dtype=torch.bfloat16, device=device) / 10.0
    kpe_cache = torch.randn(num_pages, page_size, kpe_dim, dtype=torch.bfloat16, device=device) / 10.0

    # Generate sparse indices: random valid token indices, with some -1 padding
    sparse_indices = torch.randint(
        0, total_kv_tokens, (num_tokens, topk), dtype=torch.int32, device=device,
    )
    # Mark ~5% as invalid (-1) to test padding handling
    invalid_mask = torch.rand(num_tokens, topk, device=device) < 0.05
    sparse_indices[invalid_mask] = -1

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": sm_scale,
    }


# ---------------------------------------------------------------------------
# Reference kernel runner
# ---------------------------------------------------------------------------
def _run_reference_kernel(inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the reference kernel and return (output, lse)."""
    sys.path.insert(0, str(_PROJECT_ROOT / "references"))
    import importlib
    ref_mod = importlib.import_module("dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64")
    return ref_mod.run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["sparse_indices"],
        inputs["sm_scale"],
    )


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------
def validate_kernel_output(
    kernel_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    inputs: dict,
    ref_output: Optional[torch.Tensor] = None,
    ref_lse: Optional[torch.Tensor] = None,
) -> ValidationResult:
    """
    Validate a kernel's output with all integrity defenses active.

    Args:
        kernel_fn: Callable that returns (output, lse) tensors.
        inputs: Dict of input tensors (for reference kernel if needed).
        ref_output: Pre-computed reference output. If None, runs reference kernel.
        ref_lse: Pre-computed reference LSE. If None, runs reference kernel.

    Returns:
        ValidationResult with correctness and defense check results.
    """
    defense_results = {}
    errors = []

    # --- Defense 1: Monkey-patch check ---
    mp_ok, mp_msg = _verify_no_monkey_patches()
    defense_results["monkey_patch"] = mp_ok
    if not mp_ok:
        errors.append(mp_msg)

    # --- Compute reference if not provided ---
    if ref_output is None or ref_lse is None:
        ref_output, ref_lse = _run_reference_kernel(inputs)

    # --- Pre-execution state capture ---
    before_stream = _capture_stream_state()
    before_threads = _capture_thread_count()

    # --- Execute kernel ---
    try:
        output, lse = kernel_fn()
    except Exception as e:
        return ValidationResult(
            passed=False, output_correct=False, lse_correct=False,
            output_cos_diff=float("inf"), lse_cos_diff=float("inf"),
            output_max_abs_err=float("inf"), lse_max_abs_err=float("inf"),
            defense_results=defense_results,
            error_message=f"Kernel execution failed: {e}",
        )

    # --- Defense 2: Stream check (also forces sync) ---
    stream_ok, stream_msg = _check_streams(before_stream)
    defense_results["stream_injection"] = stream_ok
    if not stream_ok:
        errors.append(stream_msg)

    # --- Defense 3: Thread check ---
    thread_ok, thread_msg = _check_threads(before_threads)
    defense_results["thread_injection"] = thread_ok
    if not thread_ok:
        errors.append(thread_msg)

    # --- Defense 4: Tensor materialization ---
    out_mat_ok, out_mat_msg = _verify_tensor_materialized(output, "output")
    lse_mat_ok, lse_mat_msg = _verify_tensor_materialized(lse, "lse")
    defense_results["output_materialized"] = out_mat_ok
    defense_results["lse_materialized"] = lse_mat_ok
    if not out_mat_ok:
        errors.append(out_mat_msg)
    if not lse_mat_ok:
        errors.append(lse_mat_msg)

    # --- Defense 5: Precision and correctness ---
    if out_mat_ok and lse_mat_ok:
        (output_ok, lse_ok, output_cos, lse_cos,
         output_abs, lse_abs, prec_msg) = _verify_precision(
            output, lse, ref_output, ref_lse,
        )
        defense_results["precision"] = output_ok and lse_ok
        if prec_msg:
            errors.append(prec_msg)
    else:
        output_ok = lse_ok = False
        output_cos = lse_cos = float("inf")
        output_abs = lse_abs = float("inf")

    all_passed = all(defense_results.values()) and output_ok and lse_ok

    return ValidationResult(
        passed=all_passed,
        output_correct=output_ok,
        lse_correct=lse_ok,
        output_cos_diff=output_cos,
        lse_cos_diff=lse_cos,
        output_max_abs_err=output_abs,
        lse_max_abs_err=lse_abs,
        defense_results=defense_results,
        error_message="; ".join(errors) if errors else None,
    )
