"""Small numeric comparison helpers for the local harness."""

from __future__ import annotations

import torch


def get_cos_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return 1 - cosine similarity for two tensors."""
    actual_f = actual.float().reshape(-1)
    expected_f = expected.float().reshape(-1)

    if actual_f.numel() == 0 and expected_f.numel() == 0:
        return 0.0

    actual_norm = torch.linalg.vector_norm(actual_f)
    expected_norm = torch.linalg.vector_norm(expected_f)
    if actual_norm.item() == 0.0 and expected_norm.item() == 0.0:
        return 0.0
    if actual_norm.item() == 0.0 or expected_norm.item() == 0.0:
        return 1.0

    cosine = torch.nn.functional.cosine_similarity(
        actual_f.unsqueeze(0),
        expected_f.unsqueeze(0),
        dim=1,
    ).item()
    return 1.0 - float(cosine)


def check_is_allclose(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    abs_tol: float,
    rel_tol: float,
    cos_diff_tol: float,
    quiet: bool = False,
) -> bool:
    """Check numeric closeness with both allclose and cosine drift guards."""
    if actual.shape != expected.shape:
        if not quiet:
            print(f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}")
        return False

    actual_f = actual.float()
    expected_f = expected.float()
    allclose_ok = torch.allclose(actual_f, expected_f, atol=abs_tol, rtol=rel_tol)
    cos_diff = abs(get_cos_diff(actual_f, expected_f))
    passed = allclose_ok and cos_diff <= cos_diff_tol

    if not quiet and not passed:
        max_abs = (actual_f - expected_f).abs().max().item()
        print(
            f"{name}: allclose={allclose_ok} cos_diff={cos_diff:.2e} "
            f"max_abs={max_abs:.2e} atol={abs_tol:.2e} rtol={rel_tol:.2e}"
        )

    return passed
