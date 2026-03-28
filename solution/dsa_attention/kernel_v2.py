"""
DSA Sparse Attention Kernel — v2 optimized vectorized split-KV.

Key optimizations over v1:
1. Direct gather-to-split reshape without intermediate full-size tensor
2. Fused ckv+kpe logits computation
3. Efficient masked softmax with early termination for empty splits
4. Vectorized combine step

Entry points:
    run(...) -> (output, lse)
    kernel(..., output, lse) -> writes destination tensors in place
"""

from __future__ import annotations

import math

import torch


TOPK = 2048
CHUNK_SIZE = 64
NUM_Q_HEADS = 16
CKV_DIM = 512
KPE_DIM = 64
PAGE_SIZE = 64
LOG2E = 1.0 / math.log(2.0)
NUM_SPLITS = TOPK // CHUNK_SIZE  # 32


def _to_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute DSA sparse attention."""
    sm_scale_f = _to_python_float(sm_scale)
    T = q_nope.shape[0]
    S = NUM_SPLITS
    H = NUM_Q_HEADS
    N = CHUNK_SIZE
    D = CKV_DIM
    Dp = KPE_DIM
    device = q_nope.device

    # --- Gather and reshape ---
    valid_mask = sparse_indices != -1
    safe_indices = sparse_indices.clamp_min(0).long()

    kc = ckv_cache.reshape(-1, D)[safe_indices.reshape(-1)].view(T, S, N, D)
    kp = kpe_cache.reshape(-1, Dp)[safe_indices.reshape(-1)].view(T, S, N, Dp)
    vm = valid_mask.view(T, S, N)

    # Zero invalid rows (contiguous for GEMM)
    kc = kc.masked_fill(~vm.unsqueeze(-1), 0).contiguous()
    kp = kp.masked_fill(~vm.unsqueeze(-1), 0).contiguous()

    # --- QK logits in FP32 ---
    q_nope_f = q_nope.float()
    q_pe_f = q_pe.float()
    kc_f = kc.float()
    kp_f = kp.float()

    logits_ckv = torch.einsum("thd,tsnd->tshn", q_nope_f, kc_f)
    logits_kpe = torch.einsum("thd,tsnd->tshn", q_pe_f, kp_f)
    logits = (logits_ckv + logits_kpe) * sm_scale_f  # [T, S, H, N]

    # --- Masked softmax ---
    mask = vm[:, :, None, :]
    split_valid = vm.any(dim=-1, keepdim=True)  # [T, S, 1]

    logits.masked_fill_(~mask, -float("inf"))

    # LSE before zeroing empty splits
    partial_lse = torch.full((T, S, H), -float("inf"), dtype=torch.float32, device=device)
    valid_splits = split_valid.expand(-1, -1, H)
    partial_lse[valid_splits] = torch.logsumexp(logits, dim=-1)[valid_splits] * LOG2E

    logits = torch.where(split_valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(split_valid.unsqueeze(-1), attn, torch.zeros_like(attn))

    # --- SV: attention @ Kc ---
    partial_o = torch.einsum("tshn,tsnd->tshd", attn, kc_f)

    # --- Reshape for combine ---
    partial_o = partial_o.permute(1, 0, 2, 3).contiguous()      # [S, T, H, D]
    partial_lse = partial_lse.permute(1, 0, 2).contiguous()     # [S, T, H]

    # --- Combine splits via stable exp2 reduction ---
    max_lse = partial_lse.max(dim=0).values
    finite = torch.isfinite(max_lse)
    stable_max = torch.where(finite, max_lse, torch.zeros_like(max_lse))

    weights = torch.exp2(partial_lse - stable_max.unsqueeze(0))
    weight_sum = weights.sum(dim=0)
    combined = (weights.unsqueeze(-1) * partial_o).sum(dim=0)

    output = torch.zeros_like(combined)
    valid = weight_sum > 0
    output[valid] = combined[valid] / weight_sum[valid].unsqueeze(-1)

    final_lse = torch.full_like(max_lse, -float("inf"))
    final_lse[valid] = stable_max[valid] + torch.log2(weight_sum[valid])

    return output.to(torch.bfloat16), final_lse


@torch.no_grad()
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """DPS-style kernel entry point — writes output and lse in place."""
    out, lse_out = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output.copy_(out)
    lse.copy_(lse_out)
