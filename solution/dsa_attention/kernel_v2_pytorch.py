"""
DSA Sparse Attention Kernel — v2 CuTeDSL-ready split-KV.

Uses batched matmul (torch.bmm) for QK and SV GEMMs instead of einsum.
PyTorch gather for sparse KV, PyTorch combine for split reduction.

The core GEMM compute path uses torch.bmm which maps directly to cuBLAS GEMMs.
The gather and combine steps remain in PyTorch.

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


def _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices):
    """Gather sparse KV rows and reshape into [T, S, N, D] chunks."""
    T = sparse_indices.shape[0]
    S = NUM_SPLITS
    N = CHUNK_SIZE
    D = CKV_DIM
    Dp = KPE_DIM

    valid_mask = sparse_indices != -1
    safe_indices = sparse_indices.clamp_min(0).long()
    vm = valid_mask.view(T, S, N)

    kc = ckv_cache.reshape(-1, D)[safe_indices.reshape(-1)].view(T, S, N, D)
    kp = kpe_cache.reshape(-1, Dp)[safe_indices.reshape(-1)].view(T, S, N, Dp)

    kc = kc.masked_fill(~vm.unsqueeze(-1), 0).contiguous()
    kp = kp.masked_fill(~vm.unsqueeze(-1), 0).contiguous()

    return kc, kp, vm


def _combine_splits(partial_o, partial_lse):
    """Combine 32 split partials via stable exp2 reduction."""
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


def _compute_split_partials(q_nope, q_pe, kc, kp, vm, sm_scale_f):
    """
    Compute partial outputs and LSEs for all (token, split) pairs.
    Uses batched matmul (torch.bmm) for GEMM operations.
    """
    T = q_nope.shape[0]
    H = NUM_Q_HEADS
    S = NUM_SPLITS
    N = CHUNK_SIZE
    D = CKV_DIM
    Dp = KPE_DIM
    device = q_nope.device

    q_nope_f = q_nope.float()
    q_pe_f = q_pe.float()
    kc_f = kc.float()
    kp_f = kp.float()

    # QK GEMM via batched matmul
    q_nope_exp = q_nope_f.unsqueeze(1).expand(T, S, H, D).reshape(T * S, H, D)
    q_pe_exp = q_pe_f.unsqueeze(1).expand(T, S, H, Dp).reshape(T * S, H, Dp)
    kc_r = kc_f.reshape(T * S, N, D).transpose(-1, -2)
    kp_r = kp_f.reshape(T * S, N, Dp).transpose(-1, -2)

    logits_ckv = torch.bmm(q_nope_exp, kc_r)
    logits_kpe = torch.bmm(q_pe_exp, kp_r)
    logits = (logits_ckv + logits_kpe) * sm_scale_f
    logits = logits.view(T, S, H, N)

    # Masked softmax
    mask = vm[:, :, None, :]
    split_valid = vm.any(dim=-1, keepdim=True)

    logits.masked_fill_(~mask, -float("inf"))

    partial_lse = torch.full((T, S, H), -float("inf"), dtype=torch.float32, device=device)
    valid_splits = split_valid.expand(-1, -1, H)
    partial_lse[valid_splits] = torch.logsumexp(logits, dim=-1)[valid_splits] * LOG2E

    logits = torch.where(split_valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(split_valid.unsqueeze(-1), attn, torch.zeros_like(attn))

    # SV GEMM via batched matmul
    attn_r = attn.reshape(T * S, H, N)
    kc_sv = kc_f.reshape(T * S, N, D)
    partial_o = torch.bmm(attn_r, kc_sv)
    partial_o = partial_o.view(T, S, H, D)

    partial_o = partial_o.permute(1, 0, 2, 3).contiguous()
    partial_lse = partial_lse.permute(1, 0, 2).contiguous()

    return partial_o, partial_lse


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute DSA sparse attention."""
    sm_scale_f = _to_python_float(sm_scale)

    kc, kp, vm = _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices)
    partial_o, partial_lse = _compute_split_partials(q_nope, q_pe, kc, kp, vm, sm_scale_f)
    output, final_lse = _combine_splits(partial_o, partial_lse)

    return output, final_lse


@torch.no_grad()
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """DPS-style kernel entry point — writes output and lse in place."""
    out, lse_out = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output.copy_(out)
    lse.copy_(lse_out)
