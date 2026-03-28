"""
DSA Sparse Attention Kernel — Reference (Triton placeholder).

This is the reference implementation wrapped as a Triton-compatible entry point.
Replace with an optimized kernel.

Signature (DPS — destination passing style):
    kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)

Inputs:
    q_nope:         [num_tokens, 16, 512]   bf16
    q_pe:           [num_tokens, 16, 64]    bf16
    ckv_cache:      [num_pages, 64, 512]    bf16
    kpe_cache:      [num_pages, 64, 64]     bf16
    sparse_indices: [num_tokens, 2048]      int32  (-1 = invalid)
    sm_scale:       float32 scalar (~0.1352)

Outputs (pre-allocated):
    output:         [num_tokens, 16, 512]   bf16
    lse:            [num_tokens, 16]         float32  (log2 base)
"""

import math
import torch


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """Reference implementation — correct but slow. Replace with optimized kernel."""
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]

    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    for t in range(num_tokens):
        indices = sparse_indices[t]
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            lse[t].fill_(float("-inf"))
            continue

        tok_idx = valid_indices.to(torch.long)
        Kc = Kc_all[tok_idx]
        Kp = Kp_all[tok_idx]
        qn = q_nope[t].to(torch.float32)
        qp = q_pe[t].to(torch.float32)

        logits = (qn @ Kc.T) + (qp @ Kp.T)
        logits_scaled = logits * sm_scale

        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)
        out = attn @ Kc
        output[t] = out.to(torch.bfloat16)
