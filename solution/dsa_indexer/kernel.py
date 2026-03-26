"""
DSA TopK Indexer Kernel — Reference (Triton placeholder).

This is the reference implementation wrapped as a DPS-compatible entry point.
Replace with an optimized kernel.

Signature (DPS — destination passing style):
    kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)

Inputs:
    q_index_fp8:      [batch_size, 64, 128]              float8_e4m3fn
    k_index_cache_fp8:[num_pages, 64, 1, 132]            int8 (deep_gemm FP8 format)
    weights:          [batch_size, 64]                   float32
    seq_lens:         [batch_size]                       int32
    block_table:      [batch_size, max_num_pages]        int32

Outputs (pre-allocated):
    topk_indices:     [batch_size, 2048]                 int32  (-1 = padding)
"""

import torch


def _dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize deep_gemm FP8 KV cache to float32."""
    k = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k.shape
    head_dim = head_dim_sf - 4  # 128

    kv_flat = k.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_float = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn).to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [num_pages, page_size, 1]

    return fp8_float * scale


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """Reference implementation — correct but slow. Replace with optimized kernel."""
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size = k_index_cache_fp8.shape[0], k_index_cache_fp8.shape[1]
    topk = topk_indices.shape[1]

    q = q_index_fp8.to(torch.float32)
    K_all = _dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, page_size, 128]

    topk_indices.fill_(-1)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, 128]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, 128]

        q_b = q[b]  # [64, 128]
        scores = torch.relu(q_b @ K.T)  # [64, seq_len]
        w = weights[b]  # [64]
        final_scores = (scores * w[:, None]).sum(dim=0)  # [seq_len]

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
