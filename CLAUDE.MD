I want to microbenchmark the B200 GPU and sm100_a compute architecture. The goal is to give a coding agent context to write an optimized deepseek sparse attention kernel for a competition.

Reference attention kernel:
```
import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        # For page_size=64, indices encode (page_idx * 64 + offset)
        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]
        qn = q_nope[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[t].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[t] = out.to(torch.bfloat16)

    return output, lse
```

Reference indexer kernel:
```
import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm format.
    
    Input: [num_pages, page_size, 1, 132] int8 (interpreted as uint8)
           Memory layout (per page): [fp8_data (page_size * 128 bytes), scales (page_size * 4 bytes)]
           After view to [num_pages, page_size, 1, 132]: NOT directly indexable as [fp8, scale] per token!
    Output: [num_pages, page_size, 128] float32
    """
    # View as uint8 for correct byte interpretation
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4  # 128
    
    # Go back to flat format to reverse the packing
    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
    
    # FP8 part: first page_size * head_dim bytes
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)
    
    # Scale part: last page_size * 4 bytes -> page_size float32 values
    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [num_pages, page_size, 1]
    
    return fp8_float * scale


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    # Check constants
    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    # Dequantize inputs
    q = q_index_fp8.to(torch.float32)  # [batch, heads, head_dim]
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, page_size, head_dim]

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    max_num_pages = block_table.shape[1]

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        
        if seq_len == 0:
            continue

        # Get pages for this sequence
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
        
        # Gather K from pages
        K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, head_dim]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, head_dim]
        
        # Query for this batch element
        q_b = q[b]  # [num_heads, head_dim]
        
        # Compute attention scores
        scores = q_b @ K.T  # [num_heads, seq_len]
        
        # Apply ReLU (deep_gemm uses ReLU activation)
        scores_relu = torch.relu(scores)  # [num_heads, seq_len]
        
        # Apply learned weights and sum across heads
        w = weights[b]  # [num_heads]
        weighted_scores = scores_relu * w[:, None]  # [num_heads, seq_len]
        final_scores = weighted_scores.sum(dim=0)  # [seq_len]
        
        # Select top-K
        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)
        
        # Convert to global token indices
        # Token index = page_idx * page_size + offset_in_page
        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token
        
        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)
```

I want to start with benchmarking the sm100 intrinsics of FlashMLA in /home/mark123/projects/FlashMLA/csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh
This project already contains a SOTA sparse attention kernel in /home/mark123/projects/FlashMLA/csrc/sm100/decode/head64/kernel.cuh

tma_gather4(...):

Some aspects I want to benchmark are:
1. memory throughput and latency
2. performance of pipelining and warp specialization
3. performance of various tensor shapes (size of row)
4. performance of various cache hints (evict_normal/first/last etc.)
5. performance of L2 partitions (fetching near vs far partition)
6. performance of various tensormaps and swizzling
7. shared memory performance (bank conflicts)

I want to measure memory throughput and latency under many combinations of the factors described above. However, if it's there's too much variance of a specific factor (ex. tensor shapes), use shapes that are used in the competition workloads.
