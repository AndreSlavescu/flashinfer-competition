---
name: v2 kernel benchmark results
description: v2 vectorized PyTorch kernel passes 23/23 workloads at 1.80x average speedup; bottleneck is gather+FP32 cast at ~1.18ms fixed cost
type: project
---

## v2 Kernel Results (2026-03-28)

**Status**: 23/23 workloads PASS, 1.80x average speedup over reference

### Performance Profile
- Latency: ~1.18ms consistent across all num_tokens (1-8)
- T=1: 1.15ms (0.96x — slightly slower than reference's 1.10ms)
- T=8: 1.19ms (2.35-2.41x speedup over reference's 2.82ms)

### Bottleneck Analysis
- Fixed cost dominated by gather: `ckv_cache.reshape(-1, D)[safe_indices]` = O(topk * D) = 2MB BF16 read
- FP32 cast doubles memory traffic (kc_f, kp_f)
- 10+ PyTorch kernel launches add overhead
- Reference scales linearly (loops over tokens), vectorized version amortizes

### Key Implementation Decisions
1. FP32 GEMM required for precision match (BF16 GEMM gives different results from reference FP32 matmul)
2. `torch.einsum` slightly faster than `torch.bmm` for these shapes (avoids reshape overhead)
3. `masked_fill` + `contiguous()` needed before GEMM for correctness
4. LSE computed from logits BEFORE zeroing empty splits (otherwise logsumexp returns wrong values)

### What Didn't Help
- `torch.bmm` — same or slightly slower than einsum, more reshape overhead
- BF16 GEMM — precision mismatch with FP32 reference

### Next Steps for v3
- CuTeDSL fused kernel to eliminate gather overhead and FP32 cast
- TMA gather4 to avoid materializing gathered KV tensors
- Fused QK+softmax+SV in single kernel launch
