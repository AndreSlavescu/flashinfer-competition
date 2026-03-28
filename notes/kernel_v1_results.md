# Kernel v1 Results

## Official Bench

| num_tokens | count | latency_ms (range) | ref_latency_ms (range) | speedup (range) |
|------------|-------|---------------------|------------------------|-----------------|
| 1 | 1 | 1.351 | 1.111 | 0.82x |
| 2 | 8 | 1.320–1.346 | 1.351–1.380 | 1.01–1.03x |
| 6 | 3 | 1.347–1.363 | 2.371–2.473 | 1.74–1.84x |
| 7 | 3 | 1.360–1.377 | 2.650–2.721 | 1.95–1.98x |
| 8 | 8 | 1.349–1.370 | 2.855–2.983 | 2.09–2.19x |

**Average speedup: 1.63x** (23/23 workloads PASSED)

Key observations:
- Our latency is nearly constant ~1.32–1.38ms regardless of num_tokens (1-8)
- Reference scales roughly linearly with num_tokens
- For num_tokens=1, we are SLOWER than reference (0.82x)
- Max correctness errors are within tolerance (max_abs_err <= 7.81e-03, max_rel_err <= 5.42e-01)

## NCU Summary

NCU was NOT run on v1 because v1 is a pure PyTorch baseline with no custom CUDA kernel. It launches dozens of internal cuBLAS/aten kernels per invocation. NCU on individual torch ops would not yield actionable single-kernel optimization data.

## SASS Pipeline Analysis

Not applicable — v1 has no custom CUDA kernel in the default benchmark path. The embedded CuTeDSL QK experiment is gated behind `DSA_ATTENTION_ENABLE_CUTEDSL_QK=1` and is not benchmarked.

## Harness Bench

Not run — v1 is not a single-kernel target suitable for harness bench. The 1.32-1.38ms range from the official bench is the ground truth.

## Algorithmic Bottleneck Analysis

### Data movement per token (num_tokens=1)

| Stage | Data Read | Data Written | Notes |
|-------|-----------|--------------|-------|
| Gather ckv | 2048 x 512 x 2B = 2.0 MB | 2.0 MB (contiguous) | Random gather from paged cache |
| Gather kpe | 2048 x 64 x 2B = 256 KB | 256 KB (contiguous) | Random gather from paged cache |
| Cast to FP32 | 2.25 MB | 4.5 MB | Doubles data size |
| QK ckv einsum | ~4 MB (Q+K) | 128 KB (logits) | torch.einsum "thd,tsnd->tshn" |
| QK kpe einsum | ~0.5 MB | 128 KB | torch.einsum "thd,tsnd->tshn" |
| Softmax | 128 KB | 128 KB | Per-split |
| SV einsum | ~4.1 MB (attn+K) | 1 MB (partial_o) | torch.einsum "tshn,tsnd->tshd" |
| Combine | ~1.1 MB | 16 KB (output) | 32 partial results |
| **Total** | **~14 MB** | **~8 MB** | Multiple HBM round-trips |

### Compute per token

| Stage | FLOPs | Notes |
|-------|-------|-------|
| QK ckv | 33.6M | 1 x 32 x 16 x 64 x 512 x 2 |
| QK kpe | 4.2M | 1 x 32 x 16 x 64 x 64 x 2 |
| SV | 33.6M | 1 x 32 x 16 x 64 x 512 x 2 |
| Softmax/combine | ~2M | |
| **Total** | **~73M** | |

### Arithmetic intensity

- Total compute: ~73M FLOPs
- Total data movement: ~22 MB (read + write)
- Arithmetic intensity: 73M / 22M = 3.3 FLOP/byte
- This is extremely memory-bound (B200 BF16 peak: ~8PF/s at 7.67 TB/s = 1040 FLOP/byte balance point)

### Why v1 is slow

PyTorch eager execution launches separate kernels for each operation:
1. `index_select` for gather (2 launches: ckv + kpe)
2. `to(float32)` casts (4 launches)
3. `masked_fill` (2 launches)
4. `einsum` for QK ckv (cublas GEMM launch)
5. `einsum` for QK kpe (cublas GEMM launch)
6. Addition + scale (2 launches)
7. `masked_fill` + `softmax` (multiple launches)
8. `einsum` for SV (cublas GEMM launch)
9. `logsumexp` + masking
10. `permute` + `contiguous`
11. Combine: max, exp2, weighted sum, div

Each kernel writes intermediates to HBM and the next kernel reads them back. The ~14 MB of data per token makes ~5+ round-trips through HBM, wasting bandwidth on intermediate tensors that a fused kernel would keep in registers/SMEM/TMEM.

### Target performance with fused CuTeDSL kernel

From measured B200 hardware (timing model):
- Per-SM with split-KV (1 chunk/SM): TMA ~3,752ns + GEMM ~1,809ns = ~5,561ns
- With TMA/GEMM pipelining: ~3,752ns (TMA-bound)
- 32 chunks on 148 SMs: 1 wave = ~5.6us per token
- At num_tokens=8: 256 CTAs on 148 SMs = 2 waves = ~11.2us
- Plus combine kernel: ~2-5us
- **Target: ~8-16us = 0.008-0.016ms** vs current 1.35ms = **84-168x improvement potential**

## Analysis

v1 is a pure PyTorch baseline. The single highest-impact optimization is replacing the entire computation path with a fused CuTeDSL warp-specialized attention kernel that:
1. Loads Q into TMEM once
2. Streams KV tiles through SMEM via TMA gather4 (one chunk of 64 tokens per CTA)
3. Computes QK (ckv + kpe) + softmax + SV in TMEM/registers
4. Writes only the final partial output and LSE
5. Combines 32 partial results in a separate kernel

This is the v2 design target.
