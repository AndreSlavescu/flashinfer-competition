---
name: Competition Kernel Design Insights from TMA Gather4 Benchmarking
description: Specific design recommendations for the DSA decode kernel derived from tma_gather4 benchmark results
type: project
---

From tma_gather4 ver4 results on B200 Modal hardware:

## TMA Configuration
- Use INT64-packed config for ckv_cache (dim0=128, box=128, 4096B/gather4, 1 col_step, SWIZZLE_NONE) — 22% faster than native BF16 box=64 (512B/gather4) for HBM-bound access
- CAVEAT: INT64 config has SWIZZLE_NONE which may require layout conversion for tcgen05.mma. Benchmark the conversion overhead before committing.
- For kpe_cache (64d BF16): use box=64, SWIZZLE_NONE, 512B/gather4. Accept ~55 GB/s per 76 CTAs — this is the per-SM TMA bandwidth ceiling for kpe.

## Pipeline Depth
- N=2 TMA pipeline stages is optimal. N=1→N=2 gives 1.31× speedup; N≥4 adds <3% more.
- In multi-warp production kernel, 2-stage double-buffering of smem (alternating ping/pong buffers with alternating mbarriers) is the recommended approach.

## SM Allocation and Working Set
- B200 L2 = ~64MB. Per SM with B_TOPK=64: 64 tokens × 1024B = 64KB ckv + 64 × 128B = 8KB kpe = 72KB per SM.
- 76 SMs × 72KB = 5.5MB total working set — fits in L2 with room to spare.
- L2-resident TMA throughput: ~900 GB/s aggregate at 76 CTAs (vs 409 GB/s HBM-bound).
- This suggests the competition kernel can be largely L2-resident if SM partitioning assigns non-overlapping index ranges.

## Index Sorting
- Sorting sparse_indices before issuing gather4 creates page-level locality.
- L2 reuse from sorted indices can deliver ~900 GB/s vs ~409 GB/s for random access — a 2.2× speedup for HBM-bound cases.
- Implement sorting as a pre-pass over sparse_indices within each B_TOPK=64 block before the TMA gather loop.

## Cache Hints
- Use evict_last for both ckv and kpe. Never use evict_first for data you want to reuse.
- At HBM-bound scale (256MB+), all hints give similar performance — but evict_last is best for L2-resident cases.

## Timing for Competition
- Per query token, at 76 CTAs (1 CTA/SM, 76 SMs of the 148):
  - ckv TMA: 2048 tokens × 1024B = 2MB; at 409 GB/s ≈ 4.9 µs (HBM-bound) or 2.2 µs (L2-resident)
  - kpe TMA: 2048 tokens × 128B = 256KB; at 55 GB/s ≈ 4.7 µs (HBM-bound) or ~0.3 µs (L2-resident)
  - kpe is the bottleneck if HBM-bound; ckv is the bottleneck if L2-resident
- True HBM-miss TMA latency for a single gather4 is NOT well measured — both L2-hot and cold cases measure ~197 ns which is TMA pipeline overhead, not memory latency.

## What's Still Unknown (requires follow-up benchmarks)
- TMA bandwidth while UTCMMA runs simultaneously (overlap efficiency)
- Dual-stream TMA (ckv + kpe simultaneously) interaction at L2/HBM
- True HBM-miss TMA latency (single access to cold cache line)
- INT64 smem layout compatibility with tcgen05.mma without reswizzle

**Why:** These derive from concrete measurements on real B200 hardware, not estimates or paper extrapolations.
**How to apply:** Use these when making design decisions for tile sizes, pipeline stages, TMA configs in the competition kernel.
