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

## Updated Timing Model (ver5, incorporating cold latency)

Per query token, 76 CTAs, B_TOPK=64:
- L2-hot: ckv 16×197ns = 3.15µs, kpe 16×182ns = 2.91µs per block
- HBM-cold: ckv 16×546ns = 8.74µs, kpe 16×492ns = 7.87µs per block
- If ckv+kpe issue simultaneously: max(ckv, kpe) = 3.15µs hot, 8.74µs cold per block
- 32 blocks / 76 SMs: 1.33µs hot or 3.68µs cold total across the GPU per query token
- UTCMMA compute: ~0.5µs total per query token (38 GFLOPS / 990 TFLOPS = 38µs kernel-wide / 76 SMs)
- KERNEL IS TMA-BOUND BY 6-7× (hot) TO 50× (cold). Optimize TMA first.

## Updated Timing Model (ver6 + dual_tma_stream ver1, 2026-03-16)

Per block (B_TOPK=64), single CTA:
- ckv TMA: 16 gather4 × 540 ns HBM-cold = 8,640 ns (no kpe interference when HBM-bound)
- kpe TMA in BOTH mode: 16 × 595 ns (L2-evicted by ckv) = 9,520 ns — kpe becomes the bottleneck
- With DIST=2 prefetch: 16 × 324 ns = 5,184 ns ckv; kpe prefetch also effective
- 32 blocks / 32 CTAs: TMA bottleneck ≈ 9,520 ns per block × (32 blocks serialized in 1 CTA) = 305 µs / token if no pipeline

With N=2 double-buffering + DIST=2 prefetch: bottleneck ≈ max(TMA, compute) per block.
UTCMMA compute still unknown — measuring this is HIGHEST PRIORITY.

## Cache Hints (updated)
- ckv_cache: `evict_last` (keeps KV in L2 during computation)
- kpe_cache: `evict_first` (+5% HBM throughput, frees L2 pressure — kpe gets evicted by ckv anyway)

## Dual-Stream TMA Finding (dual_tma_stream ver1)
- HBM-cold: ckv+kpe streams run in parallel (0% interference). Safe to issue both simultaneously.
- L2-warm: kpe serializes to HBM latency when ckv evicts its data. In production, kpe will run at ~595 ns per gather4, not 201 ns.
- Consider staggering kpe issue ~100-200 ns after ckv to reduce L2 contention (ver2 will test this).

## What's Still Unknown (requires follow-up benchmarks)
- **UTCMMA latency at competition tile sizes** (M=64, N=128 QK; M=64, N=256 SV): HIGHEST PRIORITY. Is GEMM (est. 189 ns) < TMA (595 ns)? Determines if pipeline is TMA-bound or compute-bound.
- **mbarrier chain overhead**: ~384 roundtrips per token at ~50 cycles each = 9 µs? Could flip kernel from TMA-bound to compute-bound.
- **tcgen05 fence sequence overhead**: 128 roundtrips per token in O-rescale loop.
- **TMA+UTCMMA overlap**: Does issuing TMA while UTCMMA runs preserve full throughput for both?
- **INT64 smem layout compatibility with tcgen05.mma**: Can tcgen05.mma consume SWIZZLE_NONE INT64-typed smem as BF16 without re-swizzle? (HIGH PRIORITY)
- **kpe stagger delay**: Optimal delay between ckv and kpe issue to avoid L2 serialization.

**Why:** These derive from concrete measurements on real B200 hardware (ver5 CSV), not estimates or paper extrapolations.
**How to apply:** Use these when making design decisions for tile sizes, pipeline stages, TMA configs in the competition kernel.
