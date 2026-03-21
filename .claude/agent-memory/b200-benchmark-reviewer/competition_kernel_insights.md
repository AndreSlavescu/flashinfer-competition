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

## Cache Hints (updated ver6)
- ckv_cache: `evict_last` (keeps KV in L2 during computation)
- kpe_cache: `evict_first` (+5% HBM throughput, frees L2 pressure — kpe gets evicted by ckv anyway)

## Timing for Competition
- Per query token, at 76 CTAs (1 CTA/SM, 76 SMs of the 148):
  - ckv TMA: 2048 tokens × 1024B = 2MB; at 409 GB/s ≈ 4.9 µs (HBM-bound) or 2.2 µs (L2-resident)
  - kpe TMA: 2048 tokens × 128B = 256KB; at 55 GB/s ≈ 4.7 µs (HBM-bound) or ~0.3 µs (L2-resident)
  - kpe is the bottleneck if HBM-bound; ckv is the bottleneck if L2-resident
- True HBM-miss TMA latency for a single gather4 is NOT well measured — both L2-hot and cold cases measure ~197 ns which is TMA pipeline overhead, not memory latency.

## Timing Model History (ver5/ver6 — SUPERSEDED, kept for reference)

- ver5: estimated TMA-bound 6-7× (L2-hot) to 50× (cold) — UTCMMA was unknown
- ver6: calculated per-block TMA = 16 × 595 ns (kpe concurrent) = 9,520 ns sequential; "32 blocks serialized in 1 CTA" was wrong for split-KV (32 SMs run in parallel)
- Both superseded by ver3 timing model below which adds measured UTCMMA values

## Cache Hints (updated)
- ckv_cache: `evict_last` (keeps KV in L2 during computation)
- kpe_cache: `evict_first` (+5% HBM throughput, frees L2 pressure — kpe gets evicted by ckv anyway)

## Dual-Stream TMA Finding (dual_tma_stream ver1)
- HBM-cold: ckv+kpe streams run in parallel (0% interference). Safe to issue both simultaneously.
- L2-warm: kpe serializes to HBM latency when ckv evicts its data. In production, kpe will run at ~595 ns per gather4, not 201 ns.
- Consider staggering kpe issue ~100-200 ns after ckv to reduce L2 contention (ver2 will test this).

## UTCMMA Performance (utcmma-ver3.csv, 2026-03-19 — MEASURED)

- QK GEMM (M=64 N=128 k_depth=32, RAW chain): **1,719 cycles = 928 ns** — confirmed
- SV GEMM single tile (M=64 N=256 k_depth=4, RAW chain): **320 cycles = 173 ns**; competition SV needs 2 tiles (N=256×2) = **640 cy = 346 ns**
- UTCMMA structural initiation interval: **~54 cycles/K-tile regardless of N_ACC rotation** (N_ACC=1..4 all identical). Multi-accumulator pipelining does NOT work for N_ACC ≤ 4. This means the "2-acc → 190 ns" optimization hypothesis is INVALID.
- Non-WS vs WS: WS is 17–19% faster for N=128 (64 cy non-WS vs 54 cy WS). Always use .ws variants.
- Dual-GEMM smem layout (Shape<128, K_TOTAL>): zero performance penalty vs canonical layout.
- Swizzle sensitivity: zero (INTER/SW32/SW64/SW128 all identical).

## TMEM Operation Costs (ver3 MEASURED)

- TMEM load (stable TMEM, no pending stores): ~1.79 cycles (clock-read floor; effectively free)
- TMEM store (N=64 + fence): **49.8 cycles** (linear: 12 + 0.59×N)
- Warp-collective TMEM fence pair (ld+st fence with nothing in flight): **12.83 cycles**
- Per-thread tcgen05 fence pair (before/after_thread_sync): **2.83 cycles**
- Full O-rescale 1 chunk (N=64 ld+mul+st): **168.1 cycles = 85.6 ns**
- Full O-rescale 4 chunks (D=512 width): **622.1 cycles = 316.6 ns**
- NOTE: O-rescale is only needed in online softmax designs (one SM iterating over multiple KV chunks). In split-KV with one B_TOPK=64 chunk per SM, no O-rescale is needed.

## Synchronization Costs (ver3 MEASURED)

- named_bar.sync(128): **20.67 cycles = 10.5 ns**
- named_bar.sync(32): **14.65 cycles = 7.5 ns**
- mbarrier arrive+wait (expect_tx=0 baseline): **88.51 cycles = 45.0 ns**
- MMA + umma_arrive_noelect + bar.wait: **173.03 cycles = 93.4 ns**
- tcgen05.commit overhead ≈ 30 cycles above MMA+barrier
- NOTE: mbarrier waits in a pipelined kernel synchronize TMA completion — they are part of TMA time, not additive overhead. Per-SM mbarrier count = ~32 (one per gather4: 16 ckv + 16 kpe).

## Competition Kernel Timing Model (ver3 corrected, split-KV, per-SM, 2026-03-19)

**TMA per SM/block** (16 gather4 calls for ckv + 16 for kpe, running in parallel):
- All TMA latency values (546/595 ns) are PER-GATHER4-CALL
- HBM-cold sequential: 16 × 595 ns = ~9,520 ns per stream
- With N=2 pipeline + DIST=2 prefetch: ~16 × 324 = ~5,184 ns per stream (estimated)
- ckv ∥ kpe (zero HBM interference verified) → per-block TMA = max(ckv, kpe)
- **Per-block TMA: ~5,000-9,500 ns**

**GEMM per SM/block** (serialized RAW chain, measured):
- QK ckv (M=64 N=64 K_DEPTH=32): ~1,719 cy = ~928 ns
- QK kpe (M=64 N=64 K_DEPTH=4): ~218 cy = ~118 ns
- SV (2 tiles × M=64 N=256 K_DEPTH=4): 2 × 320 = ~640 cy = ~346 ns
- **Per-block GEMM total: ~2,577 cy = ~1,392 ns**

**Pipeline structure**: With N=64 WS tiles (split-KV, 1 block/SM), GEMM needs all 64 KV tokens loaded before processing → TMA and GEMM are sequential, not overlapped.

**Per-block total = TMA + GEMM = ~6,400-10,900 ns. Kernel is TMA-BOUND by ~3.6-6.8×.**

TMA pipelining efficiency at competition scale (16 consecutive HBM-cold gather4s) is the key unmeasured parameter.

## What's Still Unknown (requires follow-up benchmarks)

- **TMA pipelining at competition scale**: 16 consecutive HBM-cold gather4 calls — what is the achieved per-call throughput with N=2 pipelining? This determines whether per-block TMA is ~5,000 or ~9,500 ns. HIGHEST PRIORITY.
- **UTCMMA N_ACC ≥ 5**: Does an 8-accumulator rotation approach 11 cycles/K-tile? M=64 N=64 with N_ACC=8 uses 256 TMEM cols. If achieved, GEMM drops to ~191 ns, making TMA dominance even more extreme.
- **TMA + UTCMMA concurrent execution**: If the kernel can issue TMA for next block while processing current block (requires multi-block per SM or online softmax with smaller N tiles), TMA/GEMM overlap becomes possible.
- **Why 11 cycles in arxiv:2512.02189**: Measurement methodology unclear. Possibly cta_group::2, independent tiles with no TMEM RAW, or multi-warp ILP.
- **INT64 smem layout compatibility with tcgen05.mma**: Still unresolved.
- **kpe stagger delay**: Still unresolved.

**Why:** These derive from concrete measurements on real B200 hardware (ver3 CSV, 2026-03-19).
**How to apply:** Use these when making design decisions for tile sizes, pipeline stages, TMA configs in the competition kernel. TMA optimization (pipelining, prefetch, cache hints) is the highest-priority area.
