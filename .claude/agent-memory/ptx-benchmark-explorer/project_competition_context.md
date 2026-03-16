---
name: competition kernel context
description: Key facts about the competition attention and indexer kernels, hardware target, and what's already benchmarked
type: project
---

Competition target: B200 sm100a, CUDA 12.9+. Two kernels to optimize:

1. Attention kernel: sparse decode attention, num_tokens=1-2, topk=2048, ckv_cache [8462,64,512] BF16 + kpe_cache [8462,64,64] BF16, sparse_indices [num_tokens,2048], output [num_tokens,16,512] BF16, LSE in log2 base. Memory-bound: 2.36 MB KV per query token, scattered across ~8462 pages.

2. Indexer kernel: FP8 input, 64 heads, dim=128, topk=2048, page_size=64.

**Why:** Competition requires lowest possible latency for the decode path. All benchmark findings directly inform kernel design choices (cache policies, TMA layout, pipeline depth, GEMM tile sizes).

**How to apply:** Every benchmark suggestion must tie back to one of these two kernels and explain how it changes a concrete design decision.

## Reference kernel critical details (from config.h + kernel.cuh)

- B_H = 64, B_TOPK = 64, NUM_BUFS = 2, NUM_THREADS = 128*3 = 384
- TiledMMA_P: SM100_MMA_F16BF16_WS_TS_NOELECT<bf16,bf16,float, B_H=64, B_TOPK*2=128, K_major, K_major>  (dual GEMM)
- TiledMMA_O: SM100_MMA_F16BF16_WS_SS_NOELECT<bf16,bf16,float, B_H=64, N=256, K_major, MN_major>
- TMEM columns: O=0..255 (256 cols), Q=256..399, P=400..464 (total 512 cols allocated)
- Warpgroup structure: WG0 = softmax/scale/store (128T), WG1 = MMA+TMA (128T, warpgroup_reg_dealloc<72>), WG2 = dequant (128T, not needed for competition BF16 kernel)
- Warp roles in WG1: warp4=MMA, warp5=nope TMA, warp6=rope TMA, warp7=index transform
- TMA cache hint in kernel: EVICT_LAST for both nope and rope KV loads
- Smem layout: Q in SW128, nope in SW128, rope in SW64 (V32) or SW128 (MODEL1)
- LDG.256 for scheduling metadata: .nc, L1::no_allocate, L2::evict_normal, L2::256B

## Current tma_gather4 benchmark coverage (8 experiments)

1. throughput_box_sweep: box_dim0 sweep, INT64 and BF16 dtypes
2. throughput_hints: EVICT_LAST/FIRST/NORMAL/NONE × 3 patterns × 4 working set sizes
3. throughput_patterns: 7 patterns × 3 working set sizes
4. throughput_swizzle: NONE/32B/64B/128B for BF16×64/128/256, INT64×128
5. throughput_l2 + throughput_l2_64b: working set sweep 1..2048 MB, L2_128B and L2_64B promotion
6. latency: sequential/random, L2-hot and HBM-cold variants
7. pipeline: N=1..32 outstanding gather4s, single CTA
8. throughput_saturation: 1..296 CTAs

**NOT covered (gaps):**
- Dual simultaneous TMA streams (ckv 512d + kpe 64d concurrently)
- TMA + UTCMMA overlap: TMA issuing while GEMM executes
- tma_gather4_prefetch (cp.async.bulk.prefetch.tensor.2d)
- cta_group::2 gather4 variant
- UTCMMA throughput/latency (tcgen05.mma) — no benchmark exists at all
- TMEM load/store bandwidth (tcgen05.ld/st)
- tcgen05 fence overhead (before/after_thread_sync)
- mbarrier overhead (init, expect_tx, arrive, try_wait.parity)
- LDG.256 cache hint matrix (.nc, L1 hints, L2 hints, prefetch sizes)
- STG.256 for output store
- cp.reduce.async.bulk (tma_bulk_reduce_add) for split-KV accumulation
- atomicadd_f32x4_with_policy for output accumulation
- SMEM swizzle layout impact on UTCMMA throughput (SW32/SW64/SW128/INTER)
- Occupancy sweep: how many CTAs/SM can coexist given 512-col TMEM allocation + 228KB smem
