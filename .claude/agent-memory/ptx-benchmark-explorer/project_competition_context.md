---
name: competition kernel context
description: Key facts about the competition attention and indexer kernels, hardware target, benchmark status, and all remaining gaps
type: project
---

Competition target: B200 sm100a, CUDA 12.9+. Two kernels to optimize:

1. Attention kernel: sparse decode attention, num_tokens=1-2, topk=2048, ckv_cache [8462,64,512] BF16 + kpe_cache [8462,64,64] BF16, sparse_indices [num_tokens,2048], output [num_tokens,16,512] BF16, LSE in log2 base. Memory-bound: 2.36 MB KV per query token, scattered across ~8462 pages.

2. Indexer kernel: FP8 input, 64 heads, dim=128, topk=2048, page_size=64.

**Why:** Competition requires lowest possible latency for the decode path. All benchmark findings directly inform kernel design choices (cache policies, TMA layout, pipeline depth, GEMM tile sizes).

**How to apply:** Every benchmark suggestion must tie back to one of these two kernels and explain how it changes a concrete design decision.

## Reference kernel critical details (from config.h + kernel.cuh)

- B_H = 64, B_TOPK = 64, NUM_BUFS = 2, NUM_THREADS = 128*3 = 384
- TiledMMA_P: SM100_MMA_F16BF16_WS_TS_NOELECT<bf16,bf16,float, M=64, N=128, K_major, K_major>
  (N=128 because dual GEMM over B_H*2=128; handles QK for both head halves in one MMA call)
- TiledMMA_O: SM100_MMA_F16BF16_WS_SS_NOELECT<bf16,bf16,float, M=64, N=256, K_major, MN_major>
  (N=256 = D_V=512/2 in MN-major B, used for SV GEMM)
- K-depth: QK GEMM = 36 K-tiles (576d / 16); SV GEMM = 4 K-tiles (B_TOPK=64 / 16)
- TMEM columns: O=0..255 (256 cols), Q=256..399, P=400..464 (total 512 cols allocated / SM)
- Warpgroup structure: WG0=softmax/scale/store (128T), WG1=MMA+TMA (128T, warpgroup_reg_dealloc<72>), WG2=dequant (128T — NOT needed for competition BF16 kernel)
- Warp roles in WG1: warp4=MMA+UTCCP, warp5=nope TMA (ckv), warp6=rope TMA (kpe), warp7=index transform (loads sparse_indices via __ldg, generates TMA coordinates)
- TMA cache hint in FlashMLA kernel: EVICT_LAST for both nope and rope KV loads
- Smem layout: Q in SW128 (512d), nope in SW128, rope in SW64 (V32) or SW128 (MODEL1)
- LDG.256 for scheduling metadata: .nc, L1::no_allocate, L2::evict_normal, L2::256B
- UTCCP: 16 calls of SM100_UTCCP_128dp256bit_1cta for Q load (128dp×256bit per call, 4 tiles × 4 subtiles)
- Named barriers in WG0: NamedBarrier::arrive_and_wait(128, wg0_sync) appears 3+ times per block
- O-rescale pattern: tmem_ld_32dp32bNx<64> → fence_view_async_tmem_load → tcgen05_before_thread_sync → __syncthreads → tcgen05_after_thread_sync → FMA → tmem_st_32dp32bNx<64> × 4 iterations (for D_V=512 in B_EPI=64 chunks)
- SM90_BULK_COPY_S2G used for split-KV output accumulation (not TMA reduce; separate combine kernel)

## Benchmark status (2026-03-16)

### Completed benchmarks
- tma_gather4 (ver6, 8 experiments): HBM-cold latency 546 ns; EVICT_FIRST best at >64MB; gather4 BW linear to 296 CTAs; L2 promotion 64B vs 128B identical
- dual_tma_stream (ver1): ckv+kpe concurrent interference measured

### dual_tma_stream ver1 KEY FINDINGS
- HBM-cold (sequential): ckv unaffected (+0.04%); kpe degrades from 466ns to 502ns (+7.6%)
- L2-warm (random): kpe catastrophically degrades from 201ns to 595ns (+196%), matching ckv latency
- Interpretation: TMA hardware parallelizes HBM-bound gather4s but SERIALIZES L2-resident gather4s from concurrent streams
- Design implication: competition kernel should stage kpe issue N nanoseconds AFTER ckv (not simultaneously), especially for L2-warm case. Delay sweep needed (ver2).

### Remaining knowledge gaps (in priority order)

**Gap 1: UTCMMA benchmark (CRITICAL — zero data)**
- Competition tiles: QK (M=64,N=128,K=36×16) WS_TS and SV (M=64,N=256,K=4×16) WS_SS
- Need: cycles per K-tile, K-depth scaling, does CUTE_UNROLL create ILP overlap?
- If 36×11=396 cycles (189 ns) < TMA cold 1150 cycles → TMA-bound (expected)
- If GEMM > TMA → pipeline redesign needed
- Also: UTCCP latency for Q loading (16 calls)

**Gap 2: TMEM load/store bandwidth + fence overhead (HIGH)**
- O-rescale: 10 ld/st+fence roundtrips per block × 32 blocks = 320 fence sequences per token
- If fence sequence = 50 cycles: 16,000 cycles = 7.6 µs = 25% of estimated decode time

**Gap 3: mbarrier overhead (HIGH)**
- 18 mbarriers, ~12 barrier ops per block, 384+ roundtrips per token
- At 50 cycles each: 19,200 cycles = 9.1 µs = 30% of estimated decode time

**Gap 4: TMA + UTCMMA overlap (HIGH)**
- Does UTCMMA block TMA issue hardware? If yes, need N≥3 pipeline depth
- Current assumption (double-buffer = 2 stages sufficient) is unverified

**Gap 5: LDG.256 cache hint matrix (MEDIUM)**
- Warp7 index streaming: .nc × L1-hint × L2-prefetch combinations unmeasured
- Current FlashMLA setting: .nc.L1::no_allocate.L2::evict_normal.L2::256B
- Competition may benefit from .L1::evict_first since indices used once per block then discarded

**Gap 6: createpolicy fraction parameter (MEDIUM)**
- create_fraction_based_cache_policy<EVICT_LAST, EVICT_FIRST>(fraction) unmeasured
- At topk=2048 with 32 concurrent CTAs, fractional eviction might reduce inter-CTA L2 thrashing

**Gap 7: cp.reduce.async.bulk.add.f32 for split-KV combine (MEDIUM)**
- Could eliminate separate combine kernel if throughput sufficient for 4.2 MB float32 accumulation
- Current design uses SM90_BULK_COPY_S2G + host-launched combine kernel

## Pipeline timing model (updated 2026-03-16)

| Stage | Est. ns | Source |
|---|---|---|
| Per-block ckv TMA (16 gather4, cold) | ~548 ns | ver5 latency |
| Per-block kpe TMA (16 gather4, concurrent+warm) | ~595 ns | dual_tma ver1 |
| UTCMMA QK (36 K-tiles × ~11 cycles) | ~189 ns | arxiv (UNVERIFIED) |
| Softmax + fence sequences | ~100-200 ns | UNKNOWN |
| mbarrier overhead per block | ~100-500 ns | UNKNOWN |
| UTCMMA SV (4 K-tiles) | ~21 ns | arxiv (UNVERIFIED) |
| **TMA-bound block time (L2-warm, concurrent)** | **~595 ns** | dual_tma ver1 |
| **Estimated per-token with N=2 pipeline** | **~10-50 µs** | model (high uncertainty) |

Key: kpe is the new bottleneck in the L2-warm case (not ckv). Pipeline should prioritize ckv and delay kpe issue.
