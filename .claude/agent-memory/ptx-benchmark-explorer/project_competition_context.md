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

## Benchmark status (2026-03-17)

### Completed benchmarks
- tma_gather4 (ver6, 8 experiments): HBM-cold latency 546 ns; EVICT_FIRST best at >64MB; gather4 BW linear to 296 CTAs; L2 promotion 64B vs 128B identical
- dual_tma_stream (ver1): ckv+kpe concurrent interference measured
- utcmma (ver1, 5 experiments): UTCMMA latency, K-depth scaling, UTCCP, swizzle — full serialized RAW dependency chain

### dual_tma_stream ver1 KEY FINDINGS
- HBM-cold (sequential): ckv unaffected (+0.04%); kpe degrades from 466ns to 502ns (+7.6%)
- L2-warm (random): kpe catastrophically degrades from 201ns to 595ns (+196%), matching ckv latency
- Interpretation: TMA hardware parallelizes HBM-bound gather4s but SERIALIZES L2-resident gather4s from concurrent streams
- Design implication: competition kernel should stage kpe issue N nanoseconds AFTER ckv (not simultaneously), especially for L2-warm case. Delay sweep needed (ver2).

### utcmma ver1 KEY FINDINGS (2026-03-17)
- Serialized RAW latency: 54.5 cy/tile for M64N64 and M64N128 (identical!); 76 cy for M64N256; 80 cy for SS M64N256
- K-depth scaling is PERFECTLY LINEAR at ~53.7-54.6 cy/tile from k_depth=1..32. No pipelining in current loop structure.
- Competition QK GEMM (M=64, N=128, k_depth=32): 1719 cy = 932 ns (MEASURED)
- Competition SV GEMM (M=64, N=256, k_depth=4, SS): 320 cy = 173 ns (MEASURED)
- UTCCP (16 copies, Q load): 1192 cy = 607 ns (MEASURED) — nearly as expensive as QK GEMM!
- Swizzle has ZERO effect on MMA latency (all 4 modes identical at 218 cy)
- CRITICAL GAP: paper claims ~11 cy/tile throughput with independent accumulators; we measured 54 cy serialized. True pipelined throughput unknown.
- If 2-acc throughput = 11 cy: QK GEMM = 352 cy = 190 ns → kernel is TMA-BOUND (TMA=595 ns > GEMM=190 ns)
- If 2-acc throughput = 54 cy: kernel is COMPUTE-BOUND (GEMM=932 ns > TMA=595 ns)
- This determines the entire pipeline design strategy.

### utcmma ver1 TMEM COLUMN BUDGET
- TiledMMA_P: M=64, N=B_TOPK*2=128 (dual-GEMM over 128-wide smem)
- TiledMMA_O: M=64, N=256, SS, MN-major B
- TMEM cols: O=0..255, Q=256..399, P=400..463
- For 2-accumulator throughput experiment: use N=64 (128 cols each) → 4 accumulators fit in 512 cols

### Remaining knowledge gaps (in priority order for ver2)

**Gap 1: UTCMMA independent-accumulator throughput (CRITICAL)**
- Use 2 TMEM regions (C0=cols 0..127, C1=cols 128..255 for M64N64), issue alternating MMA
- Sweep N_acc={1,2,4,8} → find plateau → that's true throughput
- If plateau=11 cy: design TMA-bound kernel. If plateau=54 cy: design compute-bound kernel.

**Gap 2: TMEM ld/st + fence overhead (HIGH)**
- O-rescale per chunk: tmem_ld<64> → fence_view_async_tmem_load → float2_mul × 32 → tmem_st<64> → fence_view_async_tmem_store
- 4 chunks × ~28 rescale events per token = 112 roundtrips. Cost unknown.

**Gap 3: mbarrier roundtrip overhead (HIGH)**
- NamedBarrier::arrive_and_wait(128, wg0_sync) appears 3× per block iteration in WG0
- 32 blocks × 3 × latency per token = dominant if latency > 30 cy

**Gap 4: tcgen05.commit latency (HIGH)**
- Called after every QK and SV GEMM: 64 total calls per token
- isolate by: time(MMA + commit + wait) - time(MMA only)

**Gap 5: TMA + UTCMMA overlap — pipeline depth decision (HIGH)**
- Does issuing TMA gather4 while UTCMMA executes serialize on any shared unit?
- If yes: N≥3 pipeline depth required. If no: double-buffer suffices.

**Gap 6: LDG.256 cache hint matrix (MEDIUM)** — unchanged
**Gap 7: createpolicy fraction (MEDIUM)** — unchanged
**Gap 8: cp.reduce.async.bulk combine (MEDIUM)** — unchanged

## Pipeline timing model (updated 2026-03-17)

| Stage | Est. ns | Source |
|---|---|---|
| UTCCP for Q (16 copies) | 607 ns | ver1 MEASURED |
| QK GEMM (k_depth=32, serialized) | 932 ns | ver1 MEASURED |
| QK GEMM (k_depth=32, 2-acc) | ~190 ns | arxiv, UNVERIFIED |
| SV GEMM (k_depth=4, SS) | 173 ns | ver1 MEASURED |
| tcgen05.commit × 2 per block | UNKNOWN | — |
| O-rescale (4 chunks) | UNKNOWN | — |
| mbarrier roundtrip per iteration | UNKNOWN | — |
| TMA ckv per block (L2-cold) | 548 ns | tma ver5 |
| TMA kpe (L2-warm, concurrent) | 595 ns | dual_tma ver1 |

Key open question: true 2-acc UTCMMA throughput determines whether kernel is compute-bound or TMA-bound.
