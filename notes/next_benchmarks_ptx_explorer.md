# PTX Benchmark Explorer — Next Benchmarks Plan

**Date**: 2026-03-16
**Sources**: kernel.cuh, config.h, gemm.cuh, intrinsics.cuh (sm100/sm90/sm80), dual_tma_stream ver1, tma_gather4 ver6

---

## Critical Finding: Dual TMA Stream ver1 Analysis

The dual_tma_stream ver1 results reveal a key design constraint:

| Mode | Pattern | ckv ns/gather4 | kpe ns/gather4 |
|---|---|---|---|
| ckv_only | sequential (HBM-cold) | 548.2 | — |
| kpe_only | sequential | — | 466.0 |
| **both** | **sequential** | **548.4** | **501.6** |
| ckv_only | random (L2-warm) | 575.9 | — |
| kpe_only | random | — | 201.3 |
| **both** | **random** | **595.9** | **594.8** |

**Key finding**: TMA parallelizes HBM-bound streams but **serializes L2-resident streams**. When
both streams are L2-warm, kpe degrades from 201 ns → 595 ns (matching ckv exactly). The hardware
queues both through the same L2 bottleneck. The competition kernel's operating regime (KV in L2)
is precisely where serialization is worst.

**Design implication**: The competition kernel should **stagger kpe issue after ckv** (100-200 ns
delay) to allow ckv to clear the L2 bottleneck before kpe enters. `dual_tma_stream ver2` should
sweep this delay.

---

## Priority 1: `microbenchmarks/utcmma/` — UTCMMA + UTCCP

**Priority**: CRITICAL. Zero measurements exist for either GEMM.

**Key question**: Does QK GEMM (36 K-tiles × ~11 cycles = ~189 ns) fit inside TMA fetch time
(548 ns cold, 595 ns kpe-concurrent warm)? If GEMM < TMA → double-buffering works. If GEMM >
TMA → need more pipeline stages or concurrent warpgroups.

**Experiments**:
1. `utcmma_latency_ws_ts_m64n128` — single K-tile latency at M=64, N=128, .ws. Expected ~11 cycles.
2. `utcmma_k_depth_ws_ts` — K-depth 1..36 at M=64, N=128. Tests whether unrolling creates ILP.
3. `utcmma_throughput_ws_ss_m64n256` — SV tile, 4 K-tiles at M=64, N=256.
4. `utccp_latency` — 16 back-to-back UTCCP (`tcgen05.cp.cta_group::1.128dp256bit`) calls for Q
   loading. From `kernel.cuh:482`. Unknown cost on mbarrier critical path.
5. `utcmma_swizzle_b` — SW64 vs SW128 at M=64, N=128. Bank conflict delta.

**Implementation notes**:
- Compile with `-gencode arch=compute_100a,code=sm_100a` (the `a` suffix required for `cta_group::1`)
- TMEM allocation: `cute::TMEM::Allocator1Sm().allocate(512, addr)` then `release_allocation_lock()`
- `warpgroup_reg_dealloc<72>()` for MMA warpgroup (matches FlashMLA budget)
- Use `%%clock64` with ILP=1 for latency, ILP=4+ for throughput
- Output CSV: `experiment, M, N, k_tiles, mode, swizzle, cycles_per_ktile, ns_per_ktile, effective_tflops`
- Reference: `references/NVIDIA-Hopper-Benchmark/TensorCores/mma/m16n8k16/run_m16n8k16_single_SM.cu`

---

## Priority 2: `microbenchmarks/tmem_bandwidth/` — TMEM ld/st + tcgen05 fence overhead

**Priority**: HIGH. Effort: 2/5.

**Why**: 10 TMEM ld/fence roundtrips per block × 32 blocks = 320 per token. At 50 cycles/fence:
16,000 cycles = 7.6 µs = 20-40% of estimated decode time. In WG0 (softmax warpgroup), named
barriers synchronize WG0 with WG1, so excessive fence overhead stalls the whole pipeline.

The O-rescale pattern from `kernel.cuh:278`:
```
tmem_ld_32dp32bNx<64> → fence_view_async_tmem_load → tcgen05_before_thread_sync
→ NamedBarrier(128,wg0_sync) → tcgen05_after_thread_sync → FMA×32 → tmem_st_32dp32bNx<64>
```
Occurs 4× per block × 32 blocks = 128 full fence roundtrips per token.

**Experiments**:
1. `tmem_ld_throughput_N` — N ∈ {1,2,4,8,16,32,64,128}, back-to-back, no fence.
2. `tmem_st_throughput_N` — Same for stores. Expected ~2× slower than reads.
3. `fence_sequence_cost` — Isolated cost of each fence component separately, then combined.
4. `tmem_ld_st_realistic_loop` — Exact O-rescale pattern (ld N=64 → FMA×32 → st N=64 × 4 iters
   with fences). Total cycles for the hot loop.

---

## Priority 3: `microbenchmarks/mbarrier_overhead/` — Barrier roundtrip latency

**Priority**: HIGH. Effort: 1/5. Easiest to build, potentially high impact.

**Why**: ~12 barrier operations per block × 32 blocks = 384 roundtrips per token. At 50 cycles
each: 9.1 µs = ~30% of total. Named barriers (`NamedBarrier::arrive_and_wait(128, wg0_sync)`)
appear 3+ times in WG0's inner loop and are fully serializing for 128 threads.

**Experiments**:
1. `mbarrier_single_roundtrip` — `init(1) → expect_tx(0) → arrive → try_wait.parity`. Minimum roundtrip.
2. `mbarrier_arrive_count_sweep` — init(N) for N ∈ {1, 32, 64, 128, 384}. All N threads arrive;
   measure latency from last arrive to try_wait true. Does latency scale with arrive count?
3. `mbarrier_expect_tx_size_sweep` — expect_tx(B) for B ∈ {0, 512B, 4096B, 65536B}.
4. `mbarrier_chain_18` — 18 barriers in series. Total vs N=1 baseline.
5. `named_barrier_vs_mbarrier` — NamedBarrier::arrive_and_wait(128, id) cycles vs mbarrier roundtrip.

**Implementation note**: Reuse mbarrier PTX wrappers from
`microbenchmarks/dual_tma_stream/dual_tma_kernels.cuh:13-55` — they are already standalone.

---

## Priority 4: `microbenchmarks/tma_utcmma_overlap/` — Pipeline overlap efficiency

**Priority**: HIGH. Effort: 4/5.

**Why**: Double-buffer design assumes TMA and UTCMMA run in parallel. Unverified. If UTCMMA
blocks TMA issue (shared SM resource), pipeline needs N=3 stages.

**Experiments**:
1. `tma_only_baseline` — 32 sequential gather4 calls (ckv), measure total cycles.
2. `mma_only_baseline` — 32 UTCMMA QK GEMM calls (K=36), measure total cycles.
3. `tma_during_mma` — Warp5 issues TMA while warp4 runs MMA concurrently. Per-warp `%%clock64`
   measures each independently. Interference = (concurrent_time / solo_time − 1).
4. `pipeline_depth_sweep` — N=1..4 double-buffer stages for 32 blocks end-to-end. Expected: N=2
   optimal if GEMM < TMA; N=3 optimal if GEMM ≈ TMA.

---

## Priority 5: `microbenchmarks/ldg256_hints/` — LDG.256 cache hint matrix

**Priority**: MEDIUM. Effort: 2/5.

**Why**: Warp7 loads 64 int32 indices per block via `__ldg`. FlashMLA uses
`.nc.L1::no_allocate.L2::evict_normal.L2::256B` for metadata. For the competition's index pattern
(8KB total, sequential scan, discarded after one block), `.nc.L1::evict_first` may be better —
indices have no inter-block reuse.

**Sweep**: NC × {evict_first, evict_normal, evict_last, no_allocate} × prefetch {64B, 128B, 256B}
at working sets {8KB, 256KB, 8MB}. Plus `__ldg` baseline.

---

## Improvements to Existing Benchmarks

### dual_tma_stream ver2 (immediate)
1. kpe delay sweep: issue kpe N ns after ckv (N=0,50,100,200,400). Find minimum delay to avoid
   L2-warm serialization. Key metric: kpe_ns in BOTH mode → should recover toward kpe_only baseline.
2. kpe EVICT_FIRST in concurrent mode: kpe data has lower L2 reuse than ckv.
3. 32-CTA competition-scale run.

### tma_gather4 ver7
1. kpe_bf16 EVICT_FIRST in L2 sweep (add to existing hint experiment).
2. `createpolicy` fraction sweep at 1024MB working set.

---

## Competition Pipeline Timing Model (Updated)

| Stage | Cycles (est.) | ns (est.) | Data source |
|---|---|---|---|
| ckv TMA per block (16 gather4, cold) | 1,150+ | 548+ | ver5 measured |
| kpe TMA per block (concurrent, L2-warm) | 1,250 | 595 | dual_tma ver1 |
| **TMA bottleneck per block** | **~1,250** | **~595** | kpe-limited when L2-warm |
| UTCMMA QK (36 K-tiles × 11 cyc) | 396 | 189 | arxiv:2512.02189 (unverified) |
| UTCMMA SV (4 K-tiles × 11 cyc) | 44 | 21 | arxiv:2512.02189 (unverified) |
| Softmax + fence sequences | 400? | 190? | unknown |
| mbarrier overhead per block | 600? | 285? | unknown |
| **Compute bottleneck per block** | **~1,040?** | **~495?** | mostly unknown |

If TMA (595 ns) > compute (495 ns): double-buffering (N=2) → `max(595, 495) × 32 = 19 µs/token`.
If fences/barriers dominate: compute may exceed TMA → deeper pipeline or concurrent warpgroups needed.

**Single most important experiment**: Priority 1 (UTCMMA latency) + Priority 3 (mbarrier overhead)
will determine whether the pipeline is TMA-bound or compute-bound. Every other design decision
follows from that answer.

---

## Newly Discovered PTX Variants

**UTCCP** (`tcgen05.cp.cta_group::1.128dp256bit`): Used to copy Q from smem to TMEM. Source:
`kernel.cuh:482`. 16 calls for 512-dim Q. Blocks MMA warp via `bar_q_utccp`. Latency unknown.
Competition relevance: HIGH (one-time per token but on critical path).

**Fractional cache policy** (`createpolicy.fractional.L2::evict_last.L2::evict_first.b64 %0, %fraction`):
Source: `sm80/intrinsics.cuh:40-57`. Allows pinning only `fraction` of cache lines with evict_last.
Relevant for 32 concurrent split-KV CTAs that might thrash each other's L2 allocations.

**TMA bulk reduce** (`cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`):
Source: `sm90/intrinsics.cuh:43`. Could eliminate the separate combine kernel for split-KV
accumulation. Throughput at 32 KB output size unmeasured.
