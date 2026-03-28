# PTX Benchmark Gap Analysis — 2026-03-21

## Executive Summary

As of 2026-03-21, benchmarked instruction families:
- **tma_gather4** (ver8): latency, throughput, pipeline N=1–16, prefetch, cache hints, dual-stream
- **utcmma** (ver5): completion latency, K-depth scaling, WS vs non-WS, TMEM ld/st, fences, mbarrier, tcgen05.commit, UTCCP, named barriers, multi-accumulator (null result)

**Key state**: Kernel is TMA-BOUND by 2.04× (3,690 ns TMA vs 1,809 ns GEMM per block).

---

## Section 1: Newly Discovered Variants

### 1.1 `SM90_BULK_COPY_S2G` — split-KV output write path

**PTX**: `cp.async.bulk.global.shared::cta.bulk_group [dst], [smem], bytes`
**Source**: `csrc/kerutils/include/kerutils/device/sm90/intrinsics.cuh`; used in kernel.cuh:413 — each split-KV SM writes 16 rows × 2048 bytes = 32,768 bytes of f32 partial O to `o_accum`
**sm100a-exclusive**: NO (SM90+)
**Used by reference code**: YES — on critical path for every split-KV block
**Competition relevance**: **HIGH — ON CRITICAL PATH, NEVER BENCHMARKED.** Current timing model ignores the output write entirely. If it's async and overlaps next-token TMA, cost is zero. If not, it adds to serial per-block time.
**Benchmark parameter**: Sweep write size {512B, 2KB, 32KB, 128KB} with and without explicit wait. Compare async overlap vs sequential cost.

### 1.2 TMA + UTCMMA Genuine Concurrency

Not a new instruction but the highest-priority unmeasured behavioral question. Can block i+1 TMA overlap with block i GEMM? Determines whether inter-block pipelining reduces 5,499 ns → 3,690 ns (33% gain). See Priority 1 in Section 3.

### 1.3 `createpolicy.fractional` fraction < 1.0

**Source**: `csrc/kerutils/include/kerutils/device/sm80/intrinsics.cuh:42` `create_fraction_based_cache_policy<P,S>(float fraction)`
**Valid combos**: primary ∈ {evict_first, evict_normal, evict_last, evict_unchanged} × secondary ∈ {evict_first, evict_unchanged} × fraction ∈ [0, 1]
**Used by reference code**: YES at fraction=1.0 only
**Competition relevance**: MEDIUM. At fraction=0.5 for kpe (evict_first primary), 50% of kpe data evicts from L2 immediately, reducing competition with ckv. Could recover kpe concurrent degradation (201 ns → 595 ns in dual_tma ver1). **Fraction < 1.0 never tested.**

### 1.4 `tcgen05.mma.ws M=32` — raw PTX for better tensor core utilization

**Source**: PTX ISA §9.7.16.10.9 — M=32 valid for `.ws`; blocked by CuTe `mma_traits_sm100.hpp:502` static_assert
**Competition relevance**: **HIGH.** h_q=16 → M=64 uses only 25% of tensor core rows. M=32 → 50% utilization. TMEM freed (256 cols) enables dual-QK accumulator: QK_ckv (K=32) and QK_kpe (K=4) in separate TMEM regions running in parallel.
**TMEM budget for M=32**: P(M=32,N=64)=32 cols, O(M=32,N=256)=128 cols, Q(M=32)=144 cols, QK_kpe second acc=32 cols → 336 total (well under 512).
**Never benchmarked**: Requires raw inline PTX.

### 1.5 `ex2.approx.f32` vs polynomial emulation — softmax inner loop

**Source**: FA4 `flash_fwd_sm100.py` sets `enable_ex2_emu=True` for sm100 (B200); `softmax.py` `ex2_emulation_2()` uses 3-FMA polynomial instead of `ex2.approx`
**What it does**: FA4 deliberately avoids the hardware SFU `ex2.approx.f32` on B200, using polynomial emulation instead — likely for better MMA-overlap scheduling, not a capability limitation (SM103/RTX5090 lacks SFU; FA4 uses hardware there). The B200 emulation is a deliberate throughput choice.
**Competition relevance**: MEDIUM-HIGH. Softmax inner loop (32 exp2f calls per block, WG0) is on the critical path. If hardware `ex2.approx.f32` = 4/cycle SFU but polynomial = 3 FMAs pipelined with MMA, the software path may win for overlap. **Never benchmarked on B200.**

### 1.6 `tcgen05.ld 16dp64bNx` — unbenchmarked TMEM addressing mode

**Source**: `CuTeDSL cutlass/cute/nvgpu/tcgen05/copy.py:155` `Ld16x64bOp`; valid N: x1..x128
**Competition relevance**: MEDIUM. O-rescale path uses `tmem_ld_32dp32bNx<64>` × 4 chunks (622 cy). Different bank conflict pattern with 16dp64b may reduce cost. Not needed if split-KV eliminates per-block online softmax.
**Benchmark parameter**: Add `{32dp32b, 16dp64b, 16dp128b, 16dp256b}` mode sweep to TMEM ld/st experiment at equivalent data sizes.

### 1.7 `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` — atomic accumulation

**Source**: `csrc/kerutils/include/kerutils/device/sm90/intrinsics.cuh:43` (`tma_bulk_reduce_add`)
**What it does**: Atomically adds f32 values from smem to global memory — eliminates separate combine kernel.
**Competition relevance**: MEDIUM. The combine kernel adds ~1–2 μs for num_tokens=1. If `tma_bulk_reduce_add` atomically accumulates partial O across splits, combine kernel is eliminated. **Throughput never measured.**

### 1.8 LDG.256 hint matrix for index loads

**Source**: KU_LDG_256 macro; warp 7 loads B_TOPK=64 indices (512 bytes) via `__ldg` with default hints
**Valid combos**: L1 ∈ {no_allocate, evict_first, evict_normal, evict_last, evict_unchanged} × L2 ∈ {evict_first, evict_normal, evict_last} × L2_prefetch ∈ {64B, 128B, 256B} × .nc
**Competition relevance**: MEDIUM. Indices are not reused across blocks. `L1::no_allocate + L2::evict_first` frees L1/L2 for KV cache lines. Reference uses default `__ldg` (evict_normal). **Never benchmarked with explicit hint matrix.**

---

## Section 2: Argument Combination Matrix

### `createpolicy.fractional` — untested combinations

| Primary | Secondary | Fraction | Benchmarked |
|---------|-----------|----------|-------------|
| evict_last | evict_first | 1.0 | YES (implicit EVICT_LAST) |
| evict_last | evict_first | 0.25, 0.5, 0.75 | **NO** |
| evict_first | evict_unchanged | 0.25, 0.5, 0.75, 1.0 | Only 1.0 implicit |
| evict_normal | evict_first | any | **NO** |

### `tcgen05.ld` addressing modes

| Mode | pack::16b | Benchmarked |
|------|-----------|-------------|
| 32dp32bNx | NO | YES (latency only) |
| 16dp64bNx | NO/YES | **NO** |
| 16dp128bNx | NO/YES | **NO** |
| 16dp256bNx | NO/YES | **NO** |

### TMA at competition num_tokens scale

| num_tokens | Total blocks | SMs | Multi-block? | Benchmarked |
|------------|-------------|-----|-------------|-------------|
| 1 | 32 | 32 | No | YES (ver8) |
| 2 | 64 | 64 | No | NO |
| 6 | 192 | 148 | YES (1.30×) | NO |
| 7 | 224 | 148 | YES (1.51×) | NO |
| 8 | 256 | 148 | YES (1.73×) | NO |

---

## Section 3: Prioritized Next Benchmarks

### Priority 1: TMA + UTCMMA Concurrency (utcmma ver6) — 5–8h

**Why**: Determines whether 33% speedup from inter-block pipelining is physically achievable. Critical for 61% of competition workloads (num_tokens≥5).
**Design**: warp 0 issues N=16 gather4 calls while warp 4 executes K=32 WS-TS M64N64. Measure wall time from first instruction to both complete. Compare to sequential baselines. Sweep (TMA N) × (UTCMMA K).
**Expected**: if walltime ≈ max(TMA, GEMM) → genuine overlap, design double-buffer. If ≈ A+B → serialized, stay simple.

### Priority 2: M=32 WS-TS via raw PTX (utcmma ver6) — 2–4h

**Why**: h_q=16 wastes 75% of M=64 tensor core rows. M=32 → 50% utilization + dual-QK design (+parallel QK_ckv and QK_kpe).
**Design**: Inline PTX kernel measuring M=32 WS-TS K=1 and K=32 completion latency. Compare to M=64 baseline (173 cy). If ≤173 cy, switching is free or better.

### Priority 3: SM90_BULK_COPY_S2G latency (new: bulk_copy benchmark) — 2–3h

**Why**: Split-KV output write (32,768 bytes/block) is on critical path but timing model ignores it entirely.
**Design**: Measure `cp.async.bulk.global.shared::cta.bulk_group` latency for {512B, 2KB, 32KB, 128KB}. Determine if async overlap is achievable with concurrent TMA.

### Priority 4: kpe pipeline_competition N=16 (tma_gather4 ver9) — 1h

**Why**: Per-block TMA = max(ckv, kpe). kpe latency at N=16 is estimated (~230 ns) but unmeasured.
**Design**: Replicate ver8 with bf16 dtype and 512B/gather4 box. Measure per-call kpe latency at N=16, competition FRESH_scatter pattern.

### Priority 5: createpolicy fraction sweep (dual_tma_stream ver2) — 2h

**Why**: In the L2-warm scenario (e.g., repeated in-process benchmarking), kpe concurrent degradation (201 ns → 595 ns) may be recoverable with partial eviction policy. NOTE: competition evaluation has no guaranteed L2 warmth — plan for HBM-cold as baseline. At HBM-cold scale (ver1 1024MB sequential), interference is already minimal (~0% ckv, +7% kpe). This experiment is valuable for understanding in-process repeated inference but is not critical for competition planning.
**Design**: Sweep kpe fraction={0.25, 0.5, 0.75, 1.0} with evict_first primary in concurrent mode.

### Priority 6: Dual TMA delay sweep (dual_tma_stream ver2) — 1h

**Why**: Optimal kpe stagger after ckv issue. At HBM-cold (competition baseline), the +7% kpe degradation is small and may not warrant staggering. Still worth measuring to bound the optimization.
**Design**: Sweep kpe_delay_ns={0, 50, 100, 150, 200, 300, 400} after ckv issue at HBM-cold scale.

### Priority 7: LDG.256 hints for index loads (new benchmark) — 2–3h

**Why**: Warp 7 index load policy affects L1/L2 availability for KV cache. Reference uses default; optimal is untested.
**Design**: 64-index fetch (512 bytes) under explicit {L1_hint} × {L2_hint} × {L2_prefetch} × {.nc} matrix. Expected winner: L1::no_allocate + L2::evict_first + 64B.

### Priority 8: ex2.approx.f32 throughput (new benchmark) — 2h

**Why**: FA4 uses 3-FMA polynomial over hardware SFU on B200 — reason is throughput/overlap, not capability. Need measurement to decide.
**Design**: Throughput of: hardware ex2.approx.f32, ex2.approx.f32x2 (if valid), FA4 3-FMA polynomial. Report cycles per exp2 at varying occupancy levels.

---

## Section 4: Improvements to Existing Benchmarks

**tma_gather4 ver9**:
- Add kpe (bf16, 512B) at all N pipeline depths — currently only tested at low N for kpe
- Add num_blocks sweep at [32, 64, 148, 192, 224, 256] to model multi-block/SM for num_tokens=[1,2,5,6,7,8]
- Add L2-warm fixed-indices scenario (competition eval regime — fixed sparse_indices across timing runs)

**utcmma ver6**:
- M=32 WS-TS via raw PTX (Priority 2)
- N_ACC extended to {8, 16, 32} — need N_ACC ≈ 173 cy / 11 cy ≈ 16 to test paper's throughput claim
- tcgen05.ld 16dp64bNx vs 32dp32bNx comparison

**dual_tma_stream ver2**:
- kpe delay sweep 0–400 ns (Priority 6)
- createpolicy fraction sweep for kpe (Priority 5)
- HBM-cold concurrent behavior at small working sets (competition evaluation has no guaranteed cache warmth; HBM-cold is the correct baseline — already covered by ver1 1024MB sequential)

---

## Section 5: Design Decisions This Analysis Unblocks

| Unknown | Resolving benchmark | Kernel design gate |
|---------|--------------------|--------------------|
| TMA+UTCMMA overlap? | Priority 1 | Double-buffer (33% gain) vs stay serial |
| M=32 WS-TS latency? | Priority 2 | M=32 for better utilization + dual-QK |
| Bulk_group cost? | Priority 3 | Timing model accuracy; cp.reduce vs copy+combine |
| kpe N=16 latency? | Priority 4 | Per-block TMA = max(ckv, kpe) — gap filled |
| Optimal kpe stagger? | Priority 6 | Warp 5/6 issue timing |
| ex2 throughput? | Priority 8 | Hardware SFU vs polynomial for softmax |

---

## Sources
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVIDIA CUDA Developer Forums: https://forums.developer.nvidia.com/
- arxiv:2512.02189 — Microbenchmarking NVIDIA's Blackwell Architecture
- FA4 source: `references/flash-attention/flash_attn/cute/flash_fwd_sm100.py`
- FlashMLA intrinsics: `csrc/kerutils/include/kerutils/device/sm90/intrinsics.cuh`
