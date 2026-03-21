# UTCMMA (tcgen05.mma) Analysis — B200 (sm_100a)

**Date**: 2026-03-17
**Benchmark**: microbenchmarks/utcmma/
**Results CSV**: microbenchmarks/utcmma/results/utcmma-ver1.csv
**CUDA version**: 13.0
**GPU**: NVIDIA B200 (sm_100a), 148 SMs

---

## Summary

The UTCMMA benchmark measures serialized dependency-chain latency for tcgen05.mma on B200. The primary finding is a stark discrepancy between this benchmark's measured ~54 cycles/GEMM and arxiv:2512.02189's reported ~11 cycles latency — a discrepancy that is fully explained by methodology: arxiv:2512.02189 measures throughput-mode occupancy (independent MMAs, initiation interval), while this benchmark measures the full RAW dependency chain including TMEM read-back. The true structural pipeline latency per K-tile is ~54 cycles (small N tiles) to ~80 cycles (N=256), which is what matters for competition kernel design. The K-depth sweep reveals essentially perfect linear scaling (~53–54 cycles/K-tile), and the UTCCP Q-load path shows a pipeline-initiating cost of ~168 cycles for the first copy that amortizes to ~74 cycles/copy at 16-copy batches. **NOTE (2026-03-19 correction):** The ver1 timing model below incorrectly used per-gather4 TMA latencies as per-block totals. Per-block TMA = 16 gather4 calls × per-call latency = ~5,000–9,500 ns, far exceeding GEMM (~1,392 ns). The kernel is **TMA-bound**, not compute-bound. See notes/utcmma_review.md for the corrected model.

---

## Implementation Review

### Strengths

- **Clock measurement**: Both `%clock64` and `%globaltimer` are used correctly. The `memory` clobber on all inline asm calls prevents compiler reordering around timing boundaries.
- **RAW dependency chain enforcement**: `tiled_mma.accumulate_ = UMMA::ScaleOut::One` inside the loop body forces each GEMM to read the accumulator written by the previous one. This is a correct RAW chain measuring full round-trip latency: issue MMA → TMEM write → re-read TMEM → issue next MMA.
- **TMEM allocation correctness**: `Allocator1Sm().allocate(512, &smem_tmem_addr)` from warp 0 (threadIdx.x < 32) followed by `release_allocation_lock()` and `__syncthreads()` — correct sm_100a TMEM management sequence.
- **Single-thread measurement**: Only thread 0 runs the timed section, excluding inter-thread synchronization from timing.
- **Warmup**: 100 warmup iterations (latency_ts) / 20 (kdepth/ss/swizzle). Sufficient.
- **Statistical rigor**: 11 runs with trimmed-mean (drop min/max, average 9). Spread 0.0–0.1% across all 24 experiments — exceptional stability.
- **Compilation flags**: `-gencode arch=compute_100a,code=sm_100a` correct. `compute_100a` (not `compute_100`) is required for the datacenter B200 variant.
- **Competition tile coverage**: kdepth_ts goes to k_depth=32 matching ckv K=512; ss experiment covers M=64 N=256 K_DEPTH=4 matching SV GEMM exactly.

### Issues Found

- **MINOR — CSV swizzle label**: `utcmma_latency_ts` reports "SW128" but internally uses SW32 for k_depth=1 (K=16 does not satisfy SW128's K-atom=64 requirement). Cosmetic mismatch; swizzle experiment confirms zero performance impact.
- **MINOR — UTCCP includes commit+wait in timed loop**: The barrier round-trip adds ~28 ns (52 cycles) overhead to each UTCCP measurement. This is the correct semantics for competition use but should be noted.
- **MINOR — No tcgen05.fence in MMA-only timed loops**: Correct — tcgen05 fences are only needed at `__syncthreads()` boundaries crossing TMEM paths. The MMA timed loops do not cross such boundaries.
- **MINOR — TFLOPS values are single-CTA serialized chain, not peak throughput**: 4–14 TFLOPS vs 1,926 TFLOPS peak. Intentional but should be labelled as "latency-mode TFLOPS."

No CRITICAL or MAJOR issues found.

### Recommendations

1. Add throughput-mode experiment (`accumulate_=Zero`, independent MMAs) to directly verify the 11-cycle initiation interval from arxiv:2512.02189.
2. Add combined TMA+UTCMMA overlap experiment to determine if hardware pipelines both units concurrently.
3. Fix "SW128" label in utcmma_latency_ts CSV to "SW32" for k_depth=1.

---

## Clock Rate Determination

From 64x128 latency_ts: 544,594 cycles / 10,000 iters = 54.46 cycles/iter = 29.5 ns/iter.
`%clock64` rate = 54.46 / 29.5 ns = **1.845 GHz** (sustained compute clock on this Modal B200 instance, ~18% below 2.25 GHz advertised boost).

---

## Results vs. Prior Research

### Experiment 1: utcmma_latency_ts — TS single K-tile RAW-chain latency

| Tile Shape | Measured (cycles) | Measured (ns) | arxiv:2512.02189 (throughput) | Assessment |
|---|---|---|---|---|
| m64n64k16 | 54.5 | 29.5 | ~11 cycles | EXPECTED DIVERGE |
| m64n128k16 | 54.5 | 29.5 | ~11 cycles | EXPECTED DIVERGE |
| m64n256k16 | 76.0 | 41.2 | ~11 cycles | EXPECTED DIVERGE (larger N) |
| m128n128k16 | 74.0 | 40.1 | ~11.3 cycles | EXPECTED DIVERGE |
| m128n256k16 | 138.0 | 74.8 | ~11.4 cycles | EXPECTED DIVERGE |

**Reconciliation**: arxiv:2512.02189 measures the MMA *initiation interval* (throughput rate = ~11 cycles/issue at steady state with independent MMAs). This benchmark measures the full RAW dependency latency — the time from issuing MMA[i] until MMA[i+1] can issue given that MMA[i+1] reads MMA[i]'s TMEM output. The gap: 54 − 11 = ~43 cycles represents TMEM drain + re-read latency for the accumulator read-back.

**N-scaling**: m64n64 = m64n128 = 54.5 cycles (zero cost to double N from 64 to 128). m64n256 jumps to 76.0 cycles (+40%). The N=64/128 tile output drains within a fixed overhead window; N=256 extends beyond it.

### Experiment 2: utcmma_kdepth_ts — K-depth linearity (M=64, N=128)

| K-depth | Cycles/K-tile | Linearity |
|---|---|---|
| 1 | 54.6 | baseline |
| 4 | 54.6 | 1.000 |
| 8 | 53.8 | 0.985 |
| 16 | 53.8 | 0.985 |
| 32 | 53.7 | 0.983 |

**Perfect linearity (±2.4%) from k=1 to k=32.** QK GEMM at M=64 N=128 K_DEPTH=32: **1,719 cycles = 931 ns**. (Note: benchmark uses N=128; competition QK is N=64 but per-K-tile cost is identical.)

### Experiment 3: utcmma_latency_ss — SS (Shared×Shared, SV path)

| Config | Cycles/K-tile | vs TS m64n128 |
|---|---|---|
| 64×128×1 | 52.5 | −4% (SS slightly faster) |
| 64×256×1 | 80.0 | +47% (larger N) |
| 64×256×4 (per tile) | 80.0 | same as single-tile SS |

Competition SV GEMM (M=64, N=256, K_DEPTH=4): **320 cycles = 173 ns**.

### Experiment 4: utccp_latency — UTCCP smem→TMEM copy

| Num copies | Cycles/batch | Cycles/copy | ns/copy |
|---|---|---|---|
| 1 | 168.3 | 168.3 | 91.2 |
| 2 | 194.5 | 97.2 | 52.7 |
| 4 | 379.5 | 94.9 | 51.4 |
| 8 | 688.4 | 86.1 | 46.6 |
| 16 | 1191.7 | 74.5 | 40.4 |

**Startup overhead ~168 cycles; marginal cost ~68 cycles/copy from copy 2 onward.** Competition Q load (32 copies, extrapolated): **~2,276 cycles ≈ 1,234 ns**.

### Experiment 5: utcmma_swizzle — swizzle insensitivity

All four modes (INTER/SW32/SW64/SW128) produce identical 218.4 cycles/4 K-tiles (sub-0.001% variance). **Use SW128 throughout; swizzle choice is irrelevant to UTCMMA performance.**

---

## Full Per-Block Timing Model (Updated with Measured Data)

**SUPERSEDED by ver3 timing model in notes/utcmma_review.md. Key corrections since ver1:**
- TMA "budget" values below were per-gather4-call, not per-block. Per-block TMA = 16 calls × per-call = ~5,000–9,500 ns.
- SV GEMM needs 2 tiles (N=256×2) for full D=512 output: 2 × 173 = 346 ns, not 173 ns.
- mbarrier waits synchronize TMA completion — part of TMA time, not additive overhead.
- Kernel is **TMA-BOUND** in all regimes, not compute-bound.

See notes/utcmma_review.md "Competition Kernel Timing Model" for corrected values.

---

## Implications for Competition Kernel

1. **Pipeline depth**: Kernel is TMA-bound (corrected). N=2 pipeline is sufficient for TMA. Key optimization is reducing per-block TMA time (prefetch, cache hints, pipelining multiple gather4 calls).

2. **Q loading overlap**: UTCCP (~1,234 ns extrapolated, 32 copies) can be pipelined with the SV GEMM (173 ns) and softmax. If UTCCP is async and can overlap with UTCMMA, the effective cost is much lower. The UTCCP startup cost (~91 ns) is the irreducible minimum before QK GEMM can start.

3. **Tile sizes**: M=64, N=64 for competition QK (B_TOPK=64). M=64, N=256×2 for SV (2 tiles for D=512 output). N=128 in benchmark is the FlashMLA shape (two B_TOPK=64 blocks packed), not competition-exact.

4. **Swizzle confirmed**: SW128 for all smem layouts — zero UTCMMA performance impact.

5. **Optimization priority revision (L2-hot regime)**:
   - Primary: reduce UTCCP latency / overlap Q loading with SV GEMM
   - Secondary: reduce softmax TMEM readback overhead (tcgen05.ld latency, unmeasured)
   - Tertiary: TMA prefetch (already optimal at DIST=2 from prior benchmarks)

---

## Knowledge Gaps and Open Questions

1. **UTCMMA throughput rate**: A `accumulate_=Zero` independent-MMA experiment is needed to verify the 11-cycle initiation interval on this B200 instance.
2. **TMA+UTCMMA concurrent execution**: Does issuing TMA while UTCMMA runs preserve independent throughput for both? Critical for confirming whether pipelining actually works.
3. **Softmax TMEM readback overhead**: tcgen05.ld latency, P extraction, and O rescaling path are unmeasured. Could add 500–1,000 ns/block.
4. **UTCCP + UTCMMA overlap**: If UTCCP and UTCMMA can execute concurrently, the 1,234 ns Q load cost is hidden behind compute.
5. **TMA pipelining efficiency**: Measure per-block TMA time for 16 consecutive HBM-cold gather4 calls with N=2 pipeline. This determines where in the 5,000–9,500 ns range the per-block TMA falls.

---

## References

1. **arxiv:2512.02189** — Microbenchmarking NVIDIA's Blackwell Architecture (Dec 2025): UTCMMA 11-cycle throughput, 1,926 TFLOPS BF16, TMEM specs
2. **arxiv:2507.10789** — Dissecting the NVIDIA Blackwell Architecture (Jul 2025): L1 latency, warp schedulers, sm_100a characterization
3. **tma_gather4 ver5/ver6 + dual_tma_stream ver1** (measured Modal B200): TMA latency 197/540/595 ns
4. **utcmma-ver1.csv** (measured Modal B200, 2026-03-17): primary data source
5. **csrc/sm100/decode/head64/config.h**: B_TOPK=64, TMEM column layout, tile sizes
