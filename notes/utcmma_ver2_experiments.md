# UTCMMA Ver2 Experiment Design

**Date**: 2026-03-17
**Source**: ptx-benchmark-explorer analysis of ver1 results + PTX ISA + competition kernel

---

## Background: What Ver1 Showed (and What It Missed)

The K-depth sweep is perfectly linear at ~53.7–54.6 cy/tile from k=1 through k=32. There is no pipeline effect — every MMA stalls waiting for the previous one's output because the single accumulator `tC_frag` creates a RAW dependency on every iteration. The paper's 11 cy/tile claim is for *independent* accumulators. Ver1 never measured that.

| Scenario | QK GEMM (k=32) | vs TMA cold (548 ns) | vs TMA warm (595 ns) |
|---|---|---|---|
| Serialized RAW chain (ver1 measured) | 932 ns | 1.7× slower than TMA | 1.6× slower than TMA |
| 2-acc pipelined (if paper's 11 cy/tile holds) | ~190 ns | 2.9× faster than TMA | 3.1× faster than TMA |

The two scenarios represent completely different kernel design spaces. This is the most important unknown in the entire benchmark program.

---

## Prioritized New Experiments

### Experiment 6 — `utcmma_throughput_2acc` — **CRITICAL**

**Question answered**: What is the true per-instruction throughput of UTCMMA with independent accumulators? Is it 11 cycles (paper) or ~54 cycles (implies structural serialization)?

**Why it's #1**: Every pipeline design decision — double vs. triple buffer, whether to optimize TMA vs compute, whether to add a third warpgroup — depends on whether UTCMMA throughput is 11 or 54 cycles per K-tile.

**Implementation approach**:

For the WS C accumulator (`tmem_frg_ws_1sm`, used in all competition UTCMMA), **columns = N/2** per MMA tile. Verified against config.h: P at tmem_cols::P=400..464 = 64 cols for N=128 ✓; O at 0..256 = 256 cols for D_V=512 = 2 tiles × 128 cols ✓.

| N | Columns per accumulator | Max N_acc in 512-col budget |
|---|---|---|
| 64 | 32 | 16 |
| 128 | 64 | 8 |
| 256 | 128 | 4 |

Use M=64, N=64 for the sweep:
- C0: columns 0–31 (`tC0.data().get() = 0`)
- C1: columns 32–63 (`tC1.data().get() = 32`)
- C2: columns 64–95 (`tC2.data().get() = 64`)
- C3: columns 96–127 (`tC3.data().get() = 96`)

Issue in rotation: `gemm(mma, A, B_k, C[i % N_acc])` for N_acc × K_DEPTH tiles. Separation between consecutive C[i] accesses = N_acc − 1 intervening instructions. Sweep N_acc = {1, 2, 3, 4}.

Also run M=64, N=128 with N_acc=2 as competition-exact (two 64-col accumulators = 128 total, well within budget).

**Expected result shape**:
- If true throughput = 11 cy: N_acc=1→54 cy, N_acc=2→~22 cy, N_acc=3→~11 cy, plateau at ≥3
- If true throughput = 27 cy: N_acc=1→54 cy, N_acc=2→~27 cy, plateau at ≥2
- If true throughput = 54 cy: flat ~54 cy regardless of N_acc (implies structural serialization, not RAW)

**Key implementation note**: Each TiledMMA instance needs a distinct `idesc_` pointing to a different TMEM C column range. Best to use separate fragment objects with explicit `.data().get()` column assignments rather than relying on CuTe's `gemm()` abstraction.

**Output columns**: `n_acc`, `throughput_cy_per_tile` = total_cycles / (n_acc × k_depth × iters)

---

### Experiment 7 — `tmem_ld_st_fence_overhead` — **HIGH**

**Question answered**: What is the full cost of the O-rescale TMEM read-modify-write sequence?

**Why it's #2**: From `kernel.cuh`, the O-rescale path per chunk (4 chunks per rescale event):
```
tcgen05_after_thread_sync()
[4 chunks]:
  tmem_ld_32dp32bNx<64>()
  fence_view_async_tmem_load()
  [32× float2_mul]
  tmem_st_32dp32bNx<64>()
  fence_view_async_tmem_store()
tcgen05_before_thread_sync()
```
With 32 blocks/token and rescaling on ~28 of them: 28 × 4 = 112 fence roundtrips. At 100 cycles each = 11,200 cycles = 5.3 µs hidden cost.

**Sub-experiments**:

- **Sub-A**: Raw `tmem_ld_32dp32bNx<N>` serialized chain latency. Sweep N = {1, 4, 8, 16, 32, 64, 128}.
- **Sub-B**: Raw `tmem_st_32dp32bNx<N>` serialized chain latency. Sweep N = {1, 4, 8, 16, 32, 64, 128}.
- **Sub-C**: `fence_view_async_tmem_load` latency when K loads are in-flight (issue K loads, time fence drain).
- **Sub-D**: Full roundtrip for N=64 (competition chunk size): after_thread_sync + ld + fence_ld + mul + st + fence_st + before_thread_sync. Report cycles per roundtrip.

---

### Experiment 8 — `mbarrier_roundtrip_overhead` — **HIGH**

**Question answered**: What is the cycle cost of the synchronization primitives that gate each main-loop iteration?

**Why it's #3**: Competition kernel has ~5 barrier waits per main-loop iteration in WG0, ~4 in WG1. At 32 iterations/token: ~288 barrier operations. At 50 cycles each = 14,400 cycles = 6.9 µs (potentially 20–30% of kernel time).

**Three sub-experiments**:
- **Sub-A**: `NamedBarrier::arrive_and_wait(128, N)` round-trip cost (WG0 intra-warpgroup sync).
- **Sub-B**: `transac_bar_t` round-trip: `arrive_and_expect_tx(0)` + `wait(phase)` (TMA completion wait pattern).
- **Sub-C**: Combined pattern: issue UTCMMA → `umma_arrive_noelect(bar)` → `bar.wait(phase)` + `tcgen05_after_thread_sync()`. Subtract known UTCMMA latency to isolate barrier overhead.

---

### Experiment 9 — `utcmma_commit_latency` — **HIGH**

**Question answered**: What is the raw latency of `tcgen05.commit.cta_group::1.mbarrier::arrive::one` (the `umma_arrive_noelect()` wrapper)?

**Why it's #4**: Called after every QK GEMM and SV GEMM (64 calls per token). The commit is the mechanism by which UTCMMA signals completion to WG0; its latency directly determines the pipeline WG0↔WG1 wake-up time.

**Implementation**: Measure:
1. `time(MMA × N + commit × N + wait × N)` — full serialized chain
2. `time(wait × N, no MMA)` — pure barrier roundtrip

Subtract (2) from (1), divide by N, subtract known MMA latency → commit cost in isolation.

*(Closely related to Experiment 8 Sub-C; combine into one kernel file)*

---

### Experiment 10 — `utcmma_dual_gemm_layout_check` — **HIGH** (quick verification)

**Question answered**: Does the competition kernel's dual-GEMM smem layout `Shape<B_H*2=128, K_TOTAL>` produce different UTCMMA latency than ver1's standard `Shape<N=128, K_TOTAL>`?

**Why it matters**: The competition kernel packs two B_TOPK=64 halves into a `[128, K]` B tensor. The UMMA descriptor encodes this shape. Verify that `make_umma_desc<K>` for `[128, K]` gives identical MMA performance to the standard `[64, K]` (logical N=128) layout used in ver1.

**Implementation**: Benchmark `kernel_utcmma_latency_ts<64, 128>` with the dual-GEMM smem layout `SmemLayoutKTiles_DualGemm_SW128<4>` (Shape<128, 256> in SW128). Compare to ver1's 54.5 cycles. Expected: identical. Low cost, direct competition-relevance.

---

### Experiment 11 — `utcmma_throughput_realistic_pipeline` — **HIGH** (complex)

**Question answered**: In a real double-buffered pipeline (TMA fills buffer N+1 while UTCMMA consumes buffer N), what is the actual per-block time? Do TMA and UTCMMA fully overlap?

**Why it matters**: If TMA and UTCMMA share any SM-level resource (warp scheduler, instruction queue), they may not fully overlap even if issued from different warps. Confirms or denies whether the competition kernel achieves `max(TMA, UTCMMA)` per block or `TMA + UTCMMA`.

**Implementation**: 128-thread kernel with warp4 for MMA, warp5 for TMA. Ping-pong smem buffers with mbarrier synchronization matching the competition kernel structure. Measure total time per iteration (one block of k_depth tiles with TMA overlap).

**Implementation order**: Do after Experiments 6–9 (need isolated measurements to interpret overlap results).

---

### Experiment 12 — `utcmma_ts_non_ws_comparison` — **MEDIUM**

**Question answered**: Does non-`.ws` `tcgen05.mma.cta_group::1.kind::f16` differ from `.ws` variant in latency or throughput?

**Implementation**: Add `SM100_MMA_F16BF16_TS_NOELECT` (non-WS) variants at M=64, N=128, k_depth=1 and k_depth=32. Expected: identical cycles to `.ws` variants.

---

### Experiment 13 — `utcmma_ss_kdepth_sweep` — **MEDIUM**

**Question answered**: Does SS UTCMMA (SV GEMM path) also scale linearly with k_depth? Can dual-accumulator pipelining help for N=256?

**Key architectural constraint**: One M=64 N=256 WS MMA tile uses 128 TMEM columns. The full SV GEMM (D_V=512, two tiles) uses 256 columns. Dual-pipelining the full SV GEMM (2×256 = 512 columns) exhausts the entire budget, leaving nothing for Q and P. Dual-acc pipelining is **impossible** for the full SS SV GEMM path. The SV GEMM is already fast (173 ns / 4 K-tiles) so this is not critical.

**Sweep**: k_depth = {1, 2, 4, 8} for SS M=64, N=256. Also attempt N_acc=2 with SS N=128 (each 64 columns, 128 total — comfortably fits) as proxy for throughput measurement.

---

## Argument Combination Matrix: tcgen05.mma Variants

| Variant | PTX | WS | CuTe Wrapper | Tested in Ver1 |
|---|---|---|---|---|
| TS, cta_group::1, .ws | `tcgen05.mma.ws.cta_group::1.kind::f16` | yes | `SM100_MMA_F16BF16_WS_TS_NOELECT` | Yes |
| SS, cta_group::1, .ws | same (desc_a from smem) | yes | `SM100_MMA_F16BF16_WS_SS_NOELECT` | Yes |
| TS, cta_group::1, non-.ws | `tcgen05.mma.cta_group::1.kind::f16` | no | `SM100_MMA_F16BF16_TS_NOELECT` | No → Exp 12 |
| SS, cta_group::1, non-.ws | same (desc_a from smem) | no | `SM100_MMA_F16BF16_SS_NOELECT` | No → Exp 12 |
| TS, 2-acc, N=64 | same PTX, two 32-col TMEM ranges | yes | manual column manipulation | No → Exp 6 |
| TS, 2-acc, N=128 | same, two 64-col accumulators | yes | manual | No → Exp 6 |
| tcgen05.commit | `tcgen05.commit.cta_group::1.mbarrier::arrive::one` | — | `umma_arrive_noelect()` | No → Exp 9 |
| tcgen05.ld.32dp32bNx | `tcgen05.ld.32dp32b{N×32}x` | — | `tmem_ld_32dp32bNx<N>` | No → Exp 7 |
| tcgen05.st.32dp32bNx | `tcgen05.st.32dp32b{N×32}x` | — | `tmem_st_32dp32bNx<N>` | No → Exp 7 |
| tcgen05.fence variants | `tcgen05.fence::before/after_thread_sync` | — | `tcgen05_before/after_thread_sync()` | No → Exp 7/8 |
| cta_group::2 variants | `tcgen05.mma.cta_group::2.*` | — | `SM100_MMA_F16BF16_2x1SM_*` | Skip |

---

## Key Architectural Constraints Discovered

1. **TMEM column budget for full SV GEMM accumulator**: One M=64 N=256 WS float32 MMA tile uses 128 TMEM columns. The full SV GEMM output (D_V=512) requires two N=256 tiles → 2×128 = **256 TMEM columns**. Dual-pipelining the full SV GEMM (2×256 = 512 columns) consumes the entire budget, leaving nothing for Q (~112 cols) and P (~64 cols). Dual-acc pipelining is therefore **impossible** in practice for the full-width SV GEMM path.

2. **TMEM overflow at k_depth=36 (M=64 NonInterleaved)**: Ver1 hit this bug. TMEM column address exceeds 511 at k_depth=36 with NonInterleaved tile_stride=2. Max safe k_depth=32 for M=64. Competition uses k_depth=32 exactly — safely within bounds.

3. **Dual-GEMM smem layout B_H×2=128**: Competition kernel uses `Shape<128, K_TOTAL>` for the B smem operand packing both B_TOPK=64 halves. UMMA descriptor canonicality must hold for this non-standard shape.

---

## Recommended Implementation Order for Ver2

| Order | Experiment | Effort | Unblocks |
|---|---|---|---|
| 1 | Exp 6 (2-acc throughput) | 3 hr | All pipeline design decisions |
| 2 | Exp 10 (dual-GEMM layout) | 1 hr | Validates 932 ns QK estimate |
| 3 | Exp 7 (tmem ld/st/fence) | 3 hr | O-rescale cost, TMEM BW |
| 4 | Exp 8+9 (mbarrier + commit) | 3 hr | Sync overhead, WG0↔WG1 latency |
| 5 | Exp 12 (non-WS comparison) | 1 hr | Confirms WS vs non-WS |
| 6 | Exp 13 (SS k_depth sweep) | 1 hr | Validates SV GEMM model |
| 7 | Exp 11 (realistic pipeline) | 5 hr | Final pipeline depth decision |

Total: ~17 hours. Items 1–4 (Exps 6, 7, 8/9, 10) ship as a single `ver2` binary in ~8 hours.
