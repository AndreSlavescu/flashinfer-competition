---
name: B200 Confirmed Measured Performance Values
description: Empirically confirmed B200 performance values from real Modal hardware runs, with measurement context
type: project
---

Confirmed B200 (sm_100a) performance from tma_gather4 benchmark, ver5 CSV, CUDA 13.0, 148 SMs:

**TMA gather4 latency (single CTA, pipeline N=1)**
- L2-RESIDENT total: ~225 ns (includes barrier overhead ~28 ns)
- L2-RESIDENT net (minus barrier): ~197 ns (ckv_int64, 4096B), ~182-190 ns (kpe_bf16/ckv_bf16, 512B)
- Context: 1024-iteration loop, 16MB unique data, all L2-resident after warmup
- HBM-COLD net: **546 ns (ckv_int64, 4096B)**, **492 ns (kpe_bf16, 512B)** — ver5 NEW MEASUREMENT
- HBM-cold context: sequential non-repeating access of entire 1024MB working set
- IMPORTANT: The ~197 ns L2-resident value does NOT match the published 420-cycle HBM miss latency. That 420-cycle figure is for ld.global, not TMA. TMA HBM-miss latency is ~1100 cycles (546 ns at 2.1 GHz) — 2.77× higher than ld.global because TMA goes through a DMA pipeline with additional overhead.
- Null barrier overhead (expect_tx + arrive + try_wait): ~28 ns, stable across configs

**TMA gather4 pipeline N=2 speedup (ver5 confirmed)**
- N=1: 225 ns/gather4
- N=2: 174 ns/gather4 (1.29× improvement)
- N=4..32: 168-174 ns/gather4 (diminishing, <4% more)
- Conclusion: N=2 pipeline stages is optimal for single-CTA TMA

**TMA gather4 aggregate throughput (76 CTAs, ckv_int64 4096B/gather4, ver5)**
- L2-resident (≤64MB): ~900-904 GB/s
- HBM-bound (256MB+): ~390-405 GB/s (random)
- competition_realistic/pt at 256MB: ~897-908 GB/s — MISLEADING, only 2048 unique rows = 2MB unique data, all L2-resident regardless of declared working set
- Single CTA: ~5.68 GB/s HBM-bound

**TMA gather4 multi-CTA saturation (ver5, extended to 296 CTAs)**
- 76 CTAs: 405 GB/s; 152 CTAs: 788 GB/s; 176: 906; 224: 1151; 256: 1309; 296: 1511 GB/s
- Still NO HBM saturation at 296 CTAs (= 2 CTAs/SM for all 148 SMs)
- 1511 GB/s = 18.9% of 8 TB/s HBM peak — serial-issue TMA is latency-bound, not BW-bound
- Slight superlinear efficiency improvement at 176+ CTAs (memory controller operates more efficiently under higher concurrency)

**B200 L2 cache size (empirical, confirmed ver5)**
- Cliff between 64MB (902 GB/s) and 128MB (540 GB/s)
- L2 size: ~64-80 MB (not sharp, set-associative)
- Consistent with arxiv:2512.02189 spec of ~64 MB

**L2 promotion size effect (ver5 NEW)**
- L2_PROMOTION_L2_128B vs L2_PROMOTION_L2_64B: ~0% difference at all working set sizes
- Cliff location identical for both. Use L2_128B as default; L2_64B offers no benefit.

**TMA box size effect on throughput (ver5 confirmed)**
- 256B/gather4 → 289 GB/s; 512B → 330; 1024B → 371; 2048B → 402; 4096B → 405 (HBM-bound, 76 CTAs, 256MB)
- Larger boxes amortize per-barrier overhead. Data type (INT64 vs BF16) irrelevant at same bytes/gather4.

**kpe TMA (512B/gather4, bf16 dim0=64)**
- ~53 GB/s HBM-bound at 76 CTAs
- L2-hot latency: ~182-190 ns net (single CTA)
- HBM-cold latency: ~492 ns net (NEW ver5)
- Much slower aggregate throughput than ckv because smaller box = more barriers per unit data

**Cache hints (ver5 confirmed)**
- evict_last vs evict_normal vs none: essentially identical (< 5% difference) at HBM-bound scale
- evict_first: -48% for sequential L2-resident at 64MB. Avoid for cached data.
- Competition recommendation: use evict_last (never worse, best for L2-resident cases)

**TMA gather4 prefetch effectiveness (ver6 NEW)**
- `tma_gather4_prefetch()` (`cp.async.bulk.prefetch.tensor.2d.L2...tile::gather4`) tested at DIST=0,1,2,4
- DIST=0 baseline: 537.7 ns (1.2% faster than latency_cold due to spurious same-row prefetch — negligible artifact)
- DIST=1: ~380 ns (29% latency reduction)
- DIST=2: ~324 ns (40% latency reduction) — **saturation point**
- DIST=4: ~324 ns (no improvement over DIST=2)
- Conclusion: issue prefetch for block i+2 while computing on block i; 2 outstanding requests saturate the TMA prefetch pipeline

**32-CTA competition-scale TMA throughput (ver6 NEW)**
- At 32 CTAs (competition workload: 1 CTA per split-KV block × 32 blocks):
  - 64MB (L2-resident): ~393 GB/s
  - 256MB (HBM-bound): ~174 GB/s
  - 1024MB (HBM-bound): ~166 GB/s
- 2.32× slower than 76-CTA results. L2 cliff at same 64-128MB location.

**FRESH competition scatter throughput (ver6 NEW)**
- FRESH_COMPETITION_REALISTIC at 1024MB: **385.73 GB/s** (within 1.1% of pure random 389.95 GB/s)
- competition_realistic at 1024MB: ~905 GB/s (L2-warm artifact — only 8MB unique data)
- B_TOPK=64 sort structure adds zero measurable benefit in HBM-bound regime
- The 906 GB/s figure is NOT the real competition bandwidth — use 386 GB/s for planning

**Cache hints — kpe update (ver6 NEW)**
- kpe_bf16 `evict_first` at 64MB (L2 boundary, sequential): **65 GB/s — 47% penalty** vs evict_last (122 GB/s)
- kpe_bf16 `evict_first` at HBM-bound (256MB+): **+5% faster** than evict_last
- Updated recommendation: ckv=evict_last (keeps data in L2), kpe=evict_first (frees L2 for ckv, marginal HBM gain)

**Dual TMA stream interference (dual_tma_stream ver1 NEW)**
- HBM-cold (sequential 1024MB), BOTH mode: ckv **7.58 GB/s isolated = 7.58 GB/s concurrent** (0% interference)
- L2-warm (random 256MB), BOTH mode: kpe 201 ns isolated → **595 ns concurrent** (≈ HBM cold, 3× slowdown)
- Root cause: ckv's 256MB working set evicts kpe's 8MB data from L2 → kpe serializes to HBM
- In production (8MB KV fits in L2), ckv will dominate L2 — kpe must be budgeted at HBM-cold latency (~595 ns)

**Why:** These values are from actual B200 hardware runs on Modal, ver5/ver6 and dual_tma_stream ver1. Use these over CLAUDE.md specs which are literature estimates.
**How to apply:** When analyzing future benchmarks or designing the competition kernel, use these as ground truth for B200 TMA performance.

---

## UTCMMA and TMEM Performance (utcmma-ver3.csv, 2026-03-19)

**Sustained clock rate**: 1.854 GHz (stable across ver1 and ver3; 18% below 2.25 GHz advertised boost).

**UTCMMA RAW latency (single accumulator, dependency chain)**

| Tile | cy/K-tile | ns/K-tile |
|---|---|---|
| M=64 N=64 k16 | 54.5 | 29.4 |
| M=64 N=128 k16 | 54.5 | 29.4 |
| M=64 N=256 k16 | 76.0 | 41.0 |
| M=128 N=128 k16 | 74.0 | 39.9 |
| M=128 N=256 k16 | 138.0 | 74.5 |
- K-depth linearity: ±2% from k=1 to k=32 (M=64 N=128)
- K_DEPTH = number of K-tiles (each 16 BF16 elements) along the GEMM reduction dimension
- QK GEMM (M=64 N=128 k_depth=32): **1,719 cycles = 928 ns** (competition QK is N=64, but per-K-tile cost is identical to N=128)
- SV GEMM single tile (M=64 N=256 k_depth=4 SS): **320 cycles = 173 ns**; competition SV needs 2 tiles (N=256×2=512 output) = **640 cy = 346 ns**

**CRITICAL: The arxiv:2512.02189 figure of ~11 cycles is for independent-accumulator throughput
(initiation interval), NOT the RAW dependency chain. Competition kernel serialized RAW chain
is ~54 cycles/K-tile — 5× slower than the paper's figure.**

**UTCMMA multi-accumulator throughput** (N_ACC=1..4, all identical):
- Rotating accumulator TMEM columns provides ZERO speedup for N_ACC ≤ 4.
- Structural initiation interval appears to be ~54 cycles.
- N_ACC ≥ 5 still untested; N_ACC=8 (M=64 N=64) needed for definitive conclusion.

**WS vs non-WS**:
- M=64 N=64: identical (54.5 cy both)
- M=64 N=128: WS=54.5 cy, non-WS=64.0 cy (+17%). Always use .ws variants for N≥128.

**TMEM load latency** (stable TMEM, no pending MMA): ~1.8 cycles (clock-read floor; functionally free)

**TMEM store latency** (N=64 + fence): **49.8 cycles = 26.8 ns** (linear: 12 + 0.59×N cy)

**Fence costs**:
- Warp-collective TMEM wait pair (load+store fence, nothing in flight): **12.83 cycles**
- Per-thread tcgen05 fence pair (before/after_thread_sync): **2.83 cycles**

**Full O-rescale roundtrip** (isolated measurement):
- 1 chunk (N=64 ld + 32×float2_mul + N=64 st): **168.1 cycles = 85.6 ns**
- 4 chunks (full D=512 width): **622.1 cycles = 316.6 ns**
- NOTE: O-rescale applies only to online softmax designs where one SM processes multiple KV chunks sequentially. In split-KV (competition design), each SM processes one B_TOPK=64 chunk with a single softmax pass — no O-rescale needed.

**Synchronization costs**:
- named_bar.sync(128 threads): **20.67 cycles = 10.5 ns**
- named_bar.sync(32 threads): **14.65 cycles = 7.5 ns**
- mbarrier arrive+wait (expect_tx=0): **88.51 cycles = 45.0 ns**
- Full MMA + commit + wait (M=64 N=64): **173.03 cycles = 93.4 ns**
- tcgen05.commit overhead: ~30 cycles
- NOTE: In a pipelined kernel, mbarrier waits synchronize TMA completion — they're part of TMA latency, not additive overhead. Per-SM mbarrier count = ~32 waits (one per gather4 call: 16 ckv + 16 kpe).

**UTCCP (smem→TMEM copy) latency**:
- 1 copy (128×16 bf16 tile): 168.3 cy = 85.6 ns (startup cost)
- Marginal cost at 16 copies: ~74.7 cy/copy = 38.0 ns/copy
- Competition Q load (32 copies extrapolated): ~2,276 cycles = ~1,228 ns

**Competition kernel per-SM timing model (split-KV, B_TOPK=64, ver3 corrected)**:

TMA per block (16 gather4 calls each for ckv and kpe):
- NOTE: all TMA latency values above (546/595 ns) are PER-GATHER4-CALL, not per-block total
- HBM-cold sequential (no pipelining): 16 × 595 = ~9,520 ns per stream
- With N=2 pipelining + DIST=2 prefetch: ~16 × 324 = ~5,184 ns per stream (estimated)
- ckv and kpe run in parallel (verified zero HBM interference) → per-block TMA = max(ckv, kpe)
- **Per-block TMA: ~5,000-9,500 ns** (range depends on pipelining efficiency; unmeasured for multi-call)

GEMM per block (serialized RAW chain, all values measured):
- QK ckv (M=64 N=64 K_DEPTH=32): ~1,719 cy = ~928 ns
- QK kpe (M=64 N=64 K_DEPTH=4): ~218 cy = ~118 ns
- SV (2 × M=64 N=256 K_DEPTH=4): 2 × 320 = ~640 cy = ~346 ns
- **Per-block GEMM total: ~2,577 cy = ~1,392 ns**

With N=64 WS tiles (split-KV, 1 block/SM): GEMM needs all 64 KV tokens loaded → TMA and GEMM sequential.
**Per-block total = TMA + GEMM = ~6,400-10,900 ns. Kernel is TMA-BOUND (~3.6-6.8×).**
