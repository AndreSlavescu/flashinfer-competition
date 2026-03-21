# UTCMMA Benchmark Review
*Reviewed: 2026-03-19 | Hardware: B200, CUDA 13.0, driver 580.95.05, 148 SMs*
*Source: microbenchmarks/utcmma/results/utcmma-ver3.csv*

---

## Implementation Correctness

### Threading Model (all clean)

- `tcgen05.mma` issued from `threadIdx.x == 0` — correct (1 thread/CTA for cta_group::1)
- `TMEM::Allocator1Sm().allocate()` / `release_allocation_lock()` gated `threadIdx.x < 32` — correct (warp-collective)
- TMEM ld/st/wait (`.sync.aligned`) gated `threadIdx.x < 32` — correct
- `tcgen05.fence::before/after_thread_sync` (per-thread) gated `threadIdx.x == 0` — correct separation
- `fence_view_async_tmem_load/store` = CUTLASS wrappers for `tcgen05.wait::ld/st.sync.aligned` — called inside `threadIdx.x < 32` — correct

### RAW Latency Chain (all clean)

- `ScaleOut::One` throughout → each MMA reads the accumulator the previous MMA wrote → correctly measures serialized RAW latency
- Swizzle selection: `(K_TOTAL >= 64) ? SW128 : (K_TOTAL >= 32 ? SW64 : SW32)` matches K-atom divisibility rule
- SW128 label hardcoded in Exp 10 (dual_gemm_layout) — correct, both layout modes use SW128

### TMEM Column Budget (all clean)

Static assert in Exp 6: `TMEM_COL_C + N_ACC * COLS_PER_ACC <= TMEM_COL_A_LOCAL`
N_ACC=4, N=64, COLS_PER_ACC=32 → 128 ≤ 256. Enforced at compile time.

Non-WS TS (Exp 12): N=64 → 64 cols, N=128 → 128 cols, both within [0,256). WS uses N/2 cols. Correct.

---

## Issues Found

### MAJOR — Exp 6 Multi-Accumulator Throughput: Null Result

Rotating-accumulator design (N_ACC=1..4, K_DEPTH=4 or 32) produces **zero throughput improvement** — all configs at exactly 54.60 cy/K-tile, identical to single-accumulator RAW chain. This was intended to reproduce ~11 cy/tile throughput from arxiv:2512.02189 by breaking RAW dependencies across N_ACC independent TMEM column sets.

**Root cause**: Single-thread/single-warp MMA issuance serializes all completions from the same warp even when targeting independent TMEM columns. The 11-cycle throughput figure requires distinct warps (or distinct CTAs) issuing independently. Accumulator rotation alone is insufficient.

**Impact**: Exp 6 cannot validate the 11-cycle throughput claim. Exp 11 (realistic warp-specialized pipeline with MMA and consumer warps) is needed.

### NEEDS_VERIFICATION — TMEM Load Latency Measures Fence Overhead, Not Read Latency

`tmem_ld_32dp32bNx<N>` measurements all report **1.79 cy constant regardless of N** (1 to 64 columns). This is `fence_view_async_tmem_load` overhead for a fence that drains instantly (data already written before timing starts). True TMEM read latency under load is not captured — a proper measurement requires a data dependency: load to register → use value in subsequent store.

### MINOR — SS Swizzle Label "INTER_MN128" Conflates Two Operand Layouts

Exp 3 uses INTER for A and MN_SW128 for B. Composite label is acceptable but non-standard.

### MINOR — mma_commit_wait Does Not Decompose Commit vs. Wait

173 cy total = 54.5 cy MMA + 118.5 cy commit+wait overhead. Report should document decomposition.

---

## Results vs. Prior Research

| Metric | Expected (arxiv:2512.02189) | Measured (ver3) | Status |
|--------|----------------------------|-----------------|--------|
| UTCMMA RAW latency m64n64k16 | ~11 cy (throughput) | **54.5 cy** | EXPECTED DIVERGE — methodology difference |
| UTCMMA RAW latency m64n128k16 | ~11 cy | **54.5 cy** | EXPECTED DIVERGE |
| UTCMMA pipeline throughput (N_ACC rotation) | ~11 cy | **54.5 cy** | DIVERGES — null result, Exp 11 needed |
| BF16 tensor core throughput | 1,926 TFLOPS | ~9.0 TFLOPS | Not contradictory — single CTA, serialized chain |

The 11-cycle figure from arxiv:2512.02189 is not refuted — it's simply not achievable via single-thread accumulator rotation. Architecture requires warp-level instruction-level parallelism.

---

## Measured Values (ver3, B200 hardware)

### Exp 1 — WS-TS RAW Latency

| Tile | cycles/gemm | ns |
|------|-------------|-----|
| M=64 N=64 K=16 | 54.46 | 29.4 |
| M=64 N=128 K=16 | 54.46 | 29.4 |
| M=64 N=256 K=16 | 76.01 | 41.0 |
| M=128 N=128 K=16 | 74.01 | 39.9 |
| M=128 N=256 K=16 | 138.01 | 74.5 |

Note: M=64 N=64 == M=64 N=128 (54.46 cy each) — N has no latency impact in this range.

### Exp 2 — K-Depth Sweep (M=64 N=128)

| k_depth | cycles/gemm | ns | cy/K-tile |
|---------|------------|-----|-----------|
| 1 | 54.7 | 29.4 | 54.7 |
| 32 | 1719.1 | 928.0 | 53.7 |

K_DEPTH=32 at M=64: **1719 cy = 928 ns**. Note: benchmark uses N=128 (FlashMLA shape); competition QK is N=64, but per-K-tile cost is identical (54.5 cy for both N=64 and N=128). K_DEPTH = number of K-tiles (each 16 BF16 elements) along the reduction dimension. For QK ckv: K_DEPTH=32 → 32 × 16 = 512 = head_dim_ckv.

### Exp 3 — WS-SS Latency (competition SV)

| Config | cycles/gemm | ns |
|--------|------------|-----|
| M=64 N=128 K=1 | 52.46 | 28.3 |
| M=64 N=256 K=1 | 80.01 | 43.2 |
| M=64 N=256 K=4 | 320.0 | 172.7 |

Single SV tile (N=256, K_DEPTH=4): **320 cy = 173 ns**. Competition SV needs 2 tiles (N=256×2) for full D=512 output: **640 cy = 346 ns**. K_DEPTH=4 → 4 × 16 = 64 = B_TOPK (number of KV tokens, the reduction/shared dimension in SV).

### Exp 4 — UTCCP Latency

| UTCCP cols | cy/call |
|-----------|---------|
| 16 cols (K=1) | 168.3 |
| 32 cols (K=2) | 97.2 |
| 64 cols (K=4) | 94.9 |
| 128 cols (K=8) | 86.1 |
| 256 cols (K=16) | 74.7 |

### Exp 5 — Swizzle Impact

INTER, SW32, SW64, SW128 all produce exactly **218.4 cy** at M=64 N=128 K_DEPTH=4. Zero impact — swizzle exploration complete.

### Exp 7 — TMEM ld/st/fence

| Operation | cy |
|-----------|-----|
| TMEM ld (N=1..64, fence only) | 1.79 |
| TMEM st N=1 | 13.0 |
| TMEM st N=4 | 14.6 |
| TMEM st N=8 | 16.2 |
| TMEM st N=16 | 21.6 |
| TMEM st N=32 | 31.6 |
| TMEM st N=64 (competition chunk) | **49.8** |
| fence_view_async_tmem (TMEM ld fence) | 12.8 |
| tcgen05 fence pair (before+after) | 2.83 |
| O-rescale 1 chunk (st N=64 + MMA) | 168.1 |
| O-rescale 4 chunks (competition) | **622.1 cy = 337 ns** |

### Exp 8/9 — Synchronization Overhead

| Operation | cy | ns |
|-----------|-----|-----|
| Named barrier 32 threads | 14.6 | 7.5 |
| Named barrier 128 threads | 20.7 | 10.5 |
| mbarrier roundtrip | **88.5** | **48.0** |
| MMA + commit + wait total | **173.0** | **93.4** |
| commit + wait overhead (derived) | **118.5** | **64.0** |

### Exp 12 — Non-WS TS vs. WS-TS

| Config | WS-TS (cy) | Non-WS TS (cy) | Delta |
|--------|-----------|----------------|-------|
| N=64 K=1 | 54.46 | 54.46 | 0% |
| N=128 K=1 | 54.46 | 64.00 | **+18%** |
| N=128 K=4 | 54.60 | 64.01 | **+17%** |
| N=128 K=32 | 53.72 | 64.00 | **+19%** |

WS is 17-19% faster for N≥128. For N=64 there is no benefit.

---

## Additional Findings (Second Review Pass)

### mbarrier timing decomposition

`tcgen05.commit` overhead isolated: **~30 cy** (173 cy total − 54.5 cy MMA − 88.5 cy mbarrier = 30 cy). The original design-doc estimate for mbarrier was ~50 cy; actual is 88.5 cy (+77% off).

### N_ACC needs larger rotation depth to be conclusive

With N_ACC ≤ 4, the maximum separation between consecutive accesses to the same C[i] is 3 × 11 cy = 33 cy — still below the 43-cy TMEM RAW. N_ACC=8 (8×32=256 TMEM cols, exactly fills the budget) would produce 7 × 11 = 77 cy separation and is the minimum test that could decisively show whether a structural floor exists.

### Non-WS penalty scales with N

N=64: WS = non-WS (54.46 cy each, 32 TMEM cols both modes). N=128: non-WS uses 128 TMEM cols vs WS's 64 → 17–19% slower. Penalty is proportional to the difference in TMEM column footprint, not a fixed overhead.

---

## Competition Kernel Timing Model (Split-KV, Per SM/Block)

*Note: Two reviewer passes produced conflicting conclusions. Both contained errors. This section provides corrected values.*

**Split-KV execution model**: 32 SMs each handle one B_TOPK=64 chunk. Each SM does one local softmax over 64 scores per head (16 heads) — no online rescaling across blocks. Combine kernel merges partial outputs across SMs.

**TMA per block** (all TMA latency values are PER-GATHER4-CALL, not per-block):

| Component | Per-gather4 (ns) | Calls/block | Per-block total (ns) | Source |
|-----------|-----------------|-------------|---------------------|--------|
| ckv gather4 (HBM-cold) | 546 | 16 | ~8,740 (sequential) | tma_gather4 ver6 |
| kpe gather4 (concurrent) | 595 | 16 | ~9,520 (sequential) | dual_tma ver1 |
| ckv with DIST=2 prefetch | 324 | 16 | ~5,184 (prefetched) | tma_gather4 ver6 |

ckv ∥ kpe run in parallel (zero HBM interference verified). Per-block TMA = max(ckv, kpe).
With N=2 pipelining, per-call throughput improves but multi-call pipelining efficiency is unmeasured.
**Per-block TMA estimate: ~5,000–9,500 ns** (depends on pipelining efficiency).

**GEMM per block** (all serialized RAW chains, measured):

| Component | K_DEPTH | Reduction dim | cy | ns | Source |
|-----------|---------|--------------|-----|-----|--------|
| QK ckv (M=64 N=64) | 32 | head_dim_ckv=512 | 1,719 | 928 | Exp 6 (N_ACC=1) |
| QK kpe (M=64 N=64) | 4 | head_dim_kpe=64 | ~218 | ~118 | Exp 1 extrapolated |
| SV tile 1 (M=64 N=256 SS) | 4 | B_TOPK=64 | 320 | 173 | Exp 3 |
| SV tile 2 (M=64 N=256 SS) | 4 | B_TOPK=64 | 320 | 173 | Exp 3 |
| **Total GEMM** | | | **~2,577** | **~1,392** | |

All GEMMs are serialized: K-tiles within each GEMM form RAW dependency chains. QK ckv and kpe use separate accumulators but serialize due to single-thread MMA issue. SV tiles 1+2 are independent but each needs 128 TMEM cols → 256 total = full C budget → must serialize.

**Pipeline structure**: With WS N=64 tiles, GEMM needs all 64 KV tokens loaded before processing → TMA and GEMM are **sequential within one block** (true data dependency at N=64 tile granularity).

| Component | ns | Notes |
|-----------|-----|-------|
| TMA (prefetched, pipelined) | ~5,000–9,500 | 16 gather4s, depends on pipelining |
| GEMM (all serial) | ~1,392 | QK ckv + kpe + 2× SV |
| Softmax (single pass, 64 scores × 16 heads) | ~small | unmeasured |
| **Per-block total (SERIAL baseline)** | **~6,400–10,900** | TMA + GEMM sequential, superseded by ver8 |

**NOTE (2026-03-20 correction):** This timing model is the **serial baseline for a split-KV design with 1 block/SM** only. Two corrections required for proper kernel design:

1. **Pipelining across blocks**: In a multi-block-per-SM design (double-buffered), TMA for block `i+1` overlaps with GEMM for block `i`. Effective per-block time in steady state = `max(TMA, GEMM)` = max(3,690, 1,392) = **3,690 ns** (27% improvement). This requires >1 block per SM and ping-pong smem buffers.

2. **num_tokens=1-2**: This model covers num_tokens=1 only. For num_tokens=2 (64 total blocks), pure split-KV across 64 SMs keeps latency the same (~5,082 ns, all SMs parallel). M=32 tiles could batch 2 tokens at 50% utilization vs 25% for M=64 (unbenchmarked).

**Bottleneck: TMA-BOUND by ~3.6–6.8×** (TMA ~5,000–9,500 ns >> GEMM ~1,392 ns).

**Key unmeasured parameter**: TMA pipelining efficiency for 16 consecutive HBM-cold gather4 calls. This determines where in the 5,000–9,500 ns range the per-block TMA falls.

**Previous errors in this section (corrected above):**
- Reviewer 1 used 595 ns as per-block TMA total (actually per-gather4-call × 16) → wrongly concluded "TMA-bound by 9.5×" (accidentally right direction, wrong magnitude)
- Reviewer 2 applied 28 O-rescales/token from non-split-KV design → wrongly concluded "15,491 ns compute"
- Orchestrator reconciliation used 595 ns as per-block total → wrongly concluded "compute-bound by 2.1×"
- SV GEMM was 173 ns (single tile) — competition needs 2 tiles = 346 ns

---

## Design Decisions Confirmed

1. **Use WS variants**: 18% slower without `.ws` at N≥128
2. **Use SW128 swizzle**: zero performance impact, best layout for L2 TMA fill
3. **QK ckv tile**: M=64, N=64, K_DEPTH=32 (head_dim_ckv=512). 928 ns serialized.
4. **SV tile**: 2 × (M=64, N=256, K_DEPTH=4). 346 ns total (2 tiles, each B_TOPK=64 reduction).
5. **Kernel is TMA-bound**: TMA (~5,000–9,500 ns) >> GEMM (~1,392 ns). Optimize TMA first.
6. **Single-thread accumulator rotation cannot achieve 11-cycle throughput**: warp-specialized pipeline (Exp 11) needed, though impact is limited given TMA dominance.

---

## Next Experiments (Priority Order)

1. **TMA pipelining at competition scale**: Measure per-block TMA time for 16 consecutive HBM-cold gather4 calls with N=2 pipeline + DIST=2 prefetch. This is the biggest unknown in the timing model.
2. **N_acc=3,4,5 at K_DEPTH=32**: Tests the N_ACC≥4 threshold hypothesis. Even if achieved, GEMM drops to ~191 ns — still dwarfed by TMA.
3. **M=32 WS-TS**: Competition-exact tile for 2-token batches (2×16=32 rows). May reduce smem footprint.
4. **TMA+UTCMMA concurrency (Exp 11)**: Tests whether TMA and MMA overlap — relevant if kernel processes multiple B_TOPK chunks per SM (non-split-KV or persistent kernel design).
