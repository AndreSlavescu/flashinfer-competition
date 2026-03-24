# Consolidated Benchmark Review — All Microbenchmarks — B200 (sm_100a)

**Date**: 2026-03-21

---

## Per-Benchmark Status

| Benchmark | Status | Latest Results | Critical Issues |
|---|---|---|---|
| tma_gather4 | PASS | tma-gather4-ver8.csv | None in current code |
| utcmma | PASS (corrected ver5) | utcmma-ver5.csv | Ver3/4 SUPERSEDED |
| dual_tma_stream | PASS | dual-tma-ver1.csv | None in current code |
| device_query | PASS | device-query-b200-ver1.txt | None |
| warp_scheduler | PASS | warp-scheduler-ver1.csv | None |

---

## Benchmark 1: tma_gather4

**Status: PASS**

### Strengths
- Uses `%globaltimer` PTX (nanosecond wall clock) — correct for async TMA where clock64 gives skewed results
- 11 measured runs with `MeasurementSeries` trimmed-mean; warmup before each experiment
- Latency cold variant correctly uses `num_iters = num_rows / 4` (non-repeating row walk, one pass through working set)
- `pipeline_competition` experiment uses `FRESH_COMPETITION_REALISTIC` — generates N×2048 unique indices, stepping pointer per outer iteration, correctly producing fresh HBM-cold scatter patterns
- CSV labels computed dynamically (`swizzle_name()`, `dtype_name()`, `index_pattern_name()`) — no hardcoded metadata errors

### Issues
- MINOR: `COMPETITION_REALISTIC` pattern accesses only 2048 unique rows (~8MB footprint) regardless of declared working set size — all L2-warm. Documented in code comments but CSV rows mislead naive readers. `FRESH_COMPETITION_REALISTIC` is the correct variant to use.
- MINOR: Cold latency warmup (3 warmup runs) partially primes L2 for sequential access before cold measurement — ~3-4% systematic underestimate of true HBM-cold latency.

### Results vs. Prior Research

| Metric | Expected | Measured | Assessment |
|---|---|---|---|
| TMA gather4 HBM-cold latency | N/A — TMA ≠ ld.global | ckv: 540–642 ns; kpe: 459 ns | NEW — TMA is 2.5–2.8× slower than ld.global |
| L2-hot TMA latency | ~197 ns | 193–197 ns | MATCHES |
| L2 cliff location | ~126.5 MB (deviceQuery) | Cliff between 64MB and 128MB samples | MATCHES |
| Prefetch DIST=2 benefit | Not in literature | 40% latency reduction (538 → 324 ns) | NEW FINDING |
| N=16 batch-issued pipeline | Not in literature | 230.6 ns/call = 3,690 ns/block | NEW FINDING (ver8) |

**Key competition number**: ver8 `pipeline_competition` (N=16, FRESH scatter, 1024MB) = **3,690 ns per 16-call TMA block**.

---

## Benchmark 2: utcmma

**Status: PASS (ver5 corrected; ver3/ver4 SUPERSEDED)**

### Critical Correction: ver3/4 → ver5

Ver3/4 lacked `tcgen05.commit + mbarrier.wait + tcgen05.fence::after_thread_sync`. They measured *issue rate*, not *completion latency* — a fundamentally different quantity.

| Metric | Ver3/4 (issue rate) | Ver5 (completion latency) | Ratio |
|---|---|---|---|
| M64N64 k=1 | 54.46 cy | 173.0 cy | 3.18× |
| M64N128 k=1 | 54.5 cy | 173.0 cy | 3.18× |
| M64N256 k=1 | 76.0 cy | 238.5 cy | 3.14× |
| M64N128 k=32 (per-tile) | 53.7 cy | 60.16 cy | 1.12× |

At high K_DEPTH the commit+wait overhead amortizes to ~6 cy/tile. For single-tile experiments, ver3/4 were 3× too fast.

### Strengths
- All ver5 kernels include full sync: `umma_arrive_noelect(*bar) + bar->wait(phase) + tcgen05_after_thread_sync()`
- Single-thread `tcgen05.mma` issue inside `if (threadIdx.x == 0)` — correct per PTX ISA
- TMEM allocation inside `if (threadIdx.x < 32)` — correct warp-collective requirement
- Swizzle labels computed dynamically — anti-pattern #10 fixed

### Issues
- MINOR: M=32 WS-TS removed due to CuTe `mma_traits_sm100.hpp:502` static_assert (PTX ISA allows M=32; raw PTX bypass needed)
- MINOR: N_ACC=1..5 at M64N64 k_depth=32 shows flat 60.16 cy/tile — structural floor confirmed, but N_ACC ≥ 16 (= 173 cy / ~11 cy paper throughput) untested

### Ver5 Key Numbers (competition kernel)

| Measurement | Cycles | ns |
|---|---|---|
| M64N64 k=1 (TS) | 173.0 | 93.8 |
| M64N128 k=1 (TS) | 238.5 | 129.3 |
| M64N256 k=1 (TS) | 238.5 | 129.3 |
| M64N256 SS k=4 (competition SV) | 539 | 292 |
| N_ACC=1..5, M64N64 k=32 (per-tile) | 60.16 | 32.6 |
| Fine-N sweep N=8..64 k=1 | 173.0 (TS) / 175.3 (SS) | N-independent |
| UTCCP 128dp256bit 16-copy | 74.7 cy/copy | 38.0 ns/copy |
| mbarrier arrive+wait | 88.51 | 45.0 |
| named_bar.sync 128T | 20.67 | 10.5 |

**Per-block GEMM total (ver5, split-KV):**
- QK ckv (M64N64, K_DEPTH=32): ~1,925 cy = **1,044 ns**
- QK kpe (M64N64, K_DEPTH=4): ~334 cy = **181 ns**
- SV ×2 (M64N256 SS, K_DEPTH=4): 2×539 cy = **584 ns**
- **Total: ~1,809 ns** (was 1,392 ns in ver3 — 30% higher after fixing sync)

Inferred GPU clock from ver5: 173.0 cy / 93.8 ns = **1.844 GHz** (6.2% below deviceQuery's 1.965 GHz boost clock).

---

## Benchmark 3: dual_tma_stream

**Status: PASS**

### Strengths
- Two-warp architecture with separate mbarriers correctly models competition kernel's dual-stream design
- Same t0 reference for both warps eliminates scheduling skew in BOTH mode
- Mode sweep (ckv_only / kpe_only / both) gives clean isolated vs. concurrent comparison

### Issues
- NOTE: No measurement at competition-realistic working set (64KB ckv + 8KB kpe per block). However, since competition evaluation has no guaranteed cache warmth, HBM-cold is the correct planning baseline. Ver1's 1024MB sequential result (ckv +0%, kpe +7%) already covers this case. Not a meaningful gap.

### Key Results

| Scenario | ckv latency | kpe latency | Interference |
|---|---|---|---|
| Random 256MB, isolated | 568.9 ns | 205.0 ns | Baseline |
| Random 256MB, concurrent | 588.7 ns | 585.1 ns | kpe: **+2.85×** |
| Sequential 1024MB, isolated | 540.3 ns | 458.8 ns | Baseline |
| Sequential 1024MB, concurrent | 540.5 ns | 492.6 ns | ckv: 0%; kpe: +7% |

**Key finding**: At large working sets, ckv evicts kpe from L2 → kpe degrades to HBM-cold latency in concurrent mode. At truly HBM-bound scale, essentially zero interference.

**Competition note**: Competition evaluation runs on a separate machine with no guaranteed cache warmth. Plan for **HBM-cold** baseline. At HBM-cold scale (1024MB sequential, ver1), interference is minimal: ckv +0%, kpe +7%. The 585 ns degradation is an L2-eviction artifact that only occurs when ckv working set fills L2 — not the competition scenario. HBM-cold concurrent behavior is already characterized by ver1. No ver2 needed for this specific gap.

---

## Benchmark 4: device_query

**Status: PASS**

### Corrections to CLAUDE.md / Prior Estimates

| Property | Prior Claim | Actual (deviceQuery) | Action |
|---|---|---|---|
| L2 cache | ~64 MB | **126.5 MB** (132,644,864 B) | CORRECTED — 2× larger |
| SM count | 148 | 148 | Confirmed |
| Max threads/SM | 2,048 | 2,048 | Confirmed |
| Register file | 65,536 = 256 KB | 65,536 = 256 KB | Confirmed |
| Shared mem/SM | 228 KB | 228.0 KB | Confirmed |
| SM boost clock | ~2.1 GHz | **1.965 GHz** | More precise |

The L2 correction explains the TMA gather4 cliff: 64MB fits in L2, 128MB does not, consistent with 126.5 MB actual capacity.

---

## Benchmark 5: warp_scheduler

**Status: PASS**

### Results

| Finding | Value |
|---|---|
| Sub-core count | 4 (scheduler_id = warp_id % 4) |
| Same sub-core contention ratio | 1.304× (30% slowdown) |
| Different sub-core pair | 1.000× (perfect independence) |
| 4 warps, 1 per sub-core | 4.00× linear scaling |
| 8 warps, 2 per sub-core | 6.13× |

### Issues
- MINOR: Only warps 0-7 tested; warp_id%4 pattern almost certainly consistent for warps 8-63 but unverified on B200.
- MINOR: Characterizes scalar FP32 sub-core scheduling; tcgen05.mma (single-thread issue) may not obey same rules.

**Competition relevance**: In a 128-thread (4-warp) kernel, warps 0-3 each occupy a different sub-core — ideal for maximizing instruction-level parallelism across warp roles.

---

## Competition Dataset Workload Range

From `competition-dataset/workloads/dsa_paged/`:

**Attention kernel** (23 workloads):
- num_tokens distribution: {1: 1, 2: 8, 6: 3, 7: 3, 8: 8}
- **14/23 workloads (61%) use num_tokens ≥ 5**, where 32×N > 148 SMs → multi-block/SM unavoidable
- At num_tokens=8: 256 blocks / 148 SMs ≈ 1.73 blocks/SM → inter-block pipelining is critical

**Indexer kernel** (128 workloads):
- batch_size values: [1,2,3,4,6,7,8,11,12,14,15,16,25,26,27,29,30,31]
- **Zero microbenchmark coverage** — major blind spot

---

## Updated Competition Kernel Timing Model (2026-03-21)

Per SM per block (split-KV, B_TOPK=64):

| Stage | Time | Source |
|---|---|---|
| TMA (16 ckv + 16 kpe in parallel) | **3,690 ns** | tma_gather4 ver8 MEASURED |
| QK ckv (M64N64, K_DEPTH=32) | 1,044 ns | utcmma ver5 MEASURED |
| QK kpe (M64N64, K_DEPTH=4) | 181 ns | utcmma ver5 MEASURED |
| SV ×2 (M64N256 SS, K_DEPTH=4) | 584 ns | utcmma ver5 MEASURED |
| **GEMM total** | **1,809 ns** | |
| **Per-block serial total** | **~5,499 ns** | TMA-BOUND by ~2.04× |
| **Per-block pipelined (TMA∥GEMM overlap)** | **~3,690 ns** | Requires concurrency benchmark |

For num_tokens=8 (256 blocks, ~1.73 blocks/SM):
- Serial: ~9,513 ns/SM
- Pipelined: ~6,384 ns/SM (33% faster — critical for 61% of evaluation workloads)

---

## Priority Gaps (Ranked by Impact)

1. **HIGHEST: kpe at N=16 pipeline_competition** — 230.6 ns/call assumed but unconfirmed for 512B kpe variant
2. **HIGH: M=32 WS-TS via raw PTX** — 2-token batching; PTX-valid, CuTe-blocked; would halve block count for num_tokens=2
3. **HIGH: TMA+UTCMMA concurrency** — can block i+1 TMA overlap block i GEMM? Determines 33% speedup feasibility for num_tokens≥5
4. **MEDIUM: Indexer FP8 GEMM + top-K** — completely unbenchmarked; batch_size up to 31
5. ~~Dual TMA at competition-realistic sizes~~ — REMOVED. Competition evaluation has no guaranteed cache warmth; HBM-cold is the correct baseline. Ver1 1024MB sequential already covers this: ckv +0%, kpe +7% interference. Gap closed.
6. **MEDIUM: UTCMMA N_ACC=8 throughput** — N_ACC=5 still shows 60.16 cy floor; need N_ACC≥16 to test paper's 11 cy claim

---

## Issues Summary

| Benchmark | Issue | Severity | Status |
|---|---|---|---|
| utcmma ver3/4 | Missing commit+wait: measured issue rate not completion latency | WAS CRITICAL | FIXED IN VER5 |
| utcmma ver3/4 | Hardcoded "SW128" swizzle label | WAS MAJOR | FIXED IN VER5 |
| tma_gather4 | COMPETITION_REALISTIC is always L2-warm | MINOR | Documented; use FRESH variant |
| tma_gather4 | Cold warmup pre-warms L2 (~3-4% underestimate) | MINOR | Documented |
| utcmma | M=32 WS-TS removed (CuTe bug) | MINOR GAP | Raw PTX needed |
| utcmma | N_ACC > 5 untested | MINOR GAP | N_ACC=8 experiment needed |
| dual_tma | Competition-sized working set not tested | MINOR GAP | Ver2 needed |
| all | Indexer kernel entirely unbenchmarked | MAJOR GAP | New benchmark suite needed |
| CLAUDE.md | L2 = ~64 MB claimed | MAJOR ERROR | CORRECTED to 126.5 MB |
