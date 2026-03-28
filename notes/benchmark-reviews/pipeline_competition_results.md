# TMA Gather4 Pipeline Competition Results — B200 (sm_100a)

**Date**: 2026-03-20
**Benchmark**: microbenchmarks/tma_gather4/
**Results CSV**: microbenchmarks/tma_gather4/results/tma-gather4-ver8.csv
**Experiment**: `pipeline_competition` — 16-consecutive HBM-cold gather4 pipelining

---

## Purpose

Answers the highest-priority open question from the ver6/utcmma-ver3 timing model:

> When 16 consecutive HBM-cold gather4 calls are issued (the exact pattern each SM executes per B_TOPK=64 chunk in the competition kernel), what is the achieved per-call latency? Does the TMA hardware pipeline them in parallel, and if so, to what degree?

The prior timing model bracketed per-block TMA at 5,000–9,500 ns because this was unmeasured. Ver8 closes that gap with a direct measurement at competition-exact parameters.

---

## Experiment Configuration

| Parameter | Value |
|---|---|
| Gather4 type | ckv_int64 (4096 B/gather4, 1024 B/row × 4 rows) |
| Working set | 1024 MB |
| Index pattern | FRESH_COMPETITION_REALISTIC |
| Cache state | ~90% HBM-cold (random permutation, sorted within B_TOPK=64 groups) |
| N values tested | 1, 2, 4, 8, 16 |
| Spread pct | 0.1% across all N (extremely stable) |

FRESH_COMPETITION_REALISTIC mirrors actual competition access: random permutation of all pages, sorted within each B_TOPK=64 chunk. The 1024 MB working set ensures data is well outside the 64 MB L2. SEQUENTIAL was rejected because it would give artificial DRAM row-buffer locality; the competition accesses random pages.

---

## Results

| N outstanding | ns/call | Per-block total (N×ns) | Speedup vs N=1 serial |
|---|---|---|---|
| 1 | 642.4 | 642 | 1.00× (baseline) |
| 2 | 402.9 | 806 | 1.59× |
| 4 | 306.1 | 1,224 | 2.10× |
| 8 | 260.1 | 2,081 | 3.09× |
| **16** | **230.6** | **3,690** | **2.78×** |

"Per-block speedup vs serial" = (16 × 642 ns) / (16 × 230.6 ns) = 2.78×.

---

## Key Findings

### 1. TMA pipelines N=16 outstanding HBM-miss requests (2.78× speedup)

At N=16, per-call latency drops from 642 ns to 231 ns. The TMA engine genuinely pipelines concurrent DMA requests against HBM. Issuing all 16 gather4s before the single mbarrier wait captures a free 2.78× speedup vs issuing them one-at-a-time.

The hardware achieves ~2.78 effective parallel operations at N=16 (not 16-way parallelism). The remaining latency reflects unavoidable HBM row-access overhead that cannot be hidden by pipelining alone.

### 2. Per-call latency saturates at N=8–16

| Step | Per-call reduction |
|---|---|
| N=1 → N=2 | −37% (biggest gain) |
| N=2 → N=4 | −24% |
| N=4 → N=8 | −15% |
| N=8 → N=16 | −11% |

Diminishing returns: at N=16, per-call latency is 36% of N=1. The TMA sustains ~2-3 effective concurrent HBM requests. N=16 is the right pipeline width for the competition kernel (one full B_TOPK=64 block = 16 ckv gather4s).

### 3. Competition N=1 baseline: 642 ns (not 546 ns from ver6)

The ver6 HBM-cold baseline (546 ns) used SEQUENTIAL pattern. With FRESH_COMPETITION_REALISTIC random scatter, it's 642 ns (+17%). The difference is the DRAM open-row miss penalty for random page access. Use 642 ns as the correct competition per-call baseline.

### 4. N=16 HBM-cold ≈ L2-warm latency

Ver6 L2-warm pipeline at N=16: ~200 ns/call. Ver8 HBM-cold competition-exact at N=16: 230.6 ns/call — only 15% higher. TMA pipelining at N=16 nearly collapses HBM-cold latency to the L2-warm floor.

---

## Updated Competition Kernel Timing Model

### TMA per block (N=16 all-at-once issue, ckv and kpe in parallel)

| Stream | Source | Per-block (ns) |
|---|---|---|
| ckv 16 gather4s (HBM-cold, random scatter) | ver8 measured | **3,690** |
| kpe 16 gather4s (runs in parallel, zero interference) | estimated ~symmetric | ~3,680 |
| **Total TMA (max of parallel streams)** | | **~3,690** |

### GEMM per block (from utcmma-ver3, unchanged)

| Component | ns |
|---|---|
| QK ckv (M=64 N=64 K=32) | 928 |
| QK kpe (M=64 N=64 K=4) | 118 |
| SV tile 1 + 2 (M=64 N=256 K=4 each) | 346 |
| **Total GEMM** | **1,392** |

### Summary: before vs after

| Metric | Before (estimate) | After (measured) |
|---|---|---|
| Per-block TMA | 5,000–9,500 ns | **3,690 ns** |
| Per-block GEMM | 1,392 ns | 1,392 ns |
| Per-block total | 6,400–10,900 ns | **~5,082 ns** |
| TMA:GEMM ratio | 3.6–6.8× | **2.65×** |
| N=1 HBM-cold baseline | 546 ns (SEQUENTIAL) | **642 ns** (competition scatter) |

---

## Design Implications

**Mandatory**: issue all 16 ckv gather4 calls before the single mbarrier wait. This is a free 2.78× TMA speedup. The competition kernel should always use this batch-issue pattern.

### Timing Model Scope: Serial vs. Pipelined

The 5,082 ns per-block total is the **serial baseline** for a split-KV SM that handles exactly 1 block. The stages within one block ARE sequentially constrained by true data dependencies:

```
TMA (load 64 KV tokens) → QK_ckv ∥ QK_kpe → softmax → SV
```
- SV cannot start until softmax completes (needs attention weights)
- Softmax cannot start until both QK GEMMs complete
- QK cannot start until all 64 KV tokens are loaded (N=64 tile needs full K matrix)

**Within one block, TMA and GEMM cannot overlap** (no partial-result pipelining at N=64 tile size without adding O-rescale overhead). This is unavoidable for the split-KV design.

**Across blocks (pipelined multi-block-per-SM design)**: issuing TMA for block `i+1` while GEMM runs for block `i` brings effective per-block time toward `max(3,690, 1,392) = 3,690 ns` (27% improvement over 5,082 ns). This is only applicable if the SM handles >1 block. Double-buffering (ping-pong smem) is required.

| Design | Per-block effective | Notes |
|--------|---------------------|-------|
| Split-KV serial (1 block/SM) | **5,082 ns** | baseline, no cross-block pipeline |
| Multi-block pipelined (steady state) | **3,690 ns** | TMA[i+1] overlaps GEMM[i] |
| Within-block sub-tiling (4×16 tokens) | complex | requires O-rescale ×3 ≈ +1,011 ns overhead |

### num_tokens=1 vs num_tokens=2

Competition supports `num_tokens=1-2`. The timing model above is for **num_tokens=1 only**.

With `num_tokens=2` (2 independent query tokens, each with `topk=2048`):
- Total blocks: 2 tokens × 32 blocks/token = **64 blocks**
- With split-KV: 64 SM dispatches — all 64 fit within 148 SMs → **same latency as num_tokens=1** (~5,082 ns, all SMs parallel)
- GEMM M-utilization per dispatch: M=64 with 16 heads/token → 25% (same as single-token case; each SM still handles 1 token's heads)
- **M=32 batching**: if 2 tokens' heads are packed into M=32 per SM (16+16), utilization becomes 100% — but requires both tokens to access the SAME KV pages for that block (different `sparse_indices` → generally impossible to guarantee)
- In practice: treat 2 tokens as 2 independent split-KV launches with `32 SM` reuse

**GEMM M-utilization summary** (affects instruction count, not throughput per se):
| Tile M | heads used | utilization |
|--------|-----------|-------------|
| M=64 | 16 (1 token) | 25% — current model |
| M=32 | 16 (1 token) | 50% — unbenchmarked |
| M=32 | 32 (2 tokens, same KV) | 100% — requires matching sparse_indices |

The current GEMM timing numbers (928 ns QK, 346 ns SV) are measured at M=64. M=32 measurements are needed (see Next Experiments).

**kpe pipeline**: not directly measured at N=16. Follow-up: run `pipeline_competition` with kpe_bf16 config.

---

## Open Questions

1. **kpe N=16 pipeline**: Estimated ~3,680 ns but unverified. Add kpe to `experiment_pipeline_competition`.
2. **TMA+GEMM overlap**: Can GEMM for block i overlap TMA for block i+1? If yes, effective per-block time → max(3,690, 1,392) ≈ 3,690 ns (27% improvement over sequential 5,082 ns).
3. **DIST=2 prefetch + N=16**: Prefetch reduced single-call latency to 324 ns (ver6). Combined with N=16, could push below 200 ns/call. Worth testing but likely marginal on top of the 231 ns already achieved.

---

## References

- `microbenchmarks/tma_gather4/results/tma-gather4-ver6.csv` — L2-warm pipeline and HBM-cold SEQUENTIAL baselines
- `microbenchmarks/tma_gather4/results/tma-gather4-ver8.csv` — this experiment
- `microbenchmarks/dual_tma_stream/results/dual-tma-ver1.csv` — ckv+kpe parallel interference (zero HBM interference confirmed)
- `microbenchmarks/utcmma/results/utcmma-ver3.csv` — GEMM timing model inputs
