# TMA Gather4 Improvements Review — B200 (sm_100a)

**Date**: 2026-03-16
**Benchmark**: `microbenchmarks/tma_gather4/`
**Results CSV**: `microbenchmarks/tma_gather4/results/tma-gather4-ver6.csv`
**Prior review**: `notes/tma_gather4_review_v2.md`

---

## Summary

All five improvement categories are functionally correct and produce valid, interpretable data in
ver6. There are two minor documentation bugs and one known implementation quirk (DIST=0 prefetch
kernel), none of which invalidate any measurements. Key new findings: prefetch effectiveness
saturates at ~40% latency reduction at DIST=2 (not DIST=4); the 32-block L2 sweep shows
competition-realistic throughput of only ~174 GB/s at HBM-bound sizes; the FRESH_COMPETITION_REALISTIC
pattern confirms the real competition scenario runs at ~386 GB/s (not the ~906 GB/s seen for the
L2-warm competition_realistic pattern).

---

## Part A: Experiment Expansions in main.cu

### A1: kpe config in cache hint sweep (Exp 2) — CORRECT

The outer `configs` loop over `{config_ckv_int64(), config_kpe_bf16()}` is wired correctly. Both
configs share the same experiment name strings (`throughput_hints_Xmb`), which is correct — the
`config` column differentiates them in the visualizer. At 256MB, kpe has 524288 iters vs ckv's
65536 (8× more, since kpe rows are 8× smaller), which improves statistical quality.

**New finding from kpe data**: At 64MB (the L2 boundary), `evict_first + sequential` drops kpe
throughput to **65 GB/s — a 47% penalty** vs evict_last (122 GB/s). Root cause: sequential
access + evict_first at L2 capacity creates self-evicting pathology. At HBM-bound sizes (256MB+),
evict_first is marginally **+5% faster** than evict_last.

**Competition recommendation**: Use `evict_first` for kpe_cache TMA hint (frees L2 for ckv_cache
while slightly improving HBM throughput). Use `evict_last` for ckv_cache.

### A2: 32-block L2 sweep (Exp 5) — CORRECT

Correctly adds `throughput_l2_32blk` experiment with `num_blocks=32`.

| Working Set | 76-block BW | 32-block BW | Ratio |
|---|---|---|---|
| 64MB (L2-hot) | 906 GB/s | 393 GB/s | 2.31× |
| 128MB (cliff) | 541 GB/s | 232 GB/s | 2.33× |
| 256MB (HBM) | 403 GB/s | 174 GB/s | 2.32× |
| 1024MB (HBM) | 390 GB/s | 166 GB/s | 2.35× |

**Critical competition finding**: At 32 CTAs (one per split-KV block), HBM-bound throughput is
**~166 GB/s**, not 390 GB/s. The L2 cliff location is identical to 76 blocks (64→128MB). For
the competition's 8MB KV working set per token (fits in L2 at ~64MB), expected bandwidth is
~393 GB/s at 32 CTAs — but only if the entire 8MB stays resident during computation.

### A3: kpe + competition_realistic_pt in pipeline experiment (Exp 7) — CORRECT

All 24 rows present (4 config×pattern combinations × 6 N values). Key finding: **kpe and ckv
pipeline curves are nearly identical**:

| Config | N=1 latency | N=2 latency | Saturation point |
|---|---|---|---|
| ckv_int64 + RANDOM | 222.5 ns | 172.0 ns | N=2 |
| kpe_bf16 + RANDOM | 215.6 ns | 170.6 ns | N=2 |
| ckv_int64 + comp_pt | 221.2 ns | 171.8 ns | N=2 |
| kpe_bf16 + comp_pt | 208.3 ns | 167.8 ns | N=2 |

Both configs saturate at **N=2** with a ~170 ns latency floor. The bottleneck is mbarrier
signaling overhead, not per-byte transfer time. Competition kernel should use N=2 pipeline
depth for both streams.

### A4: competition_realistic_pt in latency experiment (Exp 6) — CORRECT, one doc bug

New latency entries:
- ckv_int64 + comp_pt: 221.5 ns total, **193.2 ns net** (vs 221.9/193.7 ns for RANDOM — within 0.3%)
- kpe_bf16 + comp_pt: 208.0 ns total, **179.8 ns net** (vs 215.2/186.9 ns for RANDOM — within 3.8%)

Page-table access pattern adds zero measurable latency in the L2-warm regime.

**MINOR BUG**: Comment at `main.cu:641` says "Still shows L2-warm latency (~197 ns)" — references
the ver4 measured value. Ver5/ver6 net latency is ~193-194 ns for ckv. Documentation error only;
measurement is correct.

---

## Part B: TMA Prefetch Effectiveness

### B1: tma_gather4_prefetch() PTX wrapper — CORRECT

```ptx
cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint
  [%0, {%1, %2, %3, %4, %5}], %6;
```

Correctly omits destination smem address, mbarrier, `mbarrier::complete_tx::bytes`, and
`cta_group::1` (none required for prefetch-only). Fire-and-forget semantics are correct.

### B2: prefetch_kernel<DIST> — CORRECT, one known quirk

The sliding-window structure is correctly implemented. Prolog issues DIST prefetches before
timing starts; the main loop issues prefetch for row `i+DIST` while waiting on row `i`.

**DIST=0 quirk**: When DIST=0, condition `if (i + DIST < num_iters)` becomes `if (i < num_iters)`
— always true. So `prefetch_kernel<0>` issues `tma_gather4_prefetch` for the same row already
in flight via `tma_gather4`. Effect: measured DIST=0 latency is **537.7 ns** vs **543.9 ns**
for `latency_cold` — a 1.2% difference. Hardware appears to no-op same-address in-flight prefetch.
Comparison is valid at the ~1% level. An `if constexpr (DIST > 0)` guard would make code cleaner.

**Prefetch results** (HBM-cold, sequential, 1024MB):

| DIST | ckv latency (ns) | vs baseline | kpe latency (ns) | vs baseline |
|------|-----------------|------------|-----------------|------------|
| 0 (baseline) | 537.7 | — | — | — |
| 1 | ~380 | -29% | — | — |
| 2 | ~324 | -40% | — | — |
| 4 | ~324 | -40% | — | — |

Prefetch saturates at DIST=2 (~40% reduction). Going to DIST=4 provides no additional benefit,
meaning 2 outstanding requests fill the TMA fetch pipeline.

**Competition implication**: Issue ckv prefetch for block `i+2` while computing QK^T on block `i`.
Expected savings: 40% of 548 ns per block × 16 gather4 calls × 32 blocks = ~112 µs off per token
(from estimated 274 µs → ~162 µs for pure TMA time).

---

## Part C: Fresh Competition Scatter

### C1: FRESH_COMPETITION_REALISTIC pattern — CORRECT

Correctly generates a full permutation of `num_rows` integers, shuffles, then sorts within each
B_TOPK=64 block. For ckv_int64 at 1024MB: num_rows = 1,048,576, divisible by 64 — no edge case.

### C2: experiment_fresh() — CORRECT

Result at 1024MB:

| Pattern | 1024MB BW | Notes |
|---------|----------|-------|
| competition_realistic | 905.54 GB/s | 8MB unique footprint → always L2-warm |
| random | 389.95 GB/s | HBM-bound |
| **FRESH_competition_realistic** | **385.73 GB/s** | HBM-bound + competition structure |

FRESH is within **1.1% of pure RANDOM** — the B_TOPK=64 sort structure neither helps nor hurts
in the HBM-bound regime. Confirms the competition kernel will see ~386 GB/s at first access.
The prior "906 GB/s" figure from competition_realistic was L2-warm artifact.

---

## Part D: Comparison Against Prior Research

| Metric | Expected (arxiv:2512.02189) | Measured (ver6) | Assessment |
|---|---|---|---|
| L2 cache size | ~64 MB | 64-128MB cliff confirmed | MATCHES |
| Global mem latency | ~420 cycles / ~200 ns | 193 ns net (L2-warm) | MATCHES |
| HBM-cold TMA latency | 420 cycles + DMA overhead | 543 ns net | EXPECTED |
| Pipeline saturation | Unknown for B200 TMA | N=2 for both ckv and kpe | NEW DATA |
| Prefetch effectiveness | Unknown | 40% max at DIST≥2 | NEW DATA |
| Competition scenario BW | Unknown | 386 GB/s (fresh, 76 blocks) | NEW DATA — 2.35× slower than L2-warm estimate |

---

## Issues Summary

| ID | Severity | Description |
|---|---|---|
| B2-quirk | MINOR | `prefetch_kernel<0>` issues spurious same-row prefetch; 1.2% effect on baseline |
| A4-comment | MINOR | `main.cu:641` references outdated "~197 ns"; actual ver6 value is ~193-194 ns |

No CRITICAL or MAJOR issues. All experiments produce valid, interpretable data.

---

## Implications for Competition Kernel

**Cache policy**:
- ckv_cache: `evict_last` — retains 8MB KV in L2 during computation
- kpe_cache: `evict_first` — +5% HBM throughput, frees L2 for ckv

**Pipeline depth**: N=2 optimal for both ckv and kpe TMA streams. N=1 wastes ~50 ns per gather4.
N>2 gives no additional benefit in L2-warm conditions.

**Prefetch**: DIST=2 achieves ~40% HBM-cold latency reduction (324 ns vs 538 ns). Issue ckv
prefetch for block `i+2` while computing QK^T on block `i`.

**Bandwidth at competition scale**: ~386 GB/s (fresh/cold, 76 blocks), ~166 GB/s (32 CTAs,
HBM-bound). If KV stays L2-warm in 8MB budget: 393 GB/s at 32 CTAs.
