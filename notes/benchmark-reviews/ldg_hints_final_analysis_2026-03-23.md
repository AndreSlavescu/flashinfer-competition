# LDG.256 Hints Final Analysis — B200 (sm_100a)

**Date**: 2026-03-23
**Final canonical dataset**: `microbenchmarks/ldg_hints/results/ldg-hints-ver4.csv`
**Pchase sweep (authoritative)**: `microbenchmarks/ldg_hints/results/ldg-pchase-sweep-ver1.csv`

---

## Summary

All four LDG.256 experiments were run with fully corrected code and saved as `ldg-hints-ver4.csv`:

| Experiment | Rows | Status |
|------------|------|--------|
| ldg_throughput | 18 | Correct — sequential streaming, prefetcher-dominated |
| ldg_competition | 10 | Correct — L2-resident, confirms hint-irrelevance for competition |
| ldg_latency | 24 | Correct for 8KB/256KB/4MB; 256MB is partial probe (see caveat) |
| ldg_pchase_sweep | 57 | Authoritative B200 memory hierarchy staircase |

---

## Benchmark Version History

| File | Contents | Status |
|------|----------|--------|
| ldg-hints-ver1.csv | All experiments, OLD latency kernel (ALU overhead, +164 cy) | INVALID for latency |
| ldg-hints-ver2.csv | Correct throughput ITERS, correct gbps formula; OLD latency kernel | INVALID for latency |
| ldg-hints-ver3.csv | Absolute-pointer latency kernel; WRONG pchase_sweep iters | INVALID for pchase_sweep |
| ldg-pchase-sweep-ver1.csv | Correct pchase_sweep (iters=max(num_slots,20000)) | VALID |
| **ldg-hints-ver4.csv** | **All experiments with all fixes applied** | **CANONICAL** |

---

## Finding 1 — Throughput (sequential streaming)

All 18 hints identical at **~1,070 GB/s** for sequential 256 MB buffer access.

Root cause: sequential access pattern activates hardware prefetcher regardless of hint. The prefetcher brings L2/HBM lines ahead of loads, masking latency differences. Hint differentiation is NOT possible via sequential streaming — this is correct hardware behavior.

**Implication for competition**: HBM throughput hint comparisons require random-access patterns (p-chase) or strided patterns wider than L2 prefetch window. For competition sparse_indices (sequential within a block), any hint delivers the same throughput.

---

## Finding 2 — Competition Index Loads (L2-resident)

All 10 tested hints identical at **~1,064–1,070 GB/s** for competition scatter pattern (16.5 MB working set).

Root cause: competition working set fits comfortably in L2 (16.5 MB << 126.5 MB). All accesses are L2-warm after warmup. No hint can differentiate L2-hit bandwidth.

**Implication**: For competition inference (fixed sparse_indices across evaluate iterations, L2-warm), hint choice is irrelevant. For cold-start (first inference, L2-cold), hint also irrelevant since data is L2-resident by the time the compute loop runs. **Any hint is fine for sparse_indices.**

---

## Finding 3 — P-Chase Memory Hierarchy Staircase

Full B200 memory hierarchy measured via absolute-pointer Hamiltonian p-chase (LDG.256, en_en_128B):

| Working Set | Cycles | ns | Cache Level |
|-------------|--------|-----|-------------|
| 4 KB | 36.0 | 18.4 | **L1 hit** |
| 8 KB | 36.8 | 18.8 | L1 hit |
| 16 KB | 40.1 | 20.5 | L1 spilling |
| 32 KB | 46.9 | 23.9 | L1/L2 mix |
| 64 KB | 60.3 | 30.7 | L1→L2 transition |
| 128 KB | 87.2 | 44.5 | L1→L2 transition |
| 256 KB | 257 | 131 | L2 partial |
| 512 KB | 299 | 153 | **L2 steady-state** |
| 1 MB–32 MB | ~300 | ~153 | L2 plateau |
| 64 MB | 331 | 169 | L2 capacity pressure (L2=126.5MB) |
| 128 MB | 527 | 269 | L2→HBM transition |
| 192 MB | 640 | 326 | HBM |
| 256 MB | **689–707** | **351–360** | **HBM deep cold** |

**Clock**: 1.961–1.964 GHz confirmed via globaltimer. The 689 vs 707 cy range is run-to-run variation across Modal B200 instances (~2.5%).

**L1 boundary**: ~32–48 KB (random-access). Above this, L2 hit.
**L2 boundary**: ~64–128 MB (random-access). Above this, HBM.
**L2 size confirmed**: 126.5 MB (capacity pressure starts at 64 MB, full HBM at 128+ MB).

---

## Finding 4 — L1-Bypass Penalty (na_ef vs en_en)

| Working Set | en_en cy | na_ef cy | Penalty |
|-------------|----------|----------|---------|
| 4 KB | 36.0 | 278.5 | **7.7× L1-bypass** |
| 8 KB | 36.8 | 298.6 | **8.1× L1-bypass** |
| 32 KB | 46.9 | 278.0 | **5.9×** |
| 512 KB+ | ~299 | ~299 | **1.0× (identical in L2)** |
| 256 MB | 689 | 691 | **1.0× (identical in HBM)** |

**L1-bypass only penalizes when WS fits in L1 (≤ ~32 KB)**. At L2 or HBM, na_ef and en_en perform identically.

**Competition implication**: sparse_indices per block = 64 tokens × 4 bytes = 256 bytes → pure L1 (36 cy). Never use na_ef for index loads; use en_en_128B.

---

## Finding 5 — Non-Coherent (.nc) vs Coherent

`.nc` flag shows zero performance difference at ALL working set sizes (L1, L2, HBM). The `.nc` qualifier does not change latency for LDG.256 random-access on B200.

---

## Latency Experiment Caveat (256MB point)

The `ldg_latency` experiment at 256MB uses 500,000 iters (6% of the 8M-slot ring):
- 500K × 32B = 16MB of unique slots visited
- 16MB fits in L2 → measures partial L2→HBM mix regime
- **Measured**: 406 cy (intermediate, not true HBM)
- **True HBM cold**: 689–707 cy (from pchase_sweep with full 8M iters)

The pchase_sweep 256MB row is the authoritative HBM measurement. The 256MB latency row is a sanity check showing the L2-to-HBM transition region.

---

## Cross-Reference with Prior Work (Updated)

| Source | Metric | Their Value | Our Measurement | Notes |
|--------|--------|-------------|-----------------|-------|
| arxiv:2512.02189 (B200) | Global mem latency | ~420 cy / ~200 ns @ 2.1 GHz | **689–707 cy / 351–360 ns @ 1.965 GHz** | Their value likely L2 (300 cy), not HBM |
| arxiv:2507.10789 (GB203) | L1 latency | 30–40 cy | **36 cy ✓** | Exact match |
| arxiv:2507.10789 (GB203) | HBM latency | ~877 cy | 689–707 cy | B200 HBM faster than RTX 5080 |

---

## LDG.256 Hint Recommendations for Competition

| Use Case | WS | Recommendation | Reason |
|----------|-----|----------------|--------|
| sparse_indices per block (64 tokens) | 256 B | `en_en_128B` | L1 hit (36 cy); na_ef = 279 cy (7.7× penalty) |
| Full topk=2048 indices | 8 KB | `en_en_128B` | L1 hit (36 cy) |
| **NEVER** | any | `na_ef_*` for indices | L1-bypass; 5–8× penalty at L1-fittable sizes |
| Output store (large, cold) | MB-scale | Any — benchmark needed | Hints identical in HBM random-access; sequential BW unaffected |

---

## Methodology Notes (arxiv:1509.02308 compliance)

1. **Absolute-pointer p-chase**: `ptr = (const uint64_t*)buf[0]` — zero ALU overhead. Old approach had ~164 cy of `%` and `& ~31ULL` masking overhead.
2. **iters ≥ num_slots**: `iters = max(num_slots, 20000)` ensures full Hamiltonian ring traversal. Fewer iters visits only a cached subset, giving L1/L2 latency regardless of declared WS.
3. **Hamiltonian cycle**: Random permutation visits every slot exactly once per pass — no hot spots, defeats sequential prefetcher.
4. **Warmup = 200 iters**: Warms 6.4 KB. For WS >> 6.4 KB, timing loop is cold.
