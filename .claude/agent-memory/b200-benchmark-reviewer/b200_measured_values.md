---
name: B200 Confirmed Measured Performance Values
description: Empirically confirmed B200 performance values from real Modal hardware runs, with measurement context
type: project
---

Confirmed B200 (sm_100a) performance from tma_gather4 benchmark, ver4 CSV, CUDA 13.0, 148 SMs:

**TMA gather4 latency (single CTA, pipeline N=1)**
- Total: ~225 ns (includes barrier overhead ~28 ns)
- Net (minus barrier): ~197 ns
- Context: 1024-iteration loop, config ckv_int64 (4096B/gather4), random pattern
- CAUTION: This measures L2-resident latency, NOT HBM-miss latency. The 1024 iterations × 4 rows × 4096B = 16MB unique data, all L2-resident after warmup. True HBM-miss TMA latency has not been measured.
- The ~197 ns matches published 420-cycle HBM miss latency coincidentally (TMA pipeline overhead dominates, not memory latency).

**TMA gather4 pipeline N=2 speedup**
- N=1: 225 ns/gather4
- N=2: 172 ns/gather4 (1.31× improvement)
- N=4..32: 167-172 ns/gather4 (diminishing, <3% more)
- Conclusion: N=2 pipeline stages is optimal for single-CTA TMA

**TMA gather4 aggregate throughput (76 CTAs, ckv_int64 4096B/gather4)**
- L2-resident (≤64MB): ~900-904 GB/s
- HBM-bound (256MB+): ~394-409 GB/s (random), ~905 GB/s (competition_realistic — misleading, only 2048 unique rows)
- Single CTA: ~5.75 GB/s HBM-bound

**B200 L2 cache size (empirical)**
- Cliff between 64MB (900 GB/s) and 128MB (546 GB/s)
- L2 size confirmed: ~64-80 MB (not a sharp cliff, set-associative residency)
- Consistent with arxiv:2512.02189 spec of ~64 MB

**TMA box size effect on throughput**
- 256B/gather4 → 292 GB/s; 512B → 334; 1024B → 374; 2048B → 407; 4096B → 409 (HBM-bound, 76 CTAs, 256MB)
- Larger boxes amortize per-barrier overhead. Data type (INT64 vs BF16) irrelevant at same bytes/gather4.

**kpe TMA (512B/gather4, bf16 dim0=64)**
- ~54-55 GB/s HBM-bound at 76 CTAs
- ~9.4 ns/gather4 total latency (single CTA)
- Much slower than ckv (409 GB/s) because smaller box = more barriers per unit data

**Cache hints**
- evict_last vs evict_normal vs none: essentially identical (< 5% difference) at HBM-bound scale
- evict_first: -49% for sequential L2-resident, -12% for random L2-resident. Avoid for cached data.
- Competition recommendation: use evict_last (never worse, best for L2-resident cases)

**Why:** These values are from actual B200 hardware runs on Modal, ver4 of the benchmark. Use these over CLAUDE.md specs which are literature estimates.
**How to apply:** When analyzing future benchmarks or designing the competition kernel, use these as ground truth for B200 TMA performance.
