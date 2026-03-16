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

**Why:** These values are from actual B200 hardware runs on Modal, ver5 of the benchmark. Use these over CLAUDE.md specs which are literature estimates.
**How to apply:** When analyzing future benchmarks or designing the competition kernel, use these as ground truth for B200 TMA performance.
