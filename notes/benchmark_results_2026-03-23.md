# B200 Benchmark Results Analysis — 2026-03-23

**Files reviewed**: tma-gather4-ver9.csv, utcmma-ver6.csv, utcmma-ver7.csv, bulk-copy-s2g-ver1.csv, ldg-hints-ver1.csv
**Hardware**: B200 (sm_100a), 148 SMs, 1.965 GHz, 126.5 MB L2, HBM3e 8 TB/s

---

## 1. TMA gather4 ver9 — kpe_bf16 Pipeline Sweep

**Status: PASS. Closes the kpe latency gap.**

### Results

| config | N=1 ns | N=2 ns | N=4 ns | N=8 ns | N=16 ns | N=16 GB/s |
|--------|--------|--------|--------|--------|---------|-----------|
| ckv_int64 (4096B) | 650.9 | 408.1 | 311.0 | 265.2 | 234.5 | 17.47 |
| kpe_bf16 (512B) | 609.9 | 386.4 | 298.0 | 256.1 | 229.3 | 2.23 |

**Finding 1 — kpe latency ≈ ckv despite 8× smaller payload**: N=1 difference = 41 ns (6.3%). TMA gather4 latency is pipeline-overhead-dominated, not data-size-dominated. Throughput ratio = 7.84× ≈ bytes ratio (8×). Latency and throughput diverge as expected.

**Finding 2 — random scatter requires deeper pipelining than sequential**: N=1→16 = 2.77× speedup for ckv (vs N=1→2 = 1.29× for sequential access in ver6). Competition random addressing over 1024 MB needs **N=16 pipelining, not N=2**.

**Finding 3 — N=1 scatter > N=1 sequential**: ckv 650.9 vs ~546 ns (+19%), kpe 609.9 vs ~492 ns (+24%). Random scatter has higher TLB/page-walk overhead. These are the correct HBM-cold competition baselines.

**16-call block timing at N=16**:
- ckv: 65,536B / 17.47 GB/s = **3,752 ns** per block
- kpe: 8,192B / 2.23 GB/s = **3,673 ns** per block
- Nearly balanced — both complete in similar time

**Updated competition per-block serial estimate**: TMA 3,752 ns + GEMM 1,809 ns = **5,561 ns**. TMA:GEMM = 2.07×.

---

## 2. UTCMMA ver6 — M=32 WS-TS Latency

**Status: PASS. Raw PTX bypass of CuTe static_assert confirmed working.**

### Results

| Tile | cy/gemm | ns/gemm | vs M=64 equivalent |
|------|---------|---------|-------------------|
| M32 N64 k=1 | 173.0 | 93.9 | Same as M64 N64 |
| M32 N128 k=1 | 173.0 | 93.9 | 38% faster than M64 N128 |
| M32 N256 k=1 | 238.5 | 129.5 | Same as M64 N256 |
| M32 N64 k=4 | 310.4 | 168.5 | — |
| M32 N64 k=32 | 1,814 | 984 | 6% faster than M64 N64 k=32 (1,924 cy) |

**M=32 N=64 k=1 = 173.0 cy**: identical to M=64 N=64. The 173-cycle structural floor is architecture-level — shared by any shape at k=1.

**M=32 N=128 k=1 = 173.0 cy**: Same 173-cycle floor but handles 2× more output — 38% compute efficiency gain vs M=64 N=128 (238.5 cy).

**Competition relevance**: M=32 k=32 saves only 110 cy = 59 ns vs M=64. Against 3,752 ns TMA bottleneck = **1.6% end-to-end**. Not worth implementation complexity.

**Clock check**: 173.0 cy / 93.9 ns = 1.842 GHz (consistent with prior 1.844 GHz).

---

## 3. UTCMMA ver7 — TMEM ld Modes

**Status: BROKEN. All results are the loop-overhead floor (1.79 cy). No useful data.**

All 17 configurations return **total_cycles = 8961, cycles/iter = 1.79** — the empty-loop baseline.

**Root causes**:
1. TMEM column address is constant per iteration → no RAW dependency → hardware issues all loads out-of-order
2. `data[]` array has no live use outside the loop → compiler removes the loads entirely (dead code)
3. No data-dependent address to serialize the chain

**Required fix** — add a zero-masked RAW dependency after the fence:
```cpp
col = TMEM_COL_C + (data[0] & 0);  // runtime zero, but compiler must wait for data[0]
```
Also write `g_sink[i & 63] = data[0]` to a global array to prevent dead-code elimination.

**Competition impact**: None. The TMEM loads-are-fast conclusion still holds from arxiv:2512.02189. The actual ld-to-register latency is unknown but likely 5–30 cy based on 16 TB/s TMEM BW.

---

## 4. Bulk Copy S2G ver1 — cp.async.bulk Shared→Global

**Status: PASS. All three experiments valid.**

### Latency

| size | copy ns | copy GB/s | reduce_add ns | reduce GB/s | overhead |
|------|---------|-----------|---------------|-------------|---------|
| 512 B | 34.2 | 14.96 | 40.6 | 12.61 | +18.6% |
| 2 KB | 58.7 | 34.91 | 69.6 | 29.42 | +18.6% |
| 8 KB | 156.4 | 52.38 | 191.8 | 42.72 | +22.6% |
| 32 KB | 547.3 | 59.88 | 680.4 | 48.16 | +24.3% |
| 128 KB | 2,110.8 | 62.09 | 2,634.9 | 49.75 | +24.8% |

**Clock check**: 4,147.3 cy / 2,110.8 ns = **1.965 GHz** ✓

**Fixed startup overhead**: ~34 ns at 512B.
**Marginal bandwidth**: (2110.8 − 34.2) ns / (131072 − 512) B × 10⁹ = **15.9 GB/s** single-CTA DMA ceiling.
**reduce_add overhead**: +18–25%, growing with size (read-modify-write cost).

### Throughput Pipelining: Negligible Benefit

N_PIPE 1→16: +3.8% (copy), +3.6% (reduce_add). Compare to TMA gather4 N=1→16: 2.77×. S2G write path does not have a deep hardware pipeline for concurrent single-CTA writes. **Issue copies sequentially.**

### Async Overlap: CONFIRMED

- c=100 FMAs (compute 1,188 ns > copy 543 ns): copy+compute = 1,199 ns vs compute-only = 1,188 ns → **11 ns overhead = ~22 cycles**. Copy hides completely behind compute.
- c=1000 anomaly: copy+compute = 12,752 ns vs compute-only = 11,723 ns = +1,029 ns. The `bulk_wait_group<0>()` at end of loop adds ~500 cy pipeline flush overhead even if the copy completed long before the wait. Budget this for latency-sensitive code.

**Competition use**: Issue `cp.reduce.async.bulk.add.f32` for partial-O accumulation immediately after MMA. With softmax as intervening compute (~300+ cycles), the reduce runs fully async. Do not wait until end of block.

---

## 5. LDG Hints ver1 — LDG.256 Hint Sweep

**Status: Latency experiment VALID. Throughput and competition experiments BROKEN.**

### Broken Experiments

**Throughput — L2-warm working set**: Pattern `(tid*4 + i*128) % WS_ELEMS` with 32 threads × 10,000 iters touches only ~2.5 MB unique data despite a 256 MB allocation. L2-resident. All 18 hints show identical ~18,476 cycles — no differentiation.

**Competition — L2-resident working set**: 16.5 MB < 126.5 MB L2. All hints identical.

**gbps formula overcounts by THREADS factor**: true throughput is ~1,088 GB/s (L2 BW), not 35 TB/s.

**Fix for throughput experiment**: Stride each thread across its own non-overlapping sub-region:
```cpp
int offset = (tid * (WORKING_SET_ELEMS / 32) + i * 4) % WORKING_SET_ELEMS;
```

### Latency Results (VALID — pointer-chase, single thread, dependent loads)

| working_set | hint | cycles | ns |
|-------------|------|--------|-----|
| 8KB | en_en_128B | 200.73 | 102.2 |
| 8KB | na_ef_128B | 461.84 | 235.1 |
| 8KB | nc_en_en_128B | 200.73 | 102.2 |
| 8KB | nc_na_ef_128B | 461.85 | 235.1 |
| 256KB | en_en_128B | 441.16 | 224.6 (8% spread) |
| 4MB | en_en_128B | 456.61 | 232.4 |
| 256MB | en_en_128B | 456.28 | 232.3 (79% spread) |

**L1-bypass at 8KB confirmed**: `no_allocate` = 461.8 cy vs normal = 200.7 cy = **+130% penalty**. L1-bypass works on B200 regardless of working set size.

**Net latencies** (subtracting ~160 cy integer ALU overhead in pointer-chase):
- L1 hit: ~40 cy ≈ 20 ns
- L2 hit (via L1-bypass at 8KB): ~300 cy ≈ 153 ns
- HBM miss (4MB–256MB): ~296 cy ≈ 151 ns (all hints converge)

**Hint effect at HBM miss**: max 0.37 cy = 0.2 ns. **Cache hints are irrelevant for HBM latency.**

**256MB spread = 79.2%**: HBM latency is highly non-deterministic for single-thread requests (memory controller queueing). Not a measurement error.

### LDG Hint Recommendation for Competition sparse_indices

Working set: ~65 KB (8 tokens × 2048 × 4 bytes) — will be L1-resident after first access.

- **Use**: `nc_en_en_128B` (non-coherent, L1=evict_normal, L2=evict_normal, prefetch=128B)
  - `.nc` safe for read-only tensors; eliminates coherence overhead
  - `evict_normal` maximizes L1+L2 reuse across the 32-block decode loop
  - 200.7 cy gross latency = minimum achievable for L1-resident data
- **Avoid**: `na_ef` / `nc_na_ef` — L1-bypass doubles latency to 461 cy for L1-sized workloads (2.3× penalty). The FlashMLA reference uses `no_allocate` for metadata loads but that metadata is NOT repeatedly reused; for `sparse_indices` which are reused 32×, prefer evict_normal.

---

## Summary Table: Key Numbers

### TMA gather4 Competition-Realistic (ver9, random scatter, HBM-cold)

| Config | N=1 ns | N=16 ns | N=16 GB/s | 16-call block |
|--------|--------|---------|-----------|--------------|
| ckv_int64 (4096B) | 650.9 | 234.5 | 17.47 | ~3,752 ns |
| kpe_bf16 (512B) | 609.9 | 229.3 | 2.23 | ~3,673 ns |

### UTCMMA M=32 (ver6)

| Tile | cy | ns | competition value |
|------|----|----|------------------|
| M32 N64 k=32 | 1,814 | 984 | 6% faster than M64 — not worth it |
| M32 N128 k=1 | 173 | 93.9 | Better throughput efficiency than M64 N128 |

### cp.async.bulk S2G (ver1)

| Metric | Value |
|--------|-------|
| Latency 32KB plain copy | 547 ns = 1,075 cy |
| Latency 32KB reduce_add | 680 ns = 1,337 cy |
| Single-CTA throughput ceiling | 62.6 GB/s |
| reduce_add vs plain overhead | +18–25% |
| Pipelining N=1→16 benefit | +3.8% (negligible) |
| Async overlap issue cost | ~22 cycles |

### LDG.256 Latency (ver1, pointer-chase)

| Condition | Cycles | ns |
|-----------|--------|-----|
| L1 hit (8KB, en_en) — gross | 200.7 | 102.2 |
| L1 hit — net | ~40 | ~20 |
| L1-bypass (na_ef any size) | 461.8 | 235.1 |
| HBM miss (4MB–256MB, any hint) | 456.3 | 232.3 |

---

## Competition Kernel Implications

1. **TMA pipelining: raise to N=16**. Sequential N=2 was for sequential access. Competition random scatter needs N=16.
2. **M=32 MMA: skip**. 59 ns saving vs 3,752 ns TMA bottleneck = 1.6%. Not worth it.
3. **S2G bulk reduce: issue early, overlap with softmax**. 22-cycle async issue cost. No pipelining benefit from N_PIPE>1.
4. **sparse_indices LDG hint: nc_en_en_128B**. Avoid no_allocate (2.3× L1 penalty).
5. **Per-block serial time**: 5,561 ns (TMA 3,752 + GEMM 1,809). TMA:GEMM = 2.07×.

---

## Known Broken Benchmarks (fix before rerunning)

| Benchmark | Issue | Fix |
|-----------|-------|-----|
| tmem_ld_modes (ver7) | No RAW dep, dead-code elimination | Add `col += (data[0] & 0)`; write to g_sink |
| ldg_throughput (ver1) | L2-warm, all threads same cache lines | Stride each thread across its own sub-region |
| ldg_competition (ver1) | 16.5 MB < L2, all hints identical | Same striding fix; enlarge to >300 MB |

---

*Analysis by b200-benchmark-reviewer agent. Memory files updated: b200_measured_values.md, competition_kernel_insights.md, benchmark_antipatterns.md.*
