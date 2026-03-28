# TMA Gather4 Microbenchmark Review v2 — B200 (sm_100a)

**Date**: 2026-03-16
**Benchmark version**: ver5 (after 6 improvements from v1 review)
**Agents**: b200-benchmark-reviewer + ptx-benchmark-explorer

---

## Part 1: Code Correctness Verification (6 Improvements)

All 6 improvements from the v1 review were correctly implemented:

| # | Change | Status |
|---|---|---|
| 1 | HBM-cold latency: `num_iters=0` sentinel → `num_rows/4`, 1024MB working set | ✅ Correct |
| 2 | `COMPETITION_REALISTIC_PAGE_TABLE` pattern with block_table indirection | ✅ Correct |
| 3 | Renamed `ms_*` → `ns_*` in `run_latency()` | ✅ Correct |
| 4 | PTX spec comment on `expect_tx`-after-gather4 ordering | ✅ Correct |
| 5 | `config_ckv_int64_l2_64b()` with `CU_TENSOR_MAP_L2_PROMOTION_L2_64B` | ✅ Correct |
| 6 | Saturation sweep extended to 296 CTAs (148 SMs × 2) | ✅ Correct |

**HBM-cold implementation correctness**: With `num_iters = num_rows/4` and SEQUENTIAL pattern,
the latency kernel accesses rows `{4i, 4i+1, 4i+2, 4i+3}` for i=0..num_iters-1. Every row is
accessed exactly once. For 1024MB ckv_int64: `num_rows = 262144`, `num_iters = 65536`. Total
unique data = 262144 × 4096B = 1024MB >> 64MB L2. True cold measurement confirmed.

---

## Part 2: Ver5 Results vs Prior Research

### 2.1 True HBM-Cold TMA Latency (NEW — biggest methodological improvement)

| Config | L2-hot latency (ver4) | HBM-cold latency (ver5) | Ratio |
|---|---|---|---|
| ckv_int64 (4096B/gather4) | ~197 ns net | **~546 ns net** | **2.77×** |
| kpe_bf16 (512B/gather4) | ~197 ns net | **~492 ns net** | **2.50×** |

**Interpretation**:
- 546 ns ÷ (1/2.1 GHz) ≈ **1147 cycles** for ckv cold latency
- Previously measured 197 ns was TMA pipeline setup overhead (constant), not HBM memory latency
- True HBM-miss component: 546 - 197 = ~349 ns ≈ 733 cycles
- Published B200 HBM latency: ~420 cycles global load (arxiv:2512.02189)
- TMA overhead adds ~300 cycles on top of raw HBM latency (DMA setup, descriptor fetch, mbarrier completion token)

**Competition timing impact**:
- If KV working set is HBM-cold: each block (B_TOPK=64 tokens) costs ~546 ns × 16 gather4 calls ≈ 8.7 µs ckv TMA per block
- 32 blocks per token → ~278 µs pure TMA time (serial, no pipelining)
- With N=2 pipelining: ~139 µs
- With L2-resident data (~197 ns): ~32 µs → **4.4× faster if L2-resident**
- This makes L2 residency the #1 priority in the competition kernel

### 2.2 Extended Saturation Sweep (NEW — no saturation found)

| CTAs | BW (GB/s) | % of 8 TB/s HBM peak |
|---|---|---|
| 76 (baseline) | 409 | 5.1% |
| 96 | 518 | 6.5% |
| 128 | 690 | 8.6% |
| 152 (old max) | 796 | 9.9% |
| 176 | 905 | 11.3% |
| 192 | 990 | 12.4% |
| 224 | 1151 | 14.4% |
| 256 | 1309 | 16.4% |
| **296** | **1511** | **18.9%** |

**Finding**: TMA gather4 BW scales **near-linearly** across all 296 CTAs with no saturation.
At 296 CTAs (148 SMs × 2 CTAs each), bandwidth is still only **18.9% of HBM peak**.

**Why no saturation?** Each CTA uses a single thread issuing gather4 serially — one outstanding
TMA request at a time. Per-CTA throughput: ~5.1 GB/s. Per-SM effective BW: ~10.2 GB/s.
B200 has ~54 GB/s per SM available (8 TB/s / 148 SMs). **Single-warp-per-CTA TMA cannot
saturate HBM regardless of SM count.** The competition kernel needs multiple concurrent
TMA-issuing warps per SM, or the memory access pattern naturally saturates via many CTAs.

**For the competition kernel**: At 32 CTAs (1 per split, 32 splits for topk=2048),
expected TMA BW ≈ 32 × 5.1 = 163 GB/s aggregate. To access 2048 tokens × 1024B = 2MB:
2MB / 163 GB/s = 12.3 µs (HBM-bound). This agrees with the earlier timing estimate.

### 2.3 L2 Promotion 64B vs 128B

Both sweeps (`throughput_l2` and `throughput_l2_64b`) show **identical performance** at all
working set sizes. The L2 cliff location (between 64MB and 128MB) is unchanged. **Conclusion**:
L2 promotion sector size has no measurable impact on gather4 bandwidth. The hardware likely
upcasts to 128B sectors internally regardless of the tensor map setting. Use `L2_128B`
(the default) for all configs.

### 2.4 competition_realistic_pt Pattern

`competition_realistic_pt` shows ~896-908 GB/s at 256MB — **identical to `competition_realistic`
(~905 GB/s)**. Both patterns have only 2048 unique rows × 4096B = 8MB unique data footprint,
regardless of the physical frame mapping. L2 warming after 3 warmup runs covers all 8MB.

**Conclusion**: The PAGE_TABLE pattern correctly models address scatter, but the 8MB unique
footprint means both patterns hit L2 identically. Neither pattern can test HBM-cold page-table
scatter behavior. To properly benchmark page-table scatter, a pattern with `num_unique_rows >
L2_capacity / bytes_per_row = 64MB / 4096B = 16384` rows (i.e., fresh topk per iteration) is
needed.

---

## Part 3: Remaining Knowledge Gaps

### Gap 1: Dual TMA Stream Interference (CRITICAL)
The competition kernel runs ckv (warp 5) and kpe (warp 6) TMA streams simultaneously within
one CTA. Whether B200's TMA hardware serializes or overlaps two concurrent streams is unknown.
If streams serialize: effective kpe BW = 55 GB/s × (1 / (1 + ckv_share)) — could drop below 30 GB/s.
If they overlap: combined throughput approaches sum of individual streams.

### Gap 2: True Competition Scatter Pattern with Fresh Indices
Both `competition_realistic` and `competition_realistic_pt` reuse the same 2048 rows across all
outer iterations. The real competition issues different 2048-row sets per query token. With
num_pages=8462 and random page selection, true cold scatter behavior requires generating
fresh random indices every outer iteration (on device) rather than cycling a pre-generated pool.

### Gap 3: UTCMMA Benchmark (zero data)
No measurement of `tcgen05.mma` throughput or latency exists. The competition QK^T tile is
M=64, N=128, K=576 (36 K-tiles of 16). The TMEM constraint (512 cols/SM) limits occupancy to
1 CTA/SM. `.ws` warp-specialization overhead is unknown.

### Gap 4: TMA Gather4 + UTCMMA Overlap
All measurements are TMA-only (compute stalled waiting on mbarrier). In the actual kernel,
UTCMMA runs while TMA fetches the next block. The achievable TMA bandwidth while UTCMMA
executes simultaneously is not measured.

### Gap 5: TMA Prefetch Effectiveness for Cold Scatter
`tma_gather4_prefetch()` (from `sm100/intrinsics.cuh:26`) fires a prefetch into L2 with no
mbarrier cost. Whether B200's L2 prefetch mechanism can hide 546 ns of cold TMA latency for
random scatter patterns is untested.

---

## Part 4: Prioritized Next Experiments

### Priority 1: Dual TMA Stream Benchmark
**Why critical**: Exact competition kernel execution pattern (warp5=ckv + warp6=kpe concurrent).
A 20% interference penalty translates directly to 20% longer decode time.

**Implementation** (effort=2):
- Single CTA, 2 warps
- Warp 0: issues ckv-style gather4 (dim0=128 INT64, 4096B/step, SEQUENTIAL/RANDOM)
- Warp 1: issues kpe-style gather4 (dim0=64 BF16, 512B/step, same pattern)
- Separate mbarriers per warp; both warps time their own stream independently
- Sweep: ckv-only, kpe-only, both concurrent
- Metrics: per-stream throughput, aggregate throughput, ratio vs sum-of-individuals
- Expected finding: near-zero interference (TMA hardware likely has multiple issue slots), or
  ~20-40% serialization penalty if single-issue

**New file**: `microbenchmarks/dual_tma_stream/`

---

### Priority 2: TMA Gather4 Prefetch Effectiveness
**Why critical**: At 546 ns cold TMA latency and 32 blocks per token, prefetching block N+1
while computing on block N could reduce total token latency by up to 50%.

**Implementation** (effort=1 — wrapper already exists):
- Add to existing `tma_gather4` benchmark as a new experiment type
- Baseline: sequential gather4 with no prefetch (current pipeline)
- Prefetch: issue `tma_gather4_prefetch()` for rows[i+1] before waiting on rows[i]'s mbarrier
- Sweep: prefetch distance = 0 (none), 1, 2, 4 blocks ahead
- Use cold access pattern (SEQUENTIAL, 1024MB, num_iters=0) to see actual HBM latency hiding
- Also test with `EVICT_LAST` vs `EVICT_FIRST` on the prefetch hint

**New file**: extend `microbenchmarks/tma_gather4/gather4_kernels.cuh` + `main.cu`

---

### Priority 3: UTCMMA Throughput and Tile Shape Sweep
**Why critical**: Zero microbenchmark data for `tcgen05.mma` at competition dimensions.
The competition QK tile (M=64, N=128) and SV tile (M=64, N=256) are unmeasured.

**Implementation** (effort=3):
- New `microbenchmarks/utcmma/` directory
- Kernel: elected MMA warp issues `tcgen05.mma.ws.cta_group::1.kind::f16` in a tight loop
- Sweep M×N: M∈{32,64,128} × N∈{64,128,256} at K=16
- K-depth sweep: 1..32 K-tiles to find pipelining behavior
- Compare WS_TS vs TS_NOELECT: does `.ws` add overhead vs non-warp-specialized?
- SMEM swizzle sweep: SW128 vs SW64 vs SWIZZLE_NONE for B-matrix
- Validate: ~11 cycle latency per MMA from arxiv:2512.02189 should appear at all tile sizes

**New file**: `microbenchmarks/utcmma/`

---

### Priority 4: TMEM Load/Store Bandwidth
**Why critical**: The softmax P-extraction and O-rescaling loops in the competition kernel
repeatedly read/write TMEM. True throughput at realistic N values for `tcgen05.ld/st.32dp32bNx`
is unmeasured.

**Implementation** (effort=2):
- New `microbenchmarks/tmem_bandwidth/` or extend `utcmma/`
- Sweep N∈{1,2,4,8,16,32,64,128} for both `tmem_ld_32dp32bNx` and `tmem_st_32dp32bNx`
- Measure bandwidth vs theoretical 16 TB/s (rd) / 8 TB/s (wr)
- Fence overhead: bare `fence_view_async_tmem_load` cost
- `tcgen05.fence::before_thread_sync` + `__syncthreads()` + `after_thread_sync` roundtrip

---

### Priority 5: mbarrier Chain Overhead
**Why critical**: The competition kernel has 18+ mbarriers. At 6 barrier pairs per block ×
32 blocks per token = 192 barrier roundtrips per query. At 50 cycles each = 9600 cycles overhead.

**Implementation** (effort=1):
- Pure synchronization kernel, no data movement
- Single mbarrier: `expect_tx(0)` → `arrive` → `try_wait` roundtrip latency
- Chain of N barriers in series: N∈{1,2,4,8,16,18}
- Impact of arrive count: `init(1)` vs `init(128)` barrier wait latency

---

## Part 5: Improvements to Current Benchmark

1. **Add kpe config to cache hint sweep (Exp 2)**: Verify whether `EVICT_FIRST` is better for
   kpe (lower reuse probability than ckv). Currently only ckv_int64 is swept.

2. **Add 32-block variant to L2 sweep (Exp 5)**: Competition uses 32 CTAs per token, not 76.
   The L2 cliff location and HBM-bound throughput at 32 blocks is not measured.

3. **Extend pipeline experiment to kpe config (Exp 7)**: kpe has only 512B/gather4 (vs 4096B
   for ckv). Optimal pipeline depth for the smaller kpe stream may differ.

4. **Add `competition_realistic_pt` to pipeline and latency experiments**: The page-table
   pattern is currently only in Exp 3 (access patterns). It should inform pipeline depth choice.

5. **Record actual SM clock frequency**: Add `cudaDeviceGetAttribute(CUDA_DEVICE_ATTR_CLOCK_RATE)`
   before/after each experiment. Thermal throttling during long sweeps could bias BW by ~14%.

---

## Part 6: ptx-benchmark-explorer Additional Findings

The ptx-benchmark-explorer identified the following additional benchmark opportunities not
covered above (lower priority):

- **`cp.reduce.async.bulk.add.f32`** — atomic bulk reduce for split-KV accumulation; could
  eliminate the separate combine kernel if throughput is sufficient at 32KB output size
- **LDG.256 cache hint matrix** — `KU_LDG_256` NC×L1×L2_prefetch sweep for index streaming
- **SMEM swizzle × UTCMMA matrix** — quantify bank conflict penalty for SW128 vs SW64 vs NONE
- **`atomicadd_f32x4_with_policy`** — L2 hint impact for split-KV partial accumulation

Full details in ptx-benchmark-explorer output.

---

## Version History

| Version | Changes | Status |
|---|---|---|
| ver1 | Initial; dim0=64 for INT64 (wrong) | Obsolete |
| ver2 | Fixed dim0=128; added swizzle validity check | Obsolete |
| ver3 | Complete through Exp 6; pipeline N=32 failed | Partial |
| ver4 | All experiments complete; canonical baseline | Superseded |
| **ver5** | 6 improvements applied; new latency_cold, competition_realistic_pt, L2_64B, 296-CTA saturation | **Canonical** |
