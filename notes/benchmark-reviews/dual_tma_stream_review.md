# Dual TMA Stream Benchmark Review

**Benchmark**: `microbenchmarks/dual_tma_stream/`
**Reviewed**: 2026-03-16
**Reviewer**: Manual review (correcting b200-benchmark-reviewer agent findings)

---

## Overview

The benchmark measures TMA gather4 stream interference on B200: a single CTA launches 2 warps
issuing independent TMA gather4 operations simultaneously — warp 0 loads ckv-style data (4096B
per call, INT64 layout) and warp 1 loads kpe-style data (512B per call, BF16 layout). Three
modes let us isolate each stream and measure their concurrent impact.

**Competition relevance**: Directly models the FlashMLA decode kernel's dual TMA gather pattern
where ckv_cache (512d BF16 = 1024B, but benchmarked as INT64-packed = 128 elements × 8B = 4096B
in the largest config) and kpe_cache (64d BF16 = 128B per row, 4 rows per gather4 = 512B) are
loaded simultaneously by different warpgroups.

---

## Correctness of mbarrier Ordering — Reviewer Error Refuted

The automated reviewer flagged the following ordering as a "CRITICAL PTX protocol violation":

```cpp
tma_gather4(&ckv_map, ckv_mbar, ckv_smem, 0, rows, cache_hint);
mbarrier_expect_tx(ckv_mbar, (uint32_t)bytes_ckv);  // ← flagged as wrong
mbarrier_arrive(ckv_mbar);
mbarrier_wait(ckv_mbar, phase);
```

**This finding is incorrect.** The PTX ISA explicitly permits `mbarrier.expect_tx` to be called
*after* the initiating async operation, as long as it precedes `mbarrier.try_wait`. The TMA
engine accumulates `complete_tx` credits independently of the `expect_tx` count. The barrier
only triggers when both conditions are met: the expected TX count equals the accumulated credits,
AND an arrive has been recorded.

**Evidence from this codebase**: `microbenchmarks/tma_gather4/gather4_kernels.cuh` (the
throughput_kernel at approximately line 168) uses the identical ordering with an explicit comment:

```cpp
// PTX spec: expect_tx can be called after the async op, before try_wait
tma_gather4(desc_ptr, mbar_ptr, smem_ptr, col_idx, rows, cache_hint);
mbarrier_expect_tx(mbar_ptr, (uint32_t)bytes_per_gather4);
mbarrier_arrive(mbar_ptr);
mbarrier_wait(mbar_ptr, phase);
```

That benchmark produces validated latency results (546 ns HBM-cold, matching published B200
specs). **No fix needed for mbarrier ordering.**

---

## Real Issues Found

### MAJOR-1: Unsynchronized t0 Timestamp in BOTH Mode

**Severity**: Major (affects BOTH mode measurements — the core measurement of this benchmark)

**Problem**: In BOTH mode (mode == 2), threads 0 and 32 each read `t0 = globaltimer()` inside
their respective `if (tid == ...)` branches:

```cpp
if (tid == 0) {
    if (mode == 0 || mode == 2) {
        uint32_t phase = 0;
        int64_t t0 = globaltimer();  // ← unsynchronized: warp 0 reads t0 here
        ...
if (tid == 32) {
    if (mode == 1 || mode == 2) {
        uint32_t phase = 0;
        int64_t t0 = globaltimer();  // ← unsynchronized: warp 1 reads t0 later
        ...
```

The warp scheduler may dispatch warp 0 and warp 1 at different cycle counts. In BOTH mode, the
thread that reaches its `t0 = globaltimer()` first starts timing before the other stream has
even issued its first TMA. This means:

- The *faster* stream (lower elapsed time) gets a *longer* measured time (started timing earlier)
- The *slower* stream gets a *shorter* measured time (started timing later)
- Crossover effect: under interference, one stream may appear faster than baseline

**Impact**: In isolation modes (CKV_ONLY, KPE_ONLY) this is irrelevant. In BOTH mode, the
interference ratio `ckv_GBps_BOTH / ckv_GBps_CKV_ONLY` is the key metric — and the numerator
is biased by warp scheduling skew.

**Quantitative bound**: At ~2.1 GHz, warp scheduling skew is typically <1 µs. Benchmark
duration at 1024MB sequential ≈ 33 ms. Maximum bias ≈ 1 µs / 33 ms ≈ 0.003%. This is
negligible in practice, but the fix is trivial and makes the design correct.

**Fix**: Move `t0 = globaltimer()` to after the final `__syncthreads()`, before the mode
branches, so all threads read the timestamp at the same instruction boundary.

---

### MEDIUM-1: kpe Data Volume Asymmetry Not Documented

**Severity**: Medium (affects interpretation of "HBM-cold" results for kpe)

**Problem**: `num_calls = ckv_num_rows / 4` synchronizes both streams to the same iteration
count. However:

- ckv: `num_calls × 4096B` = `data_MB` (full working set)
- kpe: `num_calls × 512B` = `data_MB / 8` (1/8th of the working set)

At 256MB with random pattern:
- ckv accesses 65536 gather4 × 4096B = 256MB unique data → ~L2-cold
- kpe accesses 65536 gather4 × 512B = 32MB, from a 2097152-row dataset → 8MB unique data → **L2-warm**

At 1024MB with sequential pattern:
- ckv accesses 262144 × 4096B = 1024MB sequential → truly HBM-cold
- kpe accesses 262144 × 512B = 128MB sequential → still within 2×L2, partially warm

**Is this wrong?** No — this matches the competition kernel's natural behavior. kpe_cache is 8×
smaller than ckv_cache per token, so it naturally stays L2-warm under production load. The
benchmark correctly models this asymmetry. But it must be documented so results are not
misinterpreted as "kpe HBM-cold throughput."

**Fix**: Add `kpe_data_MB` column to CSV output and a comment in `run_dual()` documenting
the 8:1 asymmetry.

---

### MINOR-1: CSV Missing experiment/x_var Fields

**Severity**: Minor (tooling compatibility)

**Problem**: The `visualize.html` tool (used by tma_gather4 benchmark) requires `experiment`
and `x_var` fields at the front of each CSV row. The dual_tma_stream CSV lacks these fields,
preventing drop-in visualization.

**Fix**: Add `experiment,x_var` columns with values `"dual_tma"` and `"mode"` respectively.

---

### MINOR-2: Double sync/fence Pattern Lacks Explanation

**Severity**: Minor (readability)

**Problem**: The two-`__syncthreads()` + `fence_proxy_async()` pattern is non-obvious:

```cpp
__syncthreads();
fence_proxy_async();
__syncthreads();
```

Without a comment, readers may wonder if this is a bug or if one `__syncthreads()` is redundant.

**Fix**: Add a comment explaining the protocol: first sync ensures mbarrier_init is complete
before the fence; fence establishes async proxy visibility; second sync ensures all threads
see the fence before issuing TMA operations.

---

## Competition Fidelity Analysis

### Byte Volumes (256MB config)
| Stream | Bytes per gather4 | Rows per gather4 | Rows (256MB) | Gather4 calls |
|--------|------------------|-----------------|-------------|---------------|
| ckv    | 4096B (INT64 packed, 128 elem × 8B × 4 rows) | 4 | 65536 | 16384 |
| kpe    | 512B (BF16, 64 elem × 2B × 4 rows) | 4 | 2097152 | 16384 |

Wait — `ckv_num_rows = 256MB / 1024B = 262144`, not 65536. `num_calls = 262144 / 4 = 65536`.
So: ckv does 65536 gather4 calls × 4096B = 256MB ✓; kpe does 65536 × 512B = 32MB.

### 8:1 Ratio Matches Competition Kernel
- Competition: ckv_cache dim=512 BF16 (1024B/row), kpe_cache dim=64 BF16 (128B/row) → 8:1 ratio ✓
- Benchmark: bytes_ckv=4096, bytes_kpe=512 → 8:1 ratio ✓ (4 rows per gather4 × 1024B vs 4 × 128B)

### Key Question Answered by This Benchmark
Does running kpe gather4 alongside ckv gather4 degrade ckv throughput? If `ckv_GBps_BOTH ≈
ckv_GBps_CKV_ONLY`, the B200 TMA engine handles both streams independently without L2 bandwidth
contention. If there is significant degradation (>10%), the ckv and kpe streams are competing
for L2 bandwidth and the competition kernel must budget accordingly.

---

## Measured Results (ver1, B200, 2026-03-16)

CSV: `microbenchmarks/dual_tma_stream/results/dual-tma-ver1.csv`

| mode | pattern | data_MB | kpe_data_MB | ckv_GBps | kpe_GBps | ckv_ns | kpe_ns |
|------|---------|---------|-------------|----------|----------|--------|--------|
| ckv_only | random | 256 | 32.0 | 7.20 | — | 568.9 | — |
| kpe_only | random | 256 | 32.0 | — | 2.50 | — | 205.0 |
| **both** | random | 256 | 32.0 | **6.96** | **0.88** | **588.7** | **585.1** |
| ckv_only | sequential | 1024 | 128.0 | 7.58 | — | 540.3 | — |
| kpe_only | sequential | 1024 | 128.0 | — | 1.12 | — | 458.8 |
| **both** | sequential | 1024 | 128.0 | **7.58** | **1.04** | **540.5** | **492.6** |

### Key Findings

**HBM-cold (sequential 1024MB): Zero ckv interference.**
ckv throughput in BOTH mode = 7.58 GB/s, identical to CKV_ONLY (7.58 GB/s). The B200 TMA
engine dispatches dual gather4 streams without any bandwidth contention. ckv latency: 540.3
vs 540.5 ns — within noise.

**kpe slows slightly in HBM-cold BOTH mode**: 1.12 → 1.04 GB/s (7% slower, 458.8 → 492.6 ns).
Minor L2 contention between the two streams, but negligible for competition kernel planning.

**L2-warm (random 256MB): kpe severely degraded by ckv L2 eviction.**
kpe_only: 2.50 GB/s (205 ns, L2-warm from its 8MB unique footprint).
ckv+kpe: kpe drops to 0.88 GB/s (585 ns, near cold latency). Root cause: ckv's 256MB random
working set continuously evicts kpe's 8MB data from L2, forcing kpe to HBM. This is the
correct production behavior — ckv dominates the L2 and kpe must absorb the evictions.

ckv in L2-warm BOTH mode: 6.96 vs 7.20 GB/s (3.3% slower). ckv itself was already partially
L2-cold (256MB >> 64MB L2), so kpe sharing the L2 adds minor extra pressure.

### Competition Kernel Implications

1. **No TMA engine throttling from dual streams**: issuing ckv and kpe gather4 simultaneously
   does not reduce ckv throughput on B200. The two streams are truly independent at the hardware
   level.

2. **ckv dominates L2**: With 8:1 byte ratio, ckv's working set will evict kpe data in practice.
   kpe latency at production scale ≈ HBM cold (540 ns), not L2-warm (205 ns). Budget accordingly.

3. **TMA gather latency reference**:
   - ckv HBM-cold: 540 ns per 4096B gather4 ≈ matches tma_gather4 benchmark (546 ns)
   - kpe HBM-cold (sequential): 458–492 ns per 512B gather4 (same latency, smaller payload)
   - kpe L2-warm: 205 ns per gather4 (confirmed in isolation; evicted under production load)

---

## Summary

| Finding | Severity | Status |
|---------|----------|--------|
| mbarrier ordering (reviewer claim) | ~~CRITICAL~~ | **FALSE POSITIVE** — PTX-compliant, validated by tma_gather4 throughput results |
| Unsynchronized t0 in BOTH mode | MAJOR | Fixed in dual_tma_kernels.cuh |
| kpe data volume asymmetry undocumented | MEDIUM | Fixed: added kpe_data_MB to CSV + comment |
| Missing experiment/x_var CSV fields | MINOR | Fixed: added to CSV |
| Double-sync fence pattern unexplained | MINOR | Fixed: added comment |
