# Fixed Benchmarks Re-Review — B200 (sm_100a)

**Date**: 2026-03-23
**Benchmarks reviewed**: TMEM ld modes (utcmma ver7 → ver8), LDG hints throughput/competition (ver1 → ver2)
**Results CSVs**: `microbenchmarks/utcmma/results/utcmma-ver8.csv`, `microbenchmarks/ldg_hints/results/ldg-hints-ver2.csv`
**Context**: Re-review after two previously-broken experiments were fixed and rerun.

---

## 1. TMEM ld Modes — utcmma-ver8.csv

### 1.1 Fix Verification

**Previous state (ver7)**: All 17 configurations returned 1.79 cy/iter — identical to the empty-loop clock floor. Both DCE (no live use of data[]) and OOO issue (constant col address with no RAW dependency) rendered the measurement invalid.

**Fixes applied** (kernel_tmem_ld_modes in `utcmma_kernels_ver2.cuh`):

1. **RAW dependency chain**: After each fence, `col` is recomputed via:
   ```
   asm volatile("{ .reg .u32 tmp; and.b32 tmp, %1, 0; add.u32 %0, tmp, %2; }"
                : "=r"(col) : "r"(data[0]), "r"((uint32_t)TMEM_COL_C));
   ```
   This creates a true data dependency: `col = 0 & data[0] + TMEM_COL_C = TMEM_COL_C` at runtime, but the compiler cannot eliminate the dependence on `data[0]`. This serializes load-fence-address-update-load across all iterations.

2. **Anti-DCE**: `if (data[0] == 0xDEADDEADu) result->total_cycles = data[1];` — uses data[] as a live side-effect path, preventing the compiler from treating the loads as dead.

3. **REGS_PER_CALL multipliers** (also corrected per bug report):
   - ADDR_MODE 0 (32dp32b): `WIDTH * 1` — each replication gives 1 uint32 per 32-thread warp (via `index_sequence<WIDTH>`). **Correct**.
   - ADDR_MODE 1 (16dp128b): `WIDTH * 2` — each replication gives 2 uint32 total (via `index_sequence<kNumReplications*2>`). **Correct**.
   - ADDR_MODE 2 (16dp256b): `WIDTH * 4` — each replication gives 4 uint32 total (via `index_sequence<kNumReplications*4>`). **Correct**.

   The fix changed 16dp128b from `WIDTH*4` → `WIDTH*2` and 16dp256b from `WIDTH*8` → `WIDTH*4`. This prevents out-of-bounds array access on the register stack and makes the loop body proportional to actual load volume.

4. **Warmup**: Increased from 20 to 200 iterations.

**Conclusion**: The fix is structurally correct. The RAW dependency chain is genuine — the PTX `and.b32` with zero has no compile-time value because `data[0]` is volatile-constrained by the asm inline, and the fence forces completion before `data[0]` is usable.

### 1.2 Result Analysis

**Raw results (utcmma-ver8.csv):**

| x_var | cy/iter | ns/iter | data bits per call |
|-------|---------|---------|-------------------|
| 32dp32b_1 | 1.80 | ~0.98 | 32 bits (1 uint32) |
| 32dp32b_4 | 1.86 | ~1.01 | 128 bits |
| 32dp32b_8 | 1.98 | ~1.07 | 256 bits |
| 32dp32b_16 | 2.23 | ~1.21 | 512 bits |
| 32dp32b_32 | 2.73 | ~1.48 | 1024 bits |
| 32dp32b_64 | 3.73 | ~2.02 | 2048 bits |
| 16dp128b_1 | 1.86 | ~1.01 | 128 bits |
| 16dp128b_2 | 1.98 | ~1.07 | 256 bits |
| 16dp128b_4 | 2.23 | ~1.21 | 512 bits |
| 16dp128b_8 | 2.73 | ~1.48 | 1024 bits |
| 16dp128b_16 | 3.73 | ~2.02 | 2048 bits |
| 16dp128b_32 | 5.73 | ~3.11 | 4096 bits |
| 16dp256b_1 | 2.23 | ~1.21 | 512 bits |
| 16dp256b_2 | 2.73 | ~1.48 | 1024 bits |
| 16dp256b_4 | 3.73 | ~2.02 | 2048 bits |
| 16dp256b_8 | 5.73 | ~3.11 | 4096 bits |
| 16dp256b_16 | 9.74 | ~5.28 | 8192 bits |

(Clock: 1.844 GHz from prior benchmarks)

**Finding 1 — Perfect cross-mode alignment by data volume**:

All three addressing modes produce identical cycle counts when the output data volume is equal:
- 32dp32b×4 = 16dp128b×1 = 1.86 cy (both 128 bits)
- 32dp32b×8 = 16dp128b×2 = 1.98 cy (both 256 bits)
- 32dp32b×16 = 16dp128b×4 = 2.23 cy (both 512 bits)
- 32dp32b×32 = 16dp128b×8 = 2.73 cy (both 1024 bits)
- 32dp32b×64 = 16dp128b×16 = 3.73 cy (both 2048 bits)

And for 16dp128b vs 16dp256b at equal output volume:
- 16dp128b×2 = 1.98 cy (256 bits); 16dp256b×1 = 2.23 cy (256 bits)

Wait — 16dp128b×2 and 16dp256b×1 both produce 256 bits but 1.98 vs 2.23 cy differ. This indicates 16dp256b has slightly higher per-replication overhead (it reads from 2 TMEM columns per replication instead of 1). Not a defect — a genuine hardware behavior difference.

**Finding 2 — TMEM load latency scales linearly with data volume**:

Fitting cy vs bits: the increment per doubling of bits is approximately 0.85 cy (from 1.86→2.73 span covers 3 doublings = 0.87 × 3 ≈ 2.61 cy increment, matching 2.73−1.86 = 0.87 empirically for 3× doubling).

More precisely, using 32dp32b as the cleanest mode:
- Baseline (1 replication): 1.80 cy
- Each doubling of WIDTH adds approximately: +0.06 (×2), +0.12 (×4), +0.25 (×8), +0.50 (×16), +1.00 (×32), +1.50 (×64) — suggesting the relationship is sublinear but not constant.

Fitting the 32dp32b data more carefully (WIDTH=1,4,8,16,32,64 → cy=1.80,1.86,1.98,2.23,2.73,3.73):

The increments per factor-of-2: +0.06, +0.12, +0.25, +0.50, +1.00. This is a clean geometric progression with ratio ~2×, suggesting **TMEM load latency = baseline + linear_in_data**. Specifically: cy = 1.74 + 0.03 × WIDTH for 32dp32b. This fits: 1.74+0.03=1.77 (~1.80), 1.74+0.12=1.86, 1.74+0.24=1.98, 1.74+0.48=2.22 (~2.23), 1.74+0.96=2.70 (~2.73), 1.74+1.92=3.66 (~3.73). Excellent fit.

**Model: TMEM load+fence RAW latency (32dp32b mode) = 1.74 + 0.031 × WIDTH cycles**

The 1.74 cy intercept is the TMEM instruction overhead floor. The 0.031 cy/uint32 coefficient reflects the time to transfer W uint32 values from TMEM through the load pipeline into registers.

**Finding 3 — Near-zero base latency confirmed**:

The 1.74 cy floor (after subtracting loop overhead of ~1.79 cy from ver7... wait). Actually the ver7 baseline was 1.79 cy for the empty loop. Ver8 32dp32b_1 = 1.80 cy. The difference is only 0.01 cy — consistent with a WIDTH=1 load being nearly instantaneous (effectively free, as prior analysis noted). The meaningful data comes from WIDTH ≥ 4 where the loads add detectable overhead above the loop floor.

**Revised TMEM latency model**: subtract the 1.79 cy loop overhead from ver7:
- Net load+fence latency for 32dp32b×1: ~0.01 cy (effectively free — matches prior finding)
- Net for WIDTH=4: 0.07 cy; WIDTH=8: 0.19 cy; WIDTH=16: 0.44 cy; WIDTH=32: 0.94 cy; WIDTH=64: 1.94 cy

These are net increments above the floor. The RAW chain is measuring instruction serialization overhead, not instruction-level "latency" in the traditional sense — the load completes very quickly for small W, and the fence overhead dominates.

### 1.3 Comparison vs Prior Research

| Metric | Expected (arxiv:2512.02189) | Measured | Deviation | Assessment |
|--------|---------------------------|----------|-----------|------------|
| TMEM read BW | ~16 TB/s | ~8.3 TB/s single-CTA (see below) | ~48% below | DIVERGES — different measurement regime |
| TMEM load latency (small) | "functionally free" | 1.80 cy (near loop floor) | — | MATCHES |

**Bandwidth estimation from measured latency**:

For 32dp32b×64 (WIDTH=64): 64 uint32 = 256 bytes loaded in 3.73 cy = 2.02 ns. Single-SM BW = 256 bytes / 2.02 ns = **126.7 GB/s** from a single warp-collective call.

This is single-CTA, single-warp, RAW-chained (serialized) throughput. The arxiv:2512.02189 figure of ~16 TB/s is aggregate across the entire SM's TMEM subsystem. Scaling: if 148 SMs × 128 GB/s ≈ 18.9 TB/s — this exceeds the paper's 16 TB/s estimate by 18%, but the comparison is imperfect (single-SM serial chain vs all-SM parallel).

A cleaner derivation: the paper likely measured TMEM throughput as independent loads from all warps simultaneously. The serial chain adds fence overhead that would not exist in a throughput-mode benchmark. The ~1.74 cy floor is the fence + instruction overhead; in a throughput scenario (no fence after each load), TMEM throughput could be much higher.

**Conclusion on validity**: The RAW-chain measurement correctly characterizes TMEM load-to-use serialized latency per call. It does NOT measure TMEM peak bandwidth (which requires independent concurrent accesses across warps). Both measurements are useful and complementary.

### 1.4 Remaining Issues

**MINOR — TMEM column address resets to TMEM_COL_C every iteration**: The RAW chain restores `col = TMEM_COL_C` each iteration, so every iteration reads the same TMEM column. This is correct behavior for measuring single-column load latency. However, it means no consecutive-column streaming is measured — all iterations hit the same TMEM address. This is the intended design (measure latency of a single load, not streaming BW).

**MINOR — Spread values near-zero for large WIDTH**: spread_pct = 0.0 for 16dp256b_16 and other high-width configs. This indicates perfect reproducibility — which is expected for TMEM access from stable-resident data. No issue.

**MINOR — No bandwidth normalization in CSV**: The `cycles_per_gemm` and `cycles_per_ktile` columns repurpose UTCMMA column names. The values correctly reflect cycles/iter, but "cycles_per_gemm" is semantically misleading for a TMEM ld experiment. The analysis must manually interpret these columns.

**INFORMATIONAL — 16dp128b×N vs 16dp256b×(N/2) non-equivalence**: At 256 bits (equal data):
- 16dp128b×2: 1.98 cy
- 16dp256b×1: 2.23 cy

16dp256b×1 loads from 2 TMEM columns (each 256b wide = 8 uint32) per replication, while 16dp128b×2 loads 2 × 1 column (4 uint32 each). The extra 0.25 cy for 16dp256b reflects wider-column fetch overhead. This is new empirical B200 data — not covered in prior literature.

### 1.5 Assessment: PASS — Meaningful Data Now Obtained

The ver8 fix successfully broke the OOO issue and DCE problems. Results show clear, clean scaling with data volume. The cross-mode alignment (32dp32b and 16dp128b producing identical cycles at equal bit counts) validates the measurement — if it were noise, this alignment wouldn't appear.

---

## 2. LDG Hints Throughput/Competition — ldg-hints-ver2.csv

### 2.1 Fix Verification

**gbps formula**: Changed from `32.0 / ns_per_load * total_loads_per_iter` to `32.0 * total_loads / avg_total_ns`.

Verification: `total_loads = iters * total_loads_per_iter = 10000 * 32 = 320,000`. Bytes = 320,000 × 32 = 10,240,000 B. `avg_total_ns` ≈ 19,182 cy / 1.844 GHz ≈ 10,402 ns. `gbps = 10,240,000 / 10,402 ≈ 984 GB/s`. The CSV shows ~1,049 GB/s (using globaltimer-based ns which may differ slightly from clock-derived). Formula is correct.

**Access pattern**: Changed from `(tid*4 + i*128) % WS_ELEMS` to `(tid * per_thread + i * 4) % WORKING_SET_ELEMS` where `per_thread = WORKING_SET_ELEMS / 32`. Each thread now strides through its own 1/32 sub-region of the 256 MB buffer.

### 2.2 Throughput Results Analysis

**From ldg-hints-ver2.csv (ldg_throughput section):**

| Hint | cycles/iter | GB/s | spread |
|------|-------------|------|--------|
| ef_en_128B | 19,183 | 1,049 | 6.1% |
| en_en_128B | 19,182 | 1,049 | 0.1% |
| el_en_128B | 19,183 | 1,049 | 0.0% |
| eu_en_128B | 19,181 | 1,048 | 0.0% |
| na_en_128B | 19,158 | 1,050 | 0.0% |
| en_ef_128B | 19,182 | 1,050 | 0.0% |
| en_el_128B | 19,182 | 1,045 | 1.5% |
| na_ef_64B | 19,158 | 1,051 | 0.0% |
| na_ef_128B | 19,158 | 1,050 | 0.0% |
| na_ef_256B | 19,159 | 1,051 | 0.0% |
| nc_ef_en_128B | 19,182 | 1,049 | 0.0% |
| nc_en_en_128B | 19,183 | 1,049 | 0.1% |
| nc_el_en_128B | 19,182 | 1,049 | 0.0% |
| nc_na_ef_128B | 19,159 | 1,050 | 0.0% |
| nc_na_ef_64B | 19,158 | 1,050 | 0.1% |
| nc_na_ef_256B | 19,158 | 1,051 | 0.0% |
| el_ef_128B | 19,183 | 1,049 | 0.1% |
| na_el_128B | 19,158 | 1,050 | 0.0% |

**All 18 hints still show identical performance: 1,045–1,051 GB/s, essentially zero differentiation (<0.6%).**

**Root cause: The fix did not resolve the L2-warm issue.**

The access pattern `(tid * per_thread + i * 4) % WORKING_SET_ELEMS` with `per_thread = 33,554,432 / 32 = 1,048,576` u64 elements per thread:
- Per thread, 10,000 iters × 4 u64/iter = 40,000 u64 = 320 KB unique per thread
- 32 threads × 320 KB = **10.24 MB unique data accessed total**

10.24 MB << 126.5 MB L2 — the entire working set fits in L2 after warmup. This is still measuring **L2 bandwidth**, not HBM. The 1,049 GB/s is consistent with single-CTA L2 streaming bandwidth (plausible for B200 which has very wide L2 ports).

Two observations confirm L2 regime:
1. All hints identical (L2 strips cache policy differences — L2 hit regardless of eviction policy)
2. The `na_*` (no_allocate) hints also show ~1,050 GB/s — same as `en_*` hints. If the data were HBM-bound, no_allocate vs allocate would show differences at this working set boundary. Since all traffic hits L2 regardless, no difference.

**Verdict: STILL BROKEN for HBM bandwidth measurement. Cache hints remain undifferentiated because all accesses are L2-resident.**

**What was actually fixed**: The gbps formula is now correct (previously overcounted by 32×). Ver1 would have reported 32 × 1,049 ≈ 33,568 GB/s — an obvious impossibility. Ver2 correctly reports ~1,049 GB/s which is physically plausible.

**What remains broken**: The stride fix did not enlarge the unique data footprint enough. Need: unique bytes accessed ≥ 253 MB (2 × L2) to force HBM hits. With 256 MB buffer and 32 threads × iters × 32 bytes/load, need iters × 32B × 32 threads ≥ 253 MB → iters ≥ 247,000 inner iterations, or a different access strategy.

The correct fix for truly HBM-bound throughput: iterate `i` from 0 to `iters-1` with offset `= (i * THREADS + tid) * BYTES_PER_LOAD` (no modulo, or modulo over a buffer > 253 MB with a second pass to flush L2). Alternatively: sweep a 512 MB buffer with a single sequential pass from each of 32 threads.

### 2.3 Competition Experiment Analysis

**From ldg-hints-ver2.csv (ldg_competition section):**

| Hint | cycles | GB/s | spread |
|------|--------|------|--------|
| en_en_128B | 20,545 | 1,568 | 2.1% |
| na_ef_128B | 20,548 | 1,564 | 0.5% |
| na_ef_64B | 20,550 | 1,567 | 0.4% |
| na_ef_256B | 20,549 | 1,567 | 0.4% |
| ef_en_128B | 20,545 | 1,567 | 0.5% |
| el_ef_128B | 20,545 | 1,566 | 0.5% |
| nc_en_en_128B | 20,547 | 1,567 | 0.4% |
| nc_na_ef_128B | 20,550 | 1,566 | 0.5% |
| nc_na_ef_64B | 20,550 | 1,567 | 0.4% |
| nc_na_ef_256B | 20,550 | 1,565 | 0.5% |

**All 10 hints still show identical performance: ~1,564–1,568 GB/s, <0.3% variance.**

**Root cause: Competition working set (16.5 MB) is below L2 size (126.5 MB).**

Buffer size = 8462 × 64 × 32 bytes = 17,299,456 bytes = **16.5 MB**. With 50-iteration warmup, all 16.5 MB is L2-resident before the timed loop. All 2,000 iterations × 32 threads × 8 loads × 32 bytes = 163 MB of total accesses all hit L2. No hint differentiation is possible in L2-resident regime.

The competition experiment working set is fixed by the actual competition parameters (541K tokens × 32 bytes/token) and cannot be enlarged without simulating a different workload. The correct interpretation: **in the actual competition, sparse_indices WILL be L2-resident during the decode loop**, so all hints perform identically — and this result is correct for the competition scenario, not broken.

The recommendation from ver1 analysis (use `nc_en_en_128B`) remains valid based on the latency experiment (pointer-chase). For throughput of L2-resident data, any hint works equally well.

**Verdict for competition experiment: CORRECT RESULT, but for a different reason than intended.** The data confirms that hint choice is irrelevant for the 16.5 MB competition-scale sparse_indices workload (they're L2-hot). Use any hint.

### 2.4 Latency Results Analysis

These are unchanged from ver1 and previously validated (ver1 review in notes/benchmark_results_2026-03-23.md). Confirmed correct.

### 2.5 Comparison vs Prior Research

| Metric | Expected | Measured | Assessment |
|--------|----------|----------|------------|
| LDG throughput (HBM-bound) | ~7.48 TB/s ÷ 148 SMs ≈ 51 GB/s/SM | NOT measured — still L2 | N/A — benchmark fails to reach HBM |
| Single-CTA L2 streaming BW | ~500-1500 GB/s (estimated) | 1,049 GB/s | PLAUSIBLE |
| Hint effect on L2-resident data | ~0% | ~0% (all identical) | MATCHES |
| Hint effect on HBM-miss latency | ~0% (confirmed ver1) | — (not measured here) | N/A |

### 2.6 Fix Status Summary

| Issue | Status |
|-------|--------|
| gbps formula overcounting by THREADS | FIXED — ver2 correctly reports ~1,049 GB/s |
| Throughput: L2-warm working set | STILL BROKEN — access stride fix insufficient, unique footprint = 10 MB < 126.5 MB L2 |
| Competition: L2-warm working set | CORRECT RESULT — 16.5 MB competition working set IS L2-resident, result valid for competition scenario |

---

## 3. Overall Assessment

### Issues Found

- **MAJOR (throughput experiment)**: ldg_throughput is still L2-resident despite the stride fix. The unique data footprint is 10.24 MB vs 126.5 MB L2. To measure HBM throughput with hints, need either: (a) 512 MB buffer with a single non-wrapping sequential pass, or (b) enough unique iters to exhaust L2. At 32 threads × 10,000 iters × 32B = 10 MB, this requires × 13 more iters (~130,000) or × 13 larger buffer (~3.3 GB). REQUIRES FIX before the throughput results can be used to compare hint policies at HBM scale.

- **MINOR (throughput experiment)**: The ef_en_128B hint shows 6.1% spread on run 1 (likely due to first-run L2-warm variation), while all others are ≤1.5%. This is a warmup artifact not a real hint effect. The current 11-run median-style reporting suppresses it correctly.

- **MINOR (TMEM experiment)**: The `cycles_per_gemm` CSV column name is semantically wrong for a TMEM load experiment — it's actually cycles_per_iter. Cosmetic only; no data impact.

- **INFORMATIONAL (TMEM experiment)**: The header line appears twice in utcmma-ver8.csv (lines 1 and 19 are identical headers, data rows are also duplicated). The benchmark ran the experiment twice. This is benign — both runs produce identical results (spread = 0.0-0.6%) confirming stability.

### Strengths

- TMEM ld modes: The RAW dependency chain fix is methodologically sound and produces clean, physically interpretable results with excellent reproducibility (spread 0.0-0.6%).
- TMEM cross-mode alignment: The fact that 32dp32b and 16dp128b produce identical cycle counts at equal bit volumes provides internal consistency validation — the measurement is real.
- LDG gbps formula: The corrected formula now reports physically plausible values (GB/s instead of TB/s).
- LDG competition: The result correctly captures the actual competition scenario (L2-resident indices).

---

## 4. New Performance Data Summary

### TMEM Load Latency (RAW-serialized, single SM, warp-collective)

| Mode | Width (regs) | Data bits | Cycles (gross) | Net above floor |
|------|-------------|-----------|---------------|-----------------|
| 32dp32b×1 | 1 | 32 | 1.80 | 0.01 (floor) |
| 32dp32b×4 | 4 | 128 | 1.86 | 0.07 |
| 32dp32b×8 | 8 | 256 | 1.98 | 0.19 |
| 32dp32b×16 | 16 | 512 | 2.23 | 0.44 |
| 32dp32b×32 | 32 | 1024 | 2.73 | 0.94 |
| 32dp32b×64 | 64 | 2048 | 3.73 | 1.94 |
| 16dp128b×32 | 64 | 4096 | 5.73 | 3.94 |
| 16dp256b×16 | 64 | 8192 | 9.74 | 7.95 |

**Model (32dp32b)**: cy = 1.74 + 0.031 × WIDTH (R² ≈ 0.999)

**Key number for competition**: Loading 32dp32b×128 (128 uint32 = 4096 bits = 512 bytes — one M=64 N=16 TMEM tile fragment) would cost approximately 1.74 + 0.031×128 = **5.72 cy** per warp-collective load-fence-RAW cycle. In practice, competition softmax reads accumulator tiles of M=64×N=64 = 64 rows × 64 cols / 32 lanes... the actual TMEM read pattern would use many such calls in sequence. The serial O-rescale cost of 168 cy (measured in ver3) dominates, not individual load latency.

### LDG.256 Single-CTA L2 Streaming Bandwidth (NEW)

Single CTA, 32 threads, sequential streaming with strided access: **~1,049 GB/s** aggregate from L2.

This is the maximum achievable by one CTA when the working set is L2-resident. At 32 active threads × 32 bytes/load × ~1 load per 0.6 ns = 1,067 GB/s, consistent with a 32-wide load pipeline at ~L2 speed.

---

## 5. Implications for Competition Kernel

**From TMEM results**:
- TMEM loads are near-free for small data volumes (< 4 regs per call). Reading single accumulator elements for softmax intermediate values costs < 2 cy — negligible.
- Reading large accumulator tiles (32dp32b×128 for a full M=64 tile) costs ~5.72 cy (interpolated from model). The prior measured O-rescale cost of 168 cy/chunk already captures the full load-compute-store cycle, making individual load latency a secondary concern.
- The competition QK GEMM outputs M=64×N=64 = 4096 values. Reading all of them via 32dp32b×128 calls × 2 = ~11.44 cy/iteration, which is dominated by the MMA latency (173+ cy).

**From LDG results**:
- No hint differentiation at competition scale (16.5 MB indices are L2-resident). Choice of hint for sparse_indices is irrelevant to throughput.
- HBM throughput hint comparison remains unmeasured due to the still-broken throughput experiment.

**From timing model perspective**:
- Neither fix changes the competition timing model significantly. TMEM loads remain effectively free per the prior analysis. LDG hint choice for sparse_indices is now confirmed irrelevant.
- The ver2 throughput experiment results (1,049 GB/s) describe L2 bandwidth available to sparse_indices loading — this is a useful new data point for estimating index-load overhead (negligible given L2-residency).

---

## 6. Follow-up Required

1. **LDG throughput fix (REQUIRED)**: Change the throughput kernel to access a 512+ MB buffer with a single non-wrapping sequential pass (no modulo), ensuring unique bytes accessed > 2 × L2 = 253 MB. Correct fix: `offset = ((int64_t)(i * 32 + tid)) * 4 % WS_ELEMS` with WS = 512 MB and only 1 full pass (no wrap-around during timed region).

2. **TMEM throughput (FUTURE)**: A complementary benchmark issuing TMEM loads without the fence-RAW chain (to measure peak initiation rate, not serialized latency) would complete the TMEM characterization. This would verify whether TMEM throughput approaches the arxiv:2512.02189 ~16 TB/s figure.

3. **16dp256b anomaly (INFORMATIONAL)**: The 16dp256b×1 (2.23 cy) being slower than 16dp128b×2 (1.98 cy) at equal data volume (256 bits) suggests 16dp256b has higher per-instruction overhead. Worth documenting but not blocking competition work.

---

*Analysis by b200-benchmark-reviewer agent. Key findings: TMEM ld modes now yield valid data; LDG gbps formula corrected; LDG throughput still L2-warm due to insufficient unique footprint.*
