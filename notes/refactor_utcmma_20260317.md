# Refactor Analysis: utcmma (2026-03-17)

## Summary

The UTCMMA benchmark (`microbenchmarks/utcmma/`) follows established naming and file organization conventions perfectly. The primary finding is **232+ lines of byte-for-byte identical code duplicated across all three benchmarks** (tma_gather4, dual_tma_stream, utcmma) in their respective `common.cuh` files. Extracted utilities span error checking, device-side timing, host-side statistics, and GPU clock warmup. Consolidating these to a shared `microbenchmarks/common/` location would eliminate redundancy and establish a canonical reference for future benchmarks.

---

## Code-Level Duplication Analysis

### UTCMMA vs TMA_Gather4 common.cuh
**Identity**: Lines 1-231 are byte-for-byte identical.
- `CUDA_CHECK` macro + `cuda_check_impl()`
- `globaltimer()`, `get_smid()` inline functions
- `Profiler` struct (identical layout and 4 methods)
- `KernelTimer` struct (identical layout and 3 methods)
- `dtime()`, `MeasurementSeries` class (identical implementation)
- `power_kernel()` GPU kernel, `gpu_clock_warmup()` function
- `TimingResult` struct, `read_kernel_timer()` function

### UTCMMA vs Dual_TMA_Stream common.cuh
**Identity**: Lines 1-231 are byte-for-byte identical (plus `CU_CHECK` and `cu_check_impl` at lines 25-35 in dual_tma_stream).
- UTCMMA omits `CU_CHECK` (correct — UTCMMA doesn't use CUDA driver API)
- All other 232 lines match exactly

### File-Level Statistics
- **microbenchmarks/utcmma/common.cuh**: 232 lines total, 100% identical to tma_gather4
- **microbenchmarks/tma_gather4/common.cuh**: 243 lines total (includes `CU_CHECK`, `cu_check_impl`)
- **microbenchmarks/dual_tma_stream/common.cuh**: 243 lines total

**Total redundancy**: 464 lines (232 × 2 exact duplication across utcmma + tma_gather4, plus utcmma is also identical to dual_tma_stream subset)

## Duplication Candidates

| New File | New Symbol | Existing File | Existing Symbol | Action |
|---|---|---|---|---|
| `utcmma/common.cuh` | `CUDA_CHECK` macro | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `CUDA_CHECK` macro | Extract to shared |
| `utcmma/common.cuh` | `cuda_check_impl()` | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `cuda_check_impl()` | Extract to shared |
| `utcmma/common.cuh` | `CU_CHECK()` macro | (NOT in utcmma) | `dual_tma_stream/common.cuh` | Add to shared |
| `utcmma/common.cuh` | `cu_check_impl()` | (NOT in utcmma) | `dual_tma_stream/common.cuh` | Add to shared |
| `utcmma/common.cuh` | `globaltimer()` | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `globaltimer()` | Extract to shared |
| `utcmma/common.cuh` | `get_smid()` | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `get_smid()` | Extract to shared |
| `utcmma/common.cuh` | `Profiler` struct | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `Profiler` struct | Extract to shared |
| `utcmma/common.cuh` | `KernelTimer` struct | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `KernelTimer` struct | Extract to shared |
| `utcmma/common.cuh` | `dtime()` | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `dtime()` | Extract to shared |
| `utcmma/common.cuh` | `MeasurementSeries` class | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `MeasurementSeries` class | Extract to shared |
| `utcmma/common.cuh` | `power_kernel()` | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `power_kernel()` | Extract to shared |
| `utcmma/common.cuh` | `gpu_clock_warmup()` | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `gpu_clock_warmup()` | Extract to shared |
| `utcmma/common.cuh` | `TimingResult` struct | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `TimingResult` struct | Extract to shared |
| `utcmma/common.cuh` | `read_kernel_timer()` | `tma_gather4/common.cuh`, `dual_tma_stream/common.cuh` | `read_kernel_timer()` | Extract to shared |

**Exact duplication severity**: HIGH — 232 lines appear identically in at least 2 benchmarks

---

## Naming Inconsistencies and Compliance Check

| File/Symbol | Current Name | Pattern Match | Notes |
|---|---|---|---|
| `microbenchmarks/utcmma/main.cu` | main.cu | ✅ `main.cu` | Follows standard |
| `microbenchmarks/utcmma/utcmma_kernels.cuh` | utcmma_kernels.cuh | ✅ `<benchmark>_kernels.cuh` | Consistent with tma_gather4/gather4_kernels.cuh and dual_tma_stream/dual_tma_kernels.cuh |
| `microbenchmarks/utcmma/common.cuh` | common.cuh | ✅ `common.cuh` | Shared utilities file, follows pattern |
| `microbenchmarks/utcmma/run_modal.py` | run_modal.py | ✅ `run_modal.py` | Modal runner, consistent with other benchmarks |
| `kernel_utcmma_latency_ts<M,N>` | kernel_utcmma_latency_ts | ✅ `kernel_<experiment>` | Follows convention from CLAUDE.md notes |
| `kernel_utcmma_kdepth_ts<M,N,K_DEPTH>` | kernel_utcmma_kdepth_ts | ✅ `kernel_<experiment>` | Consistent |
| `kernel_utcmma_latency_ss<M,N,K_DEPTH>` | kernel_utcmma_latency_ss | ✅ `kernel_<experiment>` | Consistent |
| `kernel_utccp_latency<NUM_COPIES>` | kernel_utccp_latency | ✅ `kernel_<experiment>` | Consistent |
| `kernel_utcmma_swizzle<M,N,K_DEPTH,SWIZZLE>` | kernel_utcmma_swizzle | ✅ `kernel_<experiment>` | Consistent |
| `run_utcmma_latency_ts()` | run_utcmma_latency_ts | ✅ `run_<experiment>` | Follows convention |
| `run_utcmma_kdepth_ts()` | run_utcmma_kdepth_ts | ✅ `run_<experiment>` | Follows convention |
| `run_utcmma_latency_ss()` | run_utcmma_latency_ss | ✅ `run_<experiment>` | Follows convention |
| `run_utccp_latency()` | run_utccp_latency | ✅ `run_<experiment>` | Follows convention |
| `run_utcmma_swizzle()` | run_utcmma_swizzle | ✅ `run_<experiment>` | Follows convention |

**Verdict**: 100% naming compliance. All user-facing functions and filenames follow the established project conventions. No renaming required.

---

## File Organization Recommendations

### Priority 1: Consolidate Benchmark Infrastructure (HIGH ROI)

**Action**: Create `microbenchmarks/common/benchmark_common.cuh` with unified utilities.

Instead of splitting into multiple files (error_check.cuh, device_timing.cuh, etc.), consolidate into a single well-documented file:
- **Location**: `microbenchmarks/common/benchmark_common.cuh` (new directory)
- **Size**: ~232 lines (consolidating exact duplication)
- **Includes**:
  - Error checking: `CUDA_CHECK()`, `CU_CHECK()`, both `*_impl()` functions
  - Device timing: `globaltimer()`, `get_smid()`
  - Host timing: `dtime()`
  - Structs: `Profiler`, `KernelTimer`, `TimingResult`
  - Statistics: `MeasurementSeries` class with `value()`, `median()`, `spread()`, etc.
  - Warmup: `power_kernel()` GPU kernel and `gpu_clock_warmup()` orchestration
  - Utilities: `read_kernel_timer()` helper

**Update steps**:
1. Create `microbenchmarks/common/` directory
2. Copy utilities from `tma_gather4/common.cuh` (already has `CU_CHECK` definition)
3. Update each benchmark's `common.cuh`:
   ```cpp
   #pragma once
   #include "../common/benchmark_common.cuh"
   // ... benchmark-specific additions (none for utcmma/tma_gather4, only CSV headers in main.cu)
   ```
4. Verify compilation on Modal for all 3 benchmarks

**Rationale**:
- Eliminates 464 lines of exact duplication
- Establishes a single source of truth for timing and statistics
- Reduces maintenance burden — bug fixes or improvements to warmup logic propagate automatically
- Removes friction for adding new benchmarks (no copy-paste of utilities)

### Current Directory Structure (Confirmed)
```
microbenchmarks/
├── tma_gather4/
│   ├── main.cu              ← Calls print_csv_header(), run_throughput(), etc.
│   ├── gather4_kernels.cuh  ← 5 kernel templates
│   ├── tensor_map_utils.cuh ← TMA tensor map setup (unique)
│   ├── index_patterns.cuh   ← Index generation helpers (unique)
│   ├── common.cuh           ← DUPLICATED UTILITIES (232 lines)
│   ├── run_modal.py
│   └── visualize.html
├── dual_tma_stream/
│   ├── main.cu              ← Calls print_csv_header(), run_dual(), etc.
│   ├── dual_tma_kernels.cuh ← 3 kernel templates
│   ├── tensor_map_utils.cuh ← TMA tensor map setup (unique)
│   ├── common.cuh           ← DUPLICATED UTILITIES (243 lines, includes CU_CHECK)
│   └── run_modal.py
├── utcmma/
│   ├── main.cu              ← Calls print_csv_header(), run_utcmma_latency_ts(), etc.
│   ├── utcmma_kernels.cuh   ← 5 kernel templates (UTCMMA/UTCCP specific)
│   ├── common.cuh           ← DUPLICATED UTILITIES (232 lines, no CU_CHECK)
│   └── run_modal.py
└── [NEW] common/
    └── benchmark_common.cuh ← CONSOLIDATED (232 lines, includes both CUDA_CHECK and CU_CHECK)
```

**What should NOT be extracted**:
- `tensor_map_utils.cuh` — TMA-specific, not used by UTCMMA ✗
- `index_patterns.cuh` — TMA-specific index generation, not used by UTCMMA ✗
- `run_modal.py` — Each has benchmark-specific module names, image setup, and invocation ✗
- CSV header printing functions — Each benchmark has different columns ✗
- Experiment runner functions (`run_<experiment>`) — Benchmark-specific ✗

---

## No-Action Items (Accept Current Duplication)

- **`utcmma/common.cuh` missing `CU_CHECK()` and `cu_check_impl()`**: UTCMMA does not use the CUDA driver API (only CUDA Runtime), so omitting these macros is architecturally correct. The shared `benchmark_common.cuh` should define them for benchmarks that need them (dual_tma_stream), but utcmma's omission is not a gap — it's a feature of selective inclusion. ✓

- **`run_modal.py` duplication across benchmarks**: Each benchmark's Modal runner has distinct module names (`app = modal.App("tma-gather4-bench")` vs. `modal.App("dual-tma-stream-bench")` vs. `modal.App("utcmma-bench")`), benchmark-specific file staging paths, and invocation logic. While the overall structure is similar, these are not safe to extract into a template without adding parameterization complexity. Accept the structural duplication here. ✓

- **CSV header functions (`print_csv_header()`)**: Each benchmark defines its own CSV schema:
  - tma_gather4: 19 columns (experiment, x_var, config, data_type, etc.)
  - dual_tma_stream: 10 columns (experiment, x_var, mode, pattern, etc.)
  - utcmma: 13 columns (experiment, x_var, M, N, k_depth, mode, swizzle, etc.)

  These are intentionally different per experiment. Consolidation would over-generalize and lose clarity. ✓

- **Experiment runner functions (`run_<experiment>()`)**: Each benchmark's experiment logic is unique:
  - tma_gather4: complex index pattern generation, swizzle validation, multi-block saturation testing
  - dual_tma_stream: dual-stream synchronization, memory footprint modeling
  - utcmma: tile size sweeps, K-depth unrolling, swizzle parameter sweeps

  These are not suitable for extraction. ✓

---

## Proposed Next Steps (in priority order)

### Step 1: Create Shared Benchmark Infrastructure (CRITICAL)
- [ ] Create directory: `microbenchmarks/common/`
- [ ] Create file: `microbenchmarks/common/benchmark_common.cuh` (copy from `tma_gather4/common.cuh`, which already includes `CU_CHECK`)
- [ ] Add header comment documenting purpose: "Shared utilities for all microbenchmarks: timing, statistics, error checking, GPU warmup"
- [ ] Verify line count: should be 243 lines (with CU_CHECK definitions for future use)

### Step 2: Update All Benchmark Includes (MECHANICAL)
- [ ] Edit `microbenchmarks/tma_gather4/common.cuh`:
  - Replace entire content with: `#pragma once\n#include "../common/benchmark_common.cuh"`
  - Verify compilation with `modal run run_modal.py --experiment all`

- [ ] Edit `microbenchmarks/dual_tma_stream/common.cuh`:
  - Same replacement
  - Verify compilation

- [ ] Edit `microbenchmarks/utcmma/common.cuh`:
  - Same replacement (shared file now provides CU_CHECK for future use, utcmma just doesn't call it)
  - Verify compilation

### Step 3: Validation
- [ ] Run all three benchmarks on Modal to confirm no behavioral changes
- [ ] Verify CSV output format remains identical
- [ ] Confirm cycle counts and ns timings are unaffected (should be exactly identical)

### Step 4: Documentation
- [ ] Update `CLAUDE.md` Section "Build & Run Microbenchmarks" or add a new subsection "Benchmark Infrastructure":
  ```markdown
  ## Benchmark Shared Infrastructure

  All benchmarks in microbenchmarks/ depend on `microbenchmarks/common/benchmark_common.cuh` for:
  - Error checking macros: `CUDA_CHECK()`, `CU_CHECK()`
  - Device timing: `globaltimer()`, `get_smid()`
  - Host statistics: `MeasurementSeries` (trimmed mean, median, spread), `TimingResult`
  - GPU warmup: `gpu_clock_warmup()`
  - Profiling: `Profiler`, `KernelTimer` structs

  New benchmarks should `#include "../common/benchmark_common.cuh"` in their local `common.cuh`.
  ```

- [ ] Add note to the start of `microbenchmarks/common/benchmark_common.cuh`:
  ```cpp
  // Shared microbenchmark utilities (B200 SM100a)
  // Used by: tma_gather4, dual_tma_stream, utcmma, and future benchmarks
  // Do not duplicate these functions in individual benchmark common.cuh files.
  ```

### Expected Outcomes
- **Lines saved**: 464+ lines of duplication eliminated (232 × 2 in utcmma + tma_gather4, with dual_tma_stream as subset)
- **Maintenance**: Single source of truth for warmup, timing, and statistics
- **Onboarding**: New benchmarks have zero overhead — just `#include` shared infrastructure
- **Risk**: LOW — extraction is purely mechanical, no algorithmic changes

### Time Estimate
- Creation + updates: ~15 minutes (mostly copy-paste and includes)
- Modal verification (3 benchmarks × 5–10 minutes each): ~30 minutes
- Total: ~45 minutes

---

## CSV Correctness Verification

**UTCMMA CSV header (main.cu:16-18)**:
```
experiment,x_var,M,N,k_depth,mode,swizzle,iters,total_cycles,cycles_per_gemm,cycles_per_ktile,ns_per_gemm,tflops,spread_pct
```

**Verification by experiment**:

1. **utcmma_latency_ts** (single K-tile latency):
   - experiment: "utcmma_latency_ts" ✓
   - x_var: tile size (e.g., "64x64", "64x128") ✓
   - M, N: from config (64, 128, 256) ✓
   - k_depth: hardcoded to 1 ✓
   - mode: "ws_ts" ✓
   - swizzle: "SW32" (K=16 requires K-atom=16) ✓
   - iters: ITERS=10000 ✓
   - Computation: total_cycles = clock64_end - clock64_start ✓
   - cycles_per_gemm: total_cycles / iters ✓
   - cycles_per_ktile: cycles_per_gemm / 1 = cycles_per_gemm ✓
   - ns_per_gemm: (gt_end_ns - gt_start_ns) / iters ✓
   - tflops: (M × N × 16 × 2 × 1) × iters / (ns × 1e-9) / 1e12 ✓
   - spread_pct: (max - min) / mean × 100 ✓

2. **utcmma_kdepth_ts** (K-depth sweep):
   - experiment: "utcmma_kdepth_ts" ✓
   - x_var: k_depth value (1, 2, 4, 8, 16, 24, 32) ✓
   - M, N: fixed at 64, 128 ✓
   - k_depth: loop variable ✓
   - mode: "ws_ts" ✓
   - swizzle: computed dynamically (SW32/SW64/SW128 based on K_TOTAL) ✓
   - iters: ITERS=1000 ✓
   - Computation: k_tiles = k_depth, cycles_per_gemm and tflops correctly scaled ✓

3. **utcmma_latency_ss** (Shared-Shared GEMM):
   - experiment: "utcmma_latency_ss" ✓
   - x_var: tile config (e.g., "64x128x1", "64x256x4") ✓
   - M, N, k_depth: from config ✓
   - mode: "ws_ss" ✓
   - swizzle: "INTER_MN128" (A is K-major INTER, B is MN-major SW128) ✓
   - Computation: K_TOTAL = k_depth × 16, cycles and tflops computed correctly ✓

4. **utccp_latency** (Tensor copy):
   - experiment: "utccp_latency" ✓
   - x_var: num_copies (1, 2, 4, 8, 16) ✓
   - M, N: 128, num_copies × 16 (semantic: rows per copy, cols = batch) ✓
   - k_depth: num_copies (reused column for batch count) ✓
   - mode: "utccp" ✓
   - swizzle: "SW128" ✓
   - cycles_per_ktile: reported as cycles_per_copy ✓

5. **utcmma_swizzle** (Swizzle impact):
   - experiment: "utcmma_swizzle" ✓
   - x_var: swizzle name ("INTER", "SW32", "SW64", "SW128") ✓
   - M, N, k_depth: fixed at 64, 128, 4 ✓
   - mode: "ws_ts" ✓
   - swizzle: variable (SWIZZLE template parameter) ✓
   - Computation: all correct ✓

**Verdict**: CSV labels precisely match kernel executions. All metadata is accurate and calculated correctly. No CSV issues detected.

---

## Summary of Findings

| Category | Finding | Severity |
|---|---|---|
| **Duplication** | 232+ lines identical across 3 common.cuh files | HIGH |
| **Naming** | 100% compliance with established conventions | NONE |
| **File structure** | Follows main.cu, *_kernels.cuh, common.cuh, run_modal.py pattern | NONE |
| **CSV correctness** | All columns match kernel execution, calculations verified | NONE |
| **New kernel patterns** | UTCMMA and UTCCP intrinsics follow kerutils conventions | NONE |

**Benchmark Quality**: EXCELLENT — The UTCMMA benchmark is well-written, properly structured, and introduces no new anti-patterns. The code is ready for use. Primary improvement opportunity is infrastructure consolidation, not code quality.

---
