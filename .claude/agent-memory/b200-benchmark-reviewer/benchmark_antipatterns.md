---
name: B200 Benchmark Anti-Patterns and Methodological Issues
description: Known measurement pitfalls and anti-patterns that produce misleading results on sm_100a, discovered during tma_gather4 review
type: project
---

## Anti-Pattern 1: Latency kernel that accidentally measures L2-hit latency

**Problem**: A latency kernel running `num_iters=1024` with 4 row indices each = 4096 unique rows × 4096B = 16MB unique data. After 3 warmup runs, ALL accessed rows are L2-resident. The "HBM-cold random 256MB" and "L2-hot sequential 1MB" cases measure the SAME thing (L2-resident TMA latency). They appear different in intent but produce identical numbers (~197 ns net).

**Fix**: To measure true HBM-miss TMA latency: `num_iters` should equal `num_rows`, and indices should be non-repeating (walk through all rows exactly once). No warmup, or count only fresh passes.

**Where seen**: tma_gather4/gather4_kernels.cuh latency_kernel, ver4. Applied to kpe_bf16 and ckv_int64 experiments.

## Anti-Pattern 2: "competition_realistic" pattern that caps unique addresses

**Problem**: `COMPETITION_REALISTIC` pattern generates only 2048 unique token indices (topk=2048) and repeats them for all `num_indices`. For a 256MB working set with ckv_int64, `num_indices = 65536 * 4 = 262144` but only `2048 * 4096B = 8MB` of unique data is accessed. After warmup, all 8MB is L2-resident. The 256MB experiment shows ~905 GB/s — same as L2-resident performance — giving a false impression that the competition workload benefits from L2 reuse even at HBM-bound scale.

**Fix**: Generate fresh topk=2048 random indices per outer iteration (not pre-generated once). Or explicitly document that the pattern repeats and note the effective working set is 8MB.

**Where seen**: tma_gather4/index_patterns.cuh, experiment_patterns. The anomaly shows up clearly in the CSV: competition_realistic at 256MB = 905 GB/s while random at 256MB = 409 GB/s.

## Anti-Pattern 3: cudaFuncSetAttribute with function pointer

**Problem**: Calling `cudaFuncSetAttribute(kernel_fn, ...)` where `kernel_fn` is a function pointer returned from a dispatch table (e.g., `get_pipeline_kernel(N)`) fails with "invalid argument" on CUDA 12/13 because the API expects a direct kernel symbol, not a function pointer variable.

**Fix**: Either pass the kernel function directly (not via pointer), or use `cudaFuncSetAttribute` before the pointer lookup. This was the cause of ver2/ver3 pipeline experiment failures at N=32.

**Where seen**: tma_gather4/main.cu run_pipeline(), fixed in ver4.

## Anti-Pattern 4: Wrong dim0 for competition config

**Problem**: The competition ckv_cache is 512d BF16 = 1024 bytes per row = 128 INT64 elements. The initial benchmark (ver1) used dim0=64 (256B/row, 512B/gather4 for INT64) which is half the correct size. Results in ver1 are invalid — throughput and latency numbers are for a 512B/gather4 config rather than the correct 4096B/gather4.

**Fix**: Always verify dim0 × sizeof(element) = competition row bytes before accepting results.

**Where seen**: tma_gather4/tensor_map_utils.cuh, config_ckv_int64(). Fixed from dim0=64 to dim0=128 in ver3+.

## Anti-Pattern 5: Swizzle constraint mismatch causes cuTensorMapEncodeTiled failure

**Problem**: The TMA hardware enforces `box_dim0 * sizeof(element) <= swizzle_bytes`. For BF16 box=128 (256B) with SWIZZLE_128B (128B): 256 > 128, so the driver API returns "invalid argument" at runtime. The benchmark's validity check (`is_valid_swizzle_config`) catches this correctly, but early versions did not have this guard.

**Fix**: Always check box_bytes <= swizzle_bytes before calling cuTensorMapEncodeTiled. The current benchmark's validity check is correct.

**Where seen**: tma_gather4/main.cu experiment_throughput, was the cause of ver2 crash.

## Why these matter
These patterns can silently produce misleading results — the benchmark compiles and runs without error, but the measured values don't reflect what the experiment claims to measure. Always verify the effective unique data footprint vs L2 size when interpreting latency and throughput numbers.
