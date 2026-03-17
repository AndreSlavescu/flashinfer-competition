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

## Anti-Pattern 6: Warmup before a cold-latency experiment pre-warms L2

**Problem**: Running 3 warmup passes before a "HBM-cold" latency experiment that uses SEQUENTIAL access pre-populates the L2 with the last ~64 MB of the working set. If the measured run starts from row 0 (same as warmup), the first ~62K iterations hit partially warm L2, causing a systematic ~20 ns underestimate of true cold latency.

**Fix**: For cold experiments (`num_iters == 0` sentinel), skip warmup entirely. Or run a cache-invalidating kernel (access a different 256MB+ tensor) immediately before the measured run.

**Where seen**: tma_gather4/main.cu run_latency(), cold variant. The ver5 HBM-cold measurement of 546 ns is a slight lower bound on true cold latency. Effect is small (~3-4%) but systematic.

## Anti-Pattern 7: competition_realistic_page_table still measures L2-resident throughput

**Problem**: The `COMPETITION_REALISTIC_PAGE_TABLE` pattern (added in ver5) correctly models two-level indirection (scattered physical frames) but still generates only 2048 unique rows (2 MB for ckv_int64). At 256 MB declared working set, the pattern tiles these 2048 rows 128×. After warmup, the 2 MB is L2-resident. The pattern shows ~908 GB/s at 256 MB — identical to L2-resident performance — not because page-table scatter is L2-friendly, but because the footprint is small.

**Fix**: To measure real competition workload throughput, generate fresh topk=2048 indices per outer iteration (each set non-repeating with previous sets). Pre-generate `num_runs × 2048` distinct indices uploaded to device; the kernel steps the index pointer by 2048 per outer iteration.

**Where seen**: tma_gather4/index_patterns.cuh, COMPETITION_REALISTIC_PAGE_TABLE case. Same flaw as COMPETITION_REALISTIC (Anti-Pattern 2).

## Anti-Pattern 8: Declaring a valid PTX mbarrier ordering as a CRITICAL violation

**Problem**: The reviewer flagged `mbarrier_expect_tx` called AFTER `tma_gather4` as a "CRITICAL
PTX protocol violation." This is wrong. The PTX ISA explicitly allows `mbarrier.expect_tx` to be
called after the initiating async operation, as long as it precedes `mbarrier.try_wait`. The TMA
engine accumulates `complete_tx` credits independently of the `expect_tx` count. The same ordering
exists in `microbenchmarks/tma_gather4/gather4_kernels.cuh` with an explicit PTX spec comment, and
produces validated 546 ns HBM-cold results.

**Fix**: Before declaring ANY PTX instruction ordering as CRITICAL or MAJOR, consult the PTX ISA
documentation (https://docs.nvidia.com/cuda/parallel-thread-execution/) and cite the specific
section that prohibits it. If no clear prohibition exists in the spec, report as
**NEEDS_VERIFICATION** with the PTX ISA section to check — do not escalate to CRITICAL without
a spec citation. Additional sources (arxiv, reference benchmarks) strengthen confidence.

**Rule**: `expect_tx` AFTER async op but BEFORE `try_wait` = VALID per PTX ISA.
The TMA engine accumulates `complete_tx` credits independently of `expect_tx`.

**Where seen**: dual_tma_stream review, 2026-03-16.

## Anti-Pattern 9: Assuming concurrent TMA streams always run in parallel

**Problem**: Two TMA gather4 warps with separate mbarriers in the same CTA (dual_tma_stream design)
do NOT always run in parallel. In the L2-warm regime (random 256MB, kpe's 8MB unique footprint fits
in L2), kpe degrades from 201 ns isolated → 595 ns concurrent when ckv is also running. The two
streams serialize through the shared L2 bottleneck. In the HBM-bound regime (sequential 1024MB),
both streams run at full independent bandwidth (ckv: 7.58 GB/s isolated = 7.58 GB/s concurrent,
zero interference).

**Fix**: When reviewing dual-stream benchmarks, check BOTH regimes separately. The critical
question for the competition kernel: ckv's large working set evicts kpe's data from L2, so kpe
effectively runs at HBM-cold latency (~595 ns) in BOTH mode — not its isolated L2-warm latency
(201 ns). Do NOT assume "separate mbarriers = independent parallel execution."

**Where seen**: dual_tma_stream ver1, 2026-03-16.

## Why these matter
These patterns can silently produce misleading results — the benchmark compiles and runs without error, but the measured values don't reflect what the experiment claims to measure. Always verify: (a) the effective unique data footprint vs L2 size, (b) whether published PTX patterns in this repo are validated before flagging as violations, (c) cache regime (L2-warm vs HBM-bound) when evaluating concurrent stream behavior.
