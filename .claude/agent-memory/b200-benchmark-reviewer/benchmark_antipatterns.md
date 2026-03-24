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

## Anti-Pattern 10: CSV metadata labels that don't match what the kernel actually executes

**Problem**: A `run_and_report(...)` call site hardcodes a static label (e.g., `"SW128"`) for a field like swizzle mode, but the kernel selects the value dynamically based on a template parameter (e.g., `constexpr int SWIZZLE_B = (K_TOTAL >= 64) ? 128 : (K_TOTAL >= 32 ? 64 : 32)`). The hardware runs the correct swizzle, but the CSV records the wrong one. Downstream agents and training data then associate the measured cycle counts with the wrong swizzle mode, corrupting the knowledge base.

**Real example**: `kernel_utcmma_latency_ts` always uses SW32 (K=16 requires K-atom=16), but `run_and_report` reported `"SW128"`. `kernel_utcmma_kdepth_ts` auto-selects SW32/SW64/SW128 based on K_TOTAL, but `run_and_report` always reported `"SW128"` — so k_depth=1 (SW32) and k_depth=2 (SW64) were both mislabeled.

**Fix**: For any field that varies per configuration, compute the label dynamically in the launch loop to mirror the kernel's selection logic exactly. Never hardcode a static string for a field whose value is template- or runtime-dependent. Cross-check every `run_and_report` field against the kernel's actual compile-time selections.

**Why this matters**: CSV results feed downstream review agents, the knowledge base, and training data for a kernel generation agent. A wrong label (e.g., "SW128 achieves X cycles" when it was actually SW32) produces false conclusions about B200 performance characteristics that propagate indefinitely.

**Where seen**: `microbenchmarks/utcmma/main.cu`, experiments 1 and 2, discovered 2026-03-17. Fixed by computing `swizzle_name` dynamically.

## Anti-Pattern 11: UTCMMA multi-accumulator rotation with N_ACC < latency/throughput ratio

**Problem**: Rotating accumulator TMEM column addresses across k_depth inner iterations (k % N_ACC)
provides zero benefit if N_ACC × throughput_cycles < structural latency. For B200 tcgen05.mma with
~54-cycle structural initiation interval and suspected 11-cycle paper throughput, you need N_ACC ≥
54/11 ≈ 5 to fully pipeline. Testing N_ACC=1,2,3,4 and observing flat performance does NOT prove
the paper's 11-cycle figure is wrong — it only proves the rotation is insufficient to hide the
latency at those values of N_ACC. Similarly, it does NOT definitively prove that N_ACC ≥ 5 WOULD
work. Both conclusions require testing N_ACC ≥ 5.

**Observed**: utcmma_throughput_2acc experiment (ver3): N_ACC=1..4 all produce identical
54.60 cy/K-tile for M=64 N=64 k_depth=4. Consistent with structural 54-cycle floor OR insufficient
rotation depth. Follow-up with N_ACC=8 required.

**Why this matters**: The "2-accumulator pipelining → 5× throughput improvement" optimization
hypothesis was the key unknown that ver3 was designed to resolve. The zero-speedup result at
N_ACC ≤ 4 changes the competition kernel's compute model, but the ambiguity about N_ACC ≥ 5
leaves open a potential optimization path.

**Fix**: Always test N_ACC up to and beyond latency/throughput_estimate (i.e., N_ACC ≥ 5 for
UTCMMA). Use M=64 N=64 with N_ACC=8 (8×32=256 TMEM cols, fits in budget) as the definitive test.

## Anti-Pattern 12: UTCMMA latency without commit+wait measures issue rate not completion

**Problem**: Kernels that do `gemm(tiled_mma, tA, sB, tC)` in a loop without `tcgen05.commit + mbarrier.wait + tcgen05.fence::after_thread_sync` measure the rate at which MMA instructions can be issued (~54 cy/tile for M64N64), NOT when results are available in TMEM (~173 cy). Ver3/ver4 utcmma benchmarks had this bug, showing 54 cy vs the correct 173 cy.

**Impact**: 3× underestimate of competition GEMM time (1,392 ns ver3 vs 1,809 ns ver5 per block).

**Fix**: Always add `umma_arrive_noelect(*bar) + bar->wait(phase) + tcgen05_after_thread_sync()` after each timed GEMM group. This is the correct completion-latency measurement.

**Where seen**: utcmma_kernels.cuh ver3/ver4 MMA experiments. Fixed in ver5 (2026-03-21).

## Anti-Pattern 13: Throughput benchmark with insufficient unique footprint despite large buffer allocation

**Problem**: Changing the stride pattern from `(tid * 4 + i * 128) % WS_ELEMS` to `(tid * per_thread + i * 4) % WORKING_SET_ELEMS` (ver2 fix) reduces but does NOT eliminate the L2-warm issue. With 32 threads × 10,000 iters × 32 bytes/load = 10.24 MB unique data accessed — still << 126.5 MB L2. The benchmark still measures L2 bandwidth. The ver1 footprint was ~2.5 MB; ver2 is ~10 MB — both far below the 253 MB threshold for HBM-bound operation. All hints still show identical ~1,049 GB/s.

The deeper issue: the stride `(i * 4)` per iteration causes the thread to revisit the same region after `per_thread / 4 = 262,144` iters. With only 10,000 iters, each thread only touches 40,000 u64 = 320 KB of its 8 MB sub-region. The working set appears large (256 MB buffer) but the accessed subset is small.

**Fix**: To measure HBM throughput, use a SINGLE non-wrapping pass: `offset = ((int64_t)i * THREADS + tid) * 4` (no modulo) with ITERS = total_WS_elems / (THREADS * 4) = ~2.1 million iterations for 256 MB. Or use a 512 MB buffer with a single forward sweep. Ensure total unique bytes = buffer size, with no L2-warm pass before.

**Where seen**: ldg_hints/ldg_kernels.cuh kernel_ldg_throughput (ver1 and ver2 — only partially fixed).

## Anti-Pattern 14: gbps formula overcounts by total_loads_per_iter factor

**Problem**: `run_ldg` computes `ns_per_load = avg_total_ns / (iters * total_loads_per_iter)` then `gbps = 32.0 / ns_per_load * total_loads_per_iter`. Substituting: `gbps = 32.0 * total_loads_per_iter^2 * iters / avg_total_ns`. The formula overcounts by `total_loads_per_iter` (32× for throughput, 256× for competition experiment). All reported GB/s values are inflated by this factor.

**Fix**: Correct formula: `gbps = 32.0 * total_loads_per_iter * iters / avg_total_ns`. Or equivalently: `gbps = 32.0 * total_loads_per_iter / ns_per_iter` where `ns_per_iter = avg_total_ns / iters`.

**Where seen**: ldg_hints/main.cu run_ldg() (ver1).

## Anti-Pattern 15: TMEM ld benchmark without RAW dependency chain measures loop overhead floor

**Problem**: `kernel_tmem_ld_modes` measures TMEM load latency across different addressing modes (32dp32bNx vs 16dp128bNx vs 16dp256bNx) but uses the same constant TMEM column each iteration with no data-flow dependency between iterations. The compiler can remove loads or the hardware can issue all loads out-of-order. All 17 configurations produce identical 1.79 cy/iter — the clock-read floor for an empty loop. No width or mode differentiation is measurable.

**Fix**: Add a RAW dependency: `asm volatile("{ .reg .u32 tmp; and.b32 tmp, %1, 0; add.u32 %0, tmp, %2; }" : "=r"(col) : "r"(data[0]), "r"((uint32_t)TMEM_COL_C))` after fence. Anti-DCE: use `if (data[0] == 0xDEADDEAD) result->total_cycles = data[1]`. Also fix REGS_PER_CALL: 16dp128b needs WIDTH*2 (not WIDTH*4), 16dp256b needs WIDTH*4 (not WIDTH*8).

**Fixed in ver8 (2026-03-23)**: Results now show clean scaling: 32dp32b model cy = 1.74 + 0.031×WIDTH (R²≈0.999). Cross-mode alignment (32dp32b×W = 16dp128b×(W/2) at equal bit volume) validates the measurement. WIDTH=1 is near floor (1.80 cy); WIDTH=64 is clearly differentiated (3.73 cy).

**Where seen**: utcmma/utcmma_kernels_ver2.cuh kernel_tmem_ld_modes (ver7 broken, ver8 fixed).

## Why these matter
These patterns can silently produce misleading results — the benchmark compiles and runs without error, but the measured values don't reflect what the experiment claims to measure. Always verify: (a) the effective unique data footprint vs L2 size, (b) whether published PTX patterns in this repo are validated before flagging as violations, (c) cache regime (L2-warm vs HBM-bound) when evaluating concurrent stream behavior, (d) that every CSV label field matches what the kernel actually executes.
