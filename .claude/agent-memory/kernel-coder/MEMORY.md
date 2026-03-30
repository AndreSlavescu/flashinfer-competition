# Kernel Coder Memory

## Benchmark Results
- [v2_benchmark_results.md](v2_benchmark_results.md) — v2 vectorized PyTorch: 23/23 PASS, 1.80x avg speedup, ~1.18ms fixed cost

## Submission Contract
- [flashinfer_bench_contract.md](flashinfer_bench_contract.md) — kernel.py must be self-contained, no local imports, entry_point="kernel.py::kernel"

## CuTeDSL Compilation Patterns (verified on B200)

### 2026-03-27 — CuTeDSL GEMM M=64 N=64 K=512 BF16: PASS

- Verified: Full pipeline (TMA load -> SMEM -> tcgen05.mma -> TMEM -> registers -> global) compiles and runs correctly (max_err=5.96e-07).
- Key parameters: threads=128, CtaGroup.ONE, cluster=(1,1), 2 AB stages, 1 ACC stage, mma_tiler=(64,64,64), K_tile=64 (inst_k=16 * 4), TMEM cols=512.
- Pattern: scripts/test_cutedsl_gemm_compile.py

### 2026-03-27 — CuTeDSL stream argument must be cuda.CUstream

- Error: `DSLRuntimeError: expects argument to be CUstream, got int`
- Fix: `stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)`

### 2026-03-27 — PipelineTmaUmma tx_count must be compile-time resolvable

- Error: `Unable to convert dynamic Boolean value to bool at compile time` in MbarrierArray init
- Root cause: Passing tx_count as a kernel Int32 argument made it dynamic. The pipeline compares tx_count < 0 at compile time.
- Fix: Store `self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size` in `@cute.jit` __call__; `cute.size_in_bytes()` on static layouts returns Python int. Access via `self.*` in kernel.

### 2026-03-27 — local_tile requires batch dimension

- Error: `failed to construct a valid coordinate from #cute.coord<"(_,_,_)">`
- Root cause: Input tensors were 2D (M,K) but `cute.local_tile(tensor, tile, coord)` with a 3D tiler expected 3D tensors (M,K,L).
- Fix: Add batch dimension L=1 to input/output tensors. Use 3D tensors like dense_gemm example.

### 2026-03-27 — Key CuTeDSL dense_gemm pattern to follow

- Use `smem.allocate_tensor()` for SMEM (not @cute.struct for data)
- Use `cpasync.tma_partition()` for TMA src/dst partitioning
- Use `tiled_mma.make_fragment_A/B(sX)` for SMEM->MMA fragments
- Use `PipelineTmaUmma.create(...).make_participants()` -> `ab_producer, ab_consumer`
- Use `PipelineUmmaAsync` for accumulator pipeline
- Epilogue: `get_tmem_load_op` -> `make_tmem_copy` -> `get_slice(tidx)` -> `partition_S/D` -> `cute.copy` subtiles
- Barriers: @cute.struct for mbar storage, `pipeline.sync(barrier_id=1)` before tmem.free

## Toolchain Notes

### 2026-03-27 — Modal correctness run failed before upload because local packer dependency was missing

- Error:
  `scripts/bench.py` failed locally during `pack_solution_local(...)` with `No module named 'flashinfer_bench'`.
- Root cause:
  The Modal benchmark job itself runs remotely, but the solution packaging step happens on the client machine first and imports `flashinfer_bench` locally.
- Fix:
  Run the benchmark from the repo's `.venv`, where `modal`, `flashinfer_bench`, and `torch` are installed. The command path should be `.venv/bin/modal ...` moving forward.

### 2026-03-27 — CuTeDSL Modal smoke compiler only receives a single source file

- Error:
  Not a hard failure, but an important compile constraint discovered during the Modal smoke setup.
- Root cause:
  `tools/sass/dump_sass_modal.py --cutedsl ...` writes only the target `kernel.py` into the Modal container before executing it.
- Fix:
  Keep the CuTeDSL smoke path in `solution/dsa_attention/kernel.py` self-contained and avoid repo-local imports for the compile-smoke entrypoint.

### 2026-03-27 — Keep the benchmark path CuTeDSL-only

- Rule:
  The benchmark/submission entrypoint should either launch the intended CuTeDSL kernel or fail loudly. Do not silently fall back to eager PyTorch in the benchmark path.
- Why:
  Otherwise correctness or benchmark runs can pass without exercising the intended kernel, and SASS analysis can drift away from what the benchmark actually used.
- Fix:
  Keep any eager/PyTorch reference in a separate debug/reference implementation, not behind a silent fallback in the submission path.

### 2026-03-27 — `dump_sass_modal.py --cutedsl` has a strict tool-specific entrypoint contract

- Note:
  The SASS harness executes `python3 kernel.py` inside Modal with a timeout. Keep the CuTe smoke entrypoint self-contained and short-running.

### 2026-03-27 — For SM100 CuTeDSL structural plumbing, consult working reference patterns first

- Rule:
  When wiring staged SMEM layouts, `ThrMma` fragment shapes, TMEM readback views, or copy-atom tiling on SM100, consult working reference kernels before improvising the exact CuTeDSL object shapes.
- Why:
  The hard failures were almost all structural: missing MLIR context, wrong staged SMEM shape, composed-layout stride misuse, fragment rank mismatches, T2R indexing shape, and copy-atom alignment/vector-width verification.
- Fix:
  Start from known-good patterns in:
  - `references/flash-attention/flash_attn/cute/flash_fwd_sm100.py`
  - `references/CuTeDSL-kernels/quack/gemm_sm100.py`
  - `references/flash-attention/flash_attn/cute/copy_utils.py`
  - `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/blackwell_helpers.py`

### 2026-03-27 — Create DSL objects inside the JIT/launch path, not in eager Python setup

- Rule:
  When helper constructors touch MLIR-backed CuTeDSL ops, build them inside `@cute.jit` or under an explicit MLIR context.
- Why:
  Eager construction of `make_trivial_tiled_mma(...)` failed with "An MLIR function requires a Context".
- Fix:
  Move tiled MMA / tiled copy construction into the launch function so the DSL has the context it expects.

### 2026-03-27 — NEVER use `from __future__ import annotations` in CuTeDSL kernel files

- Rule:
  Do not use `from __future__ import annotations` in any file containing CuTeDSL `@cute.jit` or `@cute.kernel` decorated functions.
- Why:
  PEP 563 makes ALL annotations lazy strings. CuTeDSL decorators inspect annotations at definition time to determine parameter types (`cute.Tensor`, `cute.TiledMma`, `cute.Tile`, etc.). With lazy annotations, the JIT compiler sees strings instead of types, causing `epi_tile: cute.Tile` to be treated as dynamic rather than compile-time, which cascades into `get_tmem_load_op` failing with "Cta tile and 2sm config does not generate correct num dp."
- Fix:
  Remove the `from __future__ import annotations` import. Use explicit string annotations only where needed for forward references.

### 2026-03-27 — v3 CuTeDSL QK GEMM submission: 5/5 competition shapes PASS

- Kernel: `solution/dsa_attention/kernel.py` (v3)
- Architecture: CuTeDSL tcgen05.mma for QK GEMM (CKV + KPE fused two-phase), PyTorch for gather/softmax/SV/combine
- Accuracy: output max_abs 3.81e-06 to 1.53e-05, LSE max_abs 9.54e-07, cos_diff < 1e-9
- Competition tolerance: output abs_tol=1e-2, LSE abs_tol=1e-3, cos_diff_tol=1e-6

### 2026-03-30 — v4 proven path (BatchedQKKernel + PyTorch softmax/SV): 23/23 PASS

- Kernel: `solution/dsa_attention/kernel.py` (v4)
- Architecture: CuTeDSL BatchedQKKernel for QK GEMM + PyTorch softmax/SV/combine
- All 23 competition workloads pass. ~200-250ms latency (0.01-0.02x ref). Not competitive but correct.

### 2026-03-30 — Fused kernel hang fix: defer_sync + cluster

- [fused_kernel_hang_fix.md](fused_kernel_hang_fix.md) — Adding defer_sync=True to pipeline creates and cluster=(1,1,1) to launch fixed GPU deadlock

### 2026-03-30 — sQrow row-major fix for gather4 staging

- [cutedsl_sQrow_rowmajor_fix.md](cutedsl_sQrow_rowmajor_fix.md) — sQrow must be row-major to match gather4 write order and _make_rowmajor_mn_view strides. Fix applied but fused kernel still has remaining numerical errors.

### 2026-03-30 — Fused kernel remaining issues

- With hang fixed + sQrow row-major fix, fused kernel runs but produces wrong results
- LSE off by ~30-60 in log2 for some heads/splits; some heads have negative LSE
- Root cause likely in `_copy_rowmajor_stage_to_operand_b` or `_make_swizzled_mn_view` swizzle handling
- Q operand via TMA is bypassed (correct). K operand via gather4 + copy still broken.
- Next step: either fix the B copy or replace it with direct SMEM staging
