# CuTeDSL (Python) CODING & DEBUGGING GUIDE
===============================================================================
## NON-NEGOTIABLE RULE (READ FIRST)
**DO NOT change the intent of the code by making the CuTeDSL kernel simpler or
by using less CuTeDSL.**
- Do **not** "fix" bugs by replacing CuTeDSL tiling/partitioning/`TiledMma`/
  pipeline with a naive PyTorch or plain-CUDA implementation.
- Preferably do **not** remove async copies (`cute.copy` with TMA / `cp.async`
  atoms), swizzled smem layouts, `cute.Tensor` views, or CuTeDSL abstractions
  (`@cute.jit`, `@cute.kernel`, `cute.local_tile`, `thr_mma.partition_*`, etc.)
  just to pass correctness.
- Do **not** fall back to `torch.matmul` / `torch.softmax` / CUTLASS pre-built
  `GemmOperation` wrappers unless the original intent *already was* to call
  them. `torch.*` ops are banned in the kernel compute path per CLAUDE.md —
  they are only allowed in Python-side prologue/epilogue setup.
- Your job is to make the *same algorithm and same CuTeDSL structure* correct:
  repair layouts, strides, partitions, pipeline stages, barrier placement,
  predication, and integration wiring — **without de-CuTe-ing the kernel**.
- Do **not** change the input sizes (input tensor sizes) given in the original
  model.
This is a correctness guide, not a "simplify to green tests" guide.

===============================================================================
## LLM OPERATING PROCEDURE (How to debug without changing intent)
When you are given incorrect CuTeDSL code, follow this strict workflow:
1) **Freeze intent**
* Write down (explicitly, in the response) what the code is trying to compute:
   - inputs/outputs (tensor shapes, dtypes, majorness via `LayoutEnum.from_tensor`)
   - mathematical operation (e.g., GEMM, convolution-like, reduction, attention block, stencil, etc.)
   - invariants that must remain true (MMA tiler shape, cluster shape, pipeline
     stages, smem staging, TMEM allocation on Blackwell, etc.)
* Confirm what you will NOT change (see Non-Negotiable Rule above).
2) **Localize the bug without de-optimizing**
* You may add:
   - `cute.printf(...)` prints (guarded by `if tidx == 0 and bidx == 0:` etc.)
   - `cute.print_tensor(...)` on small fragments
   - Python-side `assert` on static shapes before `@cute.jit` entry
   - runtime asserts via `cutlass.dsl.assume` / `cute.assume`
   - extra verification kernels or temporary checksum reads
   - temporary extra `cute.arch.barrier()` / `cute.arch.sync_threads()` /
     `pipeline.*.wait()` for diagnosis (then remove once fixed)
* You may NOT replace the CuTeDSL structure with a simpler kernel "just for
  debugging" — no dropping `@cute.kernel`, no inlining with Python loops over
  global memory.
3) **Fix in the smallest possible CuTeDSL-native way**
* Typical correct fixes:
   - wrong `cute.make_layout(shape, stride=...)` (shape/stride mismatch, or
     forgetting to pass an explicit stride for non-contiguous tensors)
   - wrong tile view (`cute.local_tile`, `cute.logical_divide`,
     `cute.zipped_divide`, `thr_mma.partition_A/B/C`, `cpasync.tma_partition`)
     producing permuted coordinates
   - missing predicate on edge tiles (use `cute.make_identity_tensor` +
     comparison masks, or guard stores with an `if` on the coordinate)
   - missing `pipeline.*.wait()` / `producer_commit()` / `consumer_release()`
     between producer and consumer of an async copy stage
   - wrong smem layout (`sm100_utils.make_smem_layout_a/b/epi`) for the chosen
     MMA atom — a mismatched swizzle or K-major vs MN-major will silently
     corrupt
   - wrong accumulator dtype (`acc_dtype`) or epilogue cast placed before the
     MMA accumulate loop finishes
* Avoid "global refactors." Make one change, re-test.
4) **Prove the fix**
* Provide:
   - a short statement of what was wrong
   - what changed (which `make_layout` / `partition_*` / `pipeline.*` call)
   - why it preserves intent
   - how it was validated (small cases via `run_synthetic_check`, plus at
     least one edge case — non-power-of-two M/N/K, non-multiple-of-tile sizes,
     or boundary batch)

===============================================================================
## CORE CONCEPTS (Layouts, Shapes/Strides, Tensors, Tiles)

1) **`cute.Layout` = (Shape, Stride) and it is the truth**
* `cute.make_layout(shape, stride=...)` directly defines address mapping.
  If output is "plausible but wrong," assume layout/stride is wrong until
  proven otherwise. Dump it with `print(layout)` at trace time, or
  `cute.printf` at runtime.

2) **`cute.Tensor` = pointer + `Layout`**
* A CuTeDSL `Tensor` (constructed via `cute.make_tensor(ptr, layout)` or
  implicitly from a `torch.Tensor` argument to `@cute.jit`) is a view.
  "Correct launch" does not imply "correct indexing."

3) **Tiling/partitioning are coordinate transforms**
* `cute.local_tile`, `cute.local_partition`, `thr_mma.partition_A/B/C`,
  `cpasync.tma_partition`, `cute.slice_`, `cute.group_modes`, etc. do not
  copy — they remap indices. Debug them like math; print the resulting
  `.layout` and `.shape` before doing any compute.

4) **`cutlass.const_expr` vs runtime values**
* Values guarded by `cutlass.const_expr(...)` are resolved at trace time and
  participate in `@cute.jit` compilation. A mismatched `const_expr` branch
  often shows up as a MLIR verification error or a `TypeError` — this is
  *not* a reason to delete the branch; it is a signal that a static shape
  or dtype is wrong upstream.

===============================================================================
## CORRECTNESS DEBUGGING CHECKLIST (Silent bugs & fixes)

Fix in this order:
A) **Shape/Layout/Stride sanity**
1. Confirm each input tensor's real stride (from how `torch.Tensor` or numpy
   data was produced — transposed, `.contiguous()`, strided views, etc.).
   Pass it through `utils.LayoutEnum.from_tensor(t)` to detect majorness.
2. Confirm each `cute.Tensor` view (`local_tile`, `slice_`, `group_modes`,
   `make_fragment_A/B/C`) preserves the intended coordinates. Print the
   `.layout` at trace time.
3. Spot-check a few coordinates → physical addresses → expected values
   via `cute.crd2idx(coord, layout)`.
B) **Copy path correctness**
4. Validate copy-only loop (gmem → smem → gmem) using `cute.copy` with the
   TMA atom *before* looking at compute. Write smem back to a scratch gmem
   tensor and compare to the source.
5. If async is used (`cpasync.CopyBulkTensorTileG2SOp`, `cp.async`,
   `PipelineTmaUmma`, `PipelineUmmaAsync`), ensure required
   `producer.acquire_and_advance()`, `consumer.wait_and_advance()`,
   `consumer_handle.release()`, and `producer_commit(state)` calls exist
   before first use of the data.
C) **Compute correctness**
6. Validate partition shapes match expectations. For `cute.gemm(tiled_mma,
   acc, A_frag, B_frag, acc)`:
   - `A_frag` = `(MMA, MMA_M, MMA_K, ...)`
   - `B_frag` = `(MMA, MMA_N, MMA_K, ...)`
   - `acc`   = `(MMA, MMA_M, MMA_N)`
   Print via `print(tCrA.layout)` during trace.
7. Verify `acc_dtype` (e.g., `cutlass.Float32`) and that any cast to
   `c_dtype` happens in the epilogue, *after* the K mainloop finishes.
D) **Edges**
8. Predication for partial tiles (M/N/K remainders) must be correct. Use
   `cute.make_identity_tensor(shape)` to get coordinate tensors, then mask
   with `cute.where(coord < bound, val, zero)`.
9. Ensure stores are masked correctly (no out-of-bounds writes). TMA store
   on Blackwell (`cpasync.CopyBulkTensorTileS2GOp`) requires no-OOB tiles
   unless explicitly handled.

===============================================================================
## ASYNC COPY & SYNC (TMA / cp.async / pipelines / barriers)

**Async correctness rules (do these before performance tuning):**
- If you use `cutlass.pipeline.PipelineTmaUmma`, `PipelineUmmaAsync`,
  `PipelineAsync`, or raw `cp.async` / TMA atoms, every stage must have:
   - a clear "produce" point: `producer.acquire_and_advance()` returns a
     handle; issue `cute.copy(tma_atom, src, dst, tma_bar_ptr=handle.barrier,
     mcast_mask=...)` with that handle.
   - a clear "consume" point: `consumer.wait_and_advance(peek_status)` before
     reading; `consumer_handle.release()` after the last use.
   - correct barrier placement — `pipeline_init_arrive(cluster_shape_mn=...)`
     and `pipeline_init_wait(cluster_shape_mn=...)` flanking TMEM alloc on
     Blackwell.
- Symptom mapping:
   - stale/previous-tile values → missing `wait_and_advance` or `release`,
     or reading before `tma_bar_ptr` barrier completes
   - fails only when `num_stages > 1` → stage index math
     (`producer_handle.index` vs `.count`) or barrier-storage sizing bug
     (check `MemRange[Int64, num_stages * 2]`)
   - "hangs" → mismatched number of producer vs consumer arrivals, or
     missing `producer_commit(state)` on the accumulator pipeline
- For debugging you may temporarily serialize:
   - set `num_ab_stage = 1` (or `num_acc_stage = 1`)
   - copy then compute, with an explicit `cute.arch.barrier()` between them
   - add extra `cute.arch.sync_threads()`
Then restore the original pipeline depth once fixed.

===============================================================================
## MMA / TiledMma PITFALLS (Atom selection, partitioning, layouts)

- Atom selection must match arch + dtype. On Blackwell (sm100a) use
  `cutlass.utils.blackwell_helpers.make_trivial_tiled_mma(a_dtype, a_major,
  b_major, acc_dtype, cta_group, mma_tiler_mn)` — do **not** hand-pick an
  Ampere/Hopper atom and expect it to work. The returned `TiledMma`
  determines the expected smem layout and K-tile count.
- Smem layout must match what the MMA path expects. Use the helpers
  `sm100_utils.make_smem_layout_a / _b / _epi` — these encode the correct
  swizzle. A mismatched swizzle will compile fine and produce wrong values.
- Partitioning must align A/B fragments so K is consistent. After
  `thr_mma = tiled_mma.get_slice(mma_tile_coord_v)`, the fragments come from
  `thr_mma.partition_A(gA)`, `thr_mma.partition_B(gB)`,
  `thr_mma.partition_C(gC)`, and the compute operands from
  `tiled_mma.make_fragment_A(sA)` / `make_fragment_B(sB)` /
  `make_fragment_C(acc_shape)` where
  `acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])`.
- Debug rule: prove A/B fragments contain the intended values *before*
  blaming `cute.gemm`. Dump a small slice with `cute.print_tensor(tCrA[...,
  0, 0])` from thread 0 before the MMA loop.

### 4) CuTeDSL GEMM trace-time errors: debug partition *semantics*, not types
If `@cute.jit` compilation fails inside `cutlass.cute.algorithm.gemm` (or
`cute.gemm`) with `TypeError`, `AssertionError`, or MLIR verification errors
involving fragment-shape mismatches (e.g., "expected size of mode 1 of A to
equal size of mode 1 of B", "C mode 0 mismatch", or similar), treat this as a
**fragment-shape mismatch caused by incorrect tiling/partition mapping**. In
practice, this usually means one of the `cute.local_tile(...)`,
`cute.local_partition(...)`, `thr_mma.partition_*(...)`, tiler tuple passed
to `local_tile`, or thread-fragment layouts is assigning the wrong logical
role to a dimension. For GEMM, explicitly verify that the fragments passed
to `cute.gemm(tiled_mma, acc, A_frag, B_frag, acc)` obey the contract:
- `A_frag` layout = `(MMA, MMA_M, MMA_K[, ...])`
- `B_frag` layout = `(MMA, MMA_N, MMA_K[, ...])`
- `acc`    layout = `(MMA, MMA_M, MMA_N)`

Do **not** just tweak dtypes blindly. Instead, inspect whether
1. the CTA tiler tuple (e.g., `cute.slice_(self.mma_tiler, (None, 0, None))`
   for A vs `(0, None, None)` for B vs `(None, None, 0)` for C) mapped B
   through the wrong axis,
2. the shared→compute partition for B (`make_fragment_B`) accidentally
   broadcasts or collapses K/N because `b_smem_layout_staged` was built with
   the wrong major mode,
3. the accumulator fragment shape from `tiled_mma.partition_shape_C(...)` is
   inconsistent with the MMA atom the `tiled_mma` was built for.

When fixing this class of bug, prefer the smallest structural correction:
repair the tiler slice tuple passed to `cute.local_tile`, repair the
`partition_A/B/C` choice, or rebuild `tiled_mma` / smem layouts with the
correct major mode from `LayoutEnum.from_tensor(...).mma_major_mode()`.
After the fix, explicitly state the resulting fragment shapes, e.g.
`tCrA.shape = (MMA, 2, 4, STAGE)`, `tCrB.shape = (MMA, 2, 4, STAGE)`,
`tCtAcc.shape = (MMA, 2, 2)`, and confirm that they satisfy the GEMM
contract before recompiling.
