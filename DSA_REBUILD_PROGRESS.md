- [x] Rebuild protocol:
  Read `GPT-DSA-PLAN.md`, current `solution/dsa_attention/kernel.py`, and the relevant `csrc/cutlass/examples/python/CuTeDSL/blackwell/fmha.py` sections first.
  Implement only the next unchecked micro-task.
  If a task is too large, split it into smaller unchecked tasks before coding.
  After every micro-task, run:
  `python3 -m py_compile /home/mark123/projects/FlashMLA/solution/dsa_attention/kernel.py`
  `.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --correctness-only`
  If the benchmark passes, check off the task and move on.
  If the benchmark fails, inspect pipelines, TMA descriptors, SMEM/TMEM layouts, gather barriers, and TMA-store release ordering.
  Stop after 3 failed fix attempts on the same task and report the exact failing ownership/synchronization issue.

- [x] Add the rebuild protocol header and mark the old split-shell path as deprecated but still shadow-only.
- [x] Add new host-side fused scratch allocation helpers for padded `partial_o_tma` plus public-row slicing back to `[S, T, H, D]`.
- [x] Add warpgroup role helpers and cooperative-group constructors for WG0/WG1/WG2 in `kernel.py`.
- [x] Build a new FMHA-like split-kernel skeleton with `block=[384,1,1]`, shared storage, pipeline objects, and placeholder-output early exit behavior.
- [x] Add WG0 sparse index load and validity masks for one split.
- [x] Add WG0 `gather_mbar` initialization and phase management for one split.
- [x] Add WG0 Q-nope tile staging into operand SMEM for one CKV tile.
- [x] Add WG0 sparse CKV gather staging into operand SMEM for one CKV tile.
- [x] Add WG0 `operand_pipe` producer handoff for one CKV tile.
- [x] Add WG1 score TMEM allocation and pointer retrieval for `tStS`.
- [x] Add WG1 `operand_pipe` consumer wait/release for one CKV tile.
- [x] Add WG1 one-tile QK CKV UMMA into dedicated score TMEM `tStS`.
- [x] Add WG0 Q-pe + KPE staging and WG1 KPE accumulation into the same `tStS`.
- [x] Add WG1 -> WG2 `score_pipe` ownership handoff and WG2 wait/release skeleton around `tStS`.
- [x] Add WG2 TMEM score load into scratch and apply sparse masking plus `sm_scale * LOG2E`.
- [x] Add WG2 row-max / row-sum reduction and write `partial_lse`.
- [x] Add dedicated PV-compatible `P` TMEM region/allocation bookkeeping alongside score TMEM `tStS`.
- [x] Add WG2 BF16 `P` write-back from softmax scratch into SMEM `sP` (switched PV MMA A-operand from TMEM to SMEM source, following mixed_input_fmha_decode pattern).
- [x] Add WG0 sparse V staging and `v_pipe` handoff for one output tile.
- [x] Add WG1 PV MMA from `sP` SMEM and staged V into reused score TMEM as `tOtO` (no TMEM reallocation — reuse same allocation).
- [ ] Extend PV across all 8 output tiles.
- [ ] Add WG2 TMEM->SMEM epilogue staging plus TMA store for one output tile into padded `partial_o_tma`.
- [ ] Extend WG2 epilogue TMA store across all 8 output tiles with FMHA-style `cp_async_bulk_commit_group` and `cp_async_bulk_wait_group` ordering.
- [ ] Shadow-launch the rebuilt fused kernel from `run()` while keeping fallback authoritative.
- [ ] Add fused-vs-fallback validation hook back around split partials and require it to pass the correctness benchmark.
- [ ] Switch `run()` to fused-authoritative and keep a temporary debug env escape hatch.
- [ ] Remove the obsolete dense fallback gather/softmax/SV path and remove the old split-shell implementation.

## Helpful Sync Notes

- `https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html`
  Key takeaway: for `PipelineUmmaAsync`, the full-barrier arrive count comes from `producer_group.size`, while the empty-barrier arrive count comes from the async consumer group. For our `score_pipe`, that means a single-issuer producer group is correct, but the WG2 async consumer group should still release with its full 128-thread participation.
- `references/learn-cuda/02e_matmul_sm100/matmul_v3.cu`
  Key takeaway: raw Blackwell `tcgen05.commit...arrive::one` and TMA issue patterns are single-issuer operations. This was the clearest reference that the TMEM ownership signal should come from one producer warp/thread path, not all four MMA warps.
- `csrc/cutlass/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_decode.py`
  Key takeaway: the MMA-to-softmax handoff uses a single UMMA producer path and a larger async consumer group. Mirroring that shape fixed the deadlock in our `score_pipe` skeleton once we kept the handle SSA uniform but restricted `commit()` to the lead MMA warp.
  Secondary takeaway: the TMEM consumer-side copy path uses `get_slice(warpgroup_tidx)` because the TMEM load path is a 128-thread warpgroup operation, which matched the WG2 score-load step.
- `csrc/cutlass/examples/python/CuTeDSL/blackwell/fmha.py`
  Key takeaway: FMHA encode uses `p_source = tcgen05.OperandSource.TMEM` for PV MMA, but this requires complex TMEM P store views with `St32x32bOp` that are very sensitive to TV-layout shape. For our 64×64 tile sizes the composed layout produced 3D shapes incompatible with the store op.
- `csrc/cutlass/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_decode.py`
  Key takeaway for P staging: the decode FMHA stores P to **SMEM** (not TMEM), then the PV MMA reads P from SMEM. This avoids the TMEM store layout problem entirely. We adopted this approach: PV MMA A-operand now comes from SMEM `sP`, WG2 converts FP32→BF16 into sQrow then stages to sP via `copy_rowmajor_stage_to_operand_a`.
- `https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html`
  Key takeaway: `find_tmem_tensor_col_offset(...)` should be fed a TMEM fragment/view that matches the physical data-path layout we are sizing. For the new `P` region, the useful fake object was the shape-based `pv_tiled_mma.make_fragment_A(p_tmem_layout_staged.outer.shape)` pattern, not a tensor-based fragment built from a concrete TMEM pointer.

## Resolved Blockers

- TMEM P write-back blocker (resolved by switching to SMEM P): The `tcgen05.make_tmem_copy` store path for `St32x32bOp(Repetition(32))` required a 2D `(32,32):(1,65536)` TV-layout, but our 64×64 tile composed to `(32,16,2):(1,65536,2097152)`. Fix: adopted decode FMHA pattern — P goes through SMEM instead of TMEM.
