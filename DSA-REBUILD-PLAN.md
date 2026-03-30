## DSA Rebuild Plan: FMHA-Like CuTeDSL Kernel With WG2 TMA Epilogue

### Summary
Redesign `solution/dsa_attention/kernel.py` around a new **FMHA-like fixed-CTA fused kernel** and treat the current split-shell as obsolete bring-up code.

Chosen defaults:
- Fixed CTA per `(token, split)`, no schedulers, no persistence.
- `block=[384,1,1]` with **3 warpgroups**:
  - WG0: sparse load / staging
  - WG1: QK + PV UMMA
  - WG2: softmax + epilogue
- Keep public APIs unchanged: `run(...)`, `kernel(...)`, `run.compile_only`.
- Keep Python `_combine_splits(...)` unchanged.
- Keep fallback only as a **temporary validation bridge** until fused is benchmark-green.
- WG2 uses **FMHA-style TMA store epilogue**, not regular global stores for `partial_o`.

### Kernel Architecture
- Replace the current self-consuming 128-thread split-shell with a new kernel that mirrors FMHA’s ownership model:
  - sparse load WG owns SMEM staging
  - MMA WG owns TMEM accumulators
  - softmax/epilogue WG consumes TMEM results and releases ownership explicitly
- Keep DSA-specific algorithmic differences:
  - sparse `tma_gather4` for CKV/KPE/V
  - decoupled `q_nope` and `q_pe`
  - shared CKV rows across all Q heads
  - one split is exactly one 64-column softmax problem
- Use **separate TMEM regions**:
  - `tStS`: QK score tile, later overwritten in-place with `P`
  - `tOtO`: PV output tile
- Keep `partial_lse` as direct scalar global stores from WG2.
- Use WG2 epilogue TMA store for `partial_o`:
  - TMEM `tOtO` -> WG2 RMEM -> WG2 `sO` staging -> `tma_atom_partial_o` -> global
  - follow FMHA’s `commit_group` / `wait_group` / release ordering
- Internal storage choice for `partial_o`:
  - use a **padded fused scratch output** shaped `[NUM_SPLITS, T, M_BLOCK, CKV_DIM]` for clean 2D TMA-store tiles
  - after kernel completion, slice rows `0:NUM_Q_HEADS` to produce the public `[NUM_SPLITS, T, NUM_Q_HEADS, CKV_DIM]`
  - public API stays unchanged

### Synchronization and Ownership
- Keep sparse gather completion on its own direct `gather_mbar` path because `tma_gather4` is custom inline PTX.
- Use explicit FMHA-style participant handles for all cross-warpgroup ownership:
  - `operand_pipe`: WG0 producer -> WG1 consumer after SMEM operand staging is complete
  - `score_pipe`: WG1 producer -> WG2 consumer for `tStS` readiness; WG2 releases after writing `P` back in-place
  - `v_pipe`: WG0 producer -> WG1 consumer for V operand staging
  - `output_pipe`: WG1 producer -> WG2 consumer for `tOtO` readiness; WG2 releases only after TMA store is committed and waited
- Do not use manual pipeline state objects in the rebuilt kernel.
- Do not let the same warpgroup both produce and consume the same UMMA pipeline stage.
- Use dedicated cooperative groups per warpgroup, not CTA-wide thread groups.

### Implementation Changes
#### 1. New Rebuild Checklist
- Create a new file `DSA_REBUILD_PROGRESS.md`.
- This file supersedes `DSA_PROGRESS.md` for the rebuild effort.
- Put the workflow protocol at the top of the file:
  - read `GPT-DSA-PLAN.md`, current `kernel.py`, and the relevant `blackwell/fmha.py` sections first
  - implement only the next unchecked micro-task
  - if a task is too large, split it into smaller unchecked tasks before coding
  - after every micro-task, run:
    - `python3 -m py_compile /home/mark123/projects/FlashMLA/solution/dsa_attention/kernel.py`
    - `.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --correctness-only`
  - if benchmark passes, check off the task and move on
  - if benchmark fails, inspect pipelines, TMA descriptors, SMEM/TMEM layouts, gather barriers, and TMA-store release ordering
  - stop after 3 failed fix attempts on the same task and report the exact failing ownership/synchronization issue

#### 2. New Micro-Task Sequence For `DSA_REBUILD_PROGRESS.md`
- [ ] Add the rebuild protocol header and mark the old split-shell path as deprecated but still shadow-only.
- [ ] Add new host-side fused scratch allocation helpers for padded `partial_o_tma` plus public-row slicing back to `[S, T, H, D]`.
- [ ] Add warpgroup role helpers and cooperative-group constructors for WG0/WG1/WG2 in `kernel.py`.
- [ ] Build a new FMHA-like split-kernel skeleton with `block=[384,1,1]`, shared storage, pipeline objects, and all-invalid early exit.
- [ ] Add WG0 sparse index load, validity masks, and `gather_mbar` handling for one split.
- [ ] Add WG0 Q-nope + CKV tile staging into operand SMEM and commit `operand_pipe`.
- [ ] Add WG1 QK CKV accumulation into dedicated score TMEM `tStS`.
- [ ] Add WG0 Q-pe + KPE staging and WG1 KPE accumulation into the same `tStS`.
- [ ] Add WG1 -> WG2 `score_pipe` handoff; WG2 loads scores from TMEM, applies mask and `sm_scale * LOG2E`, computes `partial_lse`, and writes BF16 `P` back in-place to TMEM.
- [ ] Add WG0 sparse V staging and `v_pipe` handoff for one output tile.
- [ ] Add WG1 PV MMA from in-place `P` TMEM and staged V into dedicated output TMEM `tOtO`.
- [ ] Extend PV across all 8 output tiles.
- [ ] Add WG2 TMEM->SMEM epilogue staging plus TMA store for one output tile into padded `partial_o_tma`.
- [ ] Extend WG2 epilogue TMA store across all 8 output tiles with FMHA-style `cp_async_bulk_commit_group` and `cp_async_bulk_wait_group` ordering.
- [ ] Shadow-launch the rebuilt fused kernel from `run()` while keeping fallback authoritative.
- [ ] Add fused-vs-fallback validation hook back around split partials and require it to pass the correctness benchmark.
- [ ] Switch `run()` to fused-authoritative and keep a temporary debug env escape hatch.
- [ ] Remove the obsolete dense fallback gather/softmax/SV path and remove the old split-shell implementation.

#### 3. WG2 Epilogue TMA Store Details
- Build a dedicated `tma_atom_partial_o` over the padded global output tensor.
- Allocate WG2 `sO` epilogue staging in SMEM sized for one output tile.
- WG2 flow per output tile:
  - wait on `output_pipe`
  - load `tOtO` from TMEM to RMEM
  - write RMEM values into `sO`, zeroing rows `NUM_Q_HEADS:M_BLOCK`
  - issue `cute.copy(tma_atom_partial_o, sO_tile, gO_tile)`
  - `cute.arch.cp_async_bulk_commit_group()`
  - `cute.arch.cp_async_bulk_wait_group(..., read=True)`
  - release the `output_pipe` handle only after the TMA store is safe to release
- Keep `partial_lse` out of the TMA-store path; store it directly from WG2.

### Test Plan
- Required after every micro-task:
  - `python3 -m py_compile /home/mark123/projects/FlashMLA/solution/dsa_attention/kernel.py`
  - `.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --correctness-only`
- Required correctness scenarios:
  - all-valid sparse indices
  - mixed valid / `-1` / out-of-range indices
  - fully invalid split
  - random top-k patterns across multiple tokens
- Required bring-up milestones:
  1. Kernel compiles and early-exit path passes benchmark
  2. QK-only handoff is stable
  3. WG2 softmax and `partial_lse` are correct
  4. PV UMMA is correct
  5. WG2 epilogue TMA store is correct
  6. Fused partials match fallback under validation
  7. Fused path becomes authoritative
  8. Fallback path is removed

### Public Interfaces
- No public signature changes.
- `run(...)` still returns `(output_bf16, lse_fp32)`.
- `kernel(...)` remains the in-place wrapper.
- `sparse_indices` remains `[T, TOPK]`.
- New internal-only scratch may be padded for TMA-store convenience; this is not exposed publicly.

### Assumptions
- Target remains Blackwell SM100/B200 only.
- Fixed compile-time shapes remain:
  - `TOPK=2048`
  - `CHUNK_SIZE=64`
  - `NUM_Q_HEADS=16`
  - `CKV_DIM=512`
  - `KPE_DIM=64`
  - `PAGE_SIZE=64`
- Correctness and a fully working fully CuTeDSL fused kernel are higher priority than minimizing temporary internal scratch or perfectly matching FMHA’s scheduler complexity.
