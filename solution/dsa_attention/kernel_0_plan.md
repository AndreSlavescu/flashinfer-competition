# Kernel 0 plan: Deepseek sparse attention forward on B200 (sm100a)

## Problem contract and invariants

This kernel implements the exact forward semantics of `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py` for the fixed Deepseek decode shape:

- `num_qo_heads = 16`
- `head_dim_ckv = 512`
- `head_dim_kpe = 64`
- `page_size = 64`
- `topk = 2048`
- `sparse_indices[t, :]` stores token indices encoded as `page_idx * 64 + slot`, with `-1` meaning invalid

For each token `t`, the kernel computes:

1. `logits = q_nope[t] @ Kc^T + q_pe[t] @ Kp^T`
2. `logits_scaled = logits * sm_scale`
3. `lse[t, h] = logsumexp(logits_scaled[h, :]) / ln(2)`
4. `out[t, h, :] = softmax(logits_scaled[h, :]) @ V`

where `Kc` and `V` come from `ckv_cache`, and `Kp` comes from `kpe_cache`.

The round-0 design intentionally keeps the runtime path single-pass and single-kernel: there is no split-KV reducer kernel, no persistent tile scheduler, and no bootstrap fallback path in this plan.

## Kernel contract

- **Kernel granularity:** one CTA computes one token across all 16 query heads.
- **Grid:** `grid = (num_tokens, 1, 1)`.
- **Block:** `block = (224, 1, 1)` = 7 warps.
- **Architecture target:** Blackwell `sm100a` / B200 only.
- **Outputs:** `out[num_tokens, 16, 512]` in bf16 and `lse[num_tokens, 16]` in fp32 base-2 units.
- **Workspace:** no runtime workspace is required; validation mode adds optional debug GMEM surfaces only.
- **Launch requirement:** dynamic shared memory opt-in to about `154 KiB` per CTA.

The CTA uses padded MMA row extent `M_pad = 64` because Blackwell 1-CTA tcgen05 BF16 MMA requires `M in {64, 128}`. Only rows `0:16` are semantically live; rows `16:64` are forced to zero in Q and P staging and are ignored on store-out.

## Work partition

### CTA mapping

- `blockIdx.x -> token index t`
- one CTA covers the full 16-head row block for token `t`
- the sparse/top-k dimension is streamed inside the CTA in fixed tiles of `BLOCK_N = 32`

This gives `topk / BLOCK_N = 64` sparse tiles per token.

### MMA shapes

- **QK MMA tile:** `(M, N, K) = (64, 32, 16)`
- **PV MMA tile:** `(M, N, K) = (64, 128, 16)`
- **QK row padding:** 16 live heads + 48 zero rows
- **PV output slices:** 4 slices of width 128 to cover output dim 512

### Why this partition

The sparse gather is the dominant irregular component. A single CTA per token avoids inter-CTA LSE reduction and keeps sparse-index semantics easy to validate. `BLOCK_N = 32` is small enough to keep two gathered sparse stages resident in SMEM even after duplicating the ckv tile into a qk-friendly layout and a pv-friendly layout.

## Warp specialization

The CTA uses 7 warps:

- **W0: index warp**
  - prefetches TMA descriptors
  - issues TMA loads for `sparse_indices`
  - manages the index pipeline state
  - **register target:** `32` regs/thread
- **W1-W2: sparse gather warps**
  - consume staged sparse indices
  - decode `page = idx >> 6`, `slot = idx & 63`
  - gather `ckv_cache` rows and `kpe_cache` rows with non-bulk `cp.async`
  - materialize both qk-friendly and pv-friendly ckv layouts
  - write per-tile valid masks
  - **register target:** `96` regs/thread
- **W3: MMA warp**
  - performs QK tcgen05 MMA into TMEM logits
  - waits for softmax/correction completion
  - performs PV tcgen05 MMA into TMEM output accumulators
  - owns TMEM allocation lifetime
  - **register target:** `224` regs/thread
- **W4-W5: softmax warps**
  - load logits from TMEM to registers
  - apply `sm_scale * log2(e)` in fp32
  - update running row max / row sum for the 16 live rows
  - produce normalized `P` tile into SMEM
  - **register target:** `192` regs/thread
- **W6: correction + epilogue warp**
  - rescales TMEM output accumulators when the streaming row max changes
  - participates in the softmax->PV handoff
  - performs final TMEM-to-RMEM load, bf16 conversion, staged store, and TMA output store
  - **register target:** `128` regs/thread

The kernel uses `setmaxregister_decrease()` on W0-W2 and `setmaxregister_increase()` on W3-W5 after role dispatch so register pressure follows the warp-specialized work instead of the CTA maximum. There is no cluster-level specialization in kernel 0. `cta_group = tcgen05.CtaGroup.ONE` throughout, because the sparse gather is CTA-private and offers no profitable multicast reuse.

## Memory flow

### Q path

`q_nope` and `q_pe` are **not** staged as full padded `64 x K` tensors. Doing so would cost either 72 KiB of TMEM or 72 KiB of SMEM for data with an 18 KiB logical footprint. Instead:

1. W3 reloads one `K=16` slice of `q_nope` or `q_pe` at a time from GMEM.
2. The slice lands in a small SMEM A-buffer (`64 x 16`), with only rows `0:16` filled from GMEM and rows `16:64` zero-filled.
3. W3 immediately feeds that slice into tcgen05 QK MMA.

This deliberately trades a tiny amount of repeated load traffic for a large reduction in resident on-chip footprint. The 18 KiB Q working set stays L1-resident on B200 after the first few slice loads.

### Sparse index path

1. W0 TMA-loads `BLOCK_N = 32` indices from `sparse_indices[t, :]` into `sIdx[stage]`.
2. W1-W2 consume `sIdx[stage]`, build a `uint32` valid mask, and treat `-1` as invalid.

### K/V/KPE gather path

1. W1-W2 gather the selected `ckv_cache` rows into `sKc_qk[stage]`.
2. W1-W2 gather the selected `kpe_cache` rows into `sKp[stage]`.
3. W1-W2 transform `sKc_qk[stage]` into `sV_pv[stage]` with an SMEM-to-SMEM layout copy, so the ckv data is fetched from GMEM only once per sparse row.

`sKc_qk` is the canonical ckv landing buffer; `sV_pv` is the pv-friendly view used later by PV MMA.

### Logits / softmax / output path

1. W3 accumulates raw QK logits in TMEM tile `tS[64, 32]` in fp32.
2. W4-W5 load `tS` to registers, scale by `sm_scale * log2(e)`, apply the tile valid mask, and update streaming softmax state:
   - `m2[h] = running max in base-2 domain`
   - `l2[h] = running sum of exp2 shifted by m2`
3. W6 rescales the existing TMEM output accumulators `tO` by `alpha = exp2(m2_old - m2_new)` when the running max changes.
4. W4-W5 write the normalized probability tile `sP[64, 32]` in bf16.
5. W3 performs PV MMA from `sP` and `sV_pv`, accumulating into `tO[4][64][128]` in fp32.
6. After the last sparse tile, W6 converts the 4 TMEM output slices to bf16, stages each `16 x 128` live sub-tile in SMEM, and TMA-stores them to GMEM.
7. W6 stores final `lse` scalars directly to GMEM in fp32 base-2 units.

## Async pipelines

### P0: sparse-index TMA pipeline

- **Type:** `PipelineTmaAsync`
- **Producer:** W0
- **Consumer:** W1-W2
- **Depth:** 2 stages
- **Payload:** `32` sparse indices for the current token
- **SMEM budget:** `sIdx[2][32]` + barrier state = about `512 B`
- **TMEM budget:** none
- **Register budget:** W0 at `32`, W1-W2 at `96`
- **Reasoning:** the index payload is tiny, but double-buffering keeps the gather warps from stalling on TMA descriptor latency and makes the gather state machine deterministic.

### P1: sparse gather -> QK/PV pipeline

- **Type:** `PipelineAsyncUmma`
- **Producer:** W1-W2
- **Consumer:** W3
- **Depth:** 2 stages
- **Payload:** `sKc_qk[stage]`, `sV_pv[stage]`, `sKp[stage]`, and the tile valid mask
- **SMEM budget:** per stage = `32 KiB + 32 KiB + 4 KiB + mask`, double-buffered total about `136 KiB`
- **TMEM budget:** none
- **Register budget:** W1-W2 at `96`, W3 at `224`
- **Reasoning:** the sparse gather dominates memory latency, so this is the only deep data pipeline in kernel 0. While W3 computes tile `k`, W1-W2 gather tile `k+1`.

### P2: QK UMMA -> softmax/correction pipeline

- **Type:** `PipelineUmmaAsync`
- **Producer:** W3
- **Consumer:** W4-W5-W6
- **Depth:** 1 stage
- **Payload:** TMEM logits tile `tS`
- **SMEM budget:** only barrier state and row-stat scratch metadata
- **TMEM budget:** `tS[64,32]` = about `8 KiB`
- **Register budget:** W3 at `224`, W4-W5 at `192`, W6 at `128`
- **Reasoning:** only one logits tile needs to be live, because softmax/correction must finish before PV for the same sparse tile can start.

### P3: softmax/correction -> PV UMMA pipeline

- **Type:** `PipelineAsyncUmma`
- **Producer:** W4-W5-W6
- **Consumer:** W3
- **Depth:** 1 stage
- **Payload:** corrected TMEM output state + normalized `sP`
- **SMEM budget:** `sP[64,32]` = about `4 KiB`
- **TMEM budget:** `tO` stays live in place at about `128 KiB`; P3 only sequences access to it
- **Register budget:** W4-W5 at `192`, W6 at `128`, W3 at `224`
- **Reasoning:** PV can start as soon as two conditions are true: `tO` has been rescaled for the new running max, and the current probability tile is ready in SMEM.

### Prefetch policy

- W0 prefetches the sparse-index TMA descriptor and the output-store TMA descriptor at CTA start.
- W1-W2 use `cp.async` with streaming cache policy for sparse K/V/KPE rows.
- W3 does not reserve a full-Q staging buffer; it reissues small Q-slice loads that naturally hit in L1 after warmup.
- The final output store is a terminal TMA store sequence rather than a long-lived pipeline, so W6 uses `cp_async_bulk_commit_group()` / `cp_async_bulk_wait_group(read=True)` around the final `CopyBulkTensorTileS2GOp`.

## Shared-memory plan

Kernel 0 uses a single dynamic-SMEM slab carved by `SmemAllocator`. The live buffers are:

- `sIdx[2][32]` int32 = **256 B**
- `sValidMask[2]` uint32 = **8 B**
- `sKc_qk[2][32][512]` bf16 = **64 KiB**
- `sV_pv[2][32][512]` bf16 = **64 KiB**
- `sKp[2][32][64]` bf16 = **8 KiB**
- `sQ_ckv_phase[64][16]` bf16 = **2 KiB**
- `sQ_pe_phase[64][16]` bf16 = **2 KiB**
- `sP[64][32]` bf16 = **4 KiB**
- `sAlpha / row_max / row_sum` fp32 metadata = **< 256 B**
- `sO_store[16][128]` bf16 = **4 KiB**
- `launch/debug metadata` int32/fp32 scalars in validation mode = **< 512 B**
- mbarriers, named barriers, descriptor scratch, allocator bookkeeping = **~2 KiB**

**Total dynamic SMEM target:** about **154 KiB per CTA**.

This fits within B200's 228 KiB SMEM limit with opt-in, but it intentionally yields **1 CTA/SM** once TMEM usage is accounted for.

## Tensor-memory plan

TMEM holds only the tensors that truly benefit from tcgen05 accumulator locality:

- `tS[64, 32]` fp32 raw logits = **8 KiB**
- `tO[4][64, 128]` fp32 output accumulators = **128 KiB**

The live TMEM footprint is therefore about **136 KiB**. In byte terms that is roughly **272 TMEM columns**, but the Blackwell allocator in CuTeDSL allocates powers of two with 32-column granularity, so kernel 0 reserves **512 columns** via `TmemAllocator`. That makes TMEM the occupancy limiter and freezes the kernel at **1 CTA/SM**.

The important architectural choice is that full-Q residency is excluded from TMEM. If Q were kept as padded TMEM operands, the live footprint would still round to 512 columns and would crowd out useful softmax/output state.

## Synchronization and fences

- P0, P1, P2, and P3 each use their own mbarrier arrays.
- TMEM allocation uses a named barrier so W3 can allocate once and all warps can observe the pointer/state transition cleanly.
- W1-W2 call `fence_view_async_shared()` before committing P1 so W3 never sees a partially written sparse stage.
- W4-W5 call `fence_view_async_shared()` before committing P3 so W3 never sees a partially written `sP`.
- W4-W5 call `fence_view_async_tmem_load()` before releasing P2, ensuring the logits tile has been fully consumed from TMEM.
- W6 calls `fence_view_async_tmem_store()` before committing P3, ensuring the corrected `tO` values are globally visible to the PV MMA warp.
- W6 uses `cp_async_bulk_commit_group()` / `cp_async_bulk_wait_group(read=True)` around the output TMA stores before recycling `sO_store`.

There is no cluster barrier or multicast barrier in kernel 0.

## Validation strategy

Stage validation is cumulative. Each later validation entry point checks all earlier frontiers plus the new internal state introduced by the stage.

- **Stage S0:** export launch metadata and token-to-CTA mapping; validate shape guards, block size, buffer sizing, and TMEM reservation.
- **Stage S1:** export the first sparse tile's decoded indices, valid mask, gathered `ckv`, gathered `kpe`, and pv-layout `ckv` view.
- **Stage S2:** export the first sparse tile's raw QK logits (`16 x 32` live rows only) before softmax scaling.
- **Stage S3:** export streaming softmax state after tile 0:
  - `row_max_base2`
  - `row_sum_base2`
  - `alpha`
  - normalized `P`
- **Stage S4:** export the final fp32 TMEM output accumulator materialized to GMEM in validation mode, plus final fp32 base-2 `lse`.
- **Stage S5:** compare the final bf16 `out` and fp32 `lse` against the eager PyTorch reference.

Validation mode never changes the math path; it only adds GMEM materialization of otherwise internal tensors.

## CuTeDSL / Blackwell API map

- **TMA load/store atoms and descriptor handling**
  - `cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp`
  - `cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp`
  - `cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom`
  - `cutlass.cute.nvgpu.cpasync.tma_partition`
  - `cutlass.cute.nvgpu.cpasync.prefetch_descriptor`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/copy.py:142-193`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/copy.py:390-429`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py:50-279`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:437-449`

- **Irregular sparse GMEM->SMEM gather**
  - `cutlass.cute.nvgpu.cpasync.CopyG2SOp`
  - `cutlass.cute.nvgpu.cpasync.LoadCacheMode`
  - `cute.make_copy_atom`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/copy.py:36-99`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:1455-1483`

- **Blackwell tcgen05 MMA construction and staged SMEM layouts**
  - `cutlass.utils.blackwell_helpers.make_trivial_tiled_mma`
  - `cutlass.utils.blackwell_helpers.make_smem_layout_a`
  - `cutlass.utils.blackwell_helpers.make_smem_layout_b`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/utils/blackwell_helpers.py:631-685`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/utils/blackwell_helpers.py:689-744`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/utils/blackwell_helpers.py:845-926`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:376-424`

- **Blackwell MMA enums / runtime fields**
  - `cutlass.cute.nvgpu.tcgen05.OperandMajorMode`
  - `cutlass.cute.nvgpu.tcgen05.OperandSource`
  - `cutlass.cute.nvgpu.tcgen05.CtaGroup`
  - `cutlass.cute.nvgpu.tcgen05.Field`
  - `cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py:63-155`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py:567-648`

- **TMEM copy helpers used by softmax/correction/epilogue**
  - `cutlass.cute.nvgpu.tcgen05.copy.Ld32x32bOp`
  - `cutlass.cute.nvgpu.tcgen05.copy.St32x32bOp`
  - `cutlass.cute.nvgpu.tcgen05.copy.St16x256bOp`
  - `cutlass.cute.nvgpu.tcgen05.copy.Repetition`
  - `cutlass.cute.nvgpu.tcgen05.make_tmem_copy`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py:45-57`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py:370-405`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py:630-712`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/helpers.py:315-326`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:1886-1912`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:2157-2176`

- **Pipeline classes**
  - `cutlass.pipeline.PipelineTmaAsync`
  - `cutlass.pipeline.PipelineAsyncUmma`
  - `cutlass.pipeline.PipelineUmmaAsync`
  - `cutlass.pipeline.PipelineAsync`
  - `cutlass.pipeline.CooperativeGroup`
  - `cutlass.pipeline.PipelineState`
  - `cutlass.pipeline.NamedBarrier`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py:368-470`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm100.py:304-470`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm100.py:481-620`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py:46-79`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py:395-470`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py:555-620`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:653-720`

- **Shared / tensor memory allocators**
  - `cutlass.utils.smem_allocator.SmemAllocator`
  - `cutlass.utils.tmem_allocator.TmemAllocator`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/utils/smem_allocator.py:31-194`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/utils/tmem_allocator.py:31-309`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:836-840`
  - **Example region:** `references/quack/quack/gemm_sm100.py:846-857`

- **Warp identity, register shaping, fences**
  - `cute.arch.warp_idx`
  - `cute.arch.make_warp_uniform`
  - `cute.arch.setmaxregister_increase`
  - `cute.arch.setmaxregister_decrease`
  - `cute.arch.fence_view_async_shared`
  - `cute.arch.fence_view_async_tmem_load`
  - `cute.arch.fence_view_async_tmem_store`
  - `cute.arch.cp_async_bulk_commit_group`
  - `cute.arch.cp_async_bulk_wait_group`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/arch/elect.py:21-35`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py:127-139`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py:557-573`
  - **Code region:** `references/cutlass/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py:804-847`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:801-808`
  - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:1352-1410`

- **Blackwell example patterns that directly inform this design**
  - FMHA warp-specialized QK -> softmax/correction -> PV flow:
    - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:376-424`
    - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:653-720`
    - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/fmha.py:1322-1420`
  - MLA page-table async load and irregular gather patterns:
    - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:422-520`
    - **Example region:** `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:1430-1483`

## Staged round-0 implementation plan

### S0. Kernel shell, tile mapping, and resource reservation

Implement one CTA per token with `224` threads and fixed tile shapes `QK=(64,32,16)` and `PV=(64,128,16)`. The kernel owns no runtime workspace, but it does opt into about `154 KiB` of dynamic SMEM and reserves `512` TMEM columns because the live `tS + tO` footprint rounds up to the full Blackwell allocation quantum. W3 owns the `TmemAllocator` lifetime through a named barrier, and validation mode exports launch metadata so `run_stage_validation` can check block size, grid mapping, SMEM bytes, and TMEM column count before any math is turned on.

### S1. Sparse-index TMA and irregular KV/KPE gather

Wire P0 and P1. W0 TMA-loads `32` sparse indices per stage into `sIdx[2]`. W1-W2 decode page/slot, build the tile valid mask, gather `ckv_cache` once into `sKc_qk[2]`, gather `kpe_cache` into `sKp[2]`, and then derive the pv-friendly `sV_pv[2]` view from the canonical `sKc_qk[2]` buffer with an SMEM layout transform. Invalid indices are zero-filled in every gathered tensor and cleared in the valid mask.

### S2. QK tcgen05 mainloop into TMEM logits

Implement the W3 QK mainloop over the current sparse tile. For each `K=16` slice of `q_nope` and `q_pe`, W3 loads a small padded `64 x 16` A-buffer, zero-fills rows `16:64`, and issues tcgen05 BF16 MMA against `sKc_qk` and `sKp`. The result is a raw fp32 logits tile `tS[64,32]` in TMEM; only rows `0:16` are later observed.

### S3. Streaming softmax, online correction, and P materialization

Wire P2 and P3. W4-W5 load `tS` from TMEM, multiply by `sm_scale * log2(e)`, apply the tile valid mask, and update running `row_max_base2` and `row_sum_base2` for the 16 live heads. W6 computes the correction factor `alpha = exp2(m_old - m_new)` and rescales the TMEM output accumulators before PV. W4-W5 then materialize the normalized `P` tile into `sP[64,32]` in bf16, with padded rows forced to zero.

### S4. PV tcgen05 accumulation into TMEM output

After P3 commits, W3 consumes `sP` and `sV_pv` and issues four PV MMA slices of width `128`, accumulating into `tO[4][64,128]` in fp32. The gather stage is released only after the PV updates for the tile complete, so the double-buffered sparse staging remains correct under overlap.

### S5. Epilogue store and validation exports

After the last sparse tile, W6 finalizes `lse = row_max_base2 + log2(row_sum_base2)` for live rows, handles the `row_sum_base2 == 0` all-invalid case, converts the four TMEM output slices to bf16, stages each live `16 x 128` tile in `sO_store`, and TMA-stores them to GMEM. Validation mode additionally exports the selected cumulative frontier tensors: launch metadata, gathered tile 0, raw logits tile 0, softmax state tile 0, final fp32 accumulator, and final fp32 base-2 LSE before checking final bf16 `out`.