# Kernel 0 Plan — DeepSeek Sparse Attention, staged round-0 (B200 / sm100a)

## 1. Problem Specification

### Operation
- Compute sparse attention for one query token at a time:
  - `logits[h, k] = sm_scale * (dot(q_nope[t, h, :], ckv[token_k, :]) + dot(q_pe[t, h, :], kpe[token_k, :]))`
  - `o[t, h, :] = softmax(logits[h, :]) @ ckv[selected_tokens, :]`
  - `lse[t, h] = logsumexp(logits[h, :]) / ln(2)`  (base-2 LSE, exactly like the PyTorch reference)

### Inputs / outputs
- `q_nope`: shape `[T, 16, 512]`, row-major in the last mode, target round-0 dtype `bf16`
- `q_pe`: shape `[T, 16, 64]`, row-major in the last mode, target round-0 dtype `bf16`
- `ckv_cache`: shape `[P, 64, 512]`, page-major, row-major in the last mode, target round-0 dtype `bf16`
- `kpe_cache`: shape `[P, 64, 64]`, page-major, row-major in the last mode, target round-0 dtype `bf16`
- `sparse_indices`: shape `[T, 2048]`, `int32`
- `sm_scale`: scalar `float32`
- `output`: shape `[T, 16, 512]`, `bf16`
- `lse`: shape `[T, 16]`, `float32`

### Fixed round-0 kernel constants
- `num_qo_heads = 16`
- `head_dim_ckv = 512`
- `head_dim_kpe = 64`
- `page_size = 64`
- `topk = 2048 = 32 * 64`

### Round-0 support contract
Round-0 is intentionally stricter than the PyTorch reference so the kernel can use page TMA directly.

- Valid entries in `sparse_indices[t]` must be **page-dense**:
  - for page slot `j`, the 64 entries must be
    `[(page_id_j << 6) + 0, (page_id_j << 6) + 1, ..., (page_id_j << 6) + 63]`
  - trailing invalid pages are encoded as 64 copies of `-1`
- Therefore round-0 supports **0 to 32 full pages per token**, not arbitrary token gathers inside a page.
- No duplicate valid pages.
- Valid pages must appear before all invalid pages.

This contract matches the intended “paged sparse attention” path and is the key simplification that makes a TMA-based kernel feasible in round-0. Inputs that violate it should be rejected by dispatch and sent to a fallback path.

### Correctness criteria
- For every supported token `t` and head `h`, the kernel must match the PyTorch reference semantics over the valid selected tokens.
- If a token has zero valid pages:
  - `output[t, :, :] = 0`
  - `lse[t, :] = -inf`
- `lse` must be emitted in base-2, not natural log.
- Only the first 16 logical rows are meaningful; any padded MMA rows must remain numerically neutral and must never be written back.

### Edge cases
- **All `-1` pages**: early zero-output / `-inf`-LSE path.
- **Partial valid count**: supported only in units of 64 tokens (whole pages).
- **Out-of-range page id**: reject at dispatch / debug assert in kernel bring-up.
- **Non page-dense indices**: reject in round-0.
- **Numerics**:
  - accumulate QK and PV in `fp32`
  - do softmax in `fp32`
  - cast final `O` to `bf16`


## 2. Launch Configurations

### CTA / grid shape
- **Cluster:** `(1, 1, 1)`
- **Block:** `256` threads = `8` warps
- **Grid:** persistent 1D over tokens
  - `grid.x = min(T, num_sms)`; on B200, `num_sms = 148`
  - each CTA loops: `token_idx = blockIdx.x; token_idx < T; token_idx += grid.x`

### Why no 2-CTA cluster in round-0
- Blackwell tcgen05 requires `M=128` for 2-CTA UMMA.
- The logical attention row count is only `16` heads.
- Padding `16 -> 64` for 1-CTA UMMA is already acceptable for round-0; padding `16 -> 128` for 2-CTA would double the waste again.
- So round-0 chooses **1 CTA + padded `M=64`**.

### Tiling
- **QK tile:** `(M, N, K) = (64, 64, 16)`
  - logical rows used: `16`
  - padded rows: `48`
  - page-aligned: one KV page per QK tile
- **PV tile:** `(M, N, K) = (64, 128, 16)`
  - one page provides `K=64` for PV
  - four `N=128` output subtiles cover the 512 output dimension

### Work partition per token
- Preload page table: `32` page ids max
- **Pass 1:** 32 QK page tiles → row max / row sum / final `lse`
- **Pass 2:** 32 QK page tiles again → normalized `P`
- For each page in pass 2:
  - compute one `P[64,64]`
  - run 4 PV subtiles with `V[64,128]`

### Occupancy expectation
- Shared memory target: ~184 KB / CTA
- TMEM target: full 512-column allocation (256 KB logical reservation due allocator granularity)
- Result: **1 resident CTA / SM** is the intended operating point


## 3. Warp Schedule

### Warp roles
| Warp | Role | Main work | Target reg budget |
|---|---|---|---:|
| W0 | page-table warp | decode / validate `sparse_indices`, stage 32 page ids | 48 |
| W1 | load warp | Q prologue copy, K/KPE TMA, V TMA | 64 |
| W2 | MMA warp | allocate TMEM, QK UMMA, PV UMMA, TMEM stores | 96-112 |
| W3 | softmax warp | pass-1 row stats, pass-2 probability generation, `lse` store | 144-160 |
| W4 | epilogue warp 0 | write `O[:, 0:128]` | 96-112 |
| W5 | epilogue warp 1 | write `O[:, 128:256]` | 96-112 |
| W6 | epilogue warp 2 | write `O[:, 256:384]` | 96-112 |
| W7 | epilogue warp 3 | write `O[:, 384:512]` | 96-112 |

### Per-pass responsibility
- **W0**
  - reads `sparse_indices[t]`
  - verifies page-dense contract in debug mode
  - extracts up to 32 `page_id`s and `valid_page_count`
- **W1**
  - loads padded Q buffers once per token
  - pass 1 / pass 2: loads one `Kc[64,512]` page and one `Kpe[64,64]` page for QK
  - pass 2: streams four `V[64,128]` subtiles for PV
- **W2**
  - owns TMEM allocation and descriptors
  - writes QK score tile to TMEM scratch
  - accumulates final `O[64,512]` in TMEM
- **W3**
  - reads TMEM score tile
  - pass 1: updates `row_max[16]`, `row_sum[16]`
  - pass 2: computes `P = exp2(score_log2 - lse2)` for the valid 16 rows, zero-fills padded rows
  - stores final `lse[16]`
- **W4-W7**
  - wait for final `O` ready signal
  - each warp reads one `128`-wide TMEM slice
  - stores only rows `[0:16]`

### Row / column ownership
- Logical attention rows (`16` heads) are always owned by W3 for softmax.
- Output dimension is split across 4 epilogue warps:
  - W4: `d = [0, 128)`
  - W5: `d = [128, 256)`
  - W6: `d = [256, 384)`
  - W7: `d = [384, 512)`

### Padding policy
- Q rows `[16:64)` are zero-filled in SMEM once.
- Softmax ignores padded rows and writes zero probabilities for them.
- Epilogue never writes padded rows back.


## 4. Async pipelining

### Pipeline summary
| Name | Type | Producer | Consumer | Stages | Tx count | Purpose |
|---|---|---|---|---:|---:|---|
| `page_pipe` | `PipelineCpAsync` | W0 full warp (32 threads) | W1 full warp (32 threads) | 1 | n/a | preload 32 page ids once per token |
| `k_pipe` | `PipelineTmaUmma` | W1 leader lane | W2 leader lane | 1 | `64*512*2 + 64*64*2 = 73,728 B` | one K/KPE page tile for QK |
| `score_pipe` | `PipelineUmmaAsync` | W2 leader lane | W3 full warp (32 threads) | 1 | n/a | TMEM score tile ready for softmax |
| `p_pipe` | `PipelineAsyncUmma` | W3 full warp (32 threads) | W2 leader lane | 1 | n/a | probability tile in SMEM ready for PV |
| `v_pipe` | `PipelineTmaUmma` | W1 leader lane | W2 leader lane | 2 | `64*128*2 = 16,384 B` | double-buffered V subtile stream for PV |
| `o_pipe` | `PipelineUmmaAsync` | W2 leader lane | W4-W7 full group (128 threads) | 1 | n/a | final O accumulator ready for writeback |

### Cooperative-group thread counts
- `PipelineCpAsync`: **full participating warp(s)**  
  Needed because page-id copies are explicit thread-level cp.async copies.
- `PipelineTmaUmma`: **leader only** on both producer and consumer sides  
  Matches the MLA reference pattern for TMA/UMMA pipelines.
- `PipelineUmmaAsync`: **producer leader only**, **consumer full warp/group**
- `PipelineAsyncUmma`: **producer full warp/group**, **consumer leader only**

### Named barriers
| Barrier | Threads | Use |
|---|---:|---|
| `tmem_ptr_sync_bar` | `32 * (1 MMA + 1 softmax + 4 epilogue) = 192` | TMEM allocation done; consumers may retrieve pointer |
| `pass_transition_bar` | `32 * (W1 + W2 + W3) = 96` | drain pass 1 and restart page loop for pass 2 |
| `o_ready_bar` | `32 * (W2 + W4 + W5 + W6 + W7) = 160` | final O is committed before epilogue warps read TMEM |

### Fence / ordering points
- `cpasync.prefetch_descriptor(...)` once at token start for K/KPE/V TMA descriptors
- `cute.arch.fence_view_async_tmem_store()` after W2 stores a score tile or the final O accumulator
- `cute.arch.fence_view_async_tmem_load()` after W3 / W4-W7 finish consuming TMEM views

### Overlap strategy
- **Page preload** overlaps with Q prologue copy.
- **Pass 1** has minimal overlap because Q consumes a large fixed SMEM footprint; `k_pipe` is single-stage by design.
- **Pass 2** is the important overlap point:
  - W2 computes QK for page `i`
  - W3 normalizes page `i`
  - W1 starts loading `V` subtiles for page `i`
  - while W2 runs PV on page `i`, W1 can preload the **next K page** for page `i+1` because the K buffer is free after QK
- **Epilogue** is not overlapped in round-0; it starts only after `o_pipe` is signaled.

### Why the kernel is 2-pass
- A fully online softmax+PV path would require correction/rescale machinery similar to the MLA example.
- Round-0 intentionally trades extra QK work for much simpler control flow:
  1. pass 1 computes `lse`
  2. pass 2 recomputes QK, forms normalized `P`, then runs PV

This keeps the warp graph simple and avoids extra softmax-correction pipelines.


## 5. Infrastructure

### MMA atoms
- **QK**
  - `tcgen05.CtaGroup.ONE`
  - `make_trivial_tiled_mma(ab_dtype=bf16, a_leading_mode=K, b_leading_mode=K, acc_dtype=f32, mma_tiler_mn=(64, 64))`
- **PV**
  - `tcgen05.CtaGroup.ONE`
  - `make_trivial_tiled_mma(ab_dtype=bf16, a_leading_mode=K, b_leading_mode=MN, acc_dtype=f32, mma_tiler_mn=(64, 128))`

### SMEM layouts
- `sQ_nope`: padded `[64, 512]`, K-major, 1 stage
- `sQ_pe`: padded `[64, 64]`, K-major, 1 stage
- `sK_nope`: `[64, 512]`, K-major, 1 stage
- `sK_pe`: `[64, 64]`, K-major, 1 stage
- `sV`: `[64, 128, 2]`, MN-major, 2 stages
- `sP`: `[64, 64]`, K-major, 1 stage
- `sPageIds`: `[32]` `int32`

Approximate SMEM footprint:
- Q: `64*512*2 + 64*64*2 = 73,728 B`
- K/KPE: `73,728 B`
- V double-buffer: `64*128*2*2 = 32,768 B`
- P: `64*64*2 = 8,192 B`
- page ids + barriers: small
- **Total target:** ~184 KB

### TMEM layouts
- `tS_tile`: one score scratch tile, logical `[64, 64]` `float32`
- `tO_accum`: final output accumulator, logical `[64, 512]` `float32`
- Allocate **full 512 columns** from `TmemAllocator`
  - needed because allocator granularity is power-of-two columns
  - 256 columns (128 KB) is too small for `tS_tile + tO_accum`
- Use `tcgen05.find_tmem_tensor_col_offset(...)` to place `tO_accum` after `tS_tile`

### TMA descriptors / atoms
- `tma_ckv_k`: page tile `[64, 512]` from `ckv_cache` into `sK_nope`
- `tma_kpe`: page tile `[64, 64]` from `kpe_cache` into `sK_pe`
- `tma_ckv_v`: subtile `[64, 128]` from a transposed/tiled `ckv_cache` view into `sV`
- Reuse the MLA pattern for a paged non-exec TMA atom, but simplify to:
  - no clusters
  - one page per K tile
  - one `128`-wide V subtile per load

### SharedStorage fields
- pipeline barrier arrays:
  - `page_pipe_mbar_ptr`
  - `k_pipe_mbar_ptr`
  - `score_pipe_mbar_ptr`
  - `p_pipe_mbar_ptr`
  - `v_pipe_mbar_ptr`
  - `o_pipe_mbar_ptr`
- named-barrier bookkeeping / TMEM holding buffer:
  - `tmem_holding_buf`
- SMEM tensors:
  - `smem_q_nope`
  - `smem_q_pe`
  - `smem_k_nope`
  - `smem_k_pe`
  - `smem_v`
  - `smem_p`
  - `smem_page_ids`
  - `valid_page_count`

### Minimal auxiliary state
- W3 registers:
  - `row_max[16]`
  - `row_sum[16]`
  - temp score fragments / probability fragments
- No round-0 split-K workspace
- No round-0 softmax exchange buffer between multiple softmax warps


## 6. Implementation Stages

### S0 — Reference harness + support-contract checker
- **Focus:** Section 1 / semantic ground truth and round-0 input filter
- **Depends on:** none
- **Connection:** Produces the page-dense support check and the PyTorch reference outputs used by all later stage validations.
- **Validation outputs**
  - `[sources: PyTorch] ref.valid_page_count` — **full**
  - `[sources: PyTorch] ref.page_ids` — **full** (32 ints max)
  - `[sources: PyTorch] ref.output_preview` — **preview** (first 100 flattened `output` elements)
  - `[sources: PyTorch] ref.lse` — **full** (16 elements)
  - `[sources: CuTeDSL] contract.page_dense_ok` — **full**
- **Consult APIs / helpers**
  - PyTorch reference: `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py`

### S1 — Persistent token scheduler skeleton
- **Focus:** Section 2 / launch and token work assignment
- **Depends on:** S0
- **Connection:** Supplies `token_idx` to every later stage; this is the outer loop that owns one full token result per CTA.
- **Validation outputs**
  - `[sources: CuTeDSL] cfg.block_threads` — **full**
  - `[sources: CuTeDSL] cfg.grid_x` — **full**
  - `[sources: CuTeDSL] sched.token_schedule_preview` — **preview** (first 32 token indices visited by CTA 0)
- **Consult APIs / helpers**
  - `cute.arch.block_idx`, `cute.arch.grid_dim`
  - optional pattern reference: `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_helpers.py`
  - optional generic scheduler: `references/cutlass/python/CuTeDSL/cutlass/utils/static_persistent_tile_scheduler.py`

### S2 — Define MMA atoms, SMEM layouts, TMEM plan
- **Focus:** Section 5 / MMA atom and layout definition
- **Depends on:** S1
- **Connection:** All load, MMA, and epilogue stages consume these atoms/layouts.
- **Validation outputs**
  - `[sources: CuTeDSL] infra.qk_mma_shape` — **full**
  - `[sources: CuTeDSL] infra.pv_mma_shape` — **full**
  - `[sources: CuTeDSL] infra.smem_bytes` — **full**
  - `[sources: CuTeDSL] infra.tmem_cols` — **full**
  - `[sources: CuTeDSL] infra.layout_cosizes` — **full**
- **Consult APIs / helpers**
  - `cutlass.utils.blackwell_helpers.make_trivial_tiled_mma`
  - `cutlass.utils.blackwell_helpers.make_smem_layout_a`
  - `cutlass.utils.blackwell_helpers.make_smem_layout_b`
  - `cutlass.utils.SmemAllocator`
  - `cutlass.utils.TmemAllocator`
  - `tcgen05.find_tmem_tensor_col_offset`

### S3 — Page-id preload warp
- **Focus:** Section 4 / `PipelineCpAsync` page-table preload
- **Depends on:** S0, S2
- **Connection:** Produces `smem_page_ids` and `valid_page_count`, consumed by the K/V load warp.
- **Validation outputs**
  - `[sources: PyTorch, CuTeDSL] stage3.page_ids` — **full**
  - `[sources: PyTorch, CuTeDSL] stage3.valid_page_count` — **full**
  - `[sources: CuTeDSL] stage3.padding_ok` — **full**
- **Consult APIs / helpers**
  - `pipeline.PipelineCpAsync`
  - `pipeline.make_pipeline_state`
  - `cute.make_copy_atom`
  - `cpasync.CopyG2SOp`
  - `cute.copy`

### S4 — Q prologue copy + K/KPE/V TMA load warp
- **Focus:** Section 4 / TMA producer role and staged operand loads
- **Depends on:** S2, S3
- **Connection:** Feeds QK and PV compute. Q prologue is required before either pass; K/KPE feed both passes; V feeds pass 2 PV.
- **Validation outputs**
  - `[sources: PyTorch, CuTeDSL] stage4.q_nope_preview` — **preview** (first 64 flattened elements)
  - `[sources: PyTorch, CuTeDSL] stage4.q_pe_preview` — **preview** (first 64 flattened elements)
  - `[sources: PyTorch, CuTeDSL] stage4.kc_page0_preview` — **preview** (first 64 flattened elements)
  - `[sources: PyTorch, CuTeDSL] stage4.kpe_page0_preview` — **preview** (first 64 flattened elements)
  - `[sources: PyTorch, CuTeDSL] stage4.v_page0_d0_preview` — **preview** (first 64 flattened elements)
- **Consult APIs / helpers**
  - `cpasync.prefetch_descriptor`
  - `cpasync.tma_partition`
  - `pipeline.PipelineTmaUmma`
  - MLA pattern: `make_paged_tiled_tma_atom` in `mla_decode_fp16.py`

### S5 — Pass-1 QK UMMA and row-stat accumulation
- **Focus:** Section 4 / `PipelineUmmaAsync` from QK MMA to softmax stats
- **Depends on:** S4
- **Connection:** Produces final `row_max[16]` and `row_lse2[16]`, consumed by pass 2 normalization.
- **Validation outputs**
  - `[sources: PyTorch, CuTeDSL] stage5.logits_page0_preview` — **preview** (first 64 flattened elements of the valid `16x64` slice)
  - `[sources: PyTorch, CuTeDSL] stage5.row_max` — **full** (16 elements)
  - `[sources: PyTorch, CuTeDSL] stage5.row_lse2` — **full** (16 elements)
- **Consult APIs / helpers**
  - `pipeline.PipelineUmmaAsync`
  - `tcgen05.make_tmem_copy`
  - `cute.arch.fence_view_async_tmem_store`
  - `cute.arch.fence_view_async_tmem_load`
  - base-2 softmax pattern from `mla_decode_fp16.py`

### S6 — Pass-2 probability tile and PV accumulation
- **Focus:** Section 4 / `PipelineAsyncUmma` + `PipelineTmaUmma` for P→PV
- **Depends on:** S4, S5
- **Connection:** Consumes `row_lse2`, recomputes normalized `P`, and fills the TMEM output accumulator consumed by epilogue.
- **Validation outputs**
  - `[sources: PyTorch, CuTeDSL] stage6.prob_page0_preview` — **preview** (first 64 flattened elements of the valid `16x64` slice)
  - `[sources: PyTorch, CuTeDSL] stage6.o_accum_page0_d0_preview` — **preview** (first 64 flattened elements of the valid `16x128` slice after the first page)
- **Consult APIs / helpers**
  - `pipeline.PipelineAsyncUmma`
  - `pipeline.PipelineTmaUmma`
  - `cute.copy`
  - `tcgen05.make_tmem_copy`
  - `tcgen05.make_umma_smem_desc`

### S7 — Final TMEM-to-global epilogue
- **Focus:** Section 3 / writeback warps and final output store
- **Depends on:** S6
- **Connection:** Emits the final kernel outputs and closes the loop against the PyTorch reference.
- **Validation outputs**
  - `[sources: PyTorch, CuTeDSL] stage7.output_preview` — **preview** (first 100 flattened output elements)
  - `[sources: PyTorch, CuTeDSL] stage7.lse` — **full** (16 elements)
  - `[sources: PyTorch, CuTeDSL] stage7.max_abs_diff_o` — **full** (scalar)
  - `[sources: PyTorch, CuTeDSL] stage7.max_abs_diff_lse` — **full** (scalar)
- **Consult APIs / helpers**
  - `pipeline.PipelineUmmaAsync`
  - `tcgen05.make_tmem_copy`
  - `cute.copy`
  - `cute.arch.fence_view_async_tmem_store`
  - `cute.arch.fence_view_async_tmem_load`
