# DeepSeek Sparse Attention on B200 (sm100a) — final CuTeDSL kernel design

## 1. Scope and exact semantics

This design implements the baseline semantics of `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py`:

- `q_nope[t, h, 512]` and `q_pe[t, h, 64]`
- `ckv_cache[page, 64, 512]` used as both content-key and value
- `kpe_cache[page, 64, 64]`
- `sparse_indices[t, 2048]`, where each valid index is `page_idx * 64 + offset`, and `-1` is padding
- output `O[t, h, 512]` and base-2 `LSE[t, h]`

Important semantic note: attention is permutation-invariant over the selected KV set, so each CTA is allowed to reorder its assigned sparse indices before compute. That is the key enabler for TMA-friendly span packing.

## 2. Chosen kernel decomposition

I use a **two-kernel design**:

1. **Split-attention kernel**
   - computes one `(token, split_kv)` partial
   - uses TMA, tcgen05 UMMA, warp specialization, TMEM staging, and async pipelines
   - writes either final output (`split_kv == 1`) or normalized partial output + local LSE to workspace (`split_kv > 1`)

2. **Split reduction kernel**
   - combines the partial outputs across `split_kv`
   - computes final base-2 LSE and final output

There is no PyTorch/bootstrap path.

## 3. Work partition

### Main split-attention kernel

- **Grid**
  - `grid.x = num_tokens`
  - `grid.y = split_kv`
  - one CTA computes one `(token_idx, split_idx)`

- **CTA tile**
  - UMMA row tile `M = 64`
  - sparse-column microtile `N = 32`
  - QK K tile `K = 64`
  - PV output tile `D_tile = 128`

- **Why `M = 64`**
  - tcgen05 `CtaGroup.ONE` only allows `M in {64, 128}`; `M=64` is the smallest legal choice for BF16 UMMA.
  - the logical problem only has `16` heads, so each head row is **replicated 4x** into the 64-row UMMA tile.
  - all four replicas compute the same result; only replica 0 is stored.
  - this wastes compute but preserves a fully legal tcgen05 path and keeps the kernel simple.

- **Split-KV**
  - runtime `split_kv` is chosen from `{1, 2, 4, 8, 16}`
  - heuristic: smallest choice such that `num_tokens * split_kv >= #SM`, while keeping `2048 / split_kv >= 128`
  - each CTA owns `chunk_k = ceil_div(2048, split_kv)` sparse indices

- **Within a CTA**
  - the producer warp loads the chunk of sparse indices, removes `-1`, sorts valid token ids, and coalesces them into contiguous `(page, offset, length)` spans
  - spans are packed into 32-token microtiles
  - microtiles are processed sequentially

### Reduction kernel

- one CTA per token
- one warp reduces local LSEs
- remaining warps accumulate the weighted partial `O`

## 4. Warp specialization

### Main kernel: 6-warps / 192 threads

| Warp(s) | Role |
|---|---|
| warp 0 | producer warp: load sparse-index chunk, sort + span-pack, TMA prefetch, TMA load Q once, TMA load packed K/V tiles, TMA store partial/final output |
| warp 1 | UMMA warp: QK tcgen05 MMA, PV tcgen05 MMA, TMEM allocation/ownership |
| warps 2-3 | softmax warps: load `S` from TMEM, apply mask, running max/sum, write `P` to SMEM, emit correction metadata |
| warps 4-5 | correction/epilogue warps: rescale running `O` in TMEM, finalize normalized partial output, pack replica-0 rows, prepare TMA store buffer |

### Reduction kernel: 3-warps / 96 threads

| Warp(s) | Role |
|---|---|
| warp 0 | reduce local LSEs to global LSE and produce split scales |
| warp 1 | accumulate partial `O` tiles 0-255 |
| warp 2 | accumulate partial `O` tiles 256-511 and drive final TMA store |

## 5. Sparse-KV packing strategy

The producer warp makes the sparse gather TMA-friendly:

1. load `chunk_k` indices into SMEM
2. discard `-1`
3. sort valid token ids ascending
4. convert each id to `(page, offset)`
5. coalesce maximal contiguous spans inside the same page
6. greedily pack spans into 32-token microtiles

Each microtile has:

- `logical_n <= 32`
- a packed destination row range `[0, logical_n)`
- a descriptor list of spans

Padding rows `[logical_n, 32)` are filled from a static zero page so the K/V TMA stage always transfers a fixed byte count. That keeps `PipelineTmaUmma.tx_count` constant.

This design preserves semantics because attention is over a set, not an ordered sequence.

## 6. Tensor layouts and math tiles

### QK

- logical GEMM: `Q(16, 576)` × `K(576, topk_chunk)`
- implemented as:
  - `Q_nope(64, 512)` + `Q_pe(64, 64)` in replicated-head form
  - `Kc(32, 512)` + `Kpe(32, 64)` packed sparse microtile
- UMMA configuration:
  - `CtaGroup.ONE`
  - BF16 inputs
  - FP32 accumulate
  - tile `(M, N, K) = (64, 32, 64)`
  - `A major = K`
  - `B major = K`

### PV

- logical GEMM: `P(16, topk_chunk)` × `V(topk_chunk, 512)`
- implemented as:
  - `P(64, 32)` in BF16 SMEM
  - `V = Kc(32, 512)` reused from the packed K/V stage buffer
- UMMA configuration:
  - `CtaGroup.ONE`
  - BF16 inputs
  - FP32 accumulate
  - output tile `(64, 128)` with `K = 32`
  - `P major = K`
  - `V major = MN`

## 7. Memory flow

### Q tensors

- `q_nope`, `q_pe`: GMEM → TMA → SMEM (`sQ`)
- warp 0 loads the 16 physical head rows
- warp 0 replicates each head row 4x into the 64-row UMMA layout
- warp 1 reuses `sQ` for all sparse microtiles

### Sparse indices

- `sparse_indices`: GMEM → TMA/GMEM vector load → SMEM sort buffer
- warp 0 sorts and span-packs once per CTA

### K/V tensors

- `ckv_cache`, `kpe_cache`: GMEM → span-wise TMA → **double-buffered SMEM**
- same packed `ckv` buffer is used twice:
  - first as `Kc` for QK
  - then as `V` for PV
- `kpe_cache` is only used in QK

### Score tensor `S`

- QK UMMA writes `S` to TMEM stage ring
- softmax warps read `S` from TMEM into RMEM fragments

### Probability tensor `P`

- softmax warps compute `P` in RMEM
- `P`: RMEM → SMEM ring
- PV UMMA consumes `P` from SMEM

### Output accumulator `O`

- PV UMMA accumulates `O_partial` in TMEM
- correction warps rescale the prior `O_partial` in TMEM each microtile
- final normalized partial output is read from TMEM into RMEM, packed to SMEM store buffer, then TMA-stored to GMEM/workspace

### LSE

- running `(row_max, row_sum)` stays in RMEM during the split kernel
- final local `LSE = row_max + log2(row_sum)` is written to GMEM/workspace
- reduction kernel merges local LSEs into the final base-2 LSE

## 8. Async pipelines

### P0: Q prologue load

- **type:** `PipelineTmaAsync`
- **producer:** warp 0
- **consumer:** warp 1
- **stages:** 1
- **payload:** Q-nope + Q-pe for 16 heads
- **SMEM budget:** 72 KB (`64 x (512+64) x 2 B`, replicated form)
- **TMEM budget:** 0
- **register budget:** producer 64, consumer 96
- **notes:** after TMA, warp 0 replicates rows 4x and executes `fence_view_async_shared()` before commit

### P1: sparse K/V mainloop load

- **type:** `PipelineTmaUmma`
- **producer:** warp 0
- **consumer:** warp 1
- **stages:** 2
- **payload per stage:** packed `Kc(32,512)` + `Kpe(32,64)` = 36 KB
- **SMEM budget:** 72 KB ring
- **TMEM budget:** 0
- **register budget:** producer 64, consumer 96
- **notes:** stage is not released until both QK and PV are done, because `Kc` is reused as `V`

### P2: QK result handoff

- **type:** `PipelineUmmaAsync`
- **producer:** warp 1
- **consumer:** warps 2-3
- **stages:** 2
- **payload:** `S(64,32)` FP32 in TMEM
- **SMEM budget:** 0
- **TMEM budget:** 16 KB total
- **register budget:** producer 96, consumer 176

### P3: softmax-to-PV handoff

- **type:** `PipelineAsyncUmma`
- **producer:** warps 2-3
- **consumer:** warp 1
- **stages:** 2
- **payload:** `P(64,32)` BF16 in SMEM
- **SMEM budget:** 8 KB ring
- **TMEM budget:** 0
- **register budget:** producer 176, consumer 96

### P4: correction metadata handoff

- **type:** `PipelineAsync`
- **producer:** warps 2-3
- **consumer:** warps 4-5
- **stages:** 2
- **payload:** per-row `(row_sum, row_max_new, alpha, flags)`
- **SMEM/TMEM budget:** 16 KB aligned scratch
- **register budget:** producer 176, consumer 160

### P5: PV-output handoff

- **type:** `PipelineUmmaAsync`
- **producer:** warp 1
- **consumer:** warps 4-5
- **stages:** 1
- **payload:** running `O_partial(64,512)` in TMEM
- **SMEM budget:** 0
- **TMEM budget:** 128 KB
- **register budget:** producer 96, consumer 160

### P6: final/partial output store

- **type:** `PipelineTmaStore`
- **producer:** warp 0
- **stages:** 2
- **payload:** packed replica-0 output tile `(16,128)` BF16
- **SMEM budget:** 8 KB ring
- **TMEM budget:** 0
- **register budget:** producer 64

### Overlap summary

- warp 0 loads K/V tile `i+1` while warp 1 does QK/PV on tile `i`
- warps 2-3 do softmax on `S_i` while warp 1 can begin QK on `i+1`
- warps 4-5 rescale `O_i` while warp 1 computes later tiles
- warp 0 TMA-stores output tile `d+1` while warps 4-5 pack tile `d+2`

## 9. Shared-memory plan

Worst-case main-kernel SMEM allocation:

| Buffer | Size |
|---|---:|
| Q replicated staging | ~72 KB |
| K/V double buffer | ~72 KB |
| P double buffer | ~8 KB |
| output-store double buffer | ~8 KB |
| sorted-index + span descriptors | ~16 KB |
| pipeline mbarriers / scratch | ~4 KB |
| **total** | **~180 KB** |

This fits under the 228 KB opt-in limit and intentionally trades occupancy for data reuse.

Occupancy consequence:

- **1 CTA / SM**
- active warps per SM: 6
- enough parallelism is recovered by runtime `split_kv`

## 10. Tensor-memory plan

TMEM is allocated in three chunks:

| Columns | Use |
|---|---|
| 0-255 | running `O_partial(64,512)` FP32 |
| 256-287 | `S` double buffer |
| 288-319 | correction metadata / aligned scratch |

Total: **320 columns**, below the 512-column SM limit.

`O_partial` dominates TMEM usage; that is why the kernel is designed for one CTA per SM.

## 11. Online softmax / correction

Per microtile:

1. QK warp produces `S_i`
2. softmax warps compute:
   - `m_i = max(S_i)`
   - `m_new = max(m_prev, m_i)`
   - `alpha = exp2((m_prev - m_new) * sm_scale * log2(e))`
   - `P_i = exp2((S_i - m_new) * sm_scale * log2(e))`
   - `l_new = alpha * l_prev + sum(P_i)`
3. correction warps rescale running `O_prev` by `alpha`
4. PV warp accumulates `P_i @ V_i` into TMEM

At split end:

- `O_local = O_partial / l_local`
- `LSE_local = m_final + log2(l_local)`

If a split has no valid indices:

- `O_local = 0`
- `LSE_local = -inf`

## 12. Split reduction kernel

For `split_kv > 1`:

1. warp 0 reads all `LSE_local`
2. computes
   - `lse_max = max(LSE_local)`
   - `global_lse = lse_max + log2(sum(exp2(LSE_local - lse_max)))`
   - `scale_s = exp2(LSE_local[s] - global_lse)`
3. warps 1-2 accumulate
   - `O = sum_s scale_s * O_local[s]`
4. final `O` is TMA-stored, `global_lse` is scalar-stored

This is the same numerically stable merge used in Blackwell MLA reduction.

## 13. Synchronization and fences

- after mbarrier creation: `pipeline_init_wait()`
- after Q replication into SMEM: `fence_view_async_shared()` before `q_pipe.producer_commit`
- after packed K/V TMA issue: producer commits once the full 32-row microtile (including zero padding) is resident
- after softmax writes `P` to SMEM: `fence_view_async_shared()` before `p_pipe.producer_commit`
- after TMEM score loads: `fence_view_async_tmem_load()` before releasing `S` stage
- after correction metadata or `O` TMEM writes: `fence_view_async_tmem_store()` before commit
- `kv_pipe.consumer_release` happens only after PV finishes for that stage
- TMA-store pipeline uses acquire/commit per `(16,128)` output tile

## 14. Why this design matches B200 well

- uses **tcgen05 BF16 UMMA** for both QK and PV
- uses **TMA** for Q load, sparse K/V packed loads, and output/workspace stores
- uses **TMEM** for scores and running output
- uses **warp specialization** so producer, UMMA, softmax, and correction do not serialize
- uses **async pipelines** so sparse load, QK, softmax, correction, PV, and store overlap
- uses **split_kv** to recover machine-level parallelism despite 1-CTA/SM resource use

## 15. CuTeDSL / reference API map

| Design element | API / pattern | Reference |
|---|---|---|
| tcgen05 BF16 UMMA tile construction | `make_trivial_tiled_mma`, `MmaF16BF16Op`, legal `M/N` constraints | `references/cutlass/python/CuTeDSL/cutlass/utils/blackwell_helpers.py:845-926`, `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py:145-228`, `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py:567-648` |
| SMEM operand layouts | `make_smem_layout`, `make_smem_layout_a`, `make_smem_layout_b` | `references/cutlass/python/CuTeDSL/cutlass/utils/blackwell_helpers.py:583-744` |
| Generic TMA atom creation | `cpasync.make_tiled_tma_atom`, `tma_partition`, `prefetch_descriptor` | `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py:50-279` |
| UMMA-aware TMA atoms for operands | `make_tiled_tma_atom_A`, `make_tiled_tma_atom_B` | `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/helpers.py:41-164`, `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/helpers.py:168-291` |
| Paged / irregular TMA load pattern to emulate for sparse spans | `make_paged_tiled_tma_atom` example | `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:666-704` |
| Producer/consumer group sizing | `CooperativeGroup` | `references/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py:46-79` |
| Pipeline state init / cluster wait | `make_pipeline_state`, `pipeline_init_arrive`, `pipeline_init_wait` | `references/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py:652-720` |
| `TmaAsync` | `PipelineTmaAsync.create` | `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py:368-516` |
| `TmaUmma` | `PipelineTmaUmma.create` | `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm100.py:157-250` |
| `AsyncUmma` | `PipelineAsyncUmma.create` | `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm100.py:371-463` |
| `UmmaAsync` | `PipelineUmmaAsync.create` | `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm100.py:521-648` |
| pure async warp-to-warp handoff | `PipelineAsync.create` | `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py:153-206` |
| TMA store pipeline | `PipelineTmaStore.create` | `references/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py:730-776` |
| TMEM allocation | `TmemAllocator.allocate`, `wait_for_alloc`, `retrieve_ptr`, `relinquish_alloc_permit`, `free` | `references/cutlass/python/CuTeDSL/cutlass/utils/tmem_allocator.py:31-269` |
| TMEM copy helpers | `make_tmem_copy`, `make_umma_smem_desc` | `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/helpers.py:314-390` |
| TMEM load/store ops | `Ld32x32bOp`, `St32x32bOp`, etc. | `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py:99-471`, `references/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py:503-718` |
| async fences and register reconfiguration | `fence_view_async_tmem_load`, `fence_view_async_tmem_store`, `fence_view_async_shared`, `setmaxregister_increase`, `setmaxregister_decrease` | `references/cutlass/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py:768-848` |
| Shared-memory allocator | `SmemAllocator.allocate_array`, `allocate_tensor` | `references/cutlass/python/CuTeDSL/cutlass/utils/smem_allocator.py:31-255` |
| Blackwell MLA warp-specialized attention skeleton | shared storage, warp roles, pipelines, QK/PV/correction flow | `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:381-647`, `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:918-1260`, `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:1738-2187`, `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:2419-2673` |
| Decode attention TMA store pattern | TMA store atom + mixed pipeline allocation | `references/cutlass/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_decode.py:345-410`, `references/cutlass/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_decode.py:623-680` |
| Split reduction math | numerically stable base-2 LSE merge | `references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py:1308-1368` |
| Baseline semantics / edge cases | invalid `-1`, `Kc` reused as `V`, base-2 LSE | `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py:37-63` |

## 16. Final kernel summary

The final kernel is a **warp-specialized, split-KV, span-packed sparse attention kernel** for B200:

- `M=64` tcgen05 UMMA with 4x replicated heads
- `N=32` sparse microtiles
- Q loaded once, K/V packed and double-buffered
- `S` and running `O` in TMEM
- softmax/correction done online in FP32
- partial outputs merged by a second reduction kernel

This is the final CuTeDSL architecture I would implement for `kernel_0`.