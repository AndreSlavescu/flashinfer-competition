# Deepseek Sparse Attention `kernel_0` Round-0 Architecture Plan

This document proposes the first CuTeDSL kernel architecture for the Deepseek sparse-attention forward pass on NVIDIA B200 / SM100a. The goal of round-0 is **correctness-first fused execution** with a warp-specialized, persistent kernel that already matches the CuTeDSL / Blackwell programming model, while deliberately deferring riskier optimizations such as page-regrouped TMA gathers or multi-CTA split-K.

The main design choice for round-0 is:

- **one CTA processes one query token**
- **M dimension is padded from 16 heads to 64 rows** so Blackwell UMMA can be used directly
- **sparse KV accesses use decoded page+slot gathers, not TMA**, because `sparse_indices` are token-granular and non-affine
- **O and logits stay in TMEM**, so GMEM traffic is almost entirely input gathers + final writeback

This is not the final performance-optimal design, but it is the cleanest architecture that is both implementable in CuTeDSL and faithful to the baseline semantics.

---

## Problem Specification

### Kernel semantics

For each query token `t` and query/output head `h`, the kernel computes:

\[
\text{logit}_{h,j} =
\langle q\_nope[t,h,:], ckv[sparse\_indices[t,j], :] \rangle +
\langle q\_pe[t,h,:], kpe[sparse\_indices[t,j], :] \rangle
\]

\[
\text{scaled}_{h,j} = \text{logit}_{h,j} \cdot sm\_scale
\]

\[
lse[t,h] = \log_2 \left( \sum_j e^{\text{scaled}_{h,j}} \right)
= \log_2 \sum_j 2^{\text{scaled}_{h,j}\cdot \log_2 e}
\]

\[
output[t,h,:] = \sum_j \text{softmax}(\text{scaled}_{h,:})_j \cdot ckv[sparse\_indices[t,j], :]
\]

Only indices `!= -1` are valid. `ckv_cache` supplies both the non-positional key term and the value term; `kpe_cache` supplies only the positional key term.

Implementation detail: like the Blackwell MLA example, the kernel should carry

- `sm_scale_log2 = sm_scale * LOG2_E`
- `exp2(...)` / `log2(...)`

through the online softmax path, then write `lse` directly in base-2.

### Inputs / outputs

| Tensor | Shape | Dtype | Layout / stride requirements | Notes |
|---|---:|---|---|---|
| `q_nope` | `[T, 16, 512]` | `bf16` | contiguous in last dim | non-positional query |
| `q_pe` | `[T, 16, 64]` | `bf16` | contiguous in last dim | positional query |
| `ckv_cache` | `[P, 64, 512]` | `bf16` | contiguous in last dim | page-major KV/value cache |
| `kpe_cache` | `[P, 64, 64]` | `bf16` | contiguous in last dim | page-major positional key cache |
| `sparse_indices` | `[T, 2048]` | `int32` | contiguous in last dim | flattened token indices, `-1` = invalid |
| `sm_scale` | scalar | `float32` | host scalar / kernel arg | applied to logits |
| `output` | `[T, 16, 512]` | `bf16` | contiguous in last dim | final attention output |
| `lse` | `[T, 16]` | `float32` | contiguous in last dim | base-2 log-sum-exp |

### Fixed round-0 constants

Round-0 intentionally hard-codes the baseline problem class:

- `num_qo_heads = 16`
- `head_dim_ckv = 512`
- `head_dim_kpe = 64`
- `page_size = 64`
- `topk = 2048`

Derived constants used by the kernel:

- `page_bits = 6`
- `sparse_tile_n = 32`
- `num_sparse_tiles = 2048 / 32 = 64`
- `qk_k_atom = 16` for BF16 UMMA
- `qk_ckv_iters = 512 / 16 = 32`
- `qk_kpe_iters = 64 / 16 = 4`
- `pv_k_iters = 32 / 16 = 2`
- `pv_out_tile = 128`
- `pv_out_iters = 512 / 128 = 4`

### Layout / addressing model

The baseline semantics flatten the paged cache:

- `flat_tok_idx = page_idx * 64 + slot`
- `page_idx = flat_tok_idx >> 6`
- `slot = flat_tok_idx & 63`

The round-0 kernel does **not** reorder `sparse_indices`; it preserves the exact sparse ordering presented by the input tensor. That keeps semantics simple and supports repeated indices.

### Correctness criteria

The CuTeDSL kernel is correct iff:

1. For every valid sparse index, logits match the baseline to FP32 accumulation tolerance.
2. Every `-1` index contributes exactly zero probability mass and zero output contribution.
3. `lse` is returned in **base-2**, not natural log.
4. `output` is written as `bf16`, with accumulation performed in `float32`.
5. Repeated sparse indices behave exactly like repeated columns in the baseline.
6. Invalid padded heads/rows introduced only for UMMA tiling (`rows 16..63`) do not affect any stored result.

### Edge cases

1. **All indices invalid for a token**
   - `output[t, :, :] = 0`
   - `lse[t, :] = -inf`

2. **Mixed valid / invalid indices within the same 32-column sparse tile**
   - gather invalid rows as zeros
   - apply the validity mask before row-max / row-sum reduction so those columns behave as `-inf`

3. **Repeated indices**
   - preserve ordering and multiplicity; no deduplication

4. **Out-of-range non-negative index**
   - treated as invalid input in release builds; debug path should assert `idx < P * 64`

### Intentional round-0 limitations

- No split-K across CTAs; no secondary reduction kernel
- No cluster multicast / no 2-CTA UMMA
- No TMA in the sparse mainloop
- Heads are padded from `16 -> 64` rows to satisfy SM100a UMMA tile requirements

These are acceptable for round-0 because they dramatically reduce implementation risk while preserving all semantics.

---

## Launch Configurations

### CTA / cluster / grid shape

Round-0 uses:

- **cluster shape**: `(1, 1, 1)`
- **threads per CTA**: `256` (`8` warps)
- **grid shape**: persistent 1D token scheduler, launched as `(1, 1, min(T, 148))`
- **min_blocks_per_mp**: `1`

Why:

- irregular sparse gathers eliminate most benefits of 2-CTA cluster multicast
- the kernel consumes large SMEM and effectively a full TMEM allocation, so it naturally runs at **1 CTA / SM**
- one CTA per token is the cleanest mapping because sparse neighborhoods are token-specific and not reusable across CTAs

### Work partition

One CTA owns one logical query token:

- CTA work item = token `t`
- CTA computes all `16` heads for that token
- CTA loops over all `64` sparse tiles (`32` sparse columns each)
- CTA writes `output[t, :, :]` and `lse[t, :]`

The M dimension is padded:

- logical heads: `16`
- UMMA rows: `64`
- rows `16..63` are zero-padded and masked off at writeback

### Persistent scheduler

Use the generic scheduler in `cutlass.utils.static_persistent_tile_scheduler`:

- `problem_shape_ntile_mnl = (T, 1, 1)`
- `cluster_shape_mnk = (1, 1, 1)`
- `swizzle_size = 1`
- `raster_along_m = True`

Per-CTA token selection:

1. `scheduler = StaticPersistentTileScheduler.create(...)`
2. `work = scheduler.initial_work_tile_info()`
3. `token_idx = work.tile_idx[0]`
4. loop until `work.is_valid_tile == False`

Because the per-token workload is fixed (`2048` sparse entries), persistent scheduling is simple and naturally load-balanced.

### Tile sizes

| Quantity | Value | Rationale |
|---|---:|---|
| `M_tile_qk` | `64` | smallest BF16 SM100a UMMA M tile |
| `N_tile_sparse` | `32` | balances sparse gather cost, P/V staging cost, and SMEM footprint |
| `K_atom_qk` | `16` | SM100a BF16 UMMA instruction K |
| `M_tile_pv` | `64` | same padded row tile as QK |
| `N_tile_pv_out` | `128` | reasonable output-dim tile for `512`-wide V |
| `K_tile_pv` | `16` | BF16 UMMA instruction K |

### Resource budget and occupancy assumptions

Approximate per-CTA shared-memory budget:

- padded `Q_nope`: `64 x 512 x 2B = 64 KB`
- padded `Q_pe`: `64 x 64 x 2B = 8 KB`
- double-buffered `Kc` tile: `2 x (32 x 512 x 2B) = 64 KB`
- double-buffered `Kpe` tile: `2 x (32 x 64 x 2B) = 8 KB`
- double-buffered `P` tile: `2 x (64 x 32 x 2B) = 8 KB`
- double-buffered `V[128]` slice tile: `2 x (32 x 128 x 2B) = 16 KB`
- page / offset / valid-mask ring + correction metadata + mbarriers: `< 4 KB`

Total: roughly `168-176 KB`, which implies **1 CTA / SM** on B200's `228 KB` SMEM budget.

Approximate TMEM use:

- QK logits `S`: `64 x 32 x 2 stages x 4B = 16 KB`
- O accumulator: `64 x 512 x 4B = 128 KB`
- total logical use: `144 KB`
- `utils.get_num_tmem_alloc_cols(..., rounding=True)` is expected to round this to a **512-column TMEM allocation**

This is acceptable because the kernel is already constrained to one resident CTA per SM.

---

## Warp Schedule

### Warp roles

Round-0 CTA = 8 warps:

| Warp | Sub-core (`warp_id % 4`) | Role | Main responsibilities | Target reg cap |
|---|---:|---|---|---:|
| `W0` | `0` | softmax warp | consume `S` TMEM, apply sparse mask, update `row_max/row_sum`, quantize/store `P`, publish correction metadata | `~192` |
| `W1` | `1` | correction / epilogue warp | rescale `O` TMEM with correction factor, normalize final output, store `output/lse` | `~160` |
| `W2` | `2` | UMMA warp | allocate TMEM, run QK UMMA and PV UMMA, manage `ACCUMULATE` flags | `~96` |
| `W3` | `3` | page decode warp | load `sparse_indices`, decode `page_id/slot`, publish valid-mask ring | `~64` |
| `W4` | `0` | QK gather warp A | gather rows `0..15` of `Kc/Kpe` into staged QK SMEM | `~80` |
| `W5` | `1` | QK gather warp B | gather rows `16..31` of `Kc/Kpe` into staged QK SMEM | `~80` |
| `W6` | `2` | PV gather warp A | gather rows `0..15` of `V[:, d:d+128]` into staged PV SMEM | `~80` |
| `W7` | `3` | PV gather warp B | gather rows `16..31` of `V[:, d:d+128]` into staged PV SMEM | `~80` |

### Prologue / steady-state / epilogue ownership

#### Prologue

- `W3-W7` cooperatively load `Q_nope` and `Q_pe` for the token
- rows `16..63` are zero-filled
- `W2` allocates TMEM after pipeline/barrier init
- `W0/W1` wait on the TMEM pointer barrier and initialize running stats

#### Steady state over sparse tile `k`

1. `W3` stages `page_id[k+1]`, `slot[k+1]`, `valid_mask[k+1]`
2. `W4/W5` gather `Kc/Kpe[k+1]`
3. `W2` runs `QK[k+1] -> S[k+1]` in TMEM
4. `W0` consumes `S[k]`, updates online softmax stats, writes `P[k]`, publishes correction metadata
5. `W1` consumes correction metadata for `k`, rescales accumulated `O` from tile `k-1`
6. `W6/W7` gather `V[k, d:d+128]` slice-by-slice
7. `W2` consumes `P[k]` and `V[k]` slices to do `PV[k] -> O`

This is intentionally the same *style* as the MLA Blackwell example, but reduced to:

- 1 softmax warp instead of 4
- 1 correction warp instead of 4
- no cross-warp row exchange barriers, because one warp owns all 16 real heads

#### Epilogue

- `W1` waits for the last `O` stage
- if the token has any valid sparse entries, `W1` computes:
  - `O_norm = O / row_sum`
  - `lse = log2(row_sum) + row_max * sm_scale_log2`
- store only rows `0..15`
- if the token has no valid sparse entries:
  - store zero output
  - store `-inf` LSE

### Load-warp work partition details

#### `W4/W5`: `Kc/Kpe` gather

- each warp owns 16 rows of the current 32-entry sparse tile
- for each row:
  - `Kc[512]` = `1024 B` = two `16 B` passes per lane-group
  - `Kpe[64]` = `128 B` = one short vectorized pass
- invalid rows are zero-filled in SMEM

#### `W6/W7`: `V` gather

- each warp owns 16 rows for the current sparse tile
- values are gathered one `128`-dim output slice at a time:
  - `V_slice[128]` = `256 B`
- this keeps PV staging compact and allows double-buffering over output slices

---

## Async pipelining

### Main round-0 pipeline choice

The MLA example leans heavily on paged TMA. Round-0 DSA does **not** use TMA in the sparse mainloop because `sparse_indices` are token-granular and non-affine. Instead:

- the sparse address problem is split into:
  1. **page decode**
  2. **manual vectorized row gather**
- only the UMMA / TMEM boundaries use the SM100-specific pipelines directly

This is the cleanest CuTeDSL implementation path that still follows the Blackwell warp-specialized model.

### Pipeline inventory

| Pipeline | Type | Producer warps | Consumer warps | Stages | Producer threads | Consumer threads | `tx_count` | Barrier storage | Notes |
|---|---|---|---|---:|---:|---:|---|---|---|
| `page_pipe` | `PipelineAsync` | `W3` | `W4-W7` | `2` | `32` | `128` | n/a | `page_mbar_ptr` | decoded `page_id/slot/valid_mask` ring |
| `qk_load_pipe` | `PipelineAsyncUmma` | `W4-W5` | `W2` | `2` | `64` | `32` | n/a | `qk_load_mbar_ptr` | staged `Kc/Kpe` ring for QK UMMA |
| `s_pipe` | `PipelineUmmaAsync` | `W2` | `W0` | `2` | `32` | `32` | n/a | `s_mbar_ptr` | TMEM logits `S` ring |
| `p_pipe` | `PipelineAsyncUmma` | `W0` | `W2` | `2` | `32` | `32` | n/a | `p_mbar_ptr` | staged `P` ring for PV UMMA |
| `corr_pipe` | `PipelineAsync` | `W0` | `W1` | `2` | `32` | `32` | n/a | `corr_mbar_ptr` | correction metadata ring (`alpha`, `row_sum`, `row_max`, flags) |
| `v_pipe` | `PipelineAsyncUmma` | `W6-W7` | `W2` | `2` | `64` | `32` | n/a | `v_mbar_ptr` | double-buffered `V[:, d:d+128]` slices |
| `o_pipe` | `PipelineUmmaAsync` | `W2` | `W1` | `1` | `32` | `32` | n/a | `o_mbar_ptr` | current TMEM output accumulator ownership |

### Producer / consumer group choices

- **`page_pipe`**: full warp producer (`W3`), four-warp async consumer group (`W4-W7`)
- **`qk_load_pipe`**: two-warp async producer group, one-warp UMMA consumer
- **`s_pipe`**: one-warp UMMA producer, one-warp async consumer
- **`p_pipe`**: one-warp async producer, one-warp UMMA consumer
- **`corr_pipe`**: one-warp async producer, one-warp async consumer
- **`v_pipe`**: two-warp async producer, one-warp UMMA consumer
- **`o_pipe`**: one-warp UMMA producer, one-warp async consumer

Round-0 uses **full-warp or full-multi-warp cooperative groups only**. There is no leader-only pipeline participant because the sparse mainloop does not use TMA.

Because cluster shape is `(1,1,1)`, all cooperative groups are CTA-local and there is no multicast / peer-CTA mask logic.

### Named barriers and fences

#### Named barriers

Use only one named barrier in round-0:

- `tmem_ptr_sync_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=96)`
  - participating warps: `W0`, `W1`, `W2`
  - purpose: `W2` allocates TMEM; `W0/W1` cannot retrieve the pointer until allocation completes

Round-0 intentionally does **not** need MLA-style softmax-exchange or epilogue-exchange named barriers because one warp owns all 16 valid rows.

#### Fences

- after any **manual SMEM producer write** (`W3`, `W4/W5`, `W6/W7`, `W0`):
  - `cute.arch.fence_view_async_shared()`
  - then `producer_commit(...)`

- after any **TMEM consumer load** (`W0`, `W1`):
  - `cute.arch.fence_view_async_tmem_load()`
  - then `consumer_release(...)`

- after barrier init:
  - `cute.arch.mbarrier_init_fence()`
  - `pipeline.agent_sync(pipeline.Agent.ThreadBlock)`

### Prefetch strategy

1. **Q prologue**
   - load once at CTA start, keep resident for the full token

2. **Sparse page decode**
   - predecode tile `k+1` while tile `k` is in QK / softmax / PV flight

3. **QK gather**
   - use the 2-stage `qk_load_pipe` so `Kc/Kpe[k+1]` can be prepared while `W2` is busy elsewhere

4. **V slices**
   - use the 2-stage `v_pipe` inside the `4` output-slice loop so `d+128` can be prepared while `d` is consumed

5. **No TMA descriptor prefetch in round-0**
   - there are no sparse-mainloop TMA atoms to prefetch
   - optional future Q-prologue TMA can add `cpasync.prefetch_descriptor(...)`

### Why not a TMA mainloop in round-0?

The key reason is semantic, not architectural:

- TMA is ideal for affine tiles or page-granular access
- `sparse_indices` are token-granular flattened indices with arbitrary `slot` values
- the kernel must preserve that exact order

So round-0 uses **page decode + vectorized row gather**. If later profiling shows strong page locality or sorted sparse lists, the page-decode and gather stages are the obvious place to evolve toward paged non-exec TMA.

---

## Infrastructure

### MMA atoms

#### QK MMA

```python
qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
    cutlass.BFloat16,
    tcgen05.OperandMajorMode.K,
    tcgen05.OperandMajorMode.K,
    cutlass.Float32,
    tcgen05.CtaGroup.ONE,
    (64, 32),
)
```

- A operand: padded `Q_nope` / `Q_pe`
- B operand: gathered `Kc` / `Kpe`
- C / accumulator: `float32` logits in TMEM

#### PV MMA

```python
pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
    cutlass.BFloat16,
    tcgen05.OperandMajorMode.K,
    tcgen05.OperandMajorMode.MN,
    cutlass.Float32,
    tcgen05.CtaGroup.ONE,
    (64, 128),
)
```

- A operand: `P` in SMEM
- B operand: gathered `V[:, d:d+128]` slice in SMEM
- C / accumulator: `float32` output accumulator in TMEM

### SMEM layouts

Use `sm100_utils.make_smem_layout_a/b(...)` plus `cute.logical_divide(...)` exactly like the MLA example, but with CTA-local buffers and no TMA-facing layouts.

#### Q buffers

- `smem_q_ckv`
  - logical tensor: `[64, 512]`
  - layout source: `make_smem_layout_a(qk_tiled_mma, (64, 32, 16), bf16, 32)`
  - then logically divide by the `32` latent K-subtiles

- `smem_q_kpe`
  - logical tensor: `[64, 64]`
  - layout source: `make_smem_layout_a(qk_tiled_mma, (64, 32, 16), bf16, 4)`
  - then logically divide by the `4` rope K-subtiles

#### QK gather ring

- `smem_kc_qk`
  - logical tensor per stage: `[32, 512]`
  - 2 stages
  - layout source: `make_smem_layout_b(qk_tiled_mma, (64, 32, 16), bf16, 32 * 2)`
  - logically divide into `(k_subtile=32, stage=2)`

- `smem_kpe_qk`
  - logical tensor per stage: `[32, 64]`
  - 2 stages
  - layout source: `make_smem_layout_b(qk_tiled_mma, (64, 32, 16), bf16, 4 * 2)`
  - logically divide into `(k_subtile=4, stage=2)`

#### P ring

- logical tensor per stage: `[64, 32]`
- 2 stages
- layout source: `make_smem_layout_a(pv_tiled_mma, (64, 128, 16), bf16, 2 * 2)`
- logically divide into `(pv_k_subtile=2, stage=2)`

#### V-slice ring

- logical tensor per stage: `[32, 128]`
- 2 stages
- layout source: `make_smem_layout_b(pv_tiled_mma, (64, 128, 16), bf16, 2 * 2)`
- logically divide into `(pv_k_subtile=2, stage=2)`

#### Page / offset / mask ring

- `smem_page_id[2][32]` : `int32`
- `smem_page_off[2][32]` : `int32`
- `smem_valid_mask[2]` : `uint32` or `int32`

#### Correction metadata ring

Compressed to the real head count:

- `smem_corr_alpha[2][16]` : `float32`
- `smem_corr_row_sum[2][16]` : `float32`
- `smem_corr_row_max[2][16]` : `float32`
- `smem_corr_flags[2]` : packed `int32`
  - bit 0: tile has any valid sparse entry
  - bit 1: final tile
  - bit 2: token has any valid sparse entry so far

### TMEM layouts

#### Logits `S`

- logical tensor: `[64, 32, 2]`
- dtype: `float32`
- purpose: handoff from QK UMMA to softmax warp
- offset: `tmem_s_offset = 0`

#### Output accumulator `O`

- logical tensor: `[64, 512]` as `4` slices of `[64, 128]`
- dtype: `float32`
- purpose: persistent online-accumulated output across all sparse tiles
- offset: `tmem_o_offset = cols(S_stage2)`

#### TMEM allocation policy

- allocate through `utils.TmemAllocator(..., is_two_cta=False)`
- compute columns via `utils.get_num_tmem_alloc_cols([tS_stage2, tO], rounding=True, arch="sm_100")`
- expected rounded allocation: `512` columns
- **do not store correction metadata in TMEM** in round-0; keep it in SMEM instead

### TMA atoms / descriptors

#### Round-0 decision

- **none in the sparse mainloop**

Reason:

- `sparse_indices` are not affine
- preserving sparse order is more important than forcing a TMA design too early

#### Optional future-only TMA candidates

- Q prologue (`q_nope[t]`, `q_pe[t]`) because those slices are affine and contiguous
- page-grouped K/V fetch if a later pre-pass sorts / groups sparse indices by page

### SharedStorage sketch

```python
@cute.struct
class DsaKernelSharedStorage:
    # Pipeline barriers
    page_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
    qk_load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
    s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
    p_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
    corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
    v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
    o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]

    # TMEM allocator handoff
    tmem_holding_buf: cutlass.Int32

    # Sparse metadata ring
    smem_page_id: cute.struct.MemRange[cutlass.Int32, 2 * 32]
    smem_page_off: cute.struct.MemRange[cutlass.Int32, 2 * 32]
    smem_valid_mask: cute.struct.MemRange[cutlass.Int32, 2]

    # Correction metadata ring
    smem_corr_alpha: cute.struct.MemRange[cutlass.Float32, 2 * 16]
    smem_corr_row_sum: cute.struct.MemRange[cutlass.Float32, 2 * 16]
    smem_corr_row_max: cute.struct.MemRange[cutlass.Float32, 2 * 16]
    smem_corr_flags: cute.struct.MemRange[cutlass.Int32, 2]

    # Main operand storage
    smem_q_ckv: cute.struct.Align[... , 1024]
    smem_q_kpe: cute.struct.Align[... , 1024]
    smem_kc_qk: cute.struct.Align[... , 1024]
    smem_kpe_qk: cute.struct.Align[... , 1024]
    smem_p: cute.struct.Align[... , 1024]
    smem_v: cute.struct.Align[... , 1024]
```

Alignment note:

- keep UMMA-facing SMEM tensors on `1024B` alignment, mirroring the MLA example style

---

## Implementation Stages

**Validation convention:** internal SMEM / TMEM intermediates should be mirrored to dedicated GMEM debug tensors when a stage is brought up. Use a tiny test fixture with:

- at least one normal token,
- one token with mixed valid and `-1` sparse indices,
- one token with all `-1` sparse indices.

### S0 — Problem Specification / build the host reference fixture

- **Depends on:** none
- **Connection:** establishes the golden outputs and edge-case fixtures that every later stage consumes

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `ref_output` | `bf16` | `[T, 16, 512]` | `host` | `allclose` |
| `ref_lse` | `float32` | `[T, 16]` | `host` | `allclose` |
| `fixture_sparse_indices` | `int32` | `[T, 2048]` | `host` | `exact` |

- **Key helpers / APIs:** host baseline `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py`, test harness only

### S1 — Launch Configurations / instantiate the persistent token scheduler

- **Depends on:** `S0`
- **Connection:** assigns token IDs to CTAs; every later stage assumes one CTA owns one token

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `grid_shape` | `int32` tuple | `[3]` | `host` | `exact` |
| `debug_token_order` | `int32` | `[num_launched_ctas, num_scheduler_iters]` | `GMEM` | `exact` |
| `debug_token_done_count` | `int32` | `[T]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `PersistentTileSchedulerParams`, `StaticPersistentTileScheduler.create`, `StaticPersistentTileScheduler.get_grid_shape`, `cute.arch.block_idx`, `cute.arch.grid_dim`

### S2 — Infrastructure / define the SM100a QK and PV MMA atoms

- **Depends on:** `S1`
- **Connection:** `S7` consumes the QK MMA atom; `S10/S11/S12` consume the PV MMA atom

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `qk_mma_shape` | `int32` tuple | `[3]` | `host` | `exact` |
| `pv_mma_shape` | `int32` tuple | `[3]` | `host` | `exact` |
| `qk_ckv_iters` | `int32` | `[1]` | `host` | `exact` |
| `qk_kpe_iters` | `int32` | `[1]` | `host` | `exact` |
| `pv_k_iters` | `int32` | `[1]` | `host` | `exact` |
| `pv_out_iters` | `int32` | `[1]` | `host` | `exact` |

- **Key helpers / APIs:** `sm100_utils.make_trivial_tiled_mma`, `tcgen05.OperandMajorMode`, `tcgen05.CtaGroup`, `tcgen05.Field.ACCUMULATE`

### S3 — Infrastructure / define SMEM layouts, TMEM layouts, and `SharedStorage`

- **Depends on:** `S2`
- **Connection:** `S4-S12` all consume the resulting storage layout and barrier definitions

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `shared_storage_bytes` | `int32` | `[1]` | `host` | `exact` |
| `tmem_alloc_cols` | `int32` | `[1]` | `host` | `exact` |
| `q_smem_bytes` | `int32` | `[1]` | `host` | `exact` |
| `qk_ring_bytes` | `int32` | `[1]` | `host` | `exact` |
| `p_ring_bytes` | `int32` | `[1]` | `host` | `exact` |
| `v_ring_bytes` | `int32` | `[1]` | `host` | `exact` |

- **Key helpers / APIs:** `sm100_utils.make_smem_layout_a`, `sm100_utils.make_smem_layout_b`, `cute.logical_divide`, `cute.cosize`, `utils.get_num_tmem_alloc_cols`, `cute.struct.Align`, `utils.SmemAllocator`, `utils.TmemAllocator`

### S4 — Warp Schedule / implement the Q prologue load and 16→64 row padding

- **Depends on:** `S3`
- **Connection:** `S7` consumes these padded Q buffers for QK UMMA

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `debug_q_ckv_padded` | `bf16` | `[64, 512]` | `GMEM` | `exact` |
| `debug_q_kpe_padded` | `bf16` | `[64, 64]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `cute.autovec_copy`, `cute.make_identity_tensor`, `pipeline.agent_sync`, `cute.arch.setmaxregister_decrease`, CTA-wide sync after prologue

### S5 — Async pipelining / load and decode sparse indices into `(page_id, slot, valid_mask)`

- **Depends on:** `S1`, `S3`
- **Connection:** `S6` and `S9` consume the decoded sparse metadata ring

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `debug_page_id_stage0` | `int32` | `[32]` | `GMEM` | `exact` |
| `debug_page_off_stage0` | `int32` | `[32]` | `GMEM` | `exact` |
| `debug_valid_mask_stage0` | `int32` | `[1]` | `GMEM` | `exact` |
| `debug_page_id_stage1` | `int32` | `[32]` | `GMEM` | `exact` |
| `debug_page_off_stage1` | `int32` | `[32]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `pipeline.PipelineAsync.create`, `pipeline.make_pipeline_state`, `cute.make_copy_atom`, `cpasync.CopyG2SOp` (optional), integer bit ops for `>> 6` and `& 63`

### S6 — Warp Schedule / gather `Kc` and `Kpe` rows into the staged QK SMEM ring

- **Depends on:** `S5`, `S3`
- **Connection:** `S7` consumes the staged `Kc/Kpe` buffers

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `debug_kc_tile_stage0` | `bf16` | `[32, 512]` | `GMEM` | `exact` |
| `debug_kpe_tile_stage0` | `bf16` | `[32, 64]` | `GMEM` | `exact` |
| `debug_kc_tile_stage1` | `bf16` | `[32, 512]` | `GMEM` | `exact` |
| `debug_invalid_rows_zero` | `bf16` | `[32, 512]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `cute.nvgpu.CopyUniversalOp`, `cute.make_copy_atom`, `cute.make_tiled_copy_D`, `cute.copy`, `cute.arch.fence_view_async_shared`, `pipeline.PipelineAsyncUmma.create`

### S7 — Warp Schedule / run QK UMMA and materialize one logits tile in TMEM

- **Depends on:** `S4`, `S6`, `S2`, `S3`
- **Connection:** `S8` consumes the TMEM `S` stage produced here

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `debug_logits_tile0` | `float32` | `[16, 32]` | `GMEM` | `allclose` |
| `debug_logits_tile1` | `float32` | `[16, 32]` | `GMEM` | `allclose` |

- **Key helpers / APIs:** `cute.gemm`, `tcgen05.Field.ACCUMULATE`, `utils.TmemAllocator`, `tcgen05.make_tmem_copy`, `pipeline.PipelineUmmaAsync.create`

### S8 — Async pipelining / consume `S`, apply sparse masking, update online softmax, and write `P`

- **Depends on:** `S7`
- **Connection:** `S10` consumes `P`; `S11` consumes correction metadata

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `debug_p_tile0` | `bf16` | `[16, 32]` | `GMEM` | `allclose` |
| `debug_row_max0` | `float32` | `[16]` | `GMEM` | `allclose` |
| `debug_row_sum0` | `float32` | `[16]` | `GMEM` | `allclose` |
| `debug_alpha0` | `float32` | `[16]` | `GMEM` | `allclose` |
| `debug_any_valid0` | `int32` | `[1]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `tcgen05.copy.Ld32x32bOp`, `tcgen05.make_tmem_copy`, `cute.math.exp2`, `cute.math.log2`, `cute.arch.add_packed_f32x2`, `cute.arch.fence_view_async_shared`, `pipeline.PipelineAsync.create`, `pipeline.PipelineAsyncUmma.create`

### S9 — Warp Schedule / gather one `V[:, d:d+128]` slice into the PV SMEM ring

- **Depends on:** `S5`, `S3`
- **Connection:** `S10` consumes the staged V slices for PV UMMA

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `debug_v_slice0_stage0` | `bf16` | `[32, 128]` | `GMEM` | `exact` |
| `debug_v_slice0_stage1` | `bf16` | `[32, 128]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `sm100_utils.make_smem_layout_b`, `cute.nvgpu.CopyUniversalOp`, `cute.make_copy_atom`, `cute.make_tiled_copy_D`, `cute.arch.fence_view_async_shared`, `pipeline.PipelineAsyncUmma.create`

### S10 — Warp Schedule / run PV UMMA for one sparse tile and accumulate into TMEM `O`

- **Depends on:** `S8`, `S9`, `S2`, `S3`
- **Connection:** `S11` consumes the resulting `O` stage for rescaling / final ownership

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `debug_o_after_tile0` | `float32` | `[16, 512]` | `GMEM` | `allclose` |
| `debug_o_after_tile0_slice0` | `float32` | `[16, 128]` | `GMEM` | `allclose` |

- **Key helpers / APIs:** `cute.gemm`, `tcgen05.Field.ACCUMULATE`, `pipeline.PipelineAsyncUmma`, `pipeline.PipelineUmmaAsync`

### S11 — Async pipelining / rescale TMEM `O` using the correction factor from the next softmax tile

- **Depends on:** `S8`, `S10`
- **Connection:** `S10` for tile `k+1` consumes the rescaled `O`; `S12` consumes the final `O`

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `debug_o_rescaled` | `float32` | `[16, 512]` | `GMEM` | `allclose` |
| `debug_skip_corr_flag` | `int32` | `[1]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `tcgen05.copy.Ld32x32bOp`, `tcgen05.copy.St32x32bOp`, `tcgen05.make_tmem_copy`, `cute.arch.fence_view_async_tmem_load`, `cute.arch.fence_view_async_tmem_store`, `pipeline.PipelineUmmaAsync`

### S12 — Warp Schedule / finalize one token: normalize `O`, store `output`, store base-2 `lse`

- **Depends on:** `S8`, `S10`, `S11`
- **Connection:** this is the first stage that produces final user-visible outputs

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `output_token0` | `bf16` | `[16, 512]` | `GMEM` | `allclose` |
| `lse_token0` | `float32` | `[16]` | `GMEM` | `allclose` |
| `output_all_invalid` | `bf16` | `[16, 512]` | `GMEM` | `exact` |
| `lse_all_invalid` | `float32` | `[16]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `tcgen05.copy.Ld32x32bOp`, `tcgen05.make_tmem_copy`, `cute.arch.rcp_approx`, `cute.math.log2`, `cute.autovec_copy`

### S13 — Launch Configurations / stitch the full persistent token loop and validate end-to-end

- **Depends on:** `S1-S12`
- **Connection:** integrates the scheduler, all warp roles, and all pipelines into the final round-0 kernel

| Field | Dtype | Shape | Capture | Compare |
|---|---|---:|---|---|
| `output` | `bf16` | `[T, 16, 512]` | `GMEM` | `allclose` |
| `lse` | `float32` | `[T, 16]` | `GMEM` | `allclose` |
| `debug_token_done_count` | `int32` | `[T]` | `GMEM` | `exact` |

- **Key helpers / APIs:** `StaticPersistentTileScheduler`, `pipeline.make_pipeline_state`, `cute.arch.setmaxregister_increase`, `cute.arch.setmaxregister_decrease`, final kernel `.launch(...)`

---

## Summary of round-0 design decisions

1. **Use UMMA now, even though heads=16, by padding M to 64**
2. **Keep sparse gathers explicit and vectorized instead of forcing a premature TMA design**
3. **Use TMEM for `S` and `O`, but keep correction metadata in SMEM**
4. **Use one softmax warp and one correction warp because only 16 real rows exist**
5. **Use one CTA/token with a persistent scheduler over tokens**

That combination is the best low-risk starting point for `kernel_0`: it is semantically faithful, CuTeDSL-native, and already aligned with the SM100a pipeline model shown in the Blackwell MLA examples.