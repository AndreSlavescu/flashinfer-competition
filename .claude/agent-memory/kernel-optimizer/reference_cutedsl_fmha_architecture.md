---
name: CuTeDSL Blackwell FMHA architecture reference
description: Detailed architecture of CUTLASS Blackwell FMHA in CuTeDSL — warp roles, pipelines, TMEM layout, shared memory, mainloop structure — the primary template for competition kernel
type: reference
---

## Source
`NVIDIA/cutlass/examples/python/CuTeDSL/blackwell/fmha.py` (130KB, warp-specialized persistent kernel)

## Warp Specialization (16 warps = 512 threads)

| Warp IDs | Role | Description |
|----------|------|-------------|
| 0-3 | softmax0 | First softmax stage (row-max, row-sum) |
| 4-7 | softmax1 | Second softmax stage |
| 8-11 | correction | Rescale intermediate outputs with updated softmax factors |
| 12 | MMA | All tcgen05.mma operations (QK and PV GEMMs) |
| 13 | TMA load | All TMA gather/copy operations |
| 14 | epilogue | TMA store to global memory |
| 15 | empty/sync | Utility warp |

## Pipeline Architecture

| Pipeline | Type | Stages | Producer→Consumer |
|----------|------|--------|-------------------|
| load_q | PipelineTmaUmma | q_stage=2 | TMA load → MMA |
| load_kv | PipelineTmaUmma | kv_stage=3-4 | TMA load → MMA |
| mma_s0 | PipelineUmmaAsync | mma_softmax_stage=1 | MMA → softmax0 |
| mma_s1 | PipelineUmmaAsync | mma_softmax_stage=1 | MMA → softmax1 |
| s0_corr | PipelineAsync | softmax_corr_stage | softmax0 → correction |
| s1_corr | PipelineAsync | softmax_corr_stage | softmax1 → correction |
| corr_epi | PipelineAsync | - | correction → epilogue |
| mma_corr | PipelineAsync | - | MMA → correction |
| tmem_dealloc | Pipeline | - | deallocation sync |

## TMEM Layout

Allocated dynamically by MMA warp (warp 12):
- s0_offset = 0 (score matrix 0)
- s1_offset = 128 (score matrix 1)
- o0_offset = 256 (output 0)
- o1_offset = 384 (output 1)

## Shared Memory Layout

```
SharedStorage:
  load_q_mbar_ptr: [q_stage * 2] Int64     # TMA barriers
  load_kv_mbar_ptr: [kv_stage * 2] Int64
  mma_s0_mbar_ptr: [mma_softmax_stage * 2] Int64
  mma_s1_mbar_ptr: [mma_softmax_stage * 2] Int64
  s0_corr_mbar_ptr, s1_corr_mbar_ptr       # softmax→correction barriers
  tmem_holding_buf: Int32                    # TMEM address from alloc
  sO: Align[..., 1024]                      # Output staging
  sQ: Align[..., 1024]                      # Q tensor (K-major layout)
  sK: Align[..., 1024]                      # K tensor (K-major, 0-stride head broadcast)
```

## MMA Configuration

```python
qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
    q_dtype, q_major_mode, k_major_mode,
    qk_acc_dtype, cta_group, qk_mma_tiler[:2])
pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
    v_dtype, p_major_mode, v_major_mode,
    pv_acc_dtype, cta_group, pv_mma_tiler[:2], p_source)
```

CTA tiler doubles Q dimension (2×M) to process two Q tiles per block.
`tcgen05.Field.ACCUMULATE` toggled: first K-phase overwrites, subsequent accumulate.

## Mainloop Data Flow

1. **Load**: Q0, K0, Q1, V0 prefetch → then Ki, Vi in loop
2. **MMA**: GEMM_QK00 (Q0×K0→S0), GEMM_QK10 (Q1×K0→S1), GEMM_PV (P×V→O)
3. **Softmax**: Row reduction + normalization (parallel with MMA to hide latency)
4. **Correction**: Exponential normalization with accumulated max values
5. **Epilogue**: TMA store to global memory

## Key CuTeDSL Patterns

- `cute.arch.alloc_tmem(cols, holding_buf)` / `cute.arch.retrieve_tmem_ptr()`
- `setmaxregister_decrease/increase` per warp role (register pressure management)
- Named barriers with explicit thread counts
- `FmhaStaticTileScheduler` for persistent work distribution
- `FusedMask` with causal/sliding window support

## Imports Required

```python
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
```

## Adaptation Notes for Competition Kernel

1. **Remove V tensor path** — in competition, V = Kc (ckv_cache values), same data as QK key
2. **Add kpe gather** — second TMA stream for kpe_cache (64d), separate from ckv (512d)
3. **Modify softmax** — output LSE in log2 base, not natural log
4. **Add split-KV** — persistent scheduler must handle 32 blocks per query token
5. **Add sparse indexing** — TMA gather uses sparse_indices for page-level gather
6. **Reduce threads** — may not need 512 threads; competition's simpler data path might work with fewer warps
