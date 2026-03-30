# Kernel v4 Research — Optimization & Hardening

Status: **research only** — do not merge into `solution/dsa_attention/kernel.py`.

Baseline: the fused 3-warpgroup kernel as of 2026-03-29 (all checklist items complete,
fused-authoritative, register-based epilogue).

---

## 1. Profiling prerequisites

No optimization should land without NCU data confirming the bottleneck.

Key counters to collect:

| Counter | What it tells us |
|---------|-----------------|
| `sm__throughput.avg_pct_of_peak_sustained_elapsed` | Compute vs memory bound |
| `l1tex__t_sector_hit_rate.pct` | L1 hit rate for gather4 loads |
| `lts__t_sector_hit_rate.pct` | L2 hit rate (sparse indices locality) |
| `sm__warps_active.avg_pct_of_peak_sustained_elapsed` | Occupancy |
| `l1tex__data_pipe_lsu_wavefronts_mem_lg` | Register spill to local mem |
| `sm__inst_executed_pipe_tensor_op_hmma` | UTCMMA utilization |

Run with:
```bash
# correctness first
modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --correctness-only

# then NCU (if tooling supports it)
ncu --target-processes all --set full -o dsa_v3_baseline ...
```

---

## 2. Optimization candidates (ranked by expected impact)

### P1: Multi-stage V pipeline — est. 15-25% on PV phase

**Problem:** WG0->WG1 V staging is single-stage lockstep. WG0 gathers V tile N,
WG1 blocks until gather completes, computes PV, then WG0 starts tile N+1.
8 tiles total = 8 round-trip stalls.

**Design:** 2-stage `v_pipe` with double-buffered `sV`.

```
Iteration 0:  WG0 gather V[0] -> sV[0]
Iteration 1:  WG0 gather V[1] -> sV[1]  |  WG1 PV MMA on sV[0]
Iteration 2:  WG0 gather V[2] -> sV[0]  |  WG1 PV MMA on sV[1]
...
```

Changes needed:
- `pv_b_smem_layout` with `num_stages=2` (doubles sV SMEM)
- `v_pipe` with `num_stages=2`
- WG0 loop: prefetch first stage, then overlap gather/commit
- WG1 loop: ping-pong `sV[(None, None, None, stage_idx)]`

Reference: `mixed_input_fmha_decode.py:123-125` (`bs_stages=2`, `sp_stages=2`).

SMEM cost: +8KB (one extra 64x64 BF16 buffer).

**Risk:** Low. Pattern is well-established in CUTLASS FMHA.

---

### P2: TMA store epilogue — est. 20-30% on epilogue phase

**Problem:** Current epilogue is TMEM -> registers -> per-element global store.
Uncoalesced, high instruction count (4096 stores per 64x64 tile, 8 tiles = 32K stores).

**Design:** TMEM -> registers -> SMEM staging -> TMA S2G bulk store.

```
WG2 per v_tile:
  1. wait output_pipe
  2. TMEM -> registers (tiled_copy_t2r, existing)
  3. registers -> sO (SMEM, 64x64 FP32)
  4. fence_view_async_shared()
  5. TMA S2G: sO -> mPartialO[split, token, :, v_col_base:v_col_base+64]
  6. cp_async_bulk_commit_group()
  7. cp_async_bulk_wait_group(0, read=True)
  8. release output_pipe
```

Changes needed:
- `CopyBulkTensorTileS2GOp(cta_group)` for output TMA descriptor
- `cpasync.make_tiled_tma_atom(store_op, mPartialO, ...)` in `__call__`
- Pass `tma_atom_o` to kernel
- WG2: replace register->global loop with SMEM staging + TMA store
- `sO` SMEM buffer (64x64 FP32 = 16KB) — was previously allocated, we removed it; add back

Reference: `mixed_input_fmha_decode.py:338-340`, `fmha.py:1325-1345`.

**Risk:** Medium. TMA descriptor for 4D `mPartialO` needs careful coordinate mapping.

---

### P3: Persistent kernel — est. 8-15% overall

**Problem:** Grid = [T, 32, 1]. Each CTA allocates TMEM (~200-300 cycles), inits
pipelines, prefetches TMA descriptors. For 32 splits/token, that's ~10K wasted cycles.

**Design:** Grid = [T, num_SMs, 1] or smaller. Each CTA loops over splits:

```
while split_idx < NUM_SPLITS:
    load indices for split_idx
    phase 1: QK + softmax
    phase 2: PV + epilogue
    split_idx += grid_stride
```

Changes needed:
- Grid/block re-parameterization
- SMEM index/valid/mask reset between splits
- TMEM stays allocated across splits (no free/realloc)
- Persistent scheduler (CUTLASS has utilities for this)

Reference: `fmha.py:814,1007,1276` (persistent tile scheduler).

**Risk:** High complexity. SMEM/TMEM reuse across splits needs careful barrier reset.

---

### P4: Sparse index sorting — impact depends on NCU L2 data

**Problem:** gather4 loads CKV rows at random indices. If indices are scattered
across L2 sets, every gather is a miss. If sorted by page, consecutive gathers
hit the same L2 line.

**Design:** Sort `sparse_indices` on host (PyTorch) before launching the kernel.
Within each chunk of 64 indices, sort by `index // PAGE_SIZE` then by `index % PAGE_SIZE`.

Changes needed:
- Host-side sort in `_prepare_split_shell_operands` or `_compute_split_partials_fused`
- No kernel changes

**Risk:** Near zero. Pure host preprocessing. Could hurt if sorting overhead > L2 gain.

**Measure first:** NCU `lts__t_sector_hit_rate.pct`. If >80%, skip this.

---

### P5: Fuse `_combine_splits` into a CuTeDSL kernel

**Problem:** `_combine_splits` runs in PyTorch: exp2, weighted sum, normalization
over 32 splits. For T=1 (decode), this is 32 * 16 * 512 = 262K FP32 ops — tiny
but launches a separate kernel with overhead.

**Design:** A simple 1-warpgroup CuTeDSL kernel that reads `partial_o[S,T,H,D]`
and `partial_lse[S,T,H]`, computes online log-sum-exp merge, writes final
`output[T,H,D]` and `lse[T,H]`.

**Risk:** Low. Pure compute, no complex synchronization.

---

### Deprioritized: Online softmax

The decode FMHA reference doesn't use it. Softmax is ~10-15% of total latency
and already pipelined through WG2. Register pressure risk outweighs the small gain.

---

## 3. Correctness hardening

### Verified safe (false positives from audit)

| Flagged issue | Why it's OK |
|--------------|-------------|
| sQrow overflow (CHUNK_SIZE*N_BLOCK loop) | M_BLOCK = CHUNK_SIZE = 64, buffer is (64,64,1), loop is 4096 elements = exact match |
| Softmax OOB writes | 128 threads / 8 lanes = 16 heads = NUM_Q_HEADS exactly, all guarded |
| valid_wg_sync_barrier deadlock | num_threads=256 = WG0(128) + WG2(128), WG1 never touches it — correct |
| Epilogue col OOB | Identity tensor coords are in [0, N_BLOCK), always in bounds |
| gather_phase not reset | Each CTA is a fresh kernel invocation, local var starts at 0 |
| All-invalid-split NaN | row_sum=0 -> lse=-inf -> combine_splits weights=0 -> correct |

### Worth validating (low risk but no test coverage)

**A. TMEM coherence through PipelineUmmaAsync**

The epilogue reads TMEM after `output_consumer.wait_and_advance()`. This relies
on the UMMA pipeline's commit guarantee for TMEM write visibility. If stale reads
occur (non-deterministic output diffs), add:

```python
# In WG1, after gemm loop, before commit:
cute.arch.fence_view_async_tmem_store()
```

**How to test:** Run with `FLASHMLA_DSA_VALIDATE_FUSED=assert` on 100 different
seeds. Any failure = fence needed.

**B. Register spill in epilogue**

The output epilogue creates `tTR_rOutput`, `cOutput`, multiple partition views.
If NCU shows `l1tex__data_pipe_lsu_wavefronts_mem_lg` > 0 for WG2, register
pressure is causing local memory spill.

**C. Dedicated all-invalid-split test**

Add to pygpubench test generator:
```python
sparse_indices[:] = -1  # all invalid
```
Verify output = 0, lse = -inf, no NaN.

---

## 4. SMEM budget analysis

Current allocation (approximate):

| Buffer | Size | Notes |
|--------|------|-------|
| sQ (A operand) | 8 KB | 64x64 BF16, swizzled, 1 stage |
| sQrow (staging) | 8 KB | 64x64 BF16, rowmajor |
| sK (B operand) | 8 KB | 64x64 BF16, swizzled, 1 stage |
| sV (PV B operand) | 8 KB | 64x64 BF16, swizzled, 1 stage |
| sP (PV A operand) | 8 KB | 64x64 BF16, swizzled, 1 stage |
| sScoreF32 | 16 KB | 64x64 FP32 |
| sIdx + sValid + sGroupMask + sGatherBytes + sHasValid | ~0.5 KB | index metadata |
| Pipeline barriers | ~0.1 KB | mbar arrays |
| **Total** | **~57 KB** | |

SM100 shared memory limit: 228 KB per SM (configurable).

Headroom for 2-stage V: +8 KB -> ~65 KB. Comfortable.
Headroom for sO epilogue buffer: +16 KB -> ~73 KB. Still fine.
Both together: ~81 KB. Well within limits.

---

## 5. Next steps

1. **Run correctness on Modal** — confirm fused epilogue produces matching output
2. **NCU profile** — identify actual bottleneck (load? MMA? softmax? epilogue?)
3. **Implement P1** (2-stage V pipeline) in a research copy
4. **Measure** delta vs baseline
5. **Then P2** (TMA store epilogue) if epilogue is on critical path
