# Kernel v2 Design

## Hypothesis

Replacing the entire PyTorch eager execution path with a single fused CuTeDSL warp-specialized attention kernel will reduce per-call latency from ~1.35ms to ~0.01-0.02ms (67-135x improvement), because v1 makes 5+ HBM round-trips for intermediate tensors that a fused kernel eliminates.

## Evidence from previous results

From `notes/kernel_v1_results.md`:
- v1 latency is ~1.35ms constant across all num_tokens (1-8)
- v1 moves ~22MB of data per token through HBM across 15+ PyTorch kernel launches
- A fused kernel reads Q once (16KB), streams KV through SMEM (~2.25MB via TMA gather4), and writes only the output (16KB + 128B LSE)
- From measured B200 hardware: per-SM TMA = ~3,752ns, GEMM = ~1,809ns, total ~5,561ns
- 32 chunks on 148 SMs = 1 wave at ~5.6us per token; plus combine = ~8-16us total

## Changes from v1

Replace the entire `_compute_split_partials()` and `_combine_splits()` PyTorch path with:
1. A fused CuTeDSL attention kernel that produces partial outputs and LSE per split
2. A PyTorch combine kernel (temporary; to be replaced in v3)

The CuTeDSL kernel handles ALL of: sparse KV gather via TMA gather4, QK GEMM (ckv + kpe), masked softmax, SV GEMM, and partial output/LSE writeback.

## Detailed design

### Grid and launch

- Grid: `(num_tokens * 32, 1, 1)` = one CTA per (token, split) pair
- Each CTA processes one 64-token KV chunk (B_TOPK=64)
- For num_tokens=8: 256 CTAs across 148 SMs = ~2 waves
- For num_tokens=1: 32 CTAs = 1 wave (plenty of SMs idle)

### Tile sizes

- **M** = 64 (padded from 16 real query heads; this wastes 75% MMA compute but is the minimum UTCMMA M dimension — acceptable because the kernel is TMA-bound, not compute-bound)
- **B_TOPK** (N for QK) = 64 (one KV chunk per CTA, matching split-KV design)
- **K_TILE** = 16 (BF16 MMA K dimension per instruction)
- **N_OUT** (N for SV) = 256 (output accumulator tiles; 2 SV GEMM tiles of M=64 x N=256)

### Warp specialization: 8 warps (256 threads)

| Warp ID | Role | Description |
|---------|------|-------------|
| 0 | TMEM alloc + MMA | Allocates TMEM; issues all tcgen05.mma ops (QK ckv, QK kpe, SV) |
| 1 | TMA ckv | Loads 16 gather4 calls for ckv_cache (4 rows x 16 calls = 64 rows) |
| 2 | TMA kpe | Loads 16 gather4 calls for kpe_cache |
| 3 | Index prep | Loads sparse_indices, computes TMA coordinates |
| 4-7 | Softmax + epilogue | Extracts QK scores from TMEM, computes online softmax (row_max, row_sum, exp2, scale), prepares P for SV GEMM, writes final partial output + LSE to global |

**Why 8 warps**: Matches v1 design doc. Fewer warps than FA4's 16 because our problem is simpler:
- No online softmax across multiple KV tiles (each CTA sees only 64 tokens)
- No correction/rescale warps needed (single-chunk split-KV)
- Simpler epilogue (just write partial O + LSE, no global softmax finalization)

### TMA configuration

**ckv_cache tensor map** (2D, gather4 mode):
- Global tensor: `[num_pages * 64, 512]` in BF16
- Box dims: `[4, 512]` = 4 rows x 512 BF16 elements = 4096 bytes per gather4
- Swizzle: SW128 (compatible with UTCMMA SMEM descriptor)
- Cache hint: `evict_last` (keep in L2 during computation)
- Per chunk: 16 gather4 calls = 64 rows loaded

**kpe_cache tensor map** (2D, gather4 mode):
- Global tensor: `[num_pages * 64, 64]` in BF16
- Box dims: `[4, 64]` = 4 rows x 64 BF16 elements = 512 bytes per gather4
- Swizzle: SW128
- Cache hint: `evict_first` (smaller working set, can tolerate L2 eviction)
- Per chunk: 16 gather4 calls = 64 rows loaded

**sparse_indices**: Loaded via standard global load (LDG) by index prep warp. 64 int32 values per CTA = 256 bytes. Use `nc_en_en_128B` cache hint for L1 residency.

### Shared memory plan

```
Layout:

  sQ_nope : 64 x 512 x 2B = 64 KB   (Q nope, loaded once via TMA, copied to TMEM via UTCCP)
  sQ_pe   : 64 x 64 x 2B  = 8 KB    (Q PE, loaded once via TMA, copied to TMEM via UTCCP)
  sKc     : 64 x 512 x 2B  = 64 KB   (ckv KV tile, K-major for UTCMMA; stays for SV GEMM too)
  sKp     : 64 x 64 x 2B   = 8 KB    (kpe KV tile, K-major)
  mbar[4] : TMA barriers (ckv, kpe, q_nope, q_pe)
  meta    : TMEM ptr buf, named barriers, etc. = ~2 KB

  Total: ~146 KB (well within 228 KB SM limit)
```

Note: No sP buffer needed — P stays in TMEM (TS mode for SV GEMM). sQ overlaps lifetime with sKp (Q is copied to TMEM before KV is loaded), so they could share space if SMEM is tight. In v2, we keep them separate for simplicity.

Note: No double-buffering of KV in v2 (only 1 tile of 64 rows per CTA). Double-buffering would only help if we had multiple KV tiles to stream, which split-KV with B_TOPK=64 does not.

**Pipeline**: 2-stage double-buffer for TMA->MMA. While MMA processes stage[i], TMA loads stage[i+1].

**Key insight**: For split-KV with B_TOPK=64, there is only ONE full tile of 64 KV tokens per CTA. There is no main loop over multiple KV tiles. The pipeline has 2 stages to overlap the ckv and kpe loads with initial Q loading, but the steady-state is just 1 iteration:
1. Stage 0: Load Q to TMEM, load ckv[0:31] + kpe[0:31] to SMEM stage 0
2. Stage 1: Load ckv[32:63] + kpe[32:63] to SMEM stage 1
3. QK GEMM uses both stages (all 64 rows of K)
4. Softmax on QK scores
5. SV GEMM produces partial output

Actually, re-examining: with K_TILE=16 for the K dimension of the KV-dim (512 for ckv, 64 for kpe), the QK GEMM iterates over K_DEPTH:
- QK ckv: M=64, N=64, K=512 => 32 K-tiles of 16
- QK kpe: M=64, N=64, K=64 => 4 K-tiles of 16
These K-tiles iterate over the head dimension, not the sequence dimension. All 64 KV rows must be present in SMEM before QK GEMM starts.

So the pipeline stages are for the N-dimension tiling, not K-dimension. With B_TOPK=64 and gather4 loading 4 rows per call, we need 16 gather4 calls. We can pipeline these:

**Revised pipeline strategy**: Issue all 16 TMA gather4 calls for ckv and kpe in a burst, then wait for all to complete before starting QK GEMM. Per the tma-gather4-ver9 results, 16 pipelined gather4 calls achieve ~3,752ns total, which is already optimal. Using 2 mbarriers (one per 8 gather4 calls) allows the first half to be consumed while the second half is still loading — but with only 64 rows total this overlap is minor.

**Simplified approach for v2**: Issue all TMA gather4 calls, wait for completion, then run GEMM serially. This is simpler and the TMA is already deeply pipelined (16 outstanding calls).

### TMEM column assignments

| Region | Columns | Size | Contents |
|--------|---------|------|----------|
| S (QK scores) | 0-63 | 64 cols | 64x64 FP32 accumulator for QK GEMM |
| O (output) | 64-319 | 256 cols | 64x512 FP32 accumulator for SV GEMM (2 tiles: 64x256 each) |
| Q_nope | 320-447 | 128 cols | 64x512 BF16 query (packed: 512 BF16 / 2 per col x 64 rows -> need 512/4 = 128 cols for K-major BF16) |
| Q_pe | 448-463 | 16 cols | 64x64 BF16 PE query (64/4 = 16 cols K-major) |
| **Total** | 0-463 | 464 cols | Leaves 48 cols free of 512 |

**Q loading**: Q is loaded from global memory to SMEM, then copied from SMEM to TMEM via UTCCP (tcgen05.cp). Q stays in TMEM for both QK ckv and QK kpe GEMMs. Only the first 16 rows of the 64-row Q tile contain real data (16 query heads); rows 17-63 are zero-padded.

### Barrier choreography

```
Named barriers:
  NB0: tmem_alloc_barrier (256 threads) — warp 0 allocates, all warps wait
  NB1: tma_complete_barrier (256 threads) — TMA warps signal, MMA warp waits
  NB2: qk_done_barrier (256 threads) — MMA signals QK done, softmax warps wait
  NB3: p_ready_barrier (256 threads) — softmax warps signal P ready in TMEM (BF16), MMA waits for SV
  NB4: sv_done_barrier (256 threads) — MMA signals SV done, epilogue warps wait

Transactional barriers (mbarriers):
  mbar_ckv[2]: TMA ckv gather4 completion (producer: warp 1, consumer: warp 0/MMA)
  mbar_kpe[2]: TMA kpe gather4 completion (producer: warp 2, consumer: warp 0/MMA)
  mbar_q:      TMA Q load completion (producer: warp 1/2, consumer: warp 0/MMA)
  mbar_mma[1]: MMA commit completion (producer: warp 0, consumer: warps 4-7)
```

### Execution flow per CTA

```
Phase 0: Setup (~200 cycles)
  - All warps: TMEM allocation, barrier init
  - Warp 3: Load sparse_indices[token_id, split_id*64 : (split_id+1)*64]
  - Warp 1: Start TMA for Q_nope (SMEM -> TMEM via UTCCP)
  - Warp 2: Start TMA for Q_pe

Phase 1: TMA KV load (~3,752 ns)
  - Warp 1: Issue 16 gather4 calls for ckv_cache (all 64 rows)
  - Warp 2: Issue 16 gather4 calls for kpe_cache (all 64 rows, concurrent with ckv)
  - Warp 0: Wait for Q UTCCP, then wait for KV TMA completion

Phase 2: QK GEMM (~1,046 ns ckv + ~181 ns kpe = ~1,227 ns)
  - Warp 0: QK_ckv = Q_nope @ Kc^T (M=64, N=64, K=512, 32 K-tiles)
    - Uses TMEM operand A (Q_nope in TMEM), SMEM operand B (Kc in SMEM)
    - Accumulates to S region in TMEM
  - Warp 0: QK_kpe = Q_pe @ Kp^T (M=64, N=64, K=64, 4 K-tiles)
    - ACCUMULATE mode — adds to existing S in TMEM
  - Warp 0: commit + signal mbar_mma

Phase 3: Softmax (~200-400 ns estimated)
  - Warps 4-7: Wait for mbar_mma (QK GEMM complete, S in TMEM)
  - Warps 4-7: Extract S from TMEM to registers via tcgen05.ld (each warp handles 32 lanes x 64 cols)
  - Warps 4-7: Apply sm_scale in FP32 registers
  - Warps 4-7: Mask invalid positions to -inf (using valid_mask from sparse_indices)
  - Warps 4-7: Compute row_max, subtract, exp2, row_sum (all FP32)
  - Warps 4-7: Compute LSE = row_max / sm_scale + log2(row_sum) [log2 base for competition]
  - Warps 4-7: Normalize: P = exp2(S_scaled - row_max) / row_sum
  - Warps 4-7: Convert P to BF16, pack pairs into 32-bit, write back to TMEM S region via tcgen05.st
  - Warps 4-7: fence + signal p_ready_barrier

Phase 4: SV GEMM (~346 ns, TS mode)
  - Warp 0: Wait for p_ready_barrier (P ready in TMEM as BF16 K-major)
  - Warp 0: SV = P @ Kc (M=64, N=512, K=64)
    - Operand A: P from TMEM (S region, BF16 K-major, OperandSource.TMEM)
    - Operand B: Kc from SMEM (already loaded, BF16 K-major)
    - Accumulates to O region in TMEM (columns 64-319)
    - 2 tiles: M=64, N=256, K=64 each (4 K-tiles of 16 per tile)
  - Warp 0: commit + signal sv_done

Phase 5: Epilogue (~200 ns)
  - Warps 4-7: Wait for sv_done
  - Warps 4-7: Extract O from TMEM (tcgen05.ld, 4 warps needed for 128 lanes)
  - Warps 4-7: Convert FP32 -> BF16, write partial_o to global memory
  - Warps 4-7: Write partial_lse to global memory

Total per-CTA: ~5,600-6,200 ns = ~5.6-6.2 us
```

### SV GEMM operand considerations

The SV GEMM computes `P @ Kc` where P is [64, 64] (attention weights) and Kc is [64, 512] BF16 from SMEM. This is GEMM(M=64, N=512, K=64). For UTCMMA, N max is 256, so we need 2 tiles (M=64, N=256, K=64 each). K=64 with BF16 K_TILE=16 means 4 K-tiles per SV tile.

**Decision: TS mode (P from TMEM, Kc from SMEM)**. This is the FA4 pattern:
- QK GEMM produces S (scores) in the S region of TMEM (columns 0-63)
- Softmax warps extract S from TMEM via `tcgen05.ld`, compute softmax in FP32 registers
- Softmax warps convert P to BF16 and write P back to the SAME S region in TMEM via `tcgen05.st`
- MMA warp runs SV GEMM with `OperandSource.TMEM` for A (P in TMEM, K-major) and SMEM descriptor for B (Kc)
- Result accumulates to O region in TMEM

This avoids writing P to SMEM entirely and keeps P on the fast TMEM path. TS mode: 56.4 cy/K-tile at K=32 scale. For SV: 2 tiles x 4 K-tiles = 8 MMA ops x ~56 cy + overhead = ~640 cy = ~346 ns.

**P datatype**: P is FP32 from softmax, but UTCMMA kind::f16 wants BF16 operand A. Softmax warps must convert P from FP32 to BF16 before writing back to TMEM. The S region (64 FP32 cols) maps to 32 BF16 cols (packed 2:1), but TMEM stores are 32-bit cells. The softmax warps will:
1. `tcgen05.ld` S from TMEM as FP32 (row of 64 scores per warp iteration)
2. Apply scale, row_max, exp2, normalization in FP32 registers
3. Convert to BF16 and pack pairs into 32-bit
4. `tcgen05.st` packed BF16 P back to TMEM S region (now interpreted as BF16 K-major by SV GEMM)

**Key constraint**: The S region has 64 FP32 columns. After softmax converts to BF16 and packs, P occupies only 32 columns (64 BF16 elements / 2 per 32-bit cell). The MMA instruction descriptor for SV GEMM specifies BF16 input type and K-major, so it reads the packed BF16 data correctly from these 32 TMEM columns. The remaining 32 columns of the 64 allocated can be ignored.

**FA4 reference**: `flash_fwd_sm100.py` line 383: `p_source = tcgen05.OperandSource.TMEM`, `p_major_mode = tcgen05.OperandMajorMode.K`. This confirms P-from-TMEM TS mode is the production pattern.

### Split-KV combine

After the fused kernel writes `partial_o[32, num_tokens, 64, 512]` (only first 16 of 64 rows valid) and `partial_lse[32, num_tokens, 64]` (only first 16 valid), a combine step merges them.

**v2 combine**: Keep the PyTorch combine from v1 (`_combine_splits`). This is a simple exp2-weighted reduction that adds negligible latency compared to the main kernel improvement. Optimize to CuTeDSL in v3.

### Output memory layout

```
partial_o:   [32, num_tokens, 16, 512] BF16  (or [32, num_tokens, 64, 512] with zero padding)
partial_lse: [32, num_tokens, 16] FP32 (log2 base)
```

The fused kernel writes these; the combine kernel reads them and produces:
```
output: [num_tokens, 16, 512] BF16
lse:    [num_tokens, 16] FP32
```

### Q loading strategy

Q is small: q_nope [num_tokens, 16, 512] BF16 = 16 KB per token, q_pe [num_tokens, 16, 64] BF16 = 2 KB per token.

Each CTA needs the Q for its token. Multiple CTAs (32 per token) read the same Q.

**Option 1**: TMA load Q to SMEM, then UTCCP to TMEM. Standard FA4 approach.
**Option 2**: Direct LDG to registers, then reg->TMEM store. Simpler but needs TMEM store overhead.

**Decision**: Use TMA for Q load to SMEM, then UTCCP (tcgen05.cp) from SMEM to TMEM. This is the standard pattern and avoids register-file pressure for the 16KB Q_nope. Q is shared across all 32 CTAs per token, so it should be L2-hot after the first CTA loads it.

## Expected outcome

- Per-CTA execution: ~5.6-6.2 us (TMA-bound)
- For num_tokens=1: 32 CTAs, 1 wave => ~6 us + combine ~3 us = ~9 us = 0.009ms
- For num_tokens=8: 256 CTAs, 2 waves => ~12 us + combine ~5 us = ~17 us = 0.017ms
- vs v1: 1.35ms => **75-150x improvement** expected
- Average speedup over reference: >>10x (ref is ~1.1-3.0ms)

This is a best-case estimate. Actual first-version overhead from barrier setup, TMEM allocation, index computation, and softmax arithmetic will likely be 1.5-2x higher, still yielding 40-75x improvement over v1.

## Implementation notes for kernel-coder

### Phase 1 (minimum viable fused kernel)

Build a CuTeDSL kernel that:
1. Takes all inputs (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) plus output/lse buffers
2. Each CTA computes one (token, split) pair
3. Loads Q via TMA -> SMEM -> UTCCP -> TMEM
4. Loads KV via TMA gather4 -> SMEM (16 calls each for ckv and kpe)
5. Runs QK GEMM (ckv + kpe, accumulated to same S in TMEM)
6. Extracts S from TMEM, runs softmax in FP32 registers, converts P to BF16, writes P back to TMEM
7. Runs SV GEMM in TS mode (P from TMEM x Kc from SMEM = O in TMEM)
8. Extracts O from TMEM, converts to BF16, writes partial_o + partial_lse to global

### Architecture references

- **Warp specialization**: Follow FA4 pattern (`flash_fwd_sm100.py`) — single MMA warp, TMA warp, softmax/epilogue warps
- **TMEM layout**: Use `project_v1_design_decisions.md` column plan (S=64, O=256, Q_nope=128, Q_pe=16 = 464 total)
- **TMA gather4**: Use `competition_kernel_insights.md` configurations (box dims, swizzle, cache hints)
- **SMEM layout**: Use `sm100_utils.make_smem_layout_a/b()` for MMA-compatible layouts with proper swizzle
- **Softmax**: Implement in FP32 with `exp2()` (not `exp()`), produce log2-base LSE. See `references/flash-attention/flash_attn/cute/softmax.py`
- **Pipeline**: Start simple — no double-buffering for v2. Load all KV, then compute. Optimize pipeline in v3.

### Simplifications for v2

1. No double-buffering of KV tiles (all 64 rows loaded before GEMM starts)
2. TS mode for SV GEMM (P from TMEM, following FA4 pattern; avoids SMEM write for P)
3. PyTorch combine (not a fused CuTeDSL combine kernel)
4. Padding Q to M=64 (wasteful but correct; optimize packing in v3)
5. No persistent kernel / tile scheduler (one CTA per split; simple grid launch)

### Testing

The kernel MUST pass all 23 competition workloads with correctness within tolerance:
- max_abs_err < 1e-01
- max_rel_err < 1.0
(Tolerances are generous for BF16; v1 achieves max_abs_err <= 7.81e-03)
