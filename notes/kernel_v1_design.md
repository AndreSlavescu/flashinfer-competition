# Kernel v1 Design — Bootstrap CuTeDSL Kernels for DSA Competition

## Goal

Produce a first working v1 of both competition kernels in CuTeDSL (Python) that compiles on B200, passes correctness validation, and establishes a baseline speedup for iterative optimization.

The v1 design prioritizes **correctness and simplicity** over peak performance. Every design choice is annotated with its optimization ceiling so that v2+ iterations can target the highest-impact bottleneck.

---

## Part 1: Implementation Order

**Implement the Sparse Attention kernel first.**

Rationale:
1. The attention kernel is the higher-value target: 23 workloads, larger per-workload latency, and the kernel that benefits most from B200 tensor cores.
2. It exercises the full CuTeDSL stack: TMA gather, UTCMMA, warp specialization, pipeline barriers, TMEM, softmax, epilogue. Getting it working first builds the infrastructure for the indexer.
3. The indexer feeds into the attention kernel — validating attention first means we can test the full pipeline end-to-end sooner.
4. The indexer can use a simpler PyTorch-based approach initially (torch.topk is correct, just slow) while we focus on the attention kernel.

---

## Part 2: Sparse Attention Kernel (v1)

### 2.1 Algorithm Summary

Per query token t:
```
for each KV chunk of B_TOPK=64 tokens (32 chunks for topk=2048):
    1. Gather Kc[64, 512] and Kp[64, 64] from paged cache using sparse_indices
    2. S_ckv = Q_nope[16, 512] @ Kc.T[512, 64]    → [16, 64] logits (QK ckv GEMM)
    3. S_kpe = Q_pe[16, 64] @ Kp.T[64, 64]         → [16, 64] logits (QK kpe GEMM)
    4. S = (S_ckv + S_kpe) * sm_scale
    5. Online softmax: update row_max, row_sum, rescale O
    6. O += softmax(S) @ Kc[64, 512]                → [16, 512] output (SV GEMM)
finalize: lse = (row_max * scale_log2 + log2(row_sum)) * ln(2)
output = O / row_sum
```

### 2.2 Kernel Architecture: Split-KV with Per-SM Chunks

**Block mapping**: One CTA processes one (token, chunk) pair.
- Grid: (num_tokens, num_sm_parts, 1) where num_sm_parts = ceil(2048 / B_TOPK) = 32
- Each CTA handles 64 KV tokens from sparse_indices
- After all CTAs finish, a lightweight combine kernel merges the 32 partial results per token

**Why split-KV (not online softmax within one CTA)**:
- With B_TOPK=64 and 32 chunks, online softmax within a single CTA would serialize all 32 chunks, making TMA+GEMM sequential per chunk.
- Split-KV launches 32 CTAs in parallel per token. For num_tokens=1, we use 32 of 148 SMs. For num_tokens=8, we use all 256 CTAs across 148 SMs.
- Each CTA does ONE chunk (no O-rescale needed within the chunk).

**V1 simplification**: Each CTA computes a partial O and partial LSE. A separate combine kernel (FA4's `FlashAttentionForwardCombine` or a simple custom one) merges them.

### 2.3 GEMM Tile Sizes

All GEMMs within one CTA:

| GEMM | M | N | K (BF16 tiles of 16) | K total | Mode | TMEM Acc cols |
|------|---|---|---------------------|---------|------|---------------|
| QK ckv | 16 | 64 | 32 | 512 | TS (Q in TMEM) | 64 cols |
| QK kpe | 16 | 64 | 4 | 64 | TS (Q in TMEM) | same 64 cols |
| SV | 16 | 256 | 4 | 64 | SS (P in smem) | 256 cols |

**M=16** (not 64): We have 16 query heads, not 64. This is the natural M dimension.

**QK N=64**: B_TOPK=64 tokens per chunk. Perfect match.

**SV N=256**: head_dim_ckv=512, split into 2 tiles of N=256. Two MMA calls.

**SV mode SS (not TS)**: P (attention weights) comes from softmax in registers, stored to smem, then used as operand A in SS mode. V (=Kc) is operand B in smem (MN-major for V, which is K-major for Kc transposed).

### 2.4 TMEM Column Budget

```
Columns 0-63:    QK accumulator (S) — M=16, N=64, FP32 → 64 cols
Columns 64-319:  Output accumulator (O) — M=16, N=256, FP32 → 256 cols (×2 tiles)
Columns 320-383: Q_nope in TMEM — M=16, K=512 BF16 → needs 16×512/2/128 = 32 cols
Columns 384-387: Q_pe in TMEM — M=16, K=64 BF16 → needs 16×64/2/128 = 4 cols
Total: ~388 cols of 512 available
```

Wait — let me reconsider. With M=16, the TMEM layout is different. UTCMMA M dimension maps to TMEM data-paths (128 lanes). For M=16, we use only 16 of 128 lanes, which wastes 7/8 of TMEM bandwidth but is the correct mapping.

Actually, re-examining the UTCMMA tile constraints: **M must be 64 or 128** for cta_group::1. M=16 is not a valid MMA tile shape.

**Correction**: We must use M=64. This means we need to pack multiple query heads into the M dimension or pad.

**Strategy for M=64 with 16 query heads**: Replicate Q across 4 copies to fill M=64 (each head appears 4× in a "virtual" M dimension). This wastes 3/4 of compute but gives a valid MMA tile. However, the output would then have redundant rows that need to be selected.

**Better strategy**: Use M=64 with the 16 heads and pad the remaining 48 rows with zeros. The MMA hardware doesn't care about padding — it computes all 64 rows regardless. We only read out the 16 valid rows.

**Even better**: Pack 4 tokens into M=64 (16 heads × 4 tokens = 64 rows). But this complicates the split-KV design since each CTA would process 4 tokens.

**V1 decision**: Use M=64, pad 16 heads to 64 rows. Waste 75% of MMA compute but this is v1 — correctness first. Later versions can pack multiple heads or multiple tokens.

Actually wait — re-reading the FlashMLA config.h: `B_H = 64` and they have 64 heads. In our case we have 16 heads. The question is whether head_dim matters for M.

Let me reconsider. In the competition:
- Q: [num_tokens, 16, 512] — 16 rows of 512 elements
- K: [B_TOPK, 512] — 64 rows of 512 elements
- QK GEMM: M=16, N=64, K=512

The M dimension (16 query heads) is the problem. UTCMMA requires M=64 or M=128.

**V1 approach: Pad M to 64**.
- Load Q into the first 16 rows of a 64-row TMEM/smem buffer, zero the rest.
- MMA computes 64×64 output, we only read the first 16 rows.
- 75% compute waste, but QK GEMM time is ~1,044 ns measured — even at 4× waste this is ~4,176 ns, still comparable to TMA time (~3,752 ns). The kernel remains TMA-bound.
- SV GEMM: same M=64 padding. 2 × M=64 N=256 K=4 = 2 × 539 cy = 2,156 ns. With 4× waste = ~8,624 ns. Hmm, this makes compute a potential bottleneck.

**Better v1 approach: Pack 4 tokens into one CTA when num_tokens >= 4**.

Actually, re-reading more carefully — the competition has num_tokens=[1,2,6,7,8]. Let me think about this differently.

**V1 PRAGMATIC approach: Process ALL 16 heads sequentially in a simple kernel, using standard PyTorch/cuBLAS for GEMMs.**

No — the task is CuTeDSL. Let me commit to a design.

**FINAL V1 M-DIMENSION DECISION**:

Use M=64 by processing **4 query tokens together** when available, with each token contributing 16 heads. When num_tokens < 4, pad with zeros.

Actually this is getting complicated. Let me step back.

**SIMPLEST CORRECT APPROACH for V1**: Use a Triton kernel (which the bench framework supports via `lang: triton`) as v1, then move to CuTeDSL for v2.

No — let me commit to CuTeDSL. The M=64 constraint is manageable:

**V1 FINAL DECISION: M=64, pad 16→64 rows.**

Revised TMEM columns:
```
S accumulator:  M=64, N=64   → 64 cols   (cols 0-63)
O accumulator:  M=64, N=256  → 256 cols  (cols 64-319) — 2 SV tiles share this
Q_nope TMEM:    M=64, D=512  → 64*512/(2*128) = 128 cols??
```

Hmm, TMEM Q storage for M=64, D=512 BF16: that's 64×512 = 32K elements = 64KB. In TMEM: 64 data-paths × (512/2) columns (BF16 = 2 bytes, each cell is 4 bytes) = 128 cols. Total with S+O+Q would be 64+256+128 = 448 cols. This fits in 512.

Actually, Q_pe also needs space: 64×64 BF16 = 8K elements = 64 data-paths × (64/2) = 16 cols. Total: 448+16 = 464 cols. This matches the FlashMLA config exactly (they also use 464 cols for their D_Q=512+64=576 design).

### 2.5 Revised TMEM Column Plan

```
S (QK accum):    cols 0-63      64 cols  (M=64, N=64, FP32)
O (SV accum):    cols 64-319    256 cols (M=64, N=256, FP32)
Q_nope (TMEM A): cols 320-447   128 cols (M=64, K=512, BF16)
Q_pe (TMEM A):   cols 448-463   16 cols  (M=64, K=64, BF16)
Total:           464 cols of 512
```

### 2.6 Warp Specialization (v1 — Simplified)

Following FlashMLA's pattern with fewer warps:

**3 warpgroups, 384 threads (12 warps)**:

| Warpgroup | Warps | Role | Registers |
|-----------|-------|------|-----------|
| WG0 (128T) | 0-3 | Softmax + epilogue | 224 |
| WG1 (128T) | 4-7 | MMA (warp 4) + TMA load (warp 5,6) + index prep (warp 7) | 72 (deallocated from MMA+TMA warps) |
| WG2 (128T) | 8-11 | Idle in v1 (dequant warps in FlashMLA for FP8, not needed for BF16 attention) | 48 |

**V1 simplification**: Reduce to **2 warpgroups, 256 threads (8 warps)**:

| Warp | Role | Details |
|------|------|---------|
| 0-3 | Softmax + output write | WG0: read S from TMEM, online softmax, write P to smem, read O from TMEM, convert BF16, write to global |
| 4 | MMA | Single thread issues UTCMMA for QK ckv, QK kpe, SV GEMMs. Issues commit to signal softmax warps |
| 5 | TMA load (ckv) | Issues TMA gather4 for ckv_cache pages |
| 6 | TMA load (kpe) | Issues TMA gather4 for kpe_cache pages |
| 7 | Index preparation | Loads sparse_indices via LDG, computes TMA coordinates (page_idx, offset), handles -1 masking |

### 2.7 Shared Memory Layout

```
sQ_nope:  [64, 512] BF16, SW128    = 64 KB (loaded once via TMA, copied to TMEM via UTCCP)
sQ_pe:    [64, 64] BF16, SW128     = 8 KB  (loaded once, UTCCP to TMEM)
sKc:      [64, 512] BF16, SW128    = 64 KB × 2 stages = 128 KB (double-buffered for TMA pipeline)
sKp:      [64, 64] BF16, SW128     = 8 KB × 2 stages = 16 KB (double-buffered)
sP:       [64, 64] BF16, K_INTER   = 8 KB  (softmax output, written by WG0, read by MMA for SV)
sO:       [64, 512] BF16, SW128    = 64 KB (output staging for TMA store or direct write)

Barriers:  ~1 KB
TMA coords: 64 int32 × 4 buffers = 1 KB
TMEM addr:  4 bytes

Total: ~64+8+128+16+8+64+1+1 = ~290 KB
```

This exceeds the 228 KB shared memory limit. We need to reduce.

**V1 revised smem plan** — overlap Q and O (they don't coexist):
```
Union { sQ_nope[64,512], sO[64,512] }:  64 KB
sQ_pe[64,64]:                            8 KB
sKc[64,512] × 2 stages:                 128 KB
sKp[64,64] × 2 stages:                  16 KB
sP[64,64]:                               8 KB
Metadata (barriers, coords):             ~2 KB
Total: 64+8+128+16+8+2 = 226 KB ✓ (under 228 KB)
```

Wait — this is extremely tight. And we're padding Q from 16 to 64 rows, so the real Q data is only [16, 512] = 16 KB for nope and [16, 64] = 2 KB for pe. But TMA loads a full [64, 512] tile regardless.

**V1 smem revision — use actual Q sizes and pad in TMEM**:
We can't TMA a [16, 512] Q into a 64-row buffer directly if the source tensor is [num_tokens, 16, 512]. The Q heads are the second dimension. We need to think about this more carefully.

For Q: shape is [num_tokens, 16, 512] BF16. For a single token, Q_nope = [16, 512]. We need M=64 for MMA. Options:
1. Load Q[16,512] into smem, UTCCP to TMEM rows 0-15, zero TMEM rows 16-63
2. Pad at the smem level: load Q[16,512] into first 16 rows of sQ[64,512]

With approach 1, sQ only needs 16×512×2 = 16 KB, not 64 KB. Much better.

**V1 final smem plan**:
```
sQ_nope[16,512] BF16:                    16 KB (or union with sO)
sQ_pe[16,64] BF16:                       2 KB
sKc[64,512] × 2 stages:                 128 KB
sKp[64,64] × 2 stages:                  16 KB
sP[64,64] BF16:                          8 KB  (actually [16,64] since only 16 valid heads, but padded for MMA)
sO[16,512] BF16:                         16 KB (or union with sQ)
Metadata:                                ~2 KB
Total: 16+2+128+16+8+16+2 = 188 KB ✓
```

Even if we union sQ and sO (they don't coexist): 16+2+128+16+8+2 = 172 KB. Comfortable.

But wait — sP needs to be [64, 64] for MMA input (SS mode with M=64). Even though only 16 rows have real data, MMA reads all 64 rows. The garbage in rows 16-63 doesn't matter for correctness because the corresponding output rows 16-63 are also garbage (we ignore them).

Actually, for SV GEMM in SS mode: `P[M=64, K=64] @ Kc_transposed[K=64, N=256]` → `O[M=64, N=256]`. P is operand A. The padding rows in P (16-63) should be zero so that the output O rows 16-63 are zero (or at least don't corrupt the valid rows 0-15). Since P is written by the softmax stage which only processes 16 valid rows, rows 16-63 remain at whatever was in smem before. We should zero sP at init.

### 2.8 Pipeline and Data Flow

**Initialization (once per CTA)**:
1. TMA load Q_nope[16,512] and Q_pe[16,64] into smem
2. UTCCP: copy Q_nope from smem to TMEM rows 0-15 (128 copies of 16dp128b for 512d = 16 UTCCP calls?)
3. UTCCP: copy Q_pe from smem to TMEM
4. Zero TMEM rows 16-63 for Q (or just zero the O accumulator and accept garbage in S rows 16-63 which get masked in softmax)

Actually, important correction: the UTCMMA M dimension maps to TMEM lanes (data-paths), and Q occupies TMEM as operand A in TS mode. The 16 valid heads map to lanes 0-15. Lanes 16-63 will have uninitialized Q data. The resulting S accumulator rows 16-63 will be garbage. During softmax, we only process rows 0-15. For SV GEMM, P rows 16-63 should be zero. The output O rows 16-63 will be garbage but we only read rows 0-15.

So the only constraint is: **zero P (smem) rows 16-63 before SV GEMM**, or equivalently, only write P rows 0-15 during softmax and ensure the rest is zero-initialized.

**Mainloop (32 iterations for topk=2048)**:

```
for chunk = 0..31:
    // Warp 7: Load sparse_indices[chunk*64..(chunk+1)*64], compute TMA coords
    // Warp 5: TMA gather4 for Kc (4 gather4 calls × 16 pages = 64 tokens)
    // Warp 6: TMA gather4 for Kp (4 gather4 calls)
    // Wait for TMA

    // Warp 4 (MMA):
    //   S = Q_nope @ Kc.T  (TS, M=64 N=64 K=32, zero_init=true)
    //   S += Q_pe @ Kp.T   (TS, M=64 N=64 K=4, accumulate)
    //   commit → signal softmax

    // WG0 (softmax):
    //   Wait for MMA commit
    //   Read S[16,64] from TMEM (only rows 0-15)
    //   Online softmax: update row_max, row_sum, compute exp2, rescale O
    //   Write P[16,64] to smem (zero-pad rows 16-63)
    //   Signal MMA for SV

    // Warp 4 (MMA):
    //   Wait for P ready
    //   O += P @ Kc  (SS, M=64 N=256 K=4, accumulate)
    //   Two tiles: O[0:256] and O[256:512]
    //   commit → signal next iteration or epilogue
```

**Epilogue**:
```
// WG0:
//   Read O[16,512] from TMEM
//   Divide by row_sum
//   Convert to BF16
//   Write to output[token, :16, :512]
//   Compute LSE = (row_max * scale_log2 + log2(row_sum)) * ln(2)
//   Write LSE to lse[token, :16]
```

### 2.9 TMA Configuration

**ckv_cache** [num_pages, 64, 512] BF16:
- TMA 2D tensor map: dim0 = 512 (head_dim), dim1 = 64*num_pages (page_size * num_pages)
- Box size: [512, 4] = 4 tokens × 512d × 2 bytes = 4,096 bytes per gather4
- Swizzle: SW128 (for UTCMMA K-major layout)
- 16 gather4 calls per chunk (64 tokens / 4 per call)
- Cache hint: evict_last

**kpe_cache** [num_pages, 64, 64] BF16:
- TMA 2D tensor map: dim0 = 64 (head_dim_kpe), dim1 = 64*num_pages
- Box size: [64, 4] = 4 tokens × 64d × 2 bytes = 512 bytes per gather4
- Swizzle: SW128
- 16 gather4 calls per chunk
- Cache hint: evict_first

**Index mapping**: sparse_indices[t, chunk*64 + i] gives a global token index. From that:
- page_idx = token_index / 64
- offset = token_index % 64
- TMA row coordinate = page_idx * 64 + offset = token_index (it's already the flat index)
- Actually: the tensor map maps to the flattened [num_pages*64, dim] view, so TMA coordinate = token_index directly

### 2.10 Softmax Details

Online softmax with log2 base (matching FA4's SoftmaxSm100):

```python
scale_log2 = sm_scale / ln(2)  # = sm_scale * log2(e)

# Per iteration (chunk):
for row in 0..15:  # 16 heads
    S_row = TMEM_read(S, row)  # [64] float32
    row_max_cur = max(S_row) across all 64 elements
    # warp reduction for row_max (quad-reduce since 4 threads per row)

    if first_chunk:
        S_exp = exp2((S_row * scale_log2) - (row_max_cur * scale_log2))
        row_sum = sum(S_exp)
        row_max[row] = row_max_cur
        row_sum[row] = row_sum
    else:
        row_scale = exp2((row_max_prev - row_max_cur) * scale_log2)
        S_exp = exp2((S_row * scale_log2) - (row_max_cur * scale_log2))
        row_sum[row] = row_sum[row] * row_scale + sum(S_exp)
        row_max[row] = row_max_cur
        # O needs rescaling: handled in TMEM (O *= row_scale for each accumulated chunk)

    # Write S_exp as P (BF16) to smem
    P_row = convert_to_bf16(S_exp)
    smem_write(sP, row, P_row)
```

**LSE finalize** (after all chunks):
```python
lse[row] = (row_max[row] * scale_log2 + log2(row_sum[row])) * ln(2)
```

Wait, the reference says: `lse = logsumexp(logits_scaled) / log(2)`. So LSE is in **log2 base**: `lse = log2(sum(exp(logits_scaled)))`.

Using the identity: `log2(sum(exp(x))) = max(x) * log2(e) + log2(sum(exp(x - max(x))))` ... actually:

`logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`

So `lse_log2 = logsumexp(x) / log(2) = (max(x) + log(sum(exp(x-max(x))))) / log(2)`

In terms of our tracked variables (using exp2 formulation):
- `row_max` = max of unscaled logits
- `scale_log2 = sm_scale * log2(e)`
- `S_scaled = S * sm_scale`
- `exp(S_scaled - max(S_scaled)) = exp2((S * scale_log2) - (max(S) * scale_log2))`

So: `lse_log2 = max(S) * scale_log2 + log2(row_sum)` where row_sum = sum of exp2 terms.

This is clean and matches FA4's approach.

### 2.11 O Rescaling in Split-KV

In split-KV mode, each CTA processes one chunk and produces a partial result. The combine kernel handles the cross-chunk softmax correction. So within each CTA, we don't need O rescaling across chunks.

**Wait — v1 uses split-KV, one chunk per CTA.** That means each CTA computes:
- partial_O = softmax(S_chunk) @ Kc_chunk — a single chunk, no online softmax needed
- partial_lse = logsumexp(S_chunk * sm_scale) / log(2)

This is even simpler than online softmax. Within one chunk:
```python
S = QK_ckv + QK_kpe  # [16, 64]
S_scaled = S * sm_scale
row_max = max(S_scaled, dim=-1)  # [16]
S_exp = exp(S_scaled - row_max[:, None])  # [16, 64]
row_sum = sum(S_exp, dim=-1)  # [16]
P = S_exp / row_sum[:, None]  # [16, 64]
O = P @ Kc  # [16, 512]
lse = (row_max + log(row_sum)) / log(2)  # [16]
```

Actually no — we need the UNNORMALIZED partial O and partial LSE for the combine kernel to merge correctly:
```python
partial_O = S_exp @ Kc  # unnormalized: O * row_sum
partial_lse = row_max + log(row_sum)  # in natural log (or log2 base, depends on combine kernel)
```

Wait, examining FA4's FlashAttentionForwardCombine: it expects partial outputs and partial LSEs. The combine kernel does:
```python
# For each split s:
#   O_partial[s] = softmax(S_s) @ V_s  (normalized within the split)
#   lse_partial[s] = logsumexp(S_s * scale)
# Combine:
#   lse_total = logsumexp(lse_partial[0], ..., lse_partial[31])
#   O_total = sum_s(O_partial[s] * exp(lse_partial[s] - lse_total))
```

So each CTA outputs the **normalized** attention output for its chunk, plus the partial LSE. The combine kernel handles the renormalization.

This is the cleanest approach for v1.

### 2.12 Combine Kernel

Use FA4's `FlashAttentionForwardCombine` or write a simple CuTeDSL equivalent.

**V1 approach**: Write a simple PyTorch combine:
```python
# partial_O: [num_splits, num_tokens, 16, 512] BF16
# partial_lse: [num_splits, num_tokens, 16] float32
# Output: O[num_tokens, 16, 512] BF16, lse[num_tokens, 16] float32

lse_max = partial_lse.max(dim=0)  # [num_tokens, 16]
weights = exp(partial_lse - lse_max)  # [num_splits, num_tokens, 16]
weights_sum = weights.sum(dim=0)  # [num_tokens, 16]
O = (weights[:,:,:,None] * partial_O.float()).sum(dim=0) / weights_sum[:,:,None]
lse = lse_max + log(weights_sum)  # in natural log, then / log(2) for log2 base
```

**V1 simplification**: Actually, for v1, let's avoid split-KV entirely and use online softmax within a single CTA. This avoids the complexity of partial outputs, combine kernels, and intermediate storage.

### 2.13 V1 REVISED: Single-CTA Online Softmax

**Grid**: (num_tokens, 1, 1) — one CTA per token, processing all 32 chunks sequentially.

This is simpler but means:
- For num_tokens=1: only 1 SM active out of 148 (terrible utilization)
- For num_tokens=8: 8 SMs active (still terrible)

**V1 REVISED AGAIN**: Use split-KV but with a simpler combine.

Actually, let me be pragmatic. For v1, the priority is **correctness**. Let me use the single-CTA approach with online softmax. The performance will be bad (1-8 SMs) but correctness is guaranteed. V2 will add split-KV.

No — the competition measures speedup over a PyTorch reference. The reference is also slow (Python loop). We need to at least match the reference. Let's do split-KV.

**FINAL V1 DECISION**: Split-KV with 32 splits, each CTA processes one chunk. Combine via PyTorch.

### 2.14 Memory Layout for Partial Results

For split-KV, we need temporary storage:
```
partial_O:   [32, num_tokens, 16, 512] BF16  — 32 × 8 × 16 × 512 × 2 = 8 MB max
partial_lse: [32, num_tokens, 16] float32     — 32 × 8 × 16 × 4 = 16 KB
```

These are allocated by the Python wrapper and passed to the CUDA kernel.

### 2.15 Implementation Plan (CuTeDSL)

#### Phase 1: Kernel class setup
```python
class DSASparseAttention:
    def __init__(self):
        self.m_block = 64     # M dimension (padded from 16 heads)
        self.n_block = 64     # B_TOPK
        self.num_splits = 32  # topk / n_block
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.num_heads = 16
        # TMA, MMA, pipeline configs...
```

#### Phase 2: TMA descriptor creation
- ckv_cache TMA: 2D gather4, dim0=512×2B, box=[1024B, 4], SW128
- kpe_cache TMA: 2D gather4, dim0=64×2B, box=[128B, 4], SW128
- Q TMA: regular 2D copy (not gather)
- O TMA store: regular 2D copy

#### Phase 3: Kernel body
- Warp specialization per FlashMLA pattern
- Mainloop with 1 chunk per CTA (no loop needed)
- Softmax in WG0, MMA in warp 4, TMA in warps 5-6

#### Phase 4: Python wrapper
- Allocate partial_O, partial_lse
- Create TMA descriptors
- Launch attention kernel
- Launch combine kernel (PyTorch)
- Write final output and LSE

### 2.16 Key CuTeDSL References

1. **FA4 forward**: `references/flash-attention/flash_attn/cute/flash_fwd_sm100.py` — warp specialization, pipeline barriers, TMEM management, softmax
2. **FA4 combine**: `references/flash-attention/flash_attn/cute/flash_fwd_combine.py` — partial result merging
3. **Quack GEMM**: `references/CuTeDSL-kernels/quack/gemm_sm100.py` — TMA setup, smem layout, MMA configuration
4. **FA4 blackwell_helpers**: `references/flash-attention/flash_attn/cute/blackwell_helpers.py` — gemm_ptx function for UTCMMA
5. **FA4 softmax**: `references/flash-attention/flash_attn/cute/softmax.py` — online softmax, row_max/row_sum tracking
6. **FA4 pipeline**: `references/flash-attention/flash_attn/cute/pipeline.py` — PipelineStateSimple
7. **learn-cuda matmul_v7**: `references/learn-cuda/02e_matmul_sm100/` — basic CuTeDSL sm100 kernel structure

---

## Part 3: TopK Indexer Kernel (v1)

### 3.1 Algorithm Summary

Per batch element b:
```
1. For each page p in block_table[b]:
     Load K_page = k_index_cache_fp8[page_idx]  # [64, 128] FP8 + [64] scales
     Dequant: K_float = FP8_to_float(K_page) * scales
     scores_page = relu(q[b] @ K_float.T)       # [64, 64] float32 (64 heads × 64 tokens)
     weighted_page = scores_page * weights[b][:, None]  # [64, 64]
     final_page = weighted_page.sum(dim=0)       # [64] — aggregated scores for this page
2. TopK select top-2048 from all final_page scores
3. Convert to global indices
```

### 3.2 V1 Approach: PyTorch Reference (Correct Baseline)

For v1, the indexer uses the existing PyTorch reference implementation. It's correct and the bench framework supports pure Python solutions.

Rationale:
- The indexer is less latency-critical than attention (runs once per batch, not per-token)
- The FP8 dequant format is complex and error-prone
- Getting attention working first provides more competition value

### 3.3 V2 Indexer Design Preview (for future iterations)

When we implement the indexer in CuTeDSL:

**Grid**: (batch_size, 1, 1) — one CTA per batch element

**Per CTA**:
1. Stream through pages: TMA load K_page (FP8), dequant in registers
2. FP8 GEMM: q[64,128] @ K[64,128].T → [64,64] using tcgen05.mma kind::f8f6f4
   - M=64, N=64, K=128 (4 K-tiles of 32 FP8 elements)
   - Measured: 333.6 cy = 181 ns per page
3. ReLU + weighted sum: register operations, no TMEM needed
4. Accumulate scores per token across all pages
5. TopK: bitonic sort or radix select (measured ~3 µs for seq_len=5824)

**Key consideration**: The FP8 deep_gemm format stores data non-intuitively:
- Layout: [num_pages, page_size * (128 + 4)] bytes flattened
- First page_size*128 bytes: FP8 data
- Last page_size*4 bytes: float32 scales (one per token)
- NOT [page_size, 132] — the data and scales are stored in separate contiguous blocks per page

This format complicates TMA loading. V2 will address this.

---

## Part 4: Implementation Roadmap

### V1 Sprint (Current)
1. **Attention kernel** in CuTeDSL with:
   - Split-KV (32 splits, one chunk per CTA)
   - M=64 padded from 16 heads
   - Warp specialization: 8 warps (MMA, 2×TMA, index, 4×softmax+epi)
   - TMA gather4 for ckv and kpe
   - Single QK ckv + QK kpe GEMM → softmax → SV GEMM per CTA
   - Combine via PyTorch (temporary)
2. **Indexer kernel**: PyTorch reference (pass-through)
3. Target: passes correctness validation, establishes baseline speedup

### V2 Optimizations (Post-V1 profiling)
- Online softmax (eliminate combine kernel overhead)
- Multiple chunks per CTA (amortize Q load, reduce combine work)
- Pack multiple tokens into M dimension
- TMA pipeline depth N=16 for deep prefetching
- Sort sparse_indices for locality
- CuTeDSL indexer kernel with FP8 GEMM

### V3+ Advanced
- Persistent kernel with tile scheduler
- 2-CTA instructions for doubled throughput
- Custom combine kernel in CuTeDSL
- Indexer with block-scaled FP8 MMA

---

## Part 5: Expected Outcome

### V1 Performance Estimate

**Attention kernel** (per token, 32 CTAs):
- TMA per CTA: 16 ckv + 16 kpe gather4 calls ≈ 3,752 ns (pipelined N=16)
- QK GEMM: 1,044 ns (M=64 N=64 K=32 ckv) + 181 ns (kpe) = 1,225 ns
- SV GEMM: 2 × 539 ns = 1,078 ns
- Softmax: ~500 ns (estimated, unmeasured)
- Total per CTA: ~6,555 ns (sequential TMA+compute)
- All 32 CTAs run in parallel on 32 SMs
- Wall time per token ≈ 6.6 µs + combine overhead (~1-2 µs) ≈ 8 µs
- For num_tokens=8: 8 × 32 = 256 CTAs on 148 SMs, ~2 waves ≈ 16 µs

**Reference** (PyTorch): For num_tokens=1, the reference runs a Python loop over 2048 sparse indices with torch matmuls. Expected: 1-10 ms.

**Expected speedup**: 50-100× over reference for single token. The M=64 padding wastes 75% of MMA compute, which v2 will address.

### V1 Risk Assessment

1. **Highest risk**: TMA gather4 coordinate generation from sparse_indices. The index warp must translate token indices into TMA-compatible coordinates, handling -1 padding. This is the most novel piece and has no exact reference.

2. **Medium risk**: TMEM Q loading via UTCCP for 16 rows (not 64). Need to verify that UTCCP with M=16 source correctly populates TMEM lanes 0-15.

3. **Low risk**: Softmax, MMA, combine — these follow well-established FA4 patterns.
