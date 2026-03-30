# Deep Research Directions — DSA Decode Kernel on B200

Status: **research context only**. These are architectural explorations, not
incremental patches. Each direction requires prototyping and measurement.

Hardware baseline (measured on your Modal B200):
- 148 SMs, 1.965 GHz
- TMEM: 256 KB/SM, 16/8 TB/s read/write
- SMEM: 228 KB/SM, 128 B/cycle/SM bandwidth (unchanged from Hopper)
- L2: 126.5 MB, ~8.7 TB/s aggregate
- HBM3e: 7.48 TB/s sustained
- BF16 MMA: 8,192 ops/clock/SM (2x Hopper)
- Exp unit: 16 ops/clock/SM (unchanged from Hopper) — **512:1 ratio vs MMA**
- TMA gather4: 650 ns cold / 197 ns L2-resident / latency-bound not BW-bound

---

## Direction 1: Tensor-Core Softmax (FlashAttention-4 Pattern)

**The asymmetry that matters most.**

Blackwell doubled MMA throughput but left the exponential unit at 16 ops/clock.
That's a 512:1 ratio. For a 16-head × 64-column softmax, WG2 needs 1,024
exp2 calls per split. At 16 ops/clock, that's 64 cycles — comparable to or
exceeding the MMA time for a 64×64 tile.

FlashAttention-4 (arxiv:2603.05451) solves this by **emulating exp2 as a
polynomial approximation using tensor cores**:

```
exp2(x) ≈ 2^floor(x) * P(frac(x))
```

where `P(frac(x))` is a degree-2 or degree-3 polynomial evaluated via MMA:

```
// Conceptual: evaluate polynomial on an entire row at once via GEMM
// coefficients are broadcast, x values are the score row
result = c0 + c1 * frac(x) + c2 * frac(x)^2
```

The 2^floor(x) part is handled via bit manipulation (FP32 exponent field).

**Why this is deep**: It completely restructures WG2. Instead of scalar exp2
in a loop, you'd issue a small MMA for the polynomial evaluation. This means
WG2 needs TMEM access for the polynomial MMA — requiring TMEM partitioning
between the score accumulator, the polynomial workspace, and the output
accumulator.

**Expected impact**: 2-5x speedup on softmax phase. If softmax is 15-20%
of total, that's 10-15% overall. But more importantly, it **removes the
scaling bottleneck** that would otherwise cap all other optimizations.

**Research needed**:
- What polynomial degree gives sufficient precision for BF16 output?
  (FA4 uses degree-2 with error < 2^-10)
- Can we share the MMA unit between WG1 (GEMM) and WG2 (polynomial eval)?
- TMEM column budget: current QK/PV accumulator uses N columns. Polynomial
  workspace needs additional columns.

**Key reference**: FlashAttention-4 paper Section 3.2 "Tensor-core softmax",
and `csrc/cutlass/examples/python/CuTeDSL/blackwell/fmha.py` correction
epilogue which implements the polynomial evaluation.

---

## Direction 2: Cross-Split Online Accumulation (Persistent Fused Combine)

**Eliminate the 32 × partial_o HBM round-trip.**

Currently: 32 CTAs each write `partial_o[64×512]` FP32 + `partial_lse[16]`
to HBM. Then `_combine_splits` reads all 32 back and merges. That's:

```
Write: 32 × 16 × 512 × 4B = 1 MB per token (to HBM)
Read:  same 1 MB (from HBM)
Compute: online log-sum-exp merge
Write: 16 × 512 × 2B = 16 KB final output
```

2 MB of HBM traffic for 16 KB of useful output. 128:1 amplification.

**Alternative**: A persistent CTA processes all 32 splits sequentially,
maintaining a running `(O_acc, lse_acc)` pair in TMEM/registers:

```
for split in 0..31:
    # Phase 1: QK + softmax → P, lse_new
    # Phase 2: PV → O_new
    # Online merge:
    max_lse = max(lse_acc, lse_new)
    w_old = exp2(lse_acc - max_lse)
    w_new = exp2(lse_new - max_lse)
    O_acc = w_old * O_acc + w_new * O_new
    lse_acc = max_lse + log2(w_old + w_new)
```

**Total HBM traffic**: just the final 16 KB output + KV cache reads.
The 1 MB partial_o intermediate vanishes entirely.

**Why this is deep**: It requires:
1. Persistent kernel (TMEM allocated once, loops over splits)
2. TMEM budget for both `O_acc` (16×512 FP32 = 32 KB → needs 64 columns
   at 16 heads) AND the per-split QK score accumulator (16×64 FP32)
3. The online merge is 2 exp2 calls + 3 multiplies + 1 add per element.
   With tensor-core softmax (Direction 1), the exp2 becomes free.
4. Index/barrier state must reset between splits without re-allocating TMEM.

**Expected impact**: Eliminates the combine kernel entirely + 2 MB HBM
traffic. For T=1 (decode), this could be 30-50% of total time if the
current combine + HBM traffic is on the critical path.

**Synergy**: Combines naturally with Direction 1 (tensor-core exp2 for
the online merge) and Direction 5 (FP8 KV cache for faster gathers).

---

## Direction 3: Eliminating M=64 Padding via Multi-Token Batching

**75% of MMA compute is currently wasted.**

UTCMMA minimum tile is M=64. With 16 real query heads, we pad to 64 and
waste 48 rows of zeros through every GEMM.

**Insight**: Batch 4 tokens into one CTA. Each token contributes 16 heads,
4 × 16 = 64 = M_BLOCK. Zero padding eliminated.

```
Grid: [ceil(T/4), NUM_SPLITS, 1]  instead of  [T, NUM_SPLITS, 1]
Each CTA processes tokens [4*bidx .. 4*bidx+3] for one split
```

**Complications**:
1. Each token has **different sparse_indices**. The gather4 for token 0
   loads different KV rows than token 1. With 4 tokens, WG0 needs to
   issue 4 separate gather4 sequences (or use a unified index set).
2. The QK GEMM now computes 4 independent dot products stacked vertically.
   The K operand is shared across all 4 tokens (same split = same KV chunk).
   But Q_nope differs per token.
3. Softmax operates on independent rows (no cross-token interaction), so
   WG2 processes 64 rows but with 4 different validity masks.
4. PV MMA: P is 64×64, V is 64×64. Output is 64×64. The 4 tokens'
   contributions are in different row ranges of the output.

**Key question**: Do the 4 tokens in a batch share the same sparse indices?
In the competition setup, each token has independent topk indices. So the
Q operand changes per token but K/V are gathered independently.

**If indices differ**: WG0 must gather 4 different K chunks, breaking
the shared-B-operand advantage. Net gain is reduced.

**If we can sort tokens by similar index sets**: Tokens with high index
overlap could share gathers. This is a host-side scheduling problem.

**Expected impact**: Up to 4x MMA efficiency, but gather overhead may
eat 50-75% of the gain. Net: 1.5-2x on compute-bound workloads.

---

## Direction 4: Phase Overlap — QK and V Gather Concurrent

**Currently Phase 1 (QK+softmax) and Phase 2 (PV) are strictly sequential.**

After QK GEMM + softmax, the kernel knows which V tiles it needs.
But V gathering doesn't start until Phase 1 is fully complete.

**Opportunity**: Start V tile 0 gather during the softmax computation:

```
Timeline (current):
  WG0: [gather CKV] [gather KPE] ---- [gather V0] [gather V1] ...
  WG1: ------------- [QK GEMM] ------- [PV tile0] [PV tile1] ...
  WG2: -------------------------[softmax]--------- [epilogue] ...

Timeline (overlapped):
  WG0: [gather CKV] [gather KPE] [gather V0] [gather V1] [gather V2] ...
  WG1: ------------- [QK GEMM] -------------- [PV tile0] [PV tile1] ...
  WG2: -------------------------[softmax+P stg] --------- [epilogue] ...
                                      ^--- V0 gather overlaps with softmax
```

**Why this is non-trivial**: V gather uses the same `sQrow` staging buffer
as CKV/KPE. We'd need a second staging buffer, or reuse `sK` (which is
free after QK GEMM completes).

Also: the `gather_mbar` is currently single-phase. Overlapping requires
a multi-stage gather pipeline where WG0 can issue gather V0 while WG2
is still doing softmax on the QK scores.

**Expected impact**: Hides V gather latency behind softmax compute.
With gather4 at ~200-650 ns per tile and 8 tiles, that's 1.6-5.2 us
of gather time. Hiding even 1-2 tiles saves 200-400 ns per split.

---

## Direction 5: FP8 KV Cache

**The competition spec mentions FP8 KV caches. This changes everything.**

If CKV is stored as FP8 (1 byte/element instead of 2):
1. TMA gather4 payload halves: 2048 B instead of 4096 B per gather4 call
2. L2 effective capacity doubles (more KV fits in 126.5 MB)
3. MMA can use `kind::f8f6f4` for 2x throughput over BF16

**But**: FP8 QK GEMM accumulates in FP32 (same as BF16), and the output
precision may differ. The competition likely has tight tolerance thresholds.

**Design implications**:
- Need FP8→BF16 upcast for V before PV MMA (or use FP8 PV MMA if
  precision allows)
- Mixed-precision pipeline: FP8 gathers, FP8 QK MMA, FP32 softmax,
  FP8 or BF16 PV MMA
- Reference: `mixed_input_fmha_decode.py` already handles mixed FP8/BF16

**Expected impact**: 2x on gather bandwidth, 2x on QK MMA throughput.
Combined: potentially 40-60% overall speedup.

---

## Direction 6: L2 Exploitation via Index Clustering

**TMA gather4 is latency-bound, not bandwidth-bound.**

Your benchmarks show 296 CTAs achieve only 18.9% of HBM peak bandwidth
for random scatter gathers. The bottleneck is L2 miss latency (650 ns
cold → 197 ns L2-resident).

With 126.5 MB L2 and typical KV cache:
```
8462 pages × 64 tokens × 512 dims × 2 bytes = 554 MB (CKV)
8462 pages × 64 tokens × 64 dims × 2 bytes = 69 MB (KPE)
Total: 623 MB, only ~20% fits in L2
```

**Strategies**:

A. **Host-side index sorting**: Sort sparse_indices by page ID before
   kernel launch. Adjacent indices in the sorted order hit the same L2
   lines. Cost: ~10 us on CPU for 2048 indices.

B. **TMA L2 prefetch**: Issue `cp.async.bulk.prefetch` for the next
   split's indices while the current split computes. Your benchmarks
   show DIST=2 prefetch gives 40% latency reduction.

C. **Cross-CTA L2 sharing**: If multiple tokens access the same KV page
   (common in batched decode), schedule those CTAs to run concurrently
   on the same SM cluster. First CTA pays the HBM latency, others hit L2.

D. **Working-set-aware scheduling**: Process splits in page-sorted order
   so consecutive splits have maximum index overlap. Combined with the
   persistent kernel (Direction 2), this means the L2 working set evolves
   slowly rather than thrashing.

**Expected impact**: L2 hit rate from ~20% to ~60-80% = 2-3x effective
gather bandwidth. At 200 ns/gather × 8 K-tiles × 2 (CKV+KPE) = 3.2 us
per split of gather time, a 2x improvement saves 1.6 us × 32 splits = 51 us.

---

## Direction 7: Warp Specialization Redesign for Blackwell

**MMA is single-thread on SM100. The 3-warpgroup design may be overkill.**

On Hopper, wgmma required a full warpgroup (128 threads) to issue MMA.
On Blackwell, `tcgen05.mma` is issued by **a single thread**. The 128-thread
warpgroup only matters for TMEM readout (128 lanes = 4 warps × 32 lanes).

**Alternative design — 2 warpgroups + 1 MMA warp**:

```
WG0 (128 threads): Gather + Q/K/V staging + index processing
WG-MMA (32 threads): Issues all MMAs (QK, PV). Single-warp, elect_one.
WG1 (128 threads): Softmax + P staging + output epilogue (TMEM readout)
Total: 288 threads (down from 384)
```

**Benefits**:
- Higher occupancy: 288 threads/CTA allows 7 CTAs/SM (vs 5 with 384)
- MMA warp is fully dedicated — no context switching between MMA and
  barrier management
- TMEM readout still uses 128 threads (WG1) for full-lane coverage

**Complications**:
- The `PipelineUmmaAsync` requires specific producer/consumer group sizes.
  A 32-thread MMA producer with 128-thread consumer needs careful barrier
  configuration.
- WG1 does both softmax AND epilogue. Softmax needs score TMEM → registers.
  Epilogue needs output TMEM → registers. Both are 128-thread operations.

**Expected impact**: 10-20% from higher occupancy if the kernel is
latency-bound (which it likely is, given the gather4 latency).

---

## Direction 8: TMEM-Resident Score Path

**Avoid the TMEM → sScoreF32 → softmax → sQrow → sP round-trip.**

Current softmax path:
```
tStS (TMEM) → registers → sScoreF32 (SMEM, 16KB FP32)
→ softmax in SMEM → sQrow (SMEM, 8KB BF16) → sP (SMEM, 8KB swizzled)
```

That's TMEM → SMEM → SMEM → SMEM. Three SMEM copies at 128 B/cycle.

**Alternative with tensor-core softmax (Direction 1)**:
```
tStS (TMEM) → tensor-core polynomial eval → tP (TMEM)
→ PV MMA reads P from TMEM (p_source = TMEM)
```

This eliminates ALL SMEM traffic for the softmax path. The score stays
in TMEM, gets transformed in-place (or into a second TMEM region), and
PV MMA reads P directly from TMEM.

**The blocker**: The checklist notes (line 50-58 in DSA_REBUILD_PROGRESS.md)
document that TMEM P store with `St32x32bOp` produced incompatible 3D shapes
for 64×64 tiles. The decode FMHA worked around this by going through SMEM.

**Research question**: Can we use a different TMEM store layout (e.g.,
`St64x16bOp` or a custom composed layout) that's compatible with 64×64 tiles?
The FMHA encode (`fmha.py`) uses `p_source = tcgen05.OperandSource.TMEM`
successfully — what tile sizes does it use, and can we adapt them?

**Expected impact**: Eliminates ~48 KB of SMEM reads/writes per split.
At 128 B/cycle, that's ~375 cycles = ~190 ns saved per split.

---

## Direction 9: 2-CTA Cooperative Mode

**Share B-operand SMEM across two SMs.**

`cta_group::2` with `tcgen05.mma` lets two CTAs cooperate on a single MMA.
Each CTA holds half the B operand in its SMEM. The hardware fetches the
other half from the peer CTA's SMEM transparently.

For our kernel: B = K (CKV or KPE). If two adjacent CTAs process the
**same split** (same K data), they can share the K operand:

```
CTA0: holds K[0:32, :] in SMEM, processes Q[token0, :16, :] (or Q[:, :32, :])
CTA1: holds K[32:64, :] in SMEM, processes Q[token0, :16, :] (or Q[:, 32:64, :])
```

**Wait — K is already only 64 rows.** With M=64 and N=64, the B operand
is K^T (64×D_tile). Splitting across 2 CTAs means each holds 32 rows of K.

**Problem**: Our M (heads) is only 16 padded to 64. With 2-CTA mode,
M doubles to 128. Now we pad 16 → 128 (87.5% waste) instead of 16 → 64
(75% waste). This is worse unless we batch 8 tokens (8×16=128).

**Viable only if combined with Direction 3** (multi-token batching).
With 8 tokens × 16 heads = 128 = M for 2-CTA mode, zero waste + shared
K operand = significant win.

**Expected impact**: 2x MMA throughput via 2-CTA tiles, minus the
complexity of multi-token batching.

---

## Direction 10: Async Gather + Compute Overlap via SMEM Double-Buffering

**The measured numbers tell the story.**

From your TMA gather4 benchmarks:
- N=1 pipeline: 650 ns per gather (cold HBM)
- N=2 pipeline: 475 ns per gather (27% improvement)
- N=16 pipeline: 235 ns per gather (2.77x improvement)

The kernel currently uses N=1 for everything. Just adding a second pipeline
stage to the V gather path gets 27% improvement on gather latency with
minimal code change.

For the CKV/KPE gathers (8+1 = 9 K-tiles in Phase 1), double-buffering
would overlap tile N+1's gather with tile N's MMA:

```
WG0: [gather K[0]] [gather K[1]] [gather K[2]] ...
WG1:               [QK tile 0]   [QK tile 1]   ...
                   ^--- overlap
```

Currently these are sequential within a single pipeline stage.

**Implementation**: 2 stages for `operand_pipe`, double the sK SMEM buffer,
prefill the pipeline with 1 look-ahead gather.

**Expected impact**: 20-30% on Phase 1 (QK GEMM phase), which is likely
the dominant phase since it has 9 K-tiles of serial dependency.

---

## Synthesis: What to prototype first

The directions compound. Here's a dependency/synergy graph:

```
                    ┌─────────────────────┐
                    │ Direction 1          │
                    │ Tensor-core softmax  │
                    └──────┬──────────────┘
                           │ enables
                    ┌──────▼──────────────┐
                    │ Direction 8          │
                    │ TMEM-resident scores │
                    └──────┬──────────────┘
                           │ enables
                    ┌──────▼──────────────┐
            ┌───────│ Direction 2          │
            │       │ Persistent + online  │
            │       │ cross-split merge    │
            │       └──────┬──────────────┘
            │              │ combines with
            │       ┌──────▼──────────────┐
            │       │ Direction 6          │
            │       │ L2 index clustering  │
            │       └─────────────────────┘
            │
    independent
            │
            │       ┌─────────────────────┐
            ├───────│ Direction 10         │
            │       │ Double-buffer gather │◄── lowest risk, start here
            │       └─────────────────────┘
            │
            │       ┌─────────────────────┐
            └───────│ Direction 5          │
                    │ FP8 KV cache         │◄── biggest single gain if applicable
                    └─────────────────────┘

    Direction 3 (multi-token) + Direction 9 (2-CTA) form a separate cluster
    that's only viable together.
```

**Recommended prototyping order**:

1. **Direction 10** (double-buffer gather) — lowest risk, concrete measured
   gain (27% per gather), small code change. Validates the pipeline
   infrastructure for all subsequent directions.

2. **Direction 2** (persistent + online merge) — eliminates the 2 MB
   HBM round-trip. Can be prototyped independently. Biggest architectural
   change but highest ceiling.

3. **Direction 1** (tensor-core softmax) — read the FA4 paper, study the
   correction epilogue in `fmha.py`. This unlocks Directions 8 and
   makes Direction 2's online merge faster.

4. **Direction 6** (L2 clustering) — pure host-side, zero kernel risk.
   Measure L2 hit rate before and after to validate.

5. **Direction 5** (FP8) — check competition spec for FP8 cache format.
   If available, this is the single highest-impact change.
