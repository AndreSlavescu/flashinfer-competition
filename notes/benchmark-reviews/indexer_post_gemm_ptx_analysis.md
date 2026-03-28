# Indexer Post-GEMM Operations: PTX Analysis and Benchmark Recommendations

**Date**: 2026-03-24
**Context**: Two new benchmarks implemented (fp8_gemm, topk_select). This note analyzes PTX instructions for steps 3-5 of the indexer pipeline (ReLU, broadcast multiply, head reduction) and prioritizes next benchmarks.

---

## Competition Workload Reality Check

From `competition-dataset/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl`:

- **batch_size**: [1, 2, 3, 4, 6, 7, 8, 11, 12, 14, 15, 16, 25, 26, 27, 29, 30, 31]
- **max_num_pages per sequence**: 1 to ~91 (full range)
- **seq_len per batch element** = `seq_lens[b]` ≤ `max_num_pages × 64`
- **Constraint**: `topk=2048 ≤ max_num_pages × page_size` → `max_num_pages ≥ 32` required for full topk
- For `max_num_pages=31`: `seq_len = 1984`, `actual_topk = 1984` (NOT 2048) — must handle degenerate case
- **CORRECTION for topk_select benchmark**: competition seq_lens are NOT always 5824. The workload axes show `max_num_pages` up to 32-91+. A realistic competition seq_len = max_num_pages × 64, which for batch_size=1 workloads can be large (e.g., num_pages=11923 but per-batch max_num_pages capped by workload axis). Verify actual seq_len distribution from safetensors files before drawing conclusions.

**Score array dimensions after FP8 GEMM**:
- `scores[64 heads, seq_len]` stored in TMEM (float32)
- `scores` occupies N_cols TMEM columns = seq_len (or tiled if seq_len > available TMEM)
- `final_scores[seq_len]` = sum of 64 weighted, ReLU'd rows

---

## Section 1: PTX Instructions for Steps 3-5

### Step 3: ReLU Activation — `relu(scores[64, seq_len])`

**Reference formula**: `scores_relu = torch.relu(scores)` — elementwise max(0, x) on float32.

**PTX options**:

#### Option A: `max.f32 dst, src, 0f00000000`
- Standard PTX float max with literal zero.
- Latency: 4 cycles. Two-per-clock throughput.

#### Option B: `max.f32x2 dst, src, zeroVec` (sm100a exclusive)
- PTX ISA 8.5+: `max.f32x2` takes two packed f32x2 operands.
- Processes 2 scores per instruction — 2x throughput improvement.
- **Competition relevance**: HIGH.

**Recommendation**: Use `max.f32x2`. No existing benchmark covers it.

---

### Step 4: Broadcast Multiply — `w[64] × scores_relu[64, seq_len]`

#### Strategy E: Pre-absorb w into Q before FP8 GEMM (CRITICAL insight)
- `final_scores = sum_heads(relu(q_b @ K.T) * w)` can be rewritten as:
  `q_scaled = q_b * w[:, None]` → `final_scores = sum_heads(relu(q_scaled @ K.T))`
- Eliminates step 4 entirely from the post-GEMM hot path
- **Caveat**: Q is already FP8; scaling by w requires dequant→scale→requant
- This is a kernel design question, not a benchmark question

**Recommendation**: Benchmark `mul.f32x2` broadcast multiply as the building block.

---

### Step 5: Head Reduction — `sum_heads(scores_relu × w)` → `final_scores[seq_len]`

Column reduction across 64 rows of [64, seq_len] float32.

**Best approach**: 2-warp design:
1. Each warp covers 32 heads
2. TMEM read → ReLU → scale by w → intra-warp `shfl_down` sum → cross-warp SMEM exchange
3. 5 `shfl.sync.down.b32` + 5 `add.f32` = ~20 cycles per warp

**Note**: NO `redux.sync.add.f32` in PTX ISA — integer redux only. Warp shuffle is the only option.

---

## Section 2: FP8 GEMM Missing Configurations

### CRITICAL: `kind::mxf8f6f4` (block-scale variant)
- `tcgen05.mma.cta_group::1.kind::mxf8f6f4` — supports per-block scale factors in MX format
- This is what the competition k_index_cache_fp8 deep_gemm format uses
- Hardware applies scales during MMA — eliminates post-GEMM scale multiply
- SS mode only (both A and B from SMEM), scale factors stored in TMEM
- Found in CUTLASS: `SM100_MMA_MXF8F6F4_SS` at `cute/arch/mma_sm100_umma.hpp`

### Missing: SS mode for `kind::f8f6f4`
- Both A and B from SMEM (no TMEM for A)
- Frees TMEM columns for accumulator
- Requires `.ws` (warp-specialized) form

---

## Section 3: Top-K Benchmark Issues

### Algorithm Issues
1. **block_bitonic_topk**: Actually a streaming per-thread heap (not bitonic). Output is each thread's local top-8, not globally sorted top-2048. Timing valid but output incorrect.
2. **radix_select**: Has serialized `atomicAdd` in inner loop — wrong algorithm. Should use local histograms.
3. **CUB at seq_len=5824**: 4x nonlinear jump due to algorithm tier change (single-tile → multi-tile).

### Missing: `cub::BlockRadixSort` (block-scope)
- Single-CTA, no global memory round-trips
- 5824 × 4B = 23KB scores fits in 48KB smem
- Expected to match bitonic at ~3 µs with correct output

---

## Section 4: Prioritized Next Benchmarks

### Priority 1: `kind::mxf8f6f4` vs `kind::f8f6f4` + post-scale (CRITICAL)
- Hardware-accelerated scaled FP8 vs software broadcast multiply
- Potentially the biggest single optimization gap
- SS mode only, scale factors in TMEM

### Priority 2: `mul.f32x2` / `max.f32x2` sweep benchmark (HIGH)
- Building block for ReLU, broadcast scale, post-FP8 dequant
- Pure register arithmetic — ~4 cycles latency, ~0.5 cy/element throughput expected

### Priority 3: TMEM readout + warp shuffle head reduction (HIGH)
- Full post-GEMM epilogue path: TMEM → regs → ReLU → scale → shuffle → SMEM
- 2-warp design measuring total latency

### Priority 4: Fix radix_select with local histograms (MEDIUM)
- Current results invalid due to serialized atomics

### Priority 5: `ex2.approx.ftz.f32` vs polynomial softmax (MEDIUM)
- For attention kernel, not indexer
- SFU throughput unknown on B200 — this resolves it

---

## Key Architectural Decisions for Indexer Kernel

1. **Use `kind::mxf8f6f4`** if faster than `kind::f8f6f4` + manual post-scale
2. **ReLU via `max.f32x2 dst, src, 0`**: Vectorized form throughout
3. **Head reduction**: 2-warp design with TMEM read → ReLU → scale → shfl_down → SMEM
4. **Top-K**: block radix sort for seq_len > 2048; fast path (return all) for seq_len ≤ 2048
5. **Pre-absorbed w**: If Q available in FP32, scale by w before FP8 quantization to eliminate step 4
