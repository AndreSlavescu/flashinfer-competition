# UTCMMA Explorer Findings
*Generated: 2026-03-19*

---

## 1. Missing UTCMMA Variants

From the `gemm.cuh` `static_assert` constraints, the complete valid shape space:

| Mode | M valid | N granularity | N range | C TMEM cols/tile | Status |
|---|---|---|---|---|---|
| WS-TS (`.ws`, TMEM A) | **32**, 64, 128 | {64, 128, 256} only | 64–256 | N/2 | M=32 MISSING |
| WS-SS (`.ws`, smem A) | **32**, 64, 128 | {64, 128, 256} only | 64–256 | N/2 | M=32 MISSING |
| Non-WS TS | 64, 128 | M=64: step **8**; M=128: step 16 | M=64: **8–256** | N (not N/2!) | N=8..56 MISSING |
| Non-WS SS | 64, 128 | M=64: step 8; M=128: step 16 | M=64: 8–256 | N | **ZERO coverage** |
| 2-CTA TS | 128, 256 | step 16 | 16–256 | N/2 | **ZERO coverage** |
| 2-CTA SS | 128, 256 | step 16 | 16–256 | N/2 | **ZERO coverage** |

### Critical Gap: M=32 WS-TS

`SM100_MMA_F16BF16_WS_TS_NOELECT` static_assert explicitly allows `M == 32`. Never benchmarked. With competition `h_q=16`, M=32 would give 50% utilization per token (vs 25% for M=64). The WS accumulator still uses N/2 TMEM columns regardless of M value.

### Critical Gap: Non-WS SS

Experiment 12 only covers Non-WS TS. The SS variant (smem A, smem B, non-WS) has zero coverage. Non-WS TS shows a 17% latency penalty vs WS-TS (64 cy vs 54.5 cy at N=128). Does the same gap exist for SS? The answer determines whether WS is worth the register/warpgroup overhead in the SV GEMM path.

### Important Gap: Non-WS Fine-Grained N

Non-WS TS and SS support N=8, 16, 24, 32, 40, 48, 56, 64... WS is locked to N ∈ {64, 128, 256}. This N granularity difference is architectural and unexplored.

---

## 2. Next Benchmark Priorities (Competition-Grounded)

### Priority 1 — N_acc=3,4,5 at K_DEPTH=32 in Exp 6 (30 min effort, CRITICAL)

The Exp 6 result shows N_acc=1,2 at K_DEPTH=32 both produce 928 ns — no improvement. But there's a structural reason this is expected to fail at N_acc≤3 and work at N_acc≥4:

- TMEM RAW latency ≈ 43 cycles (measured: 54.5 cy/tile minus ~11 cy issue interval)
- UTCMMA initiation interval ≈ 11 cy (from arxiv:2512.02189)
- N_acc needed to clear RAW: `N_acc × 11 ≥ 43` → **N_acc ≥ 4**

| N_acc | gap (cy) | vs 43-cy RAW | Predicted |
|---|---|---|---|
| 1 | 11 | < 43 → serialized | ✓ measured |
| 2 | 22 | < 43 → serialized | ✓ measured |
| 3 | 33 | < 43 → serialized | predicted |
| **4** | **44** | **≥ 43 → breaks dependency** | **~11 cy/tile** |

If N_acc=4 at K_DEPTH=32 drops from 928 ns to ~191 ns, GEMM becomes even smaller relative to TMA (~5,000–9,500 ns per block for 16 gather4 calls), further confirming TMA dominance. TMEM budget: 4 × 32 cols = 128 of 256 available — safe.

### Priority 2 — M=32 WS-TS Latency + K_DEPTH Sweep (1-2h, HIGH)

M=32 is the competition-exact tile for 2-token batches (2 × 16 heads = 32 rows = perfect fit). If M=32 latency ≈ M=64 latency (same ~54.5 cy/tile), it's strictly better for 2-token scenarios since it reduces smem B footprint by 2× and costs nothing extra.

### Priority 3 — TMA+UTCMMA Concurrent Execution (5-8h, HIGH)

The timing model assumes TMA and UTCMMA pipeline independently. If they share an execution resource, double-buffering doesn't work. This is the biggest unmeasured system-level unknown.

### Priority 4 — Non-WS SS Latency (1-2h, MEDIUM)

Measures whether the 17% WS latency advantage seen for TS also holds for SS. The SV GEMM (SS path) latency is 80 cy/tile (WS-SS); if Non-WS SS is also 80 cy or similar, WS adds no benefit but has higher register cost.

### Priority 5 — UTCCP 64dp256b Variant (1h, MEDIUM)

The benchmark currently only measures the 128dp256b copy (128 rows per call). A competition kernel with M=64 might prefer 64dp256b (64 rows per call). The startup cost profile may differ.

---

## 3. Swizzle Layout Variants

**Conclusion from Exp 5: swizzle has zero impact on UTCMMA latency.** All configurations (INTER, SW32, SW64, SW128) produced exactly 218.4 cycles for 4 K-tiles at M=64 N=128 (sub-0.001% variance). UTCMMA reads smem via opaque 64-bit descriptors that bypass all bank conflict concerns. Swizzle exploration is complete — no further swizzle experiments are needed.

Valid swizzle selection rules:
- **SW128**: K_TOTAL % 64 == 0 (safe for K_DEPTH ≥ 4)
- **SW64**: K_TOTAL % 32 == 0 (safe for K_DEPTH ≥ 2)
- **SW32**: always valid for any K_TOTAL that's a multiple of 16
- **INTER**: always valid, no divisibility constraint

For competition tiles (K_DEPTH=32 → K_TOTAL=512), SW128 is valid and recommended to match the TMA-filled smem layout.

---

## 4. Pipeline Depth and Throughput Interaction

The multi-accumulator rotation in Exp 6 reveals a critical structural insight: **N_acc rotation does not help unless N_acc × (issue_interval) ≥ TMEM_RAW_latency**.

From ver3 data:
- N_acc=1,2,3,4 at K_DEPTH=4 (N=64): all identical at 54.6 cy/tile. Here 4×11=44 cy ≈ 43 cy RAW — right at the boundary, explains why N_acc=4 gives no benefit for K_DEPTH=4.
- N_acc=1,2 at K_DEPTH=32 (N=64): both 53.7 cy/tile. N_acc=2: gap = 2×11=22 cy < 43 cy RAW → still serialized.
- **N_acc=4 at K_DEPTH=32 is the untested critical case.** Each write to C[i%4] is spaced 4 tiles × 11 cy = 44 cy apart, which should just clear the 43-cycle TMEM RAW. This experiment would answer whether the competition QK GEMM can run at 11 cy/tile (throughput-bound) or is fundamentally serialized at 54 cy/tile (RAW-bound).

---

## Summary: Recommended Next Steps (Ordered)

1. **Exp 6 extension**: Test N_acc=3,4,5 at K_DEPTH=32 — confirms/refutes the N_acc≥4 threshold hypothesis. Fastest experiment, highest information value.
2. **M=32 WS-TS benchmark**: Add M=32 tile size to existing latency sweep.
3. **TMA+UTCMMA concurrency test**: Issue interleaved TMA and UTCMMA ops, measure if they overlap.
4. **Non-WS SS latency**: Add SS variant to Exp 12.
5. **UTCCP 64dp256b variant**: Measure copy startup overhead for smaller tile.
