---
name: Local SM100 Reference Files and arxiv Access Status
description: arxiv paper fetch results (confirmed working URLs) and local sm100 reference ground-truth patterns
type: reference
---

## arxiv Paper Access (confirmed 2026-03-21)

WebFetch succeeds on these URLs and returns readable paper body content (not just abstract):

| Paper | URL | Fetch result |
|---|---|---|
| arxiv:2512.02189 — Microbenchmarking Blackwell (Dec 2025) | https://arxiv.org/html/2512.02189v1 | SUCCESS — returns specific measured numbers from paper body |
| arxiv:2507.10789 — Dissecting Blackwell (Jul 2025) | https://arxiv.org/html/2507.10789v2 | SUCCESS — returns GB203 (RTX 5080), NOT B200 datacenter |
| arxiv:2501.12084 — Dissecting Hopper | https://arxiv.org/html/2501.12084 | SUCCESS — returns H800 PCIe numbers |

**IMPORTANT CAVEAT**: arxiv:2507.10789 covers GB203 (RTX 5080 consumer GPU), not B200 datacenter. Its L1/L2 latency numbers are for a different chip. Use only for methodology, not B200-specific values.

**What arxiv:2512.02189 actually says (confirmed by fetch):**
- tcgen05.mma latency: "11.0-11.4 cycles" (Section VI-A, Table V) — this is THROUGHPUT (initiation interval), confirmed as the paper's figure
- HBM sustained BW: "7.48 TB/s" (Section VII-C2, Table XIII) on STREAM Triad >64GB
- TMEM read: "16 TB/s per SM", write: "8 TB/s per SM" (Section V-A)
- Global memory miss latency: "420 clock cycles" (Section V-A) — this is for ld.global, NOT TMA
- FP16 tensor core: "1929.2 TFLOPS" (96.5% peak), FP8: "3851.4 TFLOPS"
- BF16 is not separately listed; 1,926 TFLOPS figure in CLAUDE.md is consistent

## Local Reference: references/learn-cuda/02e_matmul_sm100/

Ground truth for correct sm100 patterns, benchmarked on real B200 Modal hardware.

### Confirmed correct tcgen05 patterns (cross-reference against this in reviews)

**TMEM allocation**: warp 0 (elect_sync) initializes mbarriers; warp 1 runs `tcgen05.alloc`; `__syncthreads()` after both; all threads then read the TMEM address.

**tcgen05.fence placement**:
- `tcgen05.fence::after_thread_sync` goes AFTER `mbarrier_wait` for TMA data, BEFORE issuing tcgen05.mma
- `tcgen05.fence::after_thread_sync` also placed at epilogue BEFORE `tcgen05.ld`
- The `::before_thread_sync` variant does NOT appear in this reference — only `::after_thread_sync`

**tcgen05.ld — warp-collective, ALL warps**:
- In epilogue, all 4 warps call `tcgen05.ld.sync.aligned.32x32b.x8.b32` (no warp gate)
- Each warp accesses its 32-lane slice: `taddr + ((warp_id * 32) << 16) + column_offset`
- Immediately followed by `tcgen05.wait::ld.sync.aligned` (also warp-collective)
- This confirms: tcgen05.ld/.wait MUST NOT be inside `if (warp_id == 0)` guards

**mbarrier.arrive.expect_tx**: called by single thread (elect_sync), after issuing TMA, sets expected bytes AND decrements arrival count in one instruction.

**tcgen05.commit**: called by MMA warp only (elect_sync), signals accumulator readiness.

**tcgen05.dealloc**: called by warp 0 only, after `__syncthreads()` confirms all warps done with TMEM.

### Performance numbers (M=N=K=4096, BF16, B200)
- No swizzle: 256 TFLOPS (v1a/b)
- 128B TMA swizzle: 699-721 TFLOPS (v2) — 2.7× improvement from swizzle alone
- 2-stage pipelining: 914 TFLOPS (v3) — 1.3× on top of swizzle
- Warp specialization: 1054 TFLOPS (v4)
- 2-SM MMA (cta_group::2): 1170 TFLOPS (v5)
- Persistent + static scheduling: 1273-1326 TFLOPS (v6/v7)
- cuBLAS: 1359 TFLOPS

### Competition reference implementations
- `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py` — exact expected behavior
- `references/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.py` — indexer reference

**How to apply**: During any sm100 benchmark review, read relevant reference file(s) first. If the benchmark uses tcgen05.ld, tcgen05.alloc, mbarrier, or tcgen05.fence differently from this reference, trace the difference to a PTX ISA citation before flagging as correct or incorrect.
