# B200 Device Query + Warp Scheduler Analysis

Run date: 2026-03-21
GPU: NVIDIA B200 (SM 10.0), CUDA 13.0, Modal cloud

## 1. Device Query — CLAUDE.md Spec Verification

We ran `cudaGetDeviceProperties` + Driver API on a real B200 to verify claims in CLAUDE.md (which cited arxiv:2512.02189 for B200 specs and arxiv:2507.10789 for GB203/RTX 5080 specs, noting the latter might not transfer).

| Property | CLAUDE.md Claim | **Measured on B200** | Source | Status |
|---|---|---|---|---|
| SM count | 148 | **148** | arxiv:2512.02189 | CONFIRMED |
| Registers / SM | 65,536 (GB203 paper) | **65,536** | arxiv:2507.10789 | CONFIRMED (transfers from GB203) |
| Register file / SM | 256 KB | **256.0 KB** | derived | CONFIRMED |
| Max threads / SM | 2,048 | **2,048** | arxiv:2512.02189 | CONFIRMED |
| Max warps / SM | 64 | **64** | derived | CONFIRMED |
| Warp size | 32 | **32** | standard | CONFIRMED |
| Shared mem / SM | 228 KB | **228.0 KB** (233,472 B) | cudaFuncSetAttribute | CONFIRMED |
| Shared mem / block (default) | 48 KB | **48.0 KB** (49,152 B) | — | — |
| **L2 cache** | **~64 MB** | **126.5 MB** (132,644,864 B) | empirical + arxiv:2512.02189 | **WRONG — 2x larger!** |
| Total global mem | 192 GB | **178.4 GB** | product page | Expected (OS/driver reserve ~14 GB) |
| HBM3e peak BW | 8 TB/s | **7.67 TB/s** (formula) | product page | Consistent (formula is theoretical max) |
| SM clock | ~2.1 GHz (estimated) | **1.965 GHz** | first measurement | NEW DATA |
| Memory clock | — | **3,996 MHz** | first measurement | NEW DATA |
| Memory bus width | — | **7,680 bits** | first measurement | NEW DATA |
| Compute capability | sm_100a | **10.0** | — | CONFIRMED |

### Key Finding: L2 Cache is 126.5 MB, not 64 MB

CLAUDE.md says "L2 cache ~64 MB (4 partitions, 2x Hopper)" citing arxiv:2512.02189. The actual value from `cudaDeviceProp.l2CacheSize` is **132,644,864 bytes = 126.5 MB**. This is ~4x Hopper (which has ~50 MB L2) or roughly 2x what was claimed.

For the **decode workload** specifically (num_tokens=1-2), the sparse KV data accessed per launch is ~2048 indices × 576 bytes/token × 2 tokens ≈ **4.7 MB** — well within 126.5 MB L2. Repeated decode iterations against the same KV page pool can stay L2-warm after the first pass.

For **prefill workloads** (larger num_tokens), the accessed KV set grows proportionally and easily exceeds 126.5 MB. Even at num_tokens=100, 100 × 2048 × 576 bytes ≈ 118 MB approaches the L2 boundary; at num_tokens=256 or higher it overflows entirely. The competition evaluation includes prefill scenarios, so the L2-fit property cannot be assumed in general — it's a decode-only advantage.

### SM Clock: 1.965 GHz

Lower than the ~2.1 GHz often assumed. This is the `cudaDevAttrClockRate` value. The actual boost clock under load may differ (but our FFMA benchmark measured 2,300,096 cycles for 100K iterations, consistent with this clock).

## 2. Warp Scheduler — Sub-core Partitioning

### Methodology

Following Citadel (arxiv:1804.06826 §2.1): launch 1 CTA with 8 warps (256 threads), activate 2 warps at a time, measure per-warp FFMA elapsed cycles via `clock64()`. Warps sharing a sub-core contend for FP32 units → inflated cycle count.

### Result: `scheduler_id = warp_id % 4` CONFIRMED on B200

**Single-warp baseline:** All 8 warps: exactly **2,300,096 cycles** (0% spread across 20 trials).

**Two-warp sweep (28 pairs):**

| Contention? | Pairs | Per-warp cycles | Ratio vs baseline |
|---|---|---|---|
| **YES (same sub-core)** | **(0,4), (1,5), (2,6), (3,7)** | **3,000,096** | **1.304** |
| No (diff sub-core) | All other 24 pairs | 2,300,096 | 1.000 |

The 4 contention pairs are exactly `{(w, w+4)}` for w=0..3, confirming `warp_id % 4` mapping:

| Sub-core | Warps (from 0-7) |
|---|---|
| 0 | 0, 4 |
| 1 | 1, 5 |
| 2 | 2, 6 |
| 3 | 3, 7 |

Generalizing: **scheduler_id = warp_id % 4** (same as Volta, Turing, Ampere, Hopper, and now Blackwell).

### Why contention ratio is 1.3x, not 2.0x

Prediction was 2.0x (fully serialized). Measured 1.3x because:

1. Each sub-core has 32 FP32 CUDA cores. 1 warp (32 threads) saturates them.
2. With 2 warps, the scheduler interleaves instructions. While warp A's FMA is in the FP32 pipeline (~4-5 cycle depth), warp B's FMA can issue to a different pipeline stage.
3. Net effect: partial overlap. Not 2x serialization, not 1x parallel — 1.3x is the actual contention penalty.

### Scaling Analysis

| Config | Active warps | Sub-cores used | Scaling vs 1 warp | Efficiency |
|---|---|---|---|---|
| seq_1 (warp 0) | 1 | 1 | 1.00x | 100% |
| seq_2 (warps 0,1) | 2 | 2 | 2.00x | 100% |
| seq_4 (warps 0-3) | 4 | 4 | **4.00x** | **100%** |
| seq_8 (warps 0-7) | 8 | 4 (2 each) | 6.13x | 76.7% |
| s4_0_4 (warps 0,4) | 2 | 1 | 1.53x | 76.7% |
| mix_0145 (warps 0,1,4,5) | 4 | 2 (2 each) | 3.07x | 76.7% |

Key observations:
- **Perfect linear scaling up to 4 warps** when each warp is on a different sub-core (warps 0-3).
- **1.53x per sub-core with 2 warps** — consistent across all same-sub-core pairs.
- **6.13x with all 8 warps** = 4 sub-cores × 1.53x = 6.13x. Perfectly consistent.

### Cycle Count Interpretation

Per outer loop iteration (8 FMAs + loop control):
- Single warp: 2,300,096 / 100,000 = **23.0 cycles/iteration**
- Same sub-core pair: 3,000,096 / 100,000 = **30.0 cycles/iteration**
- 8 FMA instructions + ~2-3 loop instructions ≈ 10-11 instructions
- 23.0 / ~10.5 ≈ **2.2 cycles per instruction issue** (single warp on one sub-core)

This suggests each sub-core's warp scheduler issues 1 instruction approximately every 2 cycles to the FP32 units — consistent with B200's 32 FP32 cores per sub-core handling 32-wide warp operations.

### Implications for Competition Kernel

1. **Warp-specialized kernels have algorithmically fixed warp roles.** In the FlashMLA sm100 decode kernel (and in the competition kernel), warp assignments are determined by the tcgen05.mma.ws architecture: a TMA-elected warp issues gather4, an MMA-elected warp issues tcgen05.mma, etc. You cannot freely reassign which warp IDs do softmax — that is decided by the warpgroup structure and `elect_sync()` calls. The sub-core mapping is informational, not a design parameter.

2. **A 4-warp warpgroup (warps 0-3) naturally spans all 4 sub-cores.** Since `scheduler_id = warp_id % 4`, warps 0, 1, 2, 3 each land on a different sub-core. Any warpgroup organized as warps 0-3 (a standard 128-thread warpgroup) executes with full FP32 parallelism — no intra-warpgroup FP32 contention. This is the expected case for the compute warpgroup.

3. **TMA-elected warp (warp 0) and MMA-elected warp (warp 1) are on different sub-cores** (sub-core 0 and 1 respectively). They do not share FP32 units. Their contention, if any, is on other resources (instruction cache, L1, barriers) — not FP32.

4. **2 warps per sub-core gives 53% more throughput than 1**, not 2x. If the competition kernel needs two warpgroups (e.g., TMA warpgroup on warps 0-3 and compute warpgroup on warps 4-7), each sub-core handles one warp from each group. Total FP32 throughput = 4 sub-cores × 1.53x = 6.13x a single warp, not 8x. This is the steady-state limit for an 8-warp CTA doing sustained FP32 work.

## Files

- `microbenchmarks/device_query/results/device-query-b200-ver1.txt` — raw deviceQuery output
- `microbenchmarks/warp_scheduler/results/warp-scheduler-ver1.csv` — raw CSV data
- `microbenchmarks/device_query/device_query.cu` — standalone device query program
- `microbenchmarks/warp_scheduler/warp_scheduler_kernels.cuh` — FFMA contention kernel
- `microbenchmarks/warp_scheduler/main.cu` — experiment harness
