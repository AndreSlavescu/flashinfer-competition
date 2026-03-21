# UTCMMA Ver5 Benchmark Results (2026-03-21)

Results: `microbenchmarks/utcmma/results/utcmma-ver5.csv` (93 rows)
GPU: NVIDIA B200 (sm100a), CUDA 13.0, Modal cloud
Clock: 1.844 GHz (measured: 173.0 cy / 93.8 ns)

## Critical Change: Proper MMA Completion Synchronization

Ver3/ver4 benchmarks measured **issue throughput** (how fast the thread pushes async tcgen05.mma instructions into the pipeline). Ver5 adds proper completion synchronization to ALL MMA kernels:

```cpp
// After GEMM chain:
ku::umma_arrive_noelect(*bar);   // tcgen05.commit — signals barrier when MMA finishes
bar->wait(phase);                // mbarrier.wait — blocks until MMA actually done
ku::tcgen05_after_thread_sync(); // fence — required before reading TMEM results
phase ^= 1;
```

This measures **true completion latency** — when results are available in TMEM. The pattern follows `references/learn-cuda/02e_matmul_sm100/matmul_v4.cu` (correct reference implementation).

## MMA Single-Tile Latency (K=1) — Ver5 vs Ver3/4

| Config | Ver3/4 cy | Ver5 cy | Ver5 ns | Ratio |
|--------|-----------|---------|---------|-------|
| WS-TS M64N64 | 54.46 | **173.0** | 93.8 | 3.18x |
| WS-TS M64N128 | 54.46 | **173.0** | 93.8 | 3.18x |
| WS-TS M64N256 | 76.0 | **238.5** | 129.3 | 3.14x |
| WS-TS M128N128 | 114.5 | **238.5** | 129.3 | 2.08x |
| WS-TS M128N256 | N/A | **370.5** | 200.9 | — |
| WS-SS M64N128 | 54.46 | **175.3** | 95.0 | 3.22x |
| WS-SS M64N256 | 262.8 | **262.8** | 142.5 | 1.00x |
| Non-WS TS M64N64 | 54.46 | **173.0** | 93.8 | 3.18x |
| Non-WS TS M64N128 | 64.0 | **238.5** | 129.3 | 3.73x |
| Non-WS SS M64N64 | 52.46 | **175.3** | 95.0 | 3.34x |
| Non-WS SS M64N128 | 64.01 | **262.8** | 142.5 | 4.11x |
| Non-WS SS M64N256 | 128.01 | **372.8** | 202.1 | 2.91x |

**Key observations**:
- Single K-tile completion latency is ~173 cy (~94 ns) for M64 N<=128, not ~54 cy
- The ~118 cy gap (173 - 54.46) matches commit+wait+fence overhead: ~30 cy (commit) + ~88.5 cy (mbarrier) = ~118.5 cy
- N=64 and N=128 are identical at 173 cy for WS-TS (N<=128 fits in one MMA pass)
- Non-WS TS shows a step at N=128 (173 -> 238.5 cy), unlike WS-TS which stays at 173 cy through N=128
- SS variants are ~2.3 cy slower than TS (175.3 vs 173.0)

## K-Depth Amortization (WS-TS M64N128)

| K_DEPTH | Ver3/4 cy/tile | Ver5 cy/tile | Overhead/tile | Total cy | Total ns |
|---------|----------------|--------------|---------------|----------|----------|
| 1 | 54.46 | 173.1 | 118.7 | 173 | 93.8 |
| 2 | 53.2 | 113.2 | 59.4 | 226 | 122.7 |
| 4 | 54.05 | 83.4 | 29.4 | 334 | 180.9 |
| 8 | 54.05 | 68.9 | 14.9 | 551 | 299.0 |
| 16 | 54.05 | 63.3 | 9.3 | 1013 | 549.5 |
| 24 | 53.72 | 62.0 | 8.3 | 1487 | 806.4 |
| 32 | 53.72 | 60.2 | 6.2 | 1925 | 1043.8 |

**Amortization model**: `total_cycles = 56.4 * K_DEPTH + 118.5`
- **56.4 cy/K-tile**: true per-tile MMA execution cost (slightly above 54 cy issue rate)
- **118.5 cy fixed**: commit + mbarrier.wait + fence overhead (paid once per GEMM chain)
- At K=32, overhead is only 6.2 cy/tile (3.6 ns amortized)
- The ver3/4 "54 cy/K-tile" was the issue rate, which is the pipelined throughput; the 56.4 cy/tile is the true per-tile execution seen in the completion model

## Multi-Accumulator Throughput (Exp 6+14) — Still NULL

| N_ACC | K_DEPTH | cy/K-tile (ver5) |
|-------|---------|------------------|
| 1 | 4 | 83.41 |
| 2 | 4 | 83.41 |
| 3 | 4 | 83.41 |
| 4 | 4 | 83.41 |
| 1 | 32 | 60.16 |
| 2 | 32 | 60.16 |
| 3 | 32 | 60.16 |
| 4 | 32 | 60.16 |
| 5 | 32 | 60.16 |

Confirmed with proper sync: N_ACC rotation provides zero throughput benefit. The 60.16 cy floor at K=32 (vs 53.72 in ver4) is the properly-measured per-tile completion throughput.

## Non-WS TS vs SS Comparison (ver5)

| Config | TS cy | SS cy | Delta |
|--------|-------|-------|-------|
| M64 N64 K1 | 173.0 | 175.3 | SS +1.3% slower |
| M64 N128 K1 | 238.5 | 262.8 | SS +10.2% slower |
| M64 N256 K1 | — | 372.8 | — |

With proper sync, SS is **slower** than TS (opposite of ver4 finding where SS appeared 4% faster). The ver4 result was an artifact of measuring issue rate — SS had slightly lower issue backpressure but takes longer to complete.

## Fine-Grained N Sweep (K=1) — N-Independence Confirmed

| N | Non-WS TS cy | Non-WS SS cy |
|---|-------------|-------------|
| 8 | 173.0 | 175.3 |
| 16 | 173.0 | 174.8 |
| 24 | 173.0 | 175.3 |
| 32 | 173.0 | 175.3 |
| 40 | 173.0 | 175.3 |
| 48 | 173.0 | 175.3 |
| 56 | 173.0 | 175.3 |
| 64 | 173.0 | 175.3 |

Completion latency is also N-independent within N<=64, same as issue rate.

## Swizzle (Exp 5) — Still Zero Impact

| Swizzle | cy/K-tile |
|---------|-----------|
| INTER | 83.41 |
| SW32 | 83.41 |
| SW64 | 83.41 |
| SW128 | 83.41 |

Identical across all swizzle modes (M64 N128 K=4).

## Non-MMA Experiments — Unchanged

All non-MMA experiments (TMEM ld/st, fences, mbarrier, named barriers, UTCCP) produce identical results to ver3/ver4 since they were already properly synchronized.

| Experiment | cy/iter | ns/iter |
|------------|---------|---------|
| TMEM ld (N=1..64) | 1.79 | 0.9 |
| TMEM st (N=64) | 49.80 | 25.4 |
| tcgen05 fence pair | 2.83 | 1.4 |
| O-rescale 4 chunks | 622.1 | 316.7 |
| mbarrier roundtrip | 88.51 | 45.1 |
| MMA commit+wait (K=1) | 173.03 | 93.8 |
| Named barrier (128 thr) | 20.67 | 10.5 |
| UTCCP 256b 16-copy | 74.60/copy | 38.0 |
| UTCCP 128b 32-copy | 51.32/copy | 26.1 |

## Updated Competition Timing Model

| Component | Ver3/4 | Ver5 (corrected) | Change |
|-----------|--------|-------------------|--------|
| QK ckv (M64 N64, K=32) | 928 ns | **1,044 ns** | +12.5% |
| QK kpe (M64 N64, K=4) | 118 ns | **181 ns** | +53% |
| SV tile 1 (M64 N256 SS, K=4) | 173 ns | **292 ns** | +69% |
| SV tile 2 (M64 N256 SS, K=4) | 173 ns | **292 ns** | +69% |
| **Total GEMM** | **1,392 ns** | **1,809 ns** | **+30%** |
| TMA per block (unchanged) | 3,690 ns | 3,690 ns | 0% |
| **Per-block total** | **5,082 ns** | **5,499 ns** | **+8%** |
| **TMA/GEMM ratio** | 2.65x | **2.04x** | still TMA-bound |

The kernel remains TMA-bound. Inter-block pipelining steady-state is still **3,690 ns/block** (unchanged since TMA dominates).

**Important nuance**: In a real kernel with overlapped TMA/GEMM, the commit+wait overhead runs concurrently with TMA for the next block. The GEMM "wall time" that matters for pipelining is the issue-rate portion (~1,392 ns of actual MMA execution) plus the non-overlappable commit+wait at the chain boundary. Since the kernel is TMA-bound (3,690 ns >> 1,809 ns), the commit+wait overhead is fully hidden by TMA in a pipelined design.

## Key Takeaways

1. **True MMA completion latency is ~3x higher than issue rate** at K=1 (173 vs 54 cy). The ~118 cy gap is commit+wait+fence overhead.
2. **At K=32 (competition scale), overhead amortizes to +12%** (60.2 vs 53.7 cy/tile). The per-tile execution cost is ~56 cy, close to the 54 cy issue rate.
3. **N_ACC multi-accumulator is definitively NULL** — confirmed with proper sync.
4. **SS is NOT faster than TS** — ver4's "4% SS advantage" was an issue-rate artifact. With proper sync, SS is ~1-10% slower.
5. **Non-WS has a real penalty at N>=128** (238 vs 173 cy) — WS-TS remains preferred for N>=128.
6. **N-independence confirmed** — completion latency is constant for N=8..64.
7. **Competition kernel is still TMA-bound** — GEMM correction (+30%) doesn't change the design since TMA dominates 2:1.
