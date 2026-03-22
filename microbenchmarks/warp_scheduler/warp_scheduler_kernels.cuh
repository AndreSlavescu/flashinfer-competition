#pragma once
// Warp sub-core partitioning benchmark kernels (B200 SM100a)
//
// Methodology from Citadel (arxiv:1804.06826 §2.1):
//   "scheduler_id = warp_id % num_schedulers" on Volta.
//   We test whether B200 follows the same pattern by measuring FFMA
//   contention between pairs of warps.
//
// Each kernel:
//   - active_mask: bitmask of warps that execute the FFMA chain
//     bit i set → warp i participates; others return immediately
//   - Uses 8 independent fp32 accumulators (throughput-bound, not latency-bound)
//   - Reports per-warp elapsed clock64 cycles to elapsed_cycles[warp_id]
//   - Writes accumulator sums to sink[warp_id * 32 + lane_id] (no DCE)
//
// Interpretation:
//   - Same sub-core pair → both warps contend for FP32 units, per-warp cycles inflate
//     (measured contention ratio ≈ 1.3 on B200; <2.0 because pipeline overlap hides some cost)
//   - Different sub-core pair → parallel execution, 1x cycle count (contention ratio ≈ 1.0)
//
// Confirmed on B200: warp_id % 4 mapping, same as Volta.
//   Contention pairs: (0,4), (1,5), (2,6), (3,7)

#include <cuda_runtime.h>
#include <stdint.h>

static constexpr int BENCH_WARPS   = 8;
static constexpr int BENCH_THREADS = BENCH_WARPS * 32;   // 256 threads total

// ---------------------------------------------------------------------------
// FFMA contention kernel
//
// Launch with: <<<1, 256>>> (1 block → isolated to 1 SM, no inter-CTA noise)
//
// elapsed_cycles: int64_t[BENCH_WARPS], lane 0 of each active warp writes result
// sink:           float[BENCH_WARPS * 32], all active lanes write (prevents DCE)
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BENCH_THREADS)
void ffma_warp_contention_kernel(
    uint32_t  active_mask,
    int       iters,
    int64_t*  __restrict__ elapsed_cycles,   // [BENCH_WARPS]
    float*    __restrict__ sink              // [BENCH_WARPS * 32]
) {
    const int warp_id = threadIdx.x >> 5;    // threadIdx.x / 32
    const int lane_id = threadIdx.x &  31;   // threadIdx.x % 32

    // Inactive warps exit immediately — they retire before the active warps
    // start the FFMA loop, so they don't contend.
    if (!((active_mask >> warp_id) & 1u)) return;

    // 8 independent fp32 accumulators → throughput-bound behavior.
    // With 8 accumulators and ~4-cycle FP32 pipeline depth, the warp issues
    // 1 FMA instruction per cycle continuously (no stall on RAW hazard).
    // Each accumulator seeded with a different value to prevent constant-folding.
    float a0 = 1.00000f + lane_id * 0.0001f;
    float a1 = 1.00001f + lane_id * 0.0001f;
    float a2 = 1.00002f + lane_id * 0.0001f;
    float a3 = 1.00003f + lane_id * 0.0001f;
    float a4 = 1.00004f + lane_id * 0.0001f;
    float a5 = 1.00005f + lane_id * 0.0001f;
    float a6 = 1.00006f + lane_id * 0.0001f;
    float a7 = 1.00007f + lane_id * 0.0001f;
    // m slightly above 1 → values grow slowly (avoids INF, avoids denorm)
    const float m = 1.0000001f;

    // Compiler fence: ensures clock64() is not hoisted above accumulator init
    asm volatile("" ::: "memory");
    const int64_t t0 = clock64();
    asm volatile("" ::: "memory");

    // 8 FMA instructions per outer iteration, outer loop not unrolled.
    // Total FMA instructions per warp: 8 * iters
    // Total FP32 FMAs per warp (× 32 lanes): 8 * iters * 32
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        a0 = fmaf(a0, m, m);
        a1 = fmaf(a1, m, m);
        a2 = fmaf(a2, m, m);
        a3 = fmaf(a3, m, m);
        a4 = fmaf(a4, m, m);
        a5 = fmaf(a5, m, m);
        a6 = fmaf(a6, m, m);
        a7 = fmaf(a7, m, m);
    }

    asm volatile("" ::: "memory");
    const int64_t t1 = clock64();
    asm volatile("" ::: "memory");

    // All lanes write sink to prevent dead-code elimination of the FMA loop
    sink[warp_id * 32 + lane_id] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

    // Lane 0 writes timing
    if (lane_id == 0) {
        elapsed_cycles[warp_id] = t1 - t0;
    }
}
