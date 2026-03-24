#pragma once
// Bulk Copy S2G (Shared → Global) microbenchmark kernels
// Measures cp.async.bulk.global.shared::cta and cp.reduce.async.bulk.add.f32
// Target: B200 (sm100a)

#include "benchmark_common.cuh"
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// BenchResult (same layout as utcmma benchmarks)
// ---------------------------------------------------------------------------
struct BulkResult {
    uint64_t total_cycles;
    int64_t  gt_start_ns;
    int64_t  gt_end_ns;
};

// ---------------------------------------------------------------------------
// PTX inline wrappers for bulk copy S2G
// ---------------------------------------------------------------------------

// Plain cp.async.bulk shared→global copy (PTX ISA §9.7.12.1)
__device__ __forceinline__
void bulk_copy_s2g(void* dst_global, const void* src_smem, int32_t bytes) {
    // Get shared memory address for PTX. The extern __shared__ pointer is in generic
    // address space; cvta.to.shared converts it to the shared address space.
    uint32_t smem_addr;
    asm("{ .reg .u64 s; cvta.to.shared.u64 s, %1; cvt.u32.u64 %0, s; }"
        : "=r"(smem_addr) : "l"(src_smem));
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :: "l"(dst_global), "r"(smem_addr), "r"(bytes) : "memory");
}

// cp.reduce.async.bulk shared→global with f32 add (PTX ISA §9.7.12.4)
// Already available as kerutils::tma_bulk_reduce_add, but we inline here
// to avoid pulling in the full kerutils dependency tree
__device__ __forceinline__
void bulk_reduce_add_s2g(void* dst_global, const void* src_smem, int32_t bytes) {
    // Get shared memory address for PTX. The extern __shared__ pointer is in generic
    // address space; cvta.to.shared converts it to the shared address space.
    uint32_t smem_addr;
    asm("{ .reg .u64 s; cvta.to.shared.u64 s, %1; cvt.u32.u64 %0, s; }"
        : "=r"(smem_addr) : "l"(src_smem));
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
        :: "l"(dst_global), "r"(smem_addr), "r"(bytes) : "memory");
}

// Commit bulk group
__device__ __forceinline__
void bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

// Wait for bulk group (N=0 means all)
template<int N>
__device__ __forceinline__
void bulk_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// Experiment 1: Latency sweep — serialized bulk copy, measure per-copy cost
// ---------------------------------------------------------------------------
// MODE: 0 = plain copy, 1 = reduce_add
template<int MODE, int COPY_BYTES>
__global__ __launch_bounds__(128, 1)
void kernel_bulk_latency(BulkResult* result, int iters,
                         float* __restrict__ g_dst) {
    // Dynamic shared memory for source data
    extern __shared__ float smem_f32[];

    // Thread 0 initializes smem with some data
    for (int i = threadIdx.x; i < COPY_BYTES / 4; i += blockDim.x) {
        smem_f32[i] = 1.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Warmup: 100 serialized copies
        for (int i = 0; i < 100; i++) {
            if constexpr (MODE == 0) {
                bulk_copy_s2g(g_dst, smem_f32, COPY_BYTES);
            } else {
                bulk_reduce_add_s2g(g_dst, smem_f32, COPY_BYTES);
            }
            bulk_commit_group();
            bulk_wait_group<0>();
        }

        // Timed: serialized copies
        uint64_t start = clock64();
        int64_t gt_start = globaltimer();

        for (int i = 0; i < iters; i++) {
            if constexpr (MODE == 0) {
                bulk_copy_s2g(g_dst + (i % 16) * (COPY_BYTES / 4), smem_f32, COPY_BYTES);
            } else {
                bulk_reduce_add_s2g(g_dst + (i % 16) * (COPY_BYTES / 4), smem_f32, COPY_BYTES);
            }
            bulk_commit_group();
            bulk_wait_group<0>();
        }

        int64_t gt_end = globaltimer();
        uint64_t end = clock64();

        result->total_cycles = end - start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }
}

// ---------------------------------------------------------------------------
// Experiment 2: Throughput pipeline — issue N_PIPE copies before waiting
// ---------------------------------------------------------------------------
template<int MODE, int COPY_BYTES, int N_PIPE>
__global__ __launch_bounds__(128, 1)
void kernel_bulk_throughput(BulkResult* result, int iters,
                            float* __restrict__ g_dst) {
    extern __shared__ float smem_f32[];

    // Init smem
    for (int i = threadIdx.x; i < COPY_BYTES / 4; i += blockDim.x) {
        smem_f32[i] = 1.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Each pipe writes to a different global destination offset
        // Total destination space: N_PIPE * COPY_BYTES
        constexpr int STRIDE = COPY_BYTES / 4;  // in float elements

        // Warmup
        for (int w = 0; w < 50; w++) {
            for (int p = 0; p < N_PIPE; p++) {
                if constexpr (MODE == 0) {
                    bulk_copy_s2g(g_dst + p * STRIDE, smem_f32, COPY_BYTES);
                } else {
                    bulk_reduce_add_s2g(g_dst + p * STRIDE, smem_f32, COPY_BYTES);
                }
                bulk_commit_group();
            }
            bulk_wait_group<0>();
        }

        // Timed
        uint64_t start = clock64();
        int64_t gt_start = globaltimer();

        for (int i = 0; i < iters; i++) {
            // Issue N_PIPE copies
            for (int p = 0; p < N_PIPE; p++) {
                if constexpr (MODE == 0) {
                    bulk_copy_s2g(g_dst + p * STRIDE, smem_f32, COPY_BYTES);
                } else {
                    bulk_reduce_add_s2g(g_dst + p * STRIDE, smem_f32, COPY_BYTES);
                }
                bulk_commit_group();
            }
            // Wait for all
            bulk_wait_group<0>();
        }

        int64_t gt_end = globaltimer();
        uint64_t end = clock64();

        result->total_cycles = end - start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }
}

// ---------------------------------------------------------------------------
// Experiment 3: Overlap test — issue bulk copy, then do compute, then wait
// Compare (copy+compute) walltime to (copy_only + compute_only) to detect async
// ---------------------------------------------------------------------------
template<int MODE, int COPY_BYTES, int COMPUTE_ITERS>
__global__ __launch_bounds__(128, 1)
void kernel_bulk_overlap(BulkResult* result, int iters,
                         float* __restrict__ g_dst) {
    extern __shared__ float smem_f32[];

    // Init smem
    for (int i = threadIdx.x; i < COPY_BYTES / 4; i += blockDim.x) {
        smem_f32[i] = 1.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Warmup
        float acc = 1.0f;
        for (int w = 0; w < 50; w++) {
            if constexpr (MODE == 0) {
                bulk_copy_s2g(g_dst, smem_f32, COPY_BYTES);
            } else {
                bulk_reduce_add_s2g(g_dst, smem_f32, COPY_BYTES);
            }
            bulk_commit_group();
            // Compute while copy is in flight
            #pragma unroll 1
            for (int c = 0; c < COMPUTE_ITERS; c++) {
                acc = acc * 1.00001f + 0.00001f;
            }
            bulk_wait_group<0>();
        }

        // Timed
        uint64_t start = clock64();
        int64_t gt_start = globaltimer();

        for (int i = 0; i < iters; i++) {
            if constexpr (MODE == 0) {
                bulk_copy_s2g(g_dst + (i % 16) * (COPY_BYTES / 4), smem_f32, COPY_BYTES);
            } else {
                bulk_reduce_add_s2g(g_dst + (i % 16) * (COPY_BYTES / 4), smem_f32, COPY_BYTES);
            }
            bulk_commit_group();
            // Compute while copy in flight
            #pragma unroll 1
            for (int c = 0; c < COMPUTE_ITERS; c++) {
                acc = acc * 1.00001f + 0.00001f;
            }
            bulk_wait_group<0>();
        }

        int64_t gt_end = globaltimer();
        uint64_t end = clock64();

        // Prevent optimizer from removing compute
        if (acc == -999.0f) g_dst[0] = acc;

        result->total_cycles = end - start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }
}

// ---------------------------------------------------------------------------
// Experiment 4: Compute-only baseline (for overlap comparison)
// ---------------------------------------------------------------------------
template<int COMPUTE_ITERS>
__global__ __launch_bounds__(128, 1)
void kernel_compute_only(BulkResult* result, int iters,
                         float* __restrict__ g_dst) {
    if (threadIdx.x == 0) {
        float acc = 1.0f;

        // Warmup
        for (int w = 0; w < 100; w++) {
            #pragma unroll 1
            for (int c = 0; c < COMPUTE_ITERS; c++) {
                acc = acc * 1.00001f + 0.00001f;
            }
        }

        uint64_t start = clock64();
        int64_t gt_start = globaltimer();

        for (int i = 0; i < iters; i++) {
            #pragma unroll 1
            for (int c = 0; c < COMPUTE_ITERS; c++) {
                acc = acc * 1.00001f + 0.00001f;
            }
        }

        int64_t gt_end = globaltimer();
        uint64_t end = clock64();

        if (acc == -999.0f) g_dst[0] = acc;

        result->total_cycles = end - start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }
}
