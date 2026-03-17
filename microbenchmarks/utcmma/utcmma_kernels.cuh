#pragma once

// UTCMMA (tcgen05.mma) microbenchmark kernels for B200 (sm_100a)
//
// Measures latency and throughput of:
//   - TS (TMEM×Shared) warp-specialized MMA  (QK GEMM path)
//   - SS (Shared×Shared) warp-specialized MMA (SV GEMM path)
//   - UTCCP (smem→TMEM copy for Q loading)
//   - Swizzle impact on MMA performance

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>

// kerutils MMA wrappers (SM100_MMA_F16BF16_WS_TS_NOELECT, etc.)
#include <kerutils/device/common.h>
#include <kerutils/device/sm100/gemm.cuh>
#include <kerutils/device/sm100/helpers.cuh>
#include <kerutils/device/sm100/intrinsics.cuh>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

// ---------------------------------------------------------------------------
// Cycle-accurate timing helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__
uint64_t clock64_start() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__
uint64_t clock64_stop() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

// ---------------------------------------------------------------------------
// Result buffer layout: per-run results written by thread 0
// [total_cycles, globaltimer_ns_start, globaltimer_ns_end]
// ---------------------------------------------------------------------------
struct BenchResult {
    uint64_t total_cycles;
    int64_t  gt_start_ns;
    int64_t  gt_end_ns;
};

// ---------------------------------------------------------------------------
// TMEM column assignments for benchmarks
// We allocate all 512 columns and use:
//   A (input):  columns 256..383  (128 cols)
//   C (output): columns 0..255    (256 cols)
// This matches the FlashMLA decode kernel layout.
// ---------------------------------------------------------------------------
static constexpr int TMEM_COL_C = 0;
static constexpr int TMEM_COL_A = 256;

// ---------------------------------------------------------------------------
// Experiment 1: utcmma_latency_ts
// Single K-tile latency for TS (TMEM×Shared) warp-specialized MMA.
// Template params: M, N tile dimensions.
// ---------------------------------------------------------------------------
template<int M, int N>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_latency_ts(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    // K is always 16 bf16 elements (256 bits) per MMA tile
    static constexpr int K = 16;

    // --- Shared memory ---
    extern __shared__ char smem_raw[];

    // TMEM address storage (written by allocator)
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // Shared memory for B operand: [N x K] in K-major layout.
    // SW128 K-atom=64 doesn't fit K=16; use SW32 (K-atom=16) for single-tile latency.
    auto layout_B = ku::make_umma_canonical_k_major_layout<N, K, 32, bf16>();
    bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);

    // --- TMEM allocation (requires full warp) ---
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    // --- Only thread 0 runs the benchmark ---
    if (threadIdx.x == 0) {
        // Create TiledMMA
        using MMA_Atom = SM100_MMA_F16BF16_WS_TS_NOELECT<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        // Create smem tensor and descriptor for B
        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);
        UMMA::SmemDescriptor desc_B = UMMA::make_umma_desc<UMMA::Major::K>(sB);

        // Create TMEM fragments
        // A fragment: TMEM input
        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(
            partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
        tA_frag.data().get() = TMEM_COL_A;

        // C fragment: TMEM output
        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        // B fragment (descriptor in register)
        auto sB_frag = thr_mma.partition_fragment_B(sB);

        // Set accumulate mode (creates RAW dependency for latency measurement)
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;

        // Warmup: a few iterations to stabilize
        for (int i = 0; i < 100; i++) {
            gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
        }

        // --- Timed region ---
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();

    // Free TMEM
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// ---------------------------------------------------------------------------
// Experiment 2: utcmma_kdepth_ts
// K-depth sweep: measures total MMA time for k_depth K-tiles at M=64, N=128.
// Uses utcmma_ts()-style K-dimension unrolling.
// ---------------------------------------------------------------------------
template<int M, int N, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_kdepth_ts(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 16;  // bf16
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // B operand: [N x K_TOTAL] in K-major layout, swizzle chosen to fit K_TOTAL.
    // SW128 K-atom=64, SW64 K-atom=32, SW32 K-atom=16 (all multiples of 16 work).
    constexpr int SWIZZLE_B = (K_TOTAL >= 64) ? 128 : (K_TOTAL >= 32 ? 64 : 32);
    auto layout_B = ku::make_umma_canonical_k_major_layout<N, K_TOTAL, SWIZZLE_B, bf16>();
    bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_F16BF16_WS_TS_NOELECT<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(
            partition_shape_A(tiled_mma, Shape<Int<M>, Int<K_TOTAL>>{}));
        tA_frag.data().get() = TMEM_COL_A;

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        auto sB_frag = thr_mma.partition_fragment_B(sB);

        // Warmup
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
        }

        // --- Timed region ---
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// ---------------------------------------------------------------------------
// Experiment 3: utcmma_latency_ss
// SS (Shared×Shared) MMA latency — SV GEMM path.
// A from shared (K-major INTER), B from shared (MN-major SW128).
// ---------------------------------------------------------------------------
template<int M, int N, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_latency_ss(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 16;
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // A operand: [M x K_TOTAL] in K-major INTER layout (matches SmemLayoutS)
    auto layout_A = coalesce(tile_to_shape(
        UMMA::Layout_K_INTER_Atom<bf16>{},
        Shape<Int<M>, Int<K_TOTAL>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{});

    // B operand: [N x K_TOTAL] in MN-major SW128 layout (matches transposed V)
    auto layout_B = coalesce(tile_to_shape(
        UMMA::Layout_MN_SW128_Atom<bf16>{},
        Shape<Int<N>, Int<K_TOTAL>>{},
        Step<_2, _1>{}
    ), Shape<_1, _1>{});

    // Place A and B sequentially in shared memory
    bf16* smem_A = reinterpret_cast<bf16*>(smem_raw);
    bf16* smem_B = smem_A + cosize(layout_A);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_F16BF16_WS_SS_NOELECT<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::MN>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sA = make_tensor(make_smem_ptr(smem_A), layout_A);
        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto sA_frag = thr_mma.partition_fragment_A(sA);
        auto sB_frag = thr_mma.partition_fragment_B(sB);

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        // Warmup
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
        }

        // --- Timed region ---
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// ---------------------------------------------------------------------------
// Experiment 4: utccp_latency
// UTCCP (smem→TMEM copy) latency measurement.
// PTX: tcgen05.cp.cta_group::1.128x256b [tmem_col], smem_desc
// ---------------------------------------------------------------------------
template<int NUM_COPIES>
__global__ __launch_bounds__(128, 1)
void kernel_utccp_latency(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // Shared memory for source data: NUM_COPIES consecutive 128×16 bf16 tiles.
    // make_umma_desc<K> only accepts a single K-tile (K=16 bf16 = 2 u128): the canonical
    // K-major form requires mode-1 = exactly 2 u128 elements. Multi-tile K is handled by
    // advancing smem pointer per copy, matching how TiledMMA does it internally.
    static constexpr int COLS_PER_COPY = 8;  // Each UTCCP 128dp256bit copies 128×256 bits = 128×16 bf16 → 8 TMEM cols
    static constexpr int K_PER_COPY   = 16;
    static constexpr int ELEMS_PER_COPY = 128 * K_PER_COPY;  // bf16 elements per copy
    // SW32 has K-atom=16 (32 bytes / 2 bytes per bf16), fits K=16 exactly.
    auto layout_per_copy = ku::make_umma_canonical_k_major_layout<128, K_PER_COPY, 32, bf16>();
    bf16* smem_src = reinterpret_cast<bf16*>(smem_raw);

    // Barrier for commit synchronization
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Pre-compute per-copy descriptors. Each descriptor covers one 128×16 bf16 tile.
        // The c-th tile starts at smem_src + c * ELEMS_PER_COPY.
        uint64_t descs[NUM_COPIES];
        CUTE_UNROLL
        for (int c = 0; c < NUM_COPIES; c++) {
            auto sSrc_c = make_tensor(make_smem_ptr(smem_src + c * ELEMS_PER_COPY), layout_per_copy);
            descs[c] = UMMA::make_umma_desc<UMMA::Major::K>(sSrc_c).desc_;
        }

        // Warmup: alternate phases 0/1
        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            CUTE_UNROLL
            for (int c = 0; c < NUM_COPIES; c++) {
                SM100::TMEM::UTCCP::SM100_UTCCP_128dp256bit_1cta::copy(
                    descs[c],
                    TMEM_COL_A + c * COLS_PER_COPY
                );
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        // --- Timed region ---
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            CUTE_UNROLL
            for (int c = 0; c < NUM_COPIES; c++) {
                SM100::TMEM::UTCCP::SM100_UTCCP_128dp256bit_1cta::copy(
                    descs[c],
                    TMEM_COL_A + c * COLS_PER_COPY
                );
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// ---------------------------------------------------------------------------
// Experiment 5: utcmma_swizzle
// Compare different swizzle modes for B operand at fixed M=64, N=128, K_DEPTH=4.
// ---------------------------------------------------------------------------
template<int M, int N, int K_DEPTH, int SWIZZLE>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_swizzle(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 16;
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // B layout with parameterized swizzle
    // Using helpers from kerutils/device/sm100/helpers.cuh
    auto layout_B = ku::make_umma_canonical_k_major_layout<N, K_TOTAL, SWIZZLE, bf16>();
    bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_F16BF16_WS_TS_NOELECT<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(
            partition_shape_A(tiled_mma, Shape<Int<M>, Int<K_TOTAL>>{}));
        tA_frag.data().get() = TMEM_COL_A;

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        auto sB_frag = thr_mma.partition_fragment_B(sB);

        // Warmup
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
        }

        // --- Timed region ---
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}
