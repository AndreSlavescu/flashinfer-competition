#pragma once

// FP8 GEMM (tcgen05.mma kind::f8f6f4) microbenchmark kernels for B200 (sm_100a)
//
// Measures latency and throughput of tcgen05.mma.cta_group::1.kind::f8f6f4
// at competition-relevant shapes for the indexer kernel:
//   M=64 (heads), K=128 (head_dim, 4 K-tiles of 32), N=64..256
//
// FP8 K-tile = 32 elements (256 bits / 8 bits), vs BF16 K-tile = 16.
// No .ws variant exists for kind::f8f6f4 — always non-warp-specialized.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

// kerutils MMA wrappers and helpers
#include <kerutils/device/common.h>
#include <kerutils/device/sm100/gemm.cuh>
#include <kerutils/device/sm100/helpers.cuh>
#include <kerutils/device/sm100/intrinsics.cuh>

using namespace cute;
using bf16 = cutlass::bfloat16_t;
using fp8_e4m3 = cutlass::float_e4m3_t;

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
// Result buffer
// ---------------------------------------------------------------------------
struct BenchResult {
    uint64_t total_cycles;
    int64_t  gt_start_ns;
    int64_t  gt_end_ns;
};

// ---------------------------------------------------------------------------
// TMEM column assignments
//   A (input):  columns 256..383  (128 cols)
//   C (output): columns 0..255    (256 cols)
// ---------------------------------------------------------------------------
static constexpr int TMEM_COL_C = 0;
static constexpr int TMEM_COL_A = 256;

// =========================================================================
// SM100_MMA_F8F6F4_TS_NOELECT — FP8 NOELECT wrapper
//
// Mirrors SM100_MMA_F16BF16_TS_NOELECT but with kind::f8f6f4.
// No elect_one_sync() — benchmark runs thread 0 only (already elected).
// =========================================================================
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F8F6F4_TS_NOELECT
{
  static_assert(M == 64 || M == 128,
      "SM100_MMA_F8F6F4_TS_NOELECT M-mode size should be 64 or 128.");
  static_assert((M == 64  && (N % 8 == 0)  && (8 <= N)  && (N <= 256)) ||
                (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
                "SM100_MMA_F8F6F4_TS_NOELECT N-mode size invalid.");
  static_assert(a_major == UMMA::Major::K,
      "SM100_MMA_F8F6F4_TS_NOELECT A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
#else
    CUTE_INVALID_CONTROL_PATH("SM100_MMA_F8F6F4_TS_NOELECT requires CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

// MMA_Traits specialization for FP8 NOELECT TS
// Must be inside namespace cute where MMA_Traits is defined
namespace cute {
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F8F6F4_TS_NOELECT<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8,
      "SM100_MMA_F8F6F4_TS_NOELECT supports types with leq 8bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // Logical shape-K is always 256 bits; for 8-bit types -> K=32
  static constexpr int K = 32;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F8F6F4_TS_NOELECT<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};
} // namespace cute

// =========================================================================
// Experiment 1: fp8_latency_ts
// Single K-tile (K=32) RAW-chained FP8 MMA with commit+wait sync.
// Directly comparable to BF16 utcmma_latency_ts.
// =========================================================================
template<int M, int N>
__global__ __launch_bounds__(128, 1)
void kernel_fp8_latency_ts(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    // FP8: K=32 per tile (256 bits / 8 bits)
    static constexpr int K = 32;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // Swizzle for FP8: 1 byte/elem, K=32 -> 32 bytes/row -> SW32
    auto layout_B = ku::make_umma_canonical_k_major_layout<N, K, 32, fp8_e4m3>();
    fp8_e4m3* smem_B = reinterpret_cast<fp8_e4m3*>(smem_raw);

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
        using MMA_Atom = SM100_MMA_F8F6F4_TS_NOELECT<
            fp8_e4m3, fp8_e4m3, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(
            partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
        tA_frag.data().get() = TMEM_COL_A;

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        auto sB_frag = thr_mma.partition_fragment_B(sB);

        tiled_mma.accumulate_ = UMMA::ScaleOut::One;

        // Warmup
        uint32_t phase = 0;
        for (int i = 0; i < 100; i++) {
            gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
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
            gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
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

// =========================================================================
// Experiment 2: fp8_kdepth_ts
// Multi-K-tile sweep. Competition head_dim=128 -> K_DEPTH=4.
// =========================================================================
template<int M, int N, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_fp8_kdepth_ts(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 32;  // FP8: 32 elements per K-tile
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // Swizzle selection for FP8: 1 byte/elem
    // K_TOTAL=32 -> 32B/row -> SW32
    // K_TOTAL=64 -> 64B/row -> SW64
    // K_TOTAL=128 -> 128B/row -> SW128
    constexpr int SWIZZLE_B = (K_TOTAL >= 128) ? 128 : (K_TOTAL >= 64 ? 64 : 32);
    auto layout_B = ku::make_umma_canonical_k_major_layout<N, K_TOTAL, SWIZZLE_B, fp8_e4m3>();
    fp8_e4m3* smem_B = reinterpret_cast<fp8_e4m3*>(smem_raw);

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
        using MMA_Atom = SM100_MMA_F8F6F4_TS_NOELECT<
            fp8_e4m3, fp8_e4m3, float, M, N, UMMA::Major::K, UMMA::Major::K>;
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
        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        // --- Timed region ---
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
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

// =========================================================================
// Experiment 3: fp8_throughput_2acc
// Multi-accumulator rotation to break RAW dependencies.
// Measures pipeline initiation interval.
// =========================================================================
template<int M, int N, int N_ACC, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_fp8_throughput_2acc(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 32;
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;
    // Non-WS: M=64 accumulator uses N TMEM columns per accumulator
    static constexpr int COLS_PER_ACC = N;
    static constexpr int TMEM_COL_A_LOCAL = TMEM_COL_A;  // 256
    static_assert(TMEM_COL_C + N_ACC * COLS_PER_ACC <= TMEM_COL_A_LOCAL,
        "C accumulator columns overflow into A TMEM region");

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    constexpr int SWIZZLE_B = (K_TOTAL >= 128) ? 128 : (K_TOTAL >= 64 ? 64 : 32);
    auto layout_B = ku::make_umma_canonical_k_major_layout<N, K_TOTAL, SWIZZLE_B, fp8_e4m3>();
    fp8_e4m3* smem_B = reinterpret_cast<fp8_e4m3*>(smem_raw);

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
        using MMA_Atom = SM100_MMA_F8F6F4_TS_NOELECT<
            fp8_e4m3, fp8_e4m3, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(
            partition_shape_A(tiled_mma, Shape<Int<M>, Int<K_TOTAL>>{}));
        tA_frag.data().get() = TMEM_COL_A_LOCAL;

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});

        auto sB_frag = thr_mma.partition_fragment_B(sB);

        // Warmup
        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                tC_frag.data().get() = TMEM_COL_C + (k % N_ACC) * COLS_PER_ACC;
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        // --- Timed region ---
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;

        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                tC_frag.data().get() = TMEM_COL_C + (k % N_ACC) * COLS_PER_ACC;
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
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

// Experiment 4 (fp8_with_scale) removed — TMEM ld is already benchmarked
// in utcmma ver8, and post-GEMM operations (broadcast, reduction) will be
// benchmarked separately as their own building blocks.

// =========================================================================
// SM100_MMA_F8F6F4_SS_NOELECT — FP8 SS mode NOELECT wrapper
//
// Both A and B from SMEM (no A in TMEM).
// Mirrors SM100_MMA_F8F6F4_SS but removes elect_one_sync().
// Benchmark guards with threadIdx.x == 0 externally.
// =========================================================================
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F8F6F4_SS_NOELECT
{
  static_assert(M == 64 || M == 128,
      "SM100_MMA_F8F6F4_SS_NOELECT M-mode size should be 64 or 128.");
  static_assert((M == 64  && (N % 8 == 0)  && (8 <= N)  && (N <= 256)) ||
                (M == 128 && (N % 8 == 0)  && (8 <= N)  && (N <= 256)),
                "SM100_MMA_F8F6F4_SS_NOELECT N-mode size invalid.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    // No elect_one_sync() — bench uses threadIdx.x == 0 guard externally
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
#else
    CUTE_INVALID_CONTROL_PATH("SM100_MMA_F8F6F4_SS_NOELECT requires CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

// MMA_Traits specialization for FP8 SS NOELECT
// A from SMEM (desc), B from SMEM (desc), C in TMEM
namespace cute {
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F8F6F4_SS_NOELECT<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8,
      "SM100_MMA_F8F6F4_SS_NOELECT supports types with leq 8bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;  // SS: A from SMEM descriptor
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  static constexpr int K = 32;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F8F6F4_SS_NOELECT<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  a_neg, b_neg, c_sat>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};
} // namespace cute

// =========================================================================
// SM100_MMA_MXF8F6F4_SS_NOELECT — Block-scaled FP8 SS NOELECT wrapper
//
// PTX: tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale
// M=128 required (hardware constraint).
// Scale factors (E8M0) stored in TMEM at tsfa_addr / tsfb_addr.
// Identity scale: 0x7F = 2^(127-127) = 1.0
// =========================================================================
template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF8F6F4_SS_NOELECT
{
  static_assert(M == 128,
      "SM100_MMA_MXF8F6F4_SS_NOELECT M-mode size must be 128.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256),
      "SM100_MMA_MXF8F6F4_SS_NOELECT N-mode size must be multiple of 8 between 8..256.");

  using DRegisters   = void;
  using ARegisters   = uint64_t[1];
  using BRegisters   = uint64_t[1];
  using CRegisters   = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    // No elect_one_sync() — bench uses threadIdx.x == 0 guard externally
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(tsfa_addr), "r"(tsfb_addr));
#else
    CUTE_INVALID_CONTROL_PATH("SM100_MMA_MXF8F6F4_SS_NOELECT requires CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

// MMA_Traits specialization for block-scaled FP8 SS NOELECT
namespace cute {
template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_MXF8F6F4_SS_NOELECT<a_type, b_type, c_type, sf_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD   = c_type;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8,
      "SM100_MMA_MXF8F6F4_SS_NOELECT supports types with leq 8bit types");

  static constexpr int K         = 32;
  static constexpr int SFVecSize = 32;

  using FrgTypeA   = UMMA::smem_desc<a_major>;
  using FrgTypeB   = UMMA::smem_desc<b_major>;
  using FrgTypeC   = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, true>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, false>;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<>(
        traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_);

    SM100_MMA_MXF8F6F4_SS_NOELECT<a_type, b_type, c_type, sf_type,
                  M, N, a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c,
                                     uint32_t(traits.accumulate_), idesc,
                                     traits.tsfa_addr_, traits.tsfb_addr_);
  }
};
} // namespace cute

// =========================================================================
// Experiment 4: f8f6f4_latency_ss
// Single K-tile (K=32) RAW-chained FP8 SS MMA.
// A from SMEM (not TMEM). Comparable to f8f6f4_latency_ts.
// =========================================================================
template<int M, int N>
__global__ __launch_bounds__(128, 1)
void kernel_f8f6f4_latency_ss(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K         = 32;
    static constexpr int SWIZZLE   = 32;  // K=32 → 32 bytes/row → SW32
    static constexpr int TMEM_COLS = 128;

    extern __shared__ char smem_raw[];
    // A at offset 0, B immediately after A
    fp8_e4m3* smem_A = reinterpret_cast<fp8_e4m3*>(smem_raw);
    fp8_e4m3* smem_B = reinterpret_cast<fp8_e4m3*>(smem_raw + M * K * sizeof(fp8_e4m3));

    __shared__ __align__(16) uint32_t smem_tmem_addr;
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(TMEM_COLS, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_F8F6F4_SS_NOELECT<
            fp8_e4m3, fp8_e4m3, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto layout_A = ku::make_umma_canonical_k_major_layout<M, K, SWIZZLE, fp8_e4m3>();
        auto layout_B = ku::make_umma_canonical_k_major_layout<N, K, SWIZZLE, fp8_e4m3>();
        auto sA = make_tensor(make_smem_ptr(smem_A), layout_A);
        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma  = tiled_mma.get_slice(_0{});
        auto sA_frag  = thr_mma.partition_fragment_A(sA);
        auto sB_frag  = thr_mma.partition_fragment_B(sB);
        auto tC_frag  = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;  // column 0

        tiled_mma.accumulate_ = UMMA::ScaleOut::One;

        // Warmup
        uint32_t phase = 0;
        for (int i = 0; i < 100; i++) {
            gemm(tiled_mma, sA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
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
            gemm(tiled_mma, sA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
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
        TMEM::Allocator1Sm().free(0, TMEM_COLS);
    }
}

// =========================================================================
// Experiment 5: f8f6f4_kdepth_ss
// Multi-K-tile sweep in SS mode. K_DEPTH=4 matches competition head_dim=128.
// =========================================================================
template<int M, int N, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_f8f6f4_kdepth_ss(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 32;
    static constexpr int K_TOTAL    = K_DEPTH * K_PER_TILE;
    static constexpr int TMEM_COLS  = 128;

    constexpr int SWIZZLE = (K_TOTAL >= 128) ? 128 : (K_TOTAL >= 64 ? 64 : 32);

    extern __shared__ char smem_raw[];
    fp8_e4m3* smem_A = reinterpret_cast<fp8_e4m3*>(smem_raw);
    fp8_e4m3* smem_B = reinterpret_cast<fp8_e4m3*>(smem_raw + M * K_TOTAL * sizeof(fp8_e4m3));

    __shared__ __align__(16) uint32_t smem_tmem_addr;
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(TMEM_COLS, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_F8F6F4_SS_NOELECT<
            fp8_e4m3, fp8_e4m3, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto layout_A = ku::make_umma_canonical_k_major_layout<M, K_TOTAL, SWIZZLE, fp8_e4m3>();
        auto layout_B = ku::make_umma_canonical_k_major_layout<N, K_TOTAL, SWIZZLE, fp8_e4m3>();
        auto sA = make_tensor(make_smem_ptr(smem_A), layout_A);
        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma  = tiled_mma.get_slice(_0{});
        auto sA_frag  = thr_mma.partition_fragment_A(sA);
        auto sB_frag  = thr_mma.partition_fragment_B(sB);
        auto tC_frag  = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        // Warmup
        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        // --- Timed region ---
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
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
        TMEM::Allocator1Sm().free(0, TMEM_COLS);
    }
}

// =========================================================================
// Experiment 6: mxf8f6f4_latency_ss
// Single K-tile RAW-chained block-scaled FP8 SS MMA. M=128 required.
// Scale factors in TMEM at cols N (SFA) and N+1 (SFB).
// Key question: does block_scale add cycles vs unscaled f8f6f4?
// =========================================================================
using float_ue8m0_t = cutlass::float_ue8m0_t;

template<int N>
__global__ __launch_bounds__(128, 1)
void kernel_mxf8f6f4_latency_ss(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int M         = 128;  // mxf8f6f4 SS requires M=128
    static constexpr int K         = 32;
    static constexpr int SWIZZLE   = 32;
    static constexpr int TMEM_COLS = 128;
    static constexpr int TMEM_SF_COL = N;  // scale-factor base column

    extern __shared__ char smem_raw[];
    fp8_e4m3* smem_A = reinterpret_cast<fp8_e4m3*>(smem_raw);
    fp8_e4m3* smem_B = reinterpret_cast<fp8_e4m3*>(smem_raw + M * K * sizeof(fp8_e4m3));

    __shared__ __align__(16) uint32_t smem_tmem_addr;
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(TMEM_COLS, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_MXF8F6F4_SS_NOELECT<
            fp8_e4m3, fp8_e4m3, float, float_ue8m0_t,
            M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        // For ue8m0 scale factors, sub-column selection lives in the TMEM pointer encoding.
        // Do not use raw integer column + 1 (that produces misaligned TMEM accesses).
        auto tmem_sf_base = make_tmem_ptr<float_ue8m0_t>(TMEM_SF_COL);
        tiled_mma.tsfa_addr_ = raw_pointer_cast(tmem_sf_base + 0);
        tiled_mma.tsfb_addr_ = raw_pointer_cast(tmem_sf_base + 1);

        auto layout_A = ku::make_umma_canonical_k_major_layout<M, K, SWIZZLE, fp8_e4m3>();
        auto layout_B = ku::make_umma_canonical_k_major_layout<N, K, SWIZZLE, fp8_e4m3>();
        auto sA = make_tensor(make_smem_ptr(smem_A), layout_A);
        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma  = tiled_mma.get_slice(_0{});
        auto sA_frag  = thr_mma.partition_fragment_A(sA);
        auto sB_frag  = thr_mma.partition_fragment_B(sB);
        auto tC_frag  = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        tiled_mma.accumulate_ = UMMA::ScaleOut::One;

        // Warmup
        uint32_t phase = 0;
        for (int i = 0; i < 100; i++) {
            gemm(tiled_mma, sA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
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
            gemm(tiled_mma, sA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
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
        TMEM::Allocator1Sm().free(0, TMEM_COLS);
    }
}

// =========================================================================
// Experiment 7: mxf8f6f4_kdepth_ss
// K-depth sweep for block-scaled FP8. K_DEPTH=4 = competition head_dim=128.
// =========================================================================
template<int N, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_mxf8f6f4_kdepth_ss(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int M         = 128;
    static constexpr int K_PER_TILE = 32;
    static constexpr int K_TOTAL   = K_DEPTH * K_PER_TILE;
    static constexpr int TMEM_COLS = 128;
    static constexpr int TMEM_SF_COL = N;

    constexpr int SWIZZLE = (K_TOTAL >= 128) ? 128 : (K_TOTAL >= 64 ? 64 : 32);

    extern __shared__ char smem_raw[];
    fp8_e4m3* smem_A = reinterpret_cast<fp8_e4m3*>(smem_raw);
    fp8_e4m3* smem_B = reinterpret_cast<fp8_e4m3*>(smem_raw + M * K_TOTAL * sizeof(fp8_e4m3));

    __shared__ __align__(16) uint32_t smem_tmem_addr;
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(TMEM_COLS, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_MXF8F6F4_SS_NOELECT<
            fp8_e4m3, fp8_e4m3, float, float_ue8m0_t,
            M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto tmem_sf_base = make_tmem_ptr<float_ue8m0_t>(TMEM_SF_COL);
        tiled_mma.tsfa_addr_ = raw_pointer_cast(tmem_sf_base + 0);
        tiled_mma.tsfb_addr_ = raw_pointer_cast(tmem_sf_base + 1);

        auto layout_A = ku::make_umma_canonical_k_major_layout<M, K_TOTAL, SWIZZLE, fp8_e4m3>();
        auto layout_B = ku::make_umma_canonical_k_major_layout<N, K_TOTAL, SWIZZLE, fp8_e4m3>();
        auto sA = make_tensor(make_smem_ptr(smem_A), layout_A);
        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma  = tiled_mma.get_slice(_0{});
        auto sA_frag  = thr_mma.partition_fragment_A(sA);
        auto sB_frag  = thr_mma.partition_fragment_B(sB);
        auto tC_frag  = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        // Warmup
        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        // --- Timed region ---
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
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
        TMEM::Allocator1Sm().free(0, TMEM_COLS);
    }
}
