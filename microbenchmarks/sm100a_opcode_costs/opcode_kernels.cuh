#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <kerutils/device/common.h>
#include <kerutils/device/sm100/gemm.cuh>
#include <kerutils/device/sm100/helpers.cuh>
#include <kerutils/device/sm100/intrinsics.cuh>

#include "benchmark_common.cuh"
#include "tensor_map_utils.cuh"

using bf16 = cutlass::bfloat16_t;

struct BenchResult {
    uint64_t total_cycles;
    int64_t gt_start_ns;
    int64_t gt_end_ns;
    uint64_t sink0;
    uint64_t sink1;
};

static constexpr int TMEM_COL_C = 0;
static constexpr int TMEM_COL_A = 256;

__device__ __forceinline__ uint64_t clock64_start() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__ uint64_t clock64_stop() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__ uint32_t smem_addr32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t local_addr64(const volatile void* ptr) {
    uint64_t addr;
    asm volatile("cvta.to.local.u64 %0, %1;\n" : "=l"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void mbarrier_init_u64(uint64_t* mbar, uint32_t expected_count) {
    uint32_t addr = smem_addr32(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(expected_count) : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx_u64(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t addr = smem_addr32(mbar);
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_u64(uint64_t* mbar) {
    uint32_t addr = smem_addr32(mbar);
    asm volatile(
        "{\n"
        ".reg .b64 state;\n"
        "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
        "}\n"
        :: "r"(addr)
        : "memory");
}

__device__ __forceinline__ void mbarrier_wait_u64(uint64_t* mbar, uint32_t phase) {
    uint32_t addr = smem_addr32(mbar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra WAIT_%=;\n"
        "}\n"
        :: "r"(addr), "r"(phase)
        : "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_gather4_async(
    const void* desc_ptr,
    uint64_t* mbar_ptr,
    void* smem_ptr,
    int col_idx,
    int4 row_idxs,
    int64_t cache_hint) {
    uint32_t smem_addr = smem_addr32(smem_ptr);
    uint32_t mbar_addr = smem_addr32(mbar_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :: "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
           "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
           "r"(mbar_addr), "l"(cache_hint)
        : "memory");
}

__device__ __forceinline__ void tma_gather4_prefetch_async(
    const void* desc_ptr,
    int col_idx,
    int4 row_idxs,
    int64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint"
        " [%0, {%1, %2, %3, %4, %5}], %6;\n"
        :: "l"(desc_ptr), "r"(col_idx),
           "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
           "l"(cache_hint)
        : "memory");
}

__device__ __forceinline__ void cp_async_ldgsts_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem = smem_addr32(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void bulk_reduce_add_s2g(void* dst_global, const void* src_smem, int32_t bytes) {
    uint32_t smem_addr = smem_addr32(src_smem);
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
        :: "l"(dst_global), "r"(smem_addr), "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void prefetch_l2_global(const void* gmem_ptr) {
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(gmem_ptr) : "memory");
}

template <int BATCH>
__global__ void async_dummy_kernel(BenchResult* result, int outer_iters) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 1;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            asm volatile("" : "+r"(acc));
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

__global__ void sync_scalar_baseline_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 1;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("" : "+r"(acc));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

__global__ void sync_warp_baseline_kernel(BenchResult* result, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t acc = static_cast<uint32_t>(lane + 1);
    __syncwarp();
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (lane == 0) {
        gt_start = globaltimer();
        c_start = clock64_start();
    }
    for (int i = 0; i < iters; ++i) {
        asm volatile("" : "+r"(acc));
    }
    __syncwarp();
    if (lane == 0) {
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = acc;
        result->sink1 = 0;
    }
}

__global__ void sync_ffma_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    float acc = 1.0f;
    const float mul = 1.0009765625f;
    const float add = 0.0009765625f;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("fma.rn.f32 %0, %0, %1, %2;\n" : "+f"(acc) : "f"(mul), "f"(add));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = __float_as_uint(acc);
    result->sink1 = 0;
}

__global__ void sync_fadd_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    float acc = 1.0f;
    const float add = 0.0009765625f;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("add.f32 %0, %0, %1;\n" : "+f"(acc) : "f"(add));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = __float_as_uint(acc);
    result->sink1 = 0;
}

__global__ void sync_fmul_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    float acc = 1.0f;
    const float mul = 1.0009765625f;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("mul.rn.f32 %0, %0, %1;\n" : "+f"(acc) : "f"(mul));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = __float_as_uint(acc);
    result->sink1 = 0;
}

__global__ void sync_hfma2_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 0x3c003c00u;
    const uint32_t mul = 0x3c003c00u;
    const uint32_t add = 0x34003400u;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("fma.rn.f16x2 %0, %0, %1, %2;\n" : "+r"(acc) : "r"(mul), "r"(add));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

__global__ void sync_mufu_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    float acc = 0.5f;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("ex2.approx.ftz.f32 %0, %0;\n" : "+f"(acc));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = __float_as_uint(acc);
    result->sink1 = 0;
}

__global__ void sync_imad_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    int acc = 1;
    const int mul = 17;
    const int add = 3;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("mad.lo.s32 %0, %0, %1, %2;\n" : "+r"(acc) : "r"(mul), "r"(add));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = static_cast<uint32_t>(acc);
    result->sink1 = 0;
}

__global__ void sync_iadd3_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 1;
    const uint32_t add = 3;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("add.u32 %0, %0, %1;\n" : "+r"(acc) : "r"(add));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

__global__ void sync_lop3_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 0x13572468u;
    const uint32_t b = 0x24681357u;
    const uint32_t c = 0xf0f0f0f0u;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;\n" : "+r"(acc) : "r"(b), "r"(c));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

__global__ void sync_mov_kernel(BenchResult* result, int iters) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 1;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < iters; ++i) {
        asm volatile("mov.b32 %0, %0;\n" : "+r"(acc));
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

__global__ void sync_shfl_kernel(BenchResult* result, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t acc = static_cast<uint32_t>(lane + 1);
    __syncwarp();
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (lane == 0) {
        gt_start = globaltimer();
        c_start = clock64_start();
    }
    for (int i = 0; i < iters; ++i) {
        acc = __shfl_sync(0xffffffffu, acc, (lane + 1) & 31);
    }
    __syncwarp();
    if (lane == 0) {
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = acc;
        result->sink1 = 0;
    }
}

// Keep the special-register read in a separate noinline helper so ptxas
// cannot fold a single invariant S2R into the loop preheader.
__device__ __noinline__ uint32_t read_s2r_selected(uint32_t selector) {
    uint32_t special;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .u32 lane;\n"
        "setp.ne.u32 p, %1, 0;\n"
        "mov.u32 lane, %%laneid;\n"
        "selp.u32 %0, lane, %1, p;\n"
        "}\n"
        : "=r"(special)
        : "r"(selector)
        : "memory");
    return special;
}

__device__ __noinline__ uint32_t read_s2r_selected_baseline(uint32_t selector) {
    uint32_t special;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .u32 lane;\n"
        "setp.ne.u32 p, %1, 0;\n"
        "mov.u32 lane, %1;\n"
        "selp.u32 %0, lane, %1, p;\n"
        "}\n"
        : "=r"(special)
        : "r"(selector)
        : "memory");
    return special;
}

__global__ void sync_s2r_kernel(BenchResult* result, int iters) {
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    uint32_t acc = static_cast<uint32_t>(lane + 1);
    __syncwarp(mask);
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (lane == 0) {
        gt_start = globaltimer();
        c_start = clock64_start();
    }
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        uint32_t special = read_s2r_selected(acc);
        acc = ((acc << 1) | (acc >> 31)) ^ special ^ 0x9e3779b9u;
    }
    uint32_t warp_acc = __reduce_xor_sync(mask, acc);
    __syncwarp(mask);
    if (lane == 0) {
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = warp_acc;
        result->sink1 = 0;
    }
}

__global__ void sync_s2r_baseline_kernel(BenchResult* result, int iters) {
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    uint32_t acc = static_cast<uint32_t>(lane + 1);
    __syncwarp(mask);
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (lane == 0) {
        gt_start = globaltimer();
        c_start = clock64_start();
    }
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        uint32_t special = read_s2r_selected_baseline(acc);
        acc = ((acc << 1) | (acc >> 31)) ^ special ^ 0x9e3779b9u;
    }
    uint32_t warp_acc = __reduce_xor_sync(mask, acc);
    __syncwarp(mask);
    if (lane == 0) {
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = warp_acc;
        result->sink1 = 0;
    }
}

// Chase a 32-entry shared-memory ring so each LDS depends on the previous load.
__global__ void sync_lds_kernel(BenchResult* result, int iters) {
    __shared__ uint32_t smem_ring[32];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32; ++i) {
            smem_ring[i] = smem_addr32(&smem_ring[(i + 1) & 31]);
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t addr = smem_addr32(&smem_ring[0]);
    uint32_t acc = addr;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        uint32_t next_addr;
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(next_addr) : "r"(addr) : "memory");
        addr = next_addr;
        acc ^= next_addr;
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = addr;
}

// Approximate STS latency by alternating between two shared addresses, forcing
// visibility with membar.cta, and subtracting a matched membar+LDS baseline.
__global__ void sync_sts_proxy_kernel(BenchResult* result, int iters) {
    __shared__ uint32_t smem_ring[32];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32; ++i) {
            smem_ring[i] = smem_addr32(&smem_ring[(i + 1) & 31]);
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t addr = smem_addr32(&smem_ring[0]);
    uint32_t first_addr = smem_addr32(&smem_ring[0]);
    uint32_t second_addr = smem_addr32(&smem_ring[1]);
    uint32_t addr_xor = first_addr ^ second_addr;
    uint32_t acc = addr;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        uint32_t cur_addr = addr;
        uint32_t next_addr = cur_addr ^ addr_xor;
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(cur_addr), "r"(next_addr) : "memory");
        asm volatile("membar.cta;\n" ::: "memory");
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(addr) : "r"(cur_addr) : "memory");
        acc ^= addr;
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = addr;
}

// Matched control path for the STS proxy above: barrier plus dependent LDS.
__global__ void sync_sts_baseline_kernel(BenchResult* result, int iters) {
    __shared__ uint32_t smem_ring[32];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32; ++i) {
            smem_ring[i] = smem_addr32(&smem_ring[(i + 1) & 31]);
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t addr = smem_addr32(&smem_ring[0]);
    uint32_t acc = addr;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        uint32_t next_addr;
        asm volatile("membar.cta;\n" ::: "memory");
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(next_addr) : "r"(addr) : "memory");
        addr = next_addr;
        acc ^= next_addr;
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = addr;
}

template <int BATCH>
__global__ void async_tma_load_kernel(BenchResult* result, int outer_iters, const __grid_constant__ CUtensorMap tensor_map, const int* indices, int bytes_per_gather4, int64_t cache_hint) {
    extern __shared__ char smem_raw[];
    int aligned_bytes = (bytes_per_gather4 + 7) & ~7;
    char* smem_data = smem_raw;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + aligned_bytes * BATCH);
    if (threadIdx.x != 0) {
        return;
    }
    mbarrier_init_u64(mbar, 1);
    fence_proxy_async_shared();
    uint32_t phase = 0;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            int4 rows = reinterpret_cast<const int4*>(indices)[b];
            tma_gather4_async(&tensor_map, mbar, smem_data + b * aligned_bytes, 0, rows, cache_hint);
        }
        mbarrier_expect_tx_u64(mbar, static_cast<uint32_t>(BATCH * bytes_per_gather4));
        mbarrier_arrive_u64(mbar);
        mbarrier_wait_u64(mbar, phase);
        phase ^= 1;
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = phase;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_tma_load_baseline_kernel(BenchResult* result, int outer_iters) {
    __shared__ uint64_t mbar_storage;
    if (threadIdx.x != 0) {
        return;
    }
    mbarrier_init_u64(&mbar_storage, 1);
    fence_proxy_async_shared();
    uint32_t phase = 0;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        mbarrier_expect_tx_u64(&mbar_storage, 0);
        mbarrier_arrive_u64(&mbar_storage);
        mbarrier_wait_u64(&mbar_storage, phase);
        phase ^= 1;
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = phase;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_tma_prefetch_kernel(BenchResult* result, int outer_iters, const __grid_constant__ CUtensorMap tensor_map, const int* indices, int64_t cache_hint) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 0;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            int4 rows = reinterpret_cast<const int4*>(indices)[b];
            tma_gather4_prefetch_async(&tensor_map, 0, rows, cache_hint);
            acc += static_cast<uint32_t>(rows.x);
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_tma_store_kernel(BenchResult* result, int outer_iters, const __grid_constant__ CUtensorMap tensor_map, int row_stride_elems) {
    extern __shared__ char smem_raw[];
    if (threadIdx.x == 0) {
        for (int i = 0; i < row_stride_elems; ++i) {
            reinterpret_cast<bf16*>(smem_raw)[i] = bf16(1.0f);
        }
        cute::tma_store_fence();
        int64_t gt_start = globaltimer();
        uint64_t c_start = clock64_start();
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                cute::SM90_TMA_STORE::copy(&tensor_map, smem_raw, 0, b);
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = 0;
        result->sink1 = 0;
    }
}

template <int BATCH>
__global__ void async_tma_store_baseline_kernel(BenchResult* result, int outer_iters) {
    if (threadIdx.x != 0) {
        return;
    }
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = 0;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_tma_reduce_kernel(BenchResult* result, int outer_iters, const __grid_constant__ CUtensorMap tensor_map, int row_stride_elems) {
    extern __shared__ char smem_raw[];
    if (threadIdx.x == 0) {
        for (int i = 0; i < row_stride_elems; ++i) {
            reinterpret_cast<float*>(smem_raw)[i] = 1.0f;
        }
        cute::tma_store_fence();
        int64_t gt_start = globaltimer();
        uint64_t c_start = clock64_start();
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                cute::SM90_TMA_REDUCE_ADD::copy(&tensor_map, smem_raw, 0, b);
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = 0;
        result->sink1 = 0;
    }
}

template <int BATCH>
__global__ void async_bulk_copy_kernel(BenchResult* result, int outer_iters, char* gmem_ptr, int copy_bytes) {
    extern __shared__ char smem_raw[];
    if (threadIdx.x == 0) {
        cute::tma_store_fence();
        int64_t gt_start = globaltimer();
        uint64_t c_start = clock64_start();
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                cute::SM90_BULK_COPY_S2G::copy(smem_raw, gmem_ptr + b * copy_bytes, copy_bytes);
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = 0;
        result->sink1 = 0;
    }
}

template <int BATCH>
__global__ void async_bulk_reduce_kernel(BenchResult* result, int outer_iters, float* gmem_ptr, int copy_bytes) {
    extern __shared__ char smem_raw[];
    if (threadIdx.x == 0) {
        int64_t gt_start = globaltimer();
        uint64_t c_start = clock64_start();
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                bulk_reduce_add_s2g(gmem_ptr + b * (copy_bytes / sizeof(float)), smem_raw, copy_bytes);
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = 0;
        result->sink1 = 0;
    }
}

template <int BATCH>
__global__ void async_bulk_prefetch_kernel(BenchResult* result, int outer_iters, const char* gmem_ptr, int copy_bytes) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 0;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            cute::SM90_BULK_COPY_G2S::PREFETCH::copy(gmem_ptr + b * copy_bytes, copy_bytes);
            acc += static_cast<uint32_t>(b);
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_ldg_kernel(BenchResult* result, int outer_iters, const uint64_t* gmem_ptr) {
    if (threadIdx.x != 0) {
        return;
    }
    uint64_t regs[BATCH][4];
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            asm volatile(
                "ld.global.v4.u64 {%0, %1, %2, %3}, [%4];\n"
                : "=l"(regs[b][0]), "=l"(regs[b][1]), "=l"(regs[b][2]), "=l"(regs[b][3])
                : "l"(gmem_ptr + b * 4)
                : "memory");
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = regs[0][0];
    result->sink1 = regs[BATCH - 1][3];
}

template <int BATCH>
__global__ void async_stg_kernel(BenchResult* result, int outer_iters, uint64_t* gmem_ptr) {
    if (threadIdx.x != 0) {
        return;
    }
    uint64_t src0 = 1, src1 = 2, src2 = 3, src3 = 4;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            asm volatile(
                "st.global.v4.u64 [%0], {%1, %2, %3, %4};\n"
                :: "l"(gmem_ptr + b * 4), "l"(src0), "l"(src1), "l"(src2), "l"(src3)
                : "memory");
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = src0;
    result->sink1 = src3;
}

template <int BATCH>
__global__ void async_ldl_kernel(BenchResult* result, int outer_iters) {
    if (threadIdx.x != 0) {
        return;
    }
    volatile uint32_t local_buf[256];
    #pragma unroll
    for (int i = 0; i < 256; ++i) {
        local_buf[i] = static_cast<uint32_t>(i + 1);
    }
    uint32_t vals[BATCH];
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            uint64_t addr = local_addr64(&local_buf[b]);
            asm volatile("ld.local.u32 %0, [%1];\n" : "=r"(vals[b]) : "l"(addr) : "memory");
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = vals[0];
    result->sink1 = vals[BATCH - 1];
}

template <int BATCH>
__global__ void async_stl_kernel(BenchResult* result, int outer_iters) {
    if (threadIdx.x != 0) {
        return;
    }
    volatile uint32_t local_buf[256];
    uint32_t value = 1;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            uint64_t addr = local_addr64(&local_buf[b]);
            asm volatile("st.local.u32 [%0], %1;\n" :: "l"(addr), "r"(value + b) : "memory");
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = value;
    result->sink1 = local_buf[0];
}

template <int BATCH>
__global__ void async_ldgsts_kernel(BenchResult* result, int outer_iters, const char* gmem_ptr) {
    extern __shared__ char smem_raw[];
    if (threadIdx.x != 0) {
        return;
    }
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            cp_async_ldgsts_16(smem_raw + b * 16, gmem_ptr + b * 16);
        }
        cp_async_commit_group();
        cp_async_wait_group_0();
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = static_cast<uint8_t>(smem_raw[0]);
    result->sink1 = static_cast<uint8_t>(smem_raw[(BATCH - 1) * 16]);
}

template <int BATCH>
__global__ void async_ldgsts_baseline_kernel(BenchResult* result, int outer_iters) {
    if (threadIdx.x != 0) {
        return;
    }
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        cp_async_commit_group();
        cp_async_wait_group_0();
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = 0;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_cctl_prefetch_kernel(BenchResult* result, int outer_iters, const char* gmem_ptr) {
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t acc = 0;
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            prefetch_l2_global(gmem_ptr + b * 64);
            acc += static_cast<uint32_t>(b);
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = acc;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_mbar_kernel(BenchResult* result, int outer_iters) {
    __shared__ uint64_t mbar_storage;
    if (threadIdx.x != 0) {
        return;
    }
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        mbarrier_init_u64(&mbar_storage, 1);
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            mbarrier_arrive_u64(&mbar_storage);
        }
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = 0;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_mbar_baseline_kernel(BenchResult* result, int outer_iters) {
    __shared__ uint64_t mbar_storage;
    if (threadIdx.x != 0) {
        return;
    }
    int64_t gt_start = globaltimer();
    uint64_t c_start = clock64_start();
    for (int i = 0; i < outer_iters; ++i) {
        mbarrier_init_u64(&mbar_storage, 1);
    }
    uint64_t c_end = clock64_stop();
    int64_t gt_end = globaltimer();
    result->total_cycles = c_end - c_start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;
    result->sink0 = 0;
    result->sink1 = 0;
}

template <int BATCH>
__global__ void async_bar_kernel(BenchResult* result, int outer_iters) {
    const int lane = threadIdx.x & 31;
    __syncthreads();
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (threadIdx.x == 0) {
        gt_start = globaltimer();
        c_start = clock64_start();
    }
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            __syncthreads();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = lane;
        result->sink1 = 0;
    }
}

template <int BATCH>
__global__ void async_bar_baseline_kernel(BenchResult* result, int outer_iters) {
    uint32_t acc = threadIdx.x + 1;
    __syncthreads();
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (threadIdx.x == 0) {
        gt_start = globaltimer();
        c_start = clock64_start();
    }
    for (int i = 0; i < outer_iters; ++i) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            asm volatile("" : "+r"(acc));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = acc;
        result->sink1 = 0;
    }
}

template <int BATCH>
__global__ void async_utccp_kernel(BenchResult* result, int outer_iters) {
    using namespace cute;
    namespace ku = kerutils;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    static constexpr int COLS_PER_COPY = 8;
    static constexpr int K_PER_COPY = 16;
    static constexpr int ELEMS_PER_COPY = 128 * K_PER_COPY;
    auto layout_per_copy = ku::make_umma_canonical_k_major_layout<128, K_PER_COPY, 32, bf16>();
    bf16* smem_src = reinterpret_cast<bf16*>(smem_raw);

    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t descs[BATCH];
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            auto s_src = make_tensor(make_smem_ptr(smem_src + b * ELEMS_PER_COPY), layout_per_copy);
            descs[b] = UMMA::make_umma_desc<UMMA::Major::K>(s_src).desc_;
        }
        uint32_t phase = 0;
        int64_t gt_start = globaltimer();
        uint64_t c_start = clock64_start();
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                SM100::TMEM::UTCCP::SM100_UTCCP_128dp256bit_1cta::copy(descs[b], TMEM_COL_A + b * COLS_PER_COPY);
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = phase;
        result->sink1 = 0;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

template <int BATCH>
__global__ void async_utccp_baseline_kernel(BenchResult* result, int outer_iters) {
    using namespace cute;
    namespace ku = kerutils;

    __shared__ __align__(16) uint32_t smem_tmem_addr;
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t phase = 0;
        int64_t gt_start = globaltimer();
        uint64_t c_start = clock64_start();
        for (int i = 0; i < outer_iters; ++i) {
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = phase;
        result->sink1 = 0;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

template <int BATCH>
__global__ void async_ldt_kernel(BenchResult* result, int outer_iters) {
    namespace ku = kerutils;
    __shared__ __align__(16) uint32_t smem_tmem_addr;
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        uint32_t init[BATCH];
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            init[b] = static_cast<uint32_t>(b + 1);
            ku::tmem_st_32dp32bNx<1>(TMEM_COL_C + b, &init[b]);
        }
        cutlass::arch::fence_view_async_tmem_store();
        uint32_t out[BATCH];
        int64_t gt_start = 0;
        uint64_t c_start = 0;
        if (threadIdx.x == 0) {
            gt_start = globaltimer();
            c_start = clock64_start();
        }
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                ku::tmem_ld_32dp32bNx<1>(TMEM_COL_C + b, &out[b]);
            }
            cutlass::arch::fence_view_async_tmem_load();
        }
        if (threadIdx.x == 0) {
            uint64_t c_end = clock64_stop();
            int64_t gt_end = globaltimer();
            result->total_cycles = c_end - c_start;
            result->gt_start_ns = gt_start;
            result->gt_end_ns = gt_end;
            result->sink0 = out[0];
            result->sink1 = out[BATCH - 1];
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

template <int BATCH>
__global__ void async_ldtm_kernel(BenchResult* result, int outer_iters) {
    namespace ku = kerutils;
    __shared__ __align__(16) uint32_t smem_tmem_addr;
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        uint32_t init[8 * BATCH];
        #pragma unroll
        for (int i = 0; i < 8 * BATCH; ++i) {
            init[i] = static_cast<uint32_t>(i + 1);
        }
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            ku::tmem_st_32dp32bNx<8>(TMEM_COL_C + b * 8, &init[b * 8]);
        }
        cutlass::arch::fence_view_async_tmem_store();
        uint32_t out[8 * BATCH];
        int64_t gt_start = 0;
        uint64_t c_start = 0;
        if (threadIdx.x == 0) {
            gt_start = globaltimer();
            c_start = clock64_start();
        }
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                ku::tmem_ld_16dp256bNx<1>(TMEM_COL_C + b * 8, &out[b * 8]);
            }
            cutlass::arch::fence_view_async_tmem_load();
        }
        if (threadIdx.x == 0) {
            uint64_t c_end = clock64_stop();
            int64_t gt_end = globaltimer();
            result->total_cycles = c_end - c_start;
            result->gt_start_ns = gt_start;
            result->gt_end_ns = gt_end;
            result->sink0 = out[0];
            result->sink1 = out[8 * BATCH - 1];
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

template <int BATCH>
__global__ void async_ldt_baseline_kernel(BenchResult* result, int outer_iters) {
    if (threadIdx.x < 32) {
        int64_t gt_start = 0;
        uint64_t c_start = 0;
        if (threadIdx.x == 0) {
            gt_start = globaltimer();
            c_start = clock64_start();
        }
        for (int i = 0; i < outer_iters; ++i) {
            cutlass::arch::fence_view_async_tmem_load();
        }
        if (threadIdx.x == 0) {
            uint64_t c_end = clock64_stop();
            int64_t gt_end = globaltimer();
            result->total_cycles = c_end - c_start;
            result->gt_start_ns = gt_start;
            result->gt_end_ns = gt_end;
            result->sink0 = 0;
            result->sink1 = 0;
        }
    }
}

template <int BATCH>
__global__ void async_stt_kernel(BenchResult* result, int outer_iters) {
    namespace ku = kerutils;
    __shared__ __align__(16) uint32_t smem_tmem_addr;
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        uint32_t data[BATCH];
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            data[b] = static_cast<uint32_t>(b + 1);
        }
        int64_t gt_start = 0;
        uint64_t c_start = 0;
        if (threadIdx.x == 0) {
            gt_start = globaltimer();
            c_start = clock64_start();
        }
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                ku::tmem_st_32dp32bNx<1>(TMEM_COL_C + b, &data[b]);
            }
            cutlass::arch::fence_view_async_tmem_store();
        }
        if (threadIdx.x == 0) {
            uint64_t c_end = clock64_stop();
            int64_t gt_end = globaltimer();
            result->total_cycles = c_end - c_start;
            result->gt_start_ns = gt_start;
            result->gt_end_ns = gt_end;
            result->sink0 = data[0];
            result->sink1 = data[BATCH - 1];
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

template <int BATCH>
__global__ void async_sttm_kernel(BenchResult* result, int outer_iters) {
    namespace ku = kerutils;
    __shared__ __align__(16) uint32_t smem_tmem_addr;
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        uint32_t data[16 * BATCH];
        #pragma unroll
        for (int i = 0; i < 16 * BATCH; ++i) {
            data[i] = static_cast<uint32_t>(i + 1);
        }
        int64_t gt_start = 0;
        uint64_t c_start = 0;
        if (threadIdx.x == 0) {
            gt_start = globaltimer();
            c_start = clock64_start();
        }
        for (int i = 0; i < outer_iters; ++i) {
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                ku::tmem_st_32dp32bNx<16>(TMEM_COL_C + b * 16, &data[b * 16]);
            }
            cutlass::arch::fence_view_async_tmem_store();
        }
        if (threadIdx.x == 0) {
            uint64_t c_end = clock64_stop();
            int64_t gt_end = globaltimer();
            result->total_cycles = c_end - c_start;
            result->gt_start_ns = gt_start;
            result->gt_end_ns = gt_end;
            result->sink0 = data[0];
            result->sink1 = data[16 * BATCH - 1];
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

template <int BATCH>
__global__ void async_stt_baseline_kernel(BenchResult* result, int outer_iters) {
    if (threadIdx.x < 32) {
        int64_t gt_start = 0;
        uint64_t c_start = 0;
        if (threadIdx.x == 0) {
            gt_start = globaltimer();
            c_start = clock64_start();
        }
        for (int i = 0; i < outer_iters; ++i) {
            cutlass::arch::fence_view_async_tmem_store();
        }
        if (threadIdx.x == 0) {
            uint64_t c_end = clock64_stop();
            int64_t gt_end = globaltimer();
            result->total_cycles = c_end - c_start;
            result->gt_start_ns = gt_start;
            result->gt_end_ns = gt_end;
            result->sink0 = 0;
            result->sink1 = 0;
        }
    }
}

template <int BATCH>
__global__ void async_utchmma_kernel(BenchResult* result, int outer_iters) {
    using namespace cute;
    namespace ku = kerutils;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    auto layout_B = ku::make_umma_canonical_k_major_layout<64, 16, 32, bf16>();
    bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);

    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMAAtom = SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, 64, 64, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMAAtom{});
        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);
        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(partition_shape_A(tiled_mma, Shape<Int<64>, Int<16>>{}));
        tA_frag.data().get() = TMEM_COL_A;
        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<64>>{});
        auto sB_frag = thr_mma.partition_fragment_B(sB);
        uint32_t phase = 0;
        int64_t gt_start = globaltimer();
        uint64_t c_start = clock64_start();
        for (int i = 0; i < outer_iters; ++i) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            #pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                tC_frag.data().get() = TMEM_COL_C + b * 32;
                gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = phase;
        result->sink1 = 0;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

template <int BATCH>
__global__ void async_utchmma_baseline_kernel(BenchResult* result, int outer_iters) {
    using namespace cute;
    namespace ku = kerutils;

    __shared__ __align__(16) uint32_t smem_tmem_addr;
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t phase = 0;
        int64_t gt_start = globaltimer();
        uint64_t c_start = clock64_start();
        for (int i = 0; i < outer_iters; ++i) {
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }
        uint64_t c_end = clock64_stop();
        int64_t gt_end = globaltimer();
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
        result->sink0 = phase;
        result->sink1 = 0;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}
