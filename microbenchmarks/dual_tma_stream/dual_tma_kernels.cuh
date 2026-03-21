#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "../common/benchmark_common.cuh"

// ---------------------------------------------------------------------------
// Standalone PTX wrappers (copied from tma_gather4/gather4_kernels.cuh)
// ---------------------------------------------------------------------------

__device__ __forceinline__
void mbarrier_init(uint64_t* mbar, uint32_t expected_count) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(mbar_addr), "r"(expected_count)
        : "memory");
}

__device__ __forceinline__
void mbarrier_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n"
        :: "r"(mbar_addr), "r"(tx_bytes)
        : "memory");
}

__device__ __forceinline__
void mbarrier_arrive(uint64_t* mbar) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .b64 _state;\n"
        "mbarrier.arrive.shared::cta.b64 _state, [%0];\n"
        "}\n"
        :: "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__
void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra WAIT_%=;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(phase)
        : "memory");
}

__device__ __forceinline__
void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__
void tma_gather4(const void* desc_ptr, uint64_t* mbar_ptr, void* smem_ptr,
                 int col_idx, int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :: "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
           "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
           "r"(mbar_addr), "l"(cache_hint)
        : "memory");
}

constexpr int64_t HINT_EVICT_LAST = 0x14F0000000000000LL;

// ---------------------------------------------------------------------------
// Dual TMA stream kernel
//
// 2 warps (64 threads): warp 0 (thread 0) = ckv stream, warp 1 (thread 32) = kpe stream.
// Each warp has its own mbarrier and smem data region; they issue TMA gather4 independently.
//
// Modes:
//   0 = CKV_ONLY  — warp 0 active, warp 1 idle
//   1 = KPE_ONLY  — warp 1 active, warp 0 idle
//   2 = BOTH      — both warps active concurrently
//
// d_results[0] = ckv elapsed ns (0 if inactive)
// d_results[1] = kpe elapsed ns (0 if inactive)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(64)
dual_tma_kernel(
    const __grid_constant__ CUtensorMap ckv_map,
    const __grid_constant__ CUtensorMap kpe_map,
    const int* __restrict__ d_ckv_indices,
    const int* __restrict__ d_kpe_indices,
    int           ckv_num_calls,
    int           kpe_num_calls,
    int           bytes_ckv,       // bytes per ckv gather4 (4096)
    int           bytes_kpe,       // bytes per kpe gather4 (512)
    int           mode,            // 0=ckv_only, 1=kpe_only, 2=both
    int64_t       cache_hint,
    int64_t*      d_results)
{
    // Shared memory layout:
    //   [ckv_data: bytes_ckv] [kpe_data: bytes_kpe] [ckv_mbar: 8B] [kpe_mbar: 8B]
    // Both data regions are naturally aligned (4096 and 512 are powers of 2).
    extern __shared__ char smem_raw[];
    char*     ckv_smem = smem_raw;
    char*     kpe_smem = smem_raw + bytes_ckv;
    // 8B-align mbarrier placement after data regions
    int mbar_offset = ((bytes_ckv + bytes_kpe) + 7) & ~7;
    uint64_t* ckv_mbar = reinterpret_cast<uint64_t*>(smem_raw + mbar_offset);
    uint64_t* kpe_mbar = ckv_mbar + 1;

    int tid = threadIdx.x;

    // Each elected thread inits its own mbarrier
    if (tid == 0)  mbarrier_init(ckv_mbar, 1);
    if (tid == 32) mbarrier_init(kpe_mbar, 1);
    // Double sync pattern: first sync ensures both mbarrier_init calls are visible to all
    // threads before the fence; fence_proxy_async establishes async proxy visibility for
    // subsequent TMA operations; second sync ensures all threads see the fence before any
    // TMA instruction is issued.
    __syncthreads();
    fence_proxy_async();
    __syncthreads();

    // Read t0 after synchronization so warp 0 and warp 1 start timing from the same
    // instruction boundary. In BOTH mode this eliminates warp-scheduling skew from t0.
    int64_t t0 = globaltimer();

    // ---- Warp 0: CKV stream (thread 0) ----
    if (tid == 0) {
        if (mode == 0 || mode == 2) {
            uint32_t phase = 0;
            for (int i = 0; i < ckv_num_calls; i++) {
                int4 rows = *reinterpret_cast<const int4*>(d_ckv_indices + i * 4);
                tma_gather4(&ckv_map, ckv_mbar, ckv_smem, 0, rows, cache_hint);
                mbarrier_expect_tx(ckv_mbar, (uint32_t)bytes_ckv);
                mbarrier_arrive(ckv_mbar);
                mbarrier_wait(ckv_mbar, phase);
                phase ^= 1;
            }
            d_results[0] = globaltimer() - t0;
        } else {
            d_results[0] = 0;
        }
    }

    // ---- Warp 1: KPE stream (thread 32) ----
    if (tid == 32) {
        if (mode == 1 || mode == 2) {
            uint32_t phase = 0;
            for (int i = 0; i < kpe_num_calls; i++) {
                int4 rows = *reinterpret_cast<const int4*>(d_kpe_indices + i * 4);
                tma_gather4(&kpe_map, kpe_mbar, kpe_smem, 0, rows, cache_hint);
                mbarrier_expect_tx(kpe_mbar, (uint32_t)bytes_kpe);
                mbarrier_arrive(kpe_mbar);
                mbarrier_wait(kpe_mbar, phase);
                phase ^= 1;
            }
            d_results[1] = globaltimer() - t0;
        } else {
            d_results[1] = 0;
        }
    }
}
