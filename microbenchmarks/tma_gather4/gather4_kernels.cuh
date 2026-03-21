#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "../common/benchmark_common.cuh"

// ---------------------------------------------------------------------------
// Standalone PTX wrappers — no CuTe/CUTLASS dependency
// Adapted from csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh
// ---------------------------------------------------------------------------

// Raw mbarrier PTX operations
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

// Wait for mbarrier phase to complete. phase=0 for first use, toggles.
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

// Fence to ensure shared memory writes are visible
__device__ __forceinline__
void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ---------------------------------------------------------------------------
// TMA gather4 — the instruction under test
// ---------------------------------------------------------------------------
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

// TMA gather4 prefetch — fire-and-forget L2 prefetch, no mbarrier needed.
// Issues cp.async.bulk.prefetch into L2 for 4 rows; use col_idx=0 for single-step configs
// (ckv_int64 and kpe_bf16 both have exactly 1 column step).
// Adapted from csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh:26
__device__ __forceinline__
void tma_gather4_prefetch(const void* desc_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint"
        " [%0, {%1, %2, %3, %4, %5}], %6;\n"
        :: "l"(desc_ptr), "r"(col_idx),
           "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
           "l"(cache_hint)
        : "memory");
}

// Cache hint constants from cutlass/include/cute/arch/copy_sm90_desc.hpp
constexpr int64_t HINT_EVICT_NORMAL = 0x1000000000000000LL;
constexpr int64_t HINT_EVICT_FIRST  = 0x12F0000000000000LL;
constexpr int64_t HINT_EVICT_LAST   = 0x14F0000000000000LL;
constexpr int64_t HINT_NONE         = 0x0LL;

inline const char* cache_hint_name(int64_t hint) {
    if (hint == HINT_EVICT_NORMAL) return "evict_normal";
    if (hint == HINT_EVICT_FIRST)  return "evict_first";
    if (hint == HINT_EVICT_LAST)   return "evict_last";
    if (hint == HINT_NONE)         return "none";
    return "unknown";
}

// ===================================================================
// KERNEL 1: Throughput benchmark
//
// Single elected thread per CTA issues gather4 in a tight loop.
// Walks through all indices, issuing one gather4 at a time, waiting
// on barrier after each. Measures total time across all CTAs.
//
// Launch: <<<num_blocks, 1, smem_bytes>>>
// smem_bytes = max_bytes_per_gather4 + 16 (for mbarrier)
// ===================================================================


__global__ void __launch_bounds__(1)
throughput_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    const int*    __restrict__ d_indices,   // [num_gather4_calls * 4]
    int           num_gather4_calls,
    int           col_steps,        // number of col steps per row group
    int           box_dim0,         // elements per col step (for col_idx coordinate)
    int           bytes_per_step,   // box_dim0 * elem_size * 4 rows
    int           smem_data_size,   // total data region = bytes_per_step * col_steps
    int64_t       cache_hint,
    int64_t*      d_timer_buf)      // [num_blocks * 3]
{
    extern __shared__ char smem_raw[];
    // Partition: [data region (smem_data_size bytes)][8B-aligned mbarrier]
    char*     smem_data = smem_raw;
    uint64_t* mbar      = reinterpret_cast<uint64_t*>(smem_raw + smem_data_size);

    KernelTimer timer;
    timer.init(d_timer_buf, blockIdx.x);

    // Initialize mbarrier for 1 arriving thread
    mbarrier_init(mbar, 1);
    fence_proxy_async();

    int calls_per_cta = (num_gather4_calls + gridDim.x - 1) / gridDim.x;
    int my_start      = blockIdx.x * calls_per_cta;
    int my_end        = min(my_start + calls_per_cta, num_gather4_calls);

    // Phase toggles: 0, 1, 0, 1, ...
    uint32_t phase = 0;

    timer.start();

    for (int i = my_start; i < my_end; i++) {
        int4 row_idxs = *reinterpret_cast<const int4*>(d_indices + i * 4);

        for (int col = 0; col < col_steps; col++) {
            tma_gather4(&tensor_map, mbar,
                        smem_data + col * bytes_per_step,
                        col * box_dim0,  // col_idx in element coordinates
                        row_idxs, cache_hint);
        }

        // Signal expected bytes and wait.
        // Per PTX ISA: mbarrier.expect_tx may be called AFTER the async ops that will fulfill
        // it, as long as it precedes the try_wait. The TMA engine accumulates transaction
        // completion credits independently; expect_tx declares the total expected bytes against
        // which those credits are checked at try_wait time. Issuing expect_tx after the gather4
        // calls (rather than before) avoids a potential race where expect_tx could be observed
        // by the mbarrier before the gather4 has issued its completion credit.
        uint32_t total_bytes = bytes_per_step * col_steps;
        mbarrier_expect_tx(mbar, total_bytes);
        mbarrier_arrive(mbar);
        mbarrier_wait(mbar, phase);
        phase ^= 1;
    }

    timer.stop();
}


// ===================================================================
// KERNEL 2: Latency benchmark (single CTA)
//
// Measures per-gather4 round-trip latency: issue → expect_tx → arrive
// → wait → timestamp. Runs many iterations and averages.
//
// Also provides a null baseline variant (just barrier ops, no gather4)
// to isolate barrier overhead.
//
// Launch: <<<1, 1, smem_bytes>>>
// ===================================================================

__global__ void __launch_bounds__(1)
latency_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    const int*    __restrict__ d_indices,
    int           num_iters,
    int           bytes_per_gather4,    // total bytes per single gather4 call
    int64_t       cache_hint,
    int64_t*      d_results)            // [2]: total_cycles, null_baseline_cycles
{
    extern __shared__ char smem_raw[];
    char*     smem_data = smem_raw;
    uint64_t* mbar      = reinterpret_cast<uint64_t*>(smem_raw + bytes_per_gather4);

    mbarrier_init(mbar, 1);
    fence_proxy_async();

    // --- Gather4 latency ---
    uint32_t phase = 0;
    int64_t t_start = globaltimer();

    for (int i = 0; i < num_iters; i++) {
        int idx = (i * 4) % (num_iters * 4);  // wrap around indices
        int4 row_idxs = *reinterpret_cast<const int4*>(d_indices + idx);

        tma_gather4(&tensor_map, mbar, smem_data, 0, row_idxs, cache_hint);
        mbarrier_expect_tx(mbar, bytes_per_gather4);
        mbarrier_arrive(mbar);
        mbarrier_wait(mbar, phase);
        phase ^= 1;
    }

    int64_t t_gather4 = globaltimer() - t_start;

    // --- Null baseline: barrier ops only, no gather4 ---
    // Re-init barrier
    mbarrier_init(mbar, 1);
    fence_proxy_async();
    phase = 0;

    int64_t t_null_start = globaltimer();

    for (int i = 0; i < num_iters; i++) {
        // Same barrier overhead but no TMA instruction
        mbarrier_expect_tx(mbar, 0);
        mbarrier_arrive(mbar);
        mbarrier_wait(mbar, phase);
        phase ^= 1;
    }

    int64_t t_null = globaltimer() - t_null_start;

    d_results[0] = t_gather4;
    d_results[1] = t_null;
}


// ===================================================================
// KERNEL 2b: Prefetch-latency benchmark (single CTA)
//
// Issues tma_gather4_prefetch() DIST steps ahead of each gather4 call.
// Prolog: pre-issue first DIST prefetches before the timed loop begins.
//
// Goal: determine whether L2 prefetch hides the 546 ns HBM-cold TMA latency.
// Best used with SEQUENTIAL pattern + 1024MB working set + num_iters=num_rows/4
// so every row is unique (no L2 re-warming between iterations).
//
// Launch: <<<1, 1, smem_bytes>>>  where smem_bytes = bytes_per_gather4 + 16
// d_results[0] = total nanoseconds for num_iters gather4 calls (timed)
// d_results[1] = 0 (no null baseline)
// ===================================================================

template<int DIST>
__global__ void __launch_bounds__(1)
prefetch_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    const int*    __restrict__ d_indices,
    int           num_iters,
    int           bytes_per_gather4,
    int64_t       cache_hint,
    int64_t*      d_results)
{
    extern __shared__ char smem_raw[];
    char*     smem_data = smem_raw;
    uint64_t* mbar      = reinterpret_cast<uint64_t*>(smem_raw + bytes_per_gather4);

    mbarrier_init(mbar, 1);
    fence_proxy_async();

    // Prolog: pre-issue DIST prefetches before the timed section.
    // These prefetches run concurrently with the timing setup overhead.
    #pragma unroll
    for (int d = 0; d < DIST; d++) {
        if (d < num_iters) {
            int4 rows = *reinterpret_cast<const int4*>(d_indices + d * 4);
            tma_gather4_prefetch(&tensor_map, 0, rows, cache_hint);
        }
    }

    uint32_t phase = 0;
    int64_t t_start = globaltimer();

    for (int i = 0; i < num_iters; i++) {
        int4 rows = *reinterpret_cast<const int4*>(d_indices + i * 4);
        tma_gather4(&tensor_map, mbar, smem_data, 0, rows, cache_hint);

        // Issue prefetch for rows[i+DIST] while we wait for rows[i]
        if (i + DIST < num_iters) {
            int4 pf_rows = *reinterpret_cast<const int4*>(d_indices + (i + DIST) * 4);
            tma_gather4_prefetch(&tensor_map, 0, pf_rows, cache_hint);
        }

        mbarrier_expect_tx(mbar, bytes_per_gather4);
        mbarrier_arrive(mbar);
        mbarrier_wait(mbar, phase);
        phase ^= 1;
    }

    d_results[0] = globaltimer() - t_start;
    d_results[1] = 0;
}

typedef void (*prefetch_kernel_fn_t)(
    const __grid_constant__ CUtensorMap,
    const int*, int, int, int64_t, int64_t*);

inline prefetch_kernel_fn_t get_prefetch_kernel(int dist) {
    switch (dist) {
        case 0: return prefetch_kernel<0>;
        case 1: return prefetch_kernel<1>;
        case 2: return prefetch_kernel<2>;
        case 4: return prefetch_kernel<4>;
        default:
            fprintf(stderr, "Unsupported prefetch DIST=%d\n", dist);
            exit(1);
    }
}


// ===================================================================
// KERNEL 3: Pipeline depth benchmark (single CTA)
//
// Issues N gather4 calls to different smem offsets before a single
// barrier wait. Measures if batching amortizes barrier overhead.
//
// Launch: <<<1, 1, N * bytes_per_gather4 + 16>>>
// ===================================================================

// Template on N to allow compile-time unrolling
template<int N_OUTSTANDING>
__global__ void __launch_bounds__(1)
pipeline_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    const int*    __restrict__ d_indices,
    int           num_outer_iters,      // how many full batches of N
    int           bytes_per_gather4,
    int64_t       cache_hint,
    int64_t*      d_timer_buf)
{
    extern __shared__ char smem_raw[];
    // Layout: [N_OUTSTANDING * bytes_per_gather4 for data] [16 bytes for mbar]
    char*     smem_data = smem_raw;
    uint64_t* mbar      = reinterpret_cast<uint64_t*>(
        smem_raw + N_OUTSTANDING * bytes_per_gather4);

    KernelTimer timer;
    timer.init(d_timer_buf, 0);

    mbarrier_init(mbar, 1);
    fence_proxy_async();

    uint32_t phase = 0;
    int idx_offset = 0;

    timer.start();

    for (int outer = 0; outer < num_outer_iters; outer++) {
        // Issue N gather4 calls
        #pragma unroll
        for (int n = 0; n < N_OUTSTANDING; n++) {
            int4 row_idxs = *reinterpret_cast<const int4*>(
                d_indices + ((idx_offset + n) * 4) % (num_outer_iters * N_OUTSTANDING * 4));
            tma_gather4(&tensor_map, mbar,
                        smem_data + n * bytes_per_gather4,
                        0, row_idxs, cache_hint);
        }

        // Single barrier wait for all N
        uint32_t total_bytes = N_OUTSTANDING * bytes_per_gather4;
        mbarrier_expect_tx(mbar, total_bytes);
        mbarrier_arrive(mbar);
        mbarrier_wait(mbar, phase);
        phase ^= 1;
        idx_offset += N_OUTSTANDING;
    }

    timer.stop();
}


// ===================================================================
// KERNEL 4: Multi-CTA saturation benchmark
//
// Same as throughput_kernel but specifically designed for sweeping
// num_blocks to find saturation point. Identical kernel, but we
// separate it for clarity in the dispatch code.
// ===================================================================

// Reuses throughput_kernel directly — see main.cu dispatch.


// ===================================================================
// Helper: dispatch pipeline kernel by template parameter N
// ===================================================================

typedef void (*pipeline_kernel_fn_t)(
    const __grid_constant__ CUtensorMap,
    const int*, int, int, int64_t, int64_t*);

inline pipeline_kernel_fn_t get_pipeline_kernel(int n_outstanding) {
    switch (n_outstanding) {
        case  1: return pipeline_kernel<1>;
        case  2: return pipeline_kernel<2>;
        case  4: return pipeline_kernel<4>;
        case  8: return pipeline_kernel<8>;
        case 16: return pipeline_kernel<16>;
        case 32: return pipeline_kernel<32>;
        default:
            fprintf(stderr, "Unsupported N_OUTSTANDING=%d\n", n_outstanding);
            exit(1);
    }
}
