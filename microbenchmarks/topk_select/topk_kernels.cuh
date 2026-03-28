#pragma once

// Top-K selection microbenchmark kernels for B200 (sm_100a)
//
// Competition indexer: select top-2048 from F32 score arrays of size
// seq_len ∈ [1..5824], batch_size ∈ [1..31].
//
// Three algorithms:
// 1. CUB DeviceRadixSort (full sort + truncate)
// 2. Block-level warp bitonic top-K
// 3. Radix select (histogram-based, O(n))

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "benchmark_common.cuh"

// ---------------------------------------------------------------------------
// Result buffer
// ---------------------------------------------------------------------------
struct TopkResult {
    uint64_t total_cycles;
    int64_t  gt_start_ns;
    int64_t  gt_end_ns;
};

// =========================================================================
// Algorithm 1: CUB DeviceRadixSort
// Full descending sort of (key=score, value=index) pairs, take first topk.
// This is the simplest baseline — measures CUB sort overhead.
// =========================================================================
struct CubSortTopK {
    // Host-side function that allocates temp storage and calls CUB sort
    static void run(
        float* d_scores_in,     // [seq_len] input scores
        float* d_scores_out,    // [seq_len] sorted scores output
        int32_t* d_indices_in,  // [seq_len] input indices (0..seq_len-1)
        int32_t* d_indices_out, // [seq_len] sorted indices output
        int seq_len,
        void* d_temp_storage,
        size_t temp_storage_bytes)
    {
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_scores_in, d_scores_out,
            d_indices_in, d_indices_out,
            seq_len);
    }

    // Query temp storage size
    static size_t get_temp_storage_bytes(int seq_len) {
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            (float*)nullptr, (float*)nullptr,
            (int32_t*)nullptr, (int32_t*)nullptr,
            seq_len);
        return temp_bytes;
    }
};

// =========================================================================
// Algorithm 2: Block-level bitonic top-K
// Single CTA per batch element. Maintains top-K candidates in shared memory
// using a warp-level bitonic merge network.
//
// Strategy for seq_len=5824, topk=2048:
//   - 256 threads, each responsible for 8 top-K elements (2048/256)
//   - Stream through input in tiles of 256 (one element per thread)
//   - After each tile, merge new elements into per-thread top-K buffer
//   - Final reduction: bitonic sort the top-K buffer
//
// This is a simplified version — production would use CUB block radix sort.
// =========================================================================
template<int BLOCK_SIZE, int TOPK>
__global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel_block_bitonic_topk(
    const float* __restrict__ scores,   // [seq_len]
    int32_t* __restrict__ out_indices,   // [topk]
    float* __restrict__ out_scores,      // [topk]
    int seq_len,
    TopkResult* result,
    int iters)
{
    // Each thread maintains TOPK/BLOCK_SIZE candidates
    constexpr int ITEMS_PER_THREAD = TOPK / BLOCK_SIZE;
    static_assert(TOPK % BLOCK_SIZE == 0, "TOPK must be divisible by BLOCK_SIZE");
    static_assert(ITEMS_PER_THREAD >= 1, "Need at least 1 item per thread");

    const int tid = threadIdx.x;

    // Per-thread top-K buffer (score, index pairs)
    float my_scores[ITEMS_PER_THREAD];
    int32_t my_indices[ITEMS_PER_THREAD];

    // Shared memory for parallel reduction
    __shared__ float smem_scores[TOPK];
    __shared__ int32_t smem_indices[TOPK];

    // Warmup
    for (int w = 0; w < 5; w++) {
        // Initialize with -inf
        for (int j = 0; j < ITEMS_PER_THREAD; j++) {
            my_scores[j] = -1e30f;
            my_indices[j] = -1;
        }

        // Stream through input
        for (int base = 0; base < seq_len; base += BLOCK_SIZE) {
            int idx = base + tid;
            float val = (idx < seq_len) ? scores[idx] : -1e30f;

            // Simple insertion: if val > min of my buffer, replace min
            // Find min in my buffer
            int min_pos = 0;
            float min_val = my_scores[0];
            for (int j = 1; j < ITEMS_PER_THREAD; j++) {
                if (my_scores[j] < min_val) {
                    min_val = my_scores[j];
                    min_pos = j;
                }
            }
            if (val > min_val) {
                my_scores[min_pos] = val;
                my_indices[min_pos] = idx;
            }
        }
    }

    // --- Timed region ---
    __syncthreads();
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_start) :: "memory");
    }

    for (int iter = 0; iter < iters; iter++) {
        // Initialize with -inf
        for (int j = 0; j < ITEMS_PER_THREAD; j++) {
            my_scores[j] = -1e30f;
            my_indices[j] = -1;
        }

        // Stream through input, maintaining per-thread top-K
        for (int base = 0; base < seq_len; base += BLOCK_SIZE) {
            int idx = base + tid;
            float val = (idx < seq_len) ? scores[idx] : -1e30f;

            // Simple insertion: replace minimum if new value is larger
            int min_pos = 0;
            float min_val = my_scores[0];
            for (int j = 1; j < ITEMS_PER_THREAD; j++) {
                if (my_scores[j] < min_val) {
                    min_val = my_scores[j];
                    min_pos = j;
                }
            }
            if (val > min_val) {
                my_scores[min_pos] = val;
                my_indices[min_pos] = idx;
            }
        }

        // Write per-thread results to shared memory
        for (int j = 0; j < ITEMS_PER_THREAD; j++) {
            smem_scores[tid * ITEMS_PER_THREAD + j] = my_scores[j];
            smem_indices[tid * ITEMS_PER_THREAD + j] = my_indices[j];
        }
        __syncthreads();
    }

    uint64_t c_end = 0;
    int64_t gt_end = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_end) :: "memory");
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }

    // Write output (last iteration's result)
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int out_idx = tid * ITEMS_PER_THREAD + j;
        if (out_idx < TOPK) {
            out_scores[out_idx] = smem_scores[out_idx];
            out_indices[out_idx] = smem_indices[out_idx];
        }
    }
}

// =========================================================================
// Algorithm 3: Radix select (histogram-based threshold finding)
// Two-pass approach:
//   Pass 1: Build 256-bin histograms on float bits (MSB to LSB), digit by digit,
//           to find the threshold value at rank K.
//   Pass 2: Select all elements >= threshold.
// O(n) work per pass, no full sort needed.
// =========================================================================
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel_radix_select(
    const float* __restrict__ scores,   // [seq_len]
    int32_t* __restrict__ out_indices,   // [topk]
    int seq_len,
    int topk,
    TopkResult* result,
    int iters)
{
    const int tid = threadIdx.x;

    // Shared histogram (256 bins per radix digit)
    __shared__ int histogram[256];
    __shared__ int threshold_digit[4];  // One byte per pass (4 passes for 32-bit float)
    __shared__ int count_above;

    // Warmup
    for (int w = 0; w < 5; w++) {
        // Convert float to sortable uint32: flip sign bit, flip all if negative
        // This gives a monotonic mapping from float to uint32

        // Pass through 4 digits (MSB to LSB) to find threshold
        uint32_t prefix_mask = 0;
        uint32_t prefix_val = 0;

        for (int pass = 3; pass >= 0; pass--) {
            int shift = pass * 8;

            // Clear histogram
            if (tid < 256) histogram[tid] = 0;
            __syncthreads();

            // Build histogram for this digit, considering only elements matching prefix
            for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
                float f = scores[i];
                // Float to sortable uint: XOR trick
                uint32_t u;
                memcpy(&u, &f, 4);
                u ^= (-(u >> 31)) | 0x80000000u;

                // Check prefix match
                if ((u & prefix_mask) == prefix_val) {
                    int digit = (u >> shift) & 0xFF;
                    atomicAdd(&histogram[digit], 1);
                }
            }
            __syncthreads();

            // Scan from top to find threshold digit
            if (tid == 0) {
                int cumsum = 0;
                int thresh_d = 255;
                for (int d = 255; d >= 0; d--) {
                    cumsum += histogram[d];
                    if (cumsum >= topk) {
                        thresh_d = d;
                        break;
                    }
                }
                threshold_digit[pass] = thresh_d;
            }
            __syncthreads();

            // Update prefix for next pass
            prefix_mask |= (0xFFu << shift);
            prefix_val |= ((uint32_t)threshold_digit[pass] << shift);
            __syncthreads();
        }
    }

    // --- Timed region ---
    __syncthreads();
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_start) :: "memory");
    }

    for (int iter = 0; iter < iters; iter++) {
        uint32_t prefix_mask = 0;
        uint32_t prefix_val = 0;

        for (int pass = 3; pass >= 0; pass--) {
            int shift = pass * 8;

            if (tid < 256) histogram[tid] = 0;
            __syncthreads();

            for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
                float f = scores[i];
                uint32_t u;
                memcpy(&u, &f, 4);
                u ^= (-(u >> 31)) | 0x80000000u;

                if ((u & prefix_mask) == prefix_val) {
                    int digit = (u >> shift) & 0xFF;
                    atomicAdd(&histogram[digit], 1);
                }
            }
            __syncthreads();

            if (tid == 0) {
                int cumsum = 0;
                int thresh_d = 255;
                for (int d = 255; d >= 0; d--) {
                    cumsum += histogram[d];
                    if (cumsum >= topk) {
                        thresh_d = d;
                        break;
                    }
                }
                threshold_digit[pass] = thresh_d;
            }
            __syncthreads();

            prefix_mask |= (0xFFu << shift);
            prefix_val |= ((uint32_t)threshold_digit[pass] << shift);
            __syncthreads();
        }

        // Pass 2: collect elements >= threshold
        if (tid == 0) count_above = 0;
        __syncthreads();

        // Convert threshold back to float
        uint32_t thresh_u = prefix_val;
        // Reverse the float-to-sortable transform
        thresh_u ^= ((~thresh_u >> 31) - 1) | 0x80000000u;
        float threshold;
        memcpy(&threshold, &thresh_u, 4);

        for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
            if (scores[i] >= threshold) {
                int pos = atomicAdd(&count_above, 1);
                if (pos < topk) {
                    out_indices[pos] = i;
                }
            }
        }
        __syncthreads();
    }

    uint64_t c_end = 0;
    int64_t gt_end = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_end) :: "memory");
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }
}

// =========================================================================
// Algorithm 3 v2: Radix select — per-warp SMEM histograms (no atomics in hot path)
//
// Fix for kernel_radix_select: replaces single shared histogram with per-warp
// rows, eliminating 256-way atomicAdd contention in the histogram phase.
//
// SMEM: warp_hist[NUM_WARPS][256] (8 KB) + final_hist[256] (1 KB)
// =========================================================================
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel_radix_select_v2(
    const float* __restrict__ scores,   // [seq_len]
    int32_t* __restrict__ out_indices,   // [topk]
    int seq_len,
    int topk,
    TopkResult* result,
    int iters)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;   // tid / 32
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");

    // Per-warp histograms: warp w owns row warp_hist[w] (no atomics)
    __shared__ int warp_hist[NUM_WARPS][256];  // NUM_WARPS*256 ints = 8 KB for 8 warps
    __shared__ int final_hist[256];             // 1 KB
    __shared__ int threshold_digit[4];
    __shared__ int count_above;

    // Run one full pass set (warmup or timed iteration)
    // Returns the final prefix_val (sortable uint32 of the threshold)
    // ----------------------------------------------------------------
    auto do_radix_passes = [&]() -> uint32_t {
        uint32_t prefix_mask = 0;
        uint32_t prefix_val  = 0;

        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;

            // Step 1: Clear warp histograms — thread tid clears its column in all warp rows
            for (int w = 0; w < NUM_WARPS; w++) warp_hist[w][tid] = 0;
            __syncthreads();

            // Step 2: Build per-warp histogram — no atomics, each warp owns its row
            for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
                float f = scores[i];
                uint32_t u;
                memcpy(&u, &f, 4);
                u ^= (-(u >> 31)) | 0x80000000u;  // Float to monotonic uint32
                if ((u & prefix_mask) == prefix_val) {
                    warp_hist[warp_id][(u >> shift) & 0xFF]++;
                }
            }
            __syncthreads();

            // Step 3: Reduce warp rows — thread tid sums bin tid across all warps
            int sum = 0;
            for (int w = 0; w < NUM_WARPS; w++) sum += warp_hist[w][tid];
            final_hist[tid] = sum;
            __syncthreads();

            // Step 4: Thread 0 scans from top to find threshold digit
            if (tid == 0) {
                int cumsum = 0, thresh_d = 255;
                for (int d = 255; d >= 0; d--) {
                    cumsum += final_hist[d];
                    if (cumsum >= topk) { thresh_d = d; break; }
                }
                threshold_digit[pass] = thresh_d;
            }
            __syncthreads();

            // Update prefix for next pass
            prefix_mask |= (0xFFu << shift);
            prefix_val  |= ((uint32_t)threshold_digit[pass] << shift);
            __syncthreads();
        }
        return prefix_val;
    };

    // Warmup
    for (int w = 0; w < 5; w++) do_radix_passes();

    // --- Timed region ---
    __syncthreads();
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_start) :: "memory");
    }

    uint32_t threshold_u = 0;
    for (int iter = 0; iter < iters; iter++) {
        threshold_u = do_radix_passes();

        // Pass 2: collect elements >= threshold
        // Compare in sortable uint32 space (avoids the buggy float reverse transform)
        if (tid == 0) count_above = 0;
        __syncthreads();

        for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
            float f = scores[i];
            uint32_t u;
            memcpy(&u, &f, 4);
            u ^= (-(u >> 31)) | 0x80000000u;
            if (u >= threshold_u) {
                int pos = atomicAdd(&count_above, 1);
                if (pos < topk) out_indices[pos] = i;
            }
        }
        __syncthreads();
    }

    uint64_t c_end = 0;
    int64_t gt_end = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_end) :: "memory");
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }
}

// =========================================================================
// Algorithm 4: CUB BlockRadixSort (SMEM-resident, avoids global memory round-trips)
//
// CUB BlockRadixSort collectively sorts BLOCK_SIZE * ITEMS_PER_THREAD elements
// entirely within SMEM. Output in "blocked" arrangement: thread t holds items
// at global sorted positions [t*IPT, (t+1)*IPT) in descending order.
//
// ITEMS_PER_THREAD must be a compile-time constant:
//   seq_len=2048 → IPT=8  (256×8  = 2048)
//   seq_len=4096 → IPT=16 (256×16 = 4096)
//   seq_len=5824 → IPT=23 (256×23 = 5888, pad last 64 with -inf)
// =========================================================================
template<int BLOCK_SIZE, int ITEMS_PER_THREAD, int TOPK>
__global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel_cub_block_sort(
    const float* __restrict__ scores,   // [seq_len]
    int32_t* __restrict__ out_indices,   // [topk]
    int seq_len,
    TopkResult* result,
    int iters)
{
    static_assert(BLOCK_SIZE * ITEMS_PER_THREAD >= TOPK, "CAPACITY must be >= TOPK");
    typedef cub::BlockRadixSort<uint32_t, BLOCK_SIZE, ITEMS_PER_THREAD, int32_t> BlockSort;
    __shared__ typename BlockSort::TempStorage temp_storage;

    const int tid = threadIdx.x;
    uint32_t keys[ITEMS_PER_THREAD];
    int32_t  vals[ITEMS_PER_THREAD];

    // Load items in blocked arrangement: thread t owns positions [t*IPT, (t+1)*IPT)
    auto load_items = [&]() {
        for (int j = 0; j < ITEMS_PER_THREAD; j++) {
            int idx = tid * ITEMS_PER_THREAD + j;
            float score = (idx < seq_len) ? scores[idx] : -1e30f;
            uint32_t u;
            memcpy(&u, &score, 4);
            u ^= (-(u >> 31)) | 0x80000000u;  // Float to monotonic uint32 (ascending)
            keys[j] = u;
            vals[j] = idx;
        }
    };

    // Warmup
    for (int w = 0; w < 5; w++) {
        load_items();
        __syncthreads();
        BlockSort(temp_storage).SortDescending(keys, vals);
        __syncthreads();
    }

    // --- Timed region ---
    __syncthreads();
    int64_t gt_start = 0;
    uint64_t c_start = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_start) :: "memory");
    }

    for (int iter = 0; iter < iters; iter++) {
        load_items();
        __syncthreads();
        BlockSort(temp_storage).SortDescending(keys, vals);
        __syncthreads();
    }

    uint64_t c_end = 0;
    int64_t gt_end = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_end) :: "memory");
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");
        result->total_cycles = c_end - c_start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }

    // Write top-TOPK to output.
    // After SortDescending (CUB 3.x API), thread 0 holds the elements with the
    // highest float scores. Thread 0, slot 0 = globally highest.
    int start = tid * ITEMS_PER_THREAD;
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        if (start + j < TOPK) out_indices[start + j] = vals[j];
    }
}
