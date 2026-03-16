#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cassert>
#include <cstring>

#include "common.cuh"

// ---------------------------------------------------------------------------
// Host-side index generation for gather4 benchmarks.
//
// Each gather4 call needs 4 row indices (int4). We generate arrays of int
// on the host and copy them to device memory. The benchmark kernels read
// these arrays as int4 (4 consecutive ints per gather4 call).
//
// Total indices generated = num_gather4_calls * 4
// ---------------------------------------------------------------------------

enum class IndexPattern {
    SEQUENTIAL,                        // 0,1,2,3,4,5,...
    RANDOM,                            // uniform random over [0, num_rows)
    CLUSTERED_16,                      // pick random cluster of 16 consecutive rows, 4 random within
    CLUSTERED_64,                      // pick random cluster of 64 consecutive rows, 4 random within
    CLUSTERED_256,                     // pick random cluster of 256 consecutive rows, 4 random within
    COMPETITION_REALISTIC,             // 2048 random from num_rows, sorted within blocks of 64
    COMPETITION_REALISTIC_PAGE_TABLE,  // 2048 from 32 randomly-assigned physical page frames,
                                       // sorted within each B_TOPK=64 block by physical addr.
                                       // Models two-level block_table indirection: even sorted
                                       // topk indices map to scattered physical rows because
                                       // page frames are randomly assigned (not identity-mapped).
};

inline const char* index_pattern_name(IndexPattern p) {
    switch (p) {
        case IndexPattern::SEQUENTIAL:                       return "sequential";
        case IndexPattern::RANDOM:                           return "random";
        case IndexPattern::CLUSTERED_16:                     return "clustered_16";
        case IndexPattern::CLUSTERED_64:                     return "clustered_64";
        case IndexPattern::CLUSTERED_256:                    return "clustered_256";
        case IndexPattern::COMPETITION_REALISTIC:            return "competition_realistic";
        case IndexPattern::COMPETITION_REALISTIC_PAGE_TABLE: return "competition_realistic_pt";
    }
    return "unknown";
}

// Generate host-side index array. Returns vector of int, length = num_indices.
// num_indices should be a multiple of 4 (for gather4).
inline std::vector<int> generate_indices(
    IndexPattern pattern,
    int          num_indices,
    int          num_rows,
    uint32_t     seed = 42)
{
    assert(num_indices % 4 == 0);
    assert(num_rows > 0);
    std::vector<int> indices(num_indices);
    std::mt19937 rng(seed);

    switch (pattern) {
    case IndexPattern::SEQUENTIAL: {
        for (int i = 0; i < num_indices; i++) {
            indices[i] = i % num_rows;
        }
        break;
    }

    case IndexPattern::RANDOM: {
        std::uniform_int_distribution<int> dist(0, num_rows - 1);
        for (int i = 0; i < num_indices; i++) {
            indices[i] = dist(rng);
        }
        break;
    }

    case IndexPattern::CLUSTERED_16:
    case IndexPattern::CLUSTERED_64:
    case IndexPattern::CLUSTERED_256: {
        int cluster_size = 16;
        if (pattern == IndexPattern::CLUSTERED_64)  cluster_size = 64;
        if (pattern == IndexPattern::CLUSTERED_256) cluster_size = 256;

        int max_cluster_start = std::max(0, num_rows - cluster_size);
        std::uniform_int_distribution<int> cluster_dist(0, max_cluster_start);

        // Generate groups of 4 indices from random clusters
        for (int i = 0; i < num_indices; i += 4) {
            int cluster_start = cluster_dist(rng);
            std::uniform_int_distribution<int> within(0, cluster_size - 1);
            for (int j = 0; j < 4; j++) {
                indices[i + j] = cluster_start + within(rng);
            }
        }
        break;
    }

    case IndexPattern::COMPETITION_REALISTIC: {
        // Mimics the actual competition workload:
        //   - topk=2048 tokens selected from num_pages*page_size total tokens
        //   - B_TOPK=64: tokens processed in blocks of 64
        //   - Within each block of 64, indices are sorted (page-sorted)
        //   - Each gather4 loads 4 consecutive (sorted) tokens from a block
        //
        // We generate ceil(num_indices/64) blocks of 64 sorted indices.
        constexpr int block_size = 64;
        int num_blocks = (num_indices + block_size - 1) / block_size;

        // Pick 2048 random token indices (or num_indices if less)
        int total_tokens = std::min(num_indices, 2048);
        std::vector<int> all_tokens(total_tokens);
        std::uniform_int_distribution<int> dist(0, num_rows - 1);
        for (int i = 0; i < total_tokens; i++) {
            all_tokens[i] = dist(rng);
        }

        // Distribute into blocks and sort within each block
        for (int b = 0; b < num_blocks; b++) {
            int start = b * block_size;
            int end = std::min(start + block_size, total_tokens);
            if (start >= total_tokens) {
                // Pad remaining with valid indices (wrap around)
                for (int i = start; i < start + block_size && i < num_indices; i++) {
                    indices[i] = all_tokens[i % total_tokens];
                }
            } else {
                // Sort this block's portion
                std::sort(all_tokens.begin() + start,
                          all_tokens.begin() + end);
                for (int i = start; i < end && i < num_indices; i++) {
                    indices[i] = all_tokens[i];
                }
                // Pad if block is incomplete
                for (int i = end; i < start + block_size && i < num_indices; i++) {
                    indices[i] = all_tokens[end - 1];
                }
            }
        }
        break;
    }

    case IndexPattern::COMPETITION_REALISTIC_PAGE_TABLE: {
        // Models the competition kernel with actual two-level page-table indirection:
        //   physical_row = block_table[logical_page] * page_size + token_offset
        //
        // Unlike COMPETITION_REALISTIC (which sorts 2048 random token indices, giving
        // sequential-like locality if frames are identity-mapped), this pattern assigns
        // 32 random physical page frames to the 32 logical pages. Even after sorting
        // within each B_TOPK=64 block, adjacent blocks jump to entirely different physical
        // locations — matching the real workload where page frames are randomly allocated.
        //
        // Within each B_TOPK block (= one logical page): all 64 rows come from the same
        // physical frame → perfectly sequential, excellent L1/L2 locality per block.
        // Between blocks: base addresses differ by (frame_gap × page_size) rows → scattered.
        //
        // Parameters matching competition workload:
        //   topk=2048 tokens, page_size=64 tokens/page, B_TOPK=64
        constexpr int topk      = 2048;
        constexpr int page_size = 64;
        constexpr int B_TOPK    = 64;

        int num_physical_pages  = (int)(num_rows / page_size);
        int num_logical_pages   = topk / page_size;  // = 32

        if (num_physical_pages < num_logical_pages) {
            // Tensor too small for page-table simulation; fall back to random
            std::uniform_int_distribution<int> dist(0, num_rows - 1);
            for (int i = 0; i < num_indices; i++) indices[i] = dist(rng);
            break;
        }

        // Build block_table: shuffle physical pages and pick the first num_logical_pages
        std::vector<int> all_phys_pages(num_physical_pages);
        std::iota(all_phys_pages.begin(), all_phys_pages.end(), 0);
        std::shuffle(all_phys_pages.begin(), all_phys_pages.end(), rng);
        std::vector<int> block_table(all_phys_pages.begin(),
                                     all_phys_pages.begin() + num_logical_pages);

        // Generate topk physical row indices: for logical page b, all page_size rows
        // come from physical frame block_table[b] (addresses are contiguous within frame)
        std::vector<int> topk_phys_rows(topk);
        for (int b = 0; b < num_logical_pages; b++) {
            int phys_base = block_table[b] * page_size;
            for (int off = 0; off < page_size; off++) {
                topk_phys_rows[b * page_size + off] = phys_base + off;
            }
        }

        // Sort within each B_TOPK block (as the production kernel does for TMA locality).
        // Since each B_TOPK block = one physical page, rows within a block are already
        // sequential; the sort is a no-op but mirrors the kernel's pre-sort step.
        int num_topk_blocks = topk / B_TOPK;
        for (int blk = 0; blk < num_topk_blocks; blk++) {
            std::sort(topk_phys_rows.begin() + blk * B_TOPK,
                      topk_phys_rows.begin() + (blk + 1) * B_TOPK);
        }

        // Tile the topk physical rows across the index array
        for (int i = 0; i < num_indices; i++) {
            indices[i] = topk_phys_rows[i % topk];
        }
        break;
    }
    }

    return indices;
}

// Allocate device-side index buffer and copy from host
inline int* allocate_device_indices(const std::vector<int>& host_indices) {
    int* d_indices = nullptr;
    size_t bytes = host_indices.size() * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_indices, bytes));
    CUDA_CHECK(cudaMemcpy(d_indices, host_indices.data(), bytes,
                          cudaMemcpyHostToDevice));
    return d_indices;
}

// Convenience: generate + upload in one call
inline int* create_device_indices(
    IndexPattern pattern,
    int          num_indices,
    int          num_rows,
    uint32_t     seed = 42)
{
    auto h = generate_indices(pattern, num_indices, num_rows, seed);
    return allocate_device_indices(h);
}
