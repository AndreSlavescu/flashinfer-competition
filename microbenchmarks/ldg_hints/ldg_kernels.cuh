#pragma once
// LDG.256 hint sweep microbenchmark kernels
// Measures ld.global[.nc].L1::[hint].L2::[hint].v4.u64 (256-bit wide loads)
// across all valid L1/L2/NC/prefetch combinations
// Target: B200 (sm100a)

#include "benchmark_common.cuh"
#include <cstdint>

// ---------------------------------------------------------------------------
// BenchResult
// ---------------------------------------------------------------------------
struct LdgResult {
    uint64_t total_cycles;
    int64_t  gt_start_ns;
    int64_t  gt_end_ns;
};

// ---------------------------------------------------------------------------
// Hint enum — each value encodes a unique (NC, L1, L2, prefetch) combo
// We use this to template-switch at compile time into string-literal macros
// ---------------------------------------------------------------------------
enum LdgHint {
    // Coherent (.nc absent) — L1 sweep with L2=evict_normal, prefetch=128B
    H_EF_EN_128B = 0,   // L1=evict_first
    H_EN_EN_128B,        // L1=evict_normal (baseline)
    H_EL_EN_128B,        // L1=evict_last
    H_EU_EN_128B,        // L1=evict_unchanged
    H_NA_EN_128B,        // L1=no_allocate
    // Coherent — L2 sweep with L1=evict_normal, prefetch=128B
    H_EN_EF_128B,        // L2=evict_first
    H_EN_EL_128B,        // L2=evict_last
    // Coherent — prefetch sweep with L1=no_allocate, L2=evict_first
    H_NA_EF_64B,         // prefetch=64B
    H_NA_EF_128B,        // prefetch=128B (competition candidate)
    H_NA_EF_256B,        // prefetch=256B
    // Non-coherent (.nc) variants
    H_NC_EF_EN_128B,     // .nc + L1=evict_first, L2=evict_normal
    H_NC_EN_EN_128B,     // .nc + L1=evict_normal (default __ldg)
    H_NC_EL_EN_128B,     // .nc + L1=evict_last
    H_NC_NA_EF_128B,     // .nc + L1=no_allocate, L2=evict_first (top candidate)
    H_NC_NA_EF_64B,      // .nc + same but 64B prefetch
    H_NC_NA_EF_256B,     // .nc + same but 256B prefetch
    // Cross combos
    H_EL_EF_128B,        // L1=evict_last, L2=evict_first
    H_NA_EL_128B,        // L1=no_allocate, L2=evict_last
    LDG_HINT_COUNT
};

// ---------------------------------------------------------------------------
// LDG.256 dispatch macro — maps hint enum to string literals
// ---------------------------------------------------------------------------
// We inline the KU_LDG_256 macro pattern directly to avoid kerutils dependency.
// The PTX is: ld.global[.nc].L1::<L1>.L2::<L2>.L2::<prefetch>.v4.u64

#define DO_LDG256(addr, result, NC_STR, L1, L2, PF) \
    { \
        uint64_t* r = (uint64_t*)(result); \
        asm volatile( \
            "ld.global" NC_STR ".L1::" L1 ".L2::" L2 ".L2::" PF ".v4.u64 {%0, %1, %2, %3}, [%4];\n" \
            : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3]) \
            : "l"(addr) \
        ); \
    }

// Dispatch helper — calls DO_LDG256 with correct string literals
// Must be a macro because PTX string literals must be compile-time
#define DISPATCH_LDG256(HINT, addr, result) \
    if constexpr (HINT == H_EF_EN_128B)      DO_LDG256(addr, result, "", "evict_first", "evict_normal", "128B") \
    else if constexpr (HINT == H_EN_EN_128B)  DO_LDG256(addr, result, "", "evict_normal", "evict_normal", "128B") \
    else if constexpr (HINT == H_EL_EN_128B)  DO_LDG256(addr, result, "", "evict_last", "evict_normal", "128B") \
    else if constexpr (HINT == H_EU_EN_128B)  DO_LDG256(addr, result, "", "evict_unchanged", "evict_normal", "128B") \
    else if constexpr (HINT == H_NA_EN_128B)  DO_LDG256(addr, result, "", "no_allocate", "evict_normal", "128B") \
    else if constexpr (HINT == H_EN_EF_128B)  DO_LDG256(addr, result, "", "evict_normal", "evict_first", "128B") \
    else if constexpr (HINT == H_EN_EL_128B)  DO_LDG256(addr, result, "", "evict_normal", "evict_last", "128B") \
    else if constexpr (HINT == H_NA_EF_64B)   DO_LDG256(addr, result, "", "no_allocate", "evict_first", "64B") \
    else if constexpr (HINT == H_NA_EF_128B)  DO_LDG256(addr, result, "", "no_allocate", "evict_first", "128B") \
    else if constexpr (HINT == H_NA_EF_256B)  DO_LDG256(addr, result, "", "no_allocate", "evict_first", "256B") \
    else if constexpr (HINT == H_NC_EF_EN_128B) DO_LDG256(addr, result, ".nc", "evict_first", "evict_normal", "128B") \
    else if constexpr (HINT == H_NC_EN_EN_128B) DO_LDG256(addr, result, ".nc", "evict_normal", "evict_normal", "128B") \
    else if constexpr (HINT == H_NC_EL_EN_128B) DO_LDG256(addr, result, ".nc", "evict_last", "evict_normal", "128B") \
    else if constexpr (HINT == H_NC_NA_EF_128B) DO_LDG256(addr, result, ".nc", "no_allocate", "evict_first", "128B") \
    else if constexpr (HINT == H_NC_NA_EF_64B)  DO_LDG256(addr, result, ".nc", "no_allocate", "evict_first", "64B") \
    else if constexpr (HINT == H_NC_NA_EF_256B) DO_LDG256(addr, result, ".nc", "no_allocate", "evict_first", "256B") \
    else if constexpr (HINT == H_EL_EF_128B)  DO_LDG256(addr, result, "", "evict_last", "evict_first", "128B") \
    else if constexpr (HINT == H_NA_EL_128B)  DO_LDG256(addr, result, "", "no_allocate", "evict_last", "128B")

// Hint name for CSV output
__host__ const char* ldg_hint_name(int hint) {
    switch (hint) {
        case H_EF_EN_128B:      return "ef_en_128B";
        case H_EN_EN_128B:      return "en_en_128B";
        case H_EL_EN_128B:      return "el_en_128B";
        case H_EU_EN_128B:      return "eu_en_128B";
        case H_NA_EN_128B:      return "na_en_128B";
        case H_EN_EF_128B:      return "en_ef_128B";
        case H_EN_EL_128B:      return "en_el_128B";
        case H_NA_EF_64B:       return "na_ef_64B";
        case H_NA_EF_128B:      return "na_ef_128B";
        case H_NA_EF_256B:      return "na_ef_256B";
        case H_NC_EF_EN_128B:   return "nc_ef_en_128B";
        case H_NC_EN_EN_128B:   return "nc_en_en_128B";
        case H_NC_EL_EN_128B:   return "nc_el_en_128B";
        case H_NC_NA_EF_128B:   return "nc_na_ef_128B";
        case H_NC_NA_EF_64B:    return "nc_na_ef_64B";
        case H_NC_NA_EF_256B:   return "nc_na_ef_256B";
        case H_EL_EF_128B:      return "el_ef_128B";
        case H_NA_EL_128B:      return "na_el_128B";
        default:                return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: Throughput — streaming sequential 256-bit loads
// Single warp (32 threads), each thread loads from contiguous addresses.
// Measures sustained load bandwidth under different hint combos.
// ---------------------------------------------------------------------------
template<int HINT, int WORKING_SET_ELEMS>
__global__ __launch_bounds__(32, 1)
void kernel_ldg_throughput(LdgResult* result, int iters,
                           const uint64_t* __restrict__ data) {
    // Each thread loads 32 bytes (4 x u64) per iteration
    // 32 threads × 32 bytes = 1024 bytes per iteration
    constexpr int BYTES_PER_LOAD = 32;
    constexpr int LOADS_PER_ITER = 1;  // 1 load per thread per iteration

    int tid = threadIdx.x;  // 0..31

    // Each of 32 threads strides through its own non-overlapping sub-region
    // of the working set so that all threads collectively touch the full WS.
    // thread k reads: data[k * (WS/32) + iter*4], wrapping at WORKING_SET_ELEMS
    uint64_t buf[4];  // 256-bit result buffer
    const int64_t per_thread = WORKING_SET_ELEMS / 32;  // u64 elements per thread

    // Warmup
    for (int i = 0; i < 100; i++) {
        int64_t offset = ((int64_t)tid * per_thread + (int64_t)i * 4) % WORKING_SET_ELEMS;
        const uint64_t* addr = data + offset;
        DISPATCH_LDG256(HINT, addr, buf);
    }

    __syncwarp();

    // All threads load in lockstep — thread 0 times
    uint64_t start = clock64();
    int64_t gt_start = globaltimer();

    for (int i = 0; i < iters; i++) {
        int64_t offset = ((int64_t)tid * per_thread + (int64_t)i * 4) % WORKING_SET_ELEMS;
        const uint64_t* addr = data + offset;
        DISPATCH_LDG256(HINT, addr, buf);
    }

    __syncwarp();
    int64_t gt_end = globaltimer();
    uint64_t end = clock64();

    if (tid == 0) {
        result->total_cycles = end - start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }

    // Prevent optimization
    if (buf[0] == 0xDEADDEADDEADDEADull) {
        volatile uint64_t* v = (volatile uint64_t*)data;
        *v = buf[1];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Competition pattern — mimics Warp 7 index loading
// 32 threads, each loads 8 LDG.256 calls = 8 × 32B = 256 bytes per thread
// Total: 32 × 256 = 8192 bytes = 64 int32 sparse_indices per block
// Indices are random scatter across ~541K token space (8462 pages × 64)
// ---------------------------------------------------------------------------
template<int HINT>
__global__ __launch_bounds__(32, 1)
void kernel_ldg_competition(LdgResult* result, int iters,
                            const uint64_t* __restrict__ index_data,
                            int total_tokens) {
    int tid = threadIdx.x;
    uint64_t buf[4];

    // Each thread loads 8 × LDG.256 = 256 bytes = 64 int32 indices.
    // This models a single warp loading all topk=2048 indices at once (full-token index prefetch).
    // Per-block reality (B_TOPK=64): only 8 threads are active, each loading 1 LDG.256 = 8 indices.
    constexpr int LOADS_PER_THREAD = 8;

    // Warmup
    for (int w = 0; w < 50; w++) {
        for (int l = 0; l < LOADS_PER_THREAD; l++) {
            // Pseudo-random scatter: use tid + l + w to vary addresses
            int token_idx = ((tid * 31 + l * 997 + w * 7919) % total_tokens);
            int word_offset = token_idx * 4;  // 4 u64 per 32-byte line
            DISPATCH_LDG256(HINT, index_data + word_offset, buf);
        }
    }

    __syncwarp();
    uint64_t start = clock64();
    int64_t gt_start = globaltimer();

    for (int i = 0; i < iters; i++) {
        for (int l = 0; l < LOADS_PER_THREAD; l++) {
            int token_idx = ((tid * 31 + l * 997 + i * 7919) % total_tokens);
            int word_offset = token_idx * 4;
            DISPATCH_LDG256(HINT, index_data + word_offset, buf);
        }
    }

    __syncwarp();
    int64_t gt_end = globaltimer();
    uint64_t end = clock64();

    if (tid == 0) {
        result->total_cycles = end - start;
        result->gt_start_ns = gt_start;
        result->gt_end_ns = gt_end;
    }

    if (buf[0] == 0xDEADDEADDEADDEADull) {
        volatile uint64_t* v = (volatile uint64_t*)index_data;
        *v = buf[1];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Latency — absolute-pointer p-chase, single thread
// Each 32-byte slot stores an absolute device pointer in buf[0] to the next
// slot, creating a pure data-dependency chain with zero ALU overhead.
// Based on gpu-benches/gpu-latency and NVIDIA-Hopper-Benchmark p-chase method.
// ---------------------------------------------------------------------------
template<int HINT>
__global__ __launch_bounds__(32, 1)
void kernel_ldg_latency(LdgResult* result, int iters,
                         const uint64_t* __restrict__ chase_start) {
    if (threadIdx.x != 0) return;

    uint64_t buf[4];
    const uint64_t* ptr = chase_start;

    // Warmup — traverse the chase ring
    for (int w = 0; w < 200; w++) {
        DISPATCH_LDG256(HINT, ptr, buf);
        ptr = (const uint64_t*)buf[0];  // absolute device pointer — zero ALU overhead
    }

    uint64_t start = clock64();
    int64_t gt_start = globaltimer();

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        DISPATCH_LDG256(HINT, ptr, buf);
        ptr = (const uint64_t*)buf[0];
    }

    int64_t gt_end = globaltimer();
    uint64_t end = clock64();

    result->total_cycles = end - start;
    result->gt_start_ns = gt_start;
    result->gt_end_ns = gt_end;

    // Anti-DCE: prevent compiler from removing the chase
    if ((uintptr_t)ptr == 0xDEADull) {
        volatile uint64_t sink = buf[1];
        (void)sink;
    }
}
