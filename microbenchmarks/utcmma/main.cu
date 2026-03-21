#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "../common/benchmark_common.cuh"
#include "utcmma_kernels.cuh"
#include "utcmma_kernels_ver2.cuh"

// ---------------------------------------------------------------------------
// CSV header
// ---------------------------------------------------------------------------
static void print_csv_header() {
    printf("experiment,x_var,M,N,k_depth,mode,swizzle,iters,"
           "total_cycles,cycles_per_gemm,cycles_per_ktile,"
           "ns_per_gemm,tflops,spread_pct\n");
}

// ---------------------------------------------------------------------------
// Run a kernel multiple times, collect trimmed-mean statistics
// ---------------------------------------------------------------------------
template<typename KernelFunc>
static void run_and_report(
    const char* experiment,
    const char* x_var,
    int M, int N, int k_depth,
    const char* mode,
    const char* swizzle,
    int iters,
    KernelFunc kernel_fn,
    int smem_bytes,
    int num_runs,
    bool verbose)
{
    BenchResult* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(BenchResult)));

    MeasurementSeries cycles_series;
    MeasurementSeries ns_series;

    for (int run = 0; run < num_runs; run++) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(BenchResult)));

        kernel_fn(d_result, iters, smem_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());

        BenchResult h_result;
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(BenchResult), cudaMemcpyDeviceToHost));

        cycles_series.add((double)h_result.total_cycles);
        ns_series.add((double)(h_result.gt_end_ns - h_result.gt_start_ns));

        if (verbose) {
            fprintf(stderr, "  run %d/%d: %lu cycles, %.1f ns (globaltimer)\n",
                    run + 1, num_runs, h_result.total_cycles,
                    (double)(h_result.gt_end_ns - h_result.gt_start_ns));
            fflush(stderr);
        }
    }

    double avg_total_cycles = cycles_series.value();
    double avg_total_ns = ns_series.value();

    int k_tiles = (k_depth > 0) ? k_depth : 1;
    double cycles_per_gemm = avg_total_cycles / iters;
    double cycles_per_ktile = cycles_per_gemm / k_tiles;
    double ns_per_gemm = avg_total_ns / iters;

    // Compute TFLOPS: M × N × K × 2 (FMA = 2 FLOPs) per K-tile, k_tiles per GEMM
    double flops_per_gemm = (double)M * N * 16 * 2 * k_tiles;
    double tflops = (flops_per_gemm * iters) / (avg_total_ns * 1e-9) / 1e12;

    double spread = cycles_series.spread() * 100.0;

    printf("%s,%s,%d,%d,%d,%s,%s,%d,"
           "%.0f,%.1f,%.2f,"
           "%.1f,%.3f,%.1f\n",
           experiment, x_var, M, N, k_tiles, mode, swizzle, iters,
           avg_total_cycles, cycles_per_gemm, cycles_per_ktile,
           ns_per_gemm, tflops, spread);

    CUDA_CHECK(cudaFree(d_result));
}

// ---------------------------------------------------------------------------
// Experiment 1: utcmma_latency_ts — TS single K-tile latency sweep
// ---------------------------------------------------------------------------
static void run_utcmma_latency_ts(bool verbose) {
    fprintf(stderr, "\n=== Experiment 1: utcmma_latency_ts ===\n");

    const int ITERS = 10000;
    const int RUNS = 11;

    struct TileConfig {
        int M, N;
    };

    // Sweep tile sizes relevant to competition
    TileConfig configs[] = {
        {64, 64}, {64, 128}, {64, 256}, {128, 128}, {128, 256}
    };

    for (auto& cfg : configs) {
        if (verbose)
            fprintf(stderr, "  M=%d N=%d ...\n", cfg.M, cfg.N);

        // Compute shared memory needed: N × K × sizeof(bf16) with some alignment padding
        int smem_bytes = cfg.N * 16 * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            // Dispatch based on M, N
            if (cfg.M == 64 && cfg.N == 64) {
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_utcmma_latency_ts<64, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                kernel_utcmma_latency_ts<64, 64><<<1, 128, smem>>>(d_result, iters);
            } else if (cfg.M == 64 && cfg.N == 128) {
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_utcmma_latency_ts<64, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                kernel_utcmma_latency_ts<64, 128><<<1, 128, smem>>>(d_result, iters);
            } else if (cfg.M == 64 && cfg.N == 256) {
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_utcmma_latency_ts<64, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                kernel_utcmma_latency_ts<64, 256><<<1, 128, smem>>>(d_result, iters);
            } else if (cfg.M == 128 && cfg.N == 128) {
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_utcmma_latency_ts<128, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                kernel_utcmma_latency_ts<128, 128><<<1, 128, smem>>>(d_result, iters);
            } else if (cfg.M == 128 && cfg.N == 256) {
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_utcmma_latency_ts<128, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                kernel_utcmma_latency_ts<128, 256><<<1, 128, smem>>>(d_result, iters);
            }
        };

        char x_val[32];
        snprintf(x_val, sizeof(x_val), "%dx%d", cfg.M, cfg.N);

        // latency_ts kernel hardcodes SW32 (K=16 requires K-atom=16, SW32 is the only fit)
        run_and_report(
            "utcmma_latency_ts", x_val,
            cfg.M, cfg.N, 1, "ws_ts", "SW32", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 2: utcmma_kdepth_ts — K-depth sweep
// ---------------------------------------------------------------------------
static void run_utcmma_kdepth_ts(bool verbose) {
    fprintf(stderr, "\n=== Experiment 2: utcmma_kdepth_ts ===\n");

    const int ITERS = 1000;
    const int RUNS = 11;
    const int M = 64, N = 128;

    // k_depth=36 (K_TOTAL=576) exceeds TMEM capacity for M=64 NonInterleaved:
    // each K-tile consumes TMEM columns, and k_depth=36 runs off the 512-col TMEM.
    // Max competition-relevant k_depth=32 (ckv K=512 = 32 K-tiles of 16).
    int k_depths[] = {1, 2, 4, 8, 16, 24, 32};

    for (int kd : k_depths) {
        if (verbose)
            fprintf(stderr, "  k_depth=%d ...\n", kd);

        int smem_bytes = N * (kd * 16) * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            // Each template instantiation needs its own smem attribute set
            #define LAUNCH_KDEPTH(KD) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_utcmma_kdepth_ts<M, N, KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_utcmma_kdepth_ts<M, N, KD><<<1, 128, smem>>>(d_result, iters);
            switch (kd) {
                case 1:  { LAUNCH_KDEPTH(1);  break; }
                case 2:  { LAUNCH_KDEPTH(2);  break; }
                case 4:  { LAUNCH_KDEPTH(4);  break; }
                case 8:  { LAUNCH_KDEPTH(8);  break; }
                case 16: { LAUNCH_KDEPTH(16); break; }
                case 24: { LAUNCH_KDEPTH(24); break; }
                case 32: { LAUNCH_KDEPTH(32); break; }
            }
            #undef LAUNCH_KDEPTH
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", kd);

        // Swizzle matches kernel: SW128 if K_TOTAL>=64, SW64 if K_TOTAL>=32, else SW32
        int k_total = kd * 16;
        int swizzle_bits = (k_total >= 64) ? 128 : (k_total >= 32 ? 64 : 32);
        char swizzle_name[8];
        snprintf(swizzle_name, sizeof(swizzle_name), "SW%d", swizzle_bits);

        run_and_report(
            "utcmma_kdepth_ts", x_val,
            M, N, kd, "ws_ts", swizzle_name, ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 3: utcmma_latency_ss — SS variant
// ---------------------------------------------------------------------------
static void run_utcmma_latency_ss(bool verbose) {
    fprintf(stderr, "\n=== Experiment 3: utcmma_latency_ss ===\n");

    const int RUNS = 11;

    struct SSConfig {
        int M, N, k_depth, iters;
    };

    SSConfig configs[] = {
        {64, 128, 1, 10000},   // Single tile comparison with TS
        {64, 256, 1, 10000},   // Single tile at SV size
        {64, 256, 4, 1000},    // Exact competition SV GEMM: M=64, N=256, K=64 (4 tiles)
    };

    for (auto& cfg : configs) {
        if (verbose)
            fprintf(stderr, "  M=%d N=%d k_depth=%d ...\n", cfg.M, cfg.N, cfg.k_depth);

        int K_TOTAL = cfg.k_depth * 16;
        // smem for A: M × K_TOTAL + smem for B: N × K_TOTAL
        int smem_bytes = (cfg.M * K_TOTAL + cfg.N * K_TOTAL) * sizeof(__nv_bfloat16) + 512;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            if (smem > 48*1024) {
                // Set for all variants we might use
                cudaFuncSetAttribute(kernel_utcmma_latency_ss<64, 128, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utcmma_latency_ss<64, 256, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utcmma_latency_ss<64, 256, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            if (cfg.M == 64 && cfg.N == 128 && cfg.k_depth == 1) {
                kernel_utcmma_latency_ss<64, 128, 1><<<1, 128, smem>>>(d_result, iters);
            } else if (cfg.M == 64 && cfg.N == 256 && cfg.k_depth == 1) {
                kernel_utcmma_latency_ss<64, 256, 1><<<1, 128, smem>>>(d_result, iters);
            } else if (cfg.M == 64 && cfg.N == 256 && cfg.k_depth == 4) {
                kernel_utcmma_latency_ss<64, 256, 4><<<1, 128, smem>>>(d_result, iters);
            }
        };

        char x_val[32];
        snprintf(x_val, sizeof(x_val), "%dx%dx%d", cfg.M, cfg.N, cfg.k_depth);

        run_and_report(
            "utcmma_latency_ss", x_val,
            cfg.M, cfg.N, cfg.k_depth, "ws_ss", "INTER_MN128", cfg.iters,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 4: utccp_latency — UTCCP smem→TMEM copy
// ---------------------------------------------------------------------------
static void run_utccp_latency(bool verbose) {
    fprintf(stderr, "\n=== Experiment 4: utccp_latency ===\n");

    const int ITERS = 5000;
    const int RUNS = 11;

    int num_copies_list[] = {1, 2, 4, 8, 16};

    for (int nc : num_copies_list) {
        if (verbose)
            fprintf(stderr, "  num_copies=%d ...\n", nc);

        // smem: 128 × (nc * 16) bf16 elements + barrier storage
        int smem_bytes = 128 * nc * 16 * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            if (smem > 48*1024) {
                cudaFuncSetAttribute(kernel_utccp_latency<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utccp_latency<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utccp_latency<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utccp_latency<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utccp_latency<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            switch (nc) {
                case 1:  kernel_utccp_latency<1> <<<1, 128, smem>>>(d_result, iters); break;
                case 2:  kernel_utccp_latency<2> <<<1, 128, smem>>>(d_result, iters); break;
                case 4:  kernel_utccp_latency<4> <<<1, 128, smem>>>(d_result, iters); break;
                case 8:  kernel_utccp_latency<8> <<<1, 128, smem>>>(d_result, iters); break;
                case 16: kernel_utccp_latency<16><<<1, 128, smem>>>(d_result, iters); break;
            }
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", nc);

        // For UTCCP, report cycles_per_gemm as cycles_per_batch, cycles_per_ktile as cycles_per_copy
        run_and_report(
            "utccp_latency", x_val,
            128, nc * 16, nc, "utccp", "SW128", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 5: utcmma_swizzle — Swizzle impact
// ---------------------------------------------------------------------------
static void run_utcmma_swizzle(bool verbose) {
    fprintf(stderr, "\n=== Experiment 5: utcmma_swizzle ===\n");

    const int ITERS = 1000;
    const int RUNS = 11;
    const int M = 64, N = 128, K_DEPTH = 4;
    const int K_TOTAL = K_DEPTH * 16;

    struct SwConfig {
        int swizzle;
        const char* name;
    };

    SwConfig configs[] = {
        {16,  "INTER"},
        {32,  "SW32"},
        {64,  "SW64"},
        {128, "SW128"},
    };

    for (auto& sw : configs) {
        if (verbose)
            fprintf(stderr, "  swizzle=%s ...\n", sw.name);

        int smem_bytes = N * K_TOTAL * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            if (smem > 48*1024) {
                cudaFuncSetAttribute(kernel_utcmma_swizzle<64, 128, 4, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utcmma_swizzle<64, 128, 4, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utcmma_swizzle<64, 128, 4, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute(kernel_utcmma_swizzle<64, 128, 4, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            switch (sw.swizzle) {
                case 16:  kernel_utcmma_swizzle<M, N, K_DEPTH, 16> <<<1, 128, smem>>>(d_result, iters); break;
                case 32:  kernel_utcmma_swizzle<M, N, K_DEPTH, 32> <<<1, 128, smem>>>(d_result, iters); break;
                case 64:  kernel_utcmma_swizzle<M, N, K_DEPTH, 64> <<<1, 128, smem>>>(d_result, iters); break;
                case 128: kernel_utcmma_swizzle<M, N, K_DEPTH, 128><<<1, 128, smem>>>(d_result, iters); break;
            }
        };

        run_and_report(
            "utcmma_swizzle", sw.name,
            M, N, K_DEPTH, "ws_ts", sw.name, ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 6: utcmma_throughput_2acc — Multi-accumulator throughput
// ---------------------------------------------------------------------------
static void run_utcmma_throughput_2acc(bool verbose) {
    fprintf(stderr, "\n=== Experiment 6: utcmma_throughput_2acc ===\n"); fflush(stderr);

    const int RUNS = 11;

    struct AccConfig {
        int M, N, n_acc, k_depth, iters;
    };

    AccConfig configs[] = {
        // N=64 k_depth=4: baseline throughput sweep (QK kpe tile: N=64, K=64)
        {64, 64, 1, 4, 1000},
        {64, 64, 2, 4, 1000},
        {64, 64, 3, 4, 1000},
        {64, 64, 4, 4, 1000},
        // N=64 k_depth=32: competition QK ckv tile (N=64 scores, K=512 head_dim → 32 K-tiles)
        {64, 64, 1, 32, 1000},
        {64, 64, 2, 32, 1000},
        {64, 64, 3, 32, 1000},   // N_ACC=3: gap=33 cy < 43 cy RAW → predicted still serialized
        {64, 64, 4, 32, 1000},   // N_ACC=4: gap=44 cy ≥ 43 cy RAW → predicted to break dependency
        {64, 64, 5, 32, 1000},   // N_ACC=5: gap=55 cy >> 43 cy RAW → should clearly break
        // N=128 k_depth=4: FlashMLA dual-GEMM B smem packing shape (two B_TOPK=64 blocks)
        {64, 128, 1, 4, 1000},
        {64, 128, 2, 4, 1000},
    };

    for (auto& cfg : configs) {
        if (verbose) {
            fprintf(stderr, "  M=%d N=%d n_acc=%d k_depth=%d ...\n",
                    cfg.M, cfg.N, cfg.n_acc, cfg.k_depth);
            fflush(stderr);
        }

        int k_total = cfg.k_depth * 16;
        int smem_bytes = cfg.N * k_total * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_2ACC(M_, N_, NA_, KD_) \
                if (smem > 48*1024) cudaFuncSetAttribute( \
                    kernel_utcmma_throughput_2acc<M_, N_, NA_, KD_>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_utcmma_throughput_2acc<M_, N_, NA_, KD_><<<1, 128, smem>>>(d_result, iters);

            if (cfg.M == 64 && cfg.N == 64 && cfg.k_depth == 4) {
                switch (cfg.n_acc) {
                    case 1: { LAUNCH_2ACC(64, 64, 1, 4); break; }
                    case 2: { LAUNCH_2ACC(64, 64, 2, 4); break; }
                    case 3: { LAUNCH_2ACC(64, 64, 3, 4); break; }
                    case 4: { LAUNCH_2ACC(64, 64, 4, 4); break; }
                }
            } else if (cfg.M == 64 && cfg.N == 128 && cfg.k_depth == 4) {
                switch (cfg.n_acc) {
                    case 1: { LAUNCH_2ACC(64, 128, 1, 4); break; }
                    case 2: { LAUNCH_2ACC(64, 128, 2, 4); break; }
                }
            } else if (cfg.M == 64 && cfg.N == 64 && cfg.k_depth == 32) {
                switch (cfg.n_acc) {
                    case 1: { LAUNCH_2ACC(64, 64, 1, 32); break; }
                    case 2: { LAUNCH_2ACC(64, 64, 2, 32); break; }
                    case 3: { LAUNCH_2ACC(64, 64, 3, 32); break; }
                    case 4: { LAUNCH_2ACC(64, 64, 4, 32); break; }
                    case 5: { LAUNCH_2ACC(64, 64, 5, 32); break; }
                }
            } else if (cfg.M == 64 && cfg.N == 128 && cfg.k_depth == 32) {
                switch (cfg.n_acc) {
                    case 1: { LAUNCH_2ACC(64, 128, 1, 32); break; }
                    case 2: { LAUNCH_2ACC(64, 128, 2, 32); break; }
                }
            }
            #undef LAUNCH_2ACC
        };

        char x_val[32];
        snprintf(x_val, sizeof(x_val), "nacc%d", cfg.n_acc);

        int k_total_sw = cfg.k_depth * 16;
        int swizzle_bits = (k_total_sw >= 64) ? 128 : (k_total_sw >= 32 ? 64 : 32);
        char swizzle_name[8];
        snprintf(swizzle_name, sizeof(swizzle_name), "SW%d", swizzle_bits);

        run_and_report(
            "utcmma_throughput_2acc", x_val,
            cfg.M, cfg.N, cfg.k_depth, "ws_ts", swizzle_name, cfg.iters,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 7: tmem_ld_st_fence — TMEM load/store/fence overhead
// ---------------------------------------------------------------------------
static void run_tmem_ld_st_fence(bool verbose) {
    fprintf(stderr, "\n=== Experiment 7: tmem_ld_st_fence ===\n"); fflush(stderr);

    const int RUNS = 11;

    struct TmemConfig {
        int mode;
        int tmem_width;
        const char* label;
        int iters;
    };

    TmemConfig configs[] = {
        // MODE=0: tmem_ld chain
        {0, 1,  "ld_1",  5000},
        {0, 4,  "ld_4",  5000},
        {0, 8,  "ld_8",  5000},
        {0, 16, "ld_16", 5000},
        {0, 32, "ld_32", 5000},
        {0, 64, "ld_64", 5000},
        // MODE=1: tmem_st chain
        {1, 1,  "st_1",  5000},
        {1, 4,  "st_4",  5000},
        {1, 8,  "st_8",  5000},
        {1, 16, "st_16", 5000},
        {1, 32, "st_32", 5000},
        {1, 64, "st_64", 5000},
        // MODE=2: fence-only baselines
        {2, 0, "fence_tmem",    10000},
        {2, 1, "fence_tcgen05", 10000},
        // MODE=3: full rescale roundtrip
        {3, 1, "rescale_1chunk", 1000},
        {3, 4, "rescale_4chunk", 1000},
    };

    for (auto& cfg : configs) {
        if (verbose)
            fprintf(stderr, "  %s ...\n", cfg.label);

        // MODE=3 needs smem for the float2 arrays; others only need TMEM
        int smem_bytes = 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_TMEM(M_, W_) \
                kernel_tmem_ld_st_fence<M_, W_><<<1, 128, smem>>>(d_result, iters);

            switch (cfg.mode * 1000 + cfg.tmem_width) {
                case 0*1000 + 1:  { LAUNCH_TMEM(0, 1);  break; }
                case 0*1000 + 4:  { LAUNCH_TMEM(0, 4);  break; }
                case 0*1000 + 8:  { LAUNCH_TMEM(0, 8);  break; }
                case 0*1000 + 16: { LAUNCH_TMEM(0, 16); break; }
                case 0*1000 + 32: { LAUNCH_TMEM(0, 32); break; }
                case 0*1000 + 64: { LAUNCH_TMEM(0, 64); break; }
                case 1*1000 + 1:  { LAUNCH_TMEM(1, 1);  break; }
                case 1*1000 + 4:  { LAUNCH_TMEM(1, 4);  break; }
                case 1*1000 + 8:  { LAUNCH_TMEM(1, 8);  break; }
                case 1*1000 + 16: { LAUNCH_TMEM(1, 16); break; }
                case 1*1000 + 32: { LAUNCH_TMEM(1, 32); break; }
                case 1*1000 + 64: { LAUNCH_TMEM(1, 64); break; }
                case 2*1000 + 0:  { LAUNCH_TMEM(2, 0);  break; }
                case 2*1000 + 1:  { LAUNCH_TMEM(2, 1);  break; }
                case 3*1000 + 1:  { LAUNCH_TMEM(3, 1);  break; }
                case 3*1000 + 4:  { LAUNCH_TMEM(3, 4);  break; }
            }
            #undef LAUNCH_TMEM
        };

        run_and_report(
            "tmem_ld_st_fence", cfg.label,
            0, 0, 1, "tmem", "N/A", cfg.iters,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 8+9: mbarrier_commit — Barrier and commit overhead
// ---------------------------------------------------------------------------
static void run_mbarrier_commit(bool verbose) {
    fprintf(stderr, "\n=== Experiment 8+9: mbarrier_commit ===\n"); fflush(stderr);

    const int RUNS = 11;

    struct BarConfig {
        int mode;
        const char* label;
        int iters;
    };

    BarConfig configs[] = {
        {0, "named_bar_128",  10000},
        {1, "named_bar_32",   10000},
        {2, "mbarrier_tx",    10000},
        {3, "mma_commit_wait", 5000},
    };

    for (auto& cfg : configs) {
        if (verbose)
            fprintf(stderr, "  %s ...\n", cfg.label);

        // MODE=3 needs smem for B tensor (64 × 16 bf16)
        int smem_bytes = (cfg.mode == 3) ? (64 * 16 * sizeof(__nv_bfloat16) + 256) : 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            switch (cfg.mode) {
                case 0: kernel_mbarrier_commit<0><<<1, 128, smem>>>(d_result, iters); break;
                case 1: kernel_mbarrier_commit<1><<<1, 128, smem>>>(d_result, iters); break;
                case 2: kernel_mbarrier_commit<2><<<1, 128, smem>>>(d_result, iters); break;
                case 3: kernel_mbarrier_commit<3><<<1, 128, smem>>>(d_result, iters); break;
            }
        };

        run_and_report(
            "mbarrier_commit", cfg.label,
            (cfg.mode == 3) ? 64 : 0,
            (cfg.mode == 3) ? 64 : 0,
            (cfg.mode == 3) ? 1 : 0,
            "sync", "N/A", cfg.iters,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 10: utcmma_dual_gemm_layout — Layout validation
// ---------------------------------------------------------------------------
static void run_utcmma_dual_gemm_layout(bool verbose) {
    fprintf(stderr, "\n=== Experiment 10: utcmma_dual_gemm_layout ===\n"); fflush(stderr);

    const int ITERS = 1000;
    const int RUNS = 11;

    struct LayoutConfig {
        int k_depth;
        int layout_mode;
        const char* label;
    };

    LayoutConfig configs[] = {
        {4,  0, "canonical_kd4"},
        {4,  1, "competition_kd4"},
        {32, 0, "canonical_kd32"},
        {32, 1, "competition_kd32"},
    };

    const int M = 64, N = 128;

    for (auto& cfg : configs) {
        if (verbose)
            fprintf(stderr, "  %s ...\n", cfg.label);

        int k_total = cfg.k_depth * 16;
        int smem_bytes = N * k_total * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_LAYOUT(KD_, LM_) \
                if (smem > 48*1024) cudaFuncSetAttribute( \
                    kernel_utcmma_dual_gemm_layout<64, 128, KD_, LM_>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_utcmma_dual_gemm_layout<64, 128, KD_, LM_><<<1, 128, smem>>>(d_result, iters);

            if (cfg.k_depth == 4 && cfg.layout_mode == 0) { LAUNCH_LAYOUT(4, 0); }
            else if (cfg.k_depth == 4 && cfg.layout_mode == 1) { LAUNCH_LAYOUT(4, 1); }
            else if (cfg.k_depth == 32 && cfg.layout_mode == 0) { LAUNCH_LAYOUT(32, 0); }
            else if (cfg.k_depth == 32 && cfg.layout_mode == 1) { LAUNCH_LAYOUT(32, 1); }
            #undef LAUNCH_LAYOUT
        };

        run_and_report(
            "utcmma_dual_gemm_layout", cfg.label,
            M, N, cfg.k_depth, "ws_ts", "SW128", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 12: utcmma_non_ws_ts — Non-WS MMA comparison
// ---------------------------------------------------------------------------
static void run_utcmma_non_ws_ts(bool verbose) {
    fprintf(stderr, "\n=== Experiment 12: utcmma_non_ws_ts ===\n"); fflush(stderr);

    const int RUNS = 11;

    struct NWConfig {
        int M, N, k_depth, iters;
        const char* label;
    };

    NWConfig configs[] = {
        {64, 64,  1,  10000, "64x64"},
        {64, 128, 1,  10000, "64x128"},
        {64, 128, 4,  1000,  "kd4"},
        {64, 128, 32, 1000,  "kd32"},
    };

    for (auto& cfg : configs) {
        if (verbose)
            fprintf(stderr, "  %s (M=%d N=%d kd=%d) ...\n",
                    cfg.label, cfg.M, cfg.N, cfg.k_depth);

        int k_total = cfg.k_depth * 16;
        int smem_bytes = cfg.N * k_total * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_NOWS(M_, N_, KD_) \
                if (smem > 48*1024) cudaFuncSetAttribute( \
                    kernel_utcmma_non_ws_ts<M_, N_, KD_>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_utcmma_non_ws_ts<M_, N_, KD_><<<1, 128, smem>>>(d_result, iters);

            if (cfg.M == 64 && cfg.N == 64 && cfg.k_depth == 1) { LAUNCH_NOWS(64, 64, 1); }
            else if (cfg.M == 64 && cfg.N == 128 && cfg.k_depth == 1)  { LAUNCH_NOWS(64, 128, 1); }
            else if (cfg.M == 64 && cfg.N == 128 && cfg.k_depth == 4)  { LAUNCH_NOWS(64, 128, 4); }
            else if (cfg.M == 64 && cfg.N == 128 && cfg.k_depth == 32) { LAUNCH_NOWS(64, 128, 32); }
            #undef LAUNCH_NOWS
        };

        int swizzle_bits = (k_total >= 64) ? 128 : (k_total >= 32 ? 64 : 32);
        char swizzle_name[8];
        snprintf(swizzle_name, sizeof(swizzle_name), "SW%d", swizzle_bits);

        run_and_report(
            "utcmma_non_ws_ts", cfg.label,
            cfg.M, cfg.N, cfg.k_depth, "nows_ts", swizzle_name, cfg.iters,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// NOTE: Experiment 13 (M=32 WS-TS) REMOVED — M=32 is valid per PTX ISA for
// tcgen05.mma.ws.cta_group::1 but CuTe's TMEM fragment code
// (mma_traits_sm100.hpp:502) has a static_assert blocking M=32.
// To benchmark M=32, either fix CuTe or use raw PTX inline assembly.

// ---------------------------------------------------------------------------
// Experiment 15: utcmma_non_ws_ss — Non-WS SS latency
// ---------------------------------------------------------------------------
static void run_utcmma_non_ws_ss(bool verbose) {
    fprintf(stderr, "\n=== Experiment 15: utcmma_non_ws_ss ===\n"); fflush(stderr);

    const int RUNS = 11;

    struct NWSSConfig {
        int M, N, k_depth, iters;
        const char* label;
    };

    NWSSConfig configs[] = {
        {64, 64,  1, 10000, "64x64"},    // Direct comparison with WS-SS
        {64, 128, 1, 10000, "64x128"},   // Direct comparison with WS-SS
        {64, 256, 1, 10000, "64x256"},   // Direct comparison with WS-SS
        {64, 256, 4, 1000,  "kd4"},      // Competition SV tile
    };

    for (auto& cfg : configs) {
        if (verbose)
            fprintf(stderr, "  %s (M=%d N=%d kd=%d) ...\n",
                    cfg.label, cfg.M, cfg.N, cfg.k_depth);

        int K_TOTAL = cfg.k_depth * 16;
        int smem_bytes = (cfg.M * K_TOTAL + cfg.N * K_TOTAL) * sizeof(__nv_bfloat16) + 512;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_NOWS_SS(M_, N_, KD_) \
                if (smem > 48*1024) cudaFuncSetAttribute( \
                    kernel_utcmma_non_ws_ss<M_, N_, KD_>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_utcmma_non_ws_ss<M_, N_, KD_><<<1, 128, smem>>>(d_result, iters);

            if (cfg.M == 64 && cfg.N == 64 && cfg.k_depth == 1)  { LAUNCH_NOWS_SS(64, 64, 1); }
            else if (cfg.M == 64 && cfg.N == 128 && cfg.k_depth == 1) { LAUNCH_NOWS_SS(64, 128, 1); }
            else if (cfg.M == 64 && cfg.N == 256 && cfg.k_depth == 1) { LAUNCH_NOWS_SS(64, 256, 1); }
            else if (cfg.M == 64 && cfg.N == 256 && cfg.k_depth == 4) { LAUNCH_NOWS_SS(64, 256, 4); }
            #undef LAUNCH_NOWS_SS
        };

        int swizzle_bits = (K_TOTAL >= 64) ? 128 : (K_TOTAL >= 32 ? 64 : 32);
        char swizzle_name[16];
        snprintf(swizzle_name, sizeof(swizzle_name), "INTER_MN%d",
                 (K_TOTAL >= 64) ? 128 : (K_TOTAL >= 32 ? 64 : 32));

        run_and_report(
            "utcmma_non_ws_ss", cfg.label,
            cfg.M, cfg.N, cfg.k_depth, "nows_ss", swizzle_name, cfg.iters,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 16: utcmma_non_ws_fine_n — Non-WS fine-grained N sweep (TS + SS)
// ---------------------------------------------------------------------------
static void run_utcmma_non_ws_fine_n(bool verbose) {
    fprintf(stderr, "\n=== Experiment 16: utcmma_non_ws_fine_n ===\n"); fflush(stderr);

    const int ITERS = 10000;
    const int RUNS = 11;
    const int M = 64;
    const int K_DEPTH = 1;
    const int K_TOTAL = 16;

    int n_values[] = {8, 16, 24, 32, 40, 48, 56, 64};

    // --- Non-WS TS fine N ---
    for (int n : n_values) {
        if (verbose)
            fprintf(stderr, "  TS N=%d ...\n", n);

        int smem_bytes = n * K_TOTAL * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_FINE_TS(N_) \
                if (smem > 48*1024) cudaFuncSetAttribute( \
                    kernel_utcmma_non_ws_ts<64, N_, 1>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_utcmma_non_ws_ts<64, N_, 1><<<1, 128, smem>>>(d_result, iters);

            switch (n) {
                case 8:  { LAUNCH_FINE_TS(8);  break; }
                case 16: { LAUNCH_FINE_TS(16); break; }
                case 24: { LAUNCH_FINE_TS(24); break; }
                case 32: { LAUNCH_FINE_TS(32); break; }
                case 40: { LAUNCH_FINE_TS(40); break; }
                case 48: { LAUNCH_FINE_TS(48); break; }
                case 56: { LAUNCH_FINE_TS(56); break; }
                case 64: { LAUNCH_FINE_TS(64); break; }
            }
            #undef LAUNCH_FINE_TS
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "N%d", n);

        run_and_report(
            "utcmma_non_ws_ts_fine_n", x_val,
            M, n, K_DEPTH, "nows_ts", "SW32", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }

    // --- Non-WS SS fine N ---
    for (int n : n_values) {
        if (verbose)
            fprintf(stderr, "  SS N=%d ...\n", n);

        int smem_bytes = (M * K_TOTAL + n * K_TOTAL) * sizeof(__nv_bfloat16) + 512;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_FINE_SS(N_) \
                if (smem > 48*1024) cudaFuncSetAttribute( \
                    kernel_utcmma_non_ws_ss<64, N_, 1>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_utcmma_non_ws_ss<64, N_, 1><<<1, 128, smem>>>(d_result, iters);

            switch (n) {
                case 8:  { LAUNCH_FINE_SS(8);  break; }
                case 16: { LAUNCH_FINE_SS(16); break; }
                case 24: { LAUNCH_FINE_SS(24); break; }
                case 32: { LAUNCH_FINE_SS(32); break; }
                case 40: { LAUNCH_FINE_SS(40); break; }
                case 48: { LAUNCH_FINE_SS(48); break; }
                case 56: { LAUNCH_FINE_SS(56); break; }
                case 64: { LAUNCH_FINE_SS(64); break; }
            }
            #undef LAUNCH_FINE_SS
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "N%d", n);

        run_and_report(
            "utcmma_non_ws_ss_fine_n", x_val,
            M, n, K_DEPTH, "nows_ss", "INTER_SW32", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 17: utccp_128dp128b — UTCCP half-width variant
// ---------------------------------------------------------------------------
static void run_utccp_128dp128b(bool verbose) {
    fprintf(stderr, "\n=== Experiment 17: utccp_128dp128b ===\n"); fflush(stderr);

    const int ITERS = 5000;
    const int RUNS = 11;

    int num_copies_list[] = {1, 2, 4, 8, 16, 32};

    for (int nc : num_copies_list) {
        if (verbose)
            fprintf(stderr, "  num_copies=%d ...\n", nc);

        // smem: 128 × (nc * 8) bf16 elements + barrier storage
        // 128dp128bit: 128 rows × 128 bits = 128 × 8 bf16 elements per copy
        int smem_bytes = 128 * nc * 8 * sizeof(__nv_bfloat16) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_UTCCP128(NC_) \
                if (smem > 48*1024) cudaFuncSetAttribute( \
                    kernel_utccp_128dp128b<NC_>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_utccp_128dp128b<NC_><<<1, 128, smem>>>(d_result, iters);

            switch (nc) {
                case 1:  { LAUNCH_UTCCP128(1);  break; }
                case 2:  { LAUNCH_UTCCP128(2);  break; }
                case 4:  { LAUNCH_UTCCP128(4);  break; }
                case 8:  { LAUNCH_UTCCP128(8);  break; }
                case 16: { LAUNCH_UTCCP128(16); break; }
                case 32: { LAUNCH_UTCCP128(32); break; }
            }
            #undef LAUNCH_UTCCP128
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", nc);

        run_and_report(
            "utccp_128dp128b", x_val,
            128, nc * 8, nc, "utccp_128b", "INTER", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string experiment = "all";
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--experiment=", 13) == 0)
            experiment = argv[i] + 13;
        else if (strcmp(argv[i], "--verbose") == 0)
            verbose = true;
    }

    // GPU info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "GPU: %s (SM %d.%d, %d SMs)\n",
            prop.name, prop.major, prop.minor,
            prop.multiProcessorCount);
    fprintf(stderr, "Max shared memory per block: %zu bytes\n",
            prop.sharedMemPerBlockOptin);

    // Clock warmup
    fprintf(stderr, "Warming up GPU clocks...\n");
    gpu_clock_warmup(device);
    fprintf(stderr, "Warmup complete.\n");

    // Print CSV header
    print_csv_header();

    // Run experiments
    bool run_all = (experiment == "all");

    if (run_all || experiment == "utcmma_latency_ts")
        run_utcmma_latency_ts(verbose);

    if (run_all || experiment == "utcmma_kdepth_ts")
        run_utcmma_kdepth_ts(verbose);

    if (run_all || experiment == "utcmma_latency_ss")
        run_utcmma_latency_ss(verbose);

    if (run_all || experiment == "utccp_latency")
        run_utccp_latency(verbose);

    if (run_all || experiment == "utcmma_swizzle")
        run_utcmma_swizzle(verbose);

    // --- Ver2 experiments ---
    if (run_all || experiment == "utcmma_throughput_2acc")
        run_utcmma_throughput_2acc(verbose);

    if (run_all || experiment == "tmem_ld_st_fence")
        run_tmem_ld_st_fence(verbose);

    if (run_all || experiment == "mbarrier_commit")
        run_mbarrier_commit(verbose);

    if (run_all || experiment == "utcmma_dual_gemm_layout")
        run_utcmma_dual_gemm_layout(verbose);

    if (run_all || experiment == "utcmma_non_ws_ts")
        run_utcmma_non_ws_ts(verbose);

    // --- Ver4 experiments ---
    // NOTE: utcmma_m32_ws_ts removed — CuTe TMEM fragment doesn't support M=32
    // (PTX ISA allows it for .ws, but CuTe mma_traits_sm100.hpp:502 blocks it)

    if (run_all || experiment == "utcmma_non_ws_ss")
        run_utcmma_non_ws_ss(verbose);

    if (run_all || experiment == "utcmma_non_ws_fine_n")
        run_utcmma_non_ws_fine_n(verbose);

    if (run_all || experiment == "utccp_128dp128b")
        run_utccp_128dp128b(verbose);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
