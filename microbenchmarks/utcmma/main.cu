#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "../common/benchmark_common.cuh"
#include "utcmma_kernels.cuh"

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

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
