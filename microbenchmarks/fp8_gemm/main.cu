#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "../common/benchmark_common.cuh"
#include "fp8_gemm_kernels.cuh"

// ---------------------------------------------------------------------------
// CSV header
// ---------------------------------------------------------------------------
static void print_csv_header() {
    printf("experiment,x_var,M,N,k_depth,n_acc,swizzle,iters,"
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
    int M, int N, int k_depth, int n_acc,
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
            fprintf(stderr, "  run %d/%d: %lu cycles, %.1f ns\n",
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

    // FP8: K=32 per tile (vs BF16 K=16). FMA = 2 FLOPs per multiply-add.
    double flops_per_gemm = (double)M * N * 32 * 2 * k_tiles;
    double tflops = (flops_per_gemm * iters) / (avg_total_ns * 1e-9) / 1e12;

    double spread = cycles_series.spread() * 100.0;

    printf("%s,%s,%d,%d,%d,%d,%s,%d,"
           "%.0f,%.1f,%.2f,"
           "%.1f,%.3f,%.1f\n",
           experiment, x_var, M, N, k_tiles, n_acc, swizzle, iters,
           avg_total_cycles, cycles_per_gemm, cycles_per_ktile,
           ns_per_gemm, tflops, spread);
    fflush(stdout);

    CUDA_CHECK(cudaFree(d_result));
}

// ---------------------------------------------------------------------------
// Experiment 1: fp8_latency_ts — single K-tile latency sweep
// ---------------------------------------------------------------------------
static void run_fp8_latency_ts(bool verbose) {
    fprintf(stderr, "\n=== Experiment: fp8_latency_ts ===\n");

    const int ITERS = 10000;
    const int RUNS = 11;

    struct TileConfig { int M, N; };
    TileConfig configs[] = {
        {64, 64}, {64, 128}, {64, 256}, {128, 128}, {128, 256}
    };

    for (auto& cfg : configs) {
        if (verbose) fprintf(stderr, "  M=%d N=%d ...\n", cfg.M, cfg.N);

        // FP8: K=32, 1 byte/elem -> N*32 bytes for B operand + alignment
        int smem_bytes = cfg.N * 32 * sizeof(fp8_e4m3) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_FP8_LAT(m, n) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_fp8_latency_ts<m, n>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_fp8_latency_ts<m, n><<<1, 128, smem>>>(d_result, iters);
            if      (cfg.M == 64  && cfg.N == 64)  { LAUNCH_FP8_LAT(64, 64);   }
            else if (cfg.M == 64  && cfg.N == 128) { LAUNCH_FP8_LAT(64, 128);  }
            else if (cfg.M == 64  && cfg.N == 256) { LAUNCH_FP8_LAT(64, 256);  }
            else if (cfg.M == 128 && cfg.N == 128) { LAUNCH_FP8_LAT(128, 128); }
            else if (cfg.M == 128 && cfg.N == 256) { LAUNCH_FP8_LAT(128, 256); }
            #undef LAUNCH_FP8_LAT
        };

        char x_val[32];
        snprintf(x_val, sizeof(x_val), "%dx%d", cfg.M, cfg.N);

        run_and_report(
            "fp8_latency_ts", x_val,
            cfg.M, cfg.N, 1, 1, "SW32", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 2: fp8_kdepth_ts — K-depth sweep
// ---------------------------------------------------------------------------
static void run_fp8_kdepth_ts(bool verbose) {
    fprintf(stderr, "\n=== Experiment: fp8_kdepth_ts ===\n");

    const int ITERS = 1000;
    const int RUNS = 11;
    const int M = 64, N = 64;

    // FP8 K=32 per tile. Competition head_dim=128 -> K_DEPTH=4.
    int k_depths[] = {1, 2, 4, 8};

    for (int kd : k_depths) {
        if (verbose) fprintf(stderr, "  k_depth=%d ...\n", kd);

        int k_total = kd * 32;
        int smem_bytes = N * k_total * sizeof(fp8_e4m3) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_FP8_KD(KD) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_fp8_kdepth_ts<M, N, KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_fp8_kdepth_ts<M, N, KD><<<1, 128, smem>>>(d_result, iters);
            switch (kd) {
                case 1: { LAUNCH_FP8_KD(1); break; }
                case 2: { LAUNCH_FP8_KD(2); break; }
                case 4: { LAUNCH_FP8_KD(4); break; }
                case 8: { LAUNCH_FP8_KD(8); break; }
            }
            #undef LAUNCH_FP8_KD
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", kd);

        int swizzle_bits = (k_total >= 128) ? 128 : (k_total >= 64 ? 64 : 32);
        char swizzle_name[8];
        snprintf(swizzle_name, sizeof(swizzle_name), "SW%d", swizzle_bits);

        run_and_report(
            "fp8_kdepth_ts", x_val,
            M, N, kd, 1, swizzle_name, ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }

    // Also run with N=128 (wider tiles)
    const int N2 = 128;
    for (int kd : {1, 2, 4}) {
        if (verbose) fprintf(stderr, "  N=%d k_depth=%d ...\n", N2, kd);

        int k_total = kd * 32;
        int smem_bytes = N2 * k_total * sizeof(fp8_e4m3) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_FP8_KD2(KD) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_fp8_kdepth_ts<M, N2, KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_fp8_kdepth_ts<M, N2, KD><<<1, 128, smem>>>(d_result, iters);
            switch (kd) {
                case 1: { LAUNCH_FP8_KD2(1); break; }
                case 2: { LAUNCH_FP8_KD2(2); break; }
                case 4: { LAUNCH_FP8_KD2(4); break; }
            }
            #undef LAUNCH_FP8_KD2
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", kd);

        int swizzle_bits = (k_total >= 128) ? 128 : (k_total >= 64 ? 64 : 32);
        char swizzle_name[8];
        snprintf(swizzle_name, sizeof(swizzle_name), "SW%d", swizzle_bits);

        run_and_report(
            "fp8_kdepth_ts_n128", x_val,
            M, N2, kd, 1, swizzle_name, ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 3: fp8_throughput_2acc — multi-accumulator throughput
// ---------------------------------------------------------------------------
static void run_fp8_throughput_2acc(bool verbose) {
    fprintf(stderr, "\n=== Experiment: fp8_throughput_2acc ===\n");

    const int ITERS = 1000;
    const int RUNS = 11;
    const int M = 64, N = 64, K_DEPTH = 4;  // Competition: M=64, N=64, K=128

    // N_ACC sweep: 1..4 accumulators
    // Non-WS: M=64 uses N TMEM cols per accumulator
    // N=64 -> 64 cols/acc -> max 4 accs in 256 cols
    int n_accs[] = {1, 2, 3, 4};

    int k_total = K_DEPTH * 32;
    int smem_bytes = N * k_total * sizeof(fp8_e4m3) + 256;

    for (int nacc : n_accs) {
        if (verbose) fprintf(stderr, "  N_ACC=%d ...\n", nacc);

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_FP8_2ACC(NACC) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_fp8_throughput_2acc<M, N, NACC, K_DEPTH>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_fp8_throughput_2acc<M, N, NACC, K_DEPTH><<<1, 128, smem>>>(d_result, iters);
            switch (nacc) {
                case 1: { LAUNCH_FP8_2ACC(1); break; }
                case 2: { LAUNCH_FP8_2ACC(2); break; }
                case 3: { LAUNCH_FP8_2ACC(3); break; }
                case 4: { LAUNCH_FP8_2ACC(4); break; }
            }
            #undef LAUNCH_FP8_2ACC
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", nacc);

        int swizzle_bits = (k_total >= 128) ? 128 : (k_total >= 64 ? 64 : 32);
        char swizzle_name[8];
        snprintf(swizzle_name, sizeof(swizzle_name), "SW%d", swizzle_bits);

        run_and_report(
            "fp8_throughput_2acc", x_val,
            M, N, K_DEPTH, nacc, swizzle_name, ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// Experiment 4 (fp8_with_scale) removed — TMEM ld already benchmarked in utcmma ver8.
// Post-GEMM ops (broadcast, reduction) will be separate building-block benchmarks.

// ---------------------------------------------------------------------------
// Experiment 4: f8f6f4_latency_ss — single K-tile SS latency (M=128 fixed)
// ---------------------------------------------------------------------------
static void run_f8f6f4_latency_ss(bool verbose) {
    fprintf(stderr, "\n=== Experiment: f8f6f4_latency_ss ===\n");

    const int ITERS = 10000;
    const int RUNS  = 11;
    const int M     = 128;  // SS mode: M can be 64 or 128; run M=128 to match mxf8f6f4

    // N sweep for direct comparison with mxf8f6f4 (M=128, N=64)
    int ns[] = {64, 128, 256};

    for (int N : ns) {
        if (verbose) fprintf(stderr, "  M=%d N=%d ...\n", M, N);

        // SMEM: A(M×K) + B(N×K) + 256
        int smem_bytes = (M + N) * 32 * sizeof(fp8_e4m3) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_SS_LAT(n) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_f8f6f4_latency_ss<128, n>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_f8f6f4_latency_ss<128, n><<<1, 128, smem>>>(d_result, iters);
            if      (N == 64)  { LAUNCH_SS_LAT(64);  }
            else if (N == 128) { LAUNCH_SS_LAT(128); }
            else if (N == 256) { LAUNCH_SS_LAT(256); }
            #undef LAUNCH_SS_LAT
        };

        char x_val[32];
        snprintf(x_val, sizeof(x_val), "%dx%d", M, N);

        run_and_report(
            "f8f6f4_latency_ss", x_val,
            M, N, 1, 1, "SW32", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 5: f8f6f4_kdepth_ss — K-depth sweep in SS mode (M=128)
// ---------------------------------------------------------------------------
static void run_f8f6f4_kdepth_ss(bool verbose) {
    fprintf(stderr, "\n=== Experiment: f8f6f4_kdepth_ss ===\n");

    const int ITERS = 1000;
    const int RUNS  = 11;
    const int M     = 128;
    const int N     = 64;

    int k_depths[] = {1, 2, 4};

    for (int kd : k_depths) {
        if (verbose) fprintf(stderr, "  k_depth=%d ...\n", kd);

        int k_total    = kd * 32;
        int smem_bytes = (M + N) * k_total * sizeof(fp8_e4m3) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_SS_KD(KD) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_f8f6f4_kdepth_ss<128, 64, KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_f8f6f4_kdepth_ss<128, 64, KD><<<1, 128, smem>>>(d_result, iters);
            switch (kd) {
                case 1: { LAUNCH_SS_KD(1); break; }
                case 2: { LAUNCH_SS_KD(2); break; }
                case 4: { LAUNCH_SS_KD(4); break; }
            }
            #undef LAUNCH_SS_KD
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", kd);

        int swizzle_bits = (k_total >= 128) ? 128 : (k_total >= 64 ? 64 : 32);
        char swizzle_name[8];
        snprintf(swizzle_name, sizeof(swizzle_name), "SW%d", swizzle_bits);

        run_and_report(
            "f8f6f4_kdepth_ss", x_val,
            M, N, kd, 1, swizzle_name, ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 6: mxf8f6f4_latency_ss — block-scaled FP8, single K-tile (M=128)
// ---------------------------------------------------------------------------
static void run_mxf8f6f4_latency_ss(bool verbose) {
    fprintf(stderr, "\n=== Experiment: mxf8f6f4_latency_ss ===\n");

    const int ITERS = 10000;
    const int RUNS  = 11;
    const int M     = 128;  // mxf8f6f4 SS requires M=128

    int ns[] = {64, 128, 256};

    for (int N : ns) {
        if (verbose) fprintf(stderr, "  N=%d ...\n", N);

        int smem_bytes = (M + N) * 32 * sizeof(fp8_e4m3) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_MX_LAT(n) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_mxf8f6f4_latency_ss<n>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_mxf8f6f4_latency_ss<n><<<1, 128, smem>>>(d_result, iters);
            if      (N == 64)  { LAUNCH_MX_LAT(64);  }
            else if (N == 128) { LAUNCH_MX_LAT(128); }
            else if (N == 256) { LAUNCH_MX_LAT(256); }
            #undef LAUNCH_MX_LAT
        };

        char x_val[32];
        snprintf(x_val, sizeof(x_val), "%dx%d", M, N);

        run_and_report(
            "mxf8f6f4_latency_ss", x_val,
            M, N, 1, 1, "SW32", ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Experiment 7: mxf8f6f4_kdepth_ss — block-scaled K-depth sweep (M=128, N=64)
// ---------------------------------------------------------------------------
static void run_mxf8f6f4_kdepth_ss(bool verbose) {
    fprintf(stderr, "\n=== Experiment: mxf8f6f4_kdepth_ss ===\n");

    const int ITERS = 1000;
    const int RUNS  = 11;
    const int M     = 128;
    const int N     = 64;

    int k_depths[] = {1, 2, 4};

    for (int kd : k_depths) {
        if (verbose) fprintf(stderr, "  k_depth=%d ...\n", kd);

        int k_total    = kd * 32;
        int smem_bytes = (M + N) * k_total * sizeof(fp8_e4m3) + 256;

        auto launcher = [&](BenchResult* d_result, int iters, int smem) {
            #define LAUNCH_MX_KD(KD) \
                if (smem > 48*1024) cudaFuncSetAttribute(kernel_mxf8f6f4_kdepth_ss<64, KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                kernel_mxf8f6f4_kdepth_ss<64, KD><<<1, 128, smem>>>(d_result, iters);
            switch (kd) {
                case 1: { LAUNCH_MX_KD(1); break; }
                case 2: { LAUNCH_MX_KD(2); break; }
                case 4: { LAUNCH_MX_KD(4); break; }
            }
            #undef LAUNCH_MX_KD
        };

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", kd);

        int swizzle_bits = (k_total >= 128) ? 128 : (k_total >= 64 ? 64 : 32);
        char swizzle_name[8];
        snprintf(swizzle_name, sizeof(swizzle_name), "SW%d", swizzle_bits);

        run_and_report(
            "mxf8f6f4_kdepth_ss", x_val,
            M, N, kd, 1, swizzle_name, ITERS,
            launcher, smem_bytes, RUNS, verbose);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string experiment = "all";
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg.find("--experiment=") == 0) {
            experiment = arg.substr(13);
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }

    // GPU warmup
    gpu_clock_warmup();

    print_csv_header();

    if (experiment == "all" || experiment == "fp8_latency_ts") {
        run_fp8_latency_ts(verbose);
    }
    if (experiment == "all" || experiment == "fp8_kdepth_ts") {
        run_fp8_kdepth_ts(verbose);
    }
    if (experiment == "all" || experiment == "fp8_throughput_2acc") {
        run_fp8_throughput_2acc(verbose);
    }
    if (experiment == "all" || experiment == "f8f6f4_latency_ss") {
        run_f8f6f4_latency_ss(verbose);
    }
    if (experiment == "all" || experiment == "f8f6f4_kdepth_ss") {
        run_f8f6f4_kdepth_ss(verbose);
    }
    if (experiment == "all" || experiment == "mxf8f6f4_latency_ss") {
        run_mxf8f6f4_latency_ss(verbose);
    }
    if (experiment == "all" || experiment == "mxf8f6f4_kdepth_ss") {
        run_mxf8f6f4_kdepth_ss(verbose);
    }

    return 0;
}
