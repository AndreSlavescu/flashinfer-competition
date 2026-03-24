// Bulk Copy S2G (Shared → Global) microbenchmark
// Measures cp.async.bulk.global.shared::cta and cp.reduce.async.bulk.add.f32
// Target: B200 (sm100a)
//
// Experiments:
//   latency       — serialized bulk copy latency across sizes and modes
//   throughput    — pipelined bulk copy throughput (N_PIPE sweep)
//   overlap       — async overlap: copy + compute vs copy_only + compute_only
//   all           — run everything
//
// CSV schema:
//   experiment,x_var,mode,copy_bytes,n_pipe,compute_iters,iters,total_cycles,
//   cycles_per_op,ns_per_op,gbps,spread_pct

#include "bulk_kernels.cuh"
#include <string>

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------
static void print_csv_header() {
    printf("experiment,x_var,mode,copy_bytes,n_pipe,compute_iters,iters,"
           "total_cycles,cycles_per_op,ns_per_op,gbps,spread_pct\n");
}

static void print_csv_row(
    const char* experiment, const char* x_var, const char* mode,
    int copy_bytes, int n_pipe, int compute_iters, int iters,
    double total_cycles, double cycles_per_op, double ns_per_op,
    double gbps, double spread_pct)
{
    printf("%s,%s,%s,%d,%d,%d,%d,%.1f,%.2f,%.2f,%.3f,%.1f\n",
           experiment, x_var, mode,
           copy_bytes, n_pipe, compute_iters, iters,
           total_cycles, cycles_per_op, ns_per_op, gbps, spread_pct);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Generic run-and-report
// ---------------------------------------------------------------------------
using KernelFunc = void(*)(BulkResult*, int, float*);

static void run_and_report(
    const char* experiment, const char* x_var, const char* mode,
    int copy_bytes, int n_pipe, int compute_iters, int iters,
    int ops_per_iter,  // how many copies per iteration (for throughput: N_PIPE)
    KernelFunc kernel_fn, int smem_bytes,
    int num_runs, bool verbose,
    float* g_dst, int dst_bytes)
{
    BulkResult* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(BulkResult)));

    MeasurementSeries cycles_series;
    MeasurementSeries ns_series;

    for (int run = 0; run < num_runs; run++) {
        // Zero destination to avoid accumulation drift for reduce_add
        CUDA_CHECK(cudaMemset(g_dst, 0, dst_bytes));
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(BulkResult)));

        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(
                reinterpret_cast<const void*>(kernel_fn),
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }

        kernel_fn<<<1, 128, smem_bytes>>>(d_result, iters, g_dst);
        CUDA_CHECK(cudaDeviceSynchronize());

        BulkResult h_result;
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(BulkResult),
                              cudaMemcpyDeviceToHost));

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
    int total_ops = iters * ops_per_iter;
    double cycles_per_op = avg_total_cycles / total_ops;
    double ns_per_op = avg_total_ns / total_ops;
    double gbps = (copy_bytes > 0 && ns_per_op > 0)
                  ? (double)copy_bytes / ns_per_op  // bytes/ns = GB/s
                  : 0.0;
    double spread = cycles_series.spread() * 100.0;

    print_csv_row(experiment, x_var, mode,
                  copy_bytes, n_pipe, compute_iters, iters,
                  avg_total_cycles, cycles_per_op, ns_per_op, gbps, spread);

    CUDA_CHECK(cudaFree(d_result));
}

// ---------------------------------------------------------------------------
// Wrapper to cast kernel function pointers (template → KernelFunc)
// ---------------------------------------------------------------------------
// We need a trampoline because template kernels have extra params baked in.
// Instead, use a macro approach with direct launches via run_and_report_v2.

// Re-do: since KernelFunc expects (BulkResult*, int, float*), but our kernels
// have that signature too, we just need proper casting through a lambda or
// direct function pointer. The kernel signatures match:
//   kernel_bulk_latency<MODE, SIZE>(BulkResult*, int, float*)
// But they're __global__ functions, not host functions.
// We need a host-side launcher.

using LauncherFunc = void(*)(BulkResult*, int, int, float*);

static void run_bulk(
    const char* experiment, const char* x_var, const char* mode,
    int copy_bytes, int n_pipe, int compute_iters, int iters,
    int ops_per_iter,
    LauncherFunc launcher, int smem_bytes,
    int num_runs, bool verbose,
    float* g_dst, int dst_bytes)
{
    BulkResult* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(BulkResult)));

    MeasurementSeries cycles_series;
    MeasurementSeries ns_series;

    for (int run = 0; run < num_runs; run++) {
        CUDA_CHECK(cudaMemset(g_dst, 0, dst_bytes));
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(BulkResult)));

        launcher(d_result, iters, smem_bytes, g_dst);
        CUDA_CHECK(cudaDeviceSynchronize());

        BulkResult h_result;
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(BulkResult),
                              cudaMemcpyDeviceToHost));

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
    int total_ops = iters * ops_per_iter;
    double cycles_per_op = avg_total_cycles / total_ops;
    double ns_per_op = avg_total_ns / total_ops;
    double gbps = (copy_bytes > 0 && ns_per_op > 0)
                  ? (double)copy_bytes / ns_per_op
                  : 0.0;
    double spread = cycles_series.spread() * 100.0;

    print_csv_row(experiment, x_var, mode,
                  copy_bytes, n_pipe, compute_iters, iters,
                  avg_total_cycles, cycles_per_op, ns_per_op, gbps, spread);

    CUDA_CHECK(cudaFree(d_result));
}

// ---------------------------------------------------------------------------
// Experiment: latency — serialized copy, varying size and mode
// ---------------------------------------------------------------------------
static void run_latency(bool verbose) {
    fprintf(stderr, "\n=== Experiment: bulk_latency ===\n");

    // Sizes relevant to competition kernel output writes
    constexpr int SIZES[] = {512, 2048, 8192, 32768, 131072};
    constexpr int NUM_SIZES = 5;
    constexpr int ITERS = 5000;
    constexpr int RUNS = 11;

    // Allocate max global destination
    constexpr int MAX_DST = 131072 * 16 + 131072;  // 16 rotated slots + margin
    float* g_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&g_dst, MAX_DST));

    for (int si = 0; si < NUM_SIZES; si++) {
        const int sz = SIZES[si];

        // Mode 0: plain copy
        {
            char x_var[32];
            snprintf(x_var, sizeof(x_var), "copy_%d", sz);

            auto launcher = [](BulkResult* r, int iters, int smem, float* dst) {
                // Dispatch at compile time — we use a switch
                // But we can't template-switch at runtime easily.
                // Instead, we'll explicitly instantiate below.
            };
            (void)launcher;
        }
    }

    // Since we can't easily template-switch inside a lambda, expand manually:
    #define LAUNCH_LATENCY(MODE_, SZ_) \
        [](BulkResult* r, int iters, int smem, float* dst) { \
            if (smem > 48*1024) cudaFuncSetAttribute( \
                kernel_bulk_latency<MODE_, SZ_>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            kernel_bulk_latency<MODE_, SZ_><<<1, 128, smem>>>(r, iters, dst); \
        }

    struct LatencyConfig {
        int size;
        int mode;
        const char* mode_str;
        LauncherFunc launcher;
    };

    LatencyConfig configs[] = {
        // Mode 0: plain copy
        {512,    0, "copy",       LAUNCH_LATENCY(0, 512)},
        {2048,   0, "copy",       LAUNCH_LATENCY(0, 2048)},
        {8192,   0, "copy",       LAUNCH_LATENCY(0, 8192)},
        {32768,  0, "copy",       LAUNCH_LATENCY(0, 32768)},
        {131072, 0, "copy",       LAUNCH_LATENCY(0, 131072)},
        // Mode 1: reduce_add (f32)
        {512,    1, "reduce_add", LAUNCH_LATENCY(1, 512)},
        {2048,   1, "reduce_add", LAUNCH_LATENCY(1, 2048)},
        {8192,   1, "reduce_add", LAUNCH_LATENCY(1, 8192)},
        {32768,  1, "reduce_add", LAUNCH_LATENCY(1, 32768)},
        {131072, 1, "reduce_add", LAUNCH_LATENCY(1, 131072)},
    };
    #undef LAUNCH_LATENCY

    for (auto& cfg : configs) {
        char x_var[32];
        snprintf(x_var, sizeof(x_var), "%s_%d", cfg.mode_str, cfg.size);
        fprintf(stderr, "  %s...\n", x_var);

        run_bulk("bulk_latency", x_var, cfg.mode_str,
                 cfg.size, 1, 0, ITERS, 1,
                 cfg.launcher, cfg.size, RUNS, verbose,
                 g_dst, MAX_DST);
    }

    CUDA_CHECK(cudaFree(g_dst));
}

// ---------------------------------------------------------------------------
// Experiment: throughput — pipelined copies, varying N_PIPE
// ---------------------------------------------------------------------------
static void run_throughput(bool verbose) {
    fprintf(stderr, "\n=== Experiment: bulk_throughput ===\n");

    // Competition output slice: 32768 bytes (512 BF16 = 1024 B per head, 32 heads?)
    // Actually competition output = 512 * 2 = 1024 B per head, 16 heads = 16 KB per token.
    // Use 32768 (32 KB) as a representative size.
    constexpr int COPY_BYTES = 32768;
    constexpr int ITERS = 2000;
    constexpr int RUNS = 11;

    // Need space for N_PIPE * COPY_BYTES destinations
    constexpr int MAX_PIPE = 16;
    constexpr int DST_BYTES = MAX_PIPE * COPY_BYTES + COPY_BYTES;
    float* g_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&g_dst, DST_BYTES));

    #define LAUNCH_TP(MODE_, PIPE_) \
        [](BulkResult* r, int iters, int smem, float* dst) { \
            if (smem > 48*1024) cudaFuncSetAttribute( \
                kernel_bulk_throughput<MODE_, COPY_BYTES, PIPE_>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            kernel_bulk_throughput<MODE_, COPY_BYTES, PIPE_><<<1, 128, smem>>>(r, iters, dst); \
        }

    struct TpConfig {
        int n_pipe;
        int mode;
        const char* mode_str;
        LauncherFunc launcher;
    };

    TpConfig configs[] = {
        // Plain copy sweep
        {1,  0, "copy", LAUNCH_TP(0, 1)},
        {2,  0, "copy", LAUNCH_TP(0, 2)},
        {4,  0, "copy", LAUNCH_TP(0, 4)},
        {8,  0, "copy", LAUNCH_TP(0, 8)},
        {16, 0, "copy", LAUNCH_TP(0, 16)},
        // Reduce_add sweep
        {1,  1, "reduce_add", LAUNCH_TP(1, 1)},
        {2,  1, "reduce_add", LAUNCH_TP(1, 2)},
        {4,  1, "reduce_add", LAUNCH_TP(1, 4)},
        {8,  1, "reduce_add", LAUNCH_TP(1, 8)},
        {16, 1, "reduce_add", LAUNCH_TP(1, 16)},
    };
    #undef LAUNCH_TP

    for (auto& cfg : configs) {
        char x_var[32];
        snprintf(x_var, sizeof(x_var), "%s_p%d", cfg.mode_str, cfg.n_pipe);
        fprintf(stderr, "  %s...\n", x_var);

        run_bulk("bulk_throughput", x_var, cfg.mode_str,
                 COPY_BYTES, cfg.n_pipe, 0, ITERS, cfg.n_pipe,
                 cfg.launcher, COPY_BYTES, RUNS, verbose,
                 g_dst, DST_BYTES);
    }

    CUDA_CHECK(cudaFree(g_dst));
}

// ---------------------------------------------------------------------------
// Experiment: overlap — async overlap detection
// ---------------------------------------------------------------------------
static void run_overlap(bool verbose) {
    fprintf(stderr, "\n=== Experiment: bulk_overlap ===\n");

    constexpr int COPY_BYTES = 32768;
    constexpr int ITERS = 2000;
    constexpr int RUNS = 11;

    constexpr int DST_BYTES = 16 * COPY_BYTES + COPY_BYTES;
    float* g_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&g_dst, DST_BYTES));

    #define LAUNCH_OVERLAP(MODE_, COMP_) \
        [](BulkResult* r, int iters, int smem, float* dst) { \
            if (smem > 48*1024) cudaFuncSetAttribute( \
                kernel_bulk_overlap<MODE_, COPY_BYTES, COMP_>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            kernel_bulk_overlap<MODE_, COPY_BYTES, COMP_><<<1, 128, smem>>>(r, iters, dst); \
        }

    #define LAUNCH_COMPUTE(COMP_) \
        [](BulkResult* r, int iters, int smem, float* dst) { \
            kernel_compute_only<COMP_><<<1, 128>>>(r, iters, dst); \
        }

    struct OverlapConfig {
        int compute_iters;
        int mode;
        const char* mode_str;
        const char* x_var_prefix;
        LauncherFunc launcher;
    };

    // Test: copy_only (compute_iters=0 handled by latency experiment),
    //       compute_only, and copy+compute overlap
    OverlapConfig configs[] = {
        // copy + compute overlap (plain copy)
        {100,  0, "copy+compute", "copy_c100",  LAUNCH_OVERLAP(0, 100)},
        {500,  0, "copy+compute", "copy_c500",  LAUNCH_OVERLAP(0, 500)},
        {1000, 0, "copy+compute", "copy_c1000", LAUNCH_OVERLAP(0, 1000)},
        // copy + compute overlap (reduce_add)
        {100,  1, "reduce+compute", "reduce_c100",  LAUNCH_OVERLAP(1, 100)},
        {500,  1, "reduce+compute", "reduce_c500",  LAUNCH_OVERLAP(1, 500)},
        {1000, 1, "reduce+compute", "reduce_c1000", LAUNCH_OVERLAP(1, 1000)},
        // compute-only baselines
        {100,  -1, "compute_only", "compute_c100",  LAUNCH_COMPUTE(100)},
        {500,  -1, "compute_only", "compute_c500",  LAUNCH_COMPUTE(500)},
        {1000, -1, "compute_only", "compute_c1000", LAUNCH_COMPUTE(1000)},
    };
    #undef LAUNCH_OVERLAP
    #undef LAUNCH_COMPUTE

    for (auto& cfg : configs) {
        fprintf(stderr, "  %s...\n", cfg.x_var_prefix);

        int smem = (cfg.mode >= 0) ? COPY_BYTES : 0;
        run_bulk("bulk_overlap", cfg.x_var_prefix, cfg.mode_str,
                 (cfg.mode >= 0) ? COPY_BYTES : 0,
                 1, cfg.compute_iters, ITERS, 1,
                 cfg.launcher, smem, RUNS, verbose,
                 g_dst, DST_BYTES);
    }

    CUDA_CHECK(cudaFree(g_dst));
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

    print_csv_header();

    bool run_all = (experiment == "all");

    if (run_all || experiment == "latency")
        run_latency(verbose);

    if (run_all || experiment == "throughput")
        run_throughput(verbose);

    if (run_all || experiment == "overlap")
        run_overlap(verbose);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
