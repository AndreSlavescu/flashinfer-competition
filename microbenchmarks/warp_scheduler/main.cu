// Warp sub-core partitioning benchmark for B200 (SM100a)
//
// Discovers the warp-to-scheduler mapping on B200 using FFMA contention.
// Follows Citadel methodology (arxiv:1804.06826 §2.1).
//
// Experiments:
//   single_warp_baseline  — baseline cycles for each individual warp
//   two_warp_sweep        — all 28 pairs of 2 warps; contention_ratio reveals sharing
//   warp_count_scaling    — scaling throughput with 1,2,4,8 active warps
//   all                   — all of the above
//
// Usage:
//   ./warp_scheduler_bench [--experiment=<name>] [--iters=<N>] [--trials=<N>]

#include "../common/benchmark_common.cuh"
#include "warp_scheduler_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------
static constexpr int DEFAULT_ITERS   = 100000;   // FMA iterations per trial
static constexpr int DEFAULT_TRIALS  = 20;        // statistical trials

// ---------------------------------------------------------------------------
// Device-side buffers (shared across experiments)
// ---------------------------------------------------------------------------
static int64_t* d_elapsed  = nullptr;  // [BENCH_WARPS]
static float*   d_sink     = nullptr;  // [BENCH_WARPS * 32]

static void alloc_bufs() {
    CUDA_CHECK(cudaMalloc(&d_elapsed, BENCH_WARPS * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sink,    BENCH_WARPS * 32 * sizeof(float)));
}

// Run one kernel launch and return per-warp elapsed_cycles for active warps.
// Inactive warp entries are set to 0.
static void run_kernel(uint32_t active_mask, int iters,
                        int64_t out_cycles[BENCH_WARPS]) {
    // Clear elapsed buffer so inactive warp slots read as 0
    CUDA_CHECK(cudaMemset(d_elapsed, 0, BENCH_WARPS * sizeof(int64_t)));

    ffma_warp_contention_kernel<<<1, BENCH_THREADS>>>(
        active_mask, iters, d_elapsed, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out_cycles, d_elapsed,
                          BENCH_WARPS * sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
}

// Return SM clock in Hz (clockRate removed from cudaDeviceProp in CUDA 13)
static double get_sm_clock_hz() {
    int clockKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0));
    return clockKHz * 1e3;
}

// ---------------------------------------------------------------------------
// Experiment 1: single_warp_baseline
//   Run each warp in isolation; record baseline cycles.
//   Output: experiment,warp_id,cycles,gfma_per_s
// ---------------------------------------------------------------------------
// print_csv=true to emit CSV rows; false for silent baseline computation
static void run_single_warp_baseline(int iters, int num_trials,
                                      double baseline_out[BENCH_WARPS],
                                      bool print_csv = true) {
    const double sm_hz = get_sm_clock_hz();

    if (print_csv) printf("experiment,warp_id,cycles,gfma_per_s\n");

    int64_t cycles[BENCH_WARPS];
    for (int w = 0; w < BENCH_WARPS; w++) {
        uint32_t mask = 1u << w;
        MeasurementSeries cyc_series;

        // Warmup (2 runs discarded)
        for (int t = 0; t < 2; t++) {
            run_kernel(mask, iters, cycles);
        }
        // Measured trials
        for (int t = 0; t < num_trials; t++) {
            run_kernel(mask, iters, cycles);
            cyc_series.add((double)cycles[w]);
        }

        double avg_cycles  = cyc_series.value();
        double elapsed_sec = avg_cycles / sm_hz;
        // 8 FMAs per iter × 32 lanes
        double fmas        = 8.0 * iters * 32.0;
        double gfma_per_s  = fmas / (elapsed_sec * 1e9);

        baseline_out[w] = avg_cycles;
        fprintf(stderr, "  warp %d: %.0f cycles  spread=%.1f%%\n",
                w, avg_cycles, cyc_series.spread() * 100.0);
        if (print_csv) {
            printf("single_warp_baseline,%d,%.0f,%.3f\n",
                   w, avg_cycles, gfma_per_s);
        }
    }
    if (print_csv) printf("\n");
}

// ---------------------------------------------------------------------------
// Experiment 2: two_warp_sweep
//   All 28 pairs (a < b) from BENCH_WARPS=8 warps.
//   For each pair, measure per-warp cycles and compare to baseline.
//   Output: experiment,warp_a,warp_b,cycles_a,cycles_b,baseline_cycles,contention_ratio
// ---------------------------------------------------------------------------
static void run_two_warp_sweep(int iters, int num_trials,
                                const double baseline[BENCH_WARPS]) {
    printf("experiment,warp_a,warp_b,cycles_a,cycles_b,baseline_cycles,"
           "contention_ratio,likely_same_subcore\n");

    int64_t cycles[BENCH_WARPS];

    for (int a = 0; a < BENCH_WARPS; a++) {
        for (int b = a + 1; b < BENCH_WARPS; b++) {
            uint32_t mask = (1u << a) | (1u << b);
            MeasurementSeries cyc_a, cyc_b;

            // Warmup
            for (int t = 0; t < 2; t++) {
                run_kernel(mask, iters, cycles);
            }
            // Measured trials
            for (int t = 0; t < num_trials; t++) {
                run_kernel(mask, iters, cycles);
                cyc_a.add((double)cycles[a]);
                cyc_b.add((double)cycles[b]);
            }

            double avg_a = cyc_a.value();
            double avg_b = cyc_b.value();
            // Arithmetic mean of the two single-warp baselines as reference
            double base  = (baseline[a] + baseline[b]) / 2.0;
            // Contention ratio: >1 means warps shared a sub-core and contended
            // On B200: same sub-core ≈ 1.3, different ≈ 1.0 (pipeline overlap
            // prevents the theoretically-maximal 2.0x — FP32 units are shared
            // but the scheduler hides latency by interleaving warps).
            double max_cyc = std::max(avg_a, avg_b);
            double ratio   = max_cyc / base;
            // Threshold 1.1: midpoint between 1.0 (no contention) and 1.3 (contention)
            bool same_sc = (ratio > 1.1);

            fprintf(stderr,
                    "  (%d,%d): cyc_a=%.0f cyc_b=%.0f base=%.0f ratio=%.3f%s\n",
                    a, b, avg_a, avg_b, base, ratio,
                    same_sc ? " [SAME SUBCORE]" : "");
            printf("two_warp_sweep,%d,%d,%.0f,%.0f,%.0f,%.3f,%s\n",
                   a, b, avg_a, avg_b, base, ratio,
                   same_sc ? "yes" : "no");
        }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Experiment 3: warp_count_scaling
//   Tests how throughput scales with more warps per CTA.
//   Sweeps both sequential (0..N-1) and stride-4 patterns to expose sub-core groups.
//   Output: experiment,pattern,num_active,warp_mask_hex,cycles_max,gfma_per_s,scaling
// ---------------------------------------------------------------------------
static void run_warp_count_scaling(int iters, int num_trials,
                                    double single_warp_cycles) {
    const double sm_hz = get_sm_clock_hz();

    // Single-warp baseline throughput (for "scaling" ratio)
    double baseline_gfma = 8.0 * iters * 32.0 / (single_warp_cycles / sm_hz) / 1e9;

    printf("experiment,pattern,num_active,warp_mask_hex,cycles_max,gfma_per_s,scaling_vs_1warp\n");

    // (description, mask)
    struct TestCase { const char* label; uint32_t mask; };
    std::vector<TestCase> cases = {
        // Sequential 1..8 warps
        {"seq_1",  0x01},
        {"seq_2",  0x03},
        {"seq_4",  0x0f},
        {"seq_8",  0xff},
        // Stride-4 pairs (suspected same-sub-core pairs based on Volta warp_id%4)
        {"s4_0_4", 0x11},   // warps 0,4
        {"s4_1_5", 0x22},   // warps 1,5
        {"s4_2_6", 0x44},   // warps 2,6
        {"s4_3_7", 0x88},   // warps 3,7
        // Adjacent pairs (suspected different-sub-core if warp_id%4)
        {"adj_0_1", 0x03},  // warps 0,1  (same as seq_2)
        {"adj_0_2", 0x05},  // warps 0,2
        {"adj_0_3", 0x09},  // warps 0,3
        // Cross patterns
        {"mix_0145", 0x33}, // warps 0,1,4,5
        {"mix_0246", 0x55}, // warps 0,2,4,6
    };

    int64_t cycles[BENCH_WARPS];

    for (auto& tc : cases) {
        uint32_t mask = tc.mask;
        int num_active = __builtin_popcount(mask);
        MeasurementSeries cyc_max_series;

        // Warmup
        for (int t = 0; t < 2; t++) {
            run_kernel(mask, iters, cycles);
        }
        // Measured trials
        for (int t = 0; t < num_trials; t++) {
            run_kernel(mask, iters, cycles);
            int64_t mx = 0;
            for (int w = 0; w < BENCH_WARPS; w++) {
                if ((mask >> w) & 1u) mx = std::max(mx, cycles[w]);
            }
            cyc_max_series.add((double)mx);
        }

        double avg_max  = cyc_max_series.value();
        double elapsed  = avg_max / sm_hz;
        // Total FP32 FMAs across all active warps
        double total_fmas = (double)num_active * 8.0 * iters * 32.0;
        double gfma_per_s = total_fmas / (elapsed * 1e9);
        double scaling    = gfma_per_s / baseline_gfma;

        fprintf(stderr,
                "  %-12s mask=0x%02x n=%d  cycles_max=%.0f  %.3f GFMA/s  scaling=%.2fx\n",
                tc.label, mask, num_active, avg_max, gfma_per_s, scaling);
        printf("warp_count_scaling,%s,%d,0x%02x,%.0f,%.3f,%.3f\n",
               tc.label, num_active, mask, avg_max, gfma_per_s, scaling);
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------
static std::string get_arg(int argc, char** argv, const char* key, const char* def) {
    std::string prefix = std::string("--") + key + "=";
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], prefix.c_str(), prefix.size()) == 0)
            return argv[i] + prefix.size();
    }
    return def;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string experiment = get_arg(argc, argv, "experiment", "all");
    std::string iters_str  = std::to_string(DEFAULT_ITERS);
    std::string trials_str = std::to_string(DEFAULT_TRIALS);
    int iters  = std::stoi(get_arg(argc, argv, "iters",  iters_str.c_str()));
    int trials = std::stoi(get_arg(argc, argv, "trials", trials_str.c_str()));

    fprintf(stderr, "=== Warp Scheduler Benchmark ===\n");
    fprintf(stderr, "experiment=%s  iters=%d  trials=%d\n\n",
            experiment.c_str(), iters, trials);

    // Print GPU identity
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int smClockKHz = 0;
    cudaDeviceGetAttribute(&smClockKHz, cudaDevAttrClockRate, 0);
    fprintf(stderr, "GPU: %s  (SM %d.%d)  SMs=%d  clock=%.0f MHz\n\n",
            prop.name, prop.major, prop.minor,
            prop.multiProcessorCount,
            smClockKHz * 1e-3);

    // Warm up GPU clocks
    fprintf(stderr, "Warming up GPU...\n");
    gpu_clock_warmup();
    fprintf(stderr, "Warmup done.\n\n");

    // Allocate device buffers
    alloc_bufs();

    // Run experiments
    double baseline[BENCH_WARPS] = {};

    if (experiment == "single_warp_baseline" || experiment == "all") {
        fprintf(stderr, "--- single_warp_baseline ---\n");
        run_single_warp_baseline(iters, trials, baseline);
    }

    // For two_warp_sweep and warp_count_scaling we need the baseline.
    // If the user only runs one of those, compute it silently first.
    bool need_baseline = (experiment == "two_warp_sweep" ||
                          experiment == "warp_count_scaling");
    if (need_baseline) {
        fprintf(stderr, "--- computing baseline (silent) ---\n");
        run_single_warp_baseline(iters, trials, baseline, /*print_csv=*/false);
    }

    if (experiment == "two_warp_sweep" || experiment == "all") {
        fprintf(stderr, "--- two_warp_sweep ---\n");
        run_two_warp_sweep(iters, trials, baseline);
    }

    if (experiment == "warp_count_scaling" || experiment == "all") {
        // Single-warp reference: average of all 8 single-warp baselines
        double sum = 0.0;
        for (int w = 0; w < BENCH_WARPS; w++) sum += baseline[w];
        double single_warp_avg = sum / BENCH_WARPS;

        fprintf(stderr, "--- warp_count_scaling ---\n");
        fprintf(stderr, "  single_warp_avg=%.0f cycles  (throughput reference)\n",
                single_warp_avg);
        run_warp_count_scaling(iters, trials, single_warp_avg);
    }

    // Cleanup
    cudaFree(d_elapsed);
    cudaFree(d_sink);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
