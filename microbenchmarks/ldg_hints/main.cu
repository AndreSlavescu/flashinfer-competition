// LDG.256 hint sweep microbenchmark
// Measures ld.global[.nc].L1::<hint>.L2::<hint>.v4.u64 (256-bit wide loads)
// across all valid L1/L2/NC/prefetch hint combinations
// Target: B200 (sm100a)
//
// Experiments:
//   throughput    — streaming 256-bit loads with 32 threads (single warp), 256 MB
//   competition   — sparse scatter pattern mimicking Warp 7 index loads
//   latency       — pointer-chase single-thread dependent loads (4 WS sizes)
//   pchase_sweep  — fine-grained WS sweep 4KB-512MB (arxiv:1509.02308 method)
//   all           — run everything
//
// CSV schema:
//   experiment,x_var,hint,working_set_MB,threads,loads_per_thread,iters,
//   total_cycles,cycles_per_load,ns_per_load,gbps,spread_pct

#include "ldg_kernels.cuh"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// File-scope statics for use in +[] lambdas (capture-less function pointers)
static uint64_t* g_ldg_data = nullptr;
static int g_ldg_ws_bytes = 0;

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------
static void print_csv_header() {
    printf("experiment,x_var,hint,working_set_MB,threads,loads_per_thread,iters,"
           "total_cycles,cycles_per_load,ns_per_load,gbps,spread_pct\n");
}

static void print_csv_row(
    const char* experiment, const char* x_var, const char* hint,
    double working_set_MB, int threads, int loads_per_thread, int iters,
    double total_cycles, double cycles_per_load, double ns_per_load,
    double gbps, double spread_pct)
{
    printf("%s,%s,%s,%.1f,%d,%d,%d,%.1f,%.2f,%.2f,%.3f,%.1f\n",
           experiment, x_var, hint,
           working_set_MB, threads, loads_per_thread, iters,
           total_cycles, cycles_per_load, ns_per_load, gbps, spread_pct);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Generic run framework
// ---------------------------------------------------------------------------
using LauncherFunc = void(*)(LdgResult*, int, int);

static void run_ldg(
    const char* experiment, const char* x_var, const char* hint,
    double working_set_MB, int threads, int loads_per_thread, int iters,
    int total_loads_per_iter,  // threads * loads_per_thread
    LauncherFunc launcher, int smem_bytes,
    int num_runs, bool verbose)
{
    LdgResult* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(LdgResult)));

    MeasurementSeries cycles_series;
    MeasurementSeries ns_series;

    for (int run = 0; run < num_runs; run++) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(LdgResult)));

        launcher(d_result, iters, smem_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());

        LdgResult h_result;
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(LdgResult),
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
    int total_loads = iters * total_loads_per_iter;
    double cycles_per_load = avg_total_cycles / total_loads;
    double ns_per_load = avg_total_ns / total_loads;
    // Each load = 32 bytes (256 bits). Aggregate BW = total_bytes / total_time.
    double gbps = (avg_total_ns > 0)
                  ? 32.0 * total_loads / avg_total_ns  // aggregate GB/s
                  : 0.0;
    double spread = cycles_series.spread() * 100.0;

    print_csv_row(experiment, x_var, hint,
                  working_set_MB, threads, loads_per_thread, iters,
                  avg_total_cycles, cycles_per_load, ns_per_load, gbps, spread);

    CUDA_CHECK(cudaFree(d_result));
}

// ---------------------------------------------------------------------------
// Experiment: throughput — streaming sequential loads, all hints
// ---------------------------------------------------------------------------
static void run_throughput(bool verbose) {
    fprintf(stderr, "\n=== Experiment: ldg_throughput ===\n");

    // 256 MB working set → must exceed 126.5 MB L2 to hit HBM
    constexpr int WS_BYTES = 256 * 1024 * 1024;
    constexpr int WS_ELEMS = WS_BYTES / 8;  // u64 elements
    // Each of 32 threads reads 4 u64 (32B) per iter through its own WS/32 sub-region.
    // Need iters * 4 >= WS_ELEMS/32 to cover entire buffer (no revisits = cold HBM).
    constexpr int THREADS = 32;
    constexpr int PER_THREAD_ELEMS = WS_ELEMS / THREADS;  // 1,048,576
    constexpr int ITERS = PER_THREAD_ELEMS / 4;  // 262,144 — single pass through 256 MB
    constexpr int RUNS = 11;

    uint64_t* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, WS_BYTES));
    // Fill with non-zero data (some hints may behave differently on zeros)
    CUDA_CHECK(cudaMemset(d_data, 0xAB, WS_BYTES));

    g_ldg_data = d_data;

    #define LAUNCH_TP(HINT_) \
        +[](LdgResult* r, int iters, int) { \
            kernel_ldg_throughput<HINT_, WS_ELEMS><<<1, 32>>>(r, iters, g_ldg_data); \
        }

    struct TpConfig {
        int hint;
        LauncherFunc launcher;
    };

    TpConfig configs[] = {
        {H_EF_EN_128B,     LAUNCH_TP(H_EF_EN_128B)},
        {H_EN_EN_128B,     LAUNCH_TP(H_EN_EN_128B)},
        {H_EL_EN_128B,     LAUNCH_TP(H_EL_EN_128B)},
        {H_EU_EN_128B,     LAUNCH_TP(H_EU_EN_128B)},
        {H_NA_EN_128B,     LAUNCH_TP(H_NA_EN_128B)},
        {H_EN_EF_128B,     LAUNCH_TP(H_EN_EF_128B)},
        {H_EN_EL_128B,     LAUNCH_TP(H_EN_EL_128B)},
        {H_NA_EF_64B,      LAUNCH_TP(H_NA_EF_64B)},
        {H_NA_EF_128B,     LAUNCH_TP(H_NA_EF_128B)},
        {H_NA_EF_256B,     LAUNCH_TP(H_NA_EF_256B)},
        {H_NC_EF_EN_128B,  LAUNCH_TP(H_NC_EF_EN_128B)},
        {H_NC_EN_EN_128B,  LAUNCH_TP(H_NC_EN_EN_128B)},
        {H_NC_EL_EN_128B,  LAUNCH_TP(H_NC_EL_EN_128B)},
        {H_NC_NA_EF_128B,  LAUNCH_TP(H_NC_NA_EF_128B)},
        {H_NC_NA_EF_64B,   LAUNCH_TP(H_NC_NA_EF_64B)},
        {H_NC_NA_EF_256B,  LAUNCH_TP(H_NC_NA_EF_256B)},
        {H_EL_EF_128B,     LAUNCH_TP(H_EL_EF_128B)},
        {H_NA_EL_128B,     LAUNCH_TP(H_NA_EL_128B)},
    };
    #undef LAUNCH_TP

    for (auto& cfg : configs) {
        const char* name = ldg_hint_name(cfg.hint);
        fprintf(stderr, "  throughput %s...\n", name);

        run_ldg("ldg_throughput", name, name,
                256.0, THREADS, 1, ITERS, THREADS,
                cfg.launcher, 0, RUNS, verbose);
    }

    CUDA_CHECK(cudaFree(d_data));
    g_ldg_data = nullptr;
}

// ---------------------------------------------------------------------------
// Experiment: competition — sparse scatter index loads
// ---------------------------------------------------------------------------
static void run_competition(bool verbose) {
    fprintf(stderr, "\n=== Experiment: ldg_competition ===\n");

    // 8462 pages × 64 tokens × 32 bytes per 256-bit load = ~16.4 GB
    // We'll use a smaller buffer but scatter across it
    constexpr int TOTAL_TOKENS = 8462 * 64;  // 541,568
    constexpr int BYTES_PER_TOKEN = 32;       // one LDG.256 per token
    constexpr int DATA_BYTES = TOTAL_TOKENS * BYTES_PER_TOKEN;  // ~16.4 MB
    constexpr int ITERS = 2000;
    constexpr int RUNS = 11;
    constexpr int THREADS = 32;
    constexpr int LOADS_PER_THREAD = 8;  // 2048 indices / 32 threads / (32B/load / 4B/index)

    uint64_t* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, DATA_BYTES));
    CUDA_CHECK(cudaMemset(d_data, 0xCD, DATA_BYTES));

    g_ldg_data = d_data;

    #define LAUNCH_COMP(HINT_) \
        +[](LdgResult* r, int iters, int) { \
            kernel_ldg_competition<HINT_><<<1, 32>>>(r, iters, g_ldg_data, TOTAL_TOKENS); \
        }

    // Top candidates for competition + key baselines
    struct CompConfig {
        int hint;
        LauncherFunc launcher;
    };

    CompConfig configs[] = {
        {H_EN_EN_128B,     LAUNCH_COMP(H_EN_EN_128B)},
        {H_NA_EF_128B,     LAUNCH_COMP(H_NA_EF_128B)},
        {H_NA_EF_64B,      LAUNCH_COMP(H_NA_EF_64B)},
        {H_NA_EF_256B,     LAUNCH_COMP(H_NA_EF_256B)},
        {H_EF_EN_128B,     LAUNCH_COMP(H_EF_EN_128B)},
        {H_EL_EF_128B,     LAUNCH_COMP(H_EL_EF_128B)},
        {H_NC_EN_EN_128B,  LAUNCH_COMP(H_NC_EN_EN_128B)},
        {H_NC_NA_EF_128B,  LAUNCH_COMP(H_NC_NA_EF_128B)},
        {H_NC_NA_EF_64B,   LAUNCH_COMP(H_NC_NA_EF_64B)},
        {H_NC_NA_EF_256B,  LAUNCH_COMP(H_NC_NA_EF_256B)},
    };
    #undef LAUNCH_COMP

    for (auto& cfg : configs) {
        const char* name = ldg_hint_name(cfg.hint);
        fprintf(stderr, "  competition %s...\n", name);

        run_ldg("ldg_competition", name, name,
                (double)DATA_BYTES / (1024.0 * 1024.0),
                THREADS, LOADS_PER_THREAD, ITERS,
                THREADS * LOADS_PER_THREAD,
                cfg.launcher, 0, RUNS, verbose);
    }

    CUDA_CHECK(cudaFree(d_data));
    g_ldg_data = nullptr;
}

// ---------------------------------------------------------------------------
// Experiment: latency — pointer-chase dependent loads
// ---------------------------------------------------------------------------
// Absolute-pointer p-chase setup (following gpu-benches and arxiv:1509.02308 methodology).
// Each 32-byte slot (4 × u64) stores an absolute device pointer in slot[0] to the next
// slot in a random Hamiltonian cycle. The kernel chase step is just:
//     ptr = (const uint64_t*)buf[0];   // zero ALU overhead
//
// d_data is the device pointer that the chase data will be copied to.
// The chase visits every slot exactly once before cycling, ensuring the full
// working set is exercised.
static void setup_chase_data_ptrs(uint64_t* h_data, const uint64_t* d_data, int num_elems) {
    int num_slots = num_elems / 4;  // each slot = 4 u64 = 32 bytes

    // Create a random Hamiltonian cycle through all slots
    std::vector<int> perm(num_slots);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(perm.begin(), perm.end(), rng);

    // slot perm[i] → slot perm[i+1], with wrap-around forming a complete ring
    for (int i = 0; i < num_slots; i++) {
        int curr = perm[i];
        int next = perm[(i + 1) % num_slots];
        // Store absolute device pointer to next slot's first u64
        h_data[curr * 4 + 0] = (uint64_t)(d_data + next * 4);
        h_data[curr * 4 + 1] = 0xAAAAAAAAAAAAAAAAull;
        h_data[curr * 4 + 2] = 0xBBBBBBBBBBBBBBBBull;
        h_data[curr * 4 + 3] = 0xCCCCCCCCCCCCCCCCull;
    }
}

static void run_latency(bool verbose) {
    fprintf(stderr, "\n=== Experiment: ldg_latency ===\n");

    constexpr int RUNS = 11;

    struct LatencyWS {
        int bytes;
        int iters;
        const char* label;
    };

    // iters should be >= num_slots (bytes/32) for full ring traversal.
    // For quick results, use ~1× pass for small WS. For 256 MB, use 500K iters
    // (~6% of ring) as a quick probe — use pchase_sweep for thorough measurement.
    LatencyWS worksets[] = {
        {8 * 1024,         20000, "8KB"},           // L1 hot (256 slots, 78× pass)
        {256 * 1024,       10000, "256KB"},          // L2 (8K slots, 1.25× pass)
        {4 * 1024 * 1024,  131072, "4MB"},           // L2 (128K slots, 1× pass)
        {256 * 1024 * 1024, 500000, "256MB"},        // HBM probe (8M slots, ~6% — quick)
    };

    for (auto& ws : worksets) {
        int num_elems = ws.bytes / 8;  // u64 elements

        // Allocate device buffer FIRST, then setup chase with absolute device pointers
        uint64_t* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, ws.bytes));

        uint64_t* h_data = (uint64_t*)malloc(ws.bytes);
        setup_chase_data_ptrs(h_data, d_data, num_elems);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, ws.bytes, cudaMemcpyHostToDevice));
        free(h_data);

        g_ldg_data = d_data;

        #define LAUNCH_LAT(HINT_) \
            +[](LdgResult* r, int iters, int) { \
                kernel_ldg_latency<HINT_><<<1, 32>>>(r, iters, g_ldg_data); \
            }

        // Test key hints at each working set size
        struct LatConfig {
            int hint;
            LauncherFunc launcher;
        };

        LatConfig configs[] = {
            {H_EN_EN_128B,     LAUNCH_LAT(H_EN_EN_128B)},
            {H_NA_EF_128B,     LAUNCH_LAT(H_NA_EF_128B)},
            {H_EF_EN_128B,     LAUNCH_LAT(H_EF_EN_128B)},
            {H_EL_EN_128B,     LAUNCH_LAT(H_EL_EN_128B)},
            {H_NC_EN_EN_128B,  LAUNCH_LAT(H_NC_EN_EN_128B)},
            {H_NC_NA_EF_128B,  LAUNCH_LAT(H_NC_NA_EF_128B)},
        };
        #undef LAUNCH_LAT

        for (auto& cfg : configs) {
            const char* hint_name = ldg_hint_name(cfg.hint);
            char x_var[64];
            snprintf(x_var, sizeof(x_var), "%s_%s", ws.label, hint_name);
            fprintf(stderr, "  latency %s...\n", x_var);

            run_ldg("ldg_latency", x_var, hint_name,
                    (double)ws.bytes / (1024.0 * 1024.0),
                    1, 1, ws.iters, 1,
                    cfg.launcher, 0, RUNS, verbose);
        }

        CUDA_CHECK(cudaFree(d_data));
        g_ldg_data = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Experiment: pchase_sweep — fine-grained WS sweep (arxiv:1509.02308 method)
// Geometric progression from 4 KB to 256 MB to detect L1/L2/HBM boundaries.
// Uses absolute-pointer p-chase for pure load latency measurement.
//
// CRITICAL: iters must be >= num_slots (= bytes/32) to ensure every slot in
// the Hamiltonian ring is visited at least once. Otherwise the active WS is
// only iters*32 bytes (L1/L2-cached), not the allocated WS size.
// Reference: gpu-benches uses max(LEN, 100000); NVIDIA-Hopper uses max(LEN, 1000000).
// ---------------------------------------------------------------------------
static void run_pchase_sweep(bool verbose) {
    fprintf(stderr, "\n=== Experiment: ldg_pchase_sweep ===\n");

    constexpr int RUNS = 5;
    constexpr int MIN_ITERS = 20000;  // minimum for timing stability

    struct SweepPoint {
        int bytes;
        const char* label;
    };

    // Geometric progression covering L1 (≤48KB), L2 (≤126.5MB), HBM (>126.5MB)
    SweepPoint points[] = {
        {4 * 1024,         "4KB"},
        {8 * 1024,         "8KB"},
        {16 * 1024,        "16KB"},
        {32 * 1024,        "32KB"},
        {48 * 1024,        "48KB"},     // L1 boundary
        {64 * 1024,        "64KB"},
        {128 * 1024,       "128KB"},
        {256 * 1024,       "256KB"},
        {512 * 1024,       "512KB"},
        {1024 * 1024,      "1MB"},
        {2 * 1024 * 1024,  "2MB"},
        {4 * 1024 * 1024,  "4MB"},
        {8 * 1024 * 1024,  "8MB"},
        {16 * 1024 * 1024, "16MB"},
        {32 * 1024 * 1024, "32MB"},
        {64 * 1024 * 1024, "64MB"},
        {128 * 1024 * 1024,"128MB"},     // L2 boundary (~126.5 MB)
        {192 * 1024 * 1024,"192MB"},
        {256 * 1024 * 1024,"256MB"},     // deep HBM
    };

    // Key hints: baseline (L1+L2 normal), L1-bypass, non-coherent
    int sweep_hints[] = {H_EN_EN_128B, H_NA_EF_128B, H_NC_EN_EN_128B};
    constexpr int NUM_HINTS = 3;

    LdgResult* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(LdgResult)));

    for (auto& pt : points) {
        int num_elems = pt.bytes / 8;
        int num_slots = num_elems / 4;  // each slot = 32 bytes (4 × u64)
        // iters must be >= num_slots for at least one full ring traversal.
        // For small WS where num_slots < MIN_ITERS, we revisit (still exercises full WS).
        int iters = (num_slots > MIN_ITERS) ? num_slots : MIN_ITERS;

        uint64_t* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, pt.bytes));

        uint64_t* h_data = (uint64_t*)malloc(pt.bytes);
        setup_chase_data_ptrs(h_data, d_data, num_elems);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, pt.bytes, cudaMemcpyHostToDevice));
        free(h_data);

        g_ldg_data = d_data;

        fprintf(stderr, "  WS=%s  num_slots=%d  iters=%d\n", pt.label, num_slots, iters);

        for (int hi = 0; hi < NUM_HINTS; hi++) {
            int hint = sweep_hints[hi];
            const char* hint_name = ldg_hint_name(hint);

            char x_var[64];
            snprintf(x_var, sizeof(x_var), "%s_%s", pt.label, hint_name);
            fprintf(stderr, "    pchase %s ...\n", x_var);

            MeasurementSeries cycles_s, ns_s;

            for (int run = 0; run < RUNS; run++) {
                CUDA_CHECK(cudaMemset(d_result, 0, sizeof(LdgResult)));

                // Compile-time dispatch via switch on hint
                switch (hint) {
                    case H_EN_EN_128B:
                        kernel_ldg_latency<H_EN_EN_128B><<<1, 32>>>(d_result, iters, g_ldg_data);
                        break;
                    case H_NA_EF_128B:
                        kernel_ldg_latency<H_NA_EF_128B><<<1, 32>>>(d_result, iters, g_ldg_data);
                        break;
                    case H_NC_EN_EN_128B:
                        kernel_ldg_latency<H_NC_EN_EN_128B><<<1, 32>>>(d_result, iters, g_ldg_data);
                        break;
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                LdgResult h_result;
                CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(LdgResult),
                                      cudaMemcpyDeviceToHost));

                cycles_s.add((double)h_result.total_cycles);
                ns_s.add((double)(h_result.gt_end_ns - h_result.gt_start_ns));

                if (verbose) {
                    fprintf(stderr, "      run %d/%d: %lu cy, %.1f ns\n",
                            run + 1, RUNS, h_result.total_cycles,
                            (double)(h_result.gt_end_ns - h_result.gt_start_ns));
                }
            }

            double avg_cy = cycles_s.value();
            double avg_ns = ns_s.value();
            double cy_per_load = avg_cy / iters;
            double ns_per_load = avg_ns / iters;
            // Single-thread: 1 load at a time, 32 bytes each
            double gbps = (ns_per_load > 0) ? 32.0 / ns_per_load : 0.0;
            double spread = cycles_s.spread() * 100.0;

            print_csv_row("ldg_pchase_sweep", x_var, hint_name,
                          (double)pt.bytes / (1024.0 * 1024.0),
                          1, 1, iters, avg_cy,
                          cy_per_load, ns_per_load, gbps, spread);
        }

        CUDA_CHECK(cudaFree(d_data));
    }

    CUDA_CHECK(cudaFree(d_result));
    g_ldg_data = nullptr;
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

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "GPU: %s (SM %d.%d, %d SMs)\n",
            prop.name, prop.major, prop.minor,
            prop.multiProcessorCount);

    fprintf(stderr, "Warming up GPU clocks...\n");
    gpu_clock_warmup(device);
    fprintf(stderr, "Warmup complete.\n");

    print_csv_header();

    bool run_all = (experiment == "all");

    if (run_all || experiment == "throughput")
        run_throughput(verbose);

    if (run_all || experiment == "competition")
        run_competition(verbose);

    if (run_all || experiment == "latency")
        run_latency(verbose);

    if (run_all || experiment == "pchase_sweep")
        run_pchase_sweep(verbose);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
