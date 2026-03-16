#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "common.cuh"
#include "tensor_map_utils.cuh"
#include "index_patterns.cuh"
#include "gather4_kernels.cuh"

// ---------------------------------------------------------------------------
// CSV header
// ---------------------------------------------------------------------------
static void print_csv_header() {
    printf("experiment,x_var,config,data_type,dim0,box_dim0,bytes_per_row,"
           "bytes_per_gather4,col_steps,num_blocks,pattern,cache_hint,"
           "swizzle,total_data_MB,n_outstanding,num_iters,"
           "throughput_GBps,avg_latency_ns,"
           "net_latency_ns,spread_pct\n");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns the swizzle size in bytes, or 0 for SWIZZLE_NONE
static size_t swizzle_size_bytes(CUtensorMapSwizzle s) {
    switch (s) {
        case CU_TENSOR_MAP_SWIZZLE_32B:  return 32;
        case CU_TENSOR_MAP_SWIZZLE_64B:  return 64;
        case CU_TENSOR_MAP_SWIZZLE_128B: return 128;
        default: return 0;  // NONE = no constraint
    }
}

// Validate: box_dim0 * sizeof(element) <= swizzle_bytes (or swizzle=NONE)
static bool is_valid_swizzle_config(CUtensorMapDataType dtype, uint32_t box_dim0,
                                    CUtensorMapSwizzle swizzle) {
    size_t sw_bytes = swizzle_size_bytes(swizzle);
    if (sw_bytes == 0) return true;  // SWIZZLE_NONE always valid
    size_t box_bytes = box_dim0 * tensor_map_element_size(dtype);
    return box_bytes <= sw_bytes;
}

static const char* swizzle_name(CUtensorMapSwizzle s) {
    switch (s) {
        case CU_TENSOR_MAP_SWIZZLE_NONE:  return "none";
        case CU_TENSOR_MAP_SWIZZLE_32B:   return "32B";
        case CU_TENSOR_MAP_SWIZZLE_64B:   return "64B";
        case CU_TENSOR_MAP_SWIZZLE_128B:  return "128B";
        default: return "unknown";
    }
}

static const char* dtype_name(CUtensorMapDataType dt) {
    switch (dt) {
        case CU_TENSOR_MAP_DATA_TYPE_INT64:    return "int64";
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return "bf16";
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:  return "f32";
        default: return "unknown";
    }
}

struct BenchConfig {
    std::string      experiment;      // unique per experiment runner
    std::string      x_var;           // CSV column name of the swept variable (x-axis)
    TensorMapConfig  tm_cfg;
    int              num_blocks;
    IndexPattern     pattern;
    int64_t          cache_hint;
    size_t           total_data_MB;   // working set size in MB
    int              n_outstanding;   // for pipeline experiment
    int              num_iters;       // for latency experiment

    // Derived
    uint64_t num_rows() const {
        size_t row_bytes = tm_cfg.bytes_per_row();
        return (total_data_MB * 1024ULL * 1024ULL) / row_bytes;
    }
};

// ---------------------------------------------------------------------------
// Experiment 1/4/8: Throughput / Multi-CTA saturation
// ---------------------------------------------------------------------------

static void run_throughput(const BenchConfig& cfg, bool verbose) {
    uint64_t num_rows = cfg.num_rows();
    if (num_rows < 4) { fprintf(stderr, "Too few rows\n"); return; }

    // Allocate global tensor
    void* d_tensor = allocate_tensor(cfg.tm_cfg, num_rows);
    CUtensorMap tmap = create_tensor_map_from_config(cfg.tm_cfg, d_tensor, num_rows);

    // Determine how many gather4 calls total
    // Each call loads 4 rows. We want to traverse the entire working set.
    int total_gather4_calls = (int)(num_rows / 4);
    if (total_gather4_calls < cfg.num_blocks) total_gather4_calls = cfg.num_blocks;

    // Generate indices
    int num_indices = total_gather4_calls * 4;
    int* d_indices = create_device_indices(cfg.pattern, num_indices, (int)num_rows);

    // Timer buffer
    int64_t* d_timer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_timer, cfg.num_blocks * 3 * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_timer, 0, cfg.num_blocks * 3 * sizeof(int64_t)));

    int col_steps = cfg.tm_cfg.col_steps();
    int bytes_per_step = (int)(cfg.tm_cfg.box_dim0 * tensor_map_element_size(cfg.tm_cfg.data_type) * 4);
    int smem_data_size = bytes_per_step * col_steps;
    // Align to 8 bytes for mbarrier placement
    smem_data_size = (smem_data_size + 7) & ~7;
    size_t smem_bytes = smem_data_size + 16;

    if (verbose) {
        fprintf(stderr, "  throughput: blocks=%d rows=%lu gather4_calls=%d col_steps=%d\n",
                cfg.num_blocks, num_rows, total_gather4_calls, col_steps);
    }

    // Allow dynamic shared memory beyond the default 48KB limit
    if (smem_bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)throughput_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes));
    }

    // Warmup
    for (int w = 0; w < 3; w++) {
        throughput_kernel<<<cfg.num_blocks, 1, smem_bytes>>>(
            tmap, d_indices, total_gather4_calls, col_steps,
            (int)cfg.tm_cfg.box_dim0, bytes_per_step, smem_data_size,
            cfg.cache_hint, d_timer);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measured runs
    constexpr int NUM_RUNS = 11;
    MeasurementSeries ms_elapsed, ms_throughput;

    for (int r = 0; r < NUM_RUNS; r++) {
        CUDA_CHECK(cudaMemset(d_timer, 0, cfg.num_blocks * 3 * sizeof(int64_t)));

        throughput_kernel<<<cfg.num_blocks, 1, smem_bytes>>>(
            tmap, d_indices, total_gather4_calls, col_steps,
            (int)cfg.tm_cfg.box_dim0, bytes_per_step, smem_data_size,
            cfg.cache_hint, d_timer);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        auto result = read_kernel_timer(d_timer, cfg.num_blocks);
        double total_bytes = (double)total_gather4_calls * bytes_per_step * col_steps;
        double gbps = total_bytes / result.elapsed_ns;  // bytes/ns = GB/s

        ms_elapsed.add(result.elapsed_ms);
        ms_throughput.add(gbps);
    }

    // Compute per-gather4 latency: total_time / total_gather4_calls
    double avg_latency_ns = ms_elapsed.value() * 1e6 / total_gather4_calls;

    printf("%s,%s,%s,%s,%lu,%u,%zu,%zu,%d,%d,%s,%s,%s,%zu,,%d,"
           "%.2f,%.1f,,%.1f\n",
           cfg.experiment.c_str(),
           cfg.x_var.c_str(),
           cfg.tm_cfg.name,
           dtype_name(cfg.tm_cfg.data_type),
           cfg.tm_cfg.dim0,
           cfg.tm_cfg.box_dim0,
           cfg.tm_cfg.bytes_per_row(),
           cfg.tm_cfg.bytes_per_gather4(),
           col_steps,
           cfg.num_blocks,
           index_pattern_name(cfg.pattern),
           cache_hint_name(cfg.cache_hint),
           swizzle_name(cfg.tm_cfg.swizzle),
           cfg.total_data_MB,
           total_gather4_calls,
           ms_throughput.value(),
           avg_latency_ns,
           ms_throughput.spread() * 100.0);

    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_timer));
    CUDA_CHECK(cudaFree(d_tensor));
}

// ---------------------------------------------------------------------------
// Experiment 6: Latency (single CTA)
// ---------------------------------------------------------------------------

static void run_latency(const BenchConfig& cfg, bool verbose) {
    uint64_t num_rows = cfg.num_rows();
    if (num_rows < 4) { fprintf(stderr, "Too few rows\n"); return; }

    void* d_tensor = allocate_tensor(cfg.tm_cfg, num_rows);
    CUtensorMap tmap = create_tensor_map_from_config(cfg.tm_cfg, d_tensor, num_rows);

    // num_iters == 0 means "use num_rows/4": each gather4 accesses 4 unique rows,
    // stepping through the entire working set exactly once (HBM-cold measurement).
    int num_iters = (cfg.num_iters == 0) ? (int)(num_rows / 4) : cfg.num_iters;
    int num_indices = num_iters * 4;
    int* d_indices = create_device_indices(cfg.pattern, num_indices, (int)num_rows);

    int64_t* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, 2 * sizeof(int64_t)));

    int bytes_per_gather4 = (int)cfg.tm_cfg.bytes_per_gather4();
    // Align data region to 8B for mbarrier placement
    int smem_data_size = (bytes_per_gather4 + 7) & ~7;
    size_t smem_bytes = smem_data_size + 16;

    if (verbose) {
        fprintf(stderr, "  latency: rows=%lu iters=%d bytes_per_gather4=%d\n",
                num_rows, num_iters, bytes_per_gather4);
    }

    // Warmup
    for (int w = 0; w < 3; w++) {
        latency_kernel<<<1, 1, smem_bytes>>>(
            tmap, d_indices, num_iters, bytes_per_gather4,
            cfg.cache_hint, d_results);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measured runs
    constexpr int NUM_RUNS = 11;
    MeasurementSeries ns_total, ns_null, ns_net;  // store nanoseconds per iteration

    for (int r = 0; r < NUM_RUNS; r++) {
        latency_kernel<<<1, 1, smem_bytes>>>(
            tmap, d_indices, num_iters, bytes_per_gather4,
            cfg.cache_hint, d_results);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        int64_t h_results[2];
        CUDA_CHECK(cudaMemcpy(h_results, d_results, 2 * sizeof(int64_t),
                              cudaMemcpyDeviceToHost));

        double total_ns = (double)h_results[0];  // total nanoseconds for all iters
        double null_ns  = (double)h_results[1];  // null baseline nanoseconds
        double net_ns   = total_ns - null_ns;

        ns_total.add(total_ns / num_iters);  // nanoseconds per gather4
        ns_null.add(null_ns / num_iters);
        ns_net.add(net_ns / num_iters);
    }

    // Compute throughput from total latency
    double total_bytes = (double)num_iters * cfg.tm_cfg.bytes_per_gather4();
    double gbps = total_bytes / (ns_total.value() * num_iters);  // bytes / ns = GB/s

    printf("%s,%s,%s,%s,%lu,%u,%zu,%zu,%d,1,%s,%s,%s,%zu,,%d,"
           "%.2f,%.1f,%.1f,%.1f\n",
           cfg.experiment.c_str(),
           cfg.x_var.c_str(),
           cfg.tm_cfg.name,
           dtype_name(cfg.tm_cfg.data_type),
           cfg.tm_cfg.dim0,
           cfg.tm_cfg.box_dim0,
           cfg.tm_cfg.bytes_per_row(),
           cfg.tm_cfg.bytes_per_gather4(),
           cfg.tm_cfg.col_steps(),
           index_pattern_name(cfg.pattern),
           cache_hint_name(cfg.cache_hint),
           swizzle_name(cfg.tm_cfg.swizzle),
           cfg.total_data_MB,
           num_iters,
           gbps,
           ns_total.value(),
           ns_net.value(),
           ns_net.spread() * 100.0);

    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_tensor));
}

// ---------------------------------------------------------------------------
// Experiment 7: Pipeline depth (single CTA)
// ---------------------------------------------------------------------------

static void run_pipeline(const BenchConfig& cfg, bool verbose) {
    uint64_t num_rows = cfg.num_rows();
    if (num_rows < 4) { fprintf(stderr, "Too few rows\n"); return; }

    void* d_tensor = allocate_tensor(cfg.tm_cfg, num_rows);
    CUtensorMap tmap = create_tensor_map_from_config(cfg.tm_cfg, d_tensor, num_rows);

    int N = cfg.n_outstanding;
    int bytes_per_gather4 = (int)cfg.tm_cfg.bytes_per_gather4();
    int num_outer_iters = 256;  // enough iterations for stable measurement
    int num_indices = num_outer_iters * N * 4;
    int* d_indices = create_device_indices(cfg.pattern, num_indices, (int)num_rows);

    int64_t* d_timer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_timer, 3 * sizeof(int64_t)));

    int smem_data_size = N * bytes_per_gather4;
    smem_data_size = (smem_data_size + 7) & ~7;
    size_t smem_bytes = smem_data_size + 16;

    if (verbose) {
        fprintf(stderr, "  pipeline: N=%d rows=%lu outer_iters=%d smem=%zuB\n",
                N, num_rows, num_outer_iters, smem_bytes);
    }

    auto kernel_fn = get_pipeline_kernel(N);

    // Allow dynamic shared memory beyond the default 48KB limit
    if (smem_bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)kernel_fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes));
    }

    // Warmup
    for (int w = 0; w < 3; w++) {
        CUDA_CHECK(cudaMemset(d_timer, 0, 3 * sizeof(int64_t)));
        kernel_fn<<<1, 1, smem_bytes>>>(
            tmap, d_indices, num_outer_iters, bytes_per_gather4,
            cfg.cache_hint, d_timer);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measured runs
    constexpr int NUM_RUNS = 11;
    MeasurementSeries ms_per_gather4, ms_gbps;

    for (int r = 0; r < NUM_RUNS; r++) {
        CUDA_CHECK(cudaMemset(d_timer, 0, 3 * sizeof(int64_t)));
        kernel_fn<<<1, 1, smem_bytes>>>(
            tmap, d_indices, num_outer_iters, bytes_per_gather4,
            cfg.cache_hint, d_timer);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        auto result = read_kernel_timer(d_timer, 1);
        double total_gather4s = (double)num_outer_iters * N;
        double ns_per_gather4 = result.elapsed_ns / total_gather4s;
        double gbps = (total_gather4s * bytes_per_gather4) / result.elapsed_ns;

        ms_per_gather4.add(ns_per_gather4);
        ms_gbps.add(gbps);
    }

    printf("%s,%s,%s,%s,%lu,%u,%zu,%zu,%d,1,%s,%s,%s,%zu,%d,%d,"
           "%.2f,%.1f,,%.1f\n",
           cfg.experiment.c_str(),
           cfg.x_var.c_str(),
           cfg.tm_cfg.name,
           dtype_name(cfg.tm_cfg.data_type),
           cfg.tm_cfg.dim0,
           cfg.tm_cfg.box_dim0,
           cfg.tm_cfg.bytes_per_row(),
           cfg.tm_cfg.bytes_per_gather4(),
           cfg.tm_cfg.col_steps(),
           index_pattern_name(cfg.pattern),
           cache_hint_name(cfg.cache_hint),
           swizzle_name(cfg.tm_cfg.swizzle),
           cfg.total_data_MB,
           N,
           num_outer_iters,
           ms_gbps.value(),
           ms_per_gather4.value(),
           ms_per_gather4.spread() * 100.0);

    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_timer));
    CUDA_CHECK(cudaFree(d_tensor));
}

// ===================================================================
// Experiment runners
// ===================================================================

// Experiment 1: Throughput vs box size
static void experiment_throughput(bool verbose) {
    if (verbose) fprintf(stderr, "[Experiment 1: Throughput vs Box Size]\n");

    struct BoxSweep {
        CUtensorMapDataType dtype;
        uint32_t            box_dim0;
        CUtensorMapSwizzle  swizzle;
    };

    std::vector<BoxSweep> sweeps = {
        // INT64 packing: dim0=128 (512 BF16 = 1024B = 128 INT64), box varies
        {CU_TENSOR_MAP_DATA_TYPE_INT64, 8,   CU_TENSOR_MAP_SWIZZLE_NONE},   // 64B/step, 16 col_steps
        {CU_TENSOR_MAP_DATA_TYPE_INT64, 16,  CU_TENSOR_MAP_SWIZZLE_NONE},   // 128B/step, 8 col_steps
        {CU_TENSOR_MAP_DATA_TYPE_INT64, 32,  CU_TENSOR_MAP_SWIZZLE_NONE},   // 256B/step, 4 col_steps
        {CU_TENSOR_MAP_DATA_TYPE_INT64, 64,  CU_TENSOR_MAP_SWIZZLE_NONE},   // 512B/step, 2 col_steps
        {CU_TENSOR_MAP_DATA_TYPE_INT64, 128, CU_TENSOR_MAP_SWIZZLE_NONE},   // 1024B/step, 1 col_step
        // BF16 native: dim0=512, box varies
        // Swizzle constraint: box_dim0 * sizeof(element) <= swizzle_bytes
        // SWIZZLE_128B max: 128B / 2B = 64 BF16 elements
        {CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 32,  CU_TENSOR_MAP_SWIZZLE_NONE},   // 64B/step, 16 col_steps
        {CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 64,  CU_TENSOR_MAP_SWIZZLE_128B},   // 128B/step, 8 col_steps
        {CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 128, CU_TENSOR_MAP_SWIZZLE_NONE},   // 256B/step, 4 col_steps
        {CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 256, CU_TENSOR_MAP_SWIZZLE_NONE},   // 512B/step, 2 col_steps
    };

    for (auto& s : sweeps) {
        uint64_t dim0 = (s.dtype == CU_TENSOR_MAP_DATA_TYPE_INT64) ? 128 : 512;
        TensorMapConfig tm = {
            "sweep", s.dtype, dim0, s.box_dim0,
            s.swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        };

        BenchConfig cfg;
        cfg.experiment = "throughput_box_sweep";
        cfg.x_var = "box_dim0";
        cfg.tm_cfg = tm;
        cfg.num_blocks = 76;
        cfg.pattern = IndexPattern::RANDOM;
        cfg.cache_hint = HINT_EVICT_LAST;
        cfg.total_data_MB = 256;

        if (verbose) fprintf(stderr, "  %s box=%u...\n", dtype_name(s.dtype), s.box_dim0);
        run_throughput(cfg, verbose);
    }
}

// Experiment 2: Cache hint comparison (one sub-experiment per working set size)
static void experiment_cache_hints(bool verbose) {
    if (verbose) fprintf(stderr, "[Experiment 2: Cache Hints]\n");

    std::vector<int64_t> hints = {HINT_EVICT_LAST, HINT_EVICT_FIRST, HINT_EVICT_NORMAL, HINT_NONE};
    std::vector<IndexPattern> patterns = {IndexPattern::SEQUENTIAL, IndexPattern::RANDOM, IndexPattern::CLUSTERED_64};
    std::vector<size_t> data_sizes = {8, 64, 256, 1024};

    for (auto sz : data_sizes) {
        std::string exp_name = "throughput_hints_" + std::to_string(sz) + "MB";
        for (auto hint : hints) {
            for (auto pat : patterns) {
                BenchConfig cfg;
                cfg.experiment = exp_name;
                cfg.x_var = "cache_hint";
                cfg.tm_cfg = config_ckv_int64();
                cfg.num_blocks = 76;
                cfg.pattern = pat;
                cfg.cache_hint = hint;
                cfg.total_data_MB = sz;

                if (verbose) fprintf(stderr, "  hint=%s pat=%s sz=%zuMB...\n",
                                     cache_hint_name(hint), index_pattern_name(pat), sz);
                run_throughput(cfg, verbose);
            }
        }
    }
}

// Experiment 3: Access pattern impact
static void experiment_patterns(bool verbose) {
    if (verbose) fprintf(stderr, "[Experiment 3: Access Patterns]\n");

    std::vector<IndexPattern> patterns = {
        IndexPattern::SEQUENTIAL, IndexPattern::RANDOM,
        IndexPattern::CLUSTERED_16, IndexPattern::CLUSTERED_64,
        IndexPattern::CLUSTERED_256, IndexPattern::COMPETITION_REALISTIC,
        IndexPattern::COMPETITION_REALISTIC_PAGE_TABLE
    };
    std::vector<size_t> data_sizes = {64, 256, 1024};

    for (auto sz : data_sizes) {
        std::string exp_name = "throughput_patterns_" + std::to_string(sz) + "MB";
        for (auto pat : patterns) {
            BenchConfig cfg;
            cfg.experiment = exp_name;
            cfg.x_var = "pattern";
            cfg.tm_cfg = config_ckv_int64();
            cfg.num_blocks = 76;
            cfg.pattern = pat;
            cfg.cache_hint = HINT_EVICT_LAST;
            cfg.total_data_MB = sz;

            if (verbose) fprintf(stderr, "  pat=%s sz=%zuMB...\n",
                                 index_pattern_name(pat), sz);
            run_throughput(cfg, verbose);
        }
    }
}

// Experiment 4: Swizzle mode comparison
static void experiment_swizzle(bool verbose) {
    if (verbose) fprintf(stderr, "[Experiment 4: Swizzle Modes]\n");

    struct SwizzleSweep {
        CUtensorMapDataType dtype;
        uint64_t            dim0;
        uint32_t            box_dim0;
        CUtensorMapSwizzle  swizzle;
    };

    std::vector<SwizzleSweep> sweeps;

    // For each box size that makes sense, try all valid swizzle modes
    // BF16 × 64 = 128B/row: NONE, 32B, 64B, 128B
    for (auto sw : {CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
                    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B}) {
        sweeps.push_back({CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 64, 64, sw});
    }
    // BF16 × 128 = 256B/row
    for (auto sw : {CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
                    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B}) {
        sweeps.push_back({CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 512, 128, sw});
    }
    // BF16 × 256 = 512B/row
    for (auto sw : {CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
                    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B}) {
        sweeps.push_back({CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 512, 256, sw});
    }
    // INT64 × 128 = 1024B/row
    for (auto sw : {CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
                    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B}) {
        sweeps.push_back({CU_TENSOR_MAP_DATA_TYPE_INT64, 128, 128, sw});
    }

    for (auto& s : sweeps) {
        // Skip invalid: box_dim0 * sizeof(element) must be <= swizzle_bytes
        if (!is_valid_swizzle_config(s.dtype, s.box_dim0, s.swizzle)) {
            if (verbose) fprintf(stderr, "  %s box=%u swizzle=%s... SKIPPED (box exceeds swizzle)\n",
                                 dtype_name(s.dtype), s.box_dim0, swizzle_name(s.swizzle));
            continue;
        }

        TensorMapConfig tm = {
            "swizzle_sweep", s.dtype, s.dim0, s.box_dim0,
            s.swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        };

        BenchConfig cfg;
        cfg.experiment = "throughput_swizzle";
        cfg.x_var = "swizzle";
        cfg.tm_cfg = tm;
        cfg.num_blocks = 76;
        cfg.pattern = IndexPattern::RANDOM;
        cfg.cache_hint = HINT_EVICT_LAST;
        cfg.total_data_MB = 256;

        if (verbose) fprintf(stderr, "  %s box=%u swizzle=%s...\n",
                             dtype_name(s.dtype), s.box_dim0, swizzle_name(s.swizzle));
        run_throughput(cfg, verbose);
    }
}

// Experiment 5: L2 working set size sweep
static void experiment_l2(bool verbose) {
    if (verbose) fprintf(stderr, "[Experiment 5: L2 Working Set]\n");

    std::vector<size_t> sizes = {1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048};

    // Sweep 1: L2_PROMOTION_L2_128B (existing baseline)
    for (auto sz : sizes) {
        BenchConfig cfg;
        cfg.experiment = "throughput_l2";
        cfg.x_var = "total_data_MB";
        cfg.tm_cfg = config_ckv_int64();
        cfg.num_blocks = 76;
        cfg.pattern = IndexPattern::RANDOM;
        cfg.cache_hint = HINT_EVICT_LAST;
        cfg.total_data_MB = sz;

        if (verbose) fprintf(stderr, "  L2_128B sz=%zuMB...\n", sz);
        run_throughput(cfg, verbose);
    }

    // Sweep 2: L2_PROMOTION_L2_64B — test whether smaller promotion sector size
    // shifts the L2 cliff or changes HBM-bound throughput
    for (auto sz : sizes) {
        BenchConfig cfg;
        cfg.experiment = "throughput_l2_64b";
        cfg.x_var = "total_data_MB";
        cfg.tm_cfg = config_ckv_int64_l2_64b();
        cfg.num_blocks = 76;
        cfg.pattern = IndexPattern::RANDOM;
        cfg.cache_hint = HINT_EVICT_LAST;
        cfg.total_data_MB = sz;

        if (verbose) fprintf(stderr, "  L2_64B sz=%zuMB...\n", sz);
        run_throughput(cfg, verbose);
    }
}

// Experiment 6: Latency
static void experiment_latency(bool verbose) {
    if (verbose) fprintf(stderr, "[Experiment 6: Latency]\n");

    struct LatencySweep {
        TensorMapConfig tm;
        IndexPattern    pat;
        int64_t         hint;
        size_t          data_MB;
        int             num_iters = 1024;  // 0 = use num_rows/4 (HBM-cold, non-repeating)
        const char*     exp_name  = "latency";
    };

    auto cfgs = std::vector<LatencySweep>{
        // Config A (INT64, 4096B/gather4) — sequential L2-hot (1MB << 64MB L2)
        {config_ckv_int64(), IndexPattern::SEQUENTIAL, HINT_EVICT_LAST, 1},
        // Config A — random, nominally 256MB but only 16MB unique data → still L2-warm
        {config_ckv_int64(), IndexPattern::RANDOM, HINT_EVICT_LAST, 256},
        // Config A — random, no hint
        {config_ckv_int64(), IndexPattern::RANDOM, HINT_NONE, 256},
        // Config B (BF16 box=64, 512B/gather4) — sequential
        {config_ckv_bf16(), IndexPattern::SEQUENTIAL, HINT_EVICT_LAST, 1},
        // Config B — random
        {config_ckv_bf16(), IndexPattern::RANDOM, HINT_EVICT_LAST, 256},
        // Config C (BF16 64d, 512B/gather4) — sequential
        {config_kpe_bf16(), IndexPattern::SEQUENTIAL, HINT_EVICT_LAST, 1},
        // Config C — random
        {config_kpe_bf16(), IndexPattern::RANDOM, HINT_EVICT_LAST, 256},

        // HBM-cold variants: num_iters=0 → access every row exactly once per run.
        // 1024MB working set >> 64MB L2; each row is accessed cold on most iterations.
        // Sequential pattern ensures non-repeating row access (row = 4*iter .. 4*iter+3).
        // Expected net_latency_ns >> 197ns (L2-hit) → reveals true HBM-miss TMA latency.
        {config_ckv_int64(), IndexPattern::SEQUENTIAL, HINT_EVICT_LAST, 1024, 0, "latency_cold"},
        {config_kpe_bf16(),  IndexPattern::SEQUENTIAL, HINT_EVICT_LAST, 1024, 0, "latency_cold"},
    };

    for (auto& s : cfgs) {
        BenchConfig cfg;
        cfg.experiment = s.exp_name;
        cfg.x_var = "config";
        cfg.tm_cfg = s.tm;
        cfg.num_blocks = 1;
        cfg.pattern = s.pat;
        cfg.cache_hint = s.hint;
        cfg.total_data_MB = s.data_MB;
        cfg.num_iters = s.num_iters;

        if (verbose) fprintf(stderr, "  %s pat=%s hint=%s sz=%zuMB iters=%d...\n",
                             s.tm.name, index_pattern_name(s.pat),
                             cache_hint_name(s.hint), s.data_MB, s.num_iters);
        run_latency(cfg, verbose);
    }
}

// Experiment 7: Pipeline depth
static void experiment_pipeline(bool verbose) {
    if (verbose) fprintf(stderr, "[Experiment 7: Pipeline Depth]\n");

    std::vector<int> n_values = {1, 2, 4, 8, 16, 32};

    for (int N : n_values) {
        BenchConfig cfg;
        cfg.experiment = "pipeline";
        cfg.x_var = "n_outstanding";
        cfg.tm_cfg = config_ckv_int64();
        cfg.num_blocks = 1;
        cfg.pattern = IndexPattern::RANDOM;
        cfg.cache_hint = HINT_EVICT_LAST;
        cfg.total_data_MB = 256;
        cfg.n_outstanding = N;

        if (verbose) fprintf(stderr, "  N=%d...\n", N);
        run_pipeline(cfg, verbose);
    }
}

// Experiment 8: Multi-CTA saturation
static void experiment_saturation(bool verbose) {
    if (verbose) fprintf(stderr, "[Experiment 8: Multi-CTA Saturation]\n");

    std::vector<int> block_counts = {
        1, 2, 4, 8, 16, 32, 48, 64, 76, 96, 114, 128, 144, 152,
        176, 192, 224, 256, 296  // extended: up to 148 SMs × 2 CTAs/SM
    };

    for (int nb : block_counts) {
        BenchConfig cfg;
        cfg.experiment = "throughput_saturation";
        cfg.x_var = "num_blocks";
        cfg.tm_cfg = config_ckv_int64();
        cfg.num_blocks = nb;
        cfg.pattern = IndexPattern::RANDOM;
        cfg.cache_hint = HINT_EVICT_LAST;
        cfg.total_data_MB = 256;

        if (verbose) fprintf(stderr, "  blocks=%d...\n", nb);
        run_throughput(cfg, verbose);
    }
}

// ===================================================================
// CLI
// ===================================================================

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --experiment=NAME   Run specific experiment:\n"
        "                        throughput  (Exp 1: box size sweep)\n"
        "                        hints       (Exp 2: cache hints)\n"
        "                        patterns    (Exp 3: access patterns, incl. page_table)\n"
        "                        swizzle     (Exp 4: swizzle modes)\n"
        "                        l2          (Exp 5: L2 working set; 128B+64B promotion)\n"
        "                        latency     (Exp 6: latency, incl. HBM-cold variant)\n"
        "                        pipeline    (Exp 7: pipeline depth)\n"
        "                        saturation  (Exp 8: multi-CTA saturation up to 296 CTAs)\n"
        "                        all         (run all experiments)\n"
        "  --device=N          GPU device index (default: 0)\n"
        "  --verbose           Print progress to stderr\n"
        "  --help              Show this help\n",
        prog);
}

int main(int argc, char** argv) {
    std::string experiment = "all";
    int device = 0;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--experiment=") == 0) {
            experiment = arg.substr(13);
        } else if (arg.find("--device=") == 0) {
            device = std::stoi(arg.substr(9));
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    CUDA_CHECK(cudaSetDevice(device));

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (verbose) {
        fprintf(stderr, "Device %d: %s (SM %d.%d, %d SMs, %.0f GB HBM)\n",
                device, prop.name, prop.major, prop.minor,
                prop.multiProcessorCount,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    // Ramp up GPU clocks
    if (verbose) fprintf(stderr, "Warming up GPU clocks...\n");
    gpu_clock_warmup(device);
    if (verbose) fprintf(stderr, "Clock warmup done.\n\n");

    print_csv_header();

    auto run_if = [&](const char* name, auto fn) {
        if (experiment == "all" || experiment == name) {
            fn(verbose);
        }
    };

    run_if("throughput", experiment_throughput);
    run_if("hints",      experiment_cache_hints);
    run_if("patterns",   experiment_patterns);
    run_if("swizzle",    experiment_swizzle);
    run_if("l2",         experiment_l2);
    run_if("latency",    experiment_latency);
    run_if("pipeline",   experiment_pipeline);
    run_if("saturation", experiment_saturation);

    if (verbose) fprintf(stderr, "\nAll experiments complete.\n");
    return 0;
}
