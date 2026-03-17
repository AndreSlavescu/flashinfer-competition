#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include "common.cuh"
#include "tensor_map_utils.cuh"
#include "dual_tma_kernels.cuh"

// ---------------------------------------------------------------------------
// Index generation (simplified — only SEQUENTIAL and RANDOM needed here)
// ---------------------------------------------------------------------------

static int* create_device_indices(const char* pattern, int num_indices, int num_rows) {
    std::vector<int> h_indices(num_indices);
    std::mt19937 rng(42);

    if (strcmp(pattern, "sequential") == 0) {
        for (int i = 0; i < num_indices; i++)
            h_indices[i] = i % num_rows;
    } else {
        std::uniform_int_distribution<int> dist(0, num_rows - 1);
        for (int i = 0; i < num_indices; i++)
            h_indices[i] = dist(rng);
    }

    int* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, num_indices * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ptr, h_indices.data(), num_indices * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d_ptr;
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

static void print_csv_header() {
    printf("experiment,x_var,"
           "mode,pattern,data_MB,kpe_data_MB,"
           "ckv_num_calls,kpe_num_calls,"
           "ckv_GBps,kpe_GBps,"
           "ckv_ns_per_gather4,kpe_ns_per_gather4\n");
}

// ---------------------------------------------------------------------------
// Run one dual-stream configuration
// ---------------------------------------------------------------------------

struct DualConfig {
    int         mode;          // 0=ckv_only, 1=kpe_only, 2=both
    const char* pattern;       // "random" or "sequential"
    size_t      data_MB;
};

static const char* mode_name(int m) {
    switch (m) {
        case 0: return "ckv_only";
        case 1: return "kpe_only";
        case 2: return "both";
        default: return "unknown";
    }
}

static void run_dual(const DualConfig& dc, bool verbose) {
    // CKV tensor: INT64 packed, 128 elements, 1024B/row, 4096B/gather4
    auto ckv_cfg = config_ckv_int64();
    // KPE tensor: BF16 native, 64 elements, 128B/row, 512B/gather4
    auto kpe_cfg = config_kpe_bf16();

    // Both tensors use the same working set size
    uint64_t ckv_num_rows = (dc.data_MB * 1024ULL * 1024ULL) / ckv_cfg.bytes_per_row();
    uint64_t kpe_num_rows = (dc.data_MB * 1024ULL * 1024ULL) / kpe_cfg.bytes_per_row();

    void* d_ckv = allocate_tensor(ckv_cfg, ckv_num_rows);
    void* d_kpe = allocate_tensor(kpe_cfg, kpe_num_rows);

    CUtensorMap ckv_map = create_tensor_map_from_config(ckv_cfg, d_ckv, ckv_num_rows);
    CUtensorMap kpe_map = create_tensor_map_from_config(kpe_cfg, d_kpe, kpe_num_rows);

    int bytes_ckv = (int)ckv_cfg.bytes_per_gather4();  // 4096
    int bytes_kpe = (int)kpe_cfg.bytes_per_gather4();  //  512

    // Use same number of gather4 calls for both streams, synchronized to the ckv row count.
    // This matches the competition kernel's 8:1 ckv:kpe byte ratio — with num_calls based
    // on ckv rows, kpe accesses only num_calls × 512B = data_MB/8 of unique data per run.
    // Example at 256MB random: ckv touches 256MB (L2-cold), kpe touches only 32MB from a
    // 256MB pool → ~8MB unique = L2-warm. This is intentional: kpe_cache naturally stays
    // L2-warm in production because its footprint is 8× smaller than ckv_cache.
    int num_calls = (int)(ckv_num_rows / 4);
    if (num_calls < 256) num_calls = 256;
    double kpe_data_accessed_MB = (double)num_calls * bytes_kpe / (1024.0 * 1024.0);

    int* d_ckv_idx = create_device_indices(dc.pattern, num_calls * 4, (int)ckv_num_rows);
    int* d_kpe_idx = create_device_indices(dc.pattern, num_calls * 4, (int)kpe_num_rows);

    int64_t* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, 2 * sizeof(int64_t)));

    // Shared memory: ckv_data + kpe_data + 2 mbarriers
    int mbar_offset = ((bytes_ckv + bytes_kpe) + 7) & ~7;
    size_t smem_bytes = mbar_offset + 16;

    if (verbose) {
        fprintf(stderr, "  %s %s %zuMB: ckv_rows=%lu kpe_rows=%lu calls=%d smem=%zuB\n",
                mode_name(dc.mode), dc.pattern, dc.data_MB,
                ckv_num_rows, kpe_num_rows, num_calls, smem_bytes);
    }

    // Warmup
    for (int w = 0; w < 3; w++) {
        CUDA_CHECK(cudaMemset(d_results, 0, 2 * sizeof(int64_t)));
        dual_tma_kernel<<<1, 64, smem_bytes>>>(
            ckv_map, kpe_map, d_ckv_idx, d_kpe_idx,
            num_calls, num_calls, bytes_ckv, bytes_kpe,
            dc.mode, HINT_EVICT_LAST, d_results);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measured runs
    constexpr int NUM_RUNS = 11;
    MeasurementSeries ckv_gbps_ms, kpe_gbps_ms, ckv_ns_ms, kpe_ns_ms;

    for (int r = 0; r < NUM_RUNS; r++) {
        CUDA_CHECK(cudaMemset(d_results, 0, 2 * sizeof(int64_t)));
        dual_tma_kernel<<<1, 64, smem_bytes>>>(
            ckv_map, kpe_map, d_ckv_idx, d_kpe_idx,
            num_calls, num_calls, bytes_ckv, bytes_kpe,
            dc.mode, HINT_EVICT_LAST, d_results);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        int64_t h[2];
        CUDA_CHECK(cudaMemcpy(h, d_results, 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));

        double ckv_ns = (h[0] > 0) ? (double)h[0] / num_calls : 0.0;
        double kpe_ns = (h[1] > 0) ? (double)h[1] / num_calls : 0.0;
        double ckv_gbs = (h[0] > 0) ? ((double)num_calls * bytes_ckv) / (double)h[0] : 0.0;
        double kpe_gbs = (h[1] > 0) ? ((double)num_calls * bytes_kpe) / (double)h[1] : 0.0;

        ckv_gbps_ms.add(ckv_gbs);
        kpe_gbps_ms.add(kpe_gbs);
        ckv_ns_ms.add(ckv_ns);
        kpe_ns_ms.add(kpe_ns);
    }

    printf("dual_tma,mode,%s,%s,%zu,%.1f,%d,%d,%.2f,%.2f,%.1f,%.1f\n",
           mode_name(dc.mode), dc.pattern, dc.data_MB, kpe_data_accessed_MB,
           num_calls, num_calls,
           ckv_gbps_ms.value(), kpe_gbps_ms.value(),
           ckv_ns_ms.value(), kpe_ns_ms.value());

    CUDA_CHECK(cudaFree(d_ckv_idx));
    CUDA_CHECK(cudaFree(d_kpe_idx));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_ckv));
    CUDA_CHECK(cudaFree(d_kpe));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int device = 0;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--device=") == 0) {
            device = std::stoi(arg.substr(9));
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: %s [--device=N] [--verbose]\n"
                "\nMeasures dual TMA stream interference on B200.\n"
                "Runs ckv-only, kpe-only, and both-concurrent across L2-hot and HBM-cold regimes.\n",
                argv[0]);
            return 0;
        }
    }

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (verbose) {
        fprintf(stderr, "Device %d: %s (SM %d.%d, %d SMs, %.0f GB HBM)\n",
                device, prop.name, prop.major, prop.minor,
                prop.multiProcessorCount,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    if (verbose) fprintf(stderr, "Warming up GPU clocks...\n");
    gpu_clock_warmup(device);
    if (verbose) fprintf(stderr, "Clock warmup done.\n\n");

    print_csv_header();

    // Sweep: mode × pattern/size
    struct { int mode; const char* pattern; size_t data_MB; } configs[] = {
        // L2-hot regime (256MB, random — 8MB unique footprint per stream)
        {0, "random",     256},   // ckv only
        {1, "random",     256},   // kpe only
        {2, "random",     256},   // both concurrent

        // HBM-cold regime (1024MB, sequential — every row unique)
        {0, "sequential", 1024},  // ckv only
        {1, "sequential", 1024},  // kpe only
        {2, "sequential", 1024},  // both concurrent
    };

    for (auto& c : configs) {
        DualConfig dc = {c.mode, c.pattern, c.data_MB};
        run_dual(dc, verbose);
    }

    if (verbose) fprintf(stderr, "\nDual TMA stream benchmark complete.\n");
    return 0;
}
