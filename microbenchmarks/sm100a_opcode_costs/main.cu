#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

#include "opcode_kernels.cuh"


struct AggregateStats {
    double total_cycles = 0.0;
    double total_ns = 0.0;
    double cycles_per_op = 0.0;
    double ns_per_op = 0.0;
    double spread_pct = 0.0;
};


static std::string normalize_token(std::string value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        }
    }
    return out;
}


static bool matches_any(const std::string& request, std::initializer_list<const char*> aliases) {
    if (request == "ALL") {
        return true;
    }
    for (const char* alias : aliases) {
        if (request == normalize_token(alias)) {
            return true;
        }
    }
    return false;
}


static void print_csv_header() {
    printf("measurement_id,suite,role,batch,outer_iters,ops_per_iter,runs,total_cycles,total_ns,cycles_per_op,ns_per_op,spread_pct\n");
}


static void print_csv_row(
    const char* measurement_id,
    const char* suite,
    const char* role,
    int batch,
    int outer_iters,
    int ops_per_iter,
    int runs,
    const AggregateStats& stats) {
    printf(
        "%s,%s,%s,%d,%d,%d,%d,%.1f,%.1f,%.4f,%.4f,%.2f\n",
        measurement_id,
        suite,
        role,
        batch,
        outer_iters,
        ops_per_iter,
        runs,
        stats.total_cycles,
        stats.total_ns,
        stats.cycles_per_op,
        stats.ns_per_op,
        stats.spread_pct);
    fflush(stdout);
}


template <typename LaunchFn>
static AggregateStats run_aggregate(
    int runs,
    int outer_iters,
    int ops_per_iter,
    LaunchFn&& launch,
    bool verbose) {
    BenchResult* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(BenchResult)));

    MeasurementSeries cycles_series;
    MeasurementSeries ns_series;

    for (int run = 0; run < runs; ++run) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(BenchResult)));
        launch(d_result, outer_iters);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        BenchResult h_result{};
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(BenchResult), cudaMemcpyDeviceToHost));
        cycles_series.add(static_cast<double>(h_result.total_cycles));
        ns_series.add(static_cast<double>(h_result.gt_end_ns - h_result.gt_start_ns));

        if (verbose) {
            fprintf(stderr, "  run %d/%d: %.0f cycles %.1f ns\n",
                    run + 1, runs,
                    static_cast<double>(h_result.total_cycles),
                    static_cast<double>(h_result.gt_end_ns - h_result.gt_start_ns));
            fflush(stderr);
        }
    }

    CUDA_CHECK(cudaFree(d_result));

    AggregateStats stats;
    stats.total_cycles = cycles_series.value();
    stats.total_ns = ns_series.value();
    stats.cycles_per_op = stats.total_cycles / std::max(outer_iters * ops_per_iter, 1);
    stats.ns_per_op = stats.total_ns / std::max(outer_iters * ops_per_iter, 1);
    stats.spread_pct = cycles_series.spread() * 100.0;
    return stats;
}


template <typename TargetFn, typename BaselineFn>
static void run_sync_measurement(
    const char* measurement_id,
    int outer_iters,
    int runs,
    TargetFn&& target,
    BaselineFn&& baseline,
    bool verbose) {
    if (verbose) {
        fprintf(stderr, "[measure] %s (sync)\n", measurement_id);
        fflush(stderr);
    }
    auto target_stats = run_aggregate(runs, outer_iters, 1, target, verbose);
    print_csv_row(measurement_id, "sync", "target", 1, outer_iters, 1, runs, target_stats);
    auto baseline_stats = run_aggregate(runs, outer_iters, 1, baseline, verbose);
    print_csv_row(measurement_id, "sync", "baseline", 1, outer_iters, 1, runs, baseline_stats);
}


template <typename TargetFn, typename BaselineFn>
static void run_async_measurement(
    const char* measurement_id,
    int outer_iters,
    int runs,
    TargetFn&& target,
    BaselineFn&& baseline,
    bool verbose) {
    if (verbose) {
        fprintf(stderr, "[measure] %s (async)\n", measurement_id);
        fflush(stderr);
    }
    for (int batch : {1, 2, 4, 8, 16}) {
        auto target_stats = run_aggregate(
            runs,
            outer_iters,
            batch,
            [&](BenchResult* d_result, int outer) { target(batch, d_result, outer); },
            verbose);
        print_csv_row(measurement_id, "async", "target", batch, outer_iters, batch, runs, target_stats);

        auto baseline_stats = run_aggregate(
            runs,
            outer_iters,
            batch,
            [&](BenchResult* d_result, int outer) { baseline(batch, d_result, outer); },
            verbose);
        print_csv_row(measurement_id, "async", "baseline", batch, outer_iters, batch, runs, baseline_stats);
    }
}


#define DISPATCH_BATCH(batch, kernel, grid, block, smem, ...)                    \
    do {                                                                          \
        switch (batch) {                                                          \
            case 1:  kernel<1><<<grid, block, smem>>>(__VA_ARGS__); break;        \
            case 2:  kernel<2><<<grid, block, smem>>>(__VA_ARGS__); break;        \
            case 4:  kernel<4><<<grid, block, smem>>>(__VA_ARGS__); break;        \
            case 8:  kernel<8><<<grid, block, smem>>>(__VA_ARGS__); break;        \
            case 16: kernel<16><<<grid, block, smem>>>(__VA_ARGS__); break;       \
            default:                                                               \
                fprintf(stderr, "unsupported batch=%d\n", batch);                  \
                std::abort();                                                      \
        }                                                                         \
    } while (0)


static std::string get_arg(int argc, char** argv, const char* key, const char* def) {
    std::string prefix = std::string("--") + key + "=";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], prefix.c_str(), prefix.size()) == 0) {
            return argv[i] + prefix.size();
        }
    }
    return def;
}


static bool has_flag(int argc, char** argv, const char* flag) {
    std::string needle = std::string("--") + flag;
    for (int i = 1; i < argc; ++i) {
        if (needle == argv[i]) {
            return true;
        }
    }
    return false;
}


int main(int argc, char** argv) {
    std::string suite = normalize_token(get_arg(argc, argv, "suite", "all"));
    std::string op = normalize_token(get_arg(argc, argv, "op", "all"));
    bool smoke = has_flag(argc, argv, "smoke");
    bool verbose = has_flag(argc, argv, "verbose");

    const int sync_iters = smoke ? 50000 : 200000;
    const int async_outer_iters = smoke ? 2048 : 8192;
    const int runs = smoke ? 5 : 9;
    const int shared_copy_bytes = 512;
    bool ran_any = false;

    gpu_clock_warmup();
    print_csv_header();

    auto run_if_sync = [&](std::initializer_list<const char*> aliases, auto&& fn) {
        if ((suite == "ALL" || suite == "SYNC") && matches_any(op, aliases)) {
            fn();
            ran_any = true;
        }
    };
    auto run_if_async = [&](std::initializer_list<const char*> aliases, auto&& fn) {
        if ((suite == "ALL" || suite == "ASYNC") && matches_any(op, aliases)) {
            fn();
            ran_any = true;
        }
    };

    run_if_sync({"FFMA", "sync_ffma"}, [&] {
        run_sync_measurement(
            "sync_ffma", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_ffma_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"FADD", "sync_fadd"}, [&] {
        run_sync_measurement(
            "sync_fadd", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_fadd_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"FMUL", "sync_fmul"}, [&] {
        run_sync_measurement(
            "sync_fmul", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_fmul_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"HFMA2", "sync_hfma2"}, [&] {
        run_sync_measurement(
            "sync_hfma2", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_hfma2_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"MUFU", "sync_mufu"}, [&] {
        run_sync_measurement(
            "sync_mufu", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_mufu_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"IMAD", "sync_imad"}, [&] {
        run_sync_measurement(
            "sync_imad", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_imad_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"IADD3", "sync_iadd3"}, [&] {
        run_sync_measurement(
            "sync_iadd3", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_iadd3_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"LOP3", "sync_lop3"}, [&] {
        run_sync_measurement(
            "sync_lop3", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_lop3_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"MOV", "sync_mov"}, [&] {
        run_sync_measurement(
            "sync_mov", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_mov_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"SHFL", "sync_shfl"}, [&] {
        run_sync_measurement(
            "sync_shfl", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_shfl_kernel<<<1, 32>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_warp_baseline_kernel<<<1, 32>>>(d, iters); },
            verbose);
    });

    run_if_sync({"S2R", "sync_s2r"}, [&] {
        run_sync_measurement(
            "sync_s2r", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_s2r_kernel<<<1, 32>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_s2r_baseline_kernel<<<1, 32>>>(d, iters); },
            verbose);
    });

    run_if_sync({"LDS", "sync_lds"}, [&] {
        run_sync_measurement(
            "sync_lds", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_lds_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_scalar_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_sync({"STS", "sync_sts"}, [&] {
        run_sync_measurement(
            "sync_sts", sync_iters, runs,
            [&](BenchResult* d, int iters) { sync_sts_proxy_kernel<<<1, 1>>>(d, iters); },
            [&](BenchResult* d, int iters) { sync_sts_baseline_kernel<<<1, 1>>>(d, iters); },
            verbose);
    });

    run_if_async({"UTMALDG", "async_tma_load"}, [&] {
        TensorMapConfig cfg = config_kpe_bf16();
        uint64_t num_rows = 32;
        void* d_tensor = allocate_tensor(cfg, num_rows);
        CUtensorMap tmap = create_tensor_map_from_config(cfg, d_tensor, num_rows);
        int* d_indices = nullptr;
        CUDA_CHECK(cudaMalloc(&d_indices, 16 * 4 * sizeof(int)));
        std::array<int, 16 * 4> h_indices{};
        for (int b = 0; b < 16; ++b) {
            h_indices[b * 4 + 0] = (b + 0) % num_rows;
            h_indices[b * 4 + 1] = (b + 1) % num_rows;
            h_indices[b * 4 + 2] = (b + 2) % num_rows;
            h_indices[b * 4 + 3] = (b + 3) % num_rows;
        }
        CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), sizeof(h_indices), cudaMemcpyHostToDevice));
        const int bytes_per_gather4 = static_cast<int>(cfg.bytes_per_gather4());
        const int aligned_bytes = (bytes_per_gather4 + 7) & ~7;
        const int64_t cache_hint = 0x14F0000000000000LL;

        run_async_measurement(
            "async_tma_load",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                int smem_bytes = aligned_bytes * batch + 16;
                DISPATCH_BATCH(batch, async_tma_load_kernel, 1, 1, smem_bytes, d, outer, tmap, d_indices, bytes_per_gather4, cache_hint);
            },
            [&](int batch, BenchResult* d, int outer) {
                (void)batch;
                DISPATCH_BATCH(batch, async_tma_load_baseline_kernel, 1, 1, 0, d, outer);
            },
            verbose);

        CUDA_CHECK(cudaFree(d_indices));
        CUDA_CHECK(cudaFree(d_tensor));
    });

    run_if_async({"UTMAPF", "UTMACCTL", "async_tma_prefetch"}, [&] {
        TensorMapConfig cfg = config_kpe_bf16();
        uint64_t num_rows = 32;
        void* d_tensor = allocate_tensor(cfg, num_rows);
        CUtensorMap tmap = create_tensor_map_from_config(cfg, d_tensor, num_rows);
        int* d_indices = nullptr;
        CUDA_CHECK(cudaMalloc(&d_indices, 16 * 4 * sizeof(int)));
        std::array<int, 16 * 4> h_indices{};
        for (int b = 0; b < 16; ++b) {
            h_indices[b * 4 + 0] = (b + 0) % num_rows;
            h_indices[b * 4 + 1] = (b + 1) % num_rows;
            h_indices[b * 4 + 2] = (b + 2) % num_rows;
            h_indices[b * 4 + 3] = (b + 3) % num_rows;
        }
        CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), sizeof(h_indices), cudaMemcpyHostToDevice));
        const int64_t cache_hint = 0x14F0000000000000LL;

        run_async_measurement(
            "async_tma_prefetch",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_tma_prefetch_kernel, 1, 1, 0, d, outer, tmap, d_indices, cache_hint);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_dummy_kernel, 1, 1, 0, d, outer);
            },
            verbose);

        CUDA_CHECK(cudaFree(d_indices));
        CUDA_CHECK(cudaFree(d_tensor));
    });

    run_if_async({"UTMASTG", "async_tma_store"}, [&] {
        TensorMapConfig cfg = config_kpe_bf16();
        uint64_t num_rows = 32;
        void* d_tensor = allocate_tensor(cfg, num_rows);
        CUtensorMap tmap = create_tensor_map_from_config(cfg, d_tensor, num_rows);
        int row_stride_elems = static_cast<int>(cfg.dim0);
        int smem_bytes = static_cast<int>(cfg.bytes_per_row());

        run_async_measurement(
            "async_tma_store",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_tma_store_kernel, 1, 1, smem_bytes, d, outer, tmap, row_stride_elems);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_tma_store_baseline_kernel, 1, 1, 0, d, outer);
            },
            verbose);

        CUDA_CHECK(cudaFree(d_tensor));
    });

    run_if_async({"UTMAREDG", "async_tma_reduce"}, [&] {
        TensorMapConfig cfg = {"reduce_f32", CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 64, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B};
        uint64_t num_rows = 32;
        void* d_tensor = allocate_tensor(cfg, num_rows);
        CUtensorMap tmap = create_tensor_map_from_config(cfg, d_tensor, num_rows);
        int row_stride_elems = static_cast<int>(cfg.dim0);
        int smem_bytes = static_cast<int>(cfg.bytes_per_row());

        run_async_measurement(
            "async_tma_reduce",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_tma_reduce_kernel, 1, 1, smem_bytes, d, outer, tmap, row_stride_elems);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_tma_store_baseline_kernel, 1, 1, 0, d, outer);
            },
            verbose);

        CUDA_CHECK(cudaFree(d_tensor));
    });

    run_if_async({"UBLKCP", "async_bulk_copy"}, [&] {
        char* d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dst, 16 * shared_copy_bytes));

        run_async_measurement(
            "async_bulk_copy",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_bulk_copy_kernel, 1, 1, shared_copy_bytes, d, outer, d_dst, shared_copy_bytes);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_tma_store_baseline_kernel, 1, 1, 0, d, outer);
            },
            verbose);

        CUDA_CHECK(cudaFree(d_dst));
    });

    run_if_async({"UBLKRED", "async_bulk_reduce"}, [&] {
        float* d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dst, 16 * shared_copy_bytes));

        run_async_measurement(
            "async_bulk_reduce",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_bulk_reduce_kernel, 1, 1, shared_copy_bytes, d, outer, d_dst, shared_copy_bytes);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_tma_store_baseline_kernel, 1, 1, 0, d, outer);
            },
            verbose);

        CUDA_CHECK(cudaFree(d_dst));
    });

    run_if_async({"UBLKPF", "async_bulk_prefetch"}, [&] {
        char* d_src = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, 16 * shared_copy_bytes));

        run_async_measurement(
            "async_bulk_prefetch",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_bulk_prefetch_kernel, 1, 1, 0, d, outer, d_src, shared_copy_bytes);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_dummy_kernel, 1, 1, 0, d, outer);
            },
            verbose);

        CUDA_CHECK(cudaFree(d_src));
    });

    run_if_async({"LDG", "LDGMC", "async_ldg"}, [&] {
        uint64_t* d_src = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, 16 * 4 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_src, 0x42, 16 * 4 * sizeof(uint64_t)));
        run_async_measurement(
            "async_ldg",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_ldg_kernel, 1, 1, 0, d, outer, d_src);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_dummy_kernel, 1, 1, 0, d, outer);
            },
            verbose);
        CUDA_CHECK(cudaFree(d_src));
    });

    run_if_async({"STG", "async_stg"}, [&] {
        uint64_t* d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dst, 16 * 4 * sizeof(uint64_t)));
        run_async_measurement(
            "async_stg",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_stg_kernel, 1, 1, 0, d, outer, d_dst);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_dummy_kernel, 1, 1, 0, d, outer);
            },
            verbose);
        CUDA_CHECK(cudaFree(d_dst));
    });

    run_if_async({"LDL", "async_ldl"}, [&] {
        run_async_measurement(
            "async_ldl",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_ldl_kernel, 1, 1, 0, d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_dummy_kernel, 1, 1, 0, d, outer); },
            verbose);
    });

    run_if_async({"STL", "async_stl"}, [&] {
        run_async_measurement(
            "async_stl",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_stl_kernel, 1, 1, 0, d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_dummy_kernel, 1, 1, 0, d, outer); },
            verbose);
    });

    run_if_async({"LDGSTS", "async_ldgsts"}, [&] {
        char* d_src = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, 16 * 16));
        CUDA_CHECK(cudaMemset(d_src, 0x42, 16 * 16));
        run_async_measurement(
            "async_ldgsts",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_ldgsts_kernel, 1, 1, batch * 16, d, outer, d_src);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_ldgsts_baseline_kernel, 1, 1, 0, d, outer);
            },
            verbose);
        CUDA_CHECK(cudaFree(d_src));
    });

    run_if_async({"CCTL", "CCTLL", "async_cctl"}, [&] {
        char* d_src = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, 16 * 64));
        CUDA_CHECK(cudaMemset(d_src, 0x42, 16 * 64));
        run_async_measurement(
            "async_cctl",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_cctl_prefetch_kernel, 1, 1, 0, d, outer, d_src);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_dummy_kernel, 1, 1, 0, d, outer);
            },
            verbose);
        CUDA_CHECK(cudaFree(d_src));
    });

    run_if_async({"MBAR", "async_mbar"}, [&] {
        run_async_measurement(
            "async_mbar",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_mbar_kernel, 1, 1, 0, d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_mbar_baseline_kernel, 1, 1, 0, d, outer); },
            verbose);
    });

    run_if_async({"BAR", "async_bar"}, [&] {
        run_async_measurement(
            "async_bar",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_bar_kernel, 1, 128, 0, d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_bar_baseline_kernel, 1, 128, 0, d, outer); },
            verbose);
    });

    run_if_async({"UTCCP", "async_utccp"}, [&] {
        run_async_measurement(
            "async_utccp",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) {
                int smem_bytes = batch * 128 * 16 * static_cast<int>(sizeof(bf16));
                if (smem_bytes > 48 * 1024) {
                    switch (batch) {
                        case 1: CUDA_CHECK(cudaFuncSetAttribute(async_utccp_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); break;
                        case 2: CUDA_CHECK(cudaFuncSetAttribute(async_utccp_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); break;
                        case 4: CUDA_CHECK(cudaFuncSetAttribute(async_utccp_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); break;
                        case 8: CUDA_CHECK(cudaFuncSetAttribute(async_utccp_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); break;
                        case 16: CUDA_CHECK(cudaFuncSetAttribute(async_utccp_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); break;
                        default: break;
                    }
                }
                DISPATCH_BATCH(batch, async_utccp_kernel, 1, 128, smem_bytes, d, outer);
            },
            [&](int batch, BenchResult* d, int outer) {
                DISPATCH_BATCH(batch, async_utccp_baseline_kernel, 1, 128, 0, d, outer);
            },
            verbose);
    });

    run_if_async({"LDT", "async_ldt"}, [&] {
        run_async_measurement(
            "async_ldt",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_ldt_kernel, 1, 32, 0, d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_ldt_baseline_kernel, 1, 32, 0, d, outer); },
            verbose);
    });

    run_if_async({"LDTM", "UTCSHIFT", "async_ldtm"}, [&] {
        run_async_measurement(
            "async_ldtm",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_ldtm_kernel, 1, 32, 0, d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_ldt_baseline_kernel, 1, 32, 0, d, outer); },
            verbose);
    });

    run_if_async({"STT", "async_stt"}, [&] {
        run_async_measurement(
            "async_stt",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_stt_kernel, 1, 32, 0, d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_stt_baseline_kernel, 1, 32, 0, d, outer); },
            verbose);
    });

    run_if_async({"STTM", "async_sttm"}, [&] {
        run_async_measurement(
            "async_sttm",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_sttm_kernel, 1, 32, 0, d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_stt_baseline_kernel, 1, 32, 0, d, outer); },
            verbose);
    });

    run_if_async({"UTCHMMA", "UTCIMMA", "UTCOMMA", "UTCQMMA", "async_utchmma"}, [&] {
        run_async_measurement(
            "async_utchmma",
            async_outer_iters,
            runs,
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_utchmma_kernel, 1, 128, 64 * 16 * static_cast<int>(sizeof(bf16)), d, outer); },
            [&](int batch, BenchResult* d, int outer) { DISPATCH_BATCH(batch, async_utchmma_baseline_kernel, 1, 128, 0, d, outer); },
            verbose);
    });

    if (!ran_any) {
        fprintf(stderr, "No matching measurement for suite=%s op=%s\n", suite.c_str(), op.c_str());
        return 1;
    }

    return 0;
}
