#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include "benchmark_common.cuh"
#include "topk_kernels.cuh"

// ---------------------------------------------------------------------------
// CSV header
// ---------------------------------------------------------------------------
static void print_csv_header() {
    printf("experiment,x_var,algorithm,batch_size,seq_len,topk,block_size,iters,"
           "total_cycles,ns_per_call,elements_per_ns,gbps,spread_pct\n");
}

// ---------------------------------------------------------------------------
// Generate random F32 scores
// ---------------------------------------------------------------------------
static void fill_random_scores(float* h_scores, int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        h_scores[i] = dist(rng);
    }
}

// ---------------------------------------------------------------------------
// Experiment 1: CUB radix sort top-K
// ---------------------------------------------------------------------------
static void run_cub_sort_topk(bool verbose) {
    fprintf(stderr, "\n=== Experiment: cub_sort_topk ===\n");

    const int TOPK = 2048;
    const int RUNS = 11;
    const int ITERS = 100;

    int seq_lens[] = {64, 256, 1024, 2048, 4096, 5824};

    for (int seq_len : seq_lens) {
        if (verbose) fprintf(stderr, "  seq_len=%d ...\n", seq_len);

        // Allocate and fill input
        std::vector<float> h_scores(seq_len);
        fill_random_scores(h_scores.data(), seq_len);

        float *d_scores_in, *d_scores_out;
        int32_t *d_indices_in, *d_indices_out;
        CUDA_CHECK(cudaMalloc(&d_scores_in, seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scores_out, seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_indices_in, seq_len * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_indices_out, seq_len * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_scores_in, h_scores.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice));

        // Initialize indices
        std::vector<int32_t> h_indices(seq_len);
        for (int i = 0; i < seq_len; i++) h_indices[i] = i;
        CUDA_CHECK(cudaMemcpy(d_indices_in, h_indices.data(), seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));

        // Get temp storage
        size_t temp_bytes = CubSortTopK::get_temp_storage_bytes(seq_len);
        void* d_temp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

        MeasurementSeries ns_series;

        for (int run = 0; run < RUNS; run++) {
            // Reset input (CUB sort modifies in-place)
            CUDA_CHECK(cudaMemcpy(d_scores_in, h_scores.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_indices_in, h_indices.data(), seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            for (int i = 0; i < ITERS; i++) {
                CubSortTopK::run(d_scores_in, d_scores_out, d_indices_in, d_indices_out,
                                 seq_len, d_temp, temp_bytes);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            ns_series.add(elapsed_ms * 1e6);  // ms -> ns

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            if (verbose) {
                fprintf(stderr, "    run %d/%d: %.1f ns total (%.1f ns/call)\n",
                        run + 1, RUNS, elapsed_ms * 1e6, elapsed_ms * 1e6 / ITERS);
                fflush(stderr);
            }
        }

        double avg_ns = ns_series.value();
        double ns_per_call = avg_ns / ITERS;
        double elements_per_ns = (double)seq_len / ns_per_call;
        double gbps = (double)seq_len * sizeof(float) / ns_per_call;  // GB/s
        double spread = ns_series.spread() * 100.0;
        int actual_topk = (seq_len < TOPK) ? seq_len : TOPK;

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", seq_len);

        printf("cub_sort_topk,%s,cub_radix,1,%d,%d,0,%d,"
               "0,%.1f,%.3f,%.3f,%.1f\n",
               x_val, seq_len, actual_topk, ITERS,
               ns_per_call, elements_per_ns, gbps, spread);
        fflush(stdout);

        CUDA_CHECK(cudaFree(d_scores_in));
        CUDA_CHECK(cudaFree(d_scores_out));
        CUDA_CHECK(cudaFree(d_indices_in));
        CUDA_CHECK(cudaFree(d_indices_out));
        CUDA_CHECK(cudaFree(d_temp));
    }
}

// ---------------------------------------------------------------------------
// Experiment 2: Block bitonic top-K
// ---------------------------------------------------------------------------
static void run_block_bitonic_topk(bool verbose) {
    fprintf(stderr, "\n=== Experiment: block_bitonic_topk ===\n");

    const int TOPK = 2048;
    const int BLOCK_SIZE = 256;
    const int RUNS = 11;
    const int ITERS = 100;

    int seq_lens[] = {2048, 4096, 5824};

    for (int seq_len : seq_lens) {
        if (verbose) fprintf(stderr, "  seq_len=%d ...\n", seq_len);

        std::vector<float> h_scores(seq_len);
        fill_random_scores(h_scores.data(), seq_len);

        float *d_scores, *d_out_scores;
        int32_t *d_out_indices;
        TopkResult *d_result;
        CUDA_CHECK(cudaMalloc(&d_scores, seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_scores, TOPK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_indices, TOPK * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(TopkResult)));
        CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice));

        MeasurementSeries cycles_series;
        MeasurementSeries ns_series;

        for (int run = 0; run < RUNS; run++) {
            CUDA_CHECK(cudaMemset(d_result, 0, sizeof(TopkResult)));

            // smem: TOPK floats + TOPK int32s
            int smem_bytes = TOPK * (sizeof(float) + sizeof(int32_t));
            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(kernel_block_bitonic_topk<BLOCK_SIZE, TOPK>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            }

            kernel_block_bitonic_topk<BLOCK_SIZE, TOPK><<<1, BLOCK_SIZE>>>(
                d_scores, d_out_indices, d_out_scores, seq_len, d_result, ITERS);
            CUDA_CHECK(cudaDeviceSynchronize());

            TopkResult h_result;
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(TopkResult), cudaMemcpyDeviceToHost));

            cycles_series.add((double)h_result.total_cycles);
            ns_series.add((double)(h_result.gt_end_ns - h_result.gt_start_ns));

            if (verbose) {
                fprintf(stderr, "    run %d/%d: %lu cycles, %.1f ns\n",
                        run + 1, RUNS, h_result.total_cycles,
                        (double)(h_result.gt_end_ns - h_result.gt_start_ns));
                fflush(stderr);
            }
        }

        double avg_ns = ns_series.value();
        double ns_per_call = avg_ns / ITERS;
        double elements_per_ns = (double)seq_len / ns_per_call;
        double gbps = (double)seq_len * sizeof(float) / ns_per_call;
        double spread = cycles_series.spread() * 100.0;

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", seq_len);

        printf("block_bitonic_topk,%s,bitonic_merge,1,%d,%d,%d,%d,"
               "%.0f,%.1f,%.3f,%.3f,%.1f\n",
               x_val, seq_len, TOPK, BLOCK_SIZE, ITERS,
               cycles_series.value(), ns_per_call, elements_per_ns, gbps, spread);
        fflush(stdout);

        CUDA_CHECK(cudaFree(d_scores));
        CUDA_CHECK(cudaFree(d_out_scores));
        CUDA_CHECK(cudaFree(d_out_indices));
        CUDA_CHECK(cudaFree(d_result));
    }
}

// ---------------------------------------------------------------------------
// Experiment 3: Radix select top-K
// ---------------------------------------------------------------------------
static void run_radix_select_topk(bool verbose) {
    fprintf(stderr, "\n=== Experiment: radix_select_topk ===\n");

    const int TOPK = 2048;
    const int BLOCK_SIZE = 256;
    const int RUNS = 11;
    const int ITERS = 100;

    int seq_lens[] = {2048, 4096, 5824};

    for (int seq_len : seq_lens) {
        if (verbose) fprintf(stderr, "  seq_len=%d ...\n", seq_len);

        std::vector<float> h_scores(seq_len);
        fill_random_scores(h_scores.data(), seq_len);

        float *d_scores;
        int32_t *d_out_indices;
        TopkResult *d_result;
        CUDA_CHECK(cudaMalloc(&d_scores, seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_indices, TOPK * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(TopkResult)));
        CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice));

        MeasurementSeries cycles_series;
        MeasurementSeries ns_series;

        for (int run = 0; run < RUNS; run++) {
            CUDA_CHECK(cudaMemset(d_result, 0, sizeof(TopkResult)));

            kernel_radix_select<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
                d_scores, d_out_indices, seq_len, TOPK, d_result, ITERS);
            CUDA_CHECK(cudaDeviceSynchronize());

            TopkResult h_result;
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(TopkResult), cudaMemcpyDeviceToHost));

            cycles_series.add((double)h_result.total_cycles);
            ns_series.add((double)(h_result.gt_end_ns - h_result.gt_start_ns));

            if (verbose) {
                fprintf(stderr, "    run %d/%d: %lu cycles, %.1f ns\n",
                        run + 1, RUNS, h_result.total_cycles,
                        (double)(h_result.gt_end_ns - h_result.gt_start_ns));
                fflush(stderr);
            }
        }

        double avg_ns = ns_series.value();
        double ns_per_call = avg_ns / ITERS;
        double elements_per_ns = (double)seq_len / ns_per_call;
        double gbps = (double)seq_len * sizeof(float) / ns_per_call;
        double spread = cycles_series.spread() * 100.0;

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", seq_len);

        printf("radix_select_topk,%s,radix_2pass,1,%d,%d,%d,%d,"
               "%.0f,%.1f,%.3f,%.3f,%.1f\n",
               x_val, seq_len, TOPK, BLOCK_SIZE, ITERS,
               cycles_series.value(), ns_per_call, elements_per_ns, gbps, spread);
        fflush(stdout);

        CUDA_CHECK(cudaFree(d_scores));
        CUDA_CHECK(cudaFree(d_out_indices));
        CUDA_CHECK(cudaFree(d_result));
    }
}

// ---------------------------------------------------------------------------
// Experiment 4: Batched CUB sort — multiple batch elements
// ---------------------------------------------------------------------------
static void run_batched_cub(bool verbose) {
    fprintf(stderr, "\n=== Experiment: batched_cub_sort ===\n");

    const int SEQ_LEN = 5824;
    const int TOPK = 2048;
    const int RUNS = 11;
    const int ITERS = 100;

    int batch_sizes[] = {1, 4, 8, 16, 31};

    for (int batch_size : batch_sizes) {
        if (verbose) fprintf(stderr, "  batch_size=%d ...\n", batch_size);

        // Allocate per-element arrays
        std::vector<float> h_scores(SEQ_LEN);
        fill_random_scores(h_scores.data(), SEQ_LEN);

        float *d_scores_in, *d_scores_out;
        int32_t *d_indices_in, *d_indices_out;
        CUDA_CHECK(cudaMalloc(&d_scores_in, SEQ_LEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scores_out, SEQ_LEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_indices_in, SEQ_LEN * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_indices_out, SEQ_LEN * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_scores_in, h_scores.data(), SEQ_LEN * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<int32_t> h_indices(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; i++) h_indices[i] = i;
        CUDA_CHECK(cudaMemcpy(d_indices_in, h_indices.data(), SEQ_LEN * sizeof(int32_t), cudaMemcpyHostToDevice));

        size_t temp_bytes = CubSortTopK::get_temp_storage_bytes(SEQ_LEN);
        void* d_temp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

        MeasurementSeries ns_series;

        for (int run = 0; run < RUNS; run++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            for (int i = 0; i < ITERS; i++) {
                // Sort each batch element sequentially (worst case for batched)
                for (int b = 0; b < batch_size; b++) {
                    // Reset input for each element
                    CUDA_CHECK(cudaMemcpy(d_scores_in, h_scores.data(), SEQ_LEN * sizeof(float), cudaMemcpyHostToDevice));
                    CubSortTopK::run(d_scores_in, d_scores_out, d_indices_in, d_indices_out,
                                     SEQ_LEN, d_temp, temp_bytes);
                }
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            ns_series.add(elapsed_ms * 1e6);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            if (verbose) {
                fprintf(stderr, "    run %d/%d: %.1f ns total\n",
                        run + 1, RUNS, elapsed_ms * 1e6);
                fflush(stderr);
            }
        }

        double avg_ns = ns_series.value();
        double ns_per_call = avg_ns / (ITERS * batch_size);
        double elements_per_ns = (double)SEQ_LEN / ns_per_call;
        double gbps = (double)SEQ_LEN * sizeof(float) / ns_per_call;
        double spread = ns_series.spread() * 100.0;

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", batch_size);

        printf("batched_cub_sort,%s,cub_radix,%d,%d,%d,0,%d,"
               "0,%.1f,%.3f,%.3f,%.1f\n",
               x_val, batch_size, SEQ_LEN, TOPK, ITERS,
               ns_per_call, elements_per_ns, gbps, spread);
        fflush(stdout);

        CUDA_CHECK(cudaFree(d_scores_in));
        CUDA_CHECK(cudaFree(d_scores_out));
        CUDA_CHECK(cudaFree(d_indices_in));
        CUDA_CHECK(cudaFree(d_indices_out));
        CUDA_CHECK(cudaFree(d_temp));
    }
}

// ---------------------------------------------------------------------------
// Experiment 5: Radix select v2 — per-warp histograms (no atomicAdd in hot path)
// ---------------------------------------------------------------------------
static void run_radix_select_v2(bool verbose) {
    fprintf(stderr, "\n=== Experiment: radix_select_v2 ===\n");

    const int TOPK = 2048;
    const int BLOCK_SIZE = 256;
    const int RUNS = 11;
    const int ITERS = 100;

    int seq_lens[] = {2048, 4096, 5824};

    for (int seq_len : seq_lens) {
        if (verbose) fprintf(stderr, "  seq_len=%d ...\n", seq_len);

        std::vector<float> h_scores(seq_len);
        fill_random_scores(h_scores.data(), seq_len);

        float *d_scores;
        int32_t *d_out_indices;
        TopkResult *d_result;
        CUDA_CHECK(cudaMalloc(&d_scores, seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_indices, TOPK * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(TopkResult)));
        CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice));

        MeasurementSeries cycles_series;
        MeasurementSeries ns_series;

        for (int run = 0; run < RUNS; run++) {
            CUDA_CHECK(cudaMemset(d_result, 0, sizeof(TopkResult)));

            kernel_radix_select_v2<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
                d_scores, d_out_indices, seq_len, TOPK, d_result, ITERS);
            CUDA_CHECK(cudaDeviceSynchronize());

            TopkResult h_result;
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(TopkResult), cudaMemcpyDeviceToHost));

            cycles_series.add((double)h_result.total_cycles);
            ns_series.add((double)(h_result.gt_end_ns - h_result.gt_start_ns));

            if (verbose) {
                fprintf(stderr, "    run %d/%d: %lu cycles, %.1f ns\n",
                        run + 1, RUNS, h_result.total_cycles,
                        (double)(h_result.gt_end_ns - h_result.gt_start_ns));
                fflush(stderr);
            }
        }

        double avg_ns = ns_series.value();
        double ns_per_call = avg_ns / ITERS;
        double elements_per_ns = (double)seq_len / ns_per_call;
        double gbps = (double)seq_len * sizeof(float) / ns_per_call;
        double spread = cycles_series.spread() * 100.0;

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", seq_len);

        printf("radix_select_v2,%s,local_hist,1,%d,%d,%d,%d,"
               "%.0f,%.1f,%.3f,%.3f,%.1f\n",
               x_val, seq_len, TOPK, BLOCK_SIZE, ITERS,
               cycles_series.value(), ns_per_call, elements_per_ns, gbps, spread);
        fflush(stdout);

        CUDA_CHECK(cudaFree(d_scores));
        CUDA_CHECK(cudaFree(d_out_indices));
        CUDA_CHECK(cudaFree(d_result));
    }
}

// ---------------------------------------------------------------------------
// Experiment 6: CUB BlockRadixSort — SMEM-resident sort, no global mem round-trips
// ---------------------------------------------------------------------------
static void run_cub_block_sort(bool verbose) {
    fprintf(stderr, "\n=== Experiment: cub_block_sort ===\n");

    const int TOPK = 2048;
    const int BLOCK_SIZE = 256;
    const int RUNS = 11;
    const int ITERS = 100;

    // seq_len→ITEMS_PER_THREAD mapping (BLOCK_SIZE*IPT must be >= seq_len)
    struct Config { int seq_len; int ipt; };
    Config configs[] = {{2048, 8}, {4096, 16}, {5824, 23}};

    for (auto& cfg : configs) {
        int seq_len = cfg.seq_len;
        if (verbose) fprintf(stderr, "  seq_len=%d (IPT=%d) ...\n", seq_len, cfg.ipt);

        std::vector<float> h_scores(seq_len);
        fill_random_scores(h_scores.data(), seq_len);

        float *d_scores;
        int32_t *d_out_indices;
        TopkResult *d_result;
        CUDA_CHECK(cudaMalloc(&d_scores, seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_indices, TOPK * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(TopkResult)));
        CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice));

        MeasurementSeries cycles_series;
        MeasurementSeries ns_series;

        for (int run = 0; run < RUNS; run++) {
            CUDA_CHECK(cudaMemset(d_result, 0, sizeof(TopkResult)));

            // Dispatch by ITEMS_PER_THREAD (compile-time constant required by CUB)
            if (cfg.ipt == 8) {
                cudaFuncSetAttribute(kernel_cub_block_sort<BLOCK_SIZE, 8, TOPK>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 228*1024);
                kernel_cub_block_sort<BLOCK_SIZE, 8, TOPK><<<1, BLOCK_SIZE>>>(
                    d_scores, d_out_indices, seq_len, d_result, ITERS);
            } else if (cfg.ipt == 16) {
                cudaFuncSetAttribute(kernel_cub_block_sort<BLOCK_SIZE, 16, TOPK>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 228*1024);
                kernel_cub_block_sort<BLOCK_SIZE, 16, TOPK><<<1, BLOCK_SIZE>>>(
                    d_scores, d_out_indices, seq_len, d_result, ITERS);
            } else {  // ipt == 23
                cudaFuncSetAttribute(kernel_cub_block_sort<BLOCK_SIZE, 23, TOPK>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 228*1024);
                kernel_cub_block_sort<BLOCK_SIZE, 23, TOPK><<<1, BLOCK_SIZE>>>(
                    d_scores, d_out_indices, seq_len, d_result, ITERS);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            TopkResult h_result;
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(TopkResult), cudaMemcpyDeviceToHost));

            cycles_series.add((double)h_result.total_cycles);
            ns_series.add((double)(h_result.gt_end_ns - h_result.gt_start_ns));

            if (verbose) {
                fprintf(stderr, "    run %d/%d: %lu cycles, %.1f ns\n",
                        run + 1, RUNS, h_result.total_cycles,
                        (double)(h_result.gt_end_ns - h_result.gt_start_ns));
                fflush(stderr);
            }
        }

        double avg_ns = ns_series.value();
        double ns_per_call = avg_ns / ITERS;
        double elements_per_ns = (double)seq_len / ns_per_call;
        double gbps = (double)seq_len * sizeof(float) / ns_per_call;
        double spread = cycles_series.spread() * 100.0;

        char x_val[16];
        snprintf(x_val, sizeof(x_val), "%d", seq_len);

        printf("cub_block_sort,%s,cub_block,1,%d,%d,%d,%d,"
               "%.0f,%.1f,%.3f,%.3f,%.1f\n",
               x_val, seq_len, TOPK, BLOCK_SIZE, ITERS,
               cycles_series.value(), ns_per_call, elements_per_ns, gbps, spread);
        fflush(stdout);

        CUDA_CHECK(cudaFree(d_scores));
        CUDA_CHECK(cudaFree(d_out_indices));
        CUDA_CHECK(cudaFree(d_result));
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

    gpu_clock_warmup();

    print_csv_header();

    if (experiment == "all" || experiment == "cub_sort_topk") {
        run_cub_sort_topk(verbose);
    }
    if (experiment == "all" || experiment == "block_bitonic_topk") {
        run_block_bitonic_topk(verbose);
    }
    if (experiment == "all" || experiment == "radix_select_topk") {
        run_radix_select_topk(verbose);
    }
    if (experiment == "all" || experiment == "batched_cub_sort") {
        run_batched_cub(verbose);
    }
    if (experiment == "all" || experiment == "radix_select_v2") {
        run_radix_select_v2(verbose);
    }
    if (experiment == "all" || experiment == "cub_block_sort") {
        run_cub_block_sort(verbose);
    }

    return 0;
}
