#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>
#include <sys/time.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(ans) cuda_check_impl((ans), __FILE__, __LINE__)

inline void cuda_check_impl(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s  %s:%d\n",
                cudaGetErrorString(code), file, line);
        exit(1);
    }
}

#define CU_CHECK(ans) cu_check_impl((ans), __FILE__, __LINE__)

inline void cu_check_impl(CUresult code, const char* file, int line) {
    if (code != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(code, &err_str);
        fprintf(stderr, "CUDA driver error: %s  %s:%d\n",
                err_str ? err_str : "unknown", file, line);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Device-side timing (nanosecond resolution via %globaltimer)
// Adapted from microbenchmarks/kernel_timer.h
// ---------------------------------------------------------------------------
__device__ __forceinline__
int64_t globaltimer() {
    int64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__
uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// Per-CTA profiler: records (sm_id, tag, start_ns, duration_ns) tuples
// into a device buffer. Host reads back after kernel completes.
//
// Layout per CTA: [count, (sm_id, tag, start_ns, duration_ns), ...]
//   count:      int64_t
//   each entry: 4 x int64_t
struct Profiler {
    int64_t* data_ptr_;
    int      sm_id_;
    int      cnt_;

    __device__
    void init(int max_entries, int64_t* data_ptr, int bid) {
        data_ptr_ = data_ptr + bid * (1 + max_entries * 4);
        sm_id_ = get_smid();
        cnt_ = 0;
    }

    __device__
    void start(int tag) {
        data_ptr_[1 + cnt_ * 4 + 0] = sm_id_;
        data_ptr_[1 + cnt_ * 4 + 1] = tag;
        data_ptr_[1 + cnt_ * 4 + 2] = globaltimer();
    }

    __device__
    void stop() {
        data_ptr_[1 + cnt_ * 4 + 3] = globaltimer() - data_ptr_[1 + cnt_ * 4 + 2];
        cnt_ += 1;
    }

    __device__
    void flush() {
        data_ptr_[0] = cnt_;
    }
};

// Lightweight timer: just start/end per kernel, written to a small buffer.
// buf layout: [start_ns, end_ns, sm_id] per block
struct KernelTimer {
    int64_t* buf_;

    __device__
    void init(int64_t* buf, int bid) {
        buf_ = buf + bid * 3;
    }

    __device__
    void start() {
        buf_[0] = globaltimer();
        buf_[2] = get_smid();
    }

    __device__
    void stop() {
        buf_[1] = globaltimer();
    }
};

// ---------------------------------------------------------------------------
// Host-side wall-clock (for coarse validation only)
// ---------------------------------------------------------------------------
inline double dtime() {
    struct timeval t;
    gettimeofday(&t, nullptr);
    return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

// ---------------------------------------------------------------------------
// MeasurementSeries — trimmed-mean statistics
// From gpu-benches/MeasurementSeries.hpp
// ---------------------------------------------------------------------------
class MeasurementSeries {
public:
    void add(double v) { data_.push_back(v); }

    // Trimmed mean: drop min and max, average the rest
    double value() const {
        if (data_.empty()) return 0.0;
        if (data_.size() == 1) return data_[0];
        if (data_.size() == 2) return (data_[0] + data_[1]) / 2.0;
        auto sorted = data_;
        std::sort(sorted.begin(), sorted.end());
        return std::accumulate(sorted.begin() + 1, sorted.end() - 1, 0.0)
               / (double)(sorted.size() - 2);
    }

    double median() const {
        if (data_.empty()) return 0.0;
        auto sorted = data_;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        if (n % 2 == 0) return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        return sorted[n/2];
    }

    double minValue() const {
        if (data_.empty()) return 0.0;
        return *std::min_element(data_.begin(), data_.end());
    }

    double maxValue() const {
        if (data_.empty()) return 0.0;
        return *std::max_element(data_.begin(), data_.end());
    }

    // spread = (max - min) / trimmed_mean
    double spread() const {
        if (data_.size() <= 1) return 0.0;
        double v = value();
        if (v == 0.0) return 0.0;
        return (maxValue() - minValue()) / v;
    }

    size_t count() const { return data_.size(); }

private:
    std::vector<double> data_;
};

// ---------------------------------------------------------------------------
// GPU clock ramp-up kernel
// From gpu-benches/gpu-clock.cuh  —  simplified, no NVML dependency
// ---------------------------------------------------------------------------
__global__ void power_kernel(double* A, int iters) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    double val = A[0];
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        val -= (tidx * 0.1) * val;
    }
    A[0] = val;
}

inline void gpu_clock_warmup(int device = 0) {
    CUDA_CHECK(cudaSetDevice(device));
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(double)));
    double h = 1.0;
    CUDA_CHECK(cudaMemcpy(dA, &h, sizeof(double), cudaMemcpyHostToDevice));

    // Run with increasing iters until we spend >200ms to ramp clocks
    for (int iters = 100; iters < 1000000; iters *= 2) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = dtime();
        power_kernel<<<512, 256>>>(dA, iters);
        CUDA_CHECK(cudaDeviceSynchronize());
        double dt = dtime() - t0;
        if (dt > 0.2) break;
    }
    CUDA_CHECK(cudaFree(dA));
}

// ---------------------------------------------------------------------------
// Helper: read back device-side KernelTimer results and compute
// per-CTA elapsed nanoseconds. Returns (min_start, max_end) across CTAs.
// ---------------------------------------------------------------------------
struct TimingResult {
    double elapsed_ns;       // max_end - min_start across all CTAs
    double elapsed_ms;       // elapsed_ns / 1e6
    double per_cta_avg_ns;   // average per-CTA elapsed
};

inline TimingResult read_kernel_timer(int64_t* d_timer_buf, int num_blocks) {
    // Layout per block: [start_ns, end_ns, sm_id]
    std::vector<int64_t> h_buf(num_blocks * 3);
    CUDA_CHECK(cudaMemcpy(h_buf.data(), d_timer_buf,
                          num_blocks * 3 * sizeof(int64_t),
                          cudaMemcpyDeviceToHost));

    int64_t min_start = INT64_MAX;
    int64_t max_end   = INT64_MIN;
    double  sum_elapsed = 0.0;

    for (int i = 0; i < num_blocks; i++) {
        int64_t s = h_buf[i * 3 + 0];
        int64_t e = h_buf[i * 3 + 1];
        if (s < min_start) min_start = s;
        if (e > max_end)   max_end   = e;
        sum_elapsed += (double)(e - s);
    }

    TimingResult r;
    r.elapsed_ns     = (double)(max_end - min_start);
    r.elapsed_ms     = r.elapsed_ns / 1e6;
    r.per_cta_avg_ns = sum_elapsed / num_blocks;
    return r;
}
