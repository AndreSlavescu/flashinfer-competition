// Standalone B200 Device Query
// Queries CUDA Runtime + Driver API to get ground-truth hardware properties.
// No helper_cuda.h dependency — only <cuda_runtime.h> and <cuda.h>.
//
// Cross-references claims in CLAUDE.md (which cites GB203 papers, not B200).
// Key targets:
//   SM count              → CLAUDE.md says 148 (arxiv:2512.02189)
//   regsPerMultiprocessor → CLAUDE.md says 65,536 (arxiv:2507.10789, GB203!)
//   maxThreadsPerSM       → CLAUDE.md says 2,048 (arxiv:2512.02189)
//   L2 cache size         → CLAUDE.md says ~64 MB (empirical)
//   Total memory          → CLAUDE.md says 192 GB
//   sharedMemPerSM        → CLAUDE.md says up to 228 KB
//   SM clock rate         → not pinned in CLAUDE.md (~2.1 GHz expected)

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>

// Helper to check CUDA runtime errors
static void check_cuda(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
        exit(1);
    }
}

// Helper to check CUDA driver errors
static void check_cu(CUresult err, const char* ctx) {
    if (err != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(err, &s);
        fprintf(stderr, "CU error at %s: %s\n", ctx, s ? s : "unknown");
        exit(1);
    }
}

int main() {
    // Initialize Driver API
    check_cu(cuInit(0), "cuInit");

    int deviceCount = 0;
    check_cuda(cudaGetDeviceCount(&deviceCount), "GetDeviceCount");

    printf("=== B200 Device Query — Hardware Spec Verification ===\n");
    printf("Device count: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        check_cuda(cudaSetDevice(dev), "SetDevice");

        cudaDeviceProp p;
        check_cuda(cudaGetDeviceProperties(&p, dev), "GetDeviceProperties");

        int driverVer = 0, runtimeVer = 0;
        cudaDriverGetVersion(&driverVer);
        cudaRuntimeGetVersion(&runtimeVer);

        // --------------- Identity ---------------
        printf("Device %d: %s\n", dev, p.name);
        printf("  Compute Capability:      %d.%d\n", p.major, p.minor);
        printf("  CUDA Driver Version:     %d.%d\n", driverVer/1000, (driverVer%100)/10);
        printf("  CUDA Runtime Version:    %d.%d\n\n", runtimeVer/1000, (runtimeVer%100)/10);

        // --------------- SM / Compute ---------------
        // SM clock via attribute (clockRate removed from cudaDeviceProp in CUDA 13)
        int smClockKHz = 0;
        cudaDeviceGetAttribute(&smClockKHz, cudaDevAttrClockRate, dev);

        printf("--- SM / Compute ---\n");
        printf("  multiProcessorCount:     %d        (CLAUDE.md claim: 148)\n",
               p.multiProcessorCount);
        printf("  maxThreadsPerMultiProc:  %d     (CLAUDE.md claim: 2048)\n",
               p.maxThreadsPerMultiProcessor);
        printf("  warpSize:                %d\n", p.warpSize);
        printf("  maxWarpsPerSM:           %d\n",
               p.maxThreadsPerMultiProcessor / p.warpSize);
        printf("  SM Clock Rate:           %.0f MHz  (%.3f GHz)\n\n",
               smClockKHz * 1e-3,
               smClockKHz * 1e-6);

        // --------------- Register File ---------------
        printf("--- Register File ---\n");
        printf("  regsPerBlock:            %d\n", p.regsPerBlock);
        printf("  regsPerMultiprocessor:   %-8d (CLAUDE.md claim: 65536, from GB203 not B200!)\n",
               p.regsPerMultiprocessor);
        printf("  Register file / SM:      %.1f KB  (%d regs × 4 bytes)\n\n",
               p.regsPerMultiprocessor * 4.0 / 1024.0,
               p.regsPerMultiprocessor);

        // --------------- Shared Memory ---------------
        printf("--- Shared Memory ---\n");
        printf("  sharedMemPerBlock:       %zu bytes  (%.1f KB)\n",
               p.sharedMemPerBlock, p.sharedMemPerBlock / 1024.0);
        printf("  sharedMemPerMultiproc:   %zu bytes  (%.1f KB)  "
               "(CLAUDE.md claim: 228 KB)\n\n",
               p.sharedMemPerMultiprocessor,
               p.sharedMemPerMultiprocessor / 1024.0);

        // --------------- Memory ---------------
        printf("--- Memory ---\n");
        printf("  totalGlobalMem:          %.3f GB  (CLAUDE.md claim: 192 GB)\n",
               p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  l2CacheSize:             %d bytes  (%.1f MB)  "
               "(CLAUDE.md claim: ~64 MB)\n",
               p.l2CacheSize,
               p.l2CacheSize / (1024.0 * 1024.0));

        // Memory clock via attribute (stable across CUDA versions)
        int memClockKHz = 0;
        cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);
        printf("  Memory Clock Rate:       %.0f MHz\n", memClockKHz * 1e-3);
        printf("  Memory Bus Width:        %d bits\n", p.memoryBusWidth);
        // HBM is effectively DDR: peak BW = 2 * clock * bus_bytes
        double peakBW = 2.0 * memClockKHz * 1e3 * (p.memoryBusWidth / 8.0) / 1e12;
        printf("  Peak Mem BW (formula):   %.2f TB/s  (CLAUDE.md claim: 8 TB/s)\n\n",
               peakBW);

        // --------------- Other ---------------
        printf("--- Other ---\n");
        printf("  totalConstMem:           %zu bytes\n", p.totalConstMem);
        printf("  maxThreadsPerBlock:      %d\n", p.maxThreadsPerBlock);
        printf("  asyncEngineCount:        %d\n", p.asyncEngineCount);
        printf("  unifiedAddressing:       %s\n", p.unifiedAddressing ? "Yes" : "No");
        printf("  managedMemory:           %s\n", p.managedMemory ? "Yes" : "No");
        printf("  cooperativeLaunch:       %s\n", p.cooperativeLaunch ? "Yes" : "No");
        printf("  computePreemption:       %s\n\n",
               p.computePreemptionSupported ? "Yes" : "No");

        // --------------- Driver API cross-check ---------------
        CUdevice cuDev;
        check_cu(cuDeviceGet(&cuDev, dev), "cuDeviceGet");

        int cu_sm_count = 0, cu_regs = 0, cu_max_threads = 0;
        cuDeviceGetAttribute(&cu_sm_count,
                             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDev);
        cuDeviceGetAttribute(&cu_regs,
                             CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, cuDev);
        cuDeviceGetAttribute(&cu_max_threads,
                             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDev);

        printf("--- Driver API Cross-check ---\n");
        printf("  SM count  (CU_DEVICE_ATTR): %d\n", cu_sm_count);
        printf("  Regs/SM   (CU_DEVICE_ATTR): %d  (%.1f KB)\n",
               cu_regs, cu_regs * 4.0 / 1024.0);
        printf("  Threads/SM (CU_DEVICE_ATTR): %d\n\n", cu_max_threads);

        // Consistency check
        if (p.multiProcessorCount != cu_sm_count ||
            p.regsPerMultiprocessor != cu_regs ||
            p.maxThreadsPerMultiProcessor != cu_max_threads) {
            printf("  WARNING: Runtime and Driver API disagree on some properties!\n\n");
        } else {
            printf("  Runtime and Driver API agree on all cross-checked properties.\n\n");
        }

        // --------------- CSV Summary ---------------
        printf("--- CSV Summary ---\n");
        printf("device_name,sm_count,regs_per_sm,regfile_kb,max_threads_per_sm,"
               "max_warps_per_sm,l2_mb,total_mem_gb,smem_per_sm_kb,"
               "sm_clock_mhz,mem_clock_mhz,mem_bus_bits,peak_bw_tbps,"
               "compute_cap_major,compute_cap_minor\n");
        printf("%s,%d,%d,%.1f,%d,%d,%.1f,%.3f,%.1f,%.0f,%.0f,%d,%.2f,%d,%d\n",
               p.name,
               p.multiProcessorCount,
               p.regsPerMultiprocessor,
               p.regsPerMultiprocessor * 4.0 / 1024.0,
               p.maxThreadsPerMultiProcessor,
               p.maxThreadsPerMultiProcessor / p.warpSize,
               p.l2CacheSize / (1024.0 * 1024.0),
               p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               p.sharedMemPerMultiprocessor / 1024.0,
               smClockKHz * 1e-3,
               memClockKHz * 1e-3,
               p.memoryBusWidth,
               peakBW,
               p.major, p.minor);
    }

    return 0;
}
