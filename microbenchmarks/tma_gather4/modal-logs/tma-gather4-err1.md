(.venv) mark123@E9IFVFFP0841F:~/projects/FlashMLA/microbenchmarks/tma_gather4$ modal run run_modal.py
✓ Initialized. View run at https://modal.com/apps/mark123ymm/main/ap-UcKTAuFZfosVJwOdebus3g
✓ Created objects.
├── 🔨 Created mount /home/mark123/projects/FlashMLA/microbenchmarks/tma_gather4/run_modal.py
├── 🔨 Created mount common.cuh
├── 🔨 Created mount tensor_map_utils.cuh
├── 🔨 Created mount index_patterns.cuh
├── 🔨 Created mount gather4_kernels.cuh
├── 🔨 Created mount main.cu
└── 🔨 Created function run_benchmark.
Launching TMA gather4 benchmark (experiment=all) on B200...
GPU: NVIDIA B200, 580.95.05, 183359 MiB
NVCC: Build cuda_13.0.r13.0/compiler.36424714_0

Compiling with: nvcc -std=c++20 -gencode arch=compute_100a,code=sm_100a -O3 --expt-relaxed-constexpr -Xptxas -O3 -Xptxas --allow-expensive-optimizations=true -Xptxas -v -lineinfo --ftz=true -o /root/bench/tma_gather4_bench /root/bench/main.cu -lcuda
Compilation successful!
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi32EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi32EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers
ptxas info    : Compile time = 26.987 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi16EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi16EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers
ptxas info    : Compile time = 15.178 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi8EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi8EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 18 registers, used 0 barriers
ptxas info    : Compile time = 8.447 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi4EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi4EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 18 registers, used 0 barriers
ptxas info    : Compile time = 10.895 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi2EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi2EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, used 0 barriers
ptxas info    : Compile time = 13.432 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi1EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi1EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 0 barriers
ptxas info    : Compile time = 16.288 ms
ptxas info    : Compiling entry function '_Z14latency_kernel14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z14latency_kernel14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, used 0 barriers
ptxas info    : Compile time = 18.702 ms
ptxas info    : Compiling entry function '_Z17throughput_kernel14CUtensorMap_stPKiiiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z17throughput_kernel14CUtensorMap_stPKiiiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers
ptxas info    : Compile time = 12.898 ms
ptxas info    : Compiling entry function '_Z12power_kernelPdi' for 'sm_100a'
ptxas info    : Function properties for _Z12power_kernelPdi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers
ptxas info    : Compile time = 1.544 ms
/root/bench/tensor_map_utils.cuh: In function ‘CUresult (* get_cuTensorMapEncodeTiled())(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill)’:
/root/bench/tensor_map_utils.cuh:34:40: warning: ‘cudaError_t cudaGetDriverEntryPoint(const char*, void**, long long unsigned int, cudaDriverEntryPointQueryResult*)’ is deprecated [-Wdeprecated-declarations]
   34 |     CUDA_CHECK(cudaGetDriverEntryPoint(
      |                 ~~~~~~~~~~~~~~~~~~~~~~~^                                                              
/usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime_api.h:13101:46: note: declared here
13101 | extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, enum cudaDriverEntryPointQueryResult *driverStatus = NULL);
      |                                              ^~~~~~~~~~~~~~~~~~~~~~~


Running experiment: all
[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x505730=0x1702000e 0x505734=0x20 0x505728=0x1f81fb60 0x50572c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 0, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5057b0=0x1701000e 0x5057b4=0x20 0x5057a8=0x1f81fb60 0x5057ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 1, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x506730=0x1701000e 0x506734=0x20 0x506728=0x1f81fb60 0x50672c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 1, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5067b0=0x1700000e 0x5067b4=0x20 0x5067a8=0x1f81fb60 0x5067ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 3, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x508730=0x1702000e 0x508734=0x20 0x508728=0x1f81fb60 0x50872c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 3, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5087b0=0x1702000e 0x5087b4=0x20 0x5087a8=0x1f81fb60 0x5087ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 5, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x50a730=0x1703000e 0x50a734=0x20 0x50a728=0x1f81fb60 0x50a72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 5, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x50a7b0=0x1701000e 0x50a7b4=0x20 0x50a7a8=0x1f81fb60 0x50a7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 7, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x50c730=0x1702000e 0x50c734=0x20 0x50c728=0x1f81fb60 0x50c72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 7, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x50c7b0=0x1702000e 0x50c7b4=0x20 0x50c7a8=0x1f81fb60 0x50c7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 0, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x515730=0x1700000e 0x515734=0x20 0x515728=0x1f81fb60 0x51572c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 0, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5157b0=0x1700000e 0x5157b4=0x20 0x5157a8=0x1f81fb60 0x5157ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 1, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x516730=0x1701000e 0x516734=0x20 0x516728=0x1f81fb60 0x51672c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 1, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5167b0=0x1703000e 0x5167b4=0x20 0x5167a8=0x1f81fb60 0x5167ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 2, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x517730=0x1703000e 0x517734=0x20 0x517728=0x1f81fb60 0x51772c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 2, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5177b0=0x1700000e 0x5177b4=0x20 0x5177a8=0x1f81fb60 0x5177ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 3, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x518730=0x1700000e 0x518734=0x20 0x518728=0x1f81fb60 0x51872c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 3, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5187b0=0x1702000e 0x5187b4=0x20 0x5187a8=0x1f81fb60 0x5187ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 4, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x519730=0x1703000e 0x519734=0x20 0x519728=0x1f81fb60 0x51972c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 4, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5197b0=0x1702000e 0x5197b4=0x20 0x5197a8=0x1f81fb60 0x5197ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 5, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x51a730=0x1703000e 0x51a734=0x20 0x51a728=0x1f81fb60 0x51a72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 5, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x51a7b0=0x1702000e 0x51a7b4=0x20 0x51a7a8=0x1f81fb60 0x51a7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 6, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x51b730=0x1700000e 0x51b734=0x20 0x51b728=0x1f81fb60 0x51b72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 1, TPC 6, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x51b7b0=0x1702000e 0x51b7b4=0x20 0x51b7a8=0x1f81fb60 0x51b7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 0, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x525730=0x1701000e 0x525734=0x20 0x525728=0x1f81fb60 0x52572c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 0, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5257b0=0x1702000e 0x5257b4=0x20 0x5257a8=0x1f81fb60 0x5257ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 1, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x526730=0x1703000e 0x526734=0x20 0x526728=0x1f81fb60 0x52672c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 1, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5267b0=0x1703000e 0x5267b4=0x20 0x5267a8=0x1f81fb60 0x5267ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 2, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x527730=0x1701000e 0x527734=0x20 0x527728=0x1f81fb60 0x52772c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 2, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5277b0=0x1703000e 0x5277b4=0x20 0x5277a8=0x1f81fb60 0x5277ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 5, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x52a730=0x1702000e 0x52a734=0x20 0x52a728=0x1f81fb60 0x52a72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 5, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x52a7b0=0x1701000e 0x52a7b4=0x20 0x52a7a8=0x1f81fb60 0x52a7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 7, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x52c730=0x1701000e 0x52c734=0x20 0x52c728=0x1f81fb60 0x52c72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 2, TPC 7, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x52c7b0=0x1701000e 0x52c7b4=0x20 0x52c7a8=0x1f81fb60 0x52c7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 3, TPC 0, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x535730=0x1702000e 0x535734=0x20 0x535728=0x1f81fb60 0x53572c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 3, TPC 0, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5357b0=0x1700000e 0x5357b4=0x20 0x5357a8=0x1f81fb60 0x5357ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 3, TPC 2, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x537730=0x1702000e 0x537734=0x20 0x537728=0x1f81fb60 0x53772c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 3, TPC 2, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5377b0=0x1702000e 0x5377b4=0x20 0x5377a8=0x1f81fb60 0x5377ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 3, TPC 5, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x53a730=0x1702000e 0x53a734=0x20 0x53a728=0x1f81fb60 0x53a72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 3, TPC 5, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x53a7b0=0x1700000e 0x53a7b4=0x20 0x53a7a8=0x1f81fb60 0x53a7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 3, TPC 7, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x53c730=0x1702000e 0x53c734=0x20 0x53c728=0x1f81fb60 0x53c72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 3, TPC 7, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x53c7b0=0x1700000e 0x53c7b4=0x20 0x53c7a8=0x1f81fb60 0x53c7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 0, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x545730=0x1700000e 0x545734=0x20 0x545728=0x1f81fb60 0x54572c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 0, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5457b0=0x1700000e 0x5457b4=0x20 0x5457a8=0x1f81fb60 0x5457ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 3, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x548730=0x1703000e 0x548734=0x20 0x548728=0x1f81fb60 0x54872c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 3, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5487b0=0x1703000e 0x5487b4=0x20 0x5487a8=0x1f81fb60 0x5487ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 5, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x54a730=0x1700000e 0x54a734=0x20 0x54a728=0x1f81fb60 0x54a72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 5, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x54a7b0=0x1703000e 0x54a7b4=0x20 0x54a7a8=0x1f81fb60 0x54a7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 6, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x54b730=0x1701000e 0x54b734=0x20 0x54b728=0x1f81fb60 0x54b72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 6, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x54b7b0=0x1702000e 0x54b7b4=0x20 0x54b7a8=0x1f81fb60 0x54b7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 8, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x54d730=0x1703000e 0x54d734=0x20 0x54d728=0x1f81fb60 0x54d72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 4, TPC 8, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x54d7b0=0x1702000e 0x54d7b4=0x20 0x54d7a8=0x1f81fb60 0x54d7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 5, TPC 0, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x555730=0x1703000e 0x555734=0x20 0x555728=0x1f81fb60 0x55572c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 5, TPC 0, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5557b0=0x1702000e 0x5557b4=0x20 0x5557a8=0x1f81fb60 0x5557ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 5, TPC 3, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x558730=0x1701000e 0x558734=0x20 0x558728=0x1f81fb60 0x55872c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 5, TPC 3, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5587b0=0x1700000e 0x5587b4=0x20 0x5587a8=0x1f81fb60 0x5587ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 5, TPC 6, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x55b730=0x1700000e 0x55b734=0x20 0x55b728=0x1f81fb60 0x55b72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 5, TPC 6, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x55b7b0=0x1700000e 0x55b7b4=0x20 0x55b7a8=0x1f81fb60 0x55b7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 5, TPC 8, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x55d730=0x1701000e 0x55d734=0x20 0x55d728=0x1f81fb60 0x55d72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 5, TPC 8, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x55d7b0=0x1701000e 0x55d7b4=0x20 0x55d7a8=0x1f81fb60 0x55d7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 6, TPC 0, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x565730=0x1703000e 0x565734=0x20 0x565728=0x1f81fb60 0x56572c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 6, TPC 0, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5657b0=0x1700000e 0x5657b4=0x20 0x5657a8=0x1f81fb60 0x5657ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 6, TPC 3, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x568730=0x1700000e 0x568734=0x20 0x568728=0x1f81fb60 0x56872c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 6, TPC 3, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5687b0=0x1701000e 0x5687b4=0x20 0x5687a8=0x1f81fb60 0x5687ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 6, TPC 6, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x56b730=0x1700000e 0x56b734=0x20 0x56b728=0x1f81fb60 0x56b72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 6, TPC 6, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x56b7b0=0x1701000e 0x56b7b4=0x20 0x56b7a8=0x1f81fb60 0x56b7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 6, TPC 8, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x56d730=0x1700000e 0x56d734=0x20 0x56d728=0x1f81fb60 0x56d72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 6, TPC 8, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x56d7b0=0x1700000e 0x56d7b4=0x20 0x56d7a8=0x1f81fb60 0x56d7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 7, TPC 0, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x575730=0x1702000e 0x575734=0x20 0x575728=0x1f81fb60 0x57572c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 7, TPC 0, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5757b0=0x1701000e 0x5757b4=0x20 0x5757a8=0x1f81fb60 0x5757ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 7, TPC 3, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x578730=0x1702000e 0x578734=0x20 0x578728=0x1f81fb60 0x57872c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 7, TPC 3, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x5787b0=0x1700000e 0x5787b4=0x20 0x5787a8=0x1f81fb60 0x5787ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 7, TPC 6, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x57b730=0x1700000e 0x57b734=0x20 0x57b728=0x1f81fb60 0x57b72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 7, TPC 6, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x57b7b0=0x1701000e 0x57b7b4=0x20 0x57b7a8=0x1f81fb60 0x57b7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 7, TPC 8, SM 0): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x57d730=0x1703000e 0x57d734=0x20 0x57d728=0x1f81fb60 0x57d72c=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics SM Warp Exception on (GPC 7, TPC 8, SM 1): Out Of Range Address

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 13, Graphics Exception: ESR 0x57d7b0=0x1701000e 0x57d7b4=0x20 0x57d7a8=0x1f81fb60 0x57d7ac=0x1174

[gpu-health] [WARN] GPU-98645400-65f3-e471-005b-a66c1508c08d: XID: NVRM: Xid (PCI:0000:c5:00): 43, pid=260761, name=tma_gather4_ben, channel 0x00000002

Execution failed:
Device 0: NVIDIA B200 (SM 10.0, 148 SMs, 178 GB HBM)
Warming up GPU clocks...
Clock warmup done.

[Experiment 1: Throughput vs Box Size]
  int64 box=8...
  throughput: blocks=76 rows=524288 gather4_calls=131072 col_steps=8
  int64 box=16...
  throughput: blocks=76 rows=524288 gather4_calls=131072 col_steps=4
  int64 box=32...
  throughput: blocks=76 rows=524288 gather4_calls=131072 col_steps=2
  int64 box=64...
  throughput: blocks=76 rows=524288 gather4_calls=131072 col_steps=1
  bf16 box=32...
  throughput: blocks=76 rows=262144 gather4_calls=65536 col_steps=16
CUDA error: an illegal memory access was encountered  /root/bench/main.cu:105


Error: Device 0: NVIDIA B200 (SM 10.0, 148 SMs, 178 GB HBM)
Warming up GPU clocks...
Clock warmup done.

[Experiment 1: Throughput vs Box Size]
  int64 box=8...
  throughput: blocks=76 rows=524288 gather4_calls=131072 col_steps=8
  int64 box=16...
  throughput: blocks=76 rows=524288 gather4_calls=131072 col_steps=4
  int64 box=32...
  throughput: blocks=76 rows=524288 gather4_calls=131072 col_steps=2
  int64 box=64...
  throughput: blocks=76 rows=524288 gather4_calls=131072 col_steps=1
  bf16 box=32...
  throughput: blocks=76 rows=262144 gather4_calls=65536 col_steps=16
CUDA error: an illegal memory access was encountered  /root/bench/main.cu:105

Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/mark123ymm/main/ap-UcKTAuFZfosVJwOdebus3g
