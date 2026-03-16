
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
ptxas info    : Compile time = 22.057 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi16EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi16EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers
ptxas info    : Compile time = 12.427 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi8EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi8EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 18 registers, used 0 barriers
ptxas info    : Compile time = 8.133 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi4EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi4EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 18 registers, used 0 barriers
ptxas info    : Compile time = 8.764 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi2EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi2EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, used 0 barriers
ptxas info    : Compile time = 10.572 ms
ptxas info    : Compiling entry function '_Z15pipeline_kernelILi1EEv14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z15pipeline_kernelILi1EEv14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 0 barriers
ptxas info    : Compile time = 12.314 ms
ptxas info    : Compiling entry function '_Z14latency_kernel14CUtensorMap_stPKiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z14latency_kernel14CUtensorMap_stPKiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 0 barriers
ptxas info    : Compile time = 14.345 ms
ptxas info    : Compiling entry function '_Z17throughput_kernel14CUtensorMap_stPKiiiiiilPl' for 'sm_100a'
ptxas info    : Function properties for _Z17throughput_kernel14CUtensorMap_stPKiiiiiilPl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers
ptxas info    : Compile time = 9.186 ms
ptxas info    : Compiling entry function '_Z12power_kernelPdi' for 'sm_100a'
ptxas info    : Function properties for _Z12power_kernelPdi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers
ptxas info    : Compile time = 1.201 ms
/root/bench/tensor_map_utils.cuh: In function ‘CUresult (* get_cuTensorMapEncodeTiled())(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill)’:
/root/bench/tensor_map_utils.cuh:34:40: warning: ‘cudaError_t cudaGetDriverEntryPoint(const char*, void**, long long unsigned int, cudaDriverEntryPointQueryResult*)’ is deprecated [-Wdeprecated-declarations]
   34 |     CUDA_CHECK(cudaGetDriverEntryPoint(
      |                 ~~~~~~~~~~~~~~~~~~~~~~~^                                                              
/usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime_api.h:13101:46: note: declared here
13101 | extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, enum cudaDriverEntryPointQueryResult *driverStatus = NULL);
      |                                              ^~~~~~~~~~~~~~~~~~~~~~~


Running experiment: all

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
  bf16 box=64...
  throughput: blocks=76 rows=262144 gather4_calls=65536 col_steps=8
  bf16 box=128...
cuTensorMapEncodeTiled failed: invalid argument
  data_type=9 dim0=512 num_rows=262144 stride=1024 box_dim0=128 swizzle=3

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
  bf16 box=64...
  throughput: blocks=76 rows=262144 gather4_calls=65536 col_steps=8
  bf16 box=128...
cuTensorMapEncodeTiled failed: invalid argument
  data_type=9 dim0=512 num_rows=262144 stride=1024 box_dim0=128 swizzle=3
  