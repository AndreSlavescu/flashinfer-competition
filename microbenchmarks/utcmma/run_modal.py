import modal

app = modal.App("utcmma-bench")

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        # Try CUDA 13.0 first, fall back to 12.8
        "apt-get install -y cuda-toolkit-13-0 || apt-get install -y cuda-toolkit-12-8",
    )
    .env({
        "PATH": "/usr/local/cuda/bin:${PATH}",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
    })
    # Benchmark source files
    .add_local_file("microbenchmarks/common/benchmark_common.cuh", "/root/common/benchmark_common.cuh", copy=True)
    .add_local_file("microbenchmarks/utcmma/utcmma_kernels.cuh", "/root/bench/utcmma_kernels.cuh", copy=True)
    .add_local_file("microbenchmarks/utcmma/main.cu", "/root/bench/main.cu", copy=True)
    # CUTLASS/CuTe headers (needed for TMEM allocator, UMMA descriptors, MMA traits)
    .add_local_dir("csrc/cutlass/include", "/root/bench/include", copy=True)
    # kerutils headers (gemm.cuh, helpers.cuh, intrinsics.cuh)
    .add_local_dir("csrc/kerutils/include", "/root/bench/kerutils_include", copy=True)
)

NVCC_FLAGS = [
    "nvcc",
    "-std=c++20",
    "-gencode", "arch=compute_100a,code=sm_100a",
    "-O3",
    "--expt-relaxed-constexpr",
    # ptxas: max optimization + expensive opts + report resource usage
    "-Xptxas", "-O3",
    "-Xptxas", "--allow-expensive-optimizations=true",
    "-Xptxas", "-v",
    # Line info for profiling (negligible overhead, enables ncu source correlation)
    "-lineinfo",
    # Flush denormals to zero
    "--ftz=true",
    # Include paths for CuTe/CUTLASS and kerutils
    "-I/root/bench/include",
    "-I/root/bench/kerutils_include",
    # Output
    "-o", "/root/bench/utcmma_bench",
    "/root/bench/main.cu",
]


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=900,
)
def run_benchmark(experiment: str = "all"):
    import subprocess

    # Check CUDA/GPU info
    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"GPU: {info.stdout.strip()}")

    nvcc_ver = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print(f"NVCC: {nvcc_ver.stdout.strip().splitlines()[-1]}")

    # Compile
    print(f"\nCompiling with: {' '.join(NVCC_FLAGS)}")
    compile_result = subprocess.run(NVCC_FLAGS, capture_output=True, text=True, timeout=300)
    if compile_result.returncode != 0:
        print(f"Compilation failed:\n{compile_result.stderr}")
        return {"success": False, "error": compile_result.stderr}
    # ptxas -v output goes to stderr — shows register/smem usage
    print(f"Compilation successful!\n{compile_result.stderr}")

    # Run
    print(f"\nRunning experiment: {experiment}")
    run_result = subprocess.run(
        ["/root/bench/utcmma_bench", f"--experiment={experiment}", "--verbose"],
        capture_output=True, text=True,
        timeout=600,
    )
    if run_result.returncode != 0:
        print(f"Execution failed:\n{run_result.stderr}")
        return {"success": False, "error": run_result.stderr, "stderr": run_result.stderr}

    print("--- stderr (progress) ---")
    print(run_result.stderr)
    print("--- stdout (CSV) ---")
    print(run_result.stdout)

    return {
        "success": True,
        "csv": run_result.stdout,
        "log": run_result.stderr,
    }


@app.local_entrypoint()
def main(experiment: str = "all"):
    """Run UTCMMA benchmarks on B200 GPU.

    Usage:
        modal run run_modal.py                                    # all experiments
        modal run run_modal.py --experiment utcmma_latency_ts     # single experiment
        modal run run_modal.py --experiment utcmma_kdepth_ts
        modal run run_modal.py --experiment utcmma_latency_ss
        modal run run_modal.py --experiment utccp_latency
        modal run run_modal.py --experiment utcmma_swizzle
    """
    print(f"Launching UTCMMA benchmark (experiment={experiment}) on B200...")
    result = run_benchmark.remote(experiment)

    if result["success"]:
        print("\n=== CSV Results ===")
        print(result["csv"])
    else:
        print(f"\nError: {result['error']}")
