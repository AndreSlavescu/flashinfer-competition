import modal

app = modal.App("tma-gather4-bench")

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
    .add_local_file("microbenchmarks/tma_gather4/common.cuh", "/root/bench/common.cuh", copy=True)
    .add_local_file("microbenchmarks/tma_gather4/tensor_map_utils.cuh", "/root/bench/tensor_map_utils.cuh", copy=True)
    .add_local_file("microbenchmarks/tma_gather4/index_patterns.cuh", "/root/bench/index_patterns.cuh", copy=True)
    .add_local_file("microbenchmarks/tma_gather4/gather4_kernels.cuh", "/root/bench/gather4_kernels.cuh", copy=True)
    .add_local_file("microbenchmarks/tma_gather4/main.cu", "/root/bench/main.cu", copy=True)
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
    # Flush denormals to zero (avoids slow FP corner cases)
    "--ftz=true",
    # Link
    "-o", "/root/bench/tma_gather4_bench",
    "/root/bench/main.cu",
    "-lcuda",
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
    compile_result = subprocess.run(NVCC_FLAGS, capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Compilation failed:\n{compile_result.stderr}")
        return {"success": False, "error": compile_result.stderr}
    # ptxas -v output goes to stderr — shows register/smem usage
    print(f"Compilation successful!\n{compile_result.stderr}")

    # Run
    print(f"\nRunning experiment: {experiment}")
    run_result = subprocess.run(
        ["/root/bench/tma_gather4_bench", f"--experiment={experiment}", "--verbose"],
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
    """Run TMA gather4 benchmarks on B200 GPU.

    Usage:
        modal run run_modal.py                          # all experiments
        modal run run_modal.py --experiment throughput   # single experiment
        modal run run_modal.py --experiment latency
    """
    print(f"Launching TMA gather4 benchmark (experiment={experiment}) on B200...")
    result = run_benchmark.remote(experiment)

    if result["success"]:
        print("\n=== CSV Results ===")
        print(result["csv"])
    else:
        print(f"\nError: {result['error']}")
