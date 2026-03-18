import modal

app = modal.App("dual-tma-stream-bench")

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-13-0 || apt-get install -y cuda-toolkit-12-8",
    )
    .env({
        "PATH": "/usr/local/cuda/bin:${PATH}",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
    })
    .add_local_file("microbenchmarks/common/benchmark_common.cuh", "/root/common/benchmark_common.cuh", copy=True)
    .add_local_file("microbenchmarks/dual_tma_stream/tensor_map_utils.cuh", "/root/bench/tensor_map_utils.cuh", copy=True)
    .add_local_file("microbenchmarks/dual_tma_stream/dual_tma_kernels.cuh", "/root/bench/dual_tma_kernels.cuh", copy=True)
    .add_local_file("microbenchmarks/dual_tma_stream/main.cu", "/root/bench/main.cu", copy=True)
)

NVCC_FLAGS = [
    "nvcc",
    "-std=c++20",
    "-gencode", "arch=compute_100a,code=sm_100a",
    "-O3",
    "--expt-relaxed-constexpr",
    "-Xptxas", "-O3",
    "-Xptxas", "--allow-expensive-optimizations=true",
    "-Xptxas", "-v",
    "-lineinfo",
    "--ftz=true",
    "-o", "/root/bench/dual_tma_bench",
    "/root/bench/main.cu",
    "-lcuda",
]


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=900,
)
def run_benchmark():
    import subprocess

    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"GPU: {info.stdout.strip()}")

    nvcc_ver = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print(f"NVCC: {nvcc_ver.stdout.strip().splitlines()[-1]}")

    print(f"\nCompiling with: {' '.join(NVCC_FLAGS)}")
    compile_result = subprocess.run(NVCC_FLAGS, capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Compilation failed:\n{compile_result.stderr}")
        return {"success": False, "error": compile_result.stderr}
    print(f"Compilation successful!\n{compile_result.stderr}")

    print("\nRunning dual TMA stream benchmark...")
    run_result = subprocess.run(
        ["/root/bench/dual_tma_bench", "--verbose"],
        capture_output=True, text=True,
        timeout=600,
    )
    if run_result.returncode != 0:
        print(f"Execution failed:\n{run_result.stderr}")
        return {"success": False, "error": run_result.stderr}

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
def main():
    """Run dual TMA stream interference benchmark on B200 GPU."""
    print("Launching dual TMA stream benchmark on B200...")
    result = run_benchmark.remote()

    if result["success"]:
        print("\n=== CSV Results ===")
        print(result["csv"])
    else:
        print(f"\nError: {result['error']}")
