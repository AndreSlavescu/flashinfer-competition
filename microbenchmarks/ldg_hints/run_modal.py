import modal

app = modal.App("ldg-hints-bench")

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
    # Benchmark source files
    .add_local_file("microbenchmarks/common/benchmark_common.cuh", "/root/common/benchmark_common.cuh", copy=True)
    .add_local_file("microbenchmarks/ldg_hints/ldg_kernels.cuh", "/root/bench/ldg_kernels.cuh", copy=True)
    .add_local_file("microbenchmarks/ldg_hints/main.cu", "/root/bench/main.cu", copy=True)
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
    "-I/root/common",
    "-o", "/root/bench/ldg_bench",
    "/root/bench/main.cu",
]


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=600,
)
def run_benchmark(experiment: str = "all"):
    import subprocess

    # GPU info
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
    print(f"Compilation successful!\n{compile_result.stderr}")

    # Run
    print(f"\nRunning experiment: {experiment}")
    try:
        run_result = subprocess.run(
            ["/root/bench/ldg_bench", f"--experiment={experiment}", "--verbose"],
            capture_output=True, text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired as e:
        partial_stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        partial_stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        print("=== TIMEOUT after 240s ===")
        print(partial_stderr[-8000:] if len(partial_stderr) > 8000 else partial_stderr)
        print("=== partial stdout (CSV so far) ===")
        print(partial_stdout)
        return {"success": False, "error": "timeout_240s", "stderr": partial_stderr, "csv": partial_stdout}

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
    """Run LDG.256 hint sweep benchmarks on B200 GPU.

    Usage:
        modal run run_modal.py                               # all experiments
        modal run run_modal.py --experiment throughput        # streaming bandwidth (256 MB, HBM)
        modal run run_modal.py --experiment competition       # sparse scatter pattern
        modal run run_modal.py --experiment latency           # pointer-chase at 4 WS sizes
        modal run run_modal.py --experiment pchase_sweep      # fine-grained WS sweep 4KB-512MB
    """
    print(f"Launching LDG.256 hint benchmark (experiment={experiment}) on B200...")
    result = run_benchmark.remote(experiment)

    if result["success"]:
        print("\n=== CSV Results ===")
        print(result["csv"])
    else:
        print(f"\nError: {result['error']}")
