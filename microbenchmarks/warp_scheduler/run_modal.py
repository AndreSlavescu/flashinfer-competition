import modal

app = modal.App("warp-scheduler-bench")

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
    .add_local_file("microbenchmarks/common/benchmark_common.cuh",
                    "/root/common/benchmark_common.cuh", copy=True)
    .add_local_file("microbenchmarks/warp_scheduler/warp_scheduler_kernels.cuh",
                    "/root/bench/warp_scheduler_kernels.cuh", copy=True)
    .add_local_file("microbenchmarks/warp_scheduler/main.cu",
                    "/root/bench/main.cu", copy=True)
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
    "-o", "/root/bench/warp_scheduler_bench",
    "/root/bench/main.cu",
    "-lcuda",
]


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=300,
)
def run_benchmark(experiment: str = "all", iters: int = 100000, trials: int = 20):
    import subprocess

    # GPU info
    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
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
    print(f"Compilation successful!\n{compile_result.stderr}")

    # Run
    cmd = [
        "/root/bench/warp_scheduler_bench",
        f"--experiment={experiment}",
        f"--iters={iters}",
        f"--trials={trials}",
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    run_result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)

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
def main(experiment: str = "all", iters: int = 100000, trials: int = 20):
    """Warp sub-core partitioning benchmark on B200.

    Discovers whether warp_id % 4 maps to sub-cores (like Volta).
    Contention ratio ≈ 1.3 for same-sub-core pairs, ≈ 1.0 for different.

    Usage:
        modal run run_modal.py                                     # all experiments
        modal run run_modal.py --experiment single_warp_baseline   # baseline only
        modal run run_modal.py --experiment two_warp_sweep         # 28-pair sweep
        modal run run_modal.py --experiment warp_count_scaling     # scaling
    """
    print(f"Launching warp-scheduler-bench (experiment={experiment}) on B200...")
    result = run_benchmark.remote(experiment, iters, trials)

    if result["success"]:
        print("\n=== CSV Results ===")
        print(result["csv"])
    else:
        print(f"\nError: {result['error']}")
