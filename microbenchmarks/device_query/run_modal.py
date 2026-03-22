import modal

app = modal.App("device-query-bench")

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
    .add_local_file("microbenchmarks/device_query/device_query.cu",
                    "/root/bench/device_query.cu", copy=True)
)

# Pure host code — no SM100a gencode needed; -lcuda links the Driver API
NVCC_FLAGS = [
    "nvcc",
    "-std=c++17",
    "-O2",
    "-o", "/root/bench/device_query",
    "/root/bench/device_query.cu",
    "-lcuda",
]


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=120,
)
def run_device_query():
    import subprocess

    # GPU info from nvidia-smi
    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"GPU (nvidia-smi): {info.stdout.strip()}")

    nvcc_ver = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print(f"NVCC: {nvcc_ver.stdout.strip().splitlines()[-1]}\n")

    # Compile
    print(f"Compiling: {' '.join(NVCC_FLAGS)}")
    compile_result = subprocess.run(NVCC_FLAGS, capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Compilation failed:\n{compile_result.stderr}")
        return {"success": False, "error": compile_result.stderr}
    if compile_result.stderr:
        print(f"Compiler output:\n{compile_result.stderr}")

    # Run
    print("\nRunning device query...\n")
    run_result = subprocess.run(
        ["/root/bench/device_query"],
        capture_output=True, text=True,
        timeout=30,
    )
    if run_result.returncode != 0:
        print(f"Execution failed:\n{run_result.stderr}")
        return {"success": False, "error": run_result.stderr}

    print(run_result.stdout)
    if run_result.stderr:
        print(f"stderr: {run_result.stderr}")

    return {"success": True, "output": run_result.stdout}


@app.local_entrypoint()
def main():
    """Query B200 hardware properties via CUDA Runtime + Driver API.

    Usage:
        modal run run_modal.py
    """
    print("Launching device query on B200...")
    result = run_device_query.remote()
    if not result["success"]:
        print(f"\nError: {result['error']}")
    else:
        print("\n=== Device Query Complete ===")
