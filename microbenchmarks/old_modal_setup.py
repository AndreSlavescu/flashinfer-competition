import modal

app = modal.App("b200-vector-add-benchmark")

# CUDA image with basic toolkit
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential")
    .run_commands(
        # Install CUDA toolkit
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-6",
    )
    .env({
        "PATH": "/usr/local/cuda/bin:${PATH}",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
    })
    .add_local_file("vector_add.cu", "/root/vector_add.cu", copy=True)
)


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=600,
)
def run_vector_add():
    import subprocess

    # Compile the CUDA code
    print("Compiling CUDA kernel...")
    compile_result = subprocess.run(
        ["/usr/local/cuda/bin/nvcc", "-arch=sm_100a", "/root/vector_add.cu", "-o", "/root/vector_add"],
        capture_output=True,
        text=True,
    )

    if compile_result.returncode != 0:
        print(f"Compilation failed:\n{compile_result.stderr}")
        return {"success": False, "error": compile_result.stderr}

    print("Compilation successful!")

    # Run the kernel
    print("Running vector addition kernel on B200 GPU...")
    run_result = subprocess.run(
        ["/root/vector_add"],
        capture_output=True,
        text=True,
    )

    if run_result.returncode != 0:
        print(f"Execution failed:\n{run_result.stderr}")
        return {"success": False, "error": run_result.stderr}

    print("Execution output:")
    print(run_result.stdout)

    return {
        "success": True,
        "output": run_result.stdout,
        "stderr": run_result.stderr,
    }


@app.local_entrypoint()
def main():
    """Run vector addition benchmark on B200 GPU."""
    print("Running vector addition benchmark on B200 GPU...")

    result = run_vector_add.remote()

    if result["success"]:
        print("\n=== Benchmark Results ===")
        print(result["output"])
    else:
        print(f"\nError: {result['error']}")
