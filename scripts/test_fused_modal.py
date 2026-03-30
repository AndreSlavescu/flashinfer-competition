"""Test fused kernel compile/run on Modal B200."""
import modal
import os

app = modal.App("test-fused-kernel")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("nvidia-cutlass-dsl", "torch")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_fused(kernel_source: str, mode: str = "compile"):
    """Test the fused kernel on B200."""
    import subprocess, sys

    kernel_path = "/tmp/kernel.py"
    with open(kernel_path, "w") as f:
        f.write(kernel_source)

    cmd_arg = "fused-compile" if mode == "compile" else "fused-run"
    print(f"Running: python3 {kernel_path} {cmd_arg}")

    try:
        result = subprocess.run(
            [sys.executable, kernel_path, cmd_arg],
            capture_output=True, text=True, timeout=300
        )
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-3000:])
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Process did not complete within 300 seconds")
        return False


@app.local_entrypoint()
def main(mode: str = "compile"):
    # Read the kernel source
    kernel_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "solution", "dsa_attention", "kernel.py"
    )
    with open(kernel_path, "r") as f:
        kernel_source = f.read()

    print(f"Kernel source: {len(kernel_source)} chars, mode: {mode}")
    ok = test_fused.remote(kernel_source, mode)
    print(f"\nFused {mode} test: {'PASS' if ok else 'FAIL'}")
