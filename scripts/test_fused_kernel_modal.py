"""Minimal test for the fused 3-warpgroup CuTeDSL kernel on Modal B200."""

import modal

app = modal.App("test-fused-kernel")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("nvidia-cutlass-dsl", "torch")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_fused_kernel():
    import time
    import torch

    print("=== Testing fused kernel compilation ===")
    t0 = time.time()

    # Import the kernel module
    import importlib.util, sys, os
    spec = importlib.util.spec_from_file_location("kernel", "/root/kernel.py")
    kernel_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_mod)

    print(f"Module loaded in {time.time()-t0:.1f}s")

    # Create small test inputs
    T = 1
    NUM_Q_HEADS = 16
    CKV_DIM = 512
    KPE_DIM = 64
    PAGE_SIZE = 64
    TOPK = 2048

    q_nope = torch.randn((T, NUM_Q_HEADS, CKV_DIM), device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn((T, NUM_Q_HEADS, KPE_DIM), device="cuda", dtype=torch.bfloat16)
    ckv_cache = torch.randn((8, PAGE_SIZE, CKV_DIM), device="cuda", dtype=torch.bfloat16)
    kpe_cache = torch.randn((8, PAGE_SIZE, KPE_DIM), device="cuda", dtype=torch.bfloat16)
    sparse_indices = torch.randint(0, 8 * PAGE_SIZE, (T, TOPK), device="cuda", dtype=torch.int32)
    sm_scale = 0.135

    print("Inputs created")

    # Try to JIT compile the fused kernel
    print("=== Starting fused kernel JIT compilation ===")
    t1 = time.time()

    try:
        from cutlass.cute.runtime import from_dlpack
        import cuda.bindings.driver as cuda

        # Initialize helpers
        helpers = kernel_mod._get_sm100_dsl_helpers()
        print(f"SM100 helpers loaded in {time.time()-t1:.1f}s")

        # Initialize the split shell launcher (triggers JIT)
        t2 = time.time()
        launcher = kernel_mod._make_split_shell_launcher()
        print(f"Split shell launcher created in {time.time()-t2:.1f}s")

        # Prepare operands
        t3 = time.time()
        (
            _partial_o_tma, partial_o, partial_lse,
            mQ, mQp, mQn_padded, mQp_padded,
            mCkv, mKpe, mIdx, mO, mLse,
        ) = kernel_mod._prepare_split_shell_operands(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices
        )
        print(f"Operands prepared in {time.time()-t3:.1f}s")

        # Try calling the launcher (triggers JIT compilation of kernel)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        print("=== Calling launcher (JIT compile + launch) ===")
        t4 = time.time()
        launcher(
            mQ, mQp, mQn_padded, mQp_padded, mCkv, mKpe, mIdx, mO, mLse,
            float(sm_scale), stream=stream,
        )
        print(f"Launcher returned in {time.time()-t4:.1f}s")

        print("=== Synchronizing GPU ===")
        t5 = time.time()
        torch.cuda.synchronize()
        print(f"GPU synchronized in {time.time()-t5:.1f}s")

        print("=== FUSED KERNEL COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


@app.local_entrypoint()
def main():
    import os, shutil
    # Upload kernel.py to the modal function
    kernel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "solution", "dsa_attention", "kernel.py")
    print(f"Uploading kernel from: {kernel_path}")

    # Read the kernel file
    with open(kernel_path, 'r') as f:
        kernel_source = f.read()

    # We need to put the kernel file in the container
    # Use a volume or mount
    test_fused_kernel.remote()
