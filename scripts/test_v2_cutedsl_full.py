"""
Full correctness test of kernel_v2_cutedsl.py across all competition shapes.

Usage:
    .venv/bin/modal run scripts/test_v2_cutedsl_full.py
"""

from pathlib import Path

import modal

SOLUTION_DIR = Path(__file__).parent.parent / "solution" / "dsa_attention"
REFERENCE_DIR = Path(__file__).parent.parent / "references"

app = modal.App("test-v2-cutedsl-full")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "nvidia-cutlass-dsl")
    .add_local_file(
        str(SOLUTION_DIR / "kernel_v2_cutedsl.py"),
        remote_path="/root/kernel_v2_cutedsl.py",
    )
    .add_local_file(
        str(REFERENCE_DIR / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py"),
        remote_path="/root/reference.py",
    )
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_all_shapes():
    """Test across all competition num_tokens values: 1, 2, 6, 7, 8."""
    import torch
    import math
    import importlib.util

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Load modules
    spec_k = importlib.util.spec_from_file_location("kv2", "/root/kernel_v2_cutedsl.py")
    kernel_mod = importlib.util.module_from_spec(spec_k)
    spec_k.loader.exec_module(kernel_mod)

    spec_r = importlib.util.spec_from_file_location("ref", "/root/reference.py")
    ref_mod = importlib.util.module_from_spec(spec_r)
    spec_r.loader.exec_module(ref_mod)

    num_pages = 8462
    page_size = 64
    topk = 2048
    h_q = 16
    ckv_dim = 512
    kpe_dim = 64
    sm_scale = 0.1352337788608801
    device = "cuda"

    results = []

    for num_tokens in [1, 2, 6, 7, 8]:
        print(f"\n{'='*60}")
        print(f"  Testing num_tokens={num_tokens}")
        print(f"{'='*60}")

        torch.manual_seed(42 + num_tokens)
        torch.cuda.manual_seed(42 + num_tokens)

        total_kv = num_pages * page_size
        q_nope = torch.randn(num_tokens, h_q, ckv_dim, dtype=torch.bfloat16, device=device) / 10.0
        q_pe = torch.randn(num_tokens, h_q, kpe_dim, dtype=torch.bfloat16, device=device) / 10.0
        ckv_cache = torch.randn(num_pages, page_size, ckv_dim, dtype=torch.bfloat16, device=device) / 10.0
        kpe_cache = torch.randn(num_pages, page_size, kpe_dim, dtype=torch.bfloat16, device=device) / 10.0
        sparse_indices = torch.randint(0, total_kv, (num_tokens, topk), dtype=torch.int32, device=device)
        invalid_mask = torch.rand(num_tokens, topk, device=device) < 0.05
        sparse_indices[invalid_mask] = -1

        # Reference
        ref_o, ref_lse = ref_mod.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

        # Test kernel
        test_o, test_lse = kernel_mod.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

        # Check dtypes
        assert test_o.dtype == torch.bfloat16, f"Output dtype: {test_o.dtype}"
        assert test_lse.dtype == torch.float32, f"LSE dtype: {test_lse.dtype}"
        assert test_o.shape == ref_o.shape, f"Output shape: {test_o.shape} vs {ref_o.shape}"
        assert test_lse.shape == ref_lse.shape, f"LSE shape: {test_lse.shape} vs {ref_lse.shape}"

        # Numerical comparison
        o_err = (test_o.float() - ref_o.float()).abs().max().item()
        l_err = (test_lse - ref_lse).abs().max().item()

        def cos_diff(a, b):
            a, b = a.double(), b.double()
            d = (a * a + b * b).sum().item()
            return abs(1 - 2 * (a * b).sum().item() / d) if d > 1e-12 else 0.0

        o_cos = cos_diff(test_o.float(), ref_o.float())
        l_cos = cos_diff(test_lse, ref_lse)

        # Competition tolerances
        o_pass = o_err < 1e-2 and o_cos < 1e-6
        l_pass = l_err < 1e-3 and l_cos < 1e-6
        passed = o_pass and l_pass

        status = "PASS" if passed else "FAIL"
        print(f"  Output: max_abs={o_err:.2e}, cos_diff={o_cos:.2e} {'OK' if o_pass else 'FAIL'}")
        print(f"  LSE:    max_abs={l_err:.2e}, cos_diff={l_cos:.2e} {'OK' if l_pass else 'FAIL'}")
        print(f"  Result: {status}")

        results.append({
            "num_tokens": num_tokens,
            "passed": passed,
            "output_max_abs": o_err,
            "output_cos_diff": o_cos,
            "lse_max_abs": l_err,
            "lse_cos_diff": l_cos,
        })

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  num_tokens={r['num_tokens']}: {status} (o_abs={r['output_max_abs']:.2e}, l_abs={r['lse_max_abs']:.2e})")
    print(f"\n  Total: {total_pass}/{total} PASS")

    return results


@app.local_entrypoint()
def main():
    results = test_all_shapes.remote()
    total_pass = sum(1 for r in results if r["passed"])
    print(f"\nFinal: {total_pass}/{len(results)} PASS")
    if total_pass < len(results):
        print("FAILED workloads:")
        for r in results:
            if not r["passed"]:
                print(f"  num_tokens={r['num_tokens']}: o_abs={r['output_max_abs']:.2e}, l_abs={r['lse_max_abs']:.2e}")
