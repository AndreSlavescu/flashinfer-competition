"""
Quick correctness test of kernel_v2_cutedsl.py on Modal B200.

Usage:
    .venv/bin/modal run scripts/test_v2_cutedsl.py
"""

from pathlib import Path

import modal

SOLUTION_DIR = Path(__file__).parent.parent / "solution" / "dsa_attention"

app = modal.App("test-v2-cutedsl")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "nvidia-cutlass-dsl")
    .add_local_file(
        str(SOLUTION_DIR / "kernel_v2_cutedsl.py"),
        remote_path="/root/kernel_v2_cutedsl.py",
    )
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_correctness():
    """Run correctness test on Modal B200."""
    import torch
    import math
    import sys
    import importlib.util

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Generate test inputs
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_tokens = 1
    num_pages = 8462
    page_size = 64
    topk = 2048
    h_q = 16
    ckv_dim = 512
    kpe_dim = 64
    sm_scale = 0.135
    device = "cuda"
    LOG2E = 1.0 / math.log(2.0)

    total_kv = num_pages * page_size
    q_nope = torch.randn(num_tokens, h_q, ckv_dim, dtype=torch.bfloat16, device=device) / 10.0
    q_pe = torch.randn(num_tokens, h_q, kpe_dim, dtype=torch.bfloat16, device=device) / 10.0
    ckv_cache = torch.randn(num_pages, page_size, ckv_dim, dtype=torch.bfloat16, device=device) / 10.0
    kpe_cache = torch.randn(num_pages, page_size, kpe_dim, dtype=torch.bfloat16, device=device) / 10.0
    sparse_indices = torch.randint(0, total_kv, (num_tokens, topk), dtype=torch.int32, device=device)
    invalid_mask = torch.rand(num_tokens, topk, device=device) < 0.05
    sparse_indices[invalid_mask] = -1

    # --- Reference ---
    T, S, H, N, D, Dp = num_tokens, topk // 64, h_q, 64, ckv_dim, kpe_dim
    valid_mask = sparse_indices != -1
    safe = sparse_indices.clamp_min(0).long()
    vm = valid_mask.view(T, S, N)

    kc = ckv_cache.reshape(-1, D)[safe.reshape(-1)].view(T, S, N, D)
    kp = kpe_cache.reshape(-1, Dp)[safe.reshape(-1)].view(T, S, N, Dp)
    kc = kc.masked_fill(~vm.unsqueeze(-1), 0).contiguous()
    kp = kp.masked_fill(~vm.unsqueeze(-1), 0).contiguous()

    qn_f = q_nope.float(); qp_f = q_pe.float()
    kc_f = kc.float(); kp_f = kp.float()

    logits_ckv = torch.einsum("thd,tsnd->tshn", qn_f, kc_f)
    logits_kpe = torch.einsum("thd,tsnd->tshn", qp_f, kp_f)
    logits = (logits_ckv + logits_kpe) * sm_scale
    mask = vm[:, :, None, :]; sv = vm.any(dim=-1, keepdim=True)
    logits.masked_fill_(~mask, -float("inf"))

    plse = torch.full((T, S, H), -float("inf"), dtype=torch.float32, device=device)
    vs = sv.expand(-1, -1, H)
    plse[vs] = torch.logsumexp(logits, dim=-1)[vs] * LOG2E

    logits = torch.where(sv.unsqueeze(-1), logits, torch.zeros_like(logits))
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(sv.unsqueeze(-1), attn, torch.zeros_like(attn))
    po = torch.einsum("tshn,tsnd->tshd", attn, kc_f)

    po = po.permute(1, 0, 2, 3).contiguous()
    plse = plse.permute(1, 0, 2).contiguous()

    ml = plse.max(dim=0).values
    fi = torch.isfinite(ml); sm = torch.where(fi, ml, torch.zeros_like(ml))
    w = torch.exp2(plse - sm.unsqueeze(0)); ws = w.sum(dim=0)
    c = (w.unsqueeze(-1) * po).sum(dim=0)
    ref_o = torch.zeros_like(c); v = ws > 0
    ref_o[v] = c[v] / ws[v].unsqueeze(-1)
    ref_lse = torch.full_like(ml, -float("inf"))
    ref_lse[v] = sm[v] + torch.log2(ws[v])
    ref_o = ref_o.to(torch.bfloat16)

    # --- Test kernel ---
    spec = importlib.util.spec_from_file_location("kv2", "/root/kernel_v2_cutedsl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    test_o, test_lse = mod.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

    # Compare
    o_err = (test_o.float() - ref_o.float()).abs().max().item()
    l_err = (test_lse - ref_lse).abs().max().item()

    def cos_diff(a, b):
        a, b = a.double(), b.double()
        d = (a*a + b*b).sum().item()
        return abs(1 - 2*(a*b).sum().item()/d) if d > 1e-12 else 0.0

    o_cos = cos_diff(test_o.float(), ref_o.float())
    l_cos = cos_diff(test_lse, ref_lse)

    print(f"\nOutput max_abs_err: {o_err:.2e}, cos_diff: {o_cos:.2e}")
    print(f"LSE    max_abs_err: {l_err:.2e}, cos_diff: {l_cos:.2e}")

    passed = o_err < 1e-1 and l_err < 1e-1
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")

    # Also test with num_tokens=8
    if passed:
        print("\n--- Testing num_tokens=8 ---")
        torch.manual_seed(123)
        nt8 = 8
        q_nope8 = torch.randn(nt8, h_q, ckv_dim, dtype=torch.bfloat16, device=device) / 10.0
        q_pe8 = torch.randn(nt8, h_q, kpe_dim, dtype=torch.bfloat16, device=device) / 10.0
        si8 = torch.randint(0, total_kv, (nt8, topk), dtype=torch.int32, device=device)
        inv8 = torch.rand(nt8, topk, device=device) < 0.05
        si8[inv8] = -1

        test_o8, test_lse8 = mod.run(q_nope8, q_pe8, ckv_cache, kpe_cache, si8, sm_scale)
        print(f"Output shape: {test_o8.shape}, dtype: {test_o8.dtype}")
        print(f"LSE shape: {test_lse8.shape}, dtype: {test_lse8.dtype}")
        assert test_o8.shape == (nt8, h_q, ckv_dim), f"Bad shape: {test_o8.shape}"
        assert test_lse8.shape == (nt8, h_q), f"Bad shape: {test_lse8.shape}"
        print("Shape check: PASS")

    return {"passed": passed, "output_max_abs": o_err, "lse_max_abs": l_err}


@app.local_entrypoint()
def main():
    result = test_correctness.remote()
    print(f"\nFinal result: {result}")
