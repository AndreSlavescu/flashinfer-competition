"""
Bridge between the DSA sparse attention kernel and pygpubench's isolated
benchmarking framework.

pygpubench adds defenses that complement tools/harness/validate.py:
  - C++ compiled extension (no Python-level monkey-patching)
  - Isolated subprocess execution with AES-GCM encrypted results
  - L2 cache clearing between runs (hardware-level, not just torch zero)
  - Canary-value insertion (1% of memory contains wrong data)
  - Randomized result verification order
  - seccomp/landlock sandboxing (Linux only)

Usage on Modal B200:
    python -m tools.harness.pygpubench_bridge

Usage from Python:
    from tools.harness.pygpubench_bridge import bench_dsa_isolated
    result = bench_dsa_isolated(repeats=100, seed=42)
"""

import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

# ---------------------------------------------------------------------------
# Constants (must match solution/dsa_attention/kernel.py)
# ---------------------------------------------------------------------------
NUM_Q_HEADS = 16
CKV_DIM = 512
KPE_DIM = 64
TOPK = 2048
PAGE_SIZE = 64


# ---------------------------------------------------------------------------
# Test case generator — pygpubench calls this with seed + user kwargs
# ---------------------------------------------------------------------------
def generate_dsa_test_case(
    *,
    seed: int,
    num_tokens: int = 1,
    num_pages: int = 8462,
    sm_scale: float = 0.135,
) -> tuple[tuple, tuple]:
    """
    Generate a test case for the DSA decode kernel.

    pygpubench validates the first positional arg (the output buffer) against
    the expected tensor. We pack output(T,H,D) and lse(T,H,1) into a single
    FP32 tensor along the last dim so both are validated in one comparison.

    Returns:
        (kernel_args, expected_result) where:
        - kernel_args = (packed_buf, q_nope, q_pe, ckv_cache,
                         kpe_cache, sparse_indices, sm_scale_tensor)
        - expected_result = (packed_ref, atol, rtol)
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    total_kv = num_pages * PAGE_SIZE
    device = "cuda"

    q_nope = torch.randn(
        num_tokens, NUM_Q_HEADS, CKV_DIM,
        dtype=torch.bfloat16, device=device, generator=gen,
    ) / 10.0
    q_pe = torch.randn(
        num_tokens, NUM_Q_HEADS, KPE_DIM,
        dtype=torch.bfloat16, device=device, generator=gen,
    ) / 10.0
    ckv_cache = torch.randn(
        num_pages, PAGE_SIZE, CKV_DIM,
        dtype=torch.bfloat16, device=device, generator=gen,
    ) / 10.0
    kpe_cache = torch.randn(
        num_pages, PAGE_SIZE, KPE_DIM,
        dtype=torch.bfloat16, device=device, generator=gen,
    ) / 10.0

    sparse_indices = torch.randint(
        0, total_kv, (num_tokens, TOPK),
        dtype=torch.int32, device=device, generator=gen,
    )
    # Mark ~5% as invalid to test masking
    invalid_mask = torch.rand(
        num_tokens, TOPK, device=device, generator=gen,
    ) < 0.05
    sparse_indices[invalid_mask] = -1

    sm_scale_t = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    # --- Compute reference output ---
    sys.path.insert(0, str(_PROJECT_ROOT / "references"))
    import importlib
    ref_mod = importlib.import_module(
        "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
    )
    with torch.no_grad():
        ref_output, ref_lse = ref_mod.run(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        )

    # Pack into single FP32 tensor: (T, H, D+1) = [output.float() | lse.float()]
    packed_ref = torch.cat(
        [ref_output.float(), ref_lse.unsqueeze(-1).float()], dim=-1
    ).contiguous()

    # Pre-allocate packed output buffer (pygpubench validates first arg)
    packed_buf = torch.empty_like(packed_ref)

    kernel_args = (
        packed_buf,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_t,
    )
    expected = (packed_ref, 1e-2, 1e-2)

    return kernel_args, expected


# ---------------------------------------------------------------------------
# Kernel wrapper — pygpubench imports this by qualified name and calls it
# with the args from the test generator
# ---------------------------------------------------------------------------
def dsa_kernel_entry(
    packed_buf,
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_t,
):
    """
    Kernel entry that pygpubench calls per iteration.

    Writes packed [output.float() | lse.float()] into packed_buf (first arg)
    which pygpubench compares against the expected tensor from the test generator.
    """
    from solution.dsa_attention.kernel import run

    with torch.no_grad():
        out, lse_out = run(
            q_nope, q_pe, ckv_cache, kpe_cache,
            sparse_indices, sm_scale_t.item(),
        )

    # Write into the pre-allocated packed buffer (pygpubench validates this)
    packed_buf[:, :, :CKV_DIM] = out.float()
    packed_buf[:, :, CKV_DIM:] = lse_out.unsqueeze(-1).float()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def bench_dsa_isolated(
    repeats: int = 100,
    seed: int = 42,
    num_tokens: int = 1,
    num_pages: int = 8462,
    sm_scale: float = 0.135,
    discard: bool = True,
    timeout: int = 300,
):
    """
    Run the DSA kernel through pygpubench's isolated benchmark.

    Returns a pygpubench.BenchmarkResult with timings and error counts.
    """
    import pygpubench

    test_args = {
        "num_tokens": num_tokens,
        "num_pages": num_pages,
        "sm_scale": sm_scale,
    }

    result = pygpubench.do_bench_isolated(
        qualname="tools.harness.pygpubench_bridge.dsa_kernel_entry",
        test_generator=generate_dsa_test_case,
        test_args=test_args,
        repeats=repeats,
        seed=seed,
        discard=discard,
        # Modal doesn't support landlock/mseal (needs Linux kernel features)
        landlock=False,
        mseal=False,
        timeout=timeout,
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark DSA kernel with pygpubench isolation"
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--num-pages", type=int, default=8462)
    parser.add_argument("--sm-scale", type=float, default=0.135)
    parser.add_argument("--no-discard", action="store_true",
                        help="Disable L2 cache clearing between runs")
    args = parser.parse_args()

    import pygpubench

    print("=== pygpubench isolated DSA benchmark ===")
    result = bench_dsa_isolated(
        repeats=args.repeats,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_pages=args.num_pages,
        sm_scale=args.sm_scale,
        discard=not args.no_discard,
    )

    stats = pygpubench.basic_stats(result.time_us)
    status = "PASS" if result.success else "FAIL"
    errors = f" (errors: {result.errors})" if result.errors else ""
    print(f"[{status}]{errors} {stats}")
    if result.event_overhead_us is not None:
        print(f"  Event overhead: {result.event_overhead_us:.2f} us")
