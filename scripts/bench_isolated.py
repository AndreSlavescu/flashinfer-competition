"""
Isolated kernel benchmark on Modal B200 using pygpubench.

Runs the DSA kernel through pygpubench's subprocess-isolated harness with:
  - L2 cache clearing between runs
  - Canary-value memory corruption detection
  - Encrypted result transfer
  - Deterministic CUDA context

Usage:
    # Quick correctness check (10 iterations):
    modal run scripts/bench_isolated.py --repeats 10

    # Full benchmark (100 iterations):
    modal run scripts/bench_isolated.py --repeats 100

    # With specific seed:
    modal run scripts/bench_isolated.py --repeats 50 --seed 123
"""

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("flashinfer-pygpubench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nvidia-cutlass-dsl",
        "torch",
        "numpy",
        "pygpubench",
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
    mounts=[
        modal.Mount.from_local_dir(
            str(PROJECT_ROOT / "solution"),
            remote_path="/root/project/solution",
        ),
        modal.Mount.from_local_dir(
            str(PROJECT_ROOT / "tools"),
            remote_path="/root/project/tools",
        ),
        modal.Mount.from_local_dir(
            str(PROJECT_ROOT / "references"),
            remote_path="/root/project/references",
        ),
        modal.Mount.from_local_dir(
            str(PROJECT_ROOT / "tests"),
            remote_path="/root/project/tests",
        ),
    ],
)
def run_isolated_bench(
    repeats: int = 50,
    seed: int = 42,
    num_tokens: int = 1,
    num_pages: int = 8462,
    sm_scale: float = 0.135,
    discard: bool = True,
) -> dict:
    """Run pygpubench isolated benchmark on Modal B200."""
    import sys
    import torch

    sys.path.insert(0, "/root/project")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    import pygpubench
    from tools.harness.pygpubench_bridge import bench_dsa_isolated

    result = bench_dsa_isolated(
        repeats=repeats,
        seed=seed,
        num_tokens=num_tokens,
        num_pages=num_pages,
        sm_scale=sm_scale,
        discard=discard,
    )

    stats = pygpubench.basic_stats(result.time_us)

    output = {
        "success": result.success,
        "errors": result.errors,
        "full": result.full,
        "stats": {
            "mean_us": stats.mean,
            "median_us": stats.median,
            "best_us": stats.best,
            "worst_us": stats.worst,
            "std_us": stats.std,
            "runs": stats.runs,
        },
    }
    if result.event_overhead_us is not None:
        output["event_overhead_us"] = result.event_overhead_us

    status = "PASS" if result.success else "FAIL"
    errors = f" (errors: {result.errors})" if result.errors else ""
    print(f"\n[{status}]{errors} {stats}")

    return output


@app.local_entrypoint()
def main(
    repeats: int = 50,
    seed: int = 42,
    num_tokens: int = 1,
    num_pages: int = 8462,
    sm_scale: float = 0.135,
    no_discard: bool = False,
):
    import json

    result = run_isolated_bench.remote(
        repeats=repeats,
        seed=seed,
        num_tokens=num_tokens,
        num_pages=num_pages,
        sm_scale=sm_scale,
        discard=not no_discard,
    )

    print(json.dumps(result, indent=2))
