"""
FlashInfer Competition — Benchmark & Correctness Runner on Modal B200.

Packs a kernel solution, uploads the competition dataset to a Modal volume,
runs the official flashinfer-bench evaluation, and reports correctness + latency.

Usage:
    # First time — upload dataset to Modal volume:
    modal run scripts/bench.py --setup-volume

    # Run benchmark for DSA sparse attention kernel:
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention

    # Run benchmark for DSA indexer kernel:
    modal run scripts/bench.py --track dsa_indexer --solution-dir solution/dsa_indexer

    # Quick correctness-only check (1 iteration, no warmup):
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --correctness-only

    # Specify language (default: triton):
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --lang cuda

    # Entry point format: "<file>::<function>" (e.g. "kernel.py::kernel" or "kernel.cu::kernel"):
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point "kernel.py::kernel"
"""

import json
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Track definitions — maps short names to official definition names
# ---------------------------------------------------------------------------
TRACK_DEFS = {
    "dsa_attention": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    "dsa_indexer": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
}

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App("flashinfer-competition-bench")

trace_volume = modal.Volume.from_name("flashinfer-competition-trace", create_if_missing=True)
TRACE_MOUNT = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "flashinfer-bench",
        "nvidia-cutlass-dsl",
        "torch",
        "triton",
        "numpy",
        "safetensors",
    )
)


# ---------------------------------------------------------------------------
# Remote benchmark function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_MOUNT: trace_volume},
)
def run_benchmark_remote(
    solution_json: str,
    correctness_only: bool = False,
) -> dict:
    """Run flashinfer-bench evaluation on Modal B200."""
    import torch
    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Load solution
    solution = Solution.model_validate_json(solution_json)
    print(f"Solution: {solution.name} | Definition: {solution.definition}")

    # Load trace set from volume
    trace_set = TraceSet.from_path(TRACE_MOUNT)

    if solution.definition not in trace_set.definitions:
        available = list(trace_set.definitions.keys())
        return {
            "success": False,
            "error": f"Definition '{solution.definition}' not found. Available: {available}",
        }

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        return {
            "success": False,
            "error": f"No workloads found for '{solution.definition}'",
        }

    print(f"Found {len(workloads)} workloads")

    # Configure benchmark
    if correctness_only:
        config = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)
    else:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    # Build scoped trace set
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    # Run
    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    # Collect results
    traces = result_trace_set.traces.get(definition.name, [])
    results = []
    for trace in traces:
        if not trace.evaluation:
            continue
        axes = trace.workload.axes if trace.workload else {}
        entry = {
            "workload_uuid": trace.workload.uuid if trace.workload else "unknown",
            "num_tokens": axes.get("num_tokens"),
            "num_pages": axes.get("num_pages"),
            "batch_size": axes.get("batch_size"),
            "max_num_pages": axes.get("max_num_pages"),
            "status": trace.evaluation.status.value,
        }
        if trace.evaluation.log:
            entry["log"] = trace.evaluation.log
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["ref_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            entry["max_abs_err"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_err"] = trace.evaluation.correctness.max_relative_error
        results.append(entry)

    return {"success": True, "results": results}


# ---------------------------------------------------------------------------
# Volume setup — upload competition dataset
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={TRACE_MOUNT: trace_volume}, timeout=1800)
def setup_trace_volume(dataset_tar: bytes):
    """Unpack competition dataset into the Modal volume."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        f.write(dataset_tar)
        tar_path = f.name

    # Extract directly into the mount point
    subprocess.run(
        ["tar", "xzf", tar_path, "-C", TRACE_MOUNT, "--strip-components=1"],
        check=True,
    )

    # Show what we have
    import os
    for root, dirs, files in os.walk(TRACE_MOUNT):
        depth = root.replace(TRACE_MOUNT, "").count(os.sep)
        indent = " " * 2 * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth < 3:
            for f in files[:10]:
                print(f"{indent}  {f}")
            if len(files) > 10:
                print(f"{indent}  ... and {len(files) - 10} more")

    trace_volume.commit()
    print("\nDataset uploaded to volume successfully.")


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def pack_solution_local(
    solution_dir: str,
    track: str,
    lang: str = "triton",
    entry_point: str = "kernel.py::kernel",
    name: str = "dev-kernel",
    author: str = "team",
) -> str:
    """Pack solution files into a Solution JSON string."""
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    definition = TRACK_DEFS[track]

    spec = BuildSpec(
        language=lang,
        target_hardware=["cuda"],
        entry_point=entry_point,
    )

    solution = pack_solution_from_files(
        path=solution_dir,
        spec=spec,
        name=name,
        definition=definition,
        author=author,
    )

    return solution.model_dump_json(indent=2)


def print_results(results: list[dict]):
    """Pretty-print benchmark results."""
    if not results:
        print("No results returned!")
        return

    # Group by status
    passed = [r for r in results if r["status"] == "PASSED"]
    failed = [r for r in results if r["status"] != "PASSED"]

    print(f"\n{'='*80}")
    print(f"  RESULTS: {len(passed)}/{len(results)} workloads passed")
    print(f"{'='*80}\n")

    # Header
    print(f"  {'tokens':>6}  {'pages':>6}  {'status':>8}  {'latency':>10}  {'ref_lat':>10}  {'speedup':>8}  {'abs_err':>10}  {'rel_err':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

    for r in sorted(results, key=lambda x: (x.get("num_tokens") or x.get("batch_size") or 0, x.get("status", ""))):
        tokens = r.get("num_tokens") or r.get("batch_size") or "?"
        pages = r.get("num_pages") or r.get("max_num_pages") or "?"
        status = r.get("status", "?")
        lat = f"{r['latency_ms']:.3f}ms" if r.get("latency_ms") is not None else "-"
        ref_lat = f"{r['ref_latency_ms']:.3f}ms" if r.get("ref_latency_ms") is not None else "-"
        speedup = f"{r['speedup']:.2f}x" if r.get("speedup") is not None else "-"
        abs_err = f"{r['max_abs_err']:.2e}" if r.get("max_abs_err") is not None else "-"
        rel_err = f"{r['max_rel_err']:.2e}" if r.get("max_rel_err") is not None else "-"
        marker = "PASS" if status == "PASSED" else "FAIL"
        print(f"  {tokens:>6}  {pages:>6}  {marker:>8}  {lat:>10}  {ref_lat:>10}  {speedup:>8}  {abs_err:>10}  {rel_err:>10}")

    if results and results[0].get("speedup") is not None:
        speedups = [r["speedup"] for r in results if r.get("speedup") is not None]
        if speedups:
            avg = sum(speedups) / len(speedups)
            print(f"\n  Average speedup: {avg:.2f}x (arithmetic mean, competition scoring)")

    if failed:
        print(f"\n  FAILED workloads ({len(failed)}):")
        for r in failed:
            print(f"    - tokens={r.get('num_tokens')} uuid={r.get('workload_uuid', '?')[:12]}... status={r['status']}")
            if r.get("log"):
                log_lines = r["log"].strip().splitlines()
                print(f"      log: {len(log_lines)} lines")
                if len(log_lines) <= 80:
                    for line in log_lines:
                        print(f"        {line}")
                else:
                    print("        --- first 20 lines ---")
                    for line in log_lines[:20]:
                        print(f"        {line}")
                    print("        --- last 60 lines ---")
                    for line in log_lines[-60:]:
                        print(f"        {line}")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    track: str = "dsa_attention",
    solution_dir: str = "solution/dsa_attention",
    lang: str = "triton",
    entry_point: str = "kernel.py::kernel",
    name: str = "dev-kernel",
    author: str = "team",
    correctness_only: bool = False,
    setup_volume: bool = False,
):
    """Benchmark a kernel solution on Modal B200.

    Args:
        track: Track short name (dsa_attention | dsa_indexer)
        solution_dir: Path to solution source files
        lang: Language (triton | cuda)
        entry_point: Kernel entry point function name
        name: Solution name for tracking
        correctness_only: Skip perf measurement, just check correctness
        setup_volume: Upload competition dataset to Modal volume and exit
    """
    if track not in TRACK_DEFS:
        print(f"Unknown track '{track}'. Available: {list(TRACK_DEFS.keys())}")
        sys.exit(1)

    # --- Volume setup mode ---
    if setup_volume:
        import subprocess
        import tempfile

        dataset_path = PROJECT_ROOT / "competition-dataset"
        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}")
            print("Run: git submodule update --init --recursive")
            print("Then: cd competition-dataset && git lfs pull")
            sys.exit(1)

        # Check if LFS files are pulled
        sample_blob = list((dataset_path / "blob").rglob("*.safetensors"))
        if sample_blob:
            size = sample_blob[0].stat().st_size
            if size < 200:
                print("LFS pointers detected — safetensors not pulled yet.")
                print("Run: cd competition-dataset && git lfs install && git lfs pull")
                sys.exit(1)

        print(f"Packing dataset from {dataset_path}...")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tar_path = f.name

        subprocess.run(
            ["tar", "czf", tar_path, "-C", str(dataset_path), "."],
            check=True,
        )
        tar_size = Path(tar_path).stat().st_size
        print(f"Dataset archive: {tar_size / 1024 / 1024:.1f} MB")

        print("Uploading to Modal volume...")
        tar_bytes = Path(tar_path).read_bytes()
        setup_trace_volume.remote(tar_bytes)
        print("Done! Volume 'flashinfer-competition-trace' is ready.")
        return

    # --- Benchmark mode ---
    solution_path = Path(solution_dir)
    if not solution_path.is_absolute():
        solution_path = PROJECT_ROOT / solution_path

    if not solution_path.exists():
        print(f"Solution directory not found: {solution_path}")
        print(f"Create it with your kernel files first.")
        sys.exit(1)

    print(f"Track: {track} ({TRACK_DEFS[track]})")
    print(f"Solution: {solution_path}")
    print(f"Language: {lang}")
    print(f"Mode: {'correctness-only' if correctness_only else 'full benchmark'}")

    # Pack solution
    print("\nPacking solution...")
    try:
        solution_json = pack_solution_local(
            solution_dir=str(solution_path),
            track=track,
            lang=lang,
            entry_point=entry_point,
            name=name,
            author=author,
        )
    except Exception as e:
        print(f"Failed to pack solution: {e}")
        sys.exit(1)

    # Save packed solution for reference
    out_path = PROJECT_ROOT / "solution.json"
    out_path.write_text(solution_json)
    print(f"Packed solution saved to {out_path}")

    # Run on Modal
    print("\nLaunching benchmark on Modal B200...")
    result = run_benchmark_remote.remote(
        solution_json=solution_json,
        correctness_only=correctness_only,
    )

    if not result["success"]:
        print(f"\nBenchmark failed: {result['error']}")
        sys.exit(1)

    print_results(result["results"])
