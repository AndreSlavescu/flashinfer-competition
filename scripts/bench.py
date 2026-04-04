"""
word2kernel benchmark and correctness runner on Modal B200.

Packs a kernel solution, uploads the competition dataset to a Modal volume,
runs the official flashinfer-bench evaluation, or executes a fast synthetic
correctness sweep, and reports results.

Usage:
    # First time — upload dataset to Modal volume:
    modal run scripts/bench.py --setup-volume

    # Run benchmark for DSA sparse attention kernel:
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention

    # Run benchmark for DSA indexer kernel (placeholder scaffold by default):
    modal run scripts/bench.py --track dsa_indexer --solution-dir solution/dsa_indexer

    # Quick correctness-only check (1 iteration, no warmup):
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --correctness-only

    # Recommended fast synthetic path for edit-run iteration:
    .venv/bin/modal deploy scripts/bench_synthetic_service.py
    .venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention --entry-point kernel.py::kernel

    # Fallback synthetic correctness sweep (slower; uses ephemeral modal run):
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --synthetic

    # Specify language (default: python for CuTeDSL .py kernels):
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --lang python
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --lang cuda

    # Entry point format: "<file>::<function>" (e.g. "kernel.py::kernel" or "kernel.cu::kernel"):
    modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point "kernel.py::kernel"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap_runtime import require_dependencies

require_dependencies(
    {"modal": "modal"},
    entrypoint="scripts/bench.py",
)

import modal

DEFAULT_BENCH_LANGUAGE = "python"
SUPPORTED_LANGUAGES = {"python", "triton", "cuda"}
VALID_SOURCE_EXTENSIONS = {".py", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"}

# ---------------------------------------------------------------------------
# Track definitions — maps short names to official definition names
# ---------------------------------------------------------------------------
TRACK_DEFS = {
    "dsa_attention": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    "dsa_indexer": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
}

SYNTHETIC_NUM_TOKENS = (1, 2, 6, 7, 8)
SYNTHETIC_NUM_PAGES = 8462
SYNTHETIC_PAGE_SIZE = 64
SYNTHETIC_TOPK = 2048
SYNTHETIC_NUM_QO_HEADS = 16
SYNTHETIC_CKV_DIM = 512
SYNTHETIC_KPE_DIM = 64
SYNTHETIC_SM_SCALE = 0.1352337788608801

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
        "pygpubench",
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
    solution_json: str = "",
    correctness_only: bool = False,
    raw_files: Optional[dict] = None,
    pack_args: Optional[dict] = None,
) -> dict:
    """Run flashinfer-bench evaluation on Modal B200.

    Can accept either a pre-packed solution_json or raw_files + pack_args
    to pack the solution remotely (avoids local flashinfer_bench dependency).
    """
    import base64
    import tempfile

    import torch
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, Solution, TraceSet
    from flashinfer_bench.data import SourceFile

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # If raw files provided, pack solution remotely
    if raw_files and pack_args:
        with tempfile.TemporaryDirectory() as tmpdir:
            for fname, b64content in raw_files.items():
                target = Path(tmpdir) / fname
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(b64content))
            spec = BuildSpec(
                language=pack_args["lang"],
                target_hardware=["cuda"],
                entry_point=pack_args["entry_point"],
            )
            sources = [
                SourceFile(
                    path=rel_path,
                    content=(Path(tmpdir) / rel_path).read_text(encoding="utf-8"),
                )
                for rel_path in sorted(raw_files)
                if Path(rel_path).suffix.lower() in VALID_SOURCE_EXTENSIONS
            ]
            solution = Solution(
                name=pack_args["name"],
                definition=pack_args["definition"],
                author=pack_args["author"],
                spec=spec,
                sources=sources,
            )
            solution_json = solution.model_dump_json(indent=2)

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


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
)
def run_synthetic_remote(
    raw_files: dict,
    entry_file: str,
    entry_func: str,
    reference_b64: str,
) -> dict:
    """Run a fast synthetic correctness sweep on Modal B200."""
    import base64
    import importlib.util
    import sys
    import tempfile
    import traceback

    import torch

    def make_failure_results(message: str) -> list[dict]:
        return [
            {
                "workload_uuid": f"synthetic-{num_tokens}",
                "num_tokens": num_tokens,
                "num_pages": SYNTHETIC_NUM_PAGES,
                "status": "FAILED",
                "log": message,
            }
            for num_tokens in SYNTHETIC_NUM_TOKENS
        ]

    def load_module(module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec for {module_path.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    entry_file = Path(entry_file).as_posix()
    if entry_file not in raw_files:
        available = ", ".join(sorted(raw_files))
        return {
            "success": True,
            "results": make_failure_results(
                f"Entry file '{entry_file}' not found in solution files. Available: {available}"
            ),
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for fname, b64content in raw_files.items():
            target = tmpdir_path / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(b64content))
        reference_path = tmpdir_path / "reference.py"
        reference_path.write_bytes(base64.b64decode(reference_b64))
        sys.path.insert(0, tmpdir)

        try:
            reference_mod = load_module("word2kernel_reference", reference_path)
            solution_mod = load_module("word2kernel_solution", tmpdir_path / entry_file)
        except Exception:
            if sys.path and sys.path[0] == tmpdir:
                sys.path.pop(0)
            return {
                "success": True,
                "results": make_failure_results(traceback.format_exc()),
            }

        solution_run = getattr(solution_mod, "run", None)
        solution_entry = getattr(solution_mod, entry_func, None)
        use_run = callable(solution_run)

        if not use_run and not callable(solution_entry):
            if sys.path and sys.path[0] == tmpdir:
                sys.path.pop(0)
            return {
                "success": True,
                "results": make_failure_results(
                    f"Solution module '{entry_file}' is missing callable run() and '{entry_func}()'."
                ),
            }

        results = []
        total_kv = SYNTHETIC_NUM_PAGES * SYNTHETIC_PAGE_SIZE

        for num_tokens in SYNTHETIC_NUM_TOKENS:
            seed = 42 + num_tokens
            try:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                device = "cuda"
                q_nope = torch.randn(
                    num_tokens,
                    SYNTHETIC_NUM_QO_HEADS,
                    SYNTHETIC_CKV_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                ) / 10.0
                q_pe = torch.randn(
                    num_tokens,
                    SYNTHETIC_NUM_QO_HEADS,
                    SYNTHETIC_KPE_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                ) / 10.0
                ckv_cache = torch.randn(
                    SYNTHETIC_NUM_PAGES,
                    SYNTHETIC_PAGE_SIZE,
                    SYNTHETIC_CKV_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                ) / 10.0
                kpe_cache = torch.randn(
                    SYNTHETIC_NUM_PAGES,
                    SYNTHETIC_PAGE_SIZE,
                    SYNTHETIC_KPE_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                ) / 10.0
                sparse_indices = torch.randint(
                    0,
                    total_kv,
                    (num_tokens, SYNTHETIC_TOPK),
                    dtype=torch.int32,
                    device=device,
                )
                invalid_mask = torch.rand(num_tokens, SYNTHETIC_TOPK, device=device) < 0.05
                sparse_indices[invalid_mask] = -1
                sm_scale = torch.tensor(
                    SYNTHETIC_SM_SCALE,
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    ref_output, ref_lse = reference_mod.run(
                        q_nope,
                        q_pe,
                        ckv_cache,
                        kpe_cache,
                        sparse_indices,
                        sm_scale,
                    )
                    torch.cuda.synchronize()

                    if use_run:
                        test_output, test_lse = solution_run(
                            q_nope,
                            q_pe,
                            ckv_cache,
                            kpe_cache,
                            sparse_indices,
                            sm_scale,
                        )
                    else:
                        test_output = torch.empty_like(ref_output)
                        test_lse = torch.empty_like(ref_lse)
                        solution_entry(
                            q_nope,
                            q_pe,
                            ckv_cache,
                            kpe_cache,
                            sparse_indices,
                            sm_scale,
                            test_output,
                            test_lse,
                        )
                    torch.cuda.synchronize()

                output_abs_err = (test_output.float() - ref_output.float()).abs().max().item()
                lse_abs_err = (test_lse - ref_lse).abs().max().item()
                max_abs_err = max(output_abs_err, lse_abs_err)
                passed = output_abs_err < 1e-2 and lse_abs_err < 1e-3

                results.append(
                    {
                        "workload_uuid": f"synthetic-{num_tokens}",
                        "num_tokens": num_tokens,
                        "num_pages": SYNTHETIC_NUM_PAGES,
                        "status": "PASSED" if passed else "FAILED",
                        "output_abs_err": output_abs_err,
                        "lse_abs_err": lse_abs_err,
                        "max_abs_err": max_abs_err,
                    }
                )
            except Exception:
                results.append(
                    {
                        "workload_uuid": f"synthetic-{num_tokens}",
                        "num_tokens": num_tokens,
                        "num_pages": SYNTHETIC_NUM_PAGES,
                        "status": "FAILED",
                        "log": traceback.format_exc(),
                    }
                )

        if sys.path and sys.path[0] == tmpdir:
            sys.path.pop(0)

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
def collect_solution_files(solution_dir: str) -> dict:
    """Read supported source files from solution_dir into {relative_path: base64}."""
    import base64

    files: dict[str, str] = {}
    solution_path = Path(solution_dir)
    for path in sorted(solution_path.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in VALID_SOURCE_EXTENSIONS:
            continue
        rel_path = path.relative_to(solution_path).as_posix()
        if Path(rel_path).name.startswith("__"):
            continue
        files[rel_path] = base64.b64encode(path.read_bytes()).decode()
    return files


def parse_entry_point(entry_point: str) -> tuple[str, str]:
    """Parse an entry point string of the form '<file>::<function>'."""
    if "::" not in entry_point:
        raise ValueError(
            f"Invalid entry point '{entry_point}'. Expected format '<file>::<function>'."
        )
    entry_file, entry_func = entry_point.split("::", 1)
    entry_path = Path(entry_file)
    entry_file = entry_path.as_posix()
    if (
        not entry_file
        or entry_file.startswith("/")
        or ".." in entry_path.parts
        or not entry_func
    ):
        raise ValueError(
            f"Invalid entry point '{entry_point}'. Expected format '<file>::<function>'."
        )
    return entry_file, entry_func


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


def print_synthetic_results(results: list[dict]):
    """Pretty-print synthetic correctness sweep results."""
    if not results:
        print("No synthetic results returned!")
        return

    passed = [r for r in results if r["status"] == "PASSED"]
    failed = [r for r in results if r["status"] != "PASSED"]

    print(f"\n{'='*80}")
    print(f"  SYNTHETIC RESULTS: {len(passed)}/{len(results)} cases passed")
    print(f"{'='*80}\n")

    print(
        f"  {'tokens':>6}  {'pages':>6}  {'status':>8}  "
        f"{'out_abs':>12}  {'lse_abs':>12}  {'max_abs':>12}"
    )
    print(
        f"  {'-'*6}  {'-'*6}  {'-'*8}  "
        f"{'-'*12}  {'-'*12}  {'-'*12}"
    )

    for r in sorted(results, key=lambda x: x.get("num_tokens", 0)):
        tokens = r.get("num_tokens", "?")
        pages = r.get("num_pages", "?")
        status = "PASS" if r.get("status") == "PASSED" else "FAIL"
        out_abs = (
            f"{r['output_abs_err']:.2e}" if r.get("output_abs_err") is not None else "-"
        )
        lse_abs = f"{r['lse_abs_err']:.2e}" if r.get("lse_abs_err") is not None else "-"
        max_abs = f"{r['max_abs_err']:.2e}" if r.get("max_abs_err") is not None else "-"
        print(
            f"  {tokens:>6}  {pages:>6}  {status:>8}  "
            f"{out_abs:>12}  {lse_abs:>12}  {max_abs:>12}"
        )

    if failed:
        print(f"\n  FAILED synthetic cases ({len(failed)}):")
        for r in failed:
            print(f"    - num_tokens={r.get('num_tokens')} status={r['status']}")
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
    lang: str = DEFAULT_BENCH_LANGUAGE,
    entry_point: str = "kernel.py::kernel",
    name: str = "dev-kernel",
    author: str = "team",
    correctness_only: bool = False,
    synthetic: bool = False,
    setup_volume: bool = False,
):
    """Benchmark a kernel solution on Modal B200.

    Args:
        track: Track short name (dsa_attention | dsa_indexer)
        solution_dir: Path to solution source files
        lang: Language (python | triton | cuda)
        entry_point: Kernel entry point function name
        name: Solution name for tracking
        correctness_only: Skip perf measurement, just check correctness
        synthetic: Run a deterministic synthetic correctness sweep
        setup_volume: Upload competition dataset to Modal volume and exit
    """
    if track not in TRACK_DEFS:
        print(f"Unknown track '{track}'. Available: {list(TRACK_DEFS.keys())}")
        sys.exit(1)

    if lang not in SUPPORTED_LANGUAGES:
        print(f"Unknown language '{lang}'. Available: {sorted(SUPPORTED_LANGUAGES)}")
        sys.exit(1)

    if synthetic and setup_volume:
        print("--synthetic and --setup-volume are mutually exclusive")
        sys.exit(1)

    if synthetic and track != "dsa_attention":
        print("Synthetic mode currently only supports track='dsa_attention'")
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
            ["tar", "czf", tar_path, "--no-mac-metadata", "--exclude", "._*", "--exclude", ".DS_Store", "-C", str(dataset_path), "."],
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
    if track == "dsa_indexer":
        print("Note: dsa_indexer is scaffold-only in this trimmed repo until you add a real kernel.")
    if synthetic:
        mode = "synthetic correctness sweep"
    else:
        mode = "correctness-only" if correctness_only else "full benchmark"
    print(f"Mode: {mode}")

    # Collect solution files and pack remotely on Modal
    print("\nCollecting solution files...")
    raw_files = collect_solution_files(str(solution_path))
    if not raw_files:
        print(f"No supported source files found in {solution_path}")
        if track == "dsa_indexer":
            print("This scaffold is intentionally placeholder-only; add kernel.py before benchmarking.")
        else:
            print("Add a kernel entry file such as kernel.py or kernel_0.py before benchmarking.")
        sys.exit(1)
    print(f"  Files: {', '.join(raw_files.keys())}")

    if synthetic:
        import base64

        try:
            entry_file, entry_func = parse_entry_point(entry_point)
        except ValueError as exc:
            print(exc)
            sys.exit(1)

        if not entry_file.endswith(".py"):
            print(
                f"Synthetic mode requires a Python entry point file, got '{entry_file}'."
            )
            sys.exit(1)

        reference_path = (
            PROJECT_ROOT / "references" / f"{TRACK_DEFS[track]}.py"
        )
        if not reference_path.exists():
            print(f"Reference file not found: {reference_path}")
            sys.exit(1)

        print("\nLaunching synthetic correctness sweep on Modal B200...")
        print(
            "Tip: for faster edit-run iteration, deploy scripts/bench_synthetic_service.py "
            "once and use scripts/bench_synthetic.py."
        )
        result = run_synthetic_remote.remote(
            raw_files=raw_files,
            entry_file=entry_file,
            entry_func=entry_func,
            reference_b64=base64.b64encode(reference_path.read_bytes()).decode(),
        )

        if not result["success"]:
            print(f"\nSynthetic benchmark failed: {result['error']}")
            sys.exit(1)

        print_synthetic_results(result["results"])
        return

    # Auto-select best_kernel.py for full perf benchmark
    if not correctness_only and "best_kernel.py" in raw_files and entry_point == "kernel.py::kernel":
        entry_point = "best_kernel.py::kernel"
        print(f"  Using best_kernel.py for performance benchmark")

    pack_args = {
        "lang": lang,
        "entry_point": entry_point,
        "name": name,
        "definition": TRACK_DEFS[track],
        "author": author,
    }

    # Run on Modal (packing + benchmark)
    print("\nLaunching benchmark on Modal B200...")
    result = run_benchmark_remote.remote(
        raw_files=raw_files,
        pack_args=pack_args,
        correctness_only=correctness_only,
    )

    if not result["success"]:
        print(f"\nBenchmark failed: {result['error']}")
        sys.exit(1)

    print_results(result["results"])
