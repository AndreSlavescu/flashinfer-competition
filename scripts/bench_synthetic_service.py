"""
Deployed fast synthetic benchmark service for DSA sparse attention.

Usage:
    .venv/bin/modal deploy scripts/bench_synthetic_service.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not os.environ.get("MODAL_TASK_ID"):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from bootstrap_runtime import require_dependencies

    require_dependencies(
        {"modal": "modal"},
        entrypoint="scripts/bench_synthetic_service.py",
    )

import modal

from bench_synthetic_common import (
    APP_NAME,
    FIXTURE_MOUNT,
    FIXTURE_SHARED_CACHE_SEED,
    FIXTURE_VOLUME_NAME,
    FUNCTION_NAME,
    REFERENCE_FILENAME,
    STAGE_VALIDATION_NUM_TOKENS,
    SYNTHETIC_CASE_SEEDS,
    SYNTHETIC_CKV_DIM,
    SYNTHETIC_KPE_DIM,
    SYNTHETIC_NUM_PAGES,
    SYNTHETIC_NUM_QO_HEADS,
    SYNTHETIC_NUM_TOKENS,
    SYNTHETIC_PAGE_SIZE,
    SYNTHETIC_SM_SCALE,
    SYNTHETIC_TOPK,
    TRACK_DEFINITION,
    build_fixture_manifest,
    clear_directory_contents,
    clear_solution_modules,
    compute_fixture_version,
    compute_payload_hash,
    make_synthetic_failure_results,
    parse_entry_point,
    stage_validation_page_ids,
    write_raw_files,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_SOURCE_PATH = SCRIPT_DIR / "bench_synthetic_common.py"
LOCAL_REFERENCE_SOURCE_PATH = SCRIPT_DIR.parent / "references" / f"{TRACK_DEFINITION}.py"
REMOTE_COMMON_PATH = Path("/root/bench_synthetic_common.py")
REMOTE_REFERENCE_PATH = Path("/root") / REFERENCE_FILENAME
REFERENCE_HASH_SOURCE_PATH = (
    LOCAL_REFERENCE_SOURCE_PATH if LOCAL_REFERENCE_SOURCE_PATH.exists() else REMOTE_REFERENCE_PATH
)
REFERENCE_SHA256 = sha256(REFERENCE_HASH_SOURCE_PATH.read_bytes()).hexdigest()
WORK_ROOT = Path("/tmp/word2kernel_synthetic_fast")
SOLUTION_ROOT = WORK_ROOT / "solution"
FIXTURE_ROOT = Path(FIXTURE_MOUNT)
FIXTURE_VOLUME = modal.Volume.from_name(FIXTURE_VOLUME_NAME, create_if_missing=True)
ENABLE_MEMORY_SNAPSHOT = (
    os.environ.get("WORD2KERNEL_ENABLE_MEMORY_SNAPSHOT") == "1"
    or os.environ.get("FLASHMLA_ENABLE_MEMORY_SNAPSHOT") == "1"
)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "triton",
        "nvidia-cutlass-dsl",
        "numpy",
        "packaging",
        "safetensors",
    )
    .add_local_file(str(COMMON_SOURCE_PATH), remote_path=str(REMOTE_COMMON_PATH))
    .add_local_file(str(LOCAL_REFERENCE_SOURCE_PATH), remote_path=str(REMOTE_REFERENCE_PATH))
)

with image.imports():
    import importlib.util
    import torch
    from safetensors.torch import load_file, save_file

_REFERENCE_MODULE = None
_FIXTURE_CACHE: dict[str, dict] = {}
_SOLUTION_CACHE: dict[str, object] = {
    "payload_hash": None,
    "module": None,
}


def _load_python_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference_module():
    global _REFERENCE_MODULE
    if _REFERENCE_MODULE is None:
        sys.modules.pop("word2kernel_fast_reference", None)
        _REFERENCE_MODULE = _load_python_module(
            "word2kernel_fast_reference",
            REMOTE_REFERENCE_PATH,
        )
    return _REFERENCE_MODULE


def _fixture_dir(version: str) -> Path:
    return FIXTURE_ROOT / TRACK_DEFINITION / version


def _manifest_path(version: str) -> Path:
    return _fixture_dir(version) / "manifest.json"


def _shared_cache_path(version: str) -> Path:
    return _fixture_dir(version) / "shared_cache.safetensors"


def _case_inputs_path(version: str, num_tokens: int) -> Path:
    return _fixture_dir(version) / f"inputs_t{num_tokens}.safetensors"


def _case_refs_path(version: str, num_tokens: int) -> Path:
    return _fixture_dir(version) / f"refs_t{num_tokens}.safetensors"


def _build_shared_cache():
    torch.manual_seed(FIXTURE_SHARED_CACHE_SEED)
    torch.cuda.manual_seed(FIXTURE_SHARED_CACHE_SEED)

    ckv_cache = torch.randn(
        SYNTHETIC_NUM_PAGES,
        SYNTHETIC_PAGE_SIZE,
        SYNTHETIC_CKV_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ) / 10.0
    kpe_cache = torch.randn(
        SYNTHETIC_NUM_PAGES,
        SYNTHETIC_PAGE_SIZE,
        SYNTHETIC_KPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ) / 10.0
    return ckv_cache, kpe_cache


def _build_case(num_tokens: int, ckv_cache, kpe_cache):
    seed = SYNTHETIC_CASE_SEEDS[num_tokens]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    total_kv = SYNTHETIC_NUM_PAGES * SYNTHETIC_PAGE_SIZE
    q_nope = torch.randn(
        num_tokens,
        SYNTHETIC_NUM_QO_HEADS,
        SYNTHETIC_CKV_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ) / 10.0
    q_pe = torch.randn(
        num_tokens,
        SYNTHETIC_NUM_QO_HEADS,
        SYNTHETIC_KPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ) / 10.0
    sparse_indices = torch.randint(
        0,
        total_kv,
        (num_tokens, SYNTHETIC_TOPK),
        dtype=torch.int32,
        device="cuda",
    )
    invalid_mask = torch.rand(num_tokens, SYNTHETIC_TOPK, device="cuda") < 0.05
    sparse_indices[invalid_mask] = -1
    sm_scale = torch.tensor(SYNTHETIC_SM_SCALE, dtype=torch.float32, device="cuda")

    return {
        "num_tokens": num_tokens,
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": sm_scale,
    }


def _build_fixture_bundle():
    reference_mod = _load_reference_module()
    ckv_cache, kpe_cache = _build_shared_cache()
    cases = []
    reference_outputs = []

    with torch.no_grad():
        for num_tokens in SYNTHETIC_NUM_TOKENS:
            case = _build_case(num_tokens, ckv_cache, kpe_cache)
            ref_output, ref_lse = reference_mod.run(
                case["q_nope"],
                case["q_pe"],
                case["ckv_cache"],
                case["kpe_cache"],
                case["sparse_indices"],
                case["sm_scale"],
            )
            cases.append(case)
            reference_outputs.append((ref_output, ref_lse))
        torch.cuda.synchronize()

    return {
        "cases": cases,
        "reference_outputs": reference_outputs,
    }


def _write_fixture_bundle(version: str, manifest: dict, bundle: dict) -> None:
    fixture_dir = _fixture_dir(version)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    clear_directory_contents(fixture_dir)

    shared_case = bundle["cases"][0]
    save_file(
        {
            "ckv_cache": shared_case["ckv_cache"].cpu(),
            "kpe_cache": shared_case["kpe_cache"].cpu(),
        },
        _shared_cache_path(version),
        metadata={"definition": TRACK_DEFINITION},
    )

    for case, (ref_output, ref_lse) in zip(
        bundle["cases"],
        bundle["reference_outputs"],
        strict=True,
    ):
        num_tokens = case["num_tokens"]
        save_file(
            {
                "q_nope": case["q_nope"].cpu(),
                "q_pe": case["q_pe"].cpu(),
                "sparse_indices": case["sparse_indices"].cpu(),
                "sm_scale": case["sm_scale"].cpu(),
            },
            _case_inputs_path(version, num_tokens),
            metadata={"num_tokens": str(num_tokens)},
        )
        save_file(
            {
                "output": ref_output.cpu(),
                "lse": ref_lse.cpu(),
            },
            _case_refs_path(version, num_tokens),
            metadata={"num_tokens": str(num_tokens)},
        )

    _manifest_path(version).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    FIXTURE_VOLUME.commit()


def _load_fixture_bundle(version: str) -> dict:
    shared = load_file(_shared_cache_path(version), device="cuda")
    ckv_cache = shared["ckv_cache"]
    kpe_cache = shared["kpe_cache"]
    cases = []
    reference_outputs = []

    for num_tokens in SYNTHETIC_NUM_TOKENS:
        inputs = load_file(_case_inputs_path(version, num_tokens), device="cuda")
        refs = load_file(_case_refs_path(version, num_tokens), device="cuda")
        cases.append(
            {
                "num_tokens": num_tokens,
                "q_nope": inputs["q_nope"],
                "q_pe": inputs["q_pe"],
                "ckv_cache": ckv_cache,
                "kpe_cache": kpe_cache,
                "sparse_indices": inputs["sparse_indices"],
                "sm_scale": inputs["sm_scale"],
            }
        )
        reference_outputs.append((refs["output"], refs["lse"]))

    return {
        "cases": cases,
        "reference_outputs": reference_outputs,
    }


def _ensure_fixture_bundle(rebuild_fixture: bool) -> tuple[dict, str, str]:
    manifest = build_fixture_manifest(REFERENCE_SHA256)
    version = compute_fixture_version(manifest)

    if rebuild_fixture:
        _FIXTURE_CACHE.pop(version, None)

    if version in _FIXTURE_CACHE:
        return _FIXTURE_CACHE[version], version, "memory"

    FIXTURE_VOLUME.reload()
    manifest_path = _manifest_path(version)
    if rebuild_fixture or not manifest_path.exists():
        bundle = _build_fixture_bundle()
        _write_fixture_bundle(version, manifest, bundle)
        _FIXTURE_CACHE[version] = bundle
        return bundle, version, "rebuilt" if rebuild_fixture else "built"

    disk_manifest = json.loads(manifest_path.read_text())
    if disk_manifest != manifest:
        bundle = _build_fixture_bundle()
        _write_fixture_bundle(version, manifest, bundle)
        _FIXTURE_CACHE[version] = bundle
        return bundle, version, "rebuilt"

    bundle = _load_fixture_bundle(version)
    _FIXTURE_CACHE[version] = bundle
    return bundle, version, "volume"


def _ensure_solution_module(raw_files: dict[str, str], entry_point: str, entry_file: str):
    payload_hash = compute_payload_hash(entry_point, raw_files)
    cached_hash = _SOLUTION_CACHE.get("payload_hash")
    cached_module = _SOLUTION_CACHE.get("module")
    if payload_hash == cached_hash and cached_module is not None:
        return cached_module, payload_hash, "memory"

    SOLUTION_ROOT.mkdir(parents=True, exist_ok=True)
    clear_directory_contents(SOLUTION_ROOT)
    write_raw_files(SOLUTION_ROOT, raw_files)
    clear_solution_modules(raw_files, extra_module_names=("word2kernel_fast_solution",))
    if str(SOLUTION_ROOT) not in sys.path:
        sys.path.insert(0, str(SOLUTION_ROOT))
    entry_parent = str((SOLUTION_ROOT / entry_file).parent)
    if entry_parent not in sys.path:
        sys.path.insert(0, entry_parent)

    module = _load_python_module(
        "word2kernel_fast_solution",
        SOLUTION_ROOT / entry_file,
    )
    _SOLUTION_CACHE["payload_hash"] = payload_hash
    _SOLUTION_CACHE["module"] = module
    return module, payload_hash, "reloaded"


def _restrict_bundle_to_num_tokens(
    bundle: dict,
    *,
    allowed_num_tokens: tuple[int, ...],
) -> dict:
    allowed = set(allowed_num_tokens)
    cases = []
    reference_outputs = []
    for case, refs in zip(
        bundle["cases"],
        bundle["reference_outputs"],
        strict=True,
    ):
        if case["num_tokens"] in allowed:
            cases.append(case)
            reference_outputs.append(refs)
    if not cases:
        raise RuntimeError(
            f"No synthetic cases found for allowed_num_tokens={allowed_num_tokens!r}."
        )
    return {
        "cases": cases,
        "reference_outputs": reference_outputs,
    }


def _build_stage_validation_bundle(bundle: dict) -> dict:
    reference_mod = _load_reference_module()
    restricted = _restrict_bundle_to_num_tokens(
        bundle,
        allowed_num_tokens=(STAGE_VALIDATION_NUM_TOKENS,),
    )
    case = restricted["cases"][0]
    sparse_indices = torch.full_like(case["sparse_indices"], -1)
    page_ids = stage_validation_page_ids(int(case["ckv_cache"].shape[0]))

    for page_slot, page_id in enumerate(page_ids):
        start = page_slot * SYNTHETIC_PAGE_SIZE
        stop = start + SYNTHETIC_PAGE_SIZE
        sparse_indices[:, start:stop] = torch.arange(
            page_id * SYNTHETIC_PAGE_SIZE,
            (page_id + 1) * SYNTHETIC_PAGE_SIZE,
            dtype=sparse_indices.dtype,
            device=sparse_indices.device,
        )

    stage_case = {
        **case,
        "sparse_indices": sparse_indices,
    }
    with torch.no_grad():
        ref_output, ref_lse = reference_mod.run(
            stage_case["q_nope"],
            stage_case["q_pe"],
            stage_case["ckv_cache"],
            stage_case["kpe_cache"],
            stage_case["sparse_indices"],
            stage_case["sm_scale"],
        )
        torch.cuda.synchronize()

    return {
        "cases": [stage_case],
        "reference_outputs": [(ref_output, ref_lse)],
    }


def _run_synthetic_sweep(solution_mod, entry_func: str, bundle: dict) -> list[dict]:
    import contextlib
    import io

    solution_run = getattr(solution_mod, "run", None)
    solution_entry = getattr(solution_mod, entry_func, None)
    use_run = callable(solution_run)

    if not use_run and not callable(solution_entry):
        case_tokens = tuple(case["num_tokens"] for case in bundle["cases"])
        return make_synthetic_failure_results(
            f"Solution module is missing callable run() and '{entry_func}()'.",
            num_tokens_cases=case_tokens,
        )

    results = []
    for case, (ref_output, ref_lse) in zip(
        bundle["cases"],
        bundle["reference_outputs"],
        strict=True,
    ):
        num_tokens = case["num_tokens"]
        captured = io.StringIO()
        try:
            with torch.no_grad(), contextlib.redirect_stdout(captured):
                if use_run:
                    test_output, test_lse = solution_run(
                        case["q_nope"],
                        case["q_pe"],
                        case["ckv_cache"],
                        case["kpe_cache"],
                        case["sparse_indices"],
                        case["sm_scale"],
                    )
                else:
                    test_output = torch.empty_like(ref_output)
                    test_lse = torch.empty_like(ref_lse)
                    solution_entry(
                        case["q_nope"],
                        case["q_pe"],
                        case["ckv_cache"],
                        case["kpe_cache"],
                        case["sparse_indices"],
                        case["sm_scale"],
                        test_output,
                        test_lse,
                    )
                torch.cuda.synchronize()

            captured_text = captured.getvalue()
            if captured_text:
                sys.stdout.write(captured_text)

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
                    "log": captured_text,
                }
            )
        except Exception:
            import traceback

            captured_text = captured.getvalue()
            if captured_text:
                sys.stdout.write(captured_text)

            results.append(
                {
                    "workload_uuid": f"synthetic-{num_tokens}",
                    "num_tokens": num_tokens,
                    "num_pages": SYNTHETIC_NUM_PAGES,
                    "status": "FAILED",
                    "log": captured_text + "\n" + traceback.format_exc(),
                }
            )
    return results


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={FIXTURE_MOUNT: FIXTURE_VOLUME},
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    name=FUNCTION_NAME,
)
def run_dsa_synthetic_fast(
    raw_files: dict[str, str],
    entry_point: str = "kernel.py::kernel",
    rebuild_fixture: bool = False,
    stage_validation: bool = False,
) -> dict:
    """Run the optimized synthetic DSA correctness sweep on Modal."""
    start = time.perf_counter()

    try:
        entry_file, entry_func = parse_entry_point(entry_point)
    except ValueError as exc:
        return {
            "success": False,
            "error": str(exc),
        }

    if entry_file not in raw_files:
        available = ", ".join(sorted(raw_files))
        case_tokens = (
            (STAGE_VALIDATION_NUM_TOKENS,)
            if stage_validation
            else SYNTHETIC_NUM_TOKENS
        )
        return {
            "success": True,
            "results": make_synthetic_failure_results(
                f"Entry file '{entry_file}' not found in payload. Available: {available}",
                num_tokens_cases=case_tokens,
            ),
        }

    bundle, fixture_version, fixture_source = _ensure_fixture_bundle(rebuild_fixture)
    if stage_validation:
        bundle = _build_stage_validation_bundle(bundle)
    solution_mod, payload_hash, solution_source = _ensure_solution_module(
        raw_files,
        entry_point,
        entry_file,
    )
    results = _run_synthetic_sweep(solution_mod, entry_func, bundle)

    return {
        "success": True,
        "results": results,
        "fixture_version": fixture_version,
        "fixture_source": fixture_source,
        "solution_source": solution_source,
        "payload_hash": payload_hash,
        "elapsed_s": time.perf_counter() - start,
    }
