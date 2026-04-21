from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

APP_NAME = "flashinfer-competition-synthetic"
FUNCTION_NAME = "run_dsa_synthetic_fast"
FIXTURE_VOLUME_NAME = "flashinfer-competition-synthetic-fixtures"
FIXTURE_MOUNT = "/synthetic-fixtures"
TRACK = "dsa_attention"
TRACK_DEFINITION = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
REFERENCE_FILENAME = "reference.py"
FIXTURE_SCHEMA_VERSION = 1
FIXTURE_SHARED_CACHE_SEED = 20260402
SYNTHETIC_NUM_TOKENS = (1, 2, 6, 7, 8)
STAGE_VALIDATION_NUM_TOKENS = SYNTHETIC_NUM_TOKENS[0]
SYNTHETIC_CASE_SEEDS = {num_tokens: 42 + num_tokens for num_tokens in SYNTHETIC_NUM_TOKENS}
SYNTHETIC_NUM_PAGES = 8462
SYNTHETIC_PAGE_SIZE = 64
SYNTHETIC_TOPK = 2048
SYNTHETIC_NUM_QO_HEADS = 16
SYNTHETIC_CKV_DIM = 512
SYNTHETIC_KPE_DIM = 64
SYNTHETIC_SM_SCALE = 0.1352337788608801
STAGE_VALIDATION_PREFERRED_PAGE_IDS = (1, 3, 5)

STAGE_VALIDATION_INSPECTION_FOOTER = (
    "Stage validation is inspection-only.\n"
    "Inspect tagged `[PyTorch]` / `[CuTeDSL]` BEGIN/END blocks in this transcript.\n"
    "When invoked through the repo tools, use `grep_search` or `read_file` on "
    "`last_shell_dump.txt` for shell-driven review."
)


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


def make_synthetic_failure_results(
    message: str,
    *,
    num_tokens_cases: tuple[int, ...] | None = None,
) -> list[dict]:
    """Create one failure row per requested synthetic workload."""
    cases = num_tokens_cases or SYNTHETIC_NUM_TOKENS
    return [
        {
            "workload_uuid": f"synthetic-{num_tokens}",
            "num_tokens": num_tokens,
            "num_pages": SYNTHETIC_NUM_PAGES,
            "status": "FAILED",
            "log": message,
        }
        for num_tokens in cases
    ]


def print_synthetic_results(results: list[dict]) -> None:
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


def print_stage_validation_logs(results: list[dict]) -> None:
    """Print raw per-case logs for inspection-oriented stage validation.

    Unlike the generic synthetic summary, stage validation is specifically about
    inspecting tagged stdout blocks. Surface the captured log even for PASSED
    cases so the local launcher transcript mirrors what reviewers are told to
    inspect in `last_shell_dump.txt`.
    """
    logs = [
        (
            result.get("num_tokens", "?"),
            result.get("status", "?"),
            str(result.get("log", "")).rstrip(),
        )
        for result in results
        if str(result.get("log", "")).strip()
    ]

    if not logs:
        print("\nNo stage-validation logs were captured.")
        return

    print(f"\n  STAGE VALIDATION LOGS ({len(logs)}):")
    for num_tokens, status, log_text in logs:
        print(f"  --- num_tokens={num_tokens} status={status} ---")
        print(log_text)


def print_stage_validation_footer() -> None:
    """Print the shell-oriented footer for inspection-only stage validation."""
    print(STAGE_VALIDATION_INSPECTION_FOOTER)


def stage_validation_page_ids(num_pages: int) -> tuple[int, ...]:
    """Choose a small deterministic set of page ids for round-0 stage validation.

    The stage-validation harness for staged round-0 kernels expects page-dense
    sparse indices. We keep the validation case shell-friendly and deterministic
    by selecting a few whole pages and leaving the rest invalid.
    """
    if num_pages <= 0:
        return ()

    max_page_slots = SYNTHETIC_TOPK // SYNTHETIC_PAGE_SIZE
    target_count = min(len(STAGE_VALIDATION_PREFERRED_PAGE_IDS), num_pages, max_page_slots)
    page_ids: list[int] = []
    seen: set[int] = set()

    for page_id in STAGE_VALIDATION_PREFERRED_PAGE_IDS:
        if 0 <= page_id < num_pages and page_id not in seen:
            page_ids.append(page_id)
            seen.add(page_id)
        if len(page_ids) >= target_count:
            return tuple(page_ids)

    for page_id in range(num_pages):
        if page_id in seen:
            continue
        page_ids.append(page_id)
        seen.add(page_id)
        if len(page_ids) >= target_count:
            break

    return tuple(page_ids)


def write_raw_files(target_dir: Path, raw_files: dict[str, str]) -> None:
    """Materialize base64-encoded files into target_dir."""
    import base64

    target_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, b64content in raw_files.items():
        file_path = target_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(base64.b64decode(b64content))


def clear_directory_contents(target_dir: Path) -> None:
    """Remove all children from target_dir without deleting the directory itself."""
    import shutil

    if not target_dir.exists():
        return
    for child in target_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def clear_solution_modules(raw_files: dict[str, str], extra_module_names: tuple[str, ...] = ()) -> None:
    """Drop solution-related modules so a reused container reloads fresh source."""
    for module_name in extra_module_names:
        sys.modules.pop(module_name, None)
    for rel_path in raw_files:
        path = Path(rel_path)
        if path.suffix == ".py":
            sys.modules.pop(path.stem, None)


def compute_payload_hash(entry_point: str, raw_files: dict[str, str]) -> str:
    """Hash the logical solution payload."""
    hasher = hashlib.sha256()
    hasher.update(entry_point.encode())
    hasher.update(b"\0")
    for rel_path in sorted(raw_files):
        hasher.update(rel_path.encode())
        hasher.update(b"\0")
        hasher.update(raw_files[rel_path].encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def build_fixture_manifest(reference_sha256: str) -> dict:
    """Build the deterministic fixture manifest."""
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "definition": TRACK_DEFINITION,
        "reference_sha256": reference_sha256,
        "constants": {
            "num_tokens": list(SYNTHETIC_NUM_TOKENS),
            "num_pages": SYNTHETIC_NUM_PAGES,
            "page_size": SYNTHETIC_PAGE_SIZE,
            "topk": SYNTHETIC_TOPK,
            "num_qo_heads": SYNTHETIC_NUM_QO_HEADS,
            "ckv_dim": SYNTHETIC_CKV_DIM,
            "kpe_dim": SYNTHETIC_KPE_DIM,
            "sm_scale": SYNTHETIC_SM_SCALE,
        },
        "seeds": {
            "shared_cache": FIXTURE_SHARED_CACHE_SEED,
            "cases": {
                str(num_tokens): seed
                for num_tokens, seed in SYNTHETIC_CASE_SEEDS.items()
            },
        },
    }


def compute_fixture_version(manifest: dict) -> str:
    """Compute a stable version string for the fixture manifest."""
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
