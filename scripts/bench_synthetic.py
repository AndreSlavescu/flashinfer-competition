"""
Fast local launcher for the deployed synthetic DSA correctness service.

Usage:
    .venv/bin/modal deploy scripts/bench_synthetic_service.py
    .venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention
"""

from __future__ import annotations

import argparse
import ast
import base64
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap_runtime import require_dependencies

require_dependencies(
    {"modal": "modal"},
    entrypoint="scripts/bench_synthetic.py",
)

import modal

from bench_synthetic_common import (
    APP_NAME,
    FUNCTION_NAME,
    STAGE_VALIDATION_NUM_TOKENS,
    TRACK,
    parse_entry_point,
    print_synthetic_results,
)
from stage_validation_parser import compare_tagged_outputs

def resolve_solution_dir(solution_dir: str) -> Path:
    solution_path = Path(solution_dir)
    if not solution_path.is_absolute():
        solution_path = PROJECT_ROOT / solution_path
    return solution_path.resolve()


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def candidate_module_paths(base_rel_path: Path) -> list[Path]:
    if base_rel_path.suffix == ".py":
        return [base_rel_path]
    return [
        base_rel_path.with_suffix(".py"),
        base_rel_path / "__init__.py",
    ]


def resolve_local_module_paths(
    solution_path: Path,
    current_rel_path: Path,
    module_name: str | None,
    level: int,
) -> list[Path]:
    current_dir = current_rel_path.parent
    search_roots: list[Path] = []

    if level > 0:
        anchor = current_dir
        for _ in range(level - 1):
            anchor = anchor.parent
        search_roots.append(anchor)
    else:
        if current_dir != Path("."):
            search_roots.append(current_dir)
        search_roots.append(Path("."))

    module_parts = module_name.split(".") if module_name else []
    resolved: list[Path] = []
    seen: set[str] = set()

    for root in search_roots:
        candidate_base = root.joinpath(*module_parts) if module_parts else root
        for rel_candidate in candidate_module_paths(candidate_base):
            try:
                abs_candidate = (solution_path / rel_candidate).resolve()
                abs_candidate.relative_to(solution_path)
            except ValueError:
                continue
            if abs_candidate.exists():
                rel_posix = rel_candidate.as_posix()
                if rel_posix not in seen:
                    resolved.append(rel_candidate)
                    seen.add(rel_posix)
    return resolved


def discover_local_python_deps(solution_path: Path, rel_path: Path) -> set[Path]:
    source = (solution_path / rel_path).read_text()
    tree = ast.parse(source, filename=str(rel_path))
    deps: set[Path] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                deps.update(
                    resolve_local_module_paths(
                        solution_path,
                        rel_path,
                        alias.name,
                        level=0,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                deps.update(
                    resolve_local_module_paths(
                        solution_path,
                        rel_path,
                        node.module,
                        level=node.level,
                    )
                )
                for alias in node.names:
                    deps.update(
                        resolve_local_module_paths(
                            solution_path,
                            rel_path,
                            f"{node.module}.{alias.name}",
                            level=node.level,
                        )
                    )
            else:
                for alias in node.names:
                    deps.update(
                        resolve_local_module_paths(
                            solution_path,
                            rel_path,
                            alias.name,
                            level=node.level,
                        )
                    )

    return deps


def parent_init_files(solution_path: Path, rel_path: Path) -> list[Path]:
    inits: list[Path] = []
    parent = rel_path.parent
    while parent != Path("."):
        init_rel = parent / "__init__.py"
        if (solution_path / init_rel).exists():
            inits.append(init_rel)
        parent = parent.parent
    return inits


def collect_dependency_closure(solution_path: Path, entry_file: str) -> dict[str, str]:
    entry_rel = Path(entry_file)
    entry_abs = (solution_path / entry_rel).resolve()
    if not entry_abs.exists():
        raise FileNotFoundError(f"Entry file not found: {solution_path / entry_rel}")
    if entry_abs.suffix != ".py":
        raise ValueError(f"Fast synthetic mode requires a Python entry file, got '{entry_file}'.")

    to_visit = [entry_rel]
    seen: set[str] = set()
    raw_files: dict[str, str] = {}

    while to_visit:
        rel_path = to_visit.pop()
        rel_key = rel_path.as_posix()
        if rel_key in seen:
            continue

        abs_path = (solution_path / rel_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Missing local dependency: {solution_path / rel_path}")
        abs_path.relative_to(solution_path)
        seen.add(rel_key)
        raw_files[rel_key] = encode_file(abs_path)

        if abs_path.suffix != ".py":
            continue

        for init_rel in parent_init_files(solution_path, rel_path):
            if init_rel.as_posix() not in seen:
                to_visit.append(init_rel)
        for dep_rel in discover_local_python_deps(solution_path, rel_path):
            if dep_rel.as_posix() not in seen:
                to_visit.append(dep_rel)

    return dict(sorted(raw_files.items()))


def collect_all_files(solution_path: Path) -> dict[str, str]:
    raw_files: dict[str, str] = {}
    for path in sorted(solution_path.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        raw_files[path.relative_to(solution_path).as_posix()] = encode_file(path)
    return raw_files


def total_payload_bytes(raw_files: dict[str, str]) -> int:
    total = 0
    for encoded in raw_files.values():
        total += len(base64.b64decode(encoded))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deployed fast synthetic DSA correctness sweep.",
    )
    parser.add_argument("--track", default=TRACK)
    parser.add_argument("--solution-dir", default="solution/dsa_attention")
    parser.add_argument("--entry-point", default="kernel.py::kernel")
    parser.add_argument("--include-all-files", action="store_true")
    parser.add_argument("--rebuild-fixture", action="store_true")
    parser.add_argument(
        "--stage-validation",
        action="store_true",
        help=(
            "Run the entry point in dedicated single-case stage-validation mode. "
            "This is intended for parser-backed prefix_validation_harness runs."
        ),
    )
    args = parser.parse_args()

    if args.track != TRACK:
        print(
            f"Fast synthetic launcher only supports track='{TRACK}', got '{args.track}'."
        )
        sys.exit(1)

    solution_path = resolve_solution_dir(args.solution_dir)
    if not solution_path.exists():
        print(f"Solution directory not found: {solution_path}")
        sys.exit(1)

    try:
        entry_file, _ = parse_entry_point(args.entry_point)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    try:
        if args.include_all_files:
            raw_files = collect_all_files(solution_path)
        else:
            raw_files = collect_dependency_closure(solution_path, entry_file)
    except (FileNotFoundError, ValueError, SyntaxError) as exc:
        print(f"Failed to collect solution payload: {exc}")
        sys.exit(1)

    if not raw_files:
        print(f"No files collected from {solution_path}")
        sys.exit(1)

    payload_bytes = total_payload_bytes(raw_files)
    print(f"Track: {args.track}")
    print(f"Solution: {solution_path}")
    print(f"Entry point: {args.entry_point}")
    if args.stage_validation:
        print(f"Mode: stage-validation (single case, num_tokens={STAGE_VALIDATION_NUM_TOKENS})")
    print(
        f"Payload: {len(raw_files)} file(s), {payload_bytes / 1024:.1f} KiB "
        f"{'(all files)' if args.include_all_files else '(dependency closure)'}"
    )
    print("Invoking deployed synthetic runner...")

    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    start = time.perf_counter()
    try:
        result = fn.remote(
            raw_files=raw_files,
            entry_point=args.entry_point,
            rebuild_fixture=args.rebuild_fixture,
            stage_validation=args.stage_validation,
        )
    except modal.exception.NotFoundError:
        print(
            "Synthetic service is not deployed yet.\n"
            "Deploy it with: .venv/bin/modal deploy scripts/bench_synthetic_service.py"
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Remote synthetic run failed: {exc}")
        sys.exit(1)

    if not result.get("success"):
        print(f"Synthetic service failed: {result.get('error', 'unknown error')}")
        sys.exit(1)

    print(
        "Service cache: "
        f"fixture={result.get('fixture_source', '?')} "
        f"solution={result.get('solution_source', '?')}"
    )
    print(
        "Service ids: "
        f"fixture_version={str(result.get('fixture_version', '?'))[:12]} "
        f"payload_hash={str(result.get('payload_hash', '?'))[:12]}"
    )
    print(
        "Timing: "
        f"remote={result.get('elapsed_s', 0.0):.3f}s "
        f"end_to_end={time.perf_counter() - start:.3f}s"
    )
    results = result.get("results", [])
    print_synthetic_results(results)

    transcript = "\n".join(
        r.get("log", "") for r in results if r.get("log")
    )
    report = compare_tagged_outputs(transcript)
    if args.stage_validation or report.fields:
        print()
        print(report.format_summary())
        if not report.overall_pass:
            sys.exit(1)


if __name__ == "__main__":
    main()
