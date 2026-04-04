"""Shared tool definitions for all kernel generation agents.

Uses SDK built-in tools where available (ShellTool, ApplyPatchTool, WebSearchTool,
codex_tool). Repo inspection and benchmark wrappers remain custom function tools.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path

from agents import (
    ApplyPatchTool,
    RunContextWrapper,
    ShellTool,
    ShellToolLocalSkill,
    WebSearchTool,
    apply_diff,
    function_tool,
)
from agents.editor import ApplyPatchOperation, ApplyPatchResult
from agents.extensions.experimental.codex import ThreadOptions, codex_tool
from agents.tool import ShellCallOutcome, ShellCommandOutput, ShellCommandRequest, ShellResult

from kernel_agents.context import SharedContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_WORKBENCH_SKILL_DIR = (
    PROJECT_ROOT / "kernel_agents" / "skills" / "kernel-workbench"
)
DEFAULT_SEARCH_OUTPUT_LIMIT = 20_000
DEFAULT_SHELL_OUTPUT_LIMIT = 20_000
DEFAULT_READ_FILE_LIMIT = 40_000
DEFAULT_WEB_FETCH_LIMIT = 20_000
DEFAULT_BENCH_LANGUAGE = "python"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root_from_context(ctx: RunContextWrapper[SharedContext] | None) -> Path:
    """Resolve the active project root from run context or fall back to this repo."""
    if ctx is not None:
        context = getattr(ctx, "context", None)
        project_root = getattr(context, "project_root", None)
        if isinstance(project_root, str) and project_root:
            return Path(project_root).resolve()
    return PROJECT_ROOT


def _resolve_workspace_path(
    project_root: Path,
    path: str,
    *,
    ensure_parent: bool = False,
) -> Path:
    """Resolve a path inside the project root and reject workspace escapes."""
    candidate = Path(path)
    target = candidate if candidate.is_absolute() else (project_root / candidate)
    target = target.resolve()
    try:
        target.relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError(f"Path escapes project root: {path}") from exc
    if ensure_parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _display_path(path: Path, project_root: Path) -> str:
    """Return a project-relative display path."""
    try:
        rel = path.relative_to(project_root)
    except ValueError:
        return str(path)
    return rel.as_posix() or "."


def _truncate_output(
    text: str,
    limit: int = 100_000,
    suffix: str = "\n... (truncated)",
) -> str:
    """Clip tool output to keep traces readable."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(suffix):
        return text[:limit]
    keep = max(limit - len(suffix), 0)
    return text[:keep] + suffix


def _truncate_block(text: str, *, max_lines: int = 48, max_chars: int = 10_000) -> str:
    """Trim multiline output without losing the leading context."""
    stripped = text.strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]
    return _truncate_output("\n".join(lines), limit=max_chars)


def _truncate_shell_streams(stdout: str, stderr: str, limit: int | None) -> tuple[str, str]:
    """Honor max_output_length by trimming combined shell output."""
    if limit is None or limit <= 0:
        return stdout, stderr

    total = len(stdout) + len(stderr)
    if total <= limit:
        return stdout, stderr

    if len(stdout) >= limit:
        return _truncate_output(stdout, limit), ""

    stderr_limit = max(limit - len(stdout), 0)
    return stdout, _truncate_output(stderr, stderr_limit)


def _format_search_result(
    tool_name: str,
    returncode: int,
    stdout: bytes | str,
    stderr: bytes | str,
    search_path: str,
) -> str:
    """Normalize grep/rg subprocess results into trace-friendly strings."""
    out = stdout.decode("utf-8", errors="replace") if isinstance(stdout, (bytes, bytearray)) else stdout
    err = stderr.decode("utf-8", errors="replace") if isinstance(stderr, (bytes, bytearray)) else stderr

    if returncode == 0:
        return _truncate_output(
            out,
            limit=DEFAULT_SEARCH_OUTPUT_LIMIT,
            suffix="\n... (search output truncated)",
        )
    if returncode == 1 and not err.strip():
        return "No matches found."

    details = err.strip() or out.strip()
    if details:
        details = _truncate_output(details, limit=8_000)
        return f"ERROR: {tool_name} exited with code {returncode} while searching {search_path}\n{details}"
    return f"ERROR: {tool_name} exited with code {returncode} while searching {search_path}"


def _command_display(command: Sequence[str]) -> str:
    """Render a command for tool output."""
    return shlex.join(list(command))


def _extract_section(text: str, marker: str) -> str:
    """Extract a trailing section from a CLI transcript."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if marker in line:
            return "\n".join(lines[index:])
    return ""


def _build_workflow_report(
    *,
    command: Sequence[str],
    returncode: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    timeout_s: int,
    summary: str | None,
    failure_marker: str,
    extra_line: str | None = None,
) -> str:
    """Build a concise, model-friendly workflow summary."""
    lines = [f"Command: {_command_display(command)}"]
    if timed_out:
        lines.append(f"Status: timed out after {timeout_s}s")
    else:
        lines.append(f"Exit code: {returncode}")
    if summary:
        lines.append(f"Summary: {summary}")
    if extra_line:
        lines.append(extra_line)

    failure_section = _extract_section(stdout, failure_marker)
    if failure_section:
        lines.append("Failures:")
        lines.append(_truncate_block(failure_section))
    elif returncode != 0:
        details = stderr.strip() or stdout.strip()
        if details:
            lines.append("Logs:")
            lines.append(_truncate_block(details))
    elif stderr.strip():
        lines.append("Warnings:")
        lines.append(_truncate_block(stderr))

    return "\n".join(lines)


async def _run_command(
    *,
    project_root: Path,
    command: Sequence[str],
    timeout_s: int,
) -> tuple[int, str, str, bool]:
    """Run a fixed command locally and capture UTF-8 output."""
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=project_root,
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        stdout_bytes, stderr_bytes = await proc.communicate()

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return proc.returncode or 0, stdout, stderr, timed_out


def _resolve_solution_dir(
    ctx: RunContextWrapper[SharedContext],
    solution_dir: str,
) -> tuple[Path, str]:
    """Resolve the benchmark solution directory relative to project root."""
    project_root = _project_root_from_context(ctx)
    requested = solution_dir or ctx.context.solution_dir
    if not requested:
        raise RuntimeError("Solution directory is not configured.")
    resolved = _resolve_workspace_path(project_root, requested)
    return resolved, _display_path(resolved, project_root)


def _track_for_solution_dir(solution_dir: str) -> str:
    """Infer the benchmark track from the solution directory name."""
    name = Path(solution_dir).name
    return name or "dsa_attention"


# ---------------------------------------------------------------------------
# Built-in: ShellTool — local executor + shell skill
# ---------------------------------------------------------------------------

class ShellExecutor:
    """Executes shell commands locally; approvals are handled by ShellTool."""

    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = Path(cwd or PROJECT_ROOT).resolve()

    @staticmethod
    def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
        """Terminate the shell command and any children it spawned."""
        if proc.pid is None:
            proc.kill()
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def __call__(self, request: ShellCommandRequest) -> ShellResult:
        action = request.data.action
        working_directory = _project_root_from_context(request.ctx_wrapper)
        timeout = (action.timeout_ms or 0) / 1000 or None
        output_limit = action.max_output_length or DEFAULT_SHELL_OUTPUT_LIMIT

        outputs: list[ShellCommandOutput] = []
        for command in action.commands:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=working_directory,
                env=os.environ.copy(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            timed_out = False
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                self._kill_process_group(proc)
                stdout_bytes, stderr_bytes = await proc.communicate()

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            stdout, stderr = _truncate_shell_streams(stdout, stderr, output_limit)

            outputs.append(
                ShellCommandOutput(
                    command=command,
                    stdout=stdout,
                    stderr=stderr,
                    outcome=ShellCallOutcome(
                        type="timeout" if timed_out else "exit",
                        exit_code=getattr(proc, "returncode", None),
                    ),
                )
            )

            if timed_out:
                break

        return ShellResult(
            output=outputs,
            provider_data={"working_directory": str(working_directory)},
        )


def build_local_shell_skill() -> ShellToolLocalSkill:
    """Build the local shell skill bundle for canonical repo workflows."""
    return ShellToolLocalSkill(
        name="kernel-workbench",
        description="Repo command playbook for file inspection, validation, and benchmarking.",
        path=str(KERNEL_WORKBENCH_SKILL_DIR),
    )


def build_shell_environment() -> dict[str, object]:
    """Build the shell environment, attaching the local skill when present."""
    environment: dict[str, object] = {"type": "local"}
    if KERNEL_WORKBENCH_SKILL_DIR.is_dir():
        environment["skills"] = [build_local_shell_skill()]
    return environment


shell_tool = ShellTool(
    executor=ShellExecutor(PROJECT_ROOT),
    environment=build_shell_environment(),
)


# ---------------------------------------------------------------------------
# Built-in: ApplyPatchTool — local workspace editor
# ---------------------------------------------------------------------------

class WorkspaceEditor:
    """ApplyPatchEditor implementation backed by SDK-native diff application."""

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            project_root = _project_root_from_context(operation.ctx_wrapper)
            target = _resolve_workspace_path(
                project_root,
                operation.path,
                ensure_parent=True,
            )
            relative = _display_path(target, project_root)
            content = apply_diff("", operation.diff or "", mode="create")
            target.write_text(content, encoding="utf-8")
            return ApplyPatchResult(status="completed", output=f"Created {relative}")
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            project_root = _project_root_from_context(operation.ctx_wrapper)
            target = _resolve_workspace_path(project_root, operation.path)
            relative = _display_path(target, project_root)
            if not target.exists():
                return ApplyPatchResult(status="failed", output=f"File not found: {relative}")
            original = target.read_text(encoding="utf-8")
            patched = apply_diff(original, operation.diff or "")
            target.write_text(patched, encoding="utf-8")
            return ApplyPatchResult(status="completed", output=f"Updated {relative}")
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            project_root = _project_root_from_context(operation.ctx_wrapper)
            target = _resolve_workspace_path(project_root, operation.path)
            relative = _display_path(target, project_root)
            target.unlink(missing_ok=True)
            return ApplyPatchResult(status="completed", output=f"Deleted {relative}")
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))


apply_patch_tool = ApplyPatchTool(editor=WorkspaceEditor())


# ---------------------------------------------------------------------------
# Built-in: WebSearchTool + Codex helper
# ---------------------------------------------------------------------------

web_search_tool = WebSearchTool()
codex_kernel_assist = codex_tool(
    name="codex_kernel_assist",
    sandbox_mode="read-only",
    working_directory=str(PROJECT_ROOT),
    persist_session=True,
    default_thread_options=ThreadOptions(
        model="gpt-5.4",
        model_reasoning_effort="low",
        network_access_enabled=False,
        web_search_mode="disabled",
        approval_policy="never",
    ),
)


# ---------------------------------------------------------------------------
# Custom repo tools
# ---------------------------------------------------------------------------

@function_tool
async def read_file(
    ctx: RunContextWrapper[SharedContext],
    file_path: str,
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        file_path: Absolute or project-relative path inside the repo.
        start_line: First line to read (1-based). 0 = from beginning.
        end_line: Last line to read (inclusive). 0 = to end.
    """
    project_root = _project_root_from_context(ctx)
    try:
        target = _resolve_workspace_path(project_root, file_path)
    except Exception as exc:
        return f"ERROR: {exc}"

    display_path = _display_path(target, project_root)
    if not target.exists():
        return f"ERROR: File not found: {display_path}"
    if not target.is_file():
        return f"ERROR: Not a file: {display_path}"

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"ERROR: {exc}"

    lines = text.splitlines(keepends=True)
    start_index = max(0, start_line - 1) if start_line > 0 else 0
    end_index = end_line if end_line > 0 else len(lines)
    selected = lines[start_index:end_index]
    numbered = [f"{start_index + idx + 1}: {line}" for idx, line in enumerate(selected)]
    return _truncate_output(
        "".join(numbered),
        limit=DEFAULT_READ_FILE_LIMIT,
        suffix="\n... (truncated at 40k chars)",
    )


@function_tool
async def glob_files(
    ctx: RunContextWrapper[SharedContext],
    pattern: str,
    directory: str = "",
) -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern relative to ``directory`` (or the repo root), for
            example ``**/*.py`` or ``kernel_*.py``.
        directory: Optional directory to search, for example ``references`` or
            ``solution/dsa_attention``. Prefer using this instead of embedding
            the base folder into ``pattern``.
    """
    project_root = _project_root_from_context(ctx)
    try:
        search_dir = (
            _resolve_workspace_path(project_root, directory)
            if directory
            else project_root
        )
    except Exception as exc:
        return f"ERROR: {exc}"

    display_dir = _display_path(search_dir, project_root)
    if not search_dir.exists():
        return f"ERROR: Path not found: {display_dir}"
    if not search_dir.is_dir():
        return f"ERROR: Not a directory: {display_dir}"

    try:
        matches = [
            _display_path(path, project_root)
            for path in search_dir.glob(pattern)
            if path.is_file()
        ]
    except Exception as exc:
        return f"ERROR: {exc}"

    if not matches:
        return "No files found."
    return "\n".join(sorted(matches)[:200])


@function_tool
async def grep_search(
    ctx: RunContextWrapper[SharedContext],
    pattern: str,
    path: str = "",
    file_glob: str = "",
) -> str:
    """Search file contents with a regex pattern (prefers ripgrep).

    Args:
        pattern: Regular expression to search for, for example ``def kernel`` or
            ``tcgen05|cute.compile``.
        path: Optional file or directory relative to the repo root. Use an
            empty string to search the whole repo.
        file_glob: Optional filename filter such as ``*.py`` or ``**/*.md``.
    """
    project_root = _project_root_from_context(ctx)
    try:
        search_path_obj = (
            _resolve_workspace_path(project_root, path)
            if path
            else project_root
        )
    except Exception as exc:
        return f"ERROR: {exc}"

    search_path = _display_path(search_path_obj, project_root)
    if not search_path_obj.exists():
        return f"ERROR: Path not found: {search_path}"

    target = "." if search_path_obj == project_root else search_path
    cmd = ["rg", "-n", "--max-count=100", "--max-columns=200"]
    if file_glob:
        cmd.extend(["--glob", file_glob])
    cmd.extend(["-e", pattern, target])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        return _format_search_result("rg", proc.returncode, stdout, stderr, search_path)
    except FileNotFoundError:
        fallback = ["grep", "-ErnI", "-m", "100"]
        if file_glob:
            fallback.extend(["--include", file_glob])
        fallback.extend(["-e", pattern, target])
        try:
            result = subprocess.run(
                fallback,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return _format_search_result(
                "grep",
                result.returncode,
                result.stdout,
                result.stderr,
                search_path,
            )
        except Exception as exc:
            return f"ERROR: {exc}"
    except Exception as exc:
        return f"ERROR: {exc}"

# ---------------------------------------------------------------------------
# Custom workflow tools
# ---------------------------------------------------------------------------

_SYNTHETIC_SUMMARY_RE = re.compile(r"SYNTHETIC RESULTS:\s*(\d+)/(\d+)\s+cases passed")
_BENCHMARK_SUMMARY_RE = re.compile(r"RESULTS:\s*(\d+)/(\d+)\s+workloads passed")
_SPEEDUP_RE = re.compile(r"Average speedup:\s*([^\n]+)")


def _summarize_synthetic_output(
    command: Sequence[str],
    returncode: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    timeout_s: int,
) -> str:
    """Parse synthetic-check output into a concise report."""
    match = _SYNTHETIC_SUMMARY_RE.search(stdout)
    summary = f"{match.group(1)}/{match.group(2)} synthetic cases passed" if match else None
    return _build_workflow_report(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        timeout_s=timeout_s,
        summary=summary,
        failure_marker="FAILED synthetic cases (",
    )


def _summarize_benchmark_output(
    command: Sequence[str],
    returncode: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    timeout_s: int,
) -> str:
    """Parse correctness/full benchmark output into a concise report."""
    summary_match = _BENCHMARK_SUMMARY_RE.search(stdout)
    speedup_match = _SPEEDUP_RE.search(stdout)
    summary = (
        f"{summary_match.group(1)}/{summary_match.group(2)} workloads passed"
        if summary_match
        else None
    )
    speedup_line = (
        f"Average speedup: {speedup_match.group(1).strip()}"
        if speedup_match
        else None
    )
    return _build_workflow_report(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        timeout_s=timeout_s,
        summary=summary,
        failure_marker="FAILED workloads (",
        extra_line=speedup_line,
    )


@function_tool
async def run_synthetic_check(
    ctx: RunContextWrapper[SharedContext],
    solution_dir: str = "",
    entry_point: str = "kernel.py::kernel",
    include_all_files: bool = False,
    rebuild_fixture: bool = False,
) -> str:
    """Run the fast synthetic correctness sweep for a solution directory."""
    try:
        solution_path, solution_rel = _resolve_solution_dir(ctx, solution_dir)
    except Exception as exc:
        return f"ERROR: {exc}"

    if not solution_path.exists():
        return f"ERROR: Solution directory not found: {solution_rel}"

    command = [
        ".venv/bin/python",
        "scripts/bench_synthetic.py",
        "--solution-dir",
        solution_rel,
        "--entry-point",
        entry_point,
    ]
    if include_all_files:
        command.append("--include-all-files")
    if rebuild_fixture:
        command.append("--rebuild-fixture")

    timeout_s = 1800
    returncode, stdout, stderr, timed_out = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
    )
    return _summarize_synthetic_output(
        command,
        returncode,
        stdout,
        stderr,
        timed_out,
        timeout_s,
    )


@function_tool
async def run_correctness_check(
    ctx: RunContextWrapper[SharedContext],
    solution_dir: str = "",
    entry_point: str = "kernel.py::kernel",
    lang: str = DEFAULT_BENCH_LANGUAGE,
) -> str:
    """Run the Modal correctness-only benchmark flow."""
    try:
        solution_path, solution_rel = _resolve_solution_dir(ctx, solution_dir)
    except Exception as exc:
        return f"ERROR: {exc}"

    if not solution_path.exists():
        return f"ERROR: Solution directory not found: {solution_rel}"

    command = [
        ".venv/bin/modal",
        "run",
        "scripts/bench.py",
        "--track",
        _track_for_solution_dir(solution_rel),
        "--solution-dir",
        solution_rel,
        "--entry-point",
        entry_point,
        "--correctness-only",
    ]
    if lang:
        command.extend(["--lang", lang])

    timeout_s = 5400
    returncode, stdout, stderr, timed_out = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
    )
    return _summarize_benchmark_output(
        command,
        returncode,
        stdout,
        stderr,
        timed_out,
        timeout_s,
    )


@function_tool
async def run_full_benchmark(
    ctx: RunContextWrapper[SharedContext],
    solution_dir: str = "",
    entry_point: str = "kernel.py::kernel",
    lang: str = DEFAULT_BENCH_LANGUAGE,
) -> str:
    """Run the full Modal benchmark flow."""
    try:
        solution_path, solution_rel = _resolve_solution_dir(ctx, solution_dir)
    except Exception as exc:
        return f"ERROR: {exc}"

    if not solution_path.exists():
        return f"ERROR: Solution directory not found: {solution_rel}"

    command = [
        ".venv/bin/modal",
        "run",
        "scripts/bench.py",
        "--track",
        _track_for_solution_dir(solution_rel),
        "--solution-dir",
        solution_rel,
        "--entry-point",
        entry_point,
    ]
    if lang:
        command.extend(["--lang", lang])

    timeout_s = 5400
    returncode, stdout, stderr, timed_out = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
    )
    return _summarize_benchmark_output(
        command,
        returncode,
        stdout,
        stderr,
        timed_out,
        timeout_s,
    )

# ---------------------------------------------------------------------------
# Custom exact-URL fallback
# ---------------------------------------------------------------------------

@function_tool
async def web_fetch(
    ctx: RunContextWrapper[SharedContext],
    url: str,
) -> str:
    """Fetch content from a URL and return the text body (truncated to 50k chars)."""
    del ctx

    try:
        # Keep this import lazy so urllib remains a real fallback.
        import httpx
    except ImportError:
        import urllib.request

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")[:DEFAULT_WEB_FETCH_LIMIT]
        except Exception as exc:
            return f"ERROR: {exc}"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text[:DEFAULT_WEB_FETCH_LIMIT]
    except Exception as exc:
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Collected tool list for agent registration
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    shell_tool,
    apply_patch_tool,
    web_search_tool,
    web_fetch,
    codex_kernel_assist,
    read_file,
    glob_files,
    grep_search,
    run_synthetic_check,
    run_correctness_check,
    run_full_benchmark,
]
