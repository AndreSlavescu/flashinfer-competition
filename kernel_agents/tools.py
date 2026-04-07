"""Shared tool definitions for all kernel generation agents.

Uses SDK built-in tools where available (ApplyPatchTool, WebSearchTool,
codex_tool). Repo inspection, benchmark, and profiling wrappers are custom
function tools.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agents import (
    ApplyPatchTool,
    RunContextWrapper,
    WebSearchTool,
    apply_diff,
    function_tool,
)
from agents.editor import ApplyPatchOperation, ApplyPatchResult
from agents.extensions.experimental.codex import ThreadOptions, codex_tool
from agents.tool import ShellCallOutcome, ShellCommandOutput

from kernel_agents.context import SharedContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_OUTPUT_LIMIT = 20_000
DEFAULT_SHELL_OUTPUT_LIMIT = 20_000
DEFAULT_READ_FILE_LIMIT = 40_000
DEFAULT_WEB_FETCH_LIMIT = 20_000
DEFAULT_SEARCH_MATCH_LIMIT = 100
REFERENCE_READ_WINDOW_LINES = 400
DEFAULT_BENCH_LANGUAGE = "python"
SHELL_OVERFLOW_FILE_NAME = "last_shell_overflow.txt"
SHELL_OVERFLOW_BANNER = (
    "retrieved trimmed shell output; full transcript saved to "
    f"{SHELL_OVERFLOW_FILE_NAME} and it will be replaced by the next overflowing shell-like tool call. "
    "Form a concrete hypothesis before running another potentially overflowing shell command. "
    f"Use read_file or grep_search on {SHELL_OVERFLOW_FILE_NAME} if you need more detail."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandRunResult:
    """Structured result for a local subprocess helper command."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    overflow_notice: str | None = None

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


def _prepend_notice(text: str, notice: str) -> str:
    """Attach a single-line notice ahead of tool output."""
    if not notice:
        return text
    if not text:
        return notice
    return f"{notice}\n{text}"


def _is_reference_display_path(display_path: str) -> bool:
    """Return True when a project-relative path points into the references tree."""
    return display_path == "references" or display_path.startswith("references/")


def _format_search_success(output: str, search_path: str) -> str:
    """Render successful search output with trim notices when caps are hit."""
    notices: list[str] = []
    match_lines = output.splitlines()
    if len(match_lines) >= DEFAULT_SEARCH_MATCH_LIMIT:
        notices.append(
            (
                f"retrieved trimmed search results for {search_path}; showing up to the first "
                f"{DEFAULT_SEARCH_MATCH_LIMIT} matches. Refine pattern, path, or file_glob to continue."
            )
        )

    was_char_truncated = len(output) > DEFAULT_SEARCH_OUTPUT_LIMIT
    rendered = _truncate_output(
        output,
        limit=DEFAULT_SEARCH_OUTPUT_LIMIT,
        suffix="\n... (search output truncated)",
    )
    if was_char_truncated:
        notices.append(
            (
                f"retrieved trimmed search results for {search_path} at "
                f"{DEFAULT_SEARCH_OUTPUT_LIMIT} chars; refine pattern, path, or file_glob to continue."
            )
        )
    return _prepend_notice(rendered, "\n".join(notices))


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
        return _format_search_success(out, search_path)
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


def _shell_overflow_path(project_root: Path) -> Path:
    """Resolve the stable shell overflow transcript path."""
    return project_root / SHELL_OVERFLOW_FILE_NAME


def _render_shell_transcript_block(output: ShellCommandOutput) -> str:
    """Render a single shell output entry into a full overflow transcript block."""
    lines: list[str] = []
    if output.command:
        lines.append(f"Command: {output.command}")
    if output.status == "timeout":
        lines.append("Status: timeout")
    else:
        lines.append(f"Exit code: {output.exit_code}")
    lines.append("STDOUT:")
    lines.append(output.stdout.rstrip("\n") or "(empty)")
    lines.append("")
    lines.append("STDERR:")
    lines.append(output.stderr.rstrip("\n") or "(empty)")
    return "\n".join(lines)


def _build_shell_overflow_transcript(
    *,
    source_tool: str,
    working_directory: Path,
    outputs: Sequence[ShellCommandOutput],
) -> str:
    """Build the full plain-text transcript written on shell overflow."""
    lines = [
        f"Source tool: {source_tool}",
        f"Working directory: {working_directory}",
        f"Overflow file: {SHELL_OVERFLOW_FILE_NAME}",
        "This file is overwritten by the next overflowing shell-like tool call.",
        "",
    ]
    if not outputs:
        lines.append("(no output)")
    for index, output in enumerate(outputs, start=1):
        if outputs:
            lines.append(f"[Command {index}]")
            lines.append(_render_shell_transcript_block(output))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_shell_overflow(project_root: Path, transcript: str) -> str:
    """Persist the current overflowing shell transcript and return the banner."""
    overflow_path = _shell_overflow_path(project_root)
    overflow_path.write_text(transcript, encoding="utf-8")
    return SHELL_OVERFLOW_BANNER


def _maybe_write_shell_overflow(
    *,
    project_root: Path,
    source_tool: str,
    working_directory: Path,
    outputs: Sequence[ShellCommandOutput],
    limit: int,
) -> str | None:
    """Write the overflow transcript when a shell-like tool exceeds its limit."""
    transcript = _build_shell_overflow_transcript(
        source_tool=source_tool,
        working_directory=working_directory,
        outputs=outputs,
    )
    if limit > 0 and len(transcript) <= limit:
        return None
    return _write_shell_overflow(project_root, transcript)


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
    source_tool: str,
) -> CommandRunResult:
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
    output = ShellCommandOutput(
        command=_command_display(command),
        stdout=stdout,
        stderr=stderr,
        outcome=ShellCallOutcome(
            type="timeout" if timed_out else "exit",
            exit_code=getattr(proc, "returncode", None),
        ),
    )
    overflow_notice = _maybe_write_shell_overflow(
        project_root=project_root,
        source_tool=source_tool,
        working_directory=project_root,
        outputs=[output],
        limit=DEFAULT_SHELL_OUTPUT_LIMIT,
    )
    return CommandRunResult(
        returncode=proc.returncode or 0,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        overflow_notice=overflow_notice,
    )


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
    total_lines = len(lines)
    start_index = max(0, start_line - 1) if start_line > 0 else 0
    end_index = end_line if end_line > 0 else total_lines
    notices: list[str] = []

    if (
        start_line <= 0
        and end_line <= 0
        and _is_reference_display_path(display_path)
        and total_lines > REFERENCE_READ_WINDOW_LINES
    ):
        start_index = 0
        end_index = REFERENCE_READ_WINDOW_LINES
        notices.append(
            (
                f"retrieved trimmed {display_path}:[1]-[{end_index}] of {total_lines} lines; "
                "request start_line/end_line for another section."
            )
        )

    selected = lines[start_index:end_index]
    numbered = [f"{start_index + idx + 1}: {line}" for idx, line in enumerate(selected)]
    rendered_text = "".join(numbered)
    was_truncated = len(rendered_text) > DEFAULT_READ_FILE_LIMIT
    rendered = _truncate_output(
        rendered_text,
        limit=DEFAULT_READ_FILE_LIMIT,
        suffix="\n... (truncated at 40k chars)",
    )
    if was_truncated:
        notices.append(
            (
                f"retrieved trimmed {display_path}:[{start_index + 1}]-[{min(end_index, total_lines)}] "
                f"at {DEFAULT_READ_FILE_LIMIT} chars; request a narrower line range if you need more."
            )
        )
    return _prepend_notice(rendered, "\n".join(notices))


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
    cmd = ["rg", "-n", f"--max-count={DEFAULT_SEARCH_MATCH_LIMIT}", "--max-columns=200"]
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
        fallback = ["grep", "-ErnI", "-m", str(DEFAULT_SEARCH_MATCH_LIMIT)]
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
    result = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_synthetic_check",
    )
    summary = _summarize_synthetic_output(
        command,
        result.returncode,
        result.stdout,
        result.stderr,
        result.timed_out,
        timeout_s,
    )
    return _prepend_notice(summary, result.overflow_notice or "")


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
    result = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_correctness_check",
    )
    summary = _summarize_benchmark_output(
        command,
        result.returncode,
        result.stdout,
        result.stderr,
        result.timed_out,
        timeout_s,
    )
    return _prepend_notice(summary, result.overflow_notice or "")


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
    result = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_full_benchmark",
    )
    summary = _summarize_benchmark_output(
        command,
        result.returncode,
        result.stdout,
        result.stderr,
        result.timed_out,
        timeout_s,
    )
    return _prepend_notice(summary, result.overflow_notice or "")

# ---------------------------------------------------------------------------
# Custom exact-URL fallback
# ---------------------------------------------------------------------------

@function_tool
async def web_fetch(
    ctx: RunContextWrapper[SharedContext],
    url: str,
) -> str:
    """Fetch content from a URL and return the text body (truncated to 20k chars)."""
    del ctx

    try:
        # Keep this import lazy so urllib remains a real fallback.
        import httpx
    except ImportError:
        import urllib.request

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                if len(text) <= DEFAULT_WEB_FETCH_LIMIT:
                    return text
                shown = text[:DEFAULT_WEB_FETCH_LIMIT]
                return _prepend_notice(
                    shown,
                    (
                        f"retrieved trimmed {url}:[1]-[{DEFAULT_WEB_FETCH_LIMIT}] "
                        f"of {len(text)} chars"
                    ),
                )
        except Exception as exc:
            return f"ERROR: {exc}"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            if len(resp.text) <= DEFAULT_WEB_FETCH_LIMIT:
                return resp.text
            shown = resp.text[:DEFAULT_WEB_FETCH_LIMIT]
            return _prepend_notice(
                shown,
                (
                    f"retrieved trimmed {url}:[1]-[{DEFAULT_WEB_FETCH_LIMIT}] "
                    f"of {len(resp.text)} chars"
                ),
            )
    except Exception as exc:
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Directory listing + file comparison tools
# ---------------------------------------------------------------------------

@function_tool
async def list_directory(
    ctx: RunContextWrapper[SharedContext],
    path: str = "",
) -> str:
    """List files and directories at a given path.

    Args:
        path: Absolute or project-relative directory path. Empty string means
            the repo root.
    """
    project_root = _project_root_from_context(ctx)
    try:
        target = (
            _resolve_workspace_path(project_root, path) if path else project_root
        )
    except Exception as exc:
        return f"ERROR: {exc}"

    display = _display_path(target, project_root)
    if not target.exists():
        return f"ERROR: Path not found: {display}"
    if not target.is_dir():
        return f"ERROR: Not a directory: {display}"

    entries: list[str] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            if child.is_dir():
                entries.append(f"d  {child.name}/")
            else:
                size = child.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                entries.append(f"f  {child.name}  ({size_str})")
    except Exception as exc:
        return f"ERROR: {exc}"

    if not entries:
        return f"{display}/  (empty directory)"

    cap = 500
    header = f"{display}/  ({len(entries)} entries)"
    if len(entries) > cap:
        header += f" — showing first {cap}"
        entries = entries[:cap]
    return header + "\n" + "\n".join(entries)


@function_tool
async def diff_files(
    ctx: RunContextWrapper[SharedContext],
    file_a: str,
    file_b: str,
    context_lines: int = 3,
) -> str:
    """Compare two files and return a unified diff.

    Args:
        file_a: Path to the first file (project-relative or absolute).
        file_b: Path to the second file (project-relative or absolute).
        context_lines: Number of context lines around each change (default 3).
    """
    project_root = _project_root_from_context(ctx)
    try:
        path_a = _resolve_workspace_path(project_root, file_a)
        path_b = _resolve_workspace_path(project_root, file_b)
    except Exception as exc:
        return f"ERROR: {exc}"

    display_a = _display_path(path_a, project_root)
    display_b = _display_path(path_b, project_root)

    for p, d in [(path_a, display_a), (path_b, display_b)]:
        if not p.exists():
            return f"ERROR: File not found: {d}"
        if not p.is_file():
            return f"ERROR: Not a file: {d}"

    try:
        lines_a = path_a.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        lines_b = path_b.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except Exception as exc:
        return f"ERROR: {exc}"

    diff = list(difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile=display_a,
        tofile=display_b,
        n=context_lines,
    ))
    if not diff:
        return "Files are identical."
    return _truncate_output("".join(diff), limit=DEFAULT_READ_FILE_LIMIT)


# ---------------------------------------------------------------------------
# Profiling workflow tools
# ---------------------------------------------------------------------------

@function_tool
async def run_ncu_profile(
    ctx: RunContextWrapper[SharedContext],
    solution_dir: str = "",
    entry_point: str = "kernel.py::kernel",
) -> str:
    """Run NCU profiling on Modal B200 to get GPU hardware utilization metrics.

    Returns DRAM throughput, SM utilization, L2 hit rate, warp stall reasons,
    and occupancy data.

    Args:
        solution_dir: Solution directory (default: from context). Reserved for
            future parameterisation of the NCU entrypoint.
        entry_point: Kernel entry point (default kernel.py::kernel). Reserved
            for future parameterisation of the NCU entrypoint.
    """
    # Forward-compat: solution_dir and entry_point are accepted but not yet
    # wired into the command.  When tools/ncu/ncu_modal.py gains --solution-dir
    # and --entry-point flags, only the command list below needs to change.
    command = [".venv/bin/modal", "run", "tools/ncu/ncu_modal.py"]

    timeout_s = 1800
    result = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_ncu_profile",
    )

    lines = [f"Command: {_command_display(command)}"]
    if result.timed_out:
        lines.append(f"Status: timed out after {timeout_s}s")
    else:
        lines.append(f"Exit code: {result.returncode}")
    output = (result.stdout.strip() or result.stderr.strip() or "(no output)")
    lines.append(output)
    report = "\n".join(lines)
    report = _truncate_output(report, limit=DEFAULT_SHELL_OUTPUT_LIMIT)
    return _prepend_notice(report, result.overflow_notice or "")


@function_tool
async def run_sass_analysis(
    ctx: RunContextWrapper[SharedContext],
    kernel_file: str = "",
    mode: str = "cutedsl",
) -> str:
    """Run SASS instruction set analysis on a kernel via Modal B200.

    Compiles the kernel, extracts the cubin, disassembles it, and returns a
    pipeline analysis with opcode classification and cost estimates.

    Args:
        kernel_file: Path to the kernel file to analyse (project-relative or
            absolute). Defaults to the current kernel in the solution directory.
        mode: Analysis mode — ``cutedsl`` (default) or ``cubin``.
    """
    project_root = _project_root_from_context(ctx)

    if not kernel_file:
        solution_dir = ctx.context.solution_dir or "solution/dsa_attention"
        kernel_file = f"{solution_dir}/kernel.py"

    try:
        target = _resolve_workspace_path(project_root, kernel_file)
    except Exception as exc:
        return f"ERROR: {exc}"

    display = _display_path(target, project_root)
    if not target.exists():
        return f"ERROR: File not found: {display}"

    flag = "--cutedsl" if mode != "cubin" else "--cubin"
    command = [".venv/bin/modal", "run", "tools/sass/dump_sass_modal.py", flag, str(target)]

    timeout_s = 1200
    result = await _run_command(
        project_root=project_root,
        command=command,
        timeout_s=timeout_s,
        source_tool="run_sass_analysis",
    )

    lines = [f"Command: {_command_display(command)}"]
    if result.timed_out:
        lines.append(f"Status: timed out after {timeout_s}s")
    else:
        lines.append(f"Exit code: {result.returncode}")
    output = (result.stdout.strip() or result.stderr.strip() or "(no output)")
    lines.append(output)
    report = "\n".join(lines)
    report = _truncate_output(report, limit=DEFAULT_SHELL_OUTPUT_LIMIT)
    return _prepend_notice(report, result.overflow_notice or "")


# ---------------------------------------------------------------------------
# Collected tool list for agent registration
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    apply_patch_tool,
    web_search_tool,
    web_fetch,
    codex_kernel_assist,
    read_file,
    glob_files,
    grep_search,
    list_directory,
    diff_files,
    run_ncu_profile,
    run_sass_analysis,
    run_synthetic_check,
    run_correctness_check,
    run_full_benchmark,
]

# Subset for the kernel-designer agent: read-only exploration + plan writing,
# no validation/benchmark tools (those belong to the coder).
DESIGNER_TOOLS = [
    apply_patch_tool,
    web_search_tool,
    web_fetch,
    codex_kernel_assist,
    read_file,
    glob_files,
    grep_search,
    list_directory,
]
