"""Shared tool definitions for all kernel generation agents.

Uses SDK built-in tools where available (ApplyPatchTool, WebSearchTool,
codex_tool). Repo inspection, benchmark, and profiling wrappers are custom
function tools.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Literal

from agents import (
    ApplyPatchTool,
    RunContextWrapper,
    WebSearchTool,
    apply_diff,
    function_tool,
)
from agents.editor import ApplyPatchOperation, ApplyPatchResult
from agents.extensions.experimental.codex import ThreadOptions, TurnOptions, codex_tool
from agents.tool import ShellCallOutcome, ShellCommandOutput

from kernel_agents.context import (
    LEGACY_TOOL_LIMITS,
    CodexWorkerMode,
    CodexWorkerReasoningEffort,
    SharedContext,
    ToolLimitSettings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_OUTPUT_LIMIT = LEGACY_TOOL_LIMITS.search_output_limit_chars
DEFAULT_READ_FILE_LIMIT = LEGACY_TOOL_LIMITS.read_file_limit_chars
DEFAULT_WEB_FETCH_LIMIT = LEGACY_TOOL_LIMITS.web_fetch_limit_chars
DEFAULT_SEARCH_MATCH_LIMIT = LEGACY_TOOL_LIMITS.search_match_limit
REFERENCE_READ_WINDOW_LINES = LEGACY_TOOL_LIMITS.reference_read_window_lines
DEFAULT_DIFF_LIMIT = LEGACY_TOOL_LIMITS.diff_limit_chars
DEFAULT_GLOB_MATCH_LIMIT = LEGACY_TOOL_LIMITS.glob_match_limit
DEFAULT_LIST_DIRECTORY_CAP = LEGACY_TOOL_LIMITS.list_directory_cap
DEFAULT_GREP_MAX_COLUMNS = LEGACY_TOOL_LIMITS.grep_max_columns
DEFAULT_BENCH_LANGUAGE = "python"
DEFAULT_STAGE_VALIDATION_ENTRY_POINT = "kernel_0.py::prefix_validation_harness"
SHELL_DUMP_FILE_NAME = "last_shell_dump.txt"
DEFAULT_SHELL_DUMP_CONTEXT_LINES = 200


def _shell_dump_tail_banner(max_lines: int, *, char_limit: int | None = None) -> str:
    if char_limit is None:
        trimmed_at = f"to the last {max_lines} lines"
    else:
        trimmed_at = f"to the last {max_lines} lines and {char_limit} chars"
    return (
        f"retrieved trimmed shell output {trimmed_at}; "
        f"full transcript saved to {SHELL_DUMP_FILE_NAME}. "
        f"Use read_file or grep_search on {SHELL_DUMP_FILE_NAME} if you need more detail."
    )

SHELL_DUMP_TAIL_BANNER = _shell_dump_tail_banner(DEFAULT_SHELL_DUMP_CONTEXT_LINES)


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


def _tool_limits_from_context(ctx: RunContextWrapper[SharedContext] | None) -> ToolLimitSettings:
    """Resolve active tool limits from run context or fall back to the legacy profile."""
    if ctx is not None:
        context = getattr(ctx, "context", None)
        tool_limits = getattr(context, "tool_limits", None)
        if isinstance(tool_limits, ToolLimitSettings):
            return tool_limits
    return LEGACY_TOOL_LIMITS


def _current_scoped_role_from_context(
    ctx: RunContextWrapper[SharedContext] | None,
) -> str:
    """Resolve the active scoped role or fail closed for role-dependent tools."""
    if ctx is None:
        raise RuntimeError("Scoped tool call is missing run context.")

    context = getattr(ctx, "context", None)
    role = getattr(context, "current_agent_role", "")

    from kernel_agents.scoping import AGENT_SCOPES

    if role not in AGENT_SCOPES:
        raise RuntimeError("Scoped tool call is missing a valid current_agent_role.")
    return str(role)


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


def _format_search_success(output: str, search_path: str, limits: ToolLimitSettings) -> str:
    """Render successful search output with trim notices when caps are hit."""
    notices: list[str] = []
    match_lines = output.splitlines()
    if len(match_lines) >= limits.search_match_limit:
        notices.append(
            (
                f"retrieved trimmed search results for {search_path}; showing up to the first "
                f"{limits.search_match_limit} matches. Refine pattern, path, or file_glob to continue."
            )
        )

    was_char_truncated = len(output) > limits.search_output_limit_chars
    rendered = _truncate_output(
        output,
        limit=limits.search_output_limit_chars,
        suffix="\n... (search output truncated)",
    )
    if was_char_truncated:
        notices.append(
            (
                f"retrieved trimmed search results for {search_path} at "
                f"{limits.search_output_limit_chars} chars; refine pattern, path, or file_glob to continue."
            )
        )
    return _prepend_notice(rendered, "\n".join(notices))


def _format_search_result(
    tool_name: str,
    returncode: int,
    stdout: bytes | str,
    stderr: bytes | str,
    search_path: str,
    limits: ToolLimitSettings,
) -> str:
    """Normalize grep/rg subprocess results into trace-friendly strings."""
    out = stdout.decode("utf-8", errors="replace") if isinstance(stdout, (bytes, bytearray)) else stdout
    err = stderr.decode("utf-8", errors="replace") if isinstance(stderr, (bytes, bytearray)) else stderr

    if returncode == 0:
        return _format_search_success(out, search_path, limits)
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


def _shell_dump_path(project_root: Path) -> Path:
    """Resolve the stable shell dump transcript path."""
    return project_root / SHELL_DUMP_FILE_NAME


def _render_shell_transcript_block(output: ShellCommandOutput) -> str:
    """Render a single shell output entry into a plain-text transcript block."""
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


def _build_shell_dump_transcript(
    *,
    source_tool: str,
    working_directory: Path,
    outputs: Sequence[ShellCommandOutput],
) -> str:
    """Build the full plain-text transcript written to the shell dump file."""
    lines = [
        f"Source tool: {source_tool}",
        f"Working directory: {working_directory}",
        f"Dump file: {SHELL_DUMP_FILE_NAME}",
        "This file is overwritten by the next shell-like tool call.",
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


def _write_shell_dump(project_root: Path, transcript: str) -> None:
    """Persist the current shell transcript to the stable dump file."""
    dump_path = _shell_dump_path(project_root)
    dump_path.write_text(transcript, encoding="utf-8")


def _write_shell_dump_for_outputs(
    *,
    project_root: Path,
    source_tool: str,
    working_directory: Path,
    outputs: Sequence[ShellCommandOutput],
) -> None:
    """Write the current shell transcript to the stable dump file."""
    transcript = _build_shell_dump_transcript(
        source_tool=source_tool,
        working_directory=working_directory,
        outputs=outputs,
    )
    _write_shell_dump(project_root, transcript)


def _format_shell_context_window(
    text: str,
    *,
    max_lines: int,
    char_limit: int,
) -> str:
    """Keep the latest shell context window while preserving dump-search guidance."""
    lines = text.splitlines()
    was_line_trimmed = len(lines) > max_lines
    if was_line_trimmed:
        lines = lines[-max_lines:]
    rendered = "\n".join(lines)
    was_char_trimmed = len(rendered) > char_limit
    rendered = _truncate_output(rendered, limit=char_limit)

    if was_line_trimmed or was_char_trimmed:
        return _prepend_notice(
            rendered,
            _shell_dump_tail_banner(
                max_lines,
                char_limit=char_limit if was_char_trimmed else None,
            ),
        )
    return rendered


def _format_modal_shell_output(
    output: ShellCommandOutput,
    *,
    limits: ToolLimitSettings,
) -> str:
    """Render Modal-backed tool output as recent raw shell context."""
    return _format_shell_context_window(
        _render_shell_transcript_block(output),
        max_lines=DEFAULT_SHELL_DUMP_CONTEXT_LINES,
        char_limit=limits.shell_output_limit_chars,
    )

async def _run_command(
    *,
    project_root: Path,
    command: Sequence[str],
    timeout_s: int,
    source_tool: str,
) -> ShellCommandOutput:
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
    _write_shell_dump_for_outputs(
        project_root=project_root,
        source_tool=source_tool,
        working_directory=project_root,
        outputs=[output],
    )
    return output


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

_CODEX_ASSIST_PRIMARY_DIRS: dict[str, str] = {
    "designer": "references",
    "reviewer": "solution/dsa_attention",
    "coder": "solution/dsa_attention",
    "planner": "solution/dsa_attention",
    "optimizer": "solution/dsa_attention",
}


def _build_codex_kernel_assist(role: str) -> object:
    """Create a read-only Codex helper constrained to the role's directory scope."""
    from kernel_agents.scoping import AGENT_SCOPES

    directory_roots = [
        prefix.rstrip("/")
        for prefix in AGENT_SCOPES[role].read_allow
        if prefix.endswith("/")
    ]
    primary_dir = _CODEX_ASSIST_PRIMARY_DIRS[role]
    additional_directories = [
        str((PROJECT_ROOT / rel_path).resolve())
        for rel_path in directory_roots
        if rel_path != primary_dir
    ]
    return codex_tool(
        name="codex_kernel_assist",
        sandbox_mode="read-only",
        working_directory=str((PROJECT_ROOT / primary_dir).resolve()),
        persist_session=True,
        default_thread_options=ThreadOptions(
            model="gpt-5.4",
            model_reasoning_effort="low",
            network_access_enabled=False,
            web_search_mode="disabled",
            approval_policy="never",
            additional_directories=additional_directories or None,
        ),
    )


# Backward-compatible static for ad hoc imports. Role-specific tool wiring should
# use build_tools_for_role(...) so Codex assist matches the active agent scope.
codex_kernel_assist = _build_codex_kernel_assist("planner")


def _build_codex_worker_tool(
    *,
    name: str,
    run_context_thread_id_key: str,
    model: str,
    reasoning_effort: CodexWorkerReasoningEffort,
) -> object:
    """Create a write-capable Codex worker tool for bounded coding tasks."""
    return codex_tool(
        name=name,
        sandbox_mode="workspace-write",
        working_directory=str(PROJECT_ROOT),
        persist_session=True,
        use_run_context_thread_id=True,
        run_context_thread_id_key=run_context_thread_id_key,
        default_thread_options=ThreadOptions(
            model=model,
            model_reasoning_effort=reasoning_effort,
            network_access_enabled=False,
            web_search_mode="disabled",
            approval_policy="never",
        ),
        default_turn_options=TurnOptions(idle_timeout_seconds=180),
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
    limits = _tool_limits_from_context(ctx)
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
        and total_lines > limits.reference_read_window_lines
    ):
        start_index = 0
        end_index = limits.reference_read_window_lines
        notices.append(
            (
                f"retrieved trimmed {display_path}:[1]-[{end_index}] of {total_lines} lines; "
                "request start_line/end_line for another section."
            )
        )

    selected = lines[start_index:end_index]
    numbered = [f"{start_index + idx + 1}: {line}" for idx, line in enumerate(selected)]
    rendered_text = "".join(numbered)
    was_truncated = len(rendered_text) > limits.read_file_limit_chars
    rendered = _truncate_output(
        rendered_text,
        limit=limits.read_file_limit_chars,
        suffix=f"\n... (truncated at {limits.read_file_limit_chars} chars)",
    )
    if was_truncated:
        notices.append(
            (
                f"retrieved trimmed {display_path}:[{start_index + 1}]-[{min(end_index, total_lines)}] "
                f"at {limits.read_file_limit_chars} chars; request a narrower line range if you need more."
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
    limits = _tool_limits_from_context(ctx)
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
        role = _current_scoped_role_from_context(ctx)
    except Exception as exc:
        return f"ERROR: {exc}"

    from kernel_agents.scoping import AGENT_SCOPES, check_path_allowed

    read_allow = AGENT_SCOPES[role].read_allow

    try:
        matches: list[str] = []
        for path in search_dir.glob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            allowed, rel_path = check_path_allowed(resolved, project_root, read_allow)
            if allowed:
                matches.append(rel_path)
    except Exception as exc:
        return f"ERROR: {exc}"

    if not matches:
        return "No files found."
    capped = sorted(set(matches))
    if len(capped) > limits.glob_match_limit:
        shown = "\n".join(capped[:limits.glob_match_limit])
        return _prepend_notice(
            shown,
            (
                f"retrieved trimmed glob results for {display_dir}; showing up to the first "
                f"{limits.glob_match_limit} matches. Refine pattern or directory to continue."
            ),
        )
    return "\n".join(capped)


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
    limits = _tool_limits_from_context(ctx)
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
    cmd = [
        "rg",
        "-n",
        f"--max-count={limits.search_match_limit}",
        f"--max-columns={limits.grep_max_columns}",
    ]
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
        return _format_search_result("rg", proc.returncode, stdout, stderr, search_path, limits)
    except FileNotFoundError:
        fallback = ["grep", "-ErnI", "-m", str(limits.search_match_limit)]
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
                limits,
            )
        except Exception as exc:
            return f"ERROR: {exc}"
    except Exception as exc:
        return f"ERROR: {exc}"

# ---------------------------------------------------------------------------
# Custom workflow tools
# ---------------------------------------------------------------------------

@function_tool
async def run_synthetic_check(
    ctx: RunContextWrapper[SharedContext],
    solution_dir: str = "",
    entry_point: str = "kernel.py::kernel",
    include_all_files: bool = False,
    rebuild_fixture: bool = False,
) -> str:
    """Run the fast synthetic correctness sweep and return recent raw shell context."""
    limits = _tool_limits_from_context(ctx)
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

    timeout_s = 120
    output = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_synthetic_check",
    )
    return _format_modal_shell_output(output, limits=limits)


@function_tool
async def run_stage_validation(
    ctx: RunContextWrapper[SharedContext],
    stage_id: str,
    solution_dir: str = "",
    include_all_files: bool = False,
    rebuild_fixture: bool = False,
) -> str:
    """Run the synthetic stage validation entry point and return recent raw shell context."""
    limits = _tool_limits_from_context(ctx)
    try:
        solution_path, solution_rel = _resolve_solution_dir(ctx, solution_dir)
    except Exception as exc:
        return f"ERROR: {exc}"

    if not solution_path.exists():
        return f"ERROR: Solution directory not found: {solution_rel}"

    entry_point = DEFAULT_STAGE_VALIDATION_ENTRY_POINT
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

    timeout_s = 120
    output = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_stage_validation",
    )
    return _format_modal_shell_output(output, limits=limits)


@function_tool
async def run_correctness_check(
    ctx: RunContextWrapper[SharedContext],
    solution_dir: str = "",
    entry_point: str = "kernel.py::kernel",
    lang: str = DEFAULT_BENCH_LANGUAGE,
) -> str:
    """Run the Modal correctness-only benchmark flow and return recent raw shell context."""
    limits = _tool_limits_from_context(ctx)
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

    timeout_s = 120
    output = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_correctness_check",
    )
    return _format_modal_shell_output(output, limits=limits)


@function_tool
async def run_full_benchmark(
    ctx: RunContextWrapper[SharedContext],
    solution_dir: str = "",
    entry_point: str = "kernel.py::kernel",
    lang: str = DEFAULT_BENCH_LANGUAGE,
) -> str:
    """Run the full Modal benchmark flow and return recent raw shell context."""
    limits = _tool_limits_from_context(ctx)
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

    timeout_s = 300
    output = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_full_benchmark",
    )
    return _format_modal_shell_output(output, limits=limits)

# ---------------------------------------------------------------------------
# Custom exact-URL fallback
# ---------------------------------------------------------------------------

@function_tool
async def web_fetch(
    ctx: RunContextWrapper[SharedContext],
    url: str,
) -> str:
    """Fetch content from a URL and return the text body with active-profile truncation."""
    limits = _tool_limits_from_context(ctx)

    try:
        # Keep this import lazy so urllib remains a real fallback.
        import httpx
    except ImportError:
        import urllib.request

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                if len(text) <= limits.web_fetch_limit_chars:
                    return text
                shown = text[:limits.web_fetch_limit_chars]
                return _prepend_notice(
                    shown,
                    (
                        f"retrieved trimmed {url}:[1]-[{limits.web_fetch_limit_chars}] "
                        f"of {len(text)} chars"
                    ),
                )
        except Exception as exc:
            return f"ERROR: {exc}"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            if len(resp.text) <= limits.web_fetch_limit_chars:
                return resp.text
            shown = resp.text[:limits.web_fetch_limit_chars]
            return _prepend_notice(
                shown,
                (
                    f"retrieved trimmed {url}:[1]-[{limits.web_fetch_limit_chars}] "
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
    limits = _tool_limits_from_context(ctx)
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

    cap = limits.list_directory_cap
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
    limits = _tool_limits_from_context(ctx)
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
    return _truncate_output("".join(diff), limit=limits.diff_limit_chars)


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

    Returns recent raw shell context from the profiling run. Use
    ``last_shell_dump.txt`` for the full transcript when the returned context
    is trimmed.

    Args:
        solution_dir: Solution directory (default: from context). Reserved for
            future parameterisation of the NCU entrypoint.
        entry_point: Kernel entry point (default kernel.py::kernel). Reserved
            for future parameterisation of the NCU entrypoint.
    """
    # Forward-compat: solution_dir and entry_point are accepted but not yet
    # wired into the command.  When tools/ncu/ncu_modal.py gains --solution-dir
    # and --entry-point flags, only the command list below needs to change.
    limits = _tool_limits_from_context(ctx)
    command = [".venv/bin/modal", "run", "tools/ncu/ncu_modal.py"]

    timeout_s = 1800
    output = await _run_command(
        project_root=_project_root_from_context(ctx),
        command=command,
        timeout_s=timeout_s,
        source_tool="run_ncu_profile",
    )
    return _format_modal_shell_output(output, limits=limits)


@function_tool
async def run_sass_analysis(
    ctx: RunContextWrapper[SharedContext],
    kernel_file: str = "",
    mode: str = "cutedsl",
) -> str:
    """Run SASS instruction set analysis on a kernel via Modal B200.

    Compiles the kernel, extracts the cubin, disassembles it, and returns
    recent raw shell context from that analysis. Use ``last_shell_dump.txt``
    for the full transcript when the returned context is trimmed.

    Args:
        kernel_file: Path to the kernel file to analyse (project-relative or
            absolute). Defaults to the current kernel in the solution directory.
        mode: Analysis mode — ``cutedsl`` (default) or ``cubin``.
    """
    project_root = _project_root_from_context(ctx)
    limits = _tool_limits_from_context(ctx)

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
    output = await _run_command(
        project_root=project_root,
        command=command,
        timeout_s=timeout_s,
        source_tool="run_sass_analysis",
    )
    return _format_modal_shell_output(output, limits=limits)


# ---------------------------------------------------------------------------
# Collected tool list for agent registration
# ---------------------------------------------------------------------------

AgentRole = Literal["designer", "reviewer", "coder", "planner", "optimizer"]

# -- Unscoped tool pools (used only by backward-compatible statics below) ---

_BASE_REPO_TOOLS = [
    apply_patch_tool,
    codex_kernel_assist,
    read_file,
    glob_files,
    grep_search,
    list_directory,
]

_VALIDATION_TOOLS = [
    diff_files,
    run_stage_validation,
    run_synthetic_check,
    run_correctness_check,
]

_PLANNER_ANALYSIS_TOOLS = [
    run_ncu_profile,
    run_sass_analysis,
    run_full_benchmark,
]

# -- FunctionTools that carry a read-scope guardrail ------------------------

_FILE_ACCESS_TOOLS = (read_file, glob_files, grep_search, list_directory)


def _scoped_file_tools(
    guardrail: object,
) -> list[object]:
    """Return copies of the file-access FunctionTools with *guardrail* attached."""
    from agents.tool import FunctionTool

    scoped: list[object] = []
    for tool in _FILE_ACCESS_TOOLS:
        assert isinstance(tool, FunctionTool)
        existing = tool.tool_input_guardrails or []
        scoped.append(
            dataclass_replace(tool, tool_input_guardrails=[guardrail] + list(existing))
        )
    return scoped


def _scoped_diff(guardrail: object) -> object:
    """Return a copy of :data:`diff_files` with *guardrail* attached."""
    from agents.tool import FunctionTool

    assert isinstance(diff_files, FunctionTool)
    existing = diff_files.tool_input_guardrails or []
    return dataclass_replace(
        diff_files, tool_input_guardrails=[guardrail] + list(existing)
    )


def build_tools_for_role(
    role: AgentRole,
    *,
    codex_worker_mode: CodexWorkerMode = "off",
    codex_worker_model: str = "gpt-5-codex",
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort = "high",
) -> list[object]:
    """Build the tool surface for a specific agent role.

    Every file-access tool is stamped with a per-role read-scope guardrail,
    and each role gets its own :class:`ApplyPatchTool` backed by a
    :class:`ScopedWorkspaceEditor` that enforces write scoping.
    """
    from kernel_agents.scoping import ScopedWorkspaceEditor, make_scope_guardrail

    scope_guardrail = make_scope_guardrail(role)

    # Per-role ApplyPatchTool with write-scope enforcement.
    scoped_editor = ScopedWorkspaceEditor(WorkspaceEditor(), role)
    role_patch_tool = ApplyPatchTool(editor=scoped_editor)

    # Assemble the tool list.
    tools: list[object] = [
        role_patch_tool,
        _build_codex_kernel_assist(role),
    ]
    tools.extend(_scoped_file_tools(scope_guardrail))

    if role in {"coder", "optimizer", "planner"}:
        tools.append(_scoped_diff(scope_guardrail))
        tools.extend([run_synthetic_check, run_correctness_check])
        if role == "coder":
            tools.append(run_stage_validation)

    if role == "planner":
        tools.extend(_PLANNER_ANALYSIS_TOOLS)

    if codex_worker_mode == "coder_optimizer":
        if role == "coder":
            # The Codex SDK/CLI does not yet provide a clean read-root/write-root
            # split for write-capable sessions, so worker scoping remains a known
            # limitation outside this hardening pass.
            tools.append(
                _build_codex_worker_tool(
                    name="codex_coder_engineer",
                    run_context_thread_id_key="codex_thread_id_coder_engineer",
                    model=codex_worker_model,
                    reasoning_effort=codex_worker_reasoning_effort,
                )
            )
        elif role == "optimizer":
            # The Codex SDK/CLI does not yet provide a clean read-root/write-root
            # split for write-capable sessions, so worker scoping remains a known
            # limitation outside this hardening pass.
            tools.append(
                _build_codex_worker_tool(
                    name="codex_optimizer_engineer",
                    run_context_thread_id_key="codex_thread_id_optimizer_engineer",
                    model=codex_worker_model,
                    reasoning_effort=codex_worker_reasoning_effort,
                )
            )

    return tools


def tool_names(tools: Sequence[object]) -> list[str]:
    """Return the registered tool names in order."""
    return [str(getattr(tool, "name", "")) for tool in tools if getattr(tool, "name", None)]


# Backward-compatible exports for tests and ad hoc imports. New agent wiring
# should call build_tools_for_role(...) instead of relying on these statics.
ALL_TOOLS = build_tools_for_role("planner")
DESIGNER_TOOLS = build_tools_for_role("designer")
