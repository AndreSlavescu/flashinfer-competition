"""Shared tool definitions for all kernel generation agents.

Uses SDK built-in tools where available (WebSearchTool, ShellTool, ApplyPatchTool).
Custom @function_tool only for capabilities with no built-in equivalent.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from agents import (
    ApplyPatchTool,
    RunContextWrapper,
    ShellTool,
    WebSearchTool,
    function_tool,
)
from agents.editor import ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult
from agents.tool import ShellCallOutcome, ShellCommandOutput, ShellCommandRequest, ShellResult

from kernel_agents.context import SharedContext


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resolve(ctx: RunContextWrapper[SharedContext], path: str) -> Path:
    """Resolve a path relative to project root if not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(ctx.context.project_root) / p
    return p


# ---------------------------------------------------------------------------
# Built-in: ShellTool — local executor
# ---------------------------------------------------------------------------

async def _shell_executor(request: ShellCommandRequest) -> ShellResult:
    """Execute shell commands locally with timeout support."""
    timeout_ms = request.data.action.timeout_ms or 600_000  # default 10 min
    timeout_s = timeout_ms / 1000

    # Determine working directory from context if available
    cwd = None
    if request.ctx_wrapper and hasattr(request.ctx_wrapper, "context"):
        cwd = request.ctx_wrapper.context.project_root

    outputs: list[ShellCommandOutput] = []
    for command in request.data.action.commands:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
            outputs.append(ShellCommandOutput(
                command=command,
                stdout=stdout_bytes.decode(errors="replace")[:200_000],
                stderr=stderr_bytes.decode(errors="replace")[:50_000],
                outcome=ShellCallOutcome(type="exit", exit_code=proc.returncode),
            ))
        except asyncio.TimeoutError:
            outputs.append(ShellCommandOutput(
                command=command,
                stdout="",
                stderr=f"ERROR: Command timed out after {timeout_s:.0f}s",
                outcome=ShellCallOutcome(type="timeout"),
            ))
        except Exception as e:
            outputs.append(ShellCommandOutput(
                command=command,
                stdout="",
                stderr=f"ERROR: {e}",
                outcome=ShellCallOutcome(type="exit", exit_code=1),
            ))

    return ShellResult(output=outputs)


shell_tool = ShellTool(
    executor=_shell_executor,
    environment={"type": "local"},
)


# ---------------------------------------------------------------------------
# Built-in: ApplyPatchTool — local file editor
# ---------------------------------------------------------------------------

class _LocalEditor:
    """ApplyPatchEditor implementation that applies diffs on local disk."""

    def create_file(self, op: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            p = Path(op.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            content = op.diff or ""
            p.write_text(content)
            return ApplyPatchResult(status="completed", output=f"Created {p}")
        except Exception as e:
            return ApplyPatchResult(status="failed", output=str(e))

    def update_file(self, op: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            p = Path(op.path)
            if not p.exists():
                return ApplyPatchResult(status="failed", output=f"File not found: {p}")
            if op.diff:
                # Apply unified diff via patch command
                proc = subprocess.run(
                    ["patch", "--no-backup-if-mismatch", str(p)],
                    input=op.diff,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    return ApplyPatchResult(
                        status="failed",
                        output=f"patch failed: {proc.stderr or proc.stdout}",
                    )
            return ApplyPatchResult(status="completed", output=f"Updated {p}")
        except Exception as e:
            return ApplyPatchResult(status="failed", output=str(e))

    def delete_file(self, op: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            p = Path(op.path)
            if p.exists():
                p.unlink()
            return ApplyPatchResult(status="completed", output=f"Deleted {p}")
        except Exception as e:
            return ApplyPatchResult(status="failed", output=str(e))


apply_patch_tool = ApplyPatchTool(editor=_LocalEditor())


# ---------------------------------------------------------------------------
# Built-in: WebSearchTool
# ---------------------------------------------------------------------------

web_search_tool = WebSearchTool()


# ---------------------------------------------------------------------------
# Custom tools — no built-in equivalent
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
        file_path: Absolute or project-relative path.
        start_line: First line to read (1-based). 0 = from beginning.
        end_line: Last line to read (inclusive). 0 = to end.
    """
    p = _resolve(ctx, file_path)
    try:
        text = p.read_text()
    except FileNotFoundError:
        return f"ERROR: File not found: {p}"
    except Exception as e:
        return f"ERROR: {e}"

    lines = text.splitlines(keepends=True)
    s = max(0, (start_line - 1) if start_line > 0 else 0)
    e = end_line if end_line > 0 else len(lines)
    selected = lines[s:e]
    numbered = [f"{s + i + 1}: {line}" for i, line in enumerate(selected)]
    result = "".join(numbered)
    if len(result) > 200_000:
        return result[:200_000] + "\n... (truncated at 200k chars)"
    return result


@function_tool
async def glob_files(
    ctx: RunContextWrapper[SharedContext],
    pattern: str,
    directory: str = "",
) -> str:
    """Find files matching a glob pattern (e.g. '**/*.py', 'solution/**/*.cu').

    Args:
        pattern: Glob pattern.
        directory: Directory to search (default: project root). Absolute or relative.
    """
    d = _resolve(ctx, directory) if directory else Path(ctx.context.project_root)
    try:
        matches = sorted(d.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        if not matches:
            return "No files found."
        lines = [str(m) for m in matches[:200]]
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"


@function_tool
async def grep_search(
    ctx: RunContextWrapper[SharedContext],
    pattern: str,
    path: str = "",
    file_glob: str = "",
) -> str:
    """Search file contents with a regex pattern (uses ripgrep).

    Args:
        pattern: Regex pattern to search for.
        path: File or directory to search in (default: project root).
        file_glob: Optional file filter (e.g. '*.py').
    """
    search_path = str(_resolve(ctx, path)) if path else ctx.context.project_root
    cmd = ["rg", "-n", "--max-count=100", "--max-columns=200"]
    if file_glob:
        cmd.extend(["--glob", file_glob])
    cmd.extend([pattern, search_path])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        out = stdout.decode(errors="replace")
        if not out.strip():
            return "No matches found."
        if len(out) > 100_000:
            out = out[:100_000] + "\n... (truncated)"
        return out
    except FileNotFoundError:
        # rg not available — fall back to grep
        fallback = ["grep", "-rn"]
        if file_glob:
            fallback.extend(["--include", file_glob])
        fallback.extend([pattern, search_path])
        try:
            result = subprocess.run(fallback, capture_output=True, text=True, timeout=30)
            return result.stdout[:100_000] or "No matches found."
        except Exception as e:
            return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


@function_tool
async def web_fetch(
    ctx: RunContextWrapper[SharedContext],
    url: str,
) -> str:
    """Fetch content from a URL and return the text body (truncated to 50k chars).

    Args:
        url: The URL to fetch.
    """
    try:
        import httpx
    except ImportError:
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode(errors="replace")[:50_000]
        except Exception as e:
            return f"ERROR: {e}"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text[:50_000]
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Collected tool list for agent registration
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    # Built-in SDK tools
    shell_tool,
    apply_patch_tool,
    web_search_tool,
    # Custom function tools
    read_file,
    glob_files,
    grep_search,
    web_fetch,
]
