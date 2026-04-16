"""Declarative file-system scoping for per-agent read/write enforcement.

Provides two enforcement layers:
1. ToolInputGuardrail for FunctionTool-based file-access tools (read_file,
   glob_files, grep_search, list_directory, diff_files).
2. ScopedWorkspaceEditor wrapper for ApplyPatchTool write operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.editor import ApplyPatchOperation, ApplyPatchResult
from agents.tool_guardrails import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
)

if TYPE_CHECKING:
    from kernel_agents.tools import WorkspaceEditor

# ---------------------------------------------------------------------------
# Scope configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentFileScope:
    """Declarative read/write allowlist for a single agent role."""

    read_allow: tuple[str, ...]
    """Path prefixes relative to project_root. Trailing ``/`` matches directories."""

    write_allow: tuple[str, ...]
    """Path prefixes relative to project_root. Trailing ``/`` matches directories."""


AGENT_SCOPES: dict[str, AgentFileScope] = {
    "designer": AgentFileScope(
        read_allow=(
            "references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py",
            "references/cutlass/python/CuTeDSL/cutlass/cute/",
            "references/cutlass/python/CuTeDSL/cutlass/pipeline/",
            "references/cutlass/python/CuTeDSL/cutlass/utils/",
            "references/cutlass/examples/python/CuTeDSL/blackwell/",
            "references/cutlass/examples/python/CuTeDSL/notebooks/",
            "references/quack/",
            "solution/dsa_attention/",
        ),
        write_allow=(
            "solution/dsa_attention/kernel_0_plan.md",
        ),
    ),
    "coder": AgentFileScope(
        read_allow=(
            "references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py",
            "references/cutlass/python/CuTeDSL/cutlass/cute/",
            "references/cutlass/python/CuTeDSL/cutlass/pipeline/",
            "references/cutlass/python/CuTeDSL/cutlass/utils/",
            "references/cutlass/examples/python/CuTeDSL/blackwell/",
            "references/cutlass/examples/python/CuTeDSL/notebooks/",
            "references/quack/",
            "solution/dsa_attention/",
            "last_shell_dump.txt",
        ),
        write_allow=(
            "solution/dsa_attention/",
        ),
    ),
    "planner": AgentFileScope(
        read_allow=(
            "references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py",
            "references/cutlass/python/CuTeDSL/cutlass/cute/",
            "references/cutlass/python/CuTeDSL/cutlass/pipeline/",
            "references/cutlass/python/CuTeDSL/cutlass/utils/",
            "references/cutlass/examples/python/CuTeDSL/blackwell/",
            "references/cutlass/examples/python/CuTeDSL/notebooks/",
            "references/quack/",
            "solution/dsa_attention/",
            "notes/dsa_attention/",
            "last_shell_dump.txt",
        ),
        write_allow=(
            "notes/dsa_attention/",
        ),
    ),
    "optimizer": AgentFileScope(
        read_allow=(
            "references/",
            "solution/dsa_attention/",
            "notes/dsa_attention/",
            "last_shell_dump.txt",
        ),
        write_allow=(
            "solution/dsa_attention/",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def check_path_allowed(
    resolved_path: Path,
    project_root: Path,
    allowed_prefixes: tuple[str, ...],
) -> tuple[bool, str]:
    """Check whether *resolved_path* matches any entry in *allowed_prefixes*.

    Returns ``(is_allowed, relative_posix_path)``.

    Prefix rules:
    - Ending with ``/`` matches any path whose relative form starts with that
      prefix (directory scope).
    - Without a trailing ``/`` the relative path must match exactly (single
      file scope).
    """
    try:
        rel = resolved_path.relative_to(project_root)
    except ValueError:
        return False, str(resolved_path)
    rel_str = rel.as_posix()
    for prefix in allowed_prefixes:
        if prefix.endswith("/"):
            if rel_str.startswith(prefix) or rel_str + "/" == prefix:
                return True, rel_str
        else:
            if rel_str == prefix:
                return True, rel_str
    return False, rel_str


# ---------------------------------------------------------------------------
# ScopedWorkspaceEditor — write-scope enforcement for ApplyPatchTool
# ---------------------------------------------------------------------------


def _resolve_path_for_scope_check(
    project_root: Path, raw_path: str
) -> Path:
    """Resolve *raw_path* relative to *project_root* without creating dirs."""
    candidate = Path(raw_path)
    target = candidate if candidate.is_absolute() else (project_root / candidate)
    return target.resolve()


class ScopedWorkspaceEditor:
    """Wraps a :class:`WorkspaceEditor` with per-role write-scope enforcement.

    Every ``create_file`` / ``update_file`` / ``delete_file`` call is checked
    against the role's ``write_allow`` list before delegating to the inner
    editor.  Out-of-scope writes return a *failed* :class:`ApplyPatchResult`
    with a descriptive message.
    """

    def __init__(self, inner: WorkspaceEditor, role: str) -> None:
        self._inner = inner
        self._role = role
        self._scope = AGENT_SCOPES[role]

    # -- internal helpers ---------------------------------------------------

    def _project_root(self, operation: ApplyPatchOperation) -> Path:
        from kernel_agents.tools import _project_root_from_context

        return _project_root_from_context(operation.ctx_wrapper)

    def _check_write(self, operation: ApplyPatchOperation) -> str | None:
        """Return an error string if the write is disallowed, else ``None``."""
        project_root = self._project_root(operation)
        try:
            resolved = _resolve_path_for_scope_check(project_root, operation.path)
        except Exception as exc:
            return str(exc)
        allowed, rel_path = check_path_allowed(
            resolved, project_root, self._scope.write_allow
        )
        if not allowed:
            return (
                f"BLOCKED: {self._role} agent cannot write to '{rel_path}'. "
                f"Allowed write paths: {', '.join(self._scope.write_allow)}"
            )
        return None

    # -- editor interface ---------------------------------------------------

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        err = self._check_write(operation)
        if err:
            return ApplyPatchResult(status="failed", output=err)
        return self._inner.create_file(operation)

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        err = self._check_write(operation)
        if err:
            return ApplyPatchResult(status="failed", output=err)
        return self._inner.update_file(operation)

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        err = self._check_write(operation)
        if err:
            return ApplyPatchResult(status="failed", output=err)
        return self._inner.delete_file(operation)


# ---------------------------------------------------------------------------
# Tool input guardrail factory — read-scope enforcement for FunctionTools
# ---------------------------------------------------------------------------

# Maps tool name -> argument keys that hold file/directory paths.
_PATH_ARGS: dict[str, tuple[str, ...]] = {
    "read_file": ("file_path",),
    "glob_files": ("directory",),
    "grep_search": ("path",),
    "list_directory": ("path",),
    "diff_files": ("file_a", "file_b"),
}

# Tools where an empty path arg means "search the whole repo" and should be
# rejected in favour of an explicit scoped directory.
_REQUIRE_EXPLICIT_DIR: set[str] = {"glob_files", "grep_search", "list_directory"}


def make_scope_guardrail(role: str) -> ToolInputGuardrail[Any]:
    """Create a :class:`ToolInputGuardrail` that enforces read-scope for *role*.

    The returned guardrail inspects the JSON arguments of file-access tools
    and rejects calls whose path arguments fall outside the role's
    ``read_allow`` list.
    """
    scope = AGENT_SCOPES[role]

    def _guardrail_fn(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        tool_name = data.context.tool_name
        path_keys = _PATH_ARGS.get(tool_name)
        if path_keys is None:
            return ToolGuardrailFunctionOutput.allow()

        try:
            args: dict[str, Any] = json.loads(data.context.tool_arguments or "{}")
        except json.JSONDecodeError:
            return ToolGuardrailFunctionOutput.reject_content(
                message="BLOCKED: malformed tool arguments.",
            )

        # Import lazily to avoid circular dependency at module level.
        from kernel_agents.tools import _project_root_from_context

        project_root = _project_root_from_context(data.context)

        for key in path_keys:
            path_val = args.get(key, "")
            if not path_val:
                # Empty path = project root.  For search/listing tools require
                # an explicit directory within scope.
                if tool_name in _REQUIRE_EXPLICIT_DIR:
                    return ToolGuardrailFunctionOutput.reject_content(
                        message=(
                            f"BLOCKED: {role} agent must specify a directory within "
                            f"its scope. Allowed read paths: "
                            f"{', '.join(scope.read_allow)}"
                        ),
                    )
                continue

            try:
                resolved = _resolve_path_for_scope_check(project_root, path_val)
            except Exception:
                return ToolGuardrailFunctionOutput.reject_content(
                    message=f"BLOCKED: cannot resolve path '{path_val}'.",
                )

            allowed, rel_path = check_path_allowed(
                resolved, project_root, scope.read_allow
            )
            if not allowed:
                return ToolGuardrailFunctionOutput.reject_content(
                    message=(
                        f"BLOCKED: {role} agent cannot access '{rel_path}'. "
                        f"Allowed read paths: {', '.join(scope.read_allow)}"
                    ),
                )

        return ToolGuardrailFunctionOutput.allow()

    return ToolInputGuardrail(
        guardrail_function=_guardrail_fn,
        name=f"scope_guard_{role}",
    )
