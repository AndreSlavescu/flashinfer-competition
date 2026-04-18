"""Tests for kernel_agents.scoping — hard file-system scope enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kernel_agents.scoping import (
    AGENT_SCOPES,
    AgentFileScope,
    ScopedWorkspaceEditor,
    check_path_allowed,
    make_scope_guardrail,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/fake/project")


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    """Return a real temporary directory to act as project root."""
    return tmp_path


# ---------------------------------------------------------------------------
# check_path_allowed
# ---------------------------------------------------------------------------


class TestCheckPathAllowed:
    def test_directory_prefix_match(self) -> None:
        resolved = PROJECT_ROOT / "references" / "cutlass" / "foo.py"
        allowed, rel = check_path_allowed(resolved, PROJECT_ROOT, ("references/",))
        assert allowed is True
        assert rel == "references/cutlass/foo.py"

    def test_directory_prefix_self(self) -> None:
        """The directory itself (without trailing child) should match."""
        resolved = PROJECT_ROOT / "references"
        allowed, _ = check_path_allowed(resolved, PROJECT_ROOT, ("references/",))
        assert allowed is True

    def test_exact_file_match(self) -> None:
        resolved = PROJECT_ROOT / "KERNEL_CODER_CONTEXT.md"
        allowed, rel = check_path_allowed(
            resolved, PROJECT_ROOT, ("KERNEL_CODER_CONTEXT.md",)
        )
        assert allowed is True
        assert rel == "KERNEL_CODER_CONTEXT.md"

    def test_exact_file_no_directory_match(self) -> None:
        """An exact-file prefix should NOT match children."""
        resolved = PROJECT_ROOT / "KERNEL_CODER_CONTEXT.md" / "sub"
        allowed, _ = check_path_allowed(
            resolved, PROJECT_ROOT, ("KERNEL_CODER_CONTEXT.md",)
        )
        assert allowed is False

    def test_rejects_out_of_scope(self) -> None:
        resolved = PROJECT_ROOT / "main.py"
        allowed, rel = check_path_allowed(resolved, PROJECT_ROOT, ("references/",))
        assert allowed is False
        assert rel == "main.py"

    def test_rejects_outside_project_root(self) -> None:
        resolved = Path("/other/place/file.py")
        allowed, _ = check_path_allowed(resolved, PROJECT_ROOT, ("references/",))
        assert allowed is False

    def test_multiple_prefixes(self) -> None:
        resolved = PROJECT_ROOT / "notes" / "dsa_attention" / "strategy_1.md"
        allowed, _ = check_path_allowed(
            resolved,
            PROJECT_ROOT,
            ("references/", "solution/dsa_attention/", "notes/dsa_attention/"),
        )
        assert allowed is True


# ---------------------------------------------------------------------------
# ScopedWorkspaceEditor
# ---------------------------------------------------------------------------


def _make_operation(
    project_root: Path, path: str
) -> MagicMock:
    """Build a mock ApplyPatchOperation with the given path."""
    op = MagicMock()
    op.path = path
    # ctx_wrapper.context.project_root should resolve to project_root
    op.ctx_wrapper = MagicMock()
    op.ctx_wrapper.context = SimpleNamespace(project_root=str(project_root))
    return op


class TestScopedWorkspaceEditor:
    def test_designer_can_write_plan(self, project_root: Path) -> None:
        (project_root / "solution" / "dsa_attention").mkdir(parents=True)
        inner = MagicMock()
        inner.create_file.return_value = MagicMock(status="completed", output="ok")
        editor = ScopedWorkspaceEditor(inner, "designer")
        op = _make_operation(project_root, "solution/dsa_attention/kernel_0_plan.md")
        result = editor.create_file(op)
        inner.create_file.assert_called_once_with(op)
        assert result.status == "completed"

    def test_designer_cannot_write_kernel(self, project_root: Path) -> None:
        inner = MagicMock()
        editor = ScopedWorkspaceEditor(inner, "designer")
        op = _make_operation(project_root, "solution/dsa_attention/kernel_0.py")
        result = editor.create_file(op)
        inner.create_file.assert_not_called()
        assert result.status == "failed"
        assert "BLOCKED" in result.output
        assert "designer" in result.output

    def test_planner_cannot_write_kernel(self, project_root: Path) -> None:
        inner = MagicMock()
        editor = ScopedWorkspaceEditor(inner, "planner")
        op = _make_operation(project_root, "solution/dsa_attention/kernel_1.py")
        result = editor.update_file(op)
        inner.update_file.assert_not_called()
        assert result.status == "failed"
        assert "BLOCKED" in result.output

    def test_optimizer_can_write_kernel(self, project_root: Path) -> None:
        inner = MagicMock()
        inner.create_file.return_value = MagicMock(status="completed", output="ok")
        editor = ScopedWorkspaceEditor(inner, "optimizer")
        op = _make_operation(project_root, "solution/dsa_attention/kernel_3.py")
        result = editor.create_file(op)
        inner.create_file.assert_called_once_with(op)
        assert result.status == "completed"

    def test_optimizer_cannot_write_strategy(self, project_root: Path) -> None:
        inner = MagicMock()
        editor = ScopedWorkspaceEditor(inner, "optimizer")
        op = _make_operation(project_root, "notes/dsa_attention/strategy_1.md")
        result = editor.create_file(op)
        inner.create_file.assert_not_called()
        assert result.status == "failed"
        assert "BLOCKED" in result.output

    def test_coder_cannot_write_strategy(self, project_root: Path) -> None:
        inner = MagicMock()
        editor = ScopedWorkspaceEditor(inner, "coder")
        op = _make_operation(project_root, "notes/dsa_attention/strategy_1.md")
        result = editor.create_file(op)
        inner.create_file.assert_not_called()
        assert result.status == "failed"

    def test_reviewer_cannot_write_plan(self, project_root: Path) -> None:
        inner = MagicMock()
        editor = ScopedWorkspaceEditor(inner, "reviewer")
        op = _make_operation(project_root, "solution/dsa_attention/kernel_0_plan.md")
        result = editor.update_file(op)
        inner.update_file.assert_not_called()
        assert result.status == "failed"
        assert "Allowed write paths: (none)" in result.output

    def test_delete_checked(self, project_root: Path) -> None:
        inner = MagicMock()
        editor = ScopedWorkspaceEditor(inner, "designer")
        op = _make_operation(project_root, "main.py")
        result = editor.delete_file(op)
        inner.delete_file.assert_not_called()
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# make_scope_guardrail
# ---------------------------------------------------------------------------


def _make_guardrail_data(
    project_root: Path,
    tool_name: str,
    tool_args: dict,
    agent_name: str = "kernel-designer",
) -> MagicMock:
    """Build a mock ToolInputGuardrailData."""
    data = MagicMock()
    data.context.tool_name = tool_name
    data.context.tool_arguments = json.dumps(tool_args)
    # The guardrail calls _project_root_from_context(data.context), which
    # reads data.context.context.project_root.
    data.context.context = SimpleNamespace(project_root=str(project_root))
    data.agent = MagicMock()
    data.agent.name = agent_name
    return data


class TestScopeGuardrail:
    def test_allows_in_scope_read(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("designer")
        data = _make_guardrail_data(
            project_root,
            "read_file",
            {
                "file_path": "references/cutlass/python/CuTeDSL/cutlass/cute/example.py"
            },
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "allow"

    def test_rejects_out_of_scope_read(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("designer")
        data = _make_guardrail_data(
            project_root, "read_file", {"file_path": "main.py"}
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "reject_content"
        assert "BLOCKED" in result.behavior["message"]
        assert "designer" in result.behavior["message"]

    def test_rejects_empty_directory_for_glob(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("coder")
        data = _make_guardrail_data(
            project_root, "glob_files", {"pattern": "**/*.py", "directory": ""}
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "reject_content"
        assert "must specify a directory" in result.behavior["message"]

    def test_rejects_empty_path_for_grep(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("planner")
        data = _make_guardrail_data(
            project_root, "grep_search", {"pattern": "def kernel", "path": ""}
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "reject_content"

    def test_rejects_empty_path_for_list_directory(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("optimizer")
        data = _make_guardrail_data(
            project_root, "list_directory", {"path": ""}
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "reject_content"

    def test_allows_non_file_tool(self, project_root: Path) -> None:
        """Tools that don't access files (e.g. web_fetch) are always allowed."""
        guardrail = make_scope_guardrail("designer")
        data = _make_guardrail_data(
            project_root, "web_fetch", {"url": "https://example.com"}
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "allow"

    def test_planner_can_read_notes(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("planner")
        data = _make_guardrail_data(
            project_root,
            "read_file",
            {"file_path": "notes/dsa_attention/strategy_1.md"},
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "allow"

    def test_reviewer_can_read_shell_dump(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("reviewer")
        data = _make_guardrail_data(
            project_root,
            "read_file",
            {"file_path": "last_shell_dump.txt"},
            agent_name="kernel-stage-reviewer",
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "allow"

    def test_diff_both_paths_checked(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("designer")
        # file_a in scope, file_b out of scope
        data = _make_guardrail_data(
            project_root,
            "diff_files",
            {
                "file_a": "references/cutlass/foo.py",
                "file_b": "main.py",
            },
        )
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "reject_content"

    def test_malformed_args(self, project_root: Path) -> None:
        guardrail = make_scope_guardrail("designer")
        data = MagicMock()
        data.context.tool_name = "read_file"
        data.context.tool_arguments = "not json"
        data.context.context = SimpleNamespace(project_root=str(project_root))
        result = guardrail.guardrail_function(data)
        assert result.behavior["type"] == "reject_content"
        assert "malformed" in result.behavior["message"]


# ---------------------------------------------------------------------------
# AGENT_SCOPES sanity checks
# ---------------------------------------------------------------------------


class TestAgentScopesSanity:
    def test_all_roles_defined(self) -> None:
        assert set(AGENT_SCOPES.keys()) == {
            "designer",
            "reviewer",
            "coder",
            "planner",
            "optimizer",
        }

    @pytest.mark.parametrize("role", ["designer", "reviewer", "coder", "planner", "optimizer"])
    def test_references_readable(self, role: str) -> None:
        """All agents should be able to read the references directory."""
        scope = AGENT_SCOPES[role]
        assert any(p.startswith("references/") for p in scope.read_allow)

    @pytest.mark.parametrize("role", ["designer", "reviewer", "coder", "planner", "optimizer"])
    def test_solution_dir_readable(self, role: str) -> None:
        """All agents should be able to read the solution directory."""
        scope = AGENT_SCOPES[role]
        assert any("solution/dsa_attention" in p for p in scope.read_allow)
