"""Focused scoping coverage for file read/write enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kernel_agents.scoping import (
    AGENT_SCOPES,
    ScopedWorkspaceEditor,
    check_path_allowed,
    make_scope_guardrail,
)

PROJECT_ROOT = Path("/fake/project")


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.parametrize(
    ("resolved_path", "allowed_prefixes", "expected_allowed", "expected_rel"),
    [
        (
            PROJECT_ROOT / "references" / "cutlass" / "foo.py",
            ("references/",),
            True,
            "references/cutlass/foo.py",
        ),
        (
            PROJECT_ROOT / "KERNEL_CODER_CONTEXT.md",
            ("KERNEL_CODER_CONTEXT.md",),
            True,
            "KERNEL_CODER_CONTEXT.md",
        ),
        (
            PROJECT_ROOT / "main.py",
            ("references/",),
            False,
            "main.py",
        ),
    ],
)
def test_check_path_allowed_matches_expected_prefixes(
    resolved_path: Path,
    allowed_prefixes: tuple[str, ...],
    expected_allowed: bool,
    expected_rel: str,
) -> None:
    allowed, rel = check_path_allowed(resolved_path, PROJECT_ROOT, allowed_prefixes)

    assert allowed is expected_allowed
    assert rel == expected_rel


def test_check_path_allowed_rejects_paths_outside_project_root() -> None:
    allowed, rel = check_path_allowed(
        Path("/other/place/file.py"),
        PROJECT_ROOT,
        ("references/",),
    )

    assert allowed is False
    assert rel == "/other/place/file.py"


def _make_operation(project_root: Path, path: str) -> MagicMock:
    operation = MagicMock()
    operation.path = path
    operation.ctx_wrapper = MagicMock()
    operation.ctx_wrapper.context = SimpleNamespace(project_root=str(project_root))
    return operation


@pytest.mark.parametrize(
    ("role", "method_name", "path", "expected_status"),
    [
        ("designer", "create_file", "solution/dsa_attention/kernel_0_plan.md", "completed"),
        ("reviewer", "update_file", "solution/dsa_attention/kernel_0_plan.md", "failed"),
        ("optimizer", "create_file", "solution/dsa_attention/kernel_3.py", "completed"),
        ("coder", "create_file", "notes/dsa_attention/strategy_1.md", "failed"),
        ("planner", "delete_file", "solution/dsa_attention/kernel_1.py", "failed"),
    ],
)
def test_scoped_workspace_editor_enforces_role_write_permissions(
    project_root: Path,
    role: str,
    method_name: str,
    path: str,
    expected_status: str,
) -> None:
    inner = MagicMock()
    getattr(inner, method_name).return_value = MagicMock(status="completed", output="ok")
    editor = ScopedWorkspaceEditor(inner, role)
    operation = _make_operation(project_root, path)

    result = getattr(editor, method_name)(operation)

    assert result.status == expected_status
    if expected_status == "completed":
        getattr(inner, method_name).assert_called_once_with(operation)
    else:
        getattr(inner, method_name).assert_not_called()
        assert "BLOCKED" in result.output


def _make_guardrail_data(
    project_root: Path,
    tool_name: str,
    tool_args: dict[str, object],
    *,
    agent_name: str = "kernel-designer",
) -> MagicMock:
    data = MagicMock()
    data.context.tool_name = tool_name
    data.context.tool_arguments = json.dumps(tool_args)
    data.context.context = SimpleNamespace(project_root=str(project_root))
    data.agent = MagicMock()
    data.agent.name = agent_name
    return data


@pytest.mark.parametrize(
    ("role", "tool_name", "tool_args", "agent_name"),
    [
        (
            "designer",
            "read_file",
            {
                "file_path": "references/cutlass/python/CuTeDSL/cutlass/cute/example.py"
            },
            "kernel-designer",
        ),
        (
            "reviewer",
            "read_file",
            {"file_path": "last_shell_dump.txt"},
            "kernel-stage-reviewer",
        ),
        (
            "designer",
            "diff_files",
            {
                "file_a": "references/cutlass/python/CuTeDSL/cutlass/cute/example.py",
                "file_b": "solution/dsa_attention/kernel_0_plan.md",
            },
            "kernel-designer",
        ),
        (
            "reviewer",
            "diff_files",
            {
                "file_a": "solution/dsa_attention/kernel_0.py",
                "file_b": "solution/dsa_attention/round0/S0.attempt_01.kernel_0.py",
            },
            "kernel-stage-reviewer",
        ),
    ],
)
def test_scope_guardrail_allows_in_scope_reads(
    project_root: Path,
    role: str,
    tool_name: str,
    tool_args: dict[str, object],
    agent_name: str,
) -> None:
    result = make_scope_guardrail(role).guardrail_function(
        _make_guardrail_data(
            project_root,
            tool_name,
            tool_args,
            agent_name=agent_name,
        )
    )

    assert result.behavior["type"] == "allow"


@pytest.mark.parametrize(
    ("role", "tool_name", "tool_args", "message_fragment"),
    [
        ("designer", "read_file", {"file_path": "main.py"}, "BLOCKED"),
        (
            "designer",
            "diff_files",
            {"file_a": "references/cutlass/foo.py", "file_b": "main.py"},
            "BLOCKED",
        ),
        ("coder", "glob_files", {"pattern": "**/*.py", "directory": ""}, "must specify a directory"),
    ],
)
def test_scope_guardrail_rejects_out_of_scope_or_repo_wide_access(
    project_root: Path,
    role: str,
    tool_name: str,
    tool_args: dict[str, object],
    message_fragment: str,
) -> None:
    result = make_scope_guardrail(role).guardrail_function(
        _make_guardrail_data(project_root, tool_name, tool_args)
    )

    assert result.behavior["type"] == "reject_content"
    assert message_fragment in result.behavior["message"]


def test_scope_guardrail_rejects_malformed_tool_arguments(project_root: Path) -> None:
    data = MagicMock()
    data.context.tool_name = "read_file"
    data.context.tool_arguments = "not json"
    data.context.context = SimpleNamespace(project_root=str(project_root))

    result = make_scope_guardrail("designer").guardrail_function(data)

    assert result.behavior["type"] == "reject_content"
    assert "malformed" in result.behavior["message"]


def test_agent_scopes_keep_all_roles_and_shared_read_roots() -> None:
    assert set(AGENT_SCOPES) == {
        "designer",
        "reviewer",
        "coder",
        "planner",
        "optimizer",
    }

    for scope in AGENT_SCOPES.values():
        assert any(prefix.startswith("references/") for prefix in scope.read_allow)
        assert any("solution/dsa_attention" in prefix for prefix in scope.read_allow)
