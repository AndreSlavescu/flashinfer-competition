from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kernel_agents.kernel_coder import make_kernel_coder
from kernel_agents.kernel_designer import make_kernel_designer
from kernel_agents.kernel_optimizer import make_kernel_optimizer
from kernel_agents.kernel_planner import make_kernel_planner
from kernel_agents.scoping import AGENT_SCOPES
from kernel_agents.tools import PROJECT_ROOT, build_tools_for_role, tool_names
from tests.support import (
    codex_nonlocals,
    extract_markdown_section,
    resolve_instructions,
    shared_context,
    write_kernel_plan_fixture,
)


def test_codex_write_workers_and_benchmark_tools_are_role_gated() -> None:
    coder_tools = tool_names(
        build_tools_for_role("coder", codex_worker_mode="coder_optimizer")
    )
    optimizer_tools = tool_names(
        build_tools_for_role("optimizer", codex_worker_mode="coder_optimizer")
    )
    planner_tools = tool_names(
        build_tools_for_role("planner", codex_worker_mode="coder_optimizer")
    )
    designer_tools = tool_names(
        build_tools_for_role("designer", codex_worker_mode="coder_optimizer")
    )

    assert "codex_coder_engineer" in coder_tools
    assert "codex_optimizer_engineer" in optimizer_tools
    assert "codex_coder_engineer" not in designer_tools
    assert "codex_optimizer_engineer" not in planner_tools
    assert "run_full_benchmark" in planner_tools
    assert "run_full_benchmark" not in designer_tools


@pytest.mark.parametrize(
    ("role", "factory"),
    [
        ("designer", make_kernel_designer),
        ("coder", make_kernel_coder),
        ("planner", make_kernel_planner),
        ("optimizer", make_kernel_optimizer),
    ],
)
def test_primary_agent_reference_sections_follow_scope_order(
    tmp_path: Path,
    role: str,
    factory: Any,
) -> None:
    write_kernel_plan_fixture(tmp_path)
    ctx = shared_context(tmp_path)
    prompt = resolve_instructions(factory(context=ctx), ctx)
    references = extract_markdown_section(prompt, "## References")

    assert references.splitlines() == [
        "## References",
        *[f"- `{path}`" for path in AGENT_SCOPES[role].read_allow],
    ]


def test_hardware_block_only_appears_for_designer_and_planner(tmp_path: Path) -> None:
    write_kernel_plan_fixture(tmp_path)
    ctx = shared_context(tmp_path)

    designer_prompt = resolve_instructions(make_kernel_designer(context=ctx), ctx)
    coder_prompt = resolve_instructions(make_kernel_coder(context=ctx), ctx)
    planner_prompt = resolve_instructions(make_kernel_planner(context=ctx), ctx)
    optimizer_prompt = resolve_instructions(make_kernel_optimizer(context=ctx), ctx)

    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in designer_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in planner_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" not in coder_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" not in optimizer_prompt


def test_codex_kernel_assist_uses_role_scoped_workspace_roots() -> None:
    expected_primary = {
        "designer": "references",
        "reviewer": "solution/dsa_attention",
        "coder": "solution/dsa_attention",
        "planner": "solution/dsa_attention",
        "optimizer": "solution/dsa_attention",
    }

    for role, primary in expected_primary.items():
        assist_tool = next(
            tool
            for tool in build_tools_for_role(role)
            if getattr(tool, "name", None) == "codex_kernel_assist"
        )
        assist_nonlocals = codex_nonlocals(assist_tool)
        thread_options = assist_nonlocals["resolved_thread_options"]

        assert Path(thread_options.working_directory) == (PROJECT_ROOT / primary).resolve()

        expected_additional = sorted(
            str((PROJECT_ROOT / prefix.rstrip("/")).resolve())
            for prefix in AGENT_SCOPES[role].read_allow
            if prefix.endswith("/") and prefix.rstrip("/") != primary
        )
        assert sorted(thread_options.additional_directories or []) == expected_additional
