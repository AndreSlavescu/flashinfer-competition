from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

from agents import RunContextWrapper
from agents.tool_context import ToolContext

from kernel_agents.context import SharedContext, ToolLimitSettings, tool_limits_for_profile


def resolve_instructions(agent: Any, ctx: SharedContext) -> str:
    """Return the agent instructions string, resolving callables with run context."""
    instructions = agent.instructions
    if callable(instructions):
        return instructions(RunContextWrapper(context=ctx), agent)
    assert isinstance(instructions, str)
    return instructions


def extract_markdown_section(prompt: str, heading: str) -> str:
    """Extract a top-level markdown section from a rendered prompt."""
    start = prompt.index(heading)
    tail = prompt[start:]
    return tail.split("\n\n## ", 1)[0].strip()


def shared_context(
    project_root: Path,
    *,
    quality_profile: str = "legacy",
    codex_worker_mode: str = "off",
    tool_limits: ToolLimitSettings | None = None,
    current_agent_role: str = "planner",
) -> SharedContext:
    """Build a standard SharedContext for tests."""
    resolved_limits = tool_limits or tool_limits_for_profile(quality_profile)  # type: ignore[arg-type]
    return SharedContext(
        project_root=str(project_root),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
        quality_profile=quality_profile,  # type: ignore[arg-type]
        tool_limits=resolved_limits,
        codex_worker_mode=codex_worker_mode,  # type: ignore[arg-type]
        current_agent_role=current_agent_role,
    )


def write_kernel_plan_fixture(
    project_root: Path,
    content: str = "## Kernel Plan\nfixture plan\n",
) -> None:
    """Create a minimal staged round-0 plan fixture."""
    plan_path = project_root / "solution" / "dsa_attention" / "kernel_0_plan.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(content, encoding="utf-8")


async def invoke_tool(
    tool: Any,
    project_root: Path,
    *,
    context: SharedContext | None = None,
    **kwargs: Any,
) -> str:
    """Invoke a FunctionTool with JSON-serialized arguments."""
    arguments = json.dumps(kwargs)
    tool_context = ToolContext(
        context=context or shared_context(project_root),
        tool_name=tool.name,
        tool_call_id="tool-call-1",
        tool_arguments=arguments,
    )
    result = await tool.on_invoke_tool(tool_context, arguments)
    assert isinstance(result, str)
    return result


def codex_nonlocals(tool: object) -> dict[str, Any]:
    """Expose the captured configuration for codex-tool closures."""
    invoker = getattr(tool, "on_invoke_tool")
    invoke_impl = invoker.__dict__["_invoke_tool_impl"]
    return inspect.getclosurevars(invoke_impl).nonlocals


class FakeProcess:
    """Small async subprocess stub used by shell-backed tool tests."""

    def __init__(
        self,
        *,
        stdout: str,
        stderr: str = "",
        returncode: int | None = 0,
        timeout_on_first_communicate: bool = False,
    ) -> None:
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()
        self.returncode = returncode
        self.timeout_on_first_communicate = timeout_on_first_communicate
        self.communicate_calls = 0
        self.pid = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self.timeout_on_first_communicate and self.communicate_calls == 1:
            raise asyncio.TimeoutError
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

