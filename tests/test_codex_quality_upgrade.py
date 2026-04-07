from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from agents.run_config import CallModelData, ModelInputData
from agents.tool_context import ToolContext

import main
from kernel_agents.context import (
    LEGACY_TOOL_LIMITS,
    PUBLIC_CODEX_TOOL_LIMITS,
    SharedContext,
    ToolLimitSettings,
    tool_limits_for_profile,
)
from kernel_agents.kernel_coder import make_kernel_coder
from kernel_agents.kernel_designer import make_kernel_designer
from kernel_agents.kernel_optimizer import make_kernel_optimizer
from kernel_agents.kernel_planner import make_kernel_planner
from kernel_agents.tools import (
    SHELL_OVERFLOW_FILE_NAME,
    build_tools_for_role,
    diff_files,
    glob_files,
    grep_search,
    list_directory,
    read_file,
    run_synthetic_check,
    tool_names,
    web_fetch,
)


def _shared_context(
    project_root: Path,
    *,
    quality_profile: str = "legacy",
    codex_worker_mode: str = "off",
    tool_limits: ToolLimitSettings | None = None,
) -> SharedContext:
    resolved_limits = tool_limits or tool_limits_for_profile(quality_profile)  # type: ignore[arg-type]
    return SharedContext(
        project_root=str(project_root),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
        quality_profile=quality_profile,  # type: ignore[arg-type]
        tool_limits=resolved_limits,
        codex_worker_mode=codex_worker_mode,  # type: ignore[arg-type]
    )


async def _invoke_tool(
    tool: Any,
    project_root: Path,
    *,
    context: SharedContext | None = None,
    **kwargs: Any,
) -> str:
    arguments = json.dumps(kwargs)
    tool_context = ToolContext(
        context=context or _shared_context(project_root),
        tool_name=tool.name,
        tool_call_id="tool-call-1",
        tool_arguments=arguments,
    )
    result = await tool.on_invoke_tool(tool_context, arguments)
    assert isinstance(result, str)
    return result


def _codex_nonlocals(tool: object) -> dict[str, Any]:
    invoker = getattr(tool, "on_invoke_tool")
    invoke_impl = invoker.__dict__["_invoke_tool_impl"]
    return inspect.getclosurevars(invoke_impl).nonlocals


class _FakeProcess:
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int | None = 0) -> None:
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        return None


def test_tool_limit_profiles_match_expected_values() -> None:
    assert tool_limits_for_profile("legacy") == LEGACY_TOOL_LIMITS
    assert tool_limits_for_profile("public_codex") == PUBLIC_CODEX_TOOL_LIMITS
    assert PUBLIC_CODEX_TOOL_LIMITS.shell_output_limit_chars == 100_000
    assert PUBLIC_CODEX_TOOL_LIMITS.search_output_limit_chars == 100_000
    assert PUBLIC_CODEX_TOOL_LIMITS.web_fetch_limit_chars == 100_000
    assert PUBLIC_CODEX_TOOL_LIMITS.read_file_limit_chars == 200_000
    assert PUBLIC_CODEX_TOOL_LIMITS.diff_limit_chars == 200_000
    assert PUBLIC_CODEX_TOOL_LIMITS.search_match_limit == 200
    assert PUBLIC_CODEX_TOOL_LIMITS.glob_match_limit == 500
    assert PUBLIC_CODEX_TOOL_LIMITS.list_directory_cap == 500
    assert PUBLIC_CODEX_TOOL_LIMITS.reference_read_window_lines == 1200
    assert PUBLIC_CODEX_TOOL_LIMITS.grep_max_columns == 400


def test_public_codex_run_config_enables_compaction() -> None:
    run_config = main.make_run_config("public_codex")

    assert run_config.call_model_input_filter is main.public_codex_input_filter
    assert run_config.model_settings.parallel_tool_calls is False
    assert run_config.model_settings.truncation == "auto"
    assert run_config.model_settings.store is False
    assert run_config.model_settings.extra_args == {
        "context_management": [{"type": "compaction", "compact_threshold": 200000}]
    }


def test_public_codex_is_the_default_run_config() -> None:
    run_config = main.make_run_config()

    assert run_config.call_model_input_filter is main.public_codex_input_filter
    assert run_config.model_settings.truncation == "auto"


def test_public_codex_input_filter_keeps_latest_compaction_tail() -> None:
    ctx = _shared_context(Path.cwd(), quality_profile="public_codex")
    agent = make_kernel_designer(context=ctx)
    payload = CallModelData(
        model_data=ModelInputData(
            input=[
                {"type": "message", "role": "user", "content": "first user prompt"},
                {"type": "message", "role": "assistant", "content": "old assistant output"},
                {"type": "compaction", "summary": "older summary"},
                {"type": "message", "role": "assistant", "content": "stale after first compaction"},
                {"type": "message", "role": "user", "content": "second user prompt"},
                {"type": "tool_call", "name": "read_file"},
                {"type": "compaction", "summary": "latest summary"},
                {"type": "message", "role": "assistant", "content": "recent assistant output"},
                {"type": "message", "role": "user", "content": "current user follow-up"},
            ],
            instructions="system guidance",
        ),
        agent=agent,
        context=ctx,
    )

    filtered = main.public_codex_input_filter(payload)

    assert filtered.instructions == "system guidance"
    assert filtered.input == [
        {"type": "message", "role": "user", "content": "first user prompt"},
        {"type": "message", "role": "user", "content": "second user prompt"},
        {"type": "compaction", "summary": "latest summary"},
        {"type": "message", "role": "assistant", "content": "recent assistant output"},
        {"type": "message", "role": "user", "content": "current user follow-up"},
    ]


def test_coder_and_optimizer_include_write_capable_codex_only_when_enabled() -> None:
    base_ctx = _shared_context(Path.cwd(), codex_worker_mode="off")
    coder = make_kernel_coder(context=base_ctx)
    optimizer = make_kernel_optimizer(context=base_ctx)

    assert "codex_coder_engineer" not in tool_names(coder.tools)
    assert "codex_optimizer_engineer" not in tool_names(optimizer.tools)
    assert "run_ncu_profile" not in tool_names(coder.tools)
    assert "run_sass_analysis" not in tool_names(coder.tools)
    assert "run_full_benchmark" not in tool_names(coder.tools)
    assert "run_ncu_profile" not in tool_names(optimizer.tools)
    assert "run_sass_analysis" not in tool_names(optimizer.tools)
    assert "run_full_benchmark" not in tool_names(optimizer.tools)
    assert "codex_coder_engineer" not in coder.instructions
    assert "codex_optimizer_engineer" not in optimizer.instructions

    codex_ctx = _shared_context(Path.cwd(), codex_worker_mode="coder_optimizer")
    coder_with_worker = make_kernel_coder(context=codex_ctx)
    optimizer_with_worker = make_kernel_optimizer(context=codex_ctx)

    assert "codex_coder_engineer" in tool_names(coder_with_worker.tools)
    assert "codex_optimizer_engineer" in tool_names(optimizer_with_worker.tools)
    assert "codex_coder_engineer" in coder_with_worker.instructions
    assert "codex_optimizer_engineer" in optimizer_with_worker.instructions


def test_designer_and_planner_never_receive_write_capable_codex_workers() -> None:
    ctx = _shared_context(Path.cwd(), codex_worker_mode="coder_optimizer")
    designer = make_kernel_designer(context=ctx)
    planner = make_kernel_planner(context=ctx)

    designer_names = tool_names(designer.tools)
    planner_names = tool_names(planner.tools)

    assert "codex_coder_engineer" not in designer_names
    assert "codex_optimizer_engineer" not in designer_names
    assert "codex_coder_engineer" not in planner_names
    assert "codex_optimizer_engineer" not in planner_names
    assert "run_full_benchmark" not in designer_names
    assert "run_full_benchmark" in planner_names
    assert "run_ncu_profile" in planner_names
    assert "run_sass_analysis" in planner_names
    assert "codex_coder_engineer" not in designer.instructions
    assert "codex_optimizer_engineer" not in planner.instructions


def test_only_planner_currently_receives_run_full_benchmark() -> None:
    assert "run_full_benchmark" not in tool_names(build_tools_for_role("designer"))
    assert "run_full_benchmark" not in tool_names(build_tools_for_role("coder"))
    assert "run_full_benchmark" in tool_names(build_tools_for_role("planner"))
    assert "run_full_benchmark" not in tool_names(build_tools_for_role("optimizer"))


def test_agent_instructions_list_actual_registered_tools() -> None:
    ctx = _shared_context(Path.cwd(), codex_worker_mode="coder_optimizer")
    agents = [
        make_kernel_designer(context=ctx),
        make_kernel_coder(context=ctx),
        make_kernel_planner(context=ctx),
        make_kernel_optimizer(context=ctx),
    ]

    for agent in agents:
        names = tool_names(agent.tools)
        for name in names:
            assert name in agent.instructions


def test_codex_worker_tools_use_distinct_thread_keys_and_requested_options() -> None:
    coder_tools = build_tools_for_role(
        "coder",
        codex_worker_mode="coder_optimizer",
        codex_worker_model="gpt-5-codex-custom",
        codex_worker_reasoning_effort="xhigh",
    )
    optimizer_tools = build_tools_for_role(
        "optimizer",
        codex_worker_mode="coder_optimizer",
        codex_worker_model="gpt-5-codex-custom",
        codex_worker_reasoning_effort="xhigh",
    )

    coder_tool = next(tool for tool in coder_tools if getattr(tool, "name", None) == "codex_coder_engineer")
    optimizer_tool = next(
        tool for tool in optimizer_tools if getattr(tool, "name", None) == "codex_optimizer_engineer"
    )

    coder_nonlocals = _codex_nonlocals(coder_tool)
    optimizer_nonlocals = _codex_nonlocals(optimizer_tool)

    assert coder_nonlocals["resolved_run_context_thread_id_key"] == "codex_thread_id_coder_engineer"
    assert optimizer_nonlocals["resolved_run_context_thread_id_key"] == "codex_thread_id_optimizer_engineer"
    assert coder_nonlocals["resolved_options"].persist_session is True
    assert optimizer_nonlocals["resolved_options"].persist_session is True
    assert coder_nonlocals["resolved_options"].use_run_context_thread_id is True
    assert optimizer_nonlocals["resolved_options"].use_run_context_thread_id is True
    assert coder_nonlocals["resolved_thread_options"].sandbox_mode == "workspace-write"
    assert optimizer_nonlocals["resolved_thread_options"].sandbox_mode == "workspace-write"
    assert coder_nonlocals["resolved_thread_options"].working_directory
    assert optimizer_nonlocals["resolved_thread_options"].working_directory
    assert coder_nonlocals["resolved_thread_options"].model == "gpt-5-codex-custom"
    assert optimizer_nonlocals["resolved_thread_options"].model == "gpt-5-codex-custom"
    assert coder_nonlocals["resolved_thread_options"].model_reasoning_effort == "xhigh"
    assert optimizer_nonlocals["resolved_thread_options"].model_reasoning_effort == "xhigh"
    assert coder_nonlocals["resolved_thread_options"].network_access_enabled is False
    assert optimizer_nonlocals["resolved_thread_options"].network_access_enabled is False
    assert coder_nonlocals["resolved_thread_options"].web_search_mode == "disabled"
    assert optimizer_nonlocals["resolved_thread_options"].web_search_mode == "disabled"
    assert coder_nonlocals["resolved_thread_options"].approval_policy == "never"
    assert optimizer_nonlocals["resolved_thread_options"].approval_policy == "never"
    assert coder_nonlocals["resolved_options"].default_turn_options.idle_timeout_seconds == 180
    assert optimizer_nonlocals["resolved_options"].default_turn_options.idle_timeout_seconds == 180


@pytest.mark.asyncio
async def test_public_codex_read_file_uses_1200_line_reference_window(tmp_path: Path) -> None:
    target = tmp_path / "references" / "big_ref.py"
    target.parent.mkdir(parents=True)
    target.write_text("".join(f"line_{i}\n" for i in range(1, 1501)), encoding="utf-8")
    ctx = _shared_context(tmp_path, quality_profile="public_codex")

    result = await _invoke_tool(read_file, tmp_path, context=ctx, file_path="references/big_ref.py")

    assert result.startswith(
        "retrieved trimmed references/big_ref.py:[1]-[1200] of 1500 lines;"
    )
    assert "1200: line_1200" in result
    assert "1201: line_1201" not in result


@pytest.mark.asyncio
async def test_context_aware_limits_propagate_to_grep_glob_list_web_diff_and_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_limits = ToolLimitSettings(
        shell_output_limit_chars=60,
        search_output_limit_chars=60,
        web_fetch_limit_chars=10,
        read_file_limit_chars=50,
        diff_limit_chars=30,
        search_match_limit=2,
        glob_match_limit=3,
        list_directory_cap=3,
        reference_read_window_lines=5,
        grep_max_columns=80,
    )
    ctx = _shared_context(tmp_path, tool_limits=custom_limits)

    grep_target = tmp_path / "matches.txt"
    grep_target.write_text("match_1\nmatch_2\nmatch_3\nmatch_4\n", encoding="utf-8")
    grep_result = await _invoke_tool(grep_search, tmp_path, context=ctx, pattern="match_", path="matches.txt")
    assert grep_result.startswith(
        "retrieved trimmed search results for matches.txt; showing up to the first 2 matches"
    )
    assert "match_3" not in grep_result

    glob_dir = tmp_path / "globbed"
    glob_dir.mkdir()
    for index in range(1, 6):
        (glob_dir / f"file_{index}.py").write_text("pass\n", encoding="utf-8")
    glob_result = await _invoke_tool(glob_files, tmp_path, context=ctx, pattern="*.py", directory="globbed")
    assert glob_result.startswith(
        "retrieved trimmed glob results for globbed; showing up to the first 3 matches"
    )
    assert "file_3.py" in glob_result
    assert "file_4.py" not in glob_result

    list_dir = tmp_path / "listed"
    list_dir.mkdir()
    for index in range(1, 6):
        (list_dir / f"entry_{index}.txt").write_text("x\n", encoding="utf-8")
    list_result = await _invoke_tool(list_directory, tmp_path, context=ctx, path="listed")
    assert list_result.startswith("listed/  (5 entries) — showing first 3")
    assert "entry_3.txt" in list_result
    assert "entry_4.txt" not in list_result

    httpx = pytest.importorskip("httpx")

    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            assert url == "https://example.com/small-cap"
            return _FakeResponse("abcdefghijklmnop")

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    fetch_result = await _invoke_tool(
        web_fetch,
        tmp_path,
        context=ctx,
        url="https://example.com/small-cap",
    )
    assert fetch_result.startswith("retrieved trimmed https://example.com/small-cap:[1]-[10] of 16 chars")

    (tmp_path / "a.py").write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("line1\nchanged2\nchanged3\nchanged4\n", encoding="utf-8")
    diff_result = await _invoke_tool(diff_files, tmp_path, context=ctx, file_a="a.py", file_b="b.py")
    assert diff_result.endswith("... (truncated)")

    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    fake_process = _FakeProcess(stdout=("log\n" * 100) + "SYNTHETIC RESULTS: 1/1 cases passed\n")

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> _FakeProcess:
        return fake_process

    monkeypatch.setattr("kernel_agents.tools.asyncio.create_subprocess_exec", _create_subprocess_exec)
    shell_result = await _invoke_tool(run_synthetic_check, tmp_path, context=ctx)
    overflow_path = tmp_path / SHELL_OVERFLOW_FILE_NAME

    assert shell_result.startswith(
        "retrieved trimmed shell output at 60 chars; full transcript saved to last_shell_overflow.txt"
    )
    assert "Summary: 1/1 synthetic cases passed" in shell_result
    assert overflow_path.exists()
    assert "Source tool: run_synthetic_check" in overflow_path.read_text(encoding="utf-8")
