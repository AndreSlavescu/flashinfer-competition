from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from agents import RunContextWrapper, SQLiteSession
from agents.run_config import CallModelData, ModelInputData
from agents.tool_context import ToolContext

import main
from kernel_agents.context import (
    DesignerResult,
    ImplementationGraph,
    LEGACY_TOOL_LIMITS,
    PUBLIC_CODEX_TOOL_LIMITS,
    SharedContext,
    StageSpec,
    ToolLimitSettings,
    tool_limits_for_profile,
)
from kernel_agents.kernel_coder import make_kernel_coder
from kernel_agents.kernel_designer import make_kernel_designer
from kernel_agents.kernel_optimizer import make_kernel_optimizer
from kernel_agents.kernel_planner import make_kernel_planner
from kernel_agents.scoping import AGENT_SCOPES
from kernel_agents.tools import (
    PROJECT_ROOT,
    SHELL_OVERFLOW_FILE_NAME,
    build_tools_for_role,
    diff_files,
    glob_files,
    grep_search,
    list_directory,
    read_file,
    run_stage_validation,
    run_synthetic_check,
    tool_names,
    web_fetch,
)


def _resolve_instructions(agent: Any, ctx: SharedContext) -> str:
    """Return agent.instructions as a string, invoking callables with a RunContextWrapper."""
    instructions = agent.instructions
    if callable(instructions):
        return instructions(RunContextWrapper(context=ctx), agent)
    assert isinstance(instructions, str)
    return instructions


def _shared_context(
    project_root: Path,
    *,
    quality_profile: str = "legacy",
    codex_worker_mode: str = "off",
    tool_limits: ToolLimitSettings | None = None,
    current_agent_role: str = "planner",
) -> SharedContext:
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


def _write_kernel_plan_fixture(project_root: Path) -> None:
    plan_path = project_root / "solution" / "dsa_attention" / "kernel_0_plan.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("## Kernel Plan\nfixture plan\n", encoding="utf-8")


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
    assert run_config.model_settings.response_include == ["reasoning.encrypted_content"]
    assert run_config.model_settings.extra_args == {
        "context_management": [{"type": "compaction", "compact_threshold": 200000}]
    }


def test_rate_limit_error_detection_matches_tpm_and_status_signals() -> None:
    tpm_error = Exception(
        "Rate limit reached for gpt-5.4 in organization org-xxx on tokens per min "
        "(TPM): Limit 500000, Used 436369, Requested 74001. Please try again in "
        "1.244s. Visit https://platform.openai.com/account/rate-limits to learn more."
    )
    too_many = Exception("HTTP 429 Too Many Requests")

    class _StatusError(Exception):
        status_code = 429

    assert main._is_rate_limit_error(tpm_error) is True
    assert main._is_rate_limit_error(too_many) is True
    assert main._is_rate_limit_error(_StatusError("boom")) is True
    assert main._is_rate_limit_error(Exception("unrelated failure")) is False

    assert main._rate_limit_retry_after_seconds(tpm_error) == pytest.approx(1.244)
    assert main._rate_limit_retry_after_seconds(Exception("no hint")) is None


@pytest.mark.asyncio
async def test_run_agent_retries_rate_limits_with_saved_session_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _shared_context(Path.cwd(), quality_profile="public_codex")
    stage_result = object()
    calls: list[dict[str, Any]] = []
    sleep_calls: list[float] = []
    rate_limit_error = Exception(
        "Rate limit reached for gpt-5.4 in organization org-xxx on tokens per min "
        "(TPM): Limit 500000, Used 436369, Requested 74001. Please try again in "
        "1.244s. Visit https://platform.openai.com/account/rate-limits to learn more."
    )

    async def _fake_run_agent_once(
        *,
        starting_agent: object,
        input: str | list[dict[str, Any]],
        context: SharedContext,
        max_turns: int,
        verbose: bool,
        session: SQLiteSession | None = None,
    ) -> tuple[object, float]:
        calls.append(
            {
                "starting_agent": starting_agent,
                "input": input,
                "context": context,
                "max_turns": max_turns,
                "verbose": verbose,
                "session": session,
            }
        )
        if len(calls) == 1:
            raise rate_limit_error
        return stage_result, 0.25

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(main, "_run_agent_once", _fake_run_agent_once)
    monkeypatch.setattr(main.asyncio, "sleep", _fake_sleep)

    result, elapsed_s = await main._run_agent(
        stage_label="kernel-designer",
        starting_agent=object(),
        input="design the kernel",
        context=ctx,
        max_turns=80,
        verbose=True,
    )

    assert result is stage_result
    assert elapsed_s >= 0.0
    assert len(calls) == 2
    assert calls[0]["input"] == "design the kernel"
    assert calls[1]["input"] == []
    assert isinstance(calls[0]["session"], SQLiteSession)
    assert calls[0]["session"] is calls[1]["session"]
    assert calls[0]["context"] is ctx
    assert calls[1]["context"] is ctx
    assert sleep_calls == [
        pytest.approx(1.244 + main.RATE_LIMIT_RETRY_SAFETY_BUFFER_S),
    ]


@pytest.mark.asyncio
async def test_run_agent_does_not_retry_non_rate_limit_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _shared_context(Path.cwd(), quality_profile="public_codex")
    calls: list[dict[str, Any]] = []

    async def _fake_run_agent_once(
        *,
        starting_agent: object,
        input: str | list[dict[str, Any]],
        context: SharedContext,
        max_turns: int,
        verbose: bool,
        session: SQLiteSession | None = None,
    ) -> tuple[object, float]:
        calls.append({"input": input, "session": session})
        raise RuntimeError("boom")

    async def _unexpected_sleep(_: float) -> None:
        raise AssertionError("sleep should not be called for non-rate-limit failures")

    monkeypatch.setattr(main, "_run_agent_once", _fake_run_agent_once)
    monkeypatch.setattr(main.asyncio, "sleep", _unexpected_sleep)

    with pytest.raises(RuntimeError, match="boom"):
        await main._run_agent(
            stage_label="kernel-coder",
            starting_agent=object(),
            input="implement the kernel",
            context=ctx,
            max_turns=100,
            verbose=False,
        )

    assert len(calls) == 1
    assert calls[0]["input"] == "implement the kernel"
    assert isinstance(calls[0]["session"], SQLiteSession)


@pytest.mark.asyncio
async def test_run_agent_stops_after_stage_rate_limit_retry_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _shared_context(Path.cwd(), quality_profile="public_codex")
    calls: list[dict[str, Any]] = []
    sleep_calls: list[float] = []
    rate_limit_error = Exception("HTTP 429 Too Many Requests. Please try again in 0.250s.")

    async def _fake_run_agent_once(
        *,
        starting_agent: object,
        input: str | list[dict[str, Any]],
        context: SharedContext,
        max_turns: int,
        verbose: bool,
        session: SQLiteSession | None = None,
    ) -> tuple[object, float]:
        calls.append({"input": input, "session": session})
        raise rate_limit_error

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(main, "_run_agent_once", _fake_run_agent_once)
    monkeypatch.setattr(main.asyncio, "sleep", _fake_sleep)

    with pytest.raises(Exception, match="Too Many Requests"):
        await main._run_agent(
            stage_label="kernel-planner-round-1",
            starting_agent=object(),
            input="benchmark the kernel",
            context=ctx,
            max_turns=40,
            verbose=True,
        )

    assert len(calls) == main.STAGE_RATE_LIMIT_MAX_RETRIES + 1
    assert calls[0]["input"] == "benchmark the kernel"
    assert all(call["input"] == [] for call in calls[1:])
    first_session = calls[0]["session"]
    assert isinstance(first_session, SQLiteSession)
    assert all(call["session"] is first_session for call in calls)
    assert len(sleep_calls) == main.STAGE_RATE_LIMIT_MAX_RETRIES
    assert sleep_calls == [
        pytest.approx(0.250 + main.RATE_LIMIT_RETRY_SAFETY_BUFFER_S)
    ] * main.STAGE_RATE_LIMIT_MAX_RETRIES


@pytest.mark.asyncio
async def test_run_staged_designer_repairs_invalid_impl_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = _shared_context(tmp_path, quality_profile="public_codex")
    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True, exist_ok=True)
    calls: list[dict[str, Any]] = []

    valid_graph = ImplementationGraph(
        async_pipelines=[],
        stages=[
            StageSpec(
                stage_id="warp0::load_q",
                owner_warps=["warp0"],
                prerequisites=[],
                outputs=["q_tile_debug matches eager reference"],
                relevant_helpers=["cute.make_tensor"],
                plan_excerpt="## Load Q\nload q details",
            ),
            StageSpec(
                stage_id="warp1::qk_mma",
                owner_warps=["warp1"],
                prerequisites=["warp0::load_q"],
                outputs=["score_tile_debug matches eager reference"],
                relevant_helpers=["tcgen05.mma"],
                plan_excerpt="## QK Mainloop\nqk details",
            ),
        ],
    )
    invalid_graph = valid_graph.model_copy(
        update={
            "stages": [
                valid_graph.stages[0],
                valid_graph.stages[1].model_copy(update={"stage_id": "warp0::load_q"}),
            ]
        }
    )

    class _FakeRunResult:
        def __init__(self, typed_output: DesignerResult) -> None:
            self.final_output = "{}"
            self._typed_output = typed_output

        def final_output_as(self, cls: type[Any], raise_if_incorrect_type: bool = False) -> Any:
            if raise_if_incorrect_type and not isinstance(self._typed_output, cls):
                raise TypeError(
                    f"Expected {cls.__name__}, got {type(self._typed_output).__name__}"
                )
            return self._typed_output

    results = [
        _FakeRunResult(
            DesignerResult(
                plan_file="solution/dsa_attention/kernel_0_plan.md",
                impl_graph=invalid_graph,
                status="success",
                message="invalid graph first",
            )
        ),
        _FakeRunResult(
            DesignerResult(
                plan_file="solution/dsa_attention/kernel_0_plan.md",
                impl_graph=valid_graph,
                status="success",
                message="valid graph second",
            )
        ),
    ]

    async def _fake_run_agent_once(
        *,
        starting_agent: object,
        input: str | list[dict[str, Any]],
        context: SharedContext,
        max_turns: int,
        verbose: bool,
        session: SQLiteSession | None = None,
    ) -> tuple[object, float]:
        calls.append({"input": input, "session": session, "context": context})
        return results.pop(0), 0.25

    monkeypatch.setattr(main, "_run_agent_once", _fake_run_agent_once)

    raw_result, designer_out, elapsed_s = await main._run_staged_designer_with_graph_validation(
        starting_agent=object(),
        initial_input="design the staged graph",
        context=ctx,
        max_turns=80,
        verbose=False,
        solution_dir=solution_dir,
        max_graph_repairs=2,
    )

    assert isinstance(raw_result, _FakeRunResult)
    assert designer_out.message == "valid graph second"
    assert elapsed_s >= 0.0
    assert len(calls) == 2
    assert calls[0]["input"] == "design the staged graph"
    assert "duplicate stage IDs" in calls[1]["input"]
    assert isinstance(calls[0]["session"], SQLiteSession)
    assert calls[0]["session"] is calls[1]["session"]
    invalid_path = main.round0_invalid_impl_graph_path(solution_dir, 1)
    assert invalid_path.exists()
    assert (
        json.loads(invalid_path.read_text(encoding="utf-8"))["stages"][1]["stage_id"]
        == "warp0::load_q"
    )


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


def test_coder_and_optimizer_include_write_capable_codex_only_when_enabled(tmp_path: Path) -> None:
    _write_kernel_plan_fixture(tmp_path)
    base_ctx = _shared_context(tmp_path, codex_worker_mode="off")
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
    assert "codex_coder_engineer" not in _resolve_instructions(coder, base_ctx)
    assert "codex_optimizer_engineer" not in _resolve_instructions(optimizer, base_ctx)

    codex_ctx = _shared_context(tmp_path, codex_worker_mode="coder_optimizer")
    coder_with_worker = make_kernel_coder(context=codex_ctx)
    optimizer_with_worker = make_kernel_optimizer(context=codex_ctx)

    assert "codex_coder_engineer" in tool_names(coder_with_worker.tools)
    assert "codex_optimizer_engineer" in tool_names(optimizer_with_worker.tools)
    assert "codex_coder_engineer" in _resolve_instructions(coder_with_worker, codex_ctx)
    assert "codex_optimizer_engineer" in _resolve_instructions(
        optimizer_with_worker, codex_ctx
    )


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
    assert "codex_coder_engineer" not in _resolve_instructions(designer, ctx)
    assert "codex_optimizer_engineer" not in _resolve_instructions(planner, ctx)


def test_designer_prompt_matches_current_schema_driven_guidance() -> None:
    ctx = _shared_context(Path.cwd())
    instructions = _resolve_instructions(make_kernel_designer(context=ctx), ctx)

    assert "aligned with the `DesignerResult` schema" in instructions
    assert "find ALL CuTeDSL abstractions and APIs" in instructions
    assert "comprehensive staged implementation graph" in instructions
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in instructions
    assert "Validation strategy" not in instructions
    assert "debug_exports" not in instructions
    assert "target_areas" not in instructions


def test_only_planner_currently_receives_run_full_benchmark() -> None:
    assert "run_full_benchmark" not in tool_names(build_tools_for_role("designer"))
    assert "run_full_benchmark" not in tool_names(build_tools_for_role("coder"))
    assert "run_full_benchmark" in tool_names(build_tools_for_role("planner"))
    assert "run_full_benchmark" not in tool_names(build_tools_for_role("optimizer"))


def test_agent_instructions_list_actual_registered_tools(tmp_path: Path) -> None:
    _write_kernel_plan_fixture(tmp_path)
    ctx = _shared_context(tmp_path, codex_worker_mode="coder_optimizer")
    agents = [
        make_kernel_designer(context=ctx),
        make_kernel_coder(context=ctx),
        make_kernel_planner(context=ctx),
        make_kernel_optimizer(context=ctx),
    ]

    for agent in agents:
        names = tool_names(agent.tools)
        instructions = _resolve_instructions(agent, ctx)
        for name in names:
            assert name in instructions


def test_web_tools_are_removed_from_agent_tool_surfaces_and_prompt_lists(tmp_path: Path) -> None:
    _write_kernel_plan_fixture(tmp_path)
    ctx = _shared_context(tmp_path, codex_worker_mode="coder_optimizer")
    agents = [
        make_kernel_designer(context=ctx),
        make_kernel_coder(context=ctx),
        make_kernel_planner(context=ctx),
        make_kernel_optimizer(context=ctx),
    ]

    for agent in agents:
        names = tool_names(agent.tools)
        instructions = _resolve_instructions(agent, ctx)
        assert "web_search" not in names
        assert "web_fetch" not in names
        assert "web_search:" not in instructions
        assert "web_fetch:" not in instructions


def test_hardware_block_appears_for_designer_and_planner_only(tmp_path: Path) -> None:
    _write_kernel_plan_fixture(tmp_path)
    ctx = _shared_context(tmp_path)
    designer_prompt = _resolve_instructions(make_kernel_designer(context=ctx), ctx)
    coder_prompt = _resolve_instructions(make_kernel_coder(context=ctx), ctx)
    planner_prompt = _resolve_instructions(make_kernel_planner(context=ctx), ctx)
    optimizer_prompt = _resolve_instructions(make_kernel_optimizer(context=ctx), ctx)

    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in designer_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in planner_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" not in coder_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" not in optimizer_prompt


def test_narrowed_scopes_still_include_required_reference_paths() -> None:
    expected_roles = ("designer", "coder", "planner")

    for role in expected_roles:
        read_allow = AGENT_SCOPES[role].read_allow
        assert "references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py" in read_allow
        assert "references/quack/" in read_allow
        assert "solution/dsa_attention/" in read_allow


def test_codex_kernel_assist_uses_role_scoped_workspace_roots() -> None:
    expected_primary = {
        "designer": "references",
        "coder": "solution/dsa_attention",
        "planner": "solution/dsa_attention",
        "optimizer": "solution/dsa_attention",
    }

    for role, primary in expected_primary.items():
        assist_tool = next(
            tool
            for tool in build_tools_for_role(role)  # type: ignore[arg-type]
            if getattr(tool, "name", None) == "codex_kernel_assist"
        )
        assist_nonlocals = _codex_nonlocals(assist_tool)
        resolved_thread_options = assist_nonlocals["resolved_thread_options"]

        assert Path(resolved_thread_options.working_directory) == (PROJECT_ROOT / primary).resolve()

        expected_additional = sorted(
            str((PROJECT_ROOT / prefix.rstrip("/")).resolve())
            for prefix in AGENT_SCOPES[role].read_allow
            if prefix.endswith("/") and prefix.rstrip("/") != primary
        )
        actual_additional = sorted(resolved_thread_options.additional_directories or [])
        assert actual_additional == expected_additional

        file_only_entries = [
            str((PROJECT_ROOT / prefix).resolve())
            for prefix in AGENT_SCOPES[role].read_allow
            if not prefix.endswith("/")
        ]
        assert resolved_thread_options.working_directory not in file_only_entries
        for entry in file_only_entries:
            assert entry not in actual_additional


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
async def test_glob_files_filters_parent_traversal_matches(tmp_path: Path) -> None:
    references_dir = tmp_path / "references"
    references_dir.mkdir()
    (tmp_path / "main.py").write_text("def outside():\n    pass\n", encoding="utf-8")

    designer_glob = next(
        tool for tool in build_tools_for_role("designer") if getattr(tool, "name", None) == "glob_files"
    )
    ctx = _shared_context(tmp_path, current_agent_role="designer")

    result = await _invoke_tool(
        designer_glob,
        tmp_path,
        context=ctx,
        pattern="../*.py",
        directory="references",
    )

    assert result == "No files found."


@pytest.mark.asyncio
async def test_glob_files_filters_recursive_parent_traversal_matches(tmp_path: Path) -> None:
    (tmp_path / "references" / "cutlass").mkdir(parents=True)
    (tmp_path / "main.py").write_text("def outside():\n    pass\n", encoding="utf-8")

    designer_glob = next(
        tool for tool in build_tools_for_role("designer") if getattr(tool, "name", None) == "glob_files"
    )
    ctx = _shared_context(tmp_path, current_agent_role="designer")

    result = await _invoke_tool(
        designer_glob,
        tmp_path,
        context=ctx,
        pattern="**/../../main.py",
        directory="references",
    )

    assert result == "No files found."


@pytest.mark.asyncio
async def test_glob_files_filters_repo_external_matches(tmp_path: Path) -> None:
    passwd = Path("/etc/passwd")
    if not passwd.exists():
        pytest.skip("/etc/passwd not available in this environment")

    (tmp_path / "references").mkdir()
    designer_glob = next(
        tool for tool in build_tools_for_role("designer") if getattr(tool, "name", None) == "glob_files"
    )
    ctx = _shared_context(tmp_path, current_agent_role="designer")

    result = await _invoke_tool(
        designer_glob,
        tmp_path,
        context=ctx,
        pattern="../../../../../etc/passwd",
        directory="references",
    )

    assert result == "No files found."


@pytest.mark.asyncio
async def test_glob_files_returns_normalized_allowed_paths(tmp_path: Path) -> None:
    target = (
        tmp_path
        / "references"
        / "cutlass"
        / "python"
        / "CuTeDSL"
        / "cutlass"
        / "cute"
        / "example.py"
    )
    target.parent.mkdir(parents=True)
    target.write_text("pass\n", encoding="utf-8")

    designer_glob = next(
        tool for tool in build_tools_for_role("designer") if getattr(tool, "name", None) == "glob_files"
    )
    ctx = _shared_context(tmp_path, current_agent_role="designer")

    result = await _invoke_tool(
        designer_glob,
        tmp_path,
        context=ctx,
        pattern="**/*.py",
        directory="references/cutlass/python/CuTeDSL/cutlass/cute",
    )

    assert "references/cutlass/python/CuTeDSL/cutlass/cute/example.py" in result
    assert ".." not in result


@pytest.mark.asyncio
async def test_glob_files_requires_valid_current_agent_role(tmp_path: Path) -> None:
    target = tmp_path / "references" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("pass\n", encoding="utf-8")

    result = await _invoke_tool(
        glob_files,
        tmp_path,
        context=_shared_context(tmp_path, current_agent_role=""),
        pattern="*.py",
        directory="references",
    )

    assert result == "ERROR: Scoped tool call is missing a valid current_agent_role."


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
    ctx = _shared_context(tmp_path, tool_limits=custom_limits, current_agent_role="designer")

    grep_target = tmp_path / "matches.txt"
    grep_target.write_text("match_1\nmatch_2\nmatch_3\nmatch_4\n", encoding="utf-8")
    grep_result = await _invoke_tool(grep_search, tmp_path, context=ctx, pattern="match_", path="matches.txt")
    assert grep_result.startswith(
        "retrieved trimmed search results for matches.txt; showing up to the first 2 matches"
    )
    assert "match_3" not in grep_result

    glob_dir = (
        tmp_path
        / "references"
        / "cutlass"
        / "python"
        / "CuTeDSL"
        / "cutlass"
        / "cute"
        / "globbed"
    )
    glob_dir.mkdir(parents=True)
    for index in range(1, 6):
        (glob_dir / f"file_{index}.py").write_text("pass\n", encoding="utf-8")
    glob_result = await _invoke_tool(
        glob_files,
        tmp_path,
        context=ctx,
        pattern="*.py",
        directory="references/cutlass/python/CuTeDSL/cutlass/cute/globbed",
    )
    assert glob_result.startswith(
        "retrieved trimmed glob results for references/cutlass/python/CuTeDSL/cutlass/cute/globbed; showing up to the first 3 matches"
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


@pytest.mark.asyncio
async def test_run_stage_validation_reports_stage_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _shared_context(tmp_path, quality_profile="legacy", tool_limits=LEGACY_TOOL_LIMITS)
    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    fake_process = _FakeProcess(stdout=("log\n" * 100) + "SYNTHETIC RESULTS: 1/1 cases passed\n")

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> _FakeProcess:
        return fake_process

    monkeypatch.setattr("kernel_agents.tools.asyncio.create_subprocess_exec", _create_subprocess_exec)
    shell_result = await _invoke_tool(
        run_stage_validation,
        tmp_path,
        context=ctx,
        stage_id="warp1::qk_mma",
    )

    assert "Stage: warp1::qk_mma" in shell_result
    assert "Entry point: kernel_0.py::prefix_validation_harness" in shell_result
    assert "Summary: 1/1 synthetic cases passed" in shell_result
