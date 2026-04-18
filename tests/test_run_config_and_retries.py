from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents import SQLiteSession
from agents.run_config import CallModelData, ModelInputData

import main
from kernel_agents.context import (
    DesignerResult,
    ImplementationGraph,
    SharedContext,
    StageSpec,
)
from kernel_agents.kernel_designer import make_kernel_designer
from tests.support import shared_context


def test_make_run_config_defaults_to_public_codex_compaction() -> None:
    default_run_config = main.make_run_config()
    explicit_public = main.make_run_config("public_codex")
    legacy = main.make_run_config("legacy")

    assert default_run_config.call_model_input_filter is main.public_codex_input_filter
    assert explicit_public.call_model_input_filter is main.public_codex_input_filter
    assert legacy.call_model_input_filter is None

    for run_config in (default_run_config, explicit_public):
        assert run_config.model_settings.parallel_tool_calls is False
        assert run_config.model_settings.truncation == "auto"
        assert run_config.model_settings.store is False
        assert run_config.model_settings.response_include == [
            "reasoning.encrypted_content"
        ]
        assert run_config.model_settings.extra_args == {
            "context_management": [
                {"type": "compaction", "compact_threshold": 200000}
            ]
        }


def test_rate_limit_error_detection_matches_retry_hints() -> None:
    tpm_error = Exception(
        "Rate limit reached for gpt-5.4 in organization org-xxx on tokens per min "
        "(TPM): Limit 500000, Used 436369, Requested 74001. Please try again in "
        "1.244s. Visit https://platform.openai.com/account/rate-limits to learn more."
    )

    class _StatusError(Exception):
        status_code = 429

    assert main._is_rate_limit_error(tpm_error) is True
    assert main._is_rate_limit_error(Exception("HTTP 429 Too Many Requests")) is True
    assert main._is_rate_limit_error(_StatusError("boom")) is True
    assert main._is_rate_limit_error(Exception("unrelated failure")) is False
    assert main._rate_limit_retry_after_seconds(tpm_error) == pytest.approx(1.244)
    assert main._rate_limit_retry_after_seconds(Exception("no hint")) is None


@pytest.mark.asyncio
async def test_run_agent_retries_rate_limits_with_saved_session_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = shared_context(Path.cwd(), quality_profile="public_codex")
    rate_limit_error = Exception(
        "Rate limit reached for gpt-5.4 in organization org-xxx on tokens per min "
        "(TPM): Limit 500000, Used 436369, Requested 74001. Please try again in "
        "1.244s. Visit https://platform.openai.com/account/rate-limits to learn more."
    )
    calls: list[dict[str, Any]] = []
    sleep_calls: list[float] = []
    stage_result = object()

    async def _fake_run_agent_once(
        *,
        starting_agent: object,
        input: str | list[dict[str, Any]],
        context: SharedContext,
        max_turns: int,
        verbose: bool,
        session: SQLiteSession | None = None,
    ) -> tuple[object, float]:
        calls.append({"input": input, "context": context, "session": session})
        if len(calls) == 1:
            raise rate_limit_error
        return stage_result, 0.25

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(main, "_run_agent_once", _fake_run_agent_once)
    monkeypatch.setattr(main.asyncio, "sleep", _fake_sleep)

    result, _ = await main._run_agent(
        stage_label="kernel-designer",
        starting_agent=object(),
        input="design the kernel",
        context=ctx,
        max_turns=80,
        verbose=True,
    )

    assert result is stage_result
    assert len(calls) == 2
    assert calls[0]["input"] == "design the kernel"
    assert calls[1]["input"] == []
    assert calls[0]["context"] is ctx
    assert calls[1]["context"] is ctx
    assert isinstance(calls[0]["session"], SQLiteSession)
    assert calls[0]["session"] is calls[1]["session"]
    assert sleep_calls == [
        pytest.approx(1.244 + main.RATE_LIMIT_RETRY_SAFETY_BUFFER_S)
    ]


@pytest.mark.asyncio
async def test_run_agent_non_rate_limits_raise_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = shared_context(Path.cwd(), quality_profile="public_codex")
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
    ctx = shared_context(Path.cwd(), quality_profile="public_codex")
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
    assert len(sleep_calls) == main.STAGE_RATE_LIMIT_MAX_RETRIES


class _FakeRunResult:
    def __init__(self, typed_output: DesignerResult) -> None:
        self.final_output = "{}"
        self._typed_output = typed_output

    def final_output_as(
        self,
        cls: type[Any],
        raise_if_incorrect_type: bool = False,
    ) -> Any:
        if raise_if_incorrect_type and not isinstance(self._typed_output, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(self._typed_output).__name__}"
            )
        return self._typed_output


@pytest.mark.asyncio
async def test_run_staged_designer_repairs_invalid_impl_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = shared_context(tmp_path, quality_profile="public_codex")
    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True, exist_ok=True)
    calls: list[dict[str, Any]] = []

    valid_graph = ImplementationGraph(
        async_pipelines=[],
        stages=[
            StageSpec(
                stage_id="warp0::load_q",
                prerequisites=[],
                outputs=["q_tile_debug matches eager reference"],
                relevant_helpers=["cute.make_tensor"],
                plan_excerpt="## Load Q\nload q details",
            ),
            StageSpec(
                stage_id="warp1::qk_mma",
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
    results = [
        _FakeRunResult(
            DesignerResult(
                plan_file="solution/dsa_attention/kernel_0_plan.md",
                impl_graph=invalid_graph,
                status="success",
            )
        ),
        _FakeRunResult(
            DesignerResult(
                plan_file="solution/dsa_attention/kernel_0_plan.md",
                impl_graph=valid_graph,
                status="success",
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

    _, designer_out, _ = await main._run_staged_designer_with_graph_validation(
        starting_agent=object(),
        initial_input="design the staged graph",
        context=ctx,
        max_turns=80,
        verbose=False,
        solution_dir=solution_dir,
        max_graph_repairs=2,
    )

    invalid_path = main.round0_invalid_impl_graph_path(solution_dir, 1)
    assert designer_out.impl_graph == valid_graph
    assert len(calls) == 2
    assert calls[0]["input"] == "design the staged graph"
    assert "duplicate stage IDs" in calls[1]["input"]
    assert isinstance(calls[0]["session"], SQLiteSession)
    assert calls[0]["session"] is calls[1]["session"]
    assert json.loads(invalid_path.read_text(encoding="utf-8"))["stages"][1]["stage_id"] == "warp0::load_q"


def test_public_codex_input_filter_keeps_latest_compaction_tail() -> None:
    ctx = shared_context(Path.cwd(), quality_profile="public_codex")
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
        agent=make_kernel_designer(context=ctx),
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
