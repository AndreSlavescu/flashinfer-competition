from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agents import AgentOutputSchema

import main
from kernel_agents.context import (
    AsyncPipelineSpec,
    DesignerResult,
    ImplementationGraph,
    PlannerResult,
    Round0StageResult,
    RoundRecord,
    SharedContext,
    StageReviewResult,
    StageSpec,
)
from kernel_agents.kernel_stage_coder import make_round0_stage_coder
from kernel_agents.kernel_stage_fixer import make_round0_stage_fixer
from kernel_agents.kernel_stage_reviewer import make_round0_stage_reviewer
from kernel_agents.scoping import AGENT_SCOPES
from tests.support import extract_markdown_section


def _sample_impl_graph() -> ImplementationGraph:
    return ImplementationGraph(
        async_pipelines=[
            AsyncPipelineSpec(
                pipeline_id="P0",
                type="PipelineTmaUmma",
                producer_warp="warp0",
                consumer_warp="warp1",
                num_stages=1,
                payload="Q tile",
            )
        ],
        stages=[
            StageSpec(
                stage_id="warp0::load_q",
                prerequisites=[],
                outputs=["q_tile_debug matches the eager reference for the active tile"],
                relevant_helpers=["cute.make_tensor", "blackwell load example"],
                plan_excerpt="## Load Q\nload q details",
            ),
            StageSpec(
                stage_id="warp1::qk_mma",
                prerequisites=["warp0::load_q"],
                outputs=["score_tile_debug matches the eager fp32 QK tile"],
                relevant_helpers=["tcgen05.mma", "pipeline example"],
                plan_excerpt="## QK Mainloop\nqk details",
            ),
        ],
    )


def _single_stage_impl_graph() -> ImplementationGraph:
    return ImplementationGraph(
        async_pipelines=[],
        stages=[
            StageSpec(
                stage_id="warp0::load_q",
                prerequisites=[],
                outputs=["q_tile_debug matches the eager reference for the active tile"],
                relevant_helpers=["cute.make_tensor", "blackwell load example"],
                plan_excerpt="## Load Q\nload q details",
            ),
        ],
    )


def _stage_result(
    stage_id: str,
    *,
    final_correctness_verified: bool = False,
) -> Round0StageResult:
    return Round0StageResult(
        stage_id=stage_id,
        stage_output_validation_report="stage output ok",
        stage_output_verified=True,
        final_correctness_verified=final_correctness_verified,
        correctness_check_report="correct",
        status="success",
        message=f"{stage_id} ok",
    )


def _review(stage_id: str, action: str) -> StageReviewResult:
    return StageReviewResult(stage_id=stage_id, action=action, message=action)


@pytest.mark.parametrize(
    ("result_type", "expected_fields"),
    [
        (DesignerResult, {"plan_file", "impl_graph", "status"}),
        (
            Round0StageResult,
            {
                "stage_id",
                "stage_output_validation_report",
                "stage_output_verified",
                "final_correctness_verified",
                "correctness_check_report",
                "status",
                "message",
            },
        ),
        (StageReviewResult, {"stage_id", "action", "message"}),
        (
            PlannerResult,
            {
                "latency_ms",
                "bottleneck",
                "strategy_summary",
                "strategy_file",
                "is_new_best",
                "ncu_metrics",
            },
        ),
    ],
)
def test_result_models_use_top_level_structured_output(
    result_type: type[Any],
    expected_fields: set[str],
) -> None:
    schema = AgentOutputSchema(result_type).json_schema()

    assert set(schema["properties"]) == expected_fields
    assert set(schema["required"]) == expected_fields
    assert schema["additionalProperties"] is False
    assert "response" not in schema["properties"]


class _FakeTypedRunResult:
    def __init__(self, typed_output: Any) -> None:
        self.final_output = "{}"
        self._typed_output = typed_output
        self.calls: list[tuple[type[Any], bool]] = []

    def final_output_as(
        self,
        cls: type[Any],
        raise_if_incorrect_type: bool = False,
    ) -> Any:
        self.calls.append((cls, raise_if_incorrect_type))
        if raise_if_incorrect_type and not isinstance(self._typed_output, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(self._typed_output).__name__}"
            )
        return self._typed_output


def test_require_structured_output_uses_sdk_typed_accessor() -> None:
    typed_output = DesignerResult(
        plan_file="solution/dsa_attention/kernel_0_plan.md",
        impl_graph=_single_stage_impl_graph(),
        status="success",
    )
    result = _FakeTypedRunResult(typed_output)

    extracted = main.require_structured_output(result, DesignerResult)

    assert extracted == typed_output
    assert result.calls == [(DesignerResult, True)]


def test_require_structured_output_rejects_wrong_type() -> None:
    result = _FakeTypedRunResult("not a typed result")

    with pytest.raises(TypeError, match="Expected DesignerResult, got str"):
        main.require_structured_output(result, DesignerResult)


def test_state_round_trip_preserves_round0_fields(tmp_path: Path) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.current_round = 0
    ctx.best_latency_ms = 1.23
    ctx.best_round = 4
    ctx.history = [
        RoundRecord(
            round_num=1,
            kernel_file="kernel_1.py",
            latency_ms=1.5,
            correctness=True,
            strategy_summary="baseline",
            bottleneck="memory",
        )
    ]
    ctx.round0_mode = "staged"
    ctx.round0_impl_graph_file = "solution/dsa_attention/round0/kernel_0_impl_graph.json"
    ctx.round0_stage_history = [_stage_result("warp0::load_q")]
    ctx.round0_review_history = [_review("warp0::load_q", "continue_next_stage")]

    state_path = tmp_path / "loop_state.json"
    main.save_state(ctx, state_path)

    restored = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    last_round = main.load_state(restored, state_path)

    assert last_round == 0
    assert restored.round0_impl_graph_file == ctx.round0_impl_graph_file
    assert [item.stage_id for item in restored.round0_stage_history] == ["warp0::load_q"]
    assert [item.action for item in restored.round0_review_history] == [
        "continue_next_stage"
    ]


def test_load_state_tolerates_missing_round0_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "loop_state.json"
    state_path.write_text(
        json.dumps(
            {
                "current_round": 2,
                "best_latency_ms": 9.5,
                "best_round": 1,
                "model_name": "gpt-5.4",
                "quality_profile": "public_codex",
                "codex_worker_mode": "off",
                "history": [],
            }
        ),
        encoding="utf-8",
    )

    restored = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    last_round = main.load_state(restored, state_path)

    assert last_round == 2
    assert restored.round0_impl_graph_file == ""
    assert restored.round0_stage_history == []
    assert restored.round0_review_history == []


def test_stage_payload_builders_keep_only_runtime_fields(tmp_path: Path) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0.py").write_text(
        "def kernel():\n    return 'current'\n",
        encoding="utf-8",
    )

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    previous_result = _stage_result("warp0::load_q")
    current_result = _stage_result("warp1::qk_mma")
    ctx.round0_stage_history = [previous_result, current_result]
    ctx.round0_review_history = [_review("warp0::load_q", "continue_next_stage")]

    previous_snapshot = main.round0_stage_kernel_snapshot_path(
        solution_dir,
        "warp0::load_q",
        1,
    )
    previous_snapshot.parent.mkdir(parents=True, exist_ok=True)
    previous_snapshot.write_text(
        "def kernel():\n    return 'previous'\n",
        encoding="utf-8",
    )

    coder_payload = main._build_stage_coder_payload(
        graph=_sample_impl_graph(),
        stage_id="warp1::qk_mma",
        is_final_stage=False,
    )
    fixer_payload = main._build_stage_fixer_payload(
        graph=_sample_impl_graph(),
        stage_id="warp1::qk_mma",
        is_final_stage=False,
        last_review=_review("warp1::qk_mma", "retry_same_stage"),
    )
    review_payload = main._build_stage_review_payload(
        ctx=ctx,
        solution_dir=solution_dir,
        graph=_sample_impl_graph(),
        stage_result=current_result,
        stage_attempt=1,
        is_final_stage=True,
    )

    assert set(coder_payload) == {
        "current_stage",
        "prerequisite_stages",
        "is_final_stage",
    }
    assert fixer_payload["last_review"] == {
        "action": "retry_same_stage",
        "message": "retry_same_stage",
    }
    assert set(review_payload) == {
        "current_stage",
        "stage_results_history",
        "review_history",
        "file_diffs",
        "is_final_stage",
    }
    assert review_payload["file_diffs"]["diff_status"] == "available"
    assert review_payload["file_diffs"]["previous_kernel_snapshot"] == (
        "solution/dsa_attention/round0/warp0_load_q.attempt_01.kernel_0.py"
    )
    assert "-    return 'previous'" in review_payload["file_diffs"]["unified_diff"]
    assert "+    return 'current'" in review_payload["file_diffs"]["unified_diff"]


def test_stage_agent_prompts_render_dynamic_sections_and_scope_references(
    tmp_path: Path,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0_plan.md").write_text("# Plan\nplan body\n", encoding="utf-8")
    (solution_dir / "kernel_0.py").write_text("def kernel():\n    return 'current'\n", encoding="utf-8")

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )

    stage_coder = make_round0_stage_coder(context=ctx)
    coder_static = main._resolve_agent_instructions_text(stage_coder, ctx)
    assert "{..## Current Stage Specifications..}" in coder_static
    main._apply_stage_coder_prompt_sections(
        ctx,
        main._build_stage_coder_payload(
            graph=_sample_impl_graph(),
            stage_id="warp1::qk_mma",
            is_final_stage=False,
        ),
    )
    coder_prompt = main._resolve_agent_instructions_text(stage_coder, ctx)
    assert "{..## Current Stage Specifications..}" not in coder_prompt
    assert "## Current Stage Specifications" in coder_prompt
    assert "Return structured output matching `Round0StageResult`." in coder_prompt
    assert extract_markdown_section(coder_prompt, "## References").splitlines() == [
        "## References",
        *[f"- `{path}`" for path in AGENT_SCOPES["coder"].read_allow],
    ]

    stage_fixer = make_round0_stage_fixer(context=ctx)
    main._apply_stage_fixer_prompt_sections(
        ctx,
        main._build_stage_fixer_payload(
            graph=_sample_impl_graph(),
            stage_id="warp1::qk_mma",
            is_final_stage=True,
            last_review=_review("warp1::qk_mma", "retry_same_stage"),
        ),
    )
    fixer_prompt = main._resolve_agent_instructions_text(stage_fixer, ctx)
    assert "## Last Reviewer Feedback" in fixer_prompt
    assert '"action": "retry_same_stage"' in fixer_prompt

    reviewer = make_round0_stage_reviewer(context=ctx)
    stage_result = _stage_result("warp1::qk_mma")
    ctx.round0_stage_history = [stage_result]
    ctx.round0_review_history = [_review("warp0::load_q", "continue_next_stage")]
    reviewer_payload = main._build_stage_review_payload(
        ctx=ctx,
        solution_dir=solution_dir,
        graph=_sample_impl_graph(),
        stage_result=stage_result,
        stage_attempt=1,
        is_final_stage=False,
    )
    main._apply_stage_reviewer_prompt_sections(ctx, reviewer_payload)
    reviewer_prompt = main._resolve_agent_instructions_text(reviewer, ctx)
    assert "## File Diffs" in reviewer_prompt
    assert 'action="retry_same_stage"' in reviewer_prompt
    assert 'action="revise_design_then_retry"' in reviewer_prompt
    assert 'action="continue_next_stage"' in reviewer_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in reviewer_prompt
    assert extract_markdown_section(reviewer_prompt, "## References").splitlines() == [
        "## References",
        *[f"- `{path}`" for path in AGENT_SCOPES["reviewer"].read_allow],
    ]


class _FakeAgent:
    def __init__(self, instructions: object) -> None:
        self.instructions = instructions


def _prepare_staged_round0_workspace(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graph: ImplementationGraph,
) -> tuple[SharedContext, Path, Path]:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0_plan.md").write_text("# Plan\nplan body\n", encoding="utf-8")
    (solution_dir / "kernel_0.py").write_text("def kernel():\n    return 'current'\n", encoding="utf-8")

    graph_path = solution_dir / "round0" / "kernel_0_impl_graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps(graph.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_impl_graph_file = str(graph_path.relative_to(tmp_path))
    return ctx, solution_dir, tmp_path / "loop_state.json"


class _FakeStageRunResult:
    def __init__(self, typed_output: Any) -> None:
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


def test_dump_all_prompts_includes_stage_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    def _fake_agent_factory(**_: Any) -> _FakeAgent:
        return _FakeAgent("static prompt")

    monkeypatch.setattr(main, "make_kernel_designer", _fake_agent_factory)
    monkeypatch.setattr(main, "make_kernel_coder", _fake_agent_factory)
    monkeypatch.setattr(main, "make_round0_stage_coder", _fake_agent_factory)
    monkeypatch.setattr(main, "make_round0_stage_fixer", _fake_agent_factory)
    monkeypatch.setattr(main, "make_round0_stage_reviewer", _fake_agent_factory)
    monkeypatch.setattr(main, "make_kernel_planner", _fake_agent_factory)
    monkeypatch.setattr(main, "make_kernel_optimizer", _fake_agent_factory)

    output_dir = tmp_path / "prompts"
    main.dump_all_prompts(
        output_dir=output_dir,
        model="gpt-5.4",
        coder_model="gpt-5.4",
        designer_model="gpt-5.4",
        quality_profile="public_codex",
        codex_worker_mode="off",
        codex_worker_model="gpt-5-codex",
        codex_worker_reasoning_effort="high",
        coder_reasoning_effort="high",
        coder_verbosity="low",
        designer_reasoning_effort="high",
        designer_verbosity="low",
        reviewer_verbosity="low",
        planner_reasoning_effort="high",
        optimizer_reasoning_effort="high",
    )

    assert (output_dir / "static" / "kernel_stage_coder.md").read_text(
        encoding="utf-8"
    ) == "static prompt"
    assert (output_dir / "static" / "kernel_stage_fixer.md").read_text(
        encoding="utf-8"
    ) == "static prompt"
    assert (output_dir / "static" / "kernel_stage_reviewer.md").read_text(
        encoding="utf-8"
    ) == "static prompt"


def test_dump_stage_runtime_prompt_artifacts_writes_and_reuses_files(
    tmp_path: Path,
) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    prompt_dump = main._make_prompt_dump_config(tmp_path / "prompts")
    agent = _FakeAgent("runtime prompt")

    main._dump_stage_runtime_prompt_artifacts(
        prompt_dump=prompt_dump,
        agent=agent,
        ctx=ctx,
        stage_id="warp1::qk_mma",
        attempt=2,
        role_name="coder",
        payload={"step": "coder"},
    )

    coder_system = prompt_dump.runtime_dir / "stage_warp1_qk_mma.attempt_02.coder.system.md"
    coder_input = prompt_dump.runtime_dir / "stage_warp1_qk_mma.attempt_02.coder.input.json"
    assert coder_system.read_text(encoding="utf-8") == "runtime prompt"
    assert json.loads(coder_input.read_text(encoding="utf-8")) == {"step": "coder"}

    coder_input.write_text('{"step":"old"}', encoding="utf-8")
    main._dump_stage_runtime_prompt_artifacts(
        prompt_dump=prompt_dump,
        agent=agent,
        ctx=ctx,
        stage_id="warp1::qk_mma",
        attempt=2,
        role_name="coder",
        payload={"step": "new"},
        reuse_if_present=True,
    )
    assert json.loads(coder_input.read_text(encoding="utf-8")) == {"step": "old"}


def test_run_round0_staged_first_attempt_uses_coder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, solution_dir, state_path = _prepare_staged_round0_workspace(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        graph=_single_stage_impl_graph(),
    )
    stage_labels: list[str] = []

    async def _fake_run_agent(**kwargs: Any) -> tuple[_FakeStageRunResult, float]:
        stage_labels.append(kwargs["stage_label"])
        if kwargs["stage_label"].startswith("kernel-stage-coder-"):
            return _FakeStageRunResult(
                _stage_result("warp0::load_q", final_correctness_verified=True)
            ), 0.1
        if kwargs["stage_label"].startswith("kernel-stage-reviewer-"):
            return _FakeStageRunResult(
                _review("warp0::load_q", "continue_next_stage")
            ), 0.1
        raise AssertionError(f"Unexpected stage label: {kwargs['stage_label']}")

    monkeypatch.setattr(main, "_run_agent", _fake_run_agent)

    result = asyncio.run(
        main._run_round0_staged(
            ctx=ctx,
            solution_dir=solution_dir,
            state_path=state_path,
            designer=_FakeAgent("designer"),
            stage_coder=_FakeAgent("coder"),
            stage_fixer=_FakeAgent("fixer"),
            stage_reviewer=_FakeAgent("reviewer"),
            designer_max_turns=1,
            designer_verbose=False,
            coder_max_turns=1,
            coder_verbose=False,
            skip_designer=True,
            resume=False,
            prompt_dump=None,
        )
    )

    assert stage_labels[0].startswith("kernel-stage-coder-")
    assert all("kernel-stage-fixer-" not in label for label in stage_labels)
    assert result.status == "success"


def test_run_round0_staged_retry_uses_fixer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, solution_dir, state_path = _prepare_staged_round0_workspace(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        graph=_single_stage_impl_graph(),
    )
    ctx.round0_stage_history = [_stage_result("warp0::load_q")]
    ctx.round0_review_history = [_review("warp0::load_q", "retry_same_stage")]
    stage_labels: list[str] = []

    async def _fake_run_agent(**kwargs: Any) -> tuple[_FakeStageRunResult, float]:
        stage_labels.append(kwargs["stage_label"])
        if kwargs["stage_label"].startswith("kernel-stage-fixer-"):
            return _FakeStageRunResult(
                _stage_result("warp0::load_q", final_correctness_verified=True)
            ), 0.1
        if kwargs["stage_label"].startswith("kernel-stage-reviewer-"):
            return _FakeStageRunResult(
                _review("warp0::load_q", "continue_next_stage")
            ), 0.1
        raise AssertionError(f"Unexpected stage label: {kwargs['stage_label']}")

    monkeypatch.setattr(main, "_run_agent", _fake_run_agent)

    result = asyncio.run(
        main._run_round0_staged(
            ctx=ctx,
            solution_dir=solution_dir,
            state_path=state_path,
            designer=_FakeAgent("designer"),
            stage_coder=_FakeAgent("coder"),
            stage_fixer=_FakeAgent("fixer"),
            stage_reviewer=_FakeAgent("reviewer"),
            designer_max_turns=1,
            designer_verbose=False,
            coder_max_turns=1,
            coder_verbose=False,
            skip_designer=True,
            resume=False,
            prompt_dump=None,
        )
    )

    assert stage_labels[0].startswith("kernel-stage-fixer-")
    assert all("kernel-stage-coder-" not in label for label in stage_labels)
    assert result.status == "success"


@pytest.mark.parametrize(
    ("stage_history", "review_history", "expected_role", "expected_attempt"),
    [
        (
            [_stage_result("warp0::load_q", final_correctness_verified=True)],
            [],
            "coder",
            1,
        ),
        (
            [
                _stage_result("warp0::load_q"),
                _stage_result("warp0::load_q", final_correctness_verified=True),
            ],
            [_review("warp0::load_q", "retry_same_stage")],
            "fixer",
            2,
        ),
    ],
)
def test_run_round0_staged_resume_pending_review_reuses_latest_executor_prompt_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_history: list[Round0StageResult],
    review_history: list[StageReviewResult],
    expected_role: str,
    expected_attempt: int,
) -> None:
    ctx, solution_dir, state_path = _prepare_staged_round0_workspace(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        graph=_single_stage_impl_graph(),
    )
    ctx.round0_stage_history = stage_history
    ctx.round0_review_history = review_history
    prompt_dump = main._make_prompt_dump_config(tmp_path / "prompts")

    async def _fake_run_agent(**kwargs: Any) -> tuple[_FakeStageRunResult, float]:
        assert kwargs["stage_label"].startswith("kernel-stage-reviewer-")
        return _FakeStageRunResult(_review("warp0::load_q", "continue_next_stage")), 0.1

    monkeypatch.setattr(main, "_run_agent", _fake_run_agent)

    result = asyncio.run(
        main._run_round0_staged(
            ctx=ctx,
            solution_dir=solution_dir,
            state_path=state_path,
            designer=_FakeAgent("designer"),
            stage_coder=_FakeAgent("coder prompt"),
            stage_fixer=_FakeAgent("fixer prompt"),
            stage_reviewer=_FakeAgent("reviewer prompt"),
            designer_max_turns=1,
            designer_verbose=False,
            coder_max_turns=1,
            coder_verbose=False,
            skip_designer=False,
            resume=True,
            prompt_dump=prompt_dump,
        )
    )

    expected_path = (
        prompt_dump.runtime_dir
        / f"stage_warp0_load_q.attempt_{expected_attempt:02d}.{expected_role}.system.md"
    )
    unexpected_role = "fixer" if expected_role == "coder" else "coder"
    unexpected_path = (
        prompt_dump.runtime_dir
        / f"stage_warp0_load_q.attempt_{expected_attempt:02d}.{unexpected_role}.system.md"
    )

    assert expected_path.read_text(encoding="utf-8") == f"{expected_role} prompt"
    assert not unexpected_path.exists()
    assert result.status == "success"


def test_main_dump_prompts_only_exits_after_static_dump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dump_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "dump_all_prompts", lambda **kwargs: dump_calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["main.py", "--dump-prompts-only"])

    def _unexpected_asyncio_run(_: object) -> None:
        raise AssertionError("run_loop should not execute for --dump-prompts-only")

    monkeypatch.setattr(main.asyncio, "run", _unexpected_asyncio_run)

    main.main()

    assert len(dump_calls) == 1


def test_main_dump_prompts_runs_loop_with_prompt_dump_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dump_calls: list[dict[str, Any]] = []
    run_loop_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "dump_all_prompts", lambda **kwargs: dump_calls.append(kwargs))

    async def _fake_run_loop(**kwargs: Any) -> None:
        run_loop_calls.append(kwargs)

    monkeypatch.setattr(main, "run_loop", _fake_run_loop)
    monkeypatch.setattr(sys, "argv", ["main.py", "--dump-prompts"])

    main.main()

    assert len(dump_calls) == 1
    assert len(run_loop_calls) == 1
    assert run_loop_calls[0]["prompt_dump"] is not None


@pytest.mark.parametrize(
    (
        "stage_history",
        "review_history",
        "expected_completed",
        "expected_next_stage",
        "expected_pending_review",
        "expected_complete",
    ),
    [
        (
            [_stage_result("warp0::load_q")],
            [_review("warp0::load_q", "continue_next_stage")],
            ["warp0::load_q"],
            "warp1::qk_mma",
            False,
            False,
        ),
        (
            [_stage_result("warp0::load_q"), _stage_result("warp1::qk_mma")],
            [
                _review("warp0::load_q", "continue_next_stage"),
                _review("warp1::qk_mma", "retry_same_stage"),
            ],
            ["warp0::load_q"],
            "warp1::qk_mma",
            False,
            False,
        ),
        (
            [_stage_result("warp0::load_q"), _stage_result("warp1::qk_mma")],
            [
                _review("warp0::load_q", "continue_next_stage"),
                _review("warp1::qk_mma", "revise_design_then_retry"),
            ],
            [],
            "warp0::load_q",
            False,
            False,
        ),
        (
            [_stage_result("warp0::load_q"), _stage_result("warp1::qk_mma")],
            [_review("warp0::load_q", "continue_next_stage")],
            ["warp0::load_q"],
            "warp1::qk_mma",
            True,
            False,
        ),
        (
            [
                _stage_result("warp0::load_q"),
                _stage_result("warp1::qk_mma", final_correctness_verified=True),
            ],
            [
                _review("warp0::load_q", "continue_next_stage"),
                _review("warp1::qk_mma", "continue_next_stage"),
            ],
            ["warp0::load_q", "warp1::qk_mma"],
            None,
            False,
            True,
        ),
    ],
)
def test_derive_round0_progress_tracks_stage_and_review_state(
    tmp_path: Path,
    stage_history: list[Round0StageResult],
    review_history: list[StageReviewResult],
    expected_completed: list[str],
    expected_next_stage: str | None,
    expected_pending_review: bool,
    expected_complete: bool,
) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_stage_history = stage_history
    ctx.round0_review_history = review_history

    progress = main._derive_round0_progress(ctx, _sample_impl_graph())

    assert progress.completed_stage_ids == expected_completed
    assert progress.next_stage_id == expected_next_stage
    assert progress.pending_review is expected_pending_review
    assert progress.complete is expected_complete


def test_load_impl_graph_rejects_legacy_stage_schema(tmp_path: Path) -> None:
    graph_path = tmp_path / "impl_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "async_pipelines": [],
                "stages": [
                    {
                        "stage_id": "warp0::load_q",
                        "title": "Load Q",
                        "description": "Load query tiles.",
                        "owner_warps": ["warp0"],
                        "prerequisites": [],
                        "outputs": ["q_tile_debug"],
                        "checks": ["q tile matches eager reference"],
                        "validation_entry_point": "kernel_0.py::validate_stage__warp0_load_q",
                        "plan_excerpt": "## Load Q\nload q details",
                        "target_areas": ["load_q"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        main.load_impl_graph(graph_path)


def test_load_state_rejects_legacy_round0_stage_history_generated_field(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "loop_state.json"
    state_path.write_text(
        json.dumps(
            {
                "current_round": 0,
                "best_latency_ms": 1.0,
                "best_round": -1,
                "model_name": "gpt-5.4",
                "quality_profile": "public_codex",
                "codex_worker_mode": "off",
                "history": [],
                "round0_stage_history": [
                    {
                        "stage_id": "warp0::load_q",
                        "generated": ["solution/dsa_attention/kernel_0.py"],
                        "stage_output_validation_report": "ok",
                        "stage_output_verified": True,
                        "final_correctness_verified": False,
                        "correctness_check_report": "",
                        "status": "success",
                        "message": "ok",
                    }
                ],
                "round0_review_history": [],
            }
        ),
        encoding="utf-8",
    )

    restored = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )

    with pytest.raises(Exception):
        main.load_state(restored, state_path)
