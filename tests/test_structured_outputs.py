from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import pytest

from agents import AgentOutputSchema

import main
from kernel_agents.context import (
    AsyncPipelineSpec,
    CoderResult,
    DesignerResult,
    ImplementationGraph,
    KernelContractSpec,
    KeyValueNote,
    OptimizerResult,
    PlannerResult,
    ResourceLedgerSpec,
    RoundRecord,
    SharedContext,
    Round0StageResult,
    StageReviewResult,
    StageSpec,
)
from kernel_agents.kernel_coder import make_kernel_coder
from kernel_agents.kernel_stage_coder import make_round0_stage_coder


def _sample_impl_graph() -> ImplementationGraph:
    return ImplementationGraph(
        kernel_contract=KernelContractSpec(
            summary="kernel entry contract",
            items=[KeyValueNote(key="entry_point", value="kernel_0.py::kernel")],
        ),
        resource_ledger=ResourceLedgerSpec(
            summary="resource summary",
            items=[KeyValueNote(key="tmem_cols", value="0-127")],
        ),
        async_pipelines=[
            AsyncPipelineSpec(
                pipeline_id="P0",
                name="load q",
                producer_warps=["warp0"],
                consumer_warps=["warp1"],
                num_stages=1,
                payload="Q tile",
                smem_budget="72 KB",
                tmem_budget="0",
                register_budget="warp0=32,warp1=64",
                participating_stages=["warp0::load_q", "warp1::qk_mma"],
            )
        ],
        stages=[
            StageSpec(
                stage_id="warp0::load_q",
                title="Load Q",
                description="Load query tiles.",
                owner_warps=["warp0"],
                prerequisites=[],
                outputs=["q_tile_debug"],
                checks=["q tile matches eager reference"],
                validation_entry_point="kernel_0.py::validate_stage__warp0_load_q",
                plan_excerpt="## Load Q\nload q details",
                target_areas=["load_q"],
            ),
            StageSpec(
                stage_id="warp1::qk_mma",
                title="QK MMA",
                description="Compute score tiles.",
                owner_warps=["warp1"],
                prerequisites=["warp0::load_q"],
                outputs=["score_tile_debug"],
                checks=["qk tile matches eager reference"],
                validation_entry_point="kernel_0.py::validate_stage__warp1_qk_mma",
                plan_excerpt="## QK Mainloop\nqk details",
                target_areas=["run_qk_mma"],
            ),
        ],
    )


@pytest.mark.parametrize(
    ("result_type", "expected_fields"),
    [
        (DesignerResult, {"plan_file", "impl_graph", "status", "message"}),
        (
            CoderResult,
            {"generated", "correctness_verified", "status", "message", "reflection"},
        ),
        (
            Round0StageResult,
            {
                "stage_id",
                "generated",
                "stage_validation_reports",
                "current_stage_verified",
                "cumulative_regressions_verified",
                "synthetic_check_report",
                "final_correctness_verified",
                "correctness_check_report",
                "status",
                "message",
                "reflection",
            },
        ),
        (
            StageReviewResult,
            {
                "stage_id",
                "action",
                "message",
                "restart_from_stage_id",
                "replacement_impl_graph",
            },
        ),
        (
            PlannerResult,
            {"latency_ms", "bottleneck", "strategy_summary", "strategy_file", "is_new_best", "ncu_metrics"},
        ),
        (OptimizerResult, {"kernel_file", "correctness_verified", "status", "message", "reflection"}),
    ],
)
def test_result_models_use_top_level_structured_output(
    result_type: type[Any],
    expected_fields: set[str],
) -> None:
    schema = AgentOutputSchema(result_type).json_schema()

    assert set(schema["properties"]) == expected_fields
    assert set(schema["required"]) == expected_fields
    assert "response" not in schema["properties"]
    assert "response" not in schema["required"]
    assert schema["additionalProperties"] is False


class _FakeRunResult:
    def __init__(self, typed_output: Any) -> None:
        self.final_output = '{"generated":["solution/dsa_attention/kernel_0.py"]}'
        self._typed_output = typed_output
        self.calls: list[tuple[type[Any], bool]] = []

    def final_output_as(self, cls: type[Any], raise_if_incorrect_type: bool = False) -> Any:
        self.calls.append((cls, raise_if_incorrect_type))
        if raise_if_incorrect_type and not isinstance(self._typed_output, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(self._typed_output).__name__}")
        return self._typed_output


def test_require_structured_output_uses_sdk_typed_accessor() -> None:
    typed_output = CoderResult(
        generated=["solution/dsa_attention/kernel_0.py"],
        correctness_verified=True,
        status="success",
        message="validated",
        reflection="all good",
    )
    result = _FakeRunResult(typed_output)

    extracted = main.require_structured_output(result, CoderResult)

    assert extracted == typed_output
    assert result.calls == [(CoderResult, True)]


def test_require_structured_output_rejects_wrong_typed_result() -> None:
    result = _FakeRunResult("not a typed result")

    with pytest.raises(TypeError, match="Expected DesignerResult, got str"):
        main.require_structured_output(result, DesignerResult)

    assert result.calls == [(DesignerResult, True)]


def test_require_structured_output_requires_final_output_as() -> None:
    with pytest.raises(TypeError, match="final_output_as"):
        main.require_structured_output(object(), PlannerResult)


def test_validate_impl_graph_accepts_valid_graph() -> None:
    main.validate_impl_graph(_sample_impl_graph())


def test_validate_impl_graph_rejects_duplicate_stage_ids() -> None:
    graph = _sample_impl_graph()
    duplicate = StageSpec(
        stage_id="warp1::qk_mma",
        title="Duplicate",
        description="bad",
        owner_warps=["warp2"],
        prerequisites=[],
        outputs=["x"],
        checks=["y"],
        validation_entry_point="kernel_0.py::validate_stage__dup",
        plan_excerpt="## Duplicate\ndup",
        target_areas=[],
    )
    graph = graph.model_copy(update={"stages": [*graph.stages, duplicate]})

    with pytest.raises(ValueError, match="duplicate stage IDs"):
        main.validate_impl_graph(graph)


def test_validate_impl_graph_rejects_unknown_pipeline_participants() -> None:
    bad_pipeline = AsyncPipelineSpec(
        pipeline_id="P1",
        name="bad",
        producer_warps=["warp0"],
        consumer_warps=["warp1"],
        num_stages=1,
        payload="bad",
        smem_budget="0",
        tmem_budget="0",
        register_budget="0",
        participating_stages=["missing::stage"],
    )
    graph = _sample_impl_graph().model_copy(
        update={"async_pipelines": [bad_pipeline]}
    )

    with pytest.raises(ValueError, match="references unknown stages"):
        main.validate_impl_graph(graph)


def test_validate_impl_graph_rejects_prerequisites_that_point_forward() -> None:
    graph = _sample_impl_graph()
    stages = [
        graph.stages[0].model_copy(update={"prerequisites": ["warp1::qk_mma"]}),
        graph.stages[1],
    ]
    graph = graph.model_copy(update={"stages": stages})

    with pytest.raises(ValueError, match="must appear earlier in declared order"):
        main.validate_impl_graph(graph)


def test_save_and_load_state_round0_fields_round_trip(tmp_path: Path) -> None:
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
    ctx.round0_stage_history = [
        Round0StageResult(
            stage_id="warp0::load_q",
            generated=["solution/dsa_attention/kernel_0.py"],
            stage_validation_reports=[],
            current_stage_verified=True,
            cumulative_regressions_verified=True,
            synthetic_check_report="ok",
            final_correctness_verified=False,
            correctness_check_report="",
            status="success",
            message="validated",
            reflection="none",
        )
    ]
    ctx.round0_review_history = [
        StageReviewResult(
            stage_id="warp0::load_q",
            action="continue_next_stage",
            message="continue",
            restart_from_stage_id=None,
            replacement_impl_graph=None,
        )
    ]

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
    assert [item.action for item in restored.round0_review_history] == ["continue_next_stage"]


def test_load_state_tolerates_missing_round0_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "loop_state.json"
    state_path.write_text(
        """
{
  "current_round": 2,
  "best_latency_ms": 9.5,
  "best_round": 1,
  "model_name": "gpt-5.4",
  "quality_profile": "public_codex",
  "codex_worker_mode": "off",
  "history": []
}
""".strip(),
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


def _stage_result(
    stage_id: str,
    *,
    final_correctness_verified: bool = False,
) -> Round0StageResult:
    return Round0StageResult(
        stage_id=stage_id,
        generated=["solution/dsa_attention/kernel_0.py"],
        stage_validation_reports=[],
        current_stage_verified=True,
        cumulative_regressions_verified=True,
        synthetic_check_report="ok",
        final_correctness_verified=final_correctness_verified,
        correctness_check_report="correct",
        status="success",
        message=f"{stage_id} ok",
        reflection="",
    )


def _review(
    stage_id: str,
    action: str,
    *,
    restart_from_stage_id: str | None = None,
) -> StageReviewResult:
    return StageReviewResult(
        stage_id=stage_id,
        action=action,
        message=action,
        restart_from_stage_id=restart_from_stage_id,
        replacement_impl_graph=_sample_impl_graph() if action == "revise_design_then_retry" else None,
    )


def _payload_from_builder(rendered: str) -> dict[str, Any]:
    marker = "Caller payload:\n"
    _, payload = rendered.split(marker, 1)
    return json.loads(payload)


def test_build_stage_coder_input_includes_plan_excerpt_directly() -> None:
    payload = _payload_from_builder(
        main._build_stage_coder_input(
            graph=_sample_impl_graph(),
            stage_id="warp1::qk_mma",
            completed_stage_ids=["warp0::load_q"],
            latest_stage_results={"warp0::load_q": _stage_result("warp0::load_q")},
            is_final_stage=False,
        )
    )

    assert payload["current_stage"]["plan_excerpt"] == "## QK Mainloop\nqk details"
    assert "relevant_plan_excerpts" not in payload
    assert "plan_excerpt" not in payload["completed_stage_validations"][0]


def test_build_stage_review_input_includes_plan_excerpt_and_completed_validations() -> None:
    payload = _payload_from_builder(
        main._build_stage_review_input(
            graph=_sample_impl_graph(),
            stage_result=_stage_result("warp1::qk_mma"),
            completed_stage_ids=["warp0::load_q"],
            latest_stage_results={"warp0::load_q": _stage_result("warp0::load_q")},
        )
    )

    assert payload["current_stage"]["stage_id"] == "warp1::qk_mma"
    assert payload["current_stage"]["plan_excerpt"] == "## QK Mainloop\nqk details"
    assert payload["completed_stage_validations"][0]["stage_id"] == "warp0::load_q"


def test_stage_coder_inherits_shared_kernel_coder_guidance(tmp_path: Path) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0_plan.md").write_text("# Plan\nplan body\n", encoding="utf-8")

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    coder = make_kernel_coder(context=ctx)
    stage_coder = make_round0_stage_coder(context=ctx)

    coder_prompt = main._resolve_agent_instructions_text(coder, ctx)
    stage_prompt = main._resolve_agent_instructions_text(stage_coder, ctx)

    shared_phrases = [
        "FORBIDDEN: torch.matmul/bmm/einsum/softmax/logsumexp/masked_fill",
        "Use `cutlass.Constexpr` for static shapes; annotate types for the JIT.",
        "## JIT compile cache pattern",
        "references/cutlass/examples/python/CuTeDSL/blackwell",
    ]

    for phrase in shared_phrases:
        assert phrase in coder_prompt
        assert phrase in stage_prompt

    assert "Return `CoderResult`" in coder_prompt
    assert "Return `Round0StageResult`" in stage_prompt
    assert "run_stage_validation" in stage_prompt


class _FakeAgent:
    def __init__(self, instructions: object) -> None:
        self.instructions = instructions


def test_dump_all_prompts_includes_stage_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    def _fake_agent_factory(**_: Any) -> _FakeAgent:
        return _FakeAgent("static prompt")

    monkeypatch.setattr(main, "make_kernel_designer", _fake_agent_factory)
    monkeypatch.setattr(main, "make_kernel_coder", _fake_agent_factory)
    monkeypatch.setattr(main, "make_round0_stage_coder", _fake_agent_factory)
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
        planner_reasoning_effort="high",
        optimizer_reasoning_effort="high",
    )

    assert (output_dir / "kernel_stage_coder.md").read_text(encoding="utf-8") == "static prompt"
    assert (output_dir / "kernel_stage_reviewer.md").read_text(encoding="utf-8") == "static prompt"


def test_dump_stage_runtime_prompt_artifacts_writes_and_reuses_files(tmp_path: Path) -> None:
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

    main._dump_stage_runtime_prompt_artifacts(
        prompt_dump=prompt_dump,
        agent=agent,
        ctx=ctx,
        stage_id="warp1::qk_mma",
        attempt=2,
        role_name="reviewer",
        payload={"step": "reviewer"},
    )
    reviewer_input = prompt_dump.runtime_dir / "stage_warp1_qk_mma.attempt_02.reviewer.input.json"
    assert json.loads(reviewer_input.read_text(encoding="utf-8")) == {"step": "reviewer"}


def test_main_dump_prompts_only_exits_after_static_dump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dump_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main,
        "dump_all_prompts",
        lambda **kwargs: dump_calls.append(kwargs),
    )
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
    monkeypatch.setattr(
        main,
        "dump_all_prompts",
        lambda **kwargs: dump_calls.append(kwargs),
    )

    async def _fake_run_loop(**kwargs: Any) -> None:
        run_loop_calls.append(kwargs)

    monkeypatch.setattr(main, "run_loop", _fake_run_loop)
    monkeypatch.setattr(sys, "argv", ["main.py", "--dump-prompts"])

    main.main()

    assert len(dump_calls) == 1
    assert len(run_loop_calls) == 1
    assert run_loop_calls[0]["prompt_dump"] is not None


def test_derive_round0_progress_continue_advances_to_next_stage(tmp_path: Path) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_stage_history = [_stage_result("warp0::load_q")]
    ctx.round0_review_history = [_review("warp0::load_q", "continue_next_stage")]

    progress = main._derive_round0_progress(ctx, _sample_impl_graph())

    assert progress.completed_stage_ids == ["warp0::load_q"]
    assert progress.next_stage_id == "warp1::qk_mma"
    assert progress.pending_review is False


def test_derive_round0_progress_retry_keeps_same_stage_active(tmp_path: Path) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_stage_history = [
        _stage_result("warp0::load_q"),
        _stage_result("warp1::qk_mma"),
    ]
    ctx.round0_review_history = [
        _review("warp0::load_q", "continue_next_stage"),
        _review("warp1::qk_mma", "retry_same_stage"),
    ]

    progress = main._derive_round0_progress(ctx, _sample_impl_graph())

    assert progress.completed_stage_ids == ["warp0::load_q"]
    assert progress.next_stage_id == "warp1::qk_mma"


def test_derive_round0_progress_revise_rewinds_from_restart_stage(tmp_path: Path) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_stage_history = [
        _stage_result("warp0::load_q"),
        _stage_result("warp1::qk_mma"),
    ]
    ctx.round0_review_history = [
        _review("warp0::load_q", "continue_next_stage"),
        _review(
            "warp1::qk_mma",
            "revise_design_then_retry",
            restart_from_stage_id="warp1::qk_mma",
        ),
    ]

    progress = main._derive_round0_progress(ctx, _sample_impl_graph())

    assert progress.completed_stage_ids == ["warp0::load_q"]
    assert progress.next_stage_id == "warp1::qk_mma"


def test_derive_round0_progress_detects_pending_unreviewed_stage(tmp_path: Path) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_stage_history = [
        _stage_result("warp0::load_q"),
        _stage_result("warp1::qk_mma"),
    ]
    ctx.round0_review_history = [_review("warp0::load_q", "continue_next_stage")]

    progress = main._derive_round0_progress(ctx, _sample_impl_graph())

    assert progress.pending_review is True
    assert progress.pending_stage_result is not None
    assert progress.next_stage_id == "warp1::qk_mma"


def test_derive_round0_progress_marks_complete_after_final_approval(tmp_path: Path) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_stage_history = [
        _stage_result("warp0::load_q"),
        _stage_result("warp1::qk_mma", final_correctness_verified=True),
    ]
    ctx.round0_review_history = [
        _review("warp0::load_q", "continue_next_stage"),
        _review("warp1::qk_mma", "continue_next_stage"),
    ]

    progress = main._derive_round0_progress(ctx, _sample_impl_graph())

    assert progress.complete is True
    assert progress.next_stage_id is None
