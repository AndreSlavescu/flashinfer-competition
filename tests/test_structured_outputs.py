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
    OptimizerResult,
    PlannerResult,
    RoundRecord,
    SharedContext,
    Round0StageResult,
    StageReviewResult,
    StageSpec,
)
from kernel_agents.kernel_coder import make_kernel_coder
from kernel_agents.kernel_stage_coder import make_round0_stage_coder
from kernel_agents.kernel_stage_reviewer import make_round0_stage_reviewer


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
                smem_budget="72 KB",
                tmem_budget="0",
                register_budget="warp0=32,warp1=64",
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
                "stage_output_validation_report",
                "stage_output_verified",
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
        prerequisites=[],
        outputs=["x"],
        relevant_helpers=["helper"],
        plan_excerpt="## Duplicate\ndup",
    )
    graph = graph.model_copy(update={"stages": [*graph.stages, duplicate]})

    with pytest.raises(ValueError, match="duplicate stage IDs"):
        main.validate_impl_graph(graph)

def test_async_pipeline_spec_rejects_disallowed_multi_consumer_type() -> None:
    with pytest.raises(Exception, match="PipelineTmaMultiConsumersAsync"):
        AsyncPipelineSpec.model_validate(
            {
                "pipeline_id": "P1",
                "type": "PipelineTmaMultiConsumersAsync",
                "producer_warp": "warp0",
                "consumer_warp": "warp1",
                "num_stages": 1,
                "payload": "bad",
                "smem_budget": "0",
                "tmem_budget": "0",
                "register_budget": "0",
            }
        )


@pytest.mark.parametrize(
    "pipeline_type",
    [
        "PipelineAsync",
        "PipelineCpAsync",
        "PipelineTmaAsync",
        "PipelineTmaUmma",
        "PipelineAsyncUmma",
        "PipelineUmmaAsync",
        "PipelineClcFetchAsync",
    ],
)
def test_async_pipeline_spec_accepts_allowed_single_producer_single_consumer_types(
    pipeline_type: str,
) -> None:
    pipeline = AsyncPipelineSpec.model_validate(
        {
            "pipeline_id": "P1",
            "type": pipeline_type,
            "producer_warp": "warp0",
            "consumer_warp": "warp1",
            "num_stages": 1,
            "payload": "tile",
            "smem_budget": "0",
            "tmem_budget": "0",
            "register_budget": "0",
        }
    )

    assert pipeline.type == pipeline_type


def test_async_pipeline_spec_rejects_exported_type_without_consumer() -> None:
    with pytest.raises(Exception, match="PipelineTmaStore"):
        AsyncPipelineSpec.model_validate(
            {
                "pipeline_id": "P1",
                "type": "PipelineTmaStore",
                "producer_warp": "warp0",
                "consumer_warp": "warp1",
                "num_stages": 1,
                "payload": "tile",
                "smem_budget": "0",
                "tmem_budget": "0",
                "register_budget": "0",
            }
        )


@pytest.mark.parametrize(("field_name", "value"), [("producer_warp", ""), ("consumer_warp", " ")])
def test_async_pipeline_spec_rejects_blank_warp_fields(field_name: str, value: str) -> None:
    payload = {
        "pipeline_id": "P1",
        "type": "PipelineTmaUmma",
        "producer_warp": "warp0",
        "consumer_warp": "warp1",
        "num_stages": 1,
        "payload": "tile",
        "smem_budget": "0",
        "tmem_budget": "0",
        "register_budget": "0",
    }
    payload[field_name] = value

    with pytest.raises(Exception, match=field_name):
        AsyncPipelineSpec.model_validate(payload)


def test_async_pipeline_spec_rejects_blank_type() -> None:
    with pytest.raises(Exception):
        AsyncPipelineSpec.model_validate(
            {
                "pipeline_id": "P1",
                "type": "",
                "producer_warp": "warp0",
                "consumer_warp": "warp1",
                "num_stages": 1,
                "payload": "tile",
                "smem_budget": "0",
                "tmem_budget": "0",
                "register_budget": "0",
            }
        )


def test_validate_impl_graph_rejects_prerequisites_that_point_forward() -> None:
    graph = _sample_impl_graph()
    stages = [
        graph.stages[0].model_copy(update={"prerequisites": ["warp1::qk_mma"]}),
        graph.stages[1],
    ]
    graph = graph.model_copy(update={"stages": stages})

    with pytest.raises(ValueError, match="must appear earlier in declared order"):
        main.validate_impl_graph(graph)


def test_stage_spec_requires_non_empty_relevant_helpers() -> None:
    with pytest.raises(Exception, match="relevant_helpers"):
        StageSpec(
            stage_id="warp0::load_q",
            prerequisites=[],
            outputs=["q_tile_debug matches eager reference"],
            relevant_helpers=[],
            plan_excerpt="## Load Q\nload q details",
        )


def test_stage_spec_rejects_blank_relevant_helper_entries() -> None:
    with pytest.raises(Exception, match="relevant_helpers"):
        StageSpec(
            stage_id="warp0::load_q",
            prerequisites=[],
            outputs=["q_tile_debug matches eager reference"],
            relevant_helpers=[""],
            plan_excerpt="## Load Q\nload q details",
        )


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
            stage_output_validation_report="stage output ok",
            stage_output_verified=True,
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
        stage_output_validation_report="stage output ok",
        stage_output_verified=True,
        final_correctness_verified=final_correctness_verified,
        correctness_check_report="correct",
        status="success",
        message=f"{stage_id} ok",
        reflection="",
    )


def _review(
    stage_id: str,
    action: str,
) -> StageReviewResult:
    return StageReviewResult(
        stage_id=stage_id,
        action=action,
        message=action,
    )


def test_build_stage_coder_payload_keeps_only_required_fields() -> None:
    payload = main._build_stage_coder_payload(
        graph=_sample_impl_graph(),
        stage_id="warp1::qk_mma",
        is_final_stage=False,
        last_review=None,
    )

    assert set(payload) == {
        "current_stage",
        "prerequisite_stages",
        "last_review",
        "is_final_stage",
    }
    assert payload["current_stage"]["plan_excerpt"] == "## QK Mainloop\nqk details"
    assert payload["current_stage"]["outputs"] == ["score_tile_debug matches the eager fp32 QK tile"]
    assert payload["current_stage"]["relevant_helpers"] == ["tcgen05.mma", "pipeline example"]
    assert [stage["stage_id"] for stage in payload["prerequisite_stages"]] == ["warp0::load_q"]
    assert payload["prerequisite_stages"][0]["plan_excerpt"] == "## Load Q\nload q details"
    assert payload["is_final_stage"] is False
    assert payload["last_review"] is None
    assert "debug_exports" not in payload["current_stage"]
    assert "checks" not in payload["current_stage"]
    assert "target_areas" not in payload["current_stage"]
    assert "approved_frontier_summaries" not in payload


def test_build_stage_coder_payload_includes_trimmed_last_review() -> None:
    review = _review("warp1::qk_mma", "retry_same_stage")
    payload = main._build_stage_coder_payload(
        graph=_sample_impl_graph(),
        stage_id="warp1::qk_mma",
        is_final_stage=False,
        last_review=review,
    )

    assert payload["last_review"] == {
        "action": "retry_same_stage",
        "message": "retry_same_stage",
    }


def test_build_stage_review_payload_includes_diff_context_without_prior_snapshot(
    tmp_path: Path,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0.py").write_text("def kernel():\n    return 'current'\n", encoding="utf-8")

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    stage_result = _stage_result("warp1::qk_mma")
    ctx.round0_stage_history = [stage_result]

    payload = main._build_stage_review_payload(
        ctx=ctx,
        solution_dir=solution_dir,
        graph=_sample_impl_graph(),
        stage_result=stage_result,
        stage_attempt=1,
        is_final_stage=False,
    )

    assert set(payload) == {
        "current_stage",
        "stage_results_history",
        "review_history",
        "file_diffs",
        "is_final_stage",
    }
    assert payload["current_stage"]["stage_id"] == "warp1::qk_mma"
    assert payload["current_stage"]["plan_excerpt"] == "## QK Mainloop\nqk details"
    assert payload["current_stage"]["outputs"] == ["score_tile_debug matches the eager fp32 QK tile"]
    assert payload["current_stage"]["relevant_helpers"] == ["tcgen05.mma", "pipeline example"]
    assert payload["is_final_stage"] is False
    assert payload["stage_results_history"] == [
        {
            "stage_id": "warp1::qk_mma",
            "stage_output_validation_report": "stage output ok",
            "correctness_check_report": "correct",
            "message": "warp1::qk_mma ok",
        }
    ]
    assert payload["review_history"] == []
    assert payload["file_diffs"]["diff_status"] == "no_prior_snapshot"
    assert payload["file_diffs"]["previous_kernel_snapshot"] is None
    assert "NO PRIOR SNAPSHOT" in payload["file_diffs"]["unified_diff"]
    assert (
        solution_dir / "round0" / "warp1_qk_mma.attempt_01.kernel_0.py"
    ).exists()


def test_build_stage_review_payload_uses_previous_snapshot_for_unified_diff(
    tmp_path: Path,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0.py").write_text("def kernel():\n    return 'current'\n", encoding="utf-8")

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    previous_result = _stage_result("warp0::load_q")
    current_result = _stage_result("warp1::qk_mma")
    ctx.round0_stage_history = [previous_result, current_result]

    previous_snapshot = main.round0_stage_kernel_snapshot_path(
        solution_dir,
        "warp0::load_q",
        1,
    )
    previous_snapshot.parent.mkdir(parents=True, exist_ok=True)
    previous_snapshot.write_text("def kernel():\n    return 'previous'\n", encoding="utf-8")

    payload = main._build_stage_review_payload(
        ctx=ctx,
        solution_dir=solution_dir,
        graph=_sample_impl_graph(),
        stage_result=current_result,
        stage_attempt=1,
        is_final_stage=True,
    )

    assert payload["file_diffs"]["diff_status"] == "available"
    assert payload["is_final_stage"] is True
    assert payload["file_diffs"]["previous_kernel_snapshot"] == (
        "solution/dsa_attention/round0/warp0_load_q.attempt_01.kernel_0.py"
    )
    assert "--- solution/dsa_attention/round0/warp0_load_q.attempt_01.kernel_0.py" in payload["file_diffs"]["unified_diff"]
    assert "+++ solution/dsa_attention/round0/warp1_qk_mma.attempt_01.kernel_0.py" in payload["file_diffs"]["unified_diff"]
    assert "-    return 'previous'" in payload["file_diffs"]["unified_diff"]
    assert "+    return 'current'" in payload["file_diffs"]["unified_diff"]


def test_stage_coder_prompt_uses_new_body_and_dynamic_sections(tmp_path: Path) -> None:
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
    stage_coder = make_round0_stage_coder(context=ctx)

    static_prompt = main._resolve_agent_instructions_text(stage_coder, ctx)
    assert "{..## Current Stage Specifications..}" in static_prompt
    assert "{..## Pre-requisite Stage Specifications..}" in static_prompt
    assert "## References" in static_prompt
    assert "## Tools You Have" in static_prompt
    assert "## Caller payload contract" not in static_prompt
    assert "## JIT compile cache pattern" not in static_prompt
    assert "Do not stop early just to save tool calls." not in static_prompt

    payload = main._build_stage_coder_payload(
        graph=_sample_impl_graph(),
        stage_id="warp1::qk_mma",
        is_final_stage=False,
        last_review=None,
    )
    main._apply_stage_coder_prompt_sections(ctx, payload)
    stage_prompt = main._resolve_agent_instructions_text(stage_coder, ctx)

    assert "{..## Current Stage Specifications..}" not in stage_prompt
    assert "{..## Pre-requisite Stage Specifications..}" not in stage_prompt
    assert "## Current Stage Specifications" in stage_prompt
    assert "## Pre-requisite Stage Specifications" in stage_prompt
    assert "## Attempt Metadata" in stage_prompt
    assert "## References" in stage_prompt
    assert "Return structured output matching `Round0StageResult`." in stage_prompt
    assert "Read through the current stage specifications and the existing stages written in kernel_0.py" in stage_prompt
    assert "Run tests only on Modal B200 via `run_stage_validation` / `run_synthetic_check` / `run_correctness_check`" in stage_prompt
    assert "`prefix_validation_outputs_cute`" in stage_prompt
    assert "`prefix_validation_outputs_torch`" in stage_prompt
    assert "`prefix_validation_harness`" in stage_prompt
    assert "`run_stage_validation` always executes `kernel_0.py::prefix_validation_harness`" in stage_prompt
    assert "The prefix validation helpers are cumulative." in stage_prompt
    assert "references/cutlass/examples/python/CuTeDSL/blackwell" in stage_prompt
    assert '"is_final_stage": false' in stage_prompt
    assert "approved_frontier_summaries" not in stage_prompt
    assert "## Caller payload contract" not in stage_prompt
    assert "## Diff-first scoped edit discipline" not in stage_prompt
    assert "## `Round0StageResult` status rubric" not in stage_prompt


def test_stage_reviewer_prompt_uses_new_body_and_dynamic_sections(
    tmp_path: Path,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0.py").write_text("def kernel():\n    return 'current'\n", encoding="utf-8")

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    reviewer = make_round0_stage_reviewer(context=ctx)
    static_prompt = main._resolve_agent_instructions_text(reviewer, ctx)

    assert "{..## Current Stage Specifications..}" in static_prompt
    assert "{..## File Diffs..}" in static_prompt
    assert "{..## Trimmed Round0StageResult history..}" in static_prompt
    assert "{..## Trimmed StageReviewResult history..}" in static_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in static_prompt
    assert "## Tools You Have" in static_prompt
    assert "## Action gates" not in static_prompt
    assert "## Revision contract" not in static_prompt

    stage_result = _stage_result("warp1::qk_mma")
    ctx.round0_stage_history = [stage_result]
    ctx.round0_review_history = [_review("warp0::load_q", "continue_next_stage")]
    payload = main._build_stage_review_payload(
        ctx=ctx,
        solution_dir=solution_dir,
        graph=_sample_impl_graph(),
        stage_result=stage_result,
        stage_attempt=1,
        is_final_stage=False,
    )
    main._apply_stage_reviewer_prompt_sections(ctx, payload)
    reviewer_prompt = main._resolve_agent_instructions_text(reviewer, ctx)

    assert "{..## Current Stage Specifications..}" not in reviewer_prompt
    assert "{..## File Diffs..}" not in reviewer_prompt
    assert "{..## Trimmed Round0StageResult history..}" not in reviewer_prompt
    assert "{..## Trimmed StageReviewResult history..}" not in reviewer_prompt
    assert "## Current Stage Specifications" in reviewer_prompt
    assert "## File Diffs" in reviewer_prompt
    assert "## Trimmed Round0StageResult history" in reviewer_prompt
    assert "## Trimmed StageReviewResult history" in reviewer_prompt
    assert '"stage_output_validation_report": "stage output ok"' in reviewer_prompt
    assert '"action": "continue_next_stage"' in reviewer_prompt
    assert "## Attempt Metadata" in reviewer_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in reviewer_prompt
    assert 'action="retry_same_stage"' in reviewer_prompt
    assert 'action="revise_design_then_retry"' in reviewer_prompt
    assert 'action="continue_next_stage"' in reviewer_prompt
    assert "Be precise." in reviewer_prompt
    assert "`prefix_validation_outputs_cute`" in reviewer_prompt
    assert "`prefix_validation_outputs_torch`" in reviewer_prompt
    assert "`prefix_validation_harness`" in reviewer_prompt
    assert "completed prefix stages plus the current stage" in reviewer_prompt
    assert "validation_entry_point" not in reviewer_prompt
    assert "Keep `message` terse and precise." not in reviewer_prompt
    assert "## Action gates" not in reviewer_prompt
    assert "## Single-decision discipline" not in reviewer_prompt
    assert "## Revision contract" not in reviewer_prompt


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


def test_derive_round0_progress_revise_resets_all_progress(tmp_path: Path) -> None:
    """A stale revise_design_then_retry entry in the histories should reset progress.

    In normal operation the orchestrator clears both histories after a revision, so this
    code path is only reached defensively (e.g., a corrupted state file). When it is
    reached, the derivation must treat it as a full reset so no stale completions leak
    into the new graph.
    """
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
        _review("warp1::qk_mma", "revise_design_then_retry"),
    ]

    progress = main._derive_round0_progress(ctx, _sample_impl_graph())

    assert progress.completed_stage_ids == []
    assert progress.next_stage_id == "warp0::load_q"


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


def test_load_impl_graph_rejects_legacy_async_pipeline_schema(tmp_path: Path) -> None:
    graph_path = tmp_path / "impl_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "async_pipelines": [
                    {
                        "pipeline_id": "P0",
                        "name": "load q",
                        "producer_warps": ["warp0"],
                        "consumer_warps": ["warp1"],
                        "num_stages": 1,
                        "payload": "Q tile",
                        "smem_budget": "72 KB",
                        "tmem_budget": "0",
                        "register_budget": "warp0=32,warp1=64",
                        "participating_stages": ["warp0::load_q"],
                    }
                ],
                "stages": [
                    {
                        "stage_id": "warp0::load_q",
                        "prerequisites": [],
                        "outputs": ["q_tile_debug"],
                        "relevant_helpers": ["cute.make_tensor"],
                        "plan_excerpt": "## Load Q\nload q details",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        main.load_impl_graph(graph_path)


def test_load_state_rejects_legacy_round0_stage_history_with_generated_field(
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
                        "reflection": "",
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
