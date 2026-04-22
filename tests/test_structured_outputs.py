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
    DesignerResult,
    PlannerResult,
    Round0StageResult,
    RoundRecord,
    SharedContext,
    StageReviewResult,
)
from kernel_agents.kernel_stage_coder import make_round0_stage_coder
from kernel_agents.kernel_stage_fixer import make_round0_stage_fixer
from kernel_agents.kernel_stage_reviewer import make_round0_stage_reviewer
from kernel_agents.scoping import AGENT_SCOPES
from tests.support import extract_markdown_section, write_round0_stage_plan_fixture


def _sample_stage_plan() -> str:
    return """# Kernel 0 Round-0 Plan

## 1. Problem specification

Placeholder.

## 6. Round-0 implementation stages

### S0 — Load Q

Load query tiles and expose a debug output.

Depends on: none

Validation outputs:
- `q_tile_debug` [sources: PyTorch, CuTeDSL]

Key CuTeDSL helpers:
- `cute.make_tensor`
- `cute.copy`

### S1 — QK Mainloop

Consume the staged query tile and produce a score tile.

Depends on: S0

Validation outputs:
- `score_tile_debug` [sources: CuTeDSL]

Key CuTeDSL helpers:
- `tcgen05.mma`
- `pipeline.PipelineUmmaAsync.create`
"""


def _single_stage_plan() -> str:
    return """# Kernel 0 Round-0 Plan

## 6. Round-0 implementation stages

### S0 — Final Stage

Produce the final output.

Depends on: none

Validation outputs:
- `output` [sources: PyTorch, CuTeDSL]

Key CuTeDSL helpers:
- `cute.copy`
"""


def _stage_result(
    stage_id: str,
    *,
    stage_output_verified: bool = True,
    final_correctness_verified: bool = False,
    status: str = "success",
) -> Round0StageResult:
    return Round0StageResult(
        stage_id=stage_id,
        stage_output_validation_report="stage output ok",
        stage_output_verified=stage_output_verified,
        final_correctness_verified=final_correctness_verified,
        correctness_check_report="correct",
        status=status,  # type: ignore[arg-type]
        message=f"{stage_id} ok",
    )


def _review(
    stage_id: str,
    action: str,
    *,
    next_stage: str | None,
    message: str | None = None,
) -> StageReviewResult:
    return StageReviewResult(
        stage_id=stage_id,
        action=action,  # type: ignore[arg-type]
        next_stage=next_stage,
        message=message or action,
    )


@pytest.mark.parametrize(
    ("result_type", "expected_fields"),
    [
        (DesignerResult, {"plan_file", "status"}),
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
        (StageReviewResult, {"stage_id", "action", "next_stage", "message"}),
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


def test_stage_review_result_requires_action_specific_next_stage() -> None:
    with pytest.raises(Exception, match="retry_same_stage requires next_stage == stage_id"):
        _review("S0", "retry_same_stage", next_stage="S1")

    with pytest.raises(Exception, match="revise_design_then_retry requires next_stage == null"):
        _review("S0", "revise_design_then_retry", next_stage="S0")


def test_require_structured_output_uses_sdk_typed_accessor() -> None:
    typed_output = DesignerResult(
        plan_file="solution/dsa_attention/kernel_0_plan.md",
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
    ctx.round0_stage_history = [_stage_result("S0")]
    ctx.round0_review_history = [_review("S0", "continue_next_stage", next_stage="S1")]

    state_path = tmp_path / "loop_state.json"
    main.save_state(ctx, state_path)
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))

    restored = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    last_round = main.load_state(restored, state_path)

    assert last_round == 0
    assert "round0_driver_schema_version" not in saved_state
    assert [item.stage_id for item in restored.round0_stage_history] == ["S0"]
    assert [item.next_stage for item in restored.round0_review_history] == ["S1"]


def test_load_state_requires_current_round0_fields(tmp_path: Path) -> None:
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
                "codex_thread_id_coder_engineer": None,
                "codex_thread_id_optimizer_engineer": None,
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
    with pytest.raises(KeyError, match="round0_mode"):
        main.load_state(restored, state_path)


def test_load_state_rejects_round0_stage_history_with_extra_fields(
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
                "codex_thread_id_coder_engineer": None,
                "codex_thread_id_optimizer_engineer": None,
                "history": [],
                "round0_mode": "staged",
                "round0_stage_history": [
                    {
                        "stage_id": "S0",
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


def test_validate_stage_review_result_accepts_reviewer_directed_next_stage() -> None:
    main.validate_stage_review_result(
        stage_result=_stage_result("S0"),
        review=_review("S0", "continue_next_stage", next_stage="S42"),
    )


@pytest.mark.parametrize(
    ("stage_result", "review", "message"),
    [
        (
            _stage_result("S0", status="validation_failed"),
            _review("S0", "continue_next_stage", next_stage="S1"),
            "status='validation_failed'",
        ),
        (
            _stage_result("S0", stage_output_verified=False),
            _review("S0", "continue_next_stage", next_stage="S1"),
            "without passing stage validation",
        ),
        (
            _stage_result("S0", final_correctness_verified=False),
            _review("S0", "continue_next_stage", next_stage=None),
            "cannot terminate round-0 without correctness",
        ),
    ],
)
def test_validate_stage_review_result_rejects_invalid_continue_actions(
    stage_result: Round0StageResult,
    review: StageReviewResult,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        main.validate_stage_review_result(stage_result=stage_result, review=review)


@pytest.mark.parametrize(
    (
        "stage_history",
        "review_history",
        "expected_attempt_counts",
        "expected_next_stage",
        "expected_pending_review",
        "expected_complete",
        "expected_latest_stage",
    ),
    [
        ([], [], {}, "S0", False, False, None),
        (
            [_stage_result("S0")],
            [_review("S0", "continue_next_stage", next_stage="S1")],
            {"S0": 1},
            "S1",
            False,
            False,
            "S0",
        ),
        (
            [_stage_result("S0")],
            [_review("S0", "retry_same_stage", next_stage="S0")],
            {"S0": 1},
            "S0",
            False,
            False,
            "S0",
        ),
        (
            [_stage_result("S0"), _stage_result("S1")],
            [_review("S0", "continue_next_stage", next_stage="S1")],
            {"S0": 1, "S1": 1},
            "S1",
            True,
            False,
            "S1",
        ),
        (
            [
                _stage_result("S0"),
                _stage_result("S1", final_correctness_verified=True),
            ],
            [
                _review("S0", "continue_next_stage", next_stage="S1"),
                _review("S1", "continue_next_stage", next_stage=None),
            ],
            {"S0": 1, "S1": 1},
            None,
            False,
            True,
            "S1",
        ),
        (
            [_stage_result("S0")],
            [_review("S0", "revise_design_then_retry", next_stage=None)],
            {},
            None,
            False,
            False,
            None,
        ),
    ],
)
def test_derive_round0_progress_tracks_history_only(
    tmp_path: Path,
    stage_history: list[Round0StageResult],
    review_history: list[StageReviewResult],
    expected_attempt_counts: dict[str, int],
    expected_next_stage: str | None,
    expected_pending_review: bool,
    expected_complete: bool,
    expected_latest_stage: str | None,
) -> None:
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_stage_history = stage_history
    ctx.round0_review_history = review_history

    progress = main._derive_round0_progress(ctx)

    assert progress.attempt_counts == expected_attempt_counts
    assert progress.next_stage_id == expected_next_stage
    assert progress.pending_review is expected_pending_review
    assert progress.complete is expected_complete
    assert (
        progress.latest_stage_result.stage_id if progress.latest_stage_result else None
    ) == expected_latest_stage


def test_stage_payload_builders_keep_only_runtime_fields_and_create_snapshots(
    tmp_path: Path,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0.py").write_text(
        "def kernel():\n    return 'current'\n",
        encoding="utf-8",
    )

    plan_text = _sample_stage_plan()
    write_round0_stage_plan_fixture(tmp_path, plan_text)

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    stage_result = _stage_result("S1")
    ctx.round0_stage_history = [_stage_result("S0"), stage_result]
    ctx.round0_review_history = [_review("S0", "continue_next_stage", next_stage="S1")]
    previous_snapshot = main.round0_stage_kernel_snapshot_path(solution_dir, "S0", 1)
    previous_snapshot.parent.mkdir(parents=True, exist_ok=True)
    previous_snapshot.write_text("def kernel():\n    return 'previous'\n", encoding="utf-8")

    coder_payload = main._build_stage_coder_payload(
        plan_text=plan_text,
        stage_id="S1",
    )
    fixer_payload = main._build_stage_fixer_payload(
        plan_text=plan_text,
        stage_id="S1",
        last_review=_review("S1", "retry_same_stage", next_stage="S1"),
    )
    review_payload = main._build_stage_review_payload(
        ctx=ctx,
        solution_dir=solution_dir,
        plan_text=plan_text,
        stage_result=stage_result,
        stage_attempt=1,
    )
    recovery_payload = main._build_stage_recovery_payload(ctx=ctx, plan_text=plan_text)

    assert coder_payload == {
        "assigned_stage": "S1",
        "full_plan": plan_text,
    }
    assert fixer_payload == {
        "stage_to_fix": "S1",
        "full_plan": plan_text,
        "last_review": {
            "action": "retry_same_stage",
            "next_stage": "S1",
            "message": "retry_same_stage",
        },
    }
    assert review_payload == {
        "stage_under_review": "S1",
        "full_plan": plan_text,
        "kernel_snapshots": {
            "current_kernel_path": "solution/dsa_attention/kernel_0.py",
            "current_attempt_snapshot": "solution/dsa_attention/round0/S1.attempt_01.kernel_0.py",
            "previous_attempt_snapshot": "solution/dsa_attention/round0/S0.attempt_01.kernel_0.py",
        },
        "stage_results_history": [
            {
                "stage_id": "S1",
                "stage_output_validation_report": "stage output ok",
                "correctness_check_report": "correct",
                "message": "S1 ok",
            },
        ],
        "review_history": [],
        "recovery_context": "",
    }
    assert recovery_payload["stage_results_history"] == [
        {
            "stage_id": "S0",
            "stage_output_validation_report": "stage output ok",
            "correctness_check_report": "correct",
            "message": "S0 ok",
        },
        {
            "stage_id": "S1",
            "stage_output_validation_report": "stage output ok",
            "correctness_check_report": "correct",
            "message": "S1 ok",
        },
    ]
    assert recovery_payload["review_history"] == [
        {
            "action": "continue_next_stage",
            "next_stage": "S1",
            "message": "continue_next_stage",
        }
    ]
    assert "Do not return null in recovery mode." in recovery_payload["recovery_context"]

    snapshot_path = main.round0_stage_kernel_snapshot_path(solution_dir, "S1", 1)
    assert snapshot_path.exists()


def test_stage_review_payload_scopes_retry_histories_to_current_stage(
    tmp_path: Path,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    (solution_dir / "kernel_0.py").write_text(
        "def kernel():\n    return 'current'\n",
        encoding="utf-8",
    )

    plan_text = _sample_stage_plan()
    write_round0_stage_plan_fixture(tmp_path, plan_text)

    first_s2_attempt = Round0StageResult(
        stage_id="S2",
        stage_output_validation_report="S2 attempt 1 failed",
        stage_output_verified=False,
        final_correctness_verified=False,
        correctness_check_report="not run",
        status="validation_failed",
        message="S2 attempt 1",
    )
    second_s2_attempt = Round0StageResult(
        stage_id="S2",
        stage_output_validation_report="S2 attempt 2 passed",
        stage_output_verified=True,
        final_correctness_verified=False,
        correctness_check_report="not run",
        status="success",
        message="S2 attempt 2",
    )

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    ctx.round0_stage_history = [
        _stage_result("S0"),
        _stage_result("S1"),
        first_s2_attempt,
        second_s2_attempt,
    ]
    ctx.round0_review_history = [
        _review("S0", "continue_next_stage", next_stage="S1"),
        _review("S1", "continue_next_stage", next_stage="S2"),
        _review("S2", "retry_same_stage", next_stage="S2", message="retry S2"),
    ]

    review_payload = main._build_stage_review_payload(
        ctx=ctx,
        solution_dir=solution_dir,
        plan_text=plan_text,
        stage_result=second_s2_attempt,
        stage_attempt=2,
    )

    assert review_payload["stage_results_history"] == [
        {
            "stage_id": "S2",
            "stage_output_validation_report": "S2 attempt 1 failed",
            "correctness_check_report": "not run",
            "message": "S2 attempt 1",
        },
        {
            "stage_id": "S2",
            "stage_output_validation_report": "S2 attempt 2 passed",
            "correctness_check_report": "not run",
            "message": "S2 attempt 2",
        },
    ]
    assert review_payload["review_history"] == [
        {
            "action": "retry_same_stage",
            "next_stage": "S2",
            "message": "retry S2",
        }
    ]


def test_stage_agent_prompts_render_full_plan_and_no_file_diff_section(
    tmp_path: Path,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    plan_text = write_round0_stage_plan_fixture(tmp_path, _sample_stage_plan())
    (solution_dir / "kernel_0.py").write_text(
        "def kernel():\n    return 'current'\n",
        encoding="utf-8",
    )

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )

    stage_coder = make_round0_stage_coder(context=ctx)
    coder_static = main._resolve_agent_instructions_text(stage_coder, ctx)
    assert "{..## Assigned Stage..}" in coder_static
    main._apply_stage_coder_prompt_sections(
        ctx,
        main._build_stage_coder_payload(
            plan_text=plan_text,
            stage_id="S1",
        ),
    )
    coder_prompt = main._resolve_agent_instructions_text(stage_coder, ctx)
    assert "{..## Assigned Stage..}" not in coder_prompt
    assert "## Assigned Stage" in coder_prompt
    assert "## Full kernel_0_plan.md" in coder_prompt
    assert plan_text.strip() in coder_prompt
    assert "emit the `Round0StageResult` and STOP" in coder_prompt
    assert "[PyTorch] <name>: BEGIN" in coder_prompt
    assert "[CuTeDSL] <name>: BEGIN" in coder_prompt
    assert "YOU MUST USE cute.printf() and cute.print_tensor()" in coder_prompt
    assert extract_markdown_section(coder_prompt, "## References").splitlines() == [
        "## References",
        *[f"- `{path}`" for path in AGENT_SCOPES["coder"].read_allow],
    ]

    stage_fixer = make_round0_stage_fixer(context=ctx)
    main._apply_stage_fixer_prompt_sections(
        ctx,
        main._build_stage_fixer_payload(
            plan_text=plan_text,
            stage_id="S1",
            last_review=_review("S1", "retry_same_stage", next_stage="S1"),
        ),
    )
    fixer_prompt = main._resolve_agent_instructions_text(stage_fixer, ctx)
    assert "## Stage To Fix" in fixer_prompt
    assert "## Full kernel_0_plan.md" in fixer_prompt
    assert '"next_stage": "S1"' in fixer_prompt
    assert extract_markdown_section(fixer_prompt, "## References").splitlines() == [
        "## References",
        *[f"- `{path}`" for path in AGENT_SCOPES["coder"].read_allow],
    ]

    reviewer = make_round0_stage_reviewer(context=ctx)
    stage_result = _stage_result("S1")
    ctx.round0_stage_history = [stage_result]
    ctx.round0_review_history = [_review("S0", "continue_next_stage", next_stage="S1")]
    reviewer_payload = main._build_stage_review_payload(
        ctx=ctx,
        solution_dir=solution_dir,
        plan_text=plan_text,
        stage_result=stage_result,
        stage_attempt=1,
    )
    main._apply_stage_reviewer_prompt_sections(ctx, reviewer_payload)
    reviewer_prompt = main._resolve_agent_instructions_text(reviewer, ctx)
    assert "## Stage Under Review" in reviewer_prompt
    assert "## Full kernel_0_plan.md" in reviewer_prompt
    assert "## Kernel Snapshots" in reviewer_prompt
    assert "- diff_files" in reviewer_prompt
    assert "Use `grep_search` and `read_file` on `last_shell_dump.txt`" in reviewer_prompt
    assert "only include attempts for the current stage" in reviewer_prompt
    assert "## NVIDIA B200 (sm100a) Hardware Specifications" in reviewer_prompt
    assert extract_markdown_section(reviewer_prompt, "## References").splitlines() == [
        "## References",
        *[f"- `{path}`" for path in AGENT_SCOPES["reviewer"].read_allow],
    ]


def test_stage_agent_prompts_include_mlir_context_guidance(tmp_path: Path) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    plan_text = write_round0_stage_plan_fixture(tmp_path, _sample_stage_plan())
    (solution_dir / "kernel_0.py").write_text(
        "def kernel():\n    return 'current'\n",
        encoding="utf-8",
    )

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )

    stage_coder = make_round0_stage_coder(context=ctx)
    main._apply_stage_coder_prompt_sections(
        ctx,
        main._build_stage_coder_payload(
            plan_text=plan_text,
            stage_id="S2",
        ),
    )
    coder_prompt = main._resolve_agent_instructions_text(stage_coder, ctx)
    assert "MLIR CONTEXT ISSUES" in coder_prompt
    assert "@cute.jit" in coder_prompt
    assert "@cute.kernel" in coder_prompt
    assert "@cute.struct" in coder_prompt

    stage_fixer = make_round0_stage_fixer(context=ctx)
    main._apply_stage_fixer_prompt_sections(
        ctx,
        main._build_stage_fixer_payload(
            plan_text=plan_text,
            stage_id="S2",
            last_review=_review("S2", "retry_same_stage", next_stage="S2"),
        ),
    )
    fixer_prompt = main._resolve_agent_instructions_text(stage_fixer, ctx)
    assert "MLIR CONTEXT ISSUES" in fixer_prompt
    assert "@cute.jit" in fixer_prompt
    assert "@cute.kernel" in fixer_prompt
    assert "@cute.struct" in fixer_prompt


class _FakeAgent:
    def __init__(self, instructions: object) -> None:
        self.instructions = instructions


def _prepare_staged_round0_workspace(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_text: str,
) -> tuple[SharedContext, Path, Path]:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    solution_dir = tmp_path / "solution" / "dsa_attention"
    notes_dir = tmp_path / "notes" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    notes_dir.mkdir(parents=True)
    write_round0_stage_plan_fixture(tmp_path, plan_text)
    (solution_dir / "kernel_0.py").write_text(
        "def kernel():\n    return 'current'\n",
        encoding="utf-8",
    )

    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
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
        stage_id="S1",
        attempt=2,
        role_name="coder",
        payload={"step": "coder"},
    )

    coder_system = prompt_dump.runtime_dir / "stage_S1.attempt_02.coder.system.md"
    coder_input = prompt_dump.runtime_dir / "stage_S1.attempt_02.coder.input.json"
    assert coder_system.read_text(encoding="utf-8") == "runtime prompt"
    assert json.loads(coder_input.read_text(encoding="utf-8")) == {"step": "coder"}

    coder_input.write_text('{"step":"old"}', encoding="utf-8")
    main._dump_stage_runtime_prompt_artifacts(
        prompt_dump=prompt_dump,
        agent=agent,
        ctx=ctx,
        stage_id="S1",
        attempt=2,
        role_name="coder",
        payload={"step": "new"},
        reuse_if_present=True,
    )
    assert json.loads(coder_input.read_text(encoding="utf-8")) == {"step": "old"}


def test_runtime_prompt_dump_contains_full_plan_and_no_file_diffs(
    tmp_path: Path,
) -> None:
    plan_text = write_round0_stage_plan_fixture(tmp_path, _sample_stage_plan())
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )
    reviewer = make_round0_stage_reviewer(context=ctx)
    main._apply_stage_reviewer_prompt_sections(
        ctx,
        {
            "stage_under_review": "S1",
            "full_plan": plan_text,
            "kernel_snapshots": {
                "current_kernel_path": "solution/dsa_attention/kernel_0.py",
                "current_attempt_snapshot": "solution/dsa_attention/round0/S1.attempt_01.kernel_0.py",
                "previous_attempt_snapshot": None,
            },
            "stage_results_history": [],
            "review_history": [],
            "recovery_context": "",
        },
    )
    prompt_dump = main._make_prompt_dump_config(tmp_path / "prompts")

    main._dump_stage_runtime_prompt_artifacts(
        prompt_dump=prompt_dump,
        agent=reviewer,
        ctx=ctx,
        stage_id="S1",
        attempt=1,
        role_name="reviewer",
        payload={
            "stage_under_review": "S1",
            "full_plan": plan_text,
            "kernel_snapshots": {
                "current_kernel_path": "solution/dsa_attention/kernel_0.py",
                "current_attempt_snapshot": "solution/dsa_attention/round0/S1.attempt_01.kernel_0.py",
                "previous_attempt_snapshot": None,
            },
            "stage_results_history": [],
            "review_history": [],
            "recovery_context": "",
        },
    )

    system_path = prompt_dump.runtime_dir / "stage_S1.attempt_01.reviewer.system.md"
    input_path = prompt_dump.runtime_dir / "stage_S1.attempt_01.reviewer.input.json"
    assert "## Full kernel_0_plan.md" in system_path.read_text(encoding="utf-8")
    assert "## Kernel Snapshots" in system_path.read_text(encoding="utf-8")
    assert plan_text.strip() in system_path.read_text(encoding="utf-8")
    assert "File Diffs" not in system_path.read_text(encoding="utf-8")
    assert "kernel_snapshots" in input_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_recover_round0_next_stage_with_reviewer_returns_concrete_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_text = write_round0_stage_plan_fixture(tmp_path, _sample_stage_plan())
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )

    async def _fake_run_agent(**_: Any) -> tuple[_FakeStageRunResult, float]:
        return _FakeStageRunResult(
            _review("Recovery", "continue_next_stage", next_stage="S1")
        ), 0.1

    monkeypatch.setattr(main, "_run_agent", _fake_run_agent)

    review = await main._recover_round0_next_stage_with_reviewer(
        ctx=ctx,
        plan_text=plan_text,
        stage_reviewer=_FakeAgent("reviewer"),
        reviewer_max_turns=1,
        reviewer_verbose=False,
        prompt_dump=None,
    )

    assert review.next_stage == "S1"


@pytest.mark.asyncio
async def test_recover_round0_next_stage_with_reviewer_rejects_null_next_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_text = write_round0_stage_plan_fixture(tmp_path, _sample_stage_plan())
    ctx = SharedContext(
        project_root=str(tmp_path),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
    )

    async def _fake_run_agent(**_: Any) -> tuple[_FakeStageRunResult, float]:
        return _FakeStageRunResult(
            _review("Recovery", "continue_next_stage", next_stage=None)
        ), 0.1

    monkeypatch.setattr(main, "_run_agent", _fake_run_agent)

    with pytest.raises(ValueError, match="concrete next_stage"):
        await main._recover_round0_next_stage_with_reviewer(
            ctx=ctx,
            plan_text=plan_text,
            stage_reviewer=_FakeAgent("reviewer"),
            reviewer_max_turns=1,
            reviewer_verbose=False,
            prompt_dump=None,
        )


def test_run_round0_staged_first_attempt_uses_coder_and_copies_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, solution_dir, state_path = _prepare_staged_round0_workspace(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        plan_text=_single_stage_plan(),
    )
    stage_labels: list[str] = []

    async def _fake_run_agent(**kwargs: Any) -> tuple[_FakeStageRunResult, float]:
        stage_labels.append(kwargs["stage_label"])
        if kwargs["stage_label"].startswith("kernel-stage-coder-"):
            return _FakeStageRunResult(
                _stage_result("S0", final_correctness_verified=True)
            ), 0.1
        if kwargs["stage_label"].startswith("kernel-stage-reviewer-"):
            return _FakeStageRunResult(
                _review("S0", "continue_next_stage", next_stage=None)
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
    assert (solution_dir / "kernel.py").read_text(encoding="utf-8") == (
        solution_dir / "kernel_0.py"
    ).read_text(encoding="utf-8")


def test_run_round0_staged_retry_uses_fixer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, solution_dir, state_path = _prepare_staged_round0_workspace(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        plan_text=_single_stage_plan(),
    )
    ctx.round0_stage_history = [_stage_result("S0")]
    ctx.round0_review_history = [_review("S0", "retry_same_stage", next_stage="S0")]
    stage_labels: list[str] = []

    async def _fake_run_agent(**kwargs: Any) -> tuple[_FakeStageRunResult, float]:
        stage_labels.append(kwargs["stage_label"])
        if kwargs["stage_label"].startswith("kernel-stage-fixer-"):
            return _FakeStageRunResult(
                _stage_result("S0", final_correctness_verified=True)
            ), 0.1
        if kwargs["stage_label"].startswith("kernel-stage-reviewer-"):
            return _FakeStageRunResult(
                _review("S0", "continue_next_stage", next_stage=None)
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
            [_stage_result("S0", final_correctness_verified=True)],
            [],
            "coder",
            1,
        ),
        (
            [
                _stage_result("S0"),
                _stage_result("S0", final_correctness_verified=True),
            ],
            [_review("S0", "retry_same_stage", next_stage="S0")],
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
        plan_text=_single_stage_plan(),
    )
    ctx.round0_stage_history = stage_history
    ctx.round0_review_history = review_history
    prompt_dump = main._make_prompt_dump_config(tmp_path / "prompts")

    async def _fake_run_agent(**kwargs: Any) -> tuple[_FakeStageRunResult, float]:
        assert kwargs["stage_label"].startswith("kernel-stage-reviewer-")
        return _FakeStageRunResult(
            _review("S0", "continue_next_stage", next_stage=None)
        ), 0.1

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
        / f"stage_S0.attempt_{expected_attempt:02d}.{expected_role}.system.md"
    )
    unexpected_role = "fixer" if expected_role == "coder" else "coder"
    unexpected_path = (
        prompt_dump.runtime_dir
        / f"stage_S0.attempt_{expected_attempt:02d}.{unexpected_role}.system.md"
    )

    assert expected_path.read_text(encoding="utf-8") == f"{expected_role} prompt"
    assert not unexpected_path.exists()
    assert result.status == "success"


def test_run_round0_staged_resume_pending_review_scopes_reviewer_payload_to_current_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, solution_dir, state_path = _prepare_staged_round0_workspace(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        plan_text=_sample_stage_plan(),
    )
    pending_stage = _stage_result("S1", final_correctness_verified=True)
    ctx.round0_stage_history = [_stage_result("S0"), pending_stage]
    ctx.round0_review_history = [_review("S0", "continue_next_stage", next_stage="S1")]
    prompt_dump = main._make_prompt_dump_config(tmp_path / "prompts")

    async def _fake_run_agent(**kwargs: Any) -> tuple[_FakeStageRunResult, float]:
        assert kwargs["stage_label"] == "kernel-stage-reviewer-S1"
        return _FakeStageRunResult(
            _review("S1", "continue_next_stage", next_stage=None)
        ), 0.1

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

    reviewer_input_path = (
        prompt_dump.runtime_dir / "stage_S1.attempt_01.reviewer.input.json"
    )
    reviewer_input = json.loads(reviewer_input_path.read_text(encoding="utf-8"))

    assert reviewer_input["stage_results_history"] == [
        {
            "stage_id": "S1",
            "stage_output_validation_report": "stage output ok",
            "correctness_check_report": "correct",
            "message": "S1 ok",
        }
    ]
    assert reviewer_input["review_history"] == []
    assert result.status == "success"


def test_run_round0_staged_resume_next_stage_scopes_first_reviewer_payload_to_upcoming_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, solution_dir, state_path = _prepare_staged_round0_workspace(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        plan_text=_sample_stage_plan(),
    )
    ctx.round0_stage_history = [_stage_result("S0")]
    ctx.round0_review_history = [_review("S0", "continue_next_stage", next_stage="S1")]
    prompt_dump = main._make_prompt_dump_config(tmp_path / "prompts")
    stage_labels: list[str] = []

    async def _fake_run_agent(**kwargs: Any) -> tuple[_FakeStageRunResult, float]:
        stage_labels.append(kwargs["stage_label"])
        if kwargs["stage_label"] == "kernel-stage-coder-S1":
            return _FakeStageRunResult(
                _stage_result("S1", final_correctness_verified=True)
            ), 0.1
        if kwargs["stage_label"] == "kernel-stage-reviewer-S1":
            return _FakeStageRunResult(
                _review("S1", "continue_next_stage", next_stage=None)
            ), 0.1
        raise AssertionError(f"Unexpected stage label: {kwargs['stage_label']}")

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

    reviewer_input_path = (
        prompt_dump.runtime_dir / "stage_S1.attempt_01.reviewer.input.json"
    )
    reviewer_input = json.loads(reviewer_input_path.read_text(encoding="utf-8"))

    assert stage_labels == ["kernel-stage-coder-S1", "kernel-stage-reviewer-S1"]
    assert not (prompt_dump.runtime_dir / "stage_recovery.attempt_00.reviewer.input.json").exists()
    assert reviewer_input["stage_results_history"] == [
        {
            "stage_id": "S1",
            "stage_output_validation_report": "stage output ok",
            "correctness_check_report": "correct",
            "message": "S1 ok",
        }
    ]
    assert reviewer_input["review_history"] == []
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
