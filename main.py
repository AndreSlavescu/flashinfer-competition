"""Multi-agent kernel generation loop for DSA sparse attention optimization.

Usage:
    # Full loop with 5 optimization rounds:
    python main.py --num-rounds 5

    # Quick test with 1 round:
    python main.py --num-rounds 1

    # Resume from last checkpoint:
    python main.py --num-rounds 5 --resume

    # Use a different model:
    python main.py --num-rounds 5 --model gpt-4.1
"""

from __future__ import annotations

from bootstrap_runtime import load_repo_dotenv, require_dependencies

require_dependencies(
    {
        "agents": "openai-agents",
        "dotenv": "python-dotenv",
        "modal": "modal",
        "openai": "openai",
        "pydantic": "pydantic",
    },
    entrypoint="main.py",
    include_env_hint=True,
)
load_repo_dotenv()

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from agents import (
    ModelRetrySettings,
    ModelSettings,
    RunConfig,
    Runner,
    RunContextWrapper,
    SQLiteSession,
    retry_policies,
)
from agents.exceptions import MaxTurnsExceeded
from agents.run_config import CallModelData, ModelInputData

from kernel_agents.context import (
    CODEX_WORKER_MODE_CHOICES,
    CODEX_WORKER_REASONING_EFFORT_CHOICES,
    QUALITY_PROFILE_CHOICES,
    REASONING_EFFORT_CHOICES,
    ROUND0_MODE_CHOICES,
    VERBOSITY_CHOICES,
    CoderResult,
    CodexWorkerMode,
    CodexWorkerReasoningEffort,
    DesignerResult,
    ImplementationGraph,
    OptimizerResult,
    PlannerResult,
    QualityProfile,
    Round0Mode,
    Round0StageResult,
    RoundRecord,
    SharedContext,
    StageReviewResult,
    tool_limits_for_profile,
)
from kernel_agents.kernel_coder import make_kernel_coder
from kernel_agents.kernel_designer import make_kernel_designer
from kernel_agents.kernel_optimizer import make_kernel_optimizer
from kernel_agents.kernel_planner import make_kernel_planner
from kernel_agents.kernel_stage_coder import make_round0_stage_coder
from kernel_agents.kernel_stage_reviewer import make_round0_stage_reviewer
from kernel_agents.stream_logging import consume_streamed_run

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()
PUBLIC_CODEX_COMPACTION_THRESHOLD = 200_000
STAGE_RATE_LIMIT_MAX_RETRIES = 100
STAGE_RATE_LIMIT_MAX_CUMULATIVE_WAIT_S = 600.0
RATE_LIMIT_RETRY_SAFETY_BUFFER_S = 0.5
RATE_LIMIT_RETRY_BACKOFF_INITIAL_S = 2.0
RATE_LIMIT_RETRY_BACKOFF_MAX_S = 30.0
_RATE_LIMIT_RETRY_AFTER_RE = re.compile(
    r"please try again in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)
_RATE_LIMIT_MESSAGE_PATTERNS = (
    "rate limit reached",
    "too many requests",
    "tokens per min",
    "requests per min",
    "(tpm)",
    "(rpm)",
)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def _dump_round0_stage_history(history: list[Round0StageResult]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in history]


def _dump_round0_review_history(history: list[StageReviewResult]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in history]


def _load_round0_stage_history(items: list[object]) -> list[Round0StageResult]:
    return [Round0StageResult.model_validate(item) for item in items]


def _load_round0_review_history(items: list[object]) -> list[StageReviewResult]:
    return [StageReviewResult.model_validate(item) for item in items]


def save_state(ctx: SharedContext, path: Path) -> None:
    """Persist loop state to JSON for resumability."""
    state = {
        "current_round": ctx.current_round,
        "best_latency_ms": ctx.best_latency_ms,
        "best_round": ctx.best_round,
        "model_name": ctx.model_name,
        "quality_profile": ctx.quality_profile,
        "codex_worker_mode": ctx.codex_worker_mode,
        "codex_thread_id_coder_engineer": ctx.codex_thread_id_coder_engineer,
        "codex_thread_id_optimizer_engineer": ctx.codex_thread_id_optimizer_engineer,
        "history": [asdict(r) for r in ctx.history],
        "round0_mode": ctx.round0_mode,
        "round0_impl_graph_file": ctx.round0_impl_graph_file,
        "round0_stage_history": _dump_round0_stage_history(ctx.round0_stage_history),
        "round0_review_history": _dump_round0_review_history(ctx.round0_review_history),
    }
    path.write_text(json.dumps(state, indent=2))
    logger.info("State saved to %s", path)


def _round_record_from_dict(d: dict) -> RoundRecord:
    """Build a RoundRecord from a JSON dict, tolerating old/missing fields."""
    known = {f.name for f in RoundRecord.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in known}
    return RoundRecord(**filtered)


def load_state(ctx: SharedContext, path: Path) -> int:
    """Load loop state from JSON. Returns the last completed round number."""
    state = json.loads(path.read_text())
    ctx.best_latency_ms = state["best_latency_ms"]
    ctx.best_round = state["best_round"]
    ctx.history = [_round_record_from_dict(r) for r in state["history"]]
    ctx.codex_thread_id_coder_engineer = state.get("codex_thread_id_coder_engineer")
    ctx.codex_thread_id_optimizer_engineer = state.get("codex_thread_id_optimizer_engineer")
    ctx.round0_mode = state.get("round0_mode", ctx.round0_mode)
    ctx.round0_impl_graph_file = state.get("round0_impl_graph_file", "")
    ctx.round0_stage_history = _load_round0_stage_history(state.get("round0_stage_history", []))
    ctx.round0_review_history = _load_round0_review_history(state.get("round0_review_history", []))
    last_round = state["current_round"]
    logger.info("Resumed from round %d (best=%.3fms @ round %d)",
                last_round, ctx.best_latency_ms, ctx.best_round)
    return last_round


# ---------------------------------------------------------------------------
# History formatting
# ---------------------------------------------------------------------------

def format_history(history: list[RoundRecord], best_latency_ms: float = float("inf")) -> str:
    """Build a human-readable history summary for agent prompts."""
    if not history:
        return "(No prior rounds — this is the first optimization.)"
    lines = []
    if best_latency_ms < float("inf"):
        lines.append(f"  Current best latency: {best_latency_ms:.3f}ms\n")
    for r in history:
        status = "CORRECT" if r.correctness else "FAILED"
        lat = f"{r.latency_ms:.3f}ms" if r.latency_ms is not None else "N/A"
        line = (
            f"  Round {r.round_num}: [{status}] latency={lat} | "
            f"bottleneck='{r.bottleneck}' | strategy='{r.strategy_summary}'"
        )
        if r.failure_reason:
            line += f"\n    Failure reason: {r.failure_reason}"
        if r.strategy_file:
            line += f"\n    Strategy file: {r.strategy_file}"
        if r.ncu_metrics:
            line += f"\n    NCU: {json.dumps(r.ncu_metrics, separators=(',', ':'))}"
        lines.append(line)
    return "\n".join(lines)


def find_last_correct_kernel(ctx: SharedContext, solution_dir: Path) -> Path:
    """Find the most recent kernel that passed correctness validation."""
    # Walk history backwards
    for r in reversed(ctx.history):
        if r.correctness:
            p = solution_dir / r.kernel_file
            if p.exists():
                return p
    # Fallback to kernel_0.py
    k0 = solution_dir / "kernel_0.py"
    if k0.exists():
        return k0
    # Last resort: current kernel.py
    return solution_dir / "kernel.py"


def round0_artifacts_dir(solution_dir: Path) -> Path:
    """Return the staged round-0 artifact directory."""
    return solution_dir / "round0"


def round0_impl_graph_path(solution_dir: Path) -> Path:
    """Return the canonical path for the staged round-0 implementation graph."""
    return round0_artifacts_dir(solution_dir) / "kernel_0_impl_graph.json"


def round0_plan_path(solution_dir: Path) -> Path:
    """Return the canonical path for the staged round-0 human plan."""
    return solution_dir / "kernel_0_plan.md"


def _slugify_stage_id(stage_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_id).strip("_") or "stage"


def validate_impl_graph(graph: ImplementationGraph) -> None:
    """Validate basic graph invariants needed by the staged round-0 loop."""
    stage_ids = [stage.stage_id for stage in graph.stages]
    if not stage_ids:
        raise ValueError("Implementation graph must define at least one stage.")
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("Implementation graph contains duplicate stage IDs.")

    stage_index = {stage_id: index for index, stage_id in enumerate(stage_ids)}
    stage_id_set = set(stage_ids)

    for stage in graph.stages:
        missing = [dep for dep in stage.prerequisites if dep not in stage_id_set]
        if missing:
            raise ValueError(
                f"Stage '{stage.stage_id}' references unknown prerequisites: {missing}"
            )
        if not stage.debug_exports:
            raise ValueError(
                f"Stage '{stage.stage_id}' must declare at least one debug export for frontier validation."
            )
        late = [
            dep for dep in stage.prerequisites if stage_index[dep] >= stage_index[stage.stage_id]
        ]
        if late:
            raise ValueError(
                f"Stage '{stage.stage_id}' prerequisites must appear earlier in declared order: {late}"
            )

    for pipeline in graph.async_pipelines:
        missing = [
            stage_id
            for stage_id in pipeline.participating_stages
            if stage_id not in stage_id_set
        ]
        if missing:
            raise ValueError(
                f"Async pipeline '{pipeline.pipeline_id}' references unknown stages: {missing}"
            )


def iter_stage_order(graph: ImplementationGraph) -> list[str]:
    """Return the canonical stage order from the graph."""
    validate_impl_graph(graph)
    return [stage.stage_id for stage in graph.stages]


def get_stage_spec(graph: ImplementationGraph, stage_id: str):
    """Return the stage spec for *stage_id* or raise KeyError."""
    for stage in graph.stages:
        if stage.stage_id == stage_id:
            return stage
    raise KeyError(stage_id)


def relevant_async_pipelines(graph: ImplementationGraph, stage_id: str) -> list[dict[str, Any]]:
    """Return async pipeline specs that reference *stage_id*."""
    pipelines = []
    for pipeline in graph.async_pipelines:
        if stage_id in pipeline.participating_stages:
            pipelines.append(pipeline.model_dump(mode="json"))
    return pipelines


def _approved_frontier_summaries(
    *,
    graph: ImplementationGraph,
    completed_stage_ids: list[str],
    latest_stage_results: dict[str, Round0StageResult],
) -> list[dict[str, Any]]:
    """Return compact summaries for previously approved validation frontiers."""
    summaries: list[dict[str, Any]] = []
    for completed_stage_id in completed_stage_ids:
        completed_stage = get_stage_spec(graph, completed_stage_id)
        last_result = latest_stage_results[completed_stage_id]
        summaries.append(
            {
                "stage_id": completed_stage.stage_id,
                "title": completed_stage.title,
                "outputs": completed_stage.outputs,
                "debug_exports": completed_stage.debug_exports,
                "checks": completed_stage.checks,
                "frontier_verified": last_result.frontier_verified,
                "frontier_validation_report": last_result.frontier_validation_report,
            }
        )
    return summaries


@dataclass
class Round0DerivedProgress:
    """Derived staged round-0 progress reconstructed from append-only histories."""

    completed_stage_ids: list[str]
    latest_stage_results: dict[str, Round0StageResult]
    attempt_counts: dict[str, int]
    pending_stage_result: Round0StageResult | None
    pending_review: bool
    next_stage_id: str | None
    latest_review: StageReviewResult | None
    complete: bool


def _derive_round0_progress(ctx: SharedContext, graph: ImplementationGraph) -> Round0DerivedProgress:
    order = iter_stage_order(graph)
    order_index = {stage_id: index for index, stage_id in enumerate(order)}

    if len(ctx.round0_review_history) > len(ctx.round0_stage_history):
        raise ValueError("round0 review history cannot be longer than stage history.")
    if len(ctx.round0_stage_history) > len(ctx.round0_review_history) + 1:
        raise ValueError("round0 stage history can have at most one unreviewed stage attempt.")

    completed_stage_ids: list[str] = []
    latest_stage_results: dict[str, Round0StageResult] = {}
    attempt_counts: dict[str, int] = {}
    latest_review: StageReviewResult | None = None
    complete = False

    for stage_result in ctx.round0_stage_history:
        latest_stage_results[stage_result.stage_id] = stage_result
        attempt_counts[stage_result.stage_id] = attempt_counts.get(stage_result.stage_id, 0) + 1

    reviewed_attempt_count = len(ctx.round0_review_history)
    for index in range(reviewed_attempt_count):
        stage_result = ctx.round0_stage_history[index]
        review = ctx.round0_review_history[index]
        if review.stage_id != stage_result.stage_id:
            raise ValueError(
                "round0 stage/review history is out of sync: "
                f"stage attempt {index} is {stage_result.stage_id!r}, "
                f"review is {review.stage_id!r}."
            )
        latest_review = review

        if review.action == "continue_next_stage":
            if stage_result.stage_id in order_index and stage_result.stage_id not in completed_stage_ids:
                completed_stage_ids.append(stage_result.stage_id)
            if (
                stage_result.stage_id == order[-1]
                and stage_result.final_correctness_verified
            ):
                complete = True
        elif review.action == "retry_same_stage":
            continue
        elif review.action == "revise_design_then_retry":
            restart_from_stage_id = review.restart_from_stage_id
            if restart_from_stage_id is None:
                raise ValueError(
                    "round0 review history requested a design revision without restart_from_stage_id."
                )
            restart_index = order_index.get(restart_from_stage_id)
            if restart_index is None:
                raise ValueError(
                    f"restart_from_stage_id {restart_from_stage_id!r} is not present in the current graph."
                )
            completed_stage_ids = [
                stage_id
                for stage_id in completed_stage_ids
                if order_index[stage_id] < restart_index
            ]
            complete = False
        else:
            raise ValueError(f"Unknown round0 review action {review.action!r}")

    pending_stage_result = None
    pending_review = False
    if len(ctx.round0_stage_history) > reviewed_attempt_count:
        pending_stage_result = ctx.round0_stage_history[-1]
        pending_review = True

    next_stage_id = None
    if not complete:
        if pending_stage_result is not None:
            next_stage_id = pending_stage_result.stage_id
        else:
            next_stage_id = _next_incomplete_stage_id(order, completed_stage_ids)

    return Round0DerivedProgress(
        completed_stage_ids=completed_stage_ids,
        latest_stage_results=latest_stage_results,
        attempt_counts=attempt_counts,
        pending_stage_result=pending_stage_result,
        pending_review=pending_review,
        next_stage_id=next_stage_id,
        latest_review=latest_review,
        complete=complete,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_impl_graph(path: Path) -> ImplementationGraph:
    """Load and validate an implementation graph from disk."""
    graph = ImplementationGraph.model_validate(json.loads(path.read_text()))
    validate_impl_graph(graph)
    return graph


TStructuredOutput = TypeVar("TStructuredOutput")


def require_structured_output(result: object, cls: type[TStructuredOutput]) -> TStructuredOutput:
    """Extract a typed final output from an Agents SDK run result."""
    final_output_as = getattr(result, "final_output_as", None)
    if not callable(final_output_as):
        raise TypeError(f"Run result does not expose final_output_as() for {cls.__name__}")
    return final_output_as(cls, raise_if_incorrect_type=True)


def format_run_telemetry(label: str, elapsed_s: float, result: object) -> str:
    """Format wall-clock and token usage data for a completed agent run."""
    parts = [f"{label} telemetry: wall_time_s={elapsed_s:.1f}"]

    context_wrapper = getattr(result, "context_wrapper", None)
    usage = getattr(context_wrapper, "usage", None)
    if usage is None:
        return ", ".join(parts)

    requests = getattr(usage, "requests", None)
    if requests is not None:
        parts.append(f"requests={requests}")

    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is not None:
        parts.append(f"total_tokens={total_tokens}")

    output_tokens_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(output_tokens_details, "reasoning_tokens", None)
    if reasoning_tokens is not None:
        parts.append(f"reasoning_tokens={reasoning_tokens}")

    return ", ".join(parts)

def _is_user_message_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") == "message":
        return item.get("role") == "user"
    return item.get("role") == "user" and "content" in item


def _is_compaction_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "compaction"


def _iter_exception_chain(error: BaseException) -> list[BaseException]:
    """Return the exception plus chained causes/contexts without cycles."""
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    chain: list[BaseException] = []

    while pending:
        current = pending.pop(0)
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        chain.append(current)

        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)

        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            pending.append(context)

    return chain


def _is_rate_limit_error(error: BaseException) -> bool:
    """Best-effort detection for OpenAI/OpenAI SDK rate-limit failures."""
    for candidate in _iter_exception_chain(error):
        status_code = getattr(candidate, "status_code", None)
        if status_code == 429:
            return True

        message = str(candidate).lower()
        if any(pattern in message for pattern in _RATE_LIMIT_MESSAGE_PATTERNS):
            return True

    return False


def _rate_limit_retry_after_seconds(error: BaseException) -> float | None:
    """Extract retry-after seconds from a rate-limit exception string."""
    for candidate in _iter_exception_chain(error):
        message = str(candidate)
        match = _RATE_LIMIT_RETRY_AFTER_RE.search(message)
        if not match:
            continue
        try:
            return max(float(match.group(1)), 0.0)
        except ValueError:
            continue
    return None


def _stage_rate_limit_wait_seconds(error: BaseException, retry_index: int) -> float:
    """Resolve the outer retry delay after SDK-managed retries are exhausted."""
    hinted_delay = _rate_limit_retry_after_seconds(error)
    if hinted_delay is not None:
        return hinted_delay + RATE_LIMIT_RETRY_SAFETY_BUFFER_S

    fallback_delay = min(
        RATE_LIMIT_RETRY_BACKOFF_INITIAL_S * (2 ** retry_index),
        RATE_LIMIT_RETRY_BACKOFF_MAX_S,
    )
    return fallback_delay + RATE_LIMIT_RETRY_SAFETY_BUFFER_S


def public_codex_input_filter(filter_payload: CallModelData[SharedContext]) -> ModelInputData:
    """Drop transcript items that predate the latest compaction item."""
    model_data = filter_payload.model_data
    latest_compaction_index = None
    for index, item in enumerate(model_data.input):
        if _is_compaction_item(item):
            latest_compaction_index = index

    if latest_compaction_index is None:
        return model_data

    preserved_user_items = [
        item
        for item in model_data.input[:latest_compaction_index]
        if _is_user_message_item(item)
    ]
    compacted_tail = model_data.input[latest_compaction_index:]
    return ModelInputData(
        input=[*preserved_user_items, *compacted_tail],
        instructions=model_data.instructions,
    )


def make_run_config(quality_profile: QualityProfile = "public_codex") -> RunConfig:
    """Build a shared RunConfig for long tool-using agent runs."""
    retry_settings = ModelRetrySettings(
        max_retries=4,
        backoff={
            "initial_delay": 0.5,
            "max_delay": 20.0,
            "multiplier": 2.0,
            "jitter": True,
        },
        policy=retry_policies.any(
            retry_policies.provider_suggested(),
            retry_policies.retry_after(),
            retry_policies.network_error(),
            retry_policies.http_status([408, 409, 429, 500, 502, 503, 504]),
        ),
    )
    model_settings = ModelSettings(retry=retry_settings)
    call_model_input_filter = None
    if quality_profile == "public_codex":
        model_settings = model_settings.resolve(
            ModelSettings(
                parallel_tool_calls=False,
                truncation="auto",
                store=False,
                response_include=["reasoning.encrypted_content"],
                extra_args={
                    "context_management": [
                        {
                            "type": "compaction",
                            "compact_threshold": PUBLIC_CODEX_COMPACTION_THRESHOLD,
                        }
                    ]
                },
            )
        )
        call_model_input_filter = public_codex_input_filter
    return RunConfig(
        model_settings=model_settings,
        call_model_input_filter=call_model_input_filter,
    )


async def _run_agent_once(
    *,
    starting_agent: object,
    input: str | list[dict[str, Any]],
    context: SharedContext,
    max_turns: int,
    verbose: bool,
    session: SQLiteSession | None = None,
) -> tuple[object, float]:
    """Run an agent normally or via the streamed progress path."""
    started_at = time.perf_counter()
    run_config = make_run_config(context.quality_profile)
    if not verbose:
        result = await Runner.run(
            starting_agent=starting_agent,
            input=input,
            context=context,
            max_turns=max_turns,
            run_config=run_config,
            session=session,
        )
        return result, time.perf_counter() - started_at

    result = Runner.run_streamed(
        starting_agent=starting_agent,
        input=input,
        context=context,
        max_turns=max_turns,
        run_config=run_config,
        session=session,
    )
    return await consume_streamed_run(result), time.perf_counter() - started_at


async def _run_agent(
    *,
    stage_label: str,
    starting_agent: object,
    input: str,
    context: SharedContext,
    max_turns: int,
    verbose: bool,
    max_rate_limit_retries: int = STAGE_RATE_LIMIT_MAX_RETRIES,
) -> tuple[object, float]:
    """Run an agent with stage-local rate-limit recovery using session memory."""
    started_at = time.perf_counter()
    session = SQLiteSession(f"{stage_label}-{time.time_ns()}")
    current_input: str | list[dict[str, Any]] = input
    rate_limit_retries = 0
    cumulative_wait_s = 0.0

    while True:
        try:
            result, _ = await _run_agent_once(
                starting_agent=starting_agent,
                input=current_input,
                context=context,
                max_turns=max_turns,
                verbose=verbose,
                session=session,
            )
            return result, time.perf_counter() - started_at
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise

            wait_s = _stage_rate_limit_wait_seconds(exc, rate_limit_retries)
            if (
                rate_limit_retries >= max_rate_limit_retries
                or cumulative_wait_s + wait_s > STAGE_RATE_LIMIT_MAX_CUMULATIVE_WAIT_S
            ):
                raise

            next_attempt = rate_limit_retries + 2
            max_attempts = max_rate_limit_retries + 1
            print(
                f"WARNING: {stage_label} hit a rate limit after SDK retries. "
                f"Waiting {wait_s:.2f}s before retry {next_attempt}/{max_attempts} "
                f"with saved session context (cumulative wait {cumulative_wait_s + wait_s:.2f}s)."
            )
            await asyncio.sleep(wait_s)
            current_input = []
            rate_limit_retries += 1
            cumulative_wait_s += wait_s


# ---------------------------------------------------------------------------
# Prompt dumping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptDumpConfig:
    """Filesystem destinations for static and runtime prompt dumps."""

    static_dir: Path
    runtime_dir: Path


def _resolve_agent_instructions_text(agent: object, ctx: SharedContext) -> str:
    """Resolve an agent's instructions into a concrete string."""
    instructions = getattr(agent, "instructions", None)
    run_ctx = RunContextWrapper(context=ctx)
    if callable(instructions):
        return instructions(run_ctx, agent)
    if isinstance(instructions, str):
        return instructions
    return f"<agent.instructions is {type(instructions).__name__}>"


def _make_prompt_dump_config(output_dir: Path) -> PromptDumpConfig:
    """Build the prompt dump directory layout."""
    return PromptDumpConfig(
        static_dir=output_dir,
        runtime_dir=output_dir / "round0_runtime",
    )


def _build_prompt_dump_agents(
    *,
    ctx: SharedContext,
    model: str,
    coder_model: str,
    designer_model: str,
    codex_worker_model: str,
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort,
    coder_reasoning_effort: str,
    coder_verbosity: str,
    designer_reasoning_effort: str,
    designer_verbosity: str,
    planner_reasoning_effort: str,
    optimizer_reasoning_effort: str,
) -> dict[str, object]:
    """Construct the agent set whose prompts can be dumped."""
    return {
        "kernel_designer": make_kernel_designer(
            context=ctx,
            model=designer_model,
            reasoning_effort=designer_reasoning_effort,  # type: ignore[arg-type]
            verbosity=designer_verbosity,  # type: ignore[arg-type]
        ),
        "kernel_coder": make_kernel_coder(
            context=ctx,
            model=coder_model,
            reasoning_effort=coder_reasoning_effort,  # type: ignore[arg-type]
            verbosity=coder_verbosity,  # type: ignore[arg-type]
            codex_worker_model=codex_worker_model,
            codex_worker_reasoning_effort=codex_worker_reasoning_effort,
        ),
        "kernel_stage_coder": make_round0_stage_coder(
            context=ctx,
            model=coder_model,
            reasoning_effort=coder_reasoning_effort,  # type: ignore[arg-type]
            verbosity=coder_verbosity,  # type: ignore[arg-type]
            codex_worker_model=codex_worker_model,
            codex_worker_reasoning_effort=codex_worker_reasoning_effort,
        ),
        "kernel_stage_reviewer": make_round0_stage_reviewer(
            context=ctx,
            model=designer_model,
            reasoning_effort=designer_reasoning_effort,  # type: ignore[arg-type]
            verbosity=designer_verbosity,  # type: ignore[arg-type]
        ),
        "kernel_planner": make_kernel_planner(
            context=ctx,
            model=model,
            reasoning_effort=planner_reasoning_effort,  # type: ignore[arg-type]
        ),
        "kernel_optimizer": make_kernel_optimizer(
            context=ctx,
            model=model,
            reasoning_effort=optimizer_reasoning_effort,  # type: ignore[arg-type]
            codex_worker_model=codex_worker_model,
            codex_worker_reasoning_effort=codex_worker_reasoning_effort,
        ),
    }


def dump_all_prompts(
    *,
    output_dir: Path,
    model: str,
    coder_model: str,
    designer_model: str,
    quality_profile: QualityProfile,
    codex_worker_mode: CodexWorkerMode,
    codex_worker_model: str,
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort,
    coder_reasoning_effort: str,
    coder_verbosity: str,
    designer_reasoning_effort: str,
    designer_verbosity: str,
    planner_reasoning_effort: str,
    optimizer_reasoning_effort: str,
) -> None:
    """Resolve and write each agent's composed system prompt to `output_dir`.

    Builds a SharedContext identical to what `run_loop` would construct, wires
    up the round-0 and optimization agents with the same knobs, and writes each resolved prompt
    (dynamic callables invoked) to `output_dir/<agent_name>.md`.
    """
    prompt_dump = _make_prompt_dump_config(output_dir)
    prompt_dump.static_dir.mkdir(parents=True, exist_ok=True)

    ctx = SharedContext(
        project_root=str(PROJECT_ROOT),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
        model_name=model,
        quality_profile=quality_profile,
        tool_limits=tool_limits_for_profile(quality_profile),
        codex_worker_mode=codex_worker_mode,
    )

    agents = _build_prompt_dump_agents(
        ctx=ctx,
        model=model,
        coder_model=coder_model,
        designer_model=designer_model,
        codex_worker_model=codex_worker_model,
        codex_worker_reasoning_effort=codex_worker_reasoning_effort,
        coder_reasoning_effort=coder_reasoning_effort,
        coder_verbosity=coder_verbosity,
        designer_reasoning_effort=designer_reasoning_effort,
        designer_verbosity=designer_verbosity,
        planner_reasoning_effort=planner_reasoning_effort,
        optimizer_reasoning_effort=optimizer_reasoning_effort,
    )

    print(f"Dumping prompts to {prompt_dump.static_dir.relative_to(PROJECT_ROOT)}/")
    for name, agent in agents.items():
        try:
            text = _resolve_agent_instructions_text(agent, ctx)
        except FileNotFoundError as exc:
            text = (
                f"<could not resolve dynamic instructions: {exc}>\n"
                "Run kernel-designer first (or drop --skip-designer) to produce "
                "the inputs this agent reads at run time."
            )
            print(f"  WARNING: {name}: {exc}")
        out_path = prompt_dump.static_dir / f"{name}.md"
        out_path.write_text(text)
        print(
            f"  {out_path.relative_to(PROJECT_ROOT)}  "
            f"({len(text):,} chars, {len(text.splitlines())} lines)"
        )


def _dump_stage_runtime_prompt_artifacts(
    *,
    prompt_dump: PromptDumpConfig | None,
    agent: object,
    ctx: SharedContext,
    stage_id: str,
    attempt: int,
    role_name: str,
    payload: dict[str, Any],
    reuse_if_present: bool = False,
) -> None:
    """Write the fully materialized stage prompt artifacts for one attempt."""
    if prompt_dump is None:
        return

    prompt_dump.runtime_dir.mkdir(parents=True, exist_ok=True)
    stage_slug = _slugify_stage_id(stage_id)
    system_path = (
        prompt_dump.runtime_dir
        / f"stage_{stage_slug}.attempt_{attempt:02d}.{role_name}.system.md"
    )
    input_path = (
        prompt_dump.runtime_dir
        / f"stage_{stage_slug}.attempt_{attempt:02d}.{role_name}.input.json"
    )

    if reuse_if_present and system_path.exists() and input_path.exists():
        return

    system_text = _resolve_agent_instructions_text(agent, ctx)
    system_path.write_text(system_text)
    input_path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Round-0 staged bootstrap helpers
# ---------------------------------------------------------------------------


def _next_incomplete_stage_id(order: list[str], completed_stage_ids: list[str]) -> str | None:
    completed = set(completed_stage_ids)
    for stage_id in order:
        if stage_id not in completed:
            return stage_id
    return None


def _write_round0_artifact(
    solution_dir: Path,
    stage_id: str,
    attempt: int,
    suffix: str,
    payload: dict[str, Any],
) -> Path:
    stage_slug = _slugify_stage_id(stage_id)
    out_path = round0_artifacts_dir(solution_dir) / (
        f"{stage_slug}.attempt_{attempt:02d}.{suffix}.json"
    )
    _write_json(out_path, payload)
    return out_path


def _render_stage_agent_input(header: str, payload: dict[str, Any]) -> str:
    """Render a staged agent caller payload as a readable prompt string."""
    return f"{header}\n\nCaller payload:\n{json.dumps(payload, indent=2)}"


def _build_stage_coder_payload(
    *,
    graph: ImplementationGraph,
    stage_id: str,
    completed_stage_ids: list[str],
    latest_stage_results: dict[str, Round0StageResult],
    is_final_stage: bool,
) -> dict[str, Any]:
    stage = get_stage_spec(graph, stage_id)
    return {
        "current_stage": stage.model_dump(mode="json"),
        "relevant_async_pipelines": relevant_async_pipelines(graph, stage_id),
        "resource_ledger": graph.resource_ledger.model_dump(mode="json"),
        "kernel_contract": graph.kernel_contract.model_dump(mode="json"),
        "approved_frontier_summaries": _approved_frontier_summaries(
            graph=graph,
            completed_stage_ids=completed_stage_ids,
            latest_stage_results=latest_stage_results,
        ),
        "is_final_stage": is_final_stage,
    }


def _build_stage_coder_input(
    *,
    graph: ImplementationGraph,
    stage_id: str,
    completed_stage_ids: list[str],
    latest_stage_results: dict[str, Round0StageResult],
    is_final_stage: bool,
) -> str:
    payload = _build_stage_coder_payload(
        graph=graph,
        stage_id=stage_id,
        completed_stage_ids=completed_stage_ids,
        latest_stage_results=latest_stage_results,
        is_final_stage=is_final_stage,
    )
    return _render_stage_agent_input(
        "Implement the assigned staged round-0 kernel slice.",
        payload,
    )


def _build_stage_review_payload(
    *,
    graph: ImplementationGraph,
    stage_result: Round0StageResult,
    completed_stage_ids: list[str],
    latest_stage_results: dict[str, Round0StageResult],
) -> dict[str, Any]:
    stage = get_stage_spec(graph, stage_result.stage_id)
    return {
        "current_stage": stage.model_dump(mode="json"),
        "stage_result": stage_result.model_dump(mode="json"),
        "approved_frontier_summaries": _approved_frontier_summaries(
            graph=graph,
            completed_stage_ids=completed_stage_ids,
            latest_stage_results=latest_stage_results,
        ),
        "current_graph": graph.model_dump(mode="json"),
    }


def _build_stage_review_input(
    *,
    graph: ImplementationGraph,
    stage_result: Round0StageResult,
    completed_stage_ids: list[str],
    latest_stage_results: dict[str, Round0StageResult],
) -> str:
    payload = _build_stage_review_payload(
        graph=graph,
        stage_result=stage_result,
        completed_stage_ids=completed_stage_ids,
        latest_stage_results=latest_stage_results,
    )
    return _render_stage_agent_input(
        "Review the latest staged round-0 kernel implementation attempt and decide the next action.",
        payload,
    )


def _final_round0_result(
    *,
    solution_dir: Path,
    stage_result: Round0StageResult,
    impl_graph_file: str,
) -> CoderResult:
    generated = [str((solution_dir / "kernel_0.py").relative_to(PROJECT_ROOT))]
    if impl_graph_file:
        generated.append(impl_graph_file)
    return CoderResult(
        generated=generated,
        correctness_verified=stage_result.final_correctness_verified,
        status="success" if stage_result.final_correctness_verified else "validation_failed",
        message=stage_result.message,
        reflection=stage_result.reflection,
    )


async def _run_round0_staged(
    *,
    ctx: SharedContext,
    solution_dir: Path,
    state_path: Path,
    designer: object,
    stage_coder: object,
    stage_reviewer: object,
    designer_max_turns: int,
    designer_verbose: bool,
    coder_max_turns: int,
    coder_verbose: bool,
    skip_designer: bool,
    resume: bool,
    prompt_dump: PromptDumpConfig | None,
) -> CoderResult:
    round0_dir = round0_artifacts_dir(solution_dir)
    round0_dir.mkdir(parents=True, exist_ok=True)
    plan_path = round0_plan_path(solution_dir)
    if ctx.round0_impl_graph_file:
        impl_graph_file = Path(ctx.round0_impl_graph_file)
        if not impl_graph_file.is_absolute():
            impl_graph_file = PROJECT_ROOT / impl_graph_file
    else:
        impl_graph_file = round0_impl_graph_path(solution_dir)
    graph: ImplementationGraph

    if resume and ctx.round0_impl_graph_file:
        if not impl_graph_file.exists():
            print(
                "FATAL: --resume was set but "
                f"{impl_graph_file.relative_to(PROJECT_ROOT)} does not exist."
            )
            sys.exit(1)
        graph = load_impl_graph(impl_graph_file)
    elif skip_designer:
        if not plan_path.exists():
            print(
                "FATAL: --skip-designer was set but "
                f"{plan_path.relative_to(PROJECT_ROOT)} does not exist. "
                "Run kernel-designer first, or drop --skip-designer."
            )
            sys.exit(1)
        if not impl_graph_file.exists():
            print(
                "FATAL: --skip-designer was set but "
                f"{impl_graph_file.relative_to(PROJECT_ROOT)} does not exist. "
                "Run staged kernel-designer first, or drop --skip-designer."
            )
            sys.exit(1)
        graph = load_impl_graph(impl_graph_file)
        ctx.round0_impl_graph_file = str(impl_graph_file.relative_to(PROJECT_ROOT))
        print("=" * 60)
        print("ROUND 0a: SKIPPED — reusing existing staged plan + graph")
        print("=" * 60)
        print(f"  Design plan: {(solution_dir / 'kernel_0_plan.md').relative_to(PROJECT_ROOT)}")
        print(f"  Impl graph: {impl_graph_file.relative_to(PROJECT_ROOT)}")
    else:
        print("=" * 60)
        print("ROUND 0a: Design kernel_0_plan.md + implementation graph")
        print("=" * 60)
        ctx.current_agent_role = "designer"
        try:
            designer_result, designer_elapsed_s = await _run_agent(
                stage_label="kernel-designer-staged",
                starting_agent=designer,
                input=(
                    "Design the staged round-0 kernel architecture. Read the CuTeDSL references and "
                    "Blackwell kernel examples, then write solution/dsa_attention/kernel_0_plan.md "
                    "and return a compact staged implementation graph for kernel_0.py."
                ),
                context=ctx,
                max_turns=designer_max_turns,
                verbose=designer_verbose,
            )
            designer_out = require_structured_output(designer_result, DesignerResult)
        except MaxTurnsExceeded:
            print("FATAL: staged kernel-designer hit max turns without producing a plan.")
            sys.exit(1)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(
                    "FATAL: staged kernel-designer exhausted stage-local rate-limit retries "
                    f"while resuming saved context: {exc}"
                )
            else:
                print(f"FATAL: staged kernel-designer returned an invalid structured result: {exc}")
            sys.exit(1)

        if designer_out.status != "success":
            print(f"FATAL: staged kernel-designer failed: {designer_out.message}")
            sys.exit(1)

        validate_impl_graph(designer_out.impl_graph)
        _write_json(impl_graph_file, designer_out.impl_graph.model_dump(mode="json"))
        graph = designer_out.impl_graph
        ctx.round0_impl_graph_file = str(impl_graph_file.relative_to(PROJECT_ROOT))
        print(
            f"  {format_run_telemetry('kernel-designer', designer_elapsed_s, designer_result)}"
        )
        print(f"  Design plan: {designer_out.plan_file}")
        print(f"  Impl graph: {impl_graph_file.relative_to(PROJECT_ROOT)}")
        print(f"  {designer_out.message}")

    validate_impl_graph(graph)
    save_state(ctx, state_path)

    while True:
        order = iter_stage_order(graph)
        progress = _derive_round0_progress(ctx, graph)

        if progress.complete:
            final_stage_id = order[-1]
            final_stage_result = progress.latest_stage_results.get(final_stage_id)
            if final_stage_result is None or not final_stage_result.final_correctness_verified:
                print(
                    "FATAL: staged round-0 reached completion without a final correctness result."
                )
                sys.exit(1)
            shutil.copy2(solution_dir / "kernel_0.py", solution_dir / "kernel.py")
            save_state(ctx, state_path)
            return _final_round0_result(
                solution_dir=solution_dir,
                stage_result=final_stage_result,
                impl_graph_file=ctx.round0_impl_graph_file,
            )

        current_stage_id = progress.next_stage_id
        if current_stage_id is None:
            print("FATAL: staged round-0 has no remaining stage to execute.")
            sys.exit(1)

        is_final_stage = current_stage_id == order[-1]
        latest_stage_results = progress.latest_stage_results
        stage_out: Round0StageResult
        stage_attempt = progress.attempt_counts.get(current_stage_id, 0)

        print()
        print("=" * 60)
        if progress.pending_review:
            print(f"ROUND 0c[{stage_attempt}]: Review stage {current_stage_id}")
        else:
            print(f"ROUND 0b[{stage_attempt + 1}]: Stage {current_stage_id}")
        print("=" * 60)

        if progress.pending_review:
            if progress.pending_stage_result is None:
                print("FATAL: derived round-0 progress expected a pending stage result.")
                sys.exit(1)
            stage_out = progress.pending_stage_result
            if stage_out.stage_id != current_stage_id:
                print(
                    "FATAL: pending round-0 stage result does not match the next stage: "
                    f"{stage_out.stage_id!r} vs {current_stage_id!r}."
                )
                sys.exit(1)
            print("  Reusing pending stage result from history and resuming at the review step.")
            coder_payload = _build_stage_coder_payload(
                graph=graph,
                stage_id=current_stage_id,
                completed_stage_ids=progress.completed_stage_ids,
                latest_stage_results=latest_stage_results,
                is_final_stage=is_final_stage,
            )
            _dump_stage_runtime_prompt_artifacts(
                prompt_dump=prompt_dump,
                agent=stage_coder,
                ctx=ctx,
                stage_id=current_stage_id,
                attempt=stage_attempt,
                role_name="coder",
                payload=coder_payload,
                reuse_if_present=True,
            )
        else:
            stage_attempt += 1
            coder_payload = _build_stage_coder_payload(
                graph=graph,
                stage_id=current_stage_id,
                completed_stage_ids=progress.completed_stage_ids,
                latest_stage_results=latest_stage_results,
                is_final_stage=is_final_stage,
            )
            _dump_stage_runtime_prompt_artifacts(
                prompt_dump=prompt_dump,
                agent=stage_coder,
                ctx=ctx,
                stage_id=current_stage_id,
                attempt=stage_attempt,
                role_name="coder",
                payload=coder_payload,
            )
            ctx.current_agent_role = "coder"
            try:
                stage_result_raw, stage_elapsed_s = await _run_agent(
                    stage_label=f"kernel-stage-coder-{_slugify_stage_id(current_stage_id)}",
                    starting_agent=stage_coder,
                    input=_render_stage_agent_input(
                        "Implement the assigned staged round-0 kernel slice.",
                        coder_payload,
                    ),
                    context=ctx,
                    max_turns=coder_max_turns,
                    verbose=coder_verbose,
                )
                stage_out = require_structured_output(stage_result_raw, Round0StageResult)
            except MaxTurnsExceeded:
                print(f"FATAL: staged kernel-coder hit max turns on stage {current_stage_id}.")
                sys.exit(1)
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    print(
                        "FATAL: staged kernel-coder exhausted stage-local rate-limit retries "
                        f"while resuming saved context: {exc}"
                    )
                else:
                    print(f"FATAL: staged kernel-coder returned an invalid structured result: {exc}")
                sys.exit(1)

            if stage_out.stage_id != current_stage_id:
                print(
                    f"FATAL: staged kernel-coder returned stage_id={stage_out.stage_id!r}, "
                    f"expected {current_stage_id!r}."
                )
                sys.exit(1)

            stage_result_path = _write_round0_artifact(
                solution_dir,
                current_stage_id,
                stage_attempt,
                "result",
                stage_out.model_dump(mode="json"),
            )
            ctx.round0_stage_history.append(stage_out)
            print(
                f"  {format_run_telemetry('kernel-stage-coder', stage_elapsed_s, stage_result_raw)}"
            )
            print(f"  Stage result: {stage_result_path.relative_to(PROJECT_ROOT)}")
            print(f"  {stage_out.message}")
            save_state(ctx, state_path)

            progress = _derive_round0_progress(ctx, graph)
            latest_stage_results = progress.latest_stage_results

        ctx.current_agent_role = "designer"
        review_payload = _build_stage_review_payload(
            graph=graph,
            stage_result=stage_out,
            completed_stage_ids=progress.completed_stage_ids,
            latest_stage_results=latest_stage_results,
        )
        _dump_stage_runtime_prompt_artifacts(
            prompt_dump=prompt_dump,
            agent=stage_reviewer,
            ctx=ctx,
            stage_id=current_stage_id,
            attempt=stage_attempt,
            role_name="reviewer",
            payload=review_payload,
        )
        try:
            review_result_raw, review_elapsed_s = await _run_agent(
                stage_label=f"kernel-stage-reviewer-{_slugify_stage_id(current_stage_id)}",
                starting_agent=stage_reviewer,
                input=_render_stage_agent_input(
                    "Review the latest staged round-0 kernel implementation attempt and decide the next action.",
                    review_payload,
                ),
                context=ctx,
                max_turns=designer_max_turns,
                verbose=designer_verbose,
            )
            review_out = require_structured_output(review_result_raw, StageReviewResult)
        except MaxTurnsExceeded:
            print(f"FATAL: staged kernel-stage-reviewer hit max turns on stage {current_stage_id}.")
            sys.exit(1)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(
                    "FATAL: staged kernel-stage-reviewer exhausted stage-local rate-limit retries "
                    f"while resuming saved context: {exc}"
                )
            else:
                print(f"FATAL: staged kernel-stage-reviewer returned an invalid structured result: {exc}")
            sys.exit(1)

        review_path = _write_round0_artifact(
            solution_dir,
            current_stage_id,
            stage_attempt,
            "review",
            review_out.model_dump(mode="json"),
        )
        ctx.round0_review_history.append(review_out)
        print(
            f"  {format_run_telemetry('kernel-stage-reviewer', review_elapsed_s, review_result_raw)}"
        )
        print(f"  Stage review: {review_path.relative_to(PROJECT_ROOT)}")
        print(f"  Review action: {review_out.action}")
        print(f"  {review_out.message}")

        if review_out.action == "continue_next_stage":
            if review_out.replacement_impl_graph is not None or review_out.restart_from_stage_id is not None:
                print(
                    f"FATAL: reviewer returned extra revision fields while approving stage {current_stage_id}."
                )
                sys.exit(1)
            if stage_out.status != "success":
                print(
                    f"FATAL: reviewer approved stage {current_stage_id} even though coder "
                    f"returned status={stage_out.status}."
                )
                sys.exit(1)
            if not stage_out.frontier_verified:
                print(
                    f"FATAL: reviewer approved stage {current_stage_id} without passing the frontier validation."
                )
                sys.exit(1)
            if is_final_stage and not stage_out.final_correctness_verified:
                print(
                    "FATAL: staged round-0 reached the final stage without passing "
                    "the final correctness gate."
                )
                sys.exit(1)
        elif review_out.action == "retry_same_stage":
            if review_out.replacement_impl_graph is not None or review_out.restart_from_stage_id is not None:
                print(
                    f"FATAL: reviewer returned extra revision fields while retrying stage {current_stage_id}."
                )
                sys.exit(1)
        elif review_out.action == "revise_design_then_retry":
            if review_out.replacement_impl_graph is None:
                print(
                    f"FATAL: reviewer requested a design revision on stage {current_stage_id} "
                    "without providing replacement_impl_graph."
                )
                sys.exit(1)
            if review_out.restart_from_stage_id is None:
                print(
                    f"FATAL: reviewer requested a design revision on stage {current_stage_id} "
                    "without providing restart_from_stage_id."
                )
                sys.exit(1)
            validate_impl_graph(review_out.replacement_impl_graph)
            if review_out.restart_from_stage_id not in iter_stage_order(review_out.replacement_impl_graph):
                print(
                    f"FATAL: restart_from_stage_id={review_out.restart_from_stage_id!r} is not "
                    "present in the replacement implementation graph."
                )
                sys.exit(1)
            graph = review_out.replacement_impl_graph
            _write_json(impl_graph_file, graph.model_dump(mode="json"))
            ctx.round0_impl_graph_file = str(impl_graph_file.relative_to(PROJECT_ROOT))
        else:
            print(f"FATAL: unknown review action {review_out.action!r}")
            sys.exit(1)

        save_state(ctx, state_path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_loop(
    num_rounds: int = 5,
    model: str = "gpt-5.4",
    coder_model: str = "gpt-5.4",
    designer_model: str = "gpt-5.4",
    round0_mode: Round0Mode = "staged",
    quality_profile: QualityProfile = "public_codex",
    codex_worker_mode: CodexWorkerMode = "off",
    codex_worker_model: str = "gpt-5-codex",
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort = "high",
    resume: bool = False,
    skip_designer: bool = False,
    verbose: bool = False,
    coder_max_turns: int = 300,
    coder_reasoning_effort: str = "high",
    coder_verbosity: str = "low",
    designer_max_turns: int = 80,
    designer_reasoning_effort: str = "high",
    designer_verbosity: str = "low",
    planner_reasoning_effort: str = "high",
    optimizer_reasoning_effort: str = "high",
    coder_extra: str = "",
    designer_extra: str = "",
    planner_extra: str = "",
    optimizer_extra: str = "",
    prompt_dump: PromptDumpConfig | None = None,
) -> None:
    """Run the full kernel generation loop."""

    solution_dir = PROJECT_ROOT / "solution" / "dsa_attention"
    notes_dir = PROJECT_ROOT / "notes" / "dsa_attention"
    state_path = solution_dir / "loop_state.json"

    # Ensure directories exist
    solution_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    # Initialize context
    ctx = SharedContext(
        project_root=str(PROJECT_ROOT),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
        model_name=model,
        quality_profile=quality_profile,
        tool_limits=tool_limits_for_profile(quality_profile),
        codex_worker_mode=codex_worker_mode,
        round0_mode=round0_mode,
    )

    # Build agents
    designer = make_kernel_designer(
        context=ctx,
        model=designer_model,
        reasoning_effort=designer_reasoning_effort,
        verbosity=designer_verbosity,
        extra_instructions=designer_extra,
    )
    stage_coder = make_round0_stage_coder(
        context=ctx,
        model=coder_model,
        reasoning_effort=coder_reasoning_effort,
        verbosity=coder_verbosity,
        extra_instructions=coder_extra,
        codex_worker_model=codex_worker_model,
        codex_worker_reasoning_effort=codex_worker_reasoning_effort,
    )
    stage_reviewer = make_round0_stage_reviewer(
        context=ctx,
        model=designer_model,
        reasoning_effort=designer_reasoning_effort,
        verbosity=designer_verbosity,
        extra_instructions=designer_extra,
    )
    coder = make_kernel_coder(
        context=ctx,
        model=coder_model,
        reasoning_effort=coder_reasoning_effort,
        verbosity=coder_verbosity,
        extra_instructions=coder_extra,
        codex_worker_model=codex_worker_model,
        codex_worker_reasoning_effort=codex_worker_reasoning_effort,
    )
    planner = make_kernel_planner(
        context=ctx,
        model=model,
        reasoning_effort=planner_reasoning_effort,
        extra_instructions=planner_extra,
    )
    optimizer = make_kernel_optimizer(
        context=ctx,
        model=model,
        reasoning_effort=optimizer_reasoning_effort,
        extra_instructions=optimizer_extra,
        codex_worker_model=codex_worker_model,
        codex_worker_reasoning_effort=codex_worker_reasoning_effort,
    )

    # Resume handling
    start_round = 0
    if resume and state_path.exists():
        last_round = load_state(ctx, state_path)
        round0_mode = ctx.round0_mode
        if round0_mode == "staged" and last_round == 0:
            graph_path = None
            if ctx.round0_impl_graph_file:
                graph_path = Path(ctx.round0_impl_graph_file)
                if not graph_path.is_absolute():
                    graph_path = PROJECT_ROOT / graph_path

            if graph_path is not None and graph_path.exists():
                progress = _derive_round0_progress(ctx, load_impl_graph(graph_path))
                if progress.complete:
                    start_round = 1
                    print(f"Resuming from round {start_round}")
                else:
                    start_round = 0
                    current_stage = progress.next_stage_id or "(pending stage)"
                    if progress.pending_review:
                        print(f"Resuming staged round 0 at review for {current_stage}")
                    else:
                        print(f"Resuming staged round 0 from {current_stage}")
            else:
                start_round = 0
                print("Resuming staged round 0 from design")
        else:
            start_round = last_round + 1
            print(f"Resuming from round {start_round}")
    else:
        ctx.current_round = 0
        ctx.round0_mode = round0_mode
        ctx.round0_impl_graph_file = ""
        ctx.round0_stage_history = []
        ctx.round0_review_history = []

    if start_round == 0:
        if round0_mode == "staged":
            coder_out = await _run_round0_staged(
                ctx=ctx,
                solution_dir=solution_dir,
                state_path=state_path,
                designer=designer,
                stage_coder=stage_coder,
                stage_reviewer=stage_reviewer,
                designer_max_turns=designer_max_turns,
                designer_verbose=verbose,
                coder_max_turns=coder_max_turns,
                coder_verbose=verbose,
                skip_designer=skip_designer,
                resume=resume,
                prompt_dump=prompt_dump,
            )
        else:
            if skip_designer:
                plan_path = solution_dir / "kernel_0_plan.md"
                if not plan_path.exists():
                    print(
                        "FATAL: --skip-designer was set but "
                        f"{plan_path.relative_to(PROJECT_ROOT)} does not exist. "
                        "Run kernel-designer first, or drop --skip-designer."
                    )
                    sys.exit(1)
                print("=" * 60)
                print(f"ROUND 0a: SKIPPED — reusing existing {plan_path.name}")
                print("=" * 60)
                print(f"  Design plan: {plan_path.relative_to(PROJECT_ROOT)}")
            else:
                # ── Round 0a: Design kernel architecture ─────────────────────
                print("=" * 60)
                print("ROUND 0a: Design kernel_0_plan.md")
                print("=" * 60)

                ctx.current_agent_role = "designer"
                try:
                    designer_result, designer_elapsed_s = await _run_agent(
                        stage_label="kernel-designer",
                        starting_agent=designer,
                        input=(
                            "Design the kernel architecture. Read the CuTeDSL references and "
                            "Blackwell kernel examples, then write the design plan to "
                            "solution/dsa_attention/kernel_0_plan.md. The plan must describe "
                            "a CuTeDSL kernel that uses optimized B200 features (TMA, tcgen05, "
                            "warp specialization, async pipelining). Do not describe a PyTorch "
                            "fallback or a simple bootstrap path."
                        ),
                        context=ctx,
                        max_turns=designer_max_turns,
                        verbose=verbose,
                    )
                    designer_out = require_structured_output(designer_result, DesignerResult)
                except MaxTurnsExceeded:
                    print("FATAL: kernel-designer hit max turns without producing a plan.")
                    sys.exit(1)
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        print(
                            "FATAL: kernel-designer exhausted stage-local rate-limit retries "
                            f"while resuming saved context: {exc}"
                        )
                    else:
                        print(f"FATAL: kernel-designer returned an invalid structured result: {exc}")
                    sys.exit(1)

                if designer_out.status != "success":
                    print(f"FATAL: kernel-designer failed: {designer_out.message}")
                    sys.exit(1)

                print(f"  {format_run_telemetry('kernel-designer', designer_elapsed_s, designer_result)}")
                print(f"  Design plan: {designer_out.plan_file}")
                print(f"  {designer_out.message}")

            # ── Round 0b: Implement kernel from design plan ────────────────────
            print()
            print("=" * 60)
            print("ROUND 0b: Implement kernel_0.py from design plan")
            print("=" * 60)

            ctx.current_agent_role = "coder"
            try:
                result, elapsed_s = await _run_agent(
                    stage_label="kernel-coder",
                    starting_agent=coder,
                    input=(
                        "TASK\n"
                        "Implement solution/dsa_attention/kernel_0.py EXACTLY per the "
                        "design plan embedded in your system instructions. The plan is "
                        "the specification — every numbered section (work partition, "
                        "warp specialization, memory flow, async pipelines, SMEM plan, "
                        "TMEM plan, synchronization, API map) must have a visible "
                        "implementation in your kernel.\n\n"
                        "APPROACH (STRICT)\n"
                        "1. Re-read the plan top to bottom. Enumerate every pipeline, "
                        "warp role, tile, SMEM/TMEM chunk, and fence the plan names.\n"
                        "2. Implement each section verbatim. If the plan names N warps, "
                        "wire N warps. If it names pipelines P0..Pk, wire all of them. "
                        "If it allocates TMEM columns A-B/C-D, allocate all of them.\n"
                        "3. Iterate: run_synthetic_check for fast feedback, "
                        "run_correctness_check for the 23/23 gate. Trust the parsed "
                        "summaries.\n"
                        "4. On failure, fix the SMALLEST thing inside the failing "
                        "stage. Never delete, collapse, or substitute a plan-mandated "
                        "component to make a test pass — a missing component is not a "
                        "'fix', it is unfinished work.\n\n"
                        "ACCEPTANCE\n"
                        "- status=\"success\" requires BOTH: 23/23 correctness AND "
                        "every section of the plan reflected in kernel_0.py.\n"
                        "- If a plan section is genuinely impossible in CuTeDSL "
                        "(missing API, hardware limit), return "
                        "status=\"validation_failed\" and name the specific section + "
                        "API in `reflection`. Do not silently substitute a simpler "
                        "design."
                    ),
                    context=ctx,
                    max_turns=coder_max_turns,
                    verbose=verbose,
                )
                coder_out = require_structured_output(result, CoderResult)
            except MaxTurnsExceeded:
                print("FATAL: kernel-coder hit max turns without producing a result.")
                sys.exit(1)
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    print(
                        "FATAL: kernel-coder exhausted stage-local rate-limit retries "
                        f"while resuming saved context: {exc}"
                    )
                else:
                    print(f"FATAL: kernel-coder returned an invalid structured result: {exc}")
                sys.exit(1)

            if not coder_out.correctness_verified:
                print(f"FATAL: kernel-coder failed to produce a correct kernel: {coder_out.message}")
                sys.exit(1)

            print(f"  {format_run_telemetry('kernel-coder', elapsed_s, result)}")
            print(f"Round 0 complete: {coder_out.generated} [{coder_out.status}]")
            print(f"  {coder_out.message}")
            if coder_out.reflection:
                print(f"\n  Reflection:\n  {coder_out.reflection[:500]}")

            # Copy kernel_0.py → kernel.py as the initial working copy
            shutil.copy2(solution_dir / "kernel_0.py", solution_dir / "kernel.py")

        start_round = 1
        save_state(ctx, state_path)

    # ── Rounds 1..N: Planner → Optimizer cycle ─────────────────────────
    for i in range(start_round, num_rounds + 1):
        ctx.current_round = i

        print()
        print("=" * 60)
        print(f"ROUND {i}/{num_rounds}: Optimize")
        print("=" * 60)

        # Copy last correct kernel to working copy
        last_correct = find_last_correct_kernel(ctx, solution_dir)
        shutil.copy2(last_correct, solution_dir / "kernel.py")
        print(f"Working copy: kernel.py ← {last_correct.name}")

        # Build history string
        history_str = format_history(ctx.history, best_latency_ms=ctx.best_latency_ms)
        # ── Planner ────────────────────────────────────────────────────
        print(f"\n--- Planner (round {i}) ---")
        planner_input = (
            f"Round {i}. The current kernel.py was copied from {last_correct.name}.\n\n"
            f"Current best latency: {ctx.best_latency_ms:.3f}ms (round {ctx.best_round}).\n\n"
            f"History of prior rounds:\n{history_str}\n\n"
            f"Benchmark the current kernel, diagnose the bottleneck, and write "
            f"strategy_{i}.md to notes/dsa_attention/."
        )

        ctx.current_agent_role = "planner"
        try:
            planner_result, planner_elapsed_s = await _run_agent(
                stage_label=f"kernel-planner-round-{i}",
                starting_agent=planner,
                input=planner_input,
                context=ctx,
                max_turns=40,
                verbose=verbose,
            )
            print(f"  {format_run_telemetry('kernel-planner', planner_elapsed_s, planner_result)}")
            pr = require_structured_output(planner_result, PlannerResult)
        except MaxTurnsExceeded:
            print(f"WARNING: Planner hit max turns in round {i}. Skipping this round.")
            ctx.history.append(RoundRecord(
                round_num=i, kernel_file=f"kernel_{i}.py",
                latency_ms=None, correctness=False,
                strategy_summary="planner_timeout", bottleneck="unknown",
            ))
            save_state(ctx, state_path)
            continue
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(
                    "FATAL: kernel-planner exhausted stage-local rate-limit retries "
                    f"in round {i} while resuming saved context: {exc}"
                )
            else:
                print(
                    f"FATAL: kernel-planner returned an invalid structured result in round {i}: "
                    f"{exc}"
                )
            sys.exit(1)

        print(f"  Latency: {pr.latency_ms:.3f}ms")
        print(f"  Bottleneck: {pr.bottleneck}")
        print(f"  Strategy: {pr.strategy_summary}")

        # Track best
        if pr.latency_ms < ctx.best_latency_ms:
            ctx.best_latency_ms = pr.latency_ms
            ctx.best_round = i - 1  # the kernel we just benchmarked
            shutil.copy2(solution_dir / "kernel.py", solution_dir / "best_kernel.py")
            print(f"  NEW BEST: {pr.latency_ms:.3f}ms (from round {i - 1})")

        # ── Optimizer ──────────────────────────────────────────────────
        print(f"\n--- Optimizer (round {i}) ---")
        optimizer_input = (
            f"Round {i}. Implement the strategy in {pr.strategy_file}.\n"
            f"Read the current kernel at solution/dsa_attention/kernel.py.\n"
            f"Write the optimized kernel to solution/dsa_attention/kernel_{i}.py.\n"
            f"Validate with --correctness-only. You have up to 5 attempts.\n\n"
            f"Current best latency: {ctx.best_latency_ms:.3f}ms (round {ctx.best_round}).\n\n"
            f"History of prior rounds:\n{history_str}\n\n"
            f"SCOPE: Implement EXACTLY and ONLY what the strategy specifies. "
            f"Do not refactor unrelated code or change the kernel interface."
        )

        ctx.current_agent_role = "optimizer"
        try:
            optimizer_result, optimizer_elapsed_s = await _run_agent(
                stage_label=f"kernel-optimizer-round-{i}",
                starting_agent=optimizer,
                input=optimizer_input,
                context=ctx,
                max_turns=60,
                verbose=verbose,
            )
            print(
                f"  {format_run_telemetry('kernel-optimizer', optimizer_elapsed_s, optimizer_result)}"
            )
            opt = require_structured_output(optimizer_result, OptimizerResult)
        except MaxTurnsExceeded:
            print(f"WARNING: Optimizer hit max turns in round {i}.")
            opt = OptimizerResult(
                kernel_file=f"kernel_{i}.py",
                correctness_verified=False,
                status="timeout",
                message="Hit max_turns limit",
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(
                    "FATAL: kernel-optimizer exhausted stage-local rate-limit retries "
                    f"in round {i} while resuming saved context: {exc}"
                )
            else:
                print(
                    f"FATAL: kernel-optimizer returned an invalid structured result in round {i}: "
                    f"{exc}"
                )
            sys.exit(1)

        status_icon = "OK" if opt.correctness_verified else "FAIL"
        print(f"  [{status_icon}] {opt.kernel_file}: {opt.message}")
        if opt.reflection:
            print(f"  Reflection: {opt.reflection[:300]}")

        # Record round
        ctx.history.append(RoundRecord(
            round_num=i,
            kernel_file=f"kernel_{i}.py",
            latency_ms=pr.latency_ms,  # latency of kernel_{i-1}
            correctness=opt.correctness_verified,
            strategy_summary=pr.strategy_summary,
            bottleneck=pr.bottleneck,
            failure_reason=opt.message if not opt.correctness_verified else "",
            strategy_file=pr.strategy_file,
            ncu_metrics=pr.ncu_metrics.model_dump() if pr.ncu_metrics else None,
        ))

        save_state(ctx, state_path)

    # ── Final summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if (solution_dir / "best_kernel.py").exists() and ctx.best_round >= 0:
        print(f"Best kernel so far: round {ctx.best_round} ({ctx.best_latency_ms:.3f}ms)")
    else:
        print("No best kernel has been recorded yet.")

    save_state(ctx, state_path)
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent kernel generation loop for DSA sparse attention"
    )
    parser.add_argument(
        "--num-rounds", type=int, default=5,
        help="Number of optimization rounds (default: 5)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5.4",
        help="LLM model for planner and optimizer agents (default: gpt-5.4)",
    )
    parser.add_argument(
        "--coder-model", type=str, default="gpt-5.4",
        help="LLM model for the round-0 kernel-coder agent (default: gpt-5.4)",
    )
    parser.add_argument(
        "--designer-model", type=str, default="gpt-5.4",
        help="LLM model for the round-0 kernel-designer agent (default: gpt-5.4)",
    )
    parser.add_argument(
        "--round0-mode",
        choices=ROUND0_MODE_CHOICES,
        default="staged",
        help=(
            "Round-0 bootstrap mode: staged (default) or legacy monolithic "
            "designer->coder flow."
        ),
    )
    parser.add_argument(
        "--quality-profile",
        choices=QUALITY_PROFILE_CHOICES,
        default="public_codex",
        help="Tool-limit and run-config profile to use (default: public_codex)",
    )
    parser.add_argument(
        "--codex-worker-mode",
        choices=CODEX_WORKER_MODE_CHOICES,
        default="off",
        help="Expose write-capable Codex workers for selected agents (default: off)",
    )
    parser.add_argument(
        "--codex-worker-model",
        type=str,
        default="gpt-5-codex",
        help="Model used by the write-capable Codex worker tools (default: gpt-5-codex)",
    )
    parser.add_argument(
        "--codex-worker-reasoning-effort",
        choices=CODEX_WORKER_REASONING_EFFORT_CHOICES,
        default="high",
        help="Reasoning effort for the write-capable Codex worker tools (default: high)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint (loop_state.json)",
    )
    parser.add_argument(
        "--skip-designer", action="store_true",
        help=(
            "Skip round 0a (kernel-designer). In staged mode this reuses both "
            "solution/dsa_attention/kernel_0_plan.md and the persisted round0 "
            "implementation graph; in legacy mode it reuses only the plan. "
            "Ignored when --resume successfully loads saved state."
        ),
    )
    parser.add_argument(
        "--dump-prompts", action="store_true",
        help=(
            "Resolve and write the static agent prompts to prompts/, then continue "
            "running and capture fully materialized staged round-0 coder/reviewer "
            "system prompts and caller payloads under prompts/round0_runtime/."
        ),
    )
    parser.add_argument(
        "--dump-prompts-only", action="store_true",
        help=(
            "Resolve and write the static agent prompts to prompts/ and exit without "
            "running any agent."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show structured agent progress",
    )
    parser.add_argument(
        "--coder-max-turns", type=int, default=300,
        help="Max LLM turns for the round-0 kernel-coder agent (default: 300)",
    )
    parser.add_argument(
        "--coder-reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default="high",
        help="Reasoning effort for the round-0 kernel-coder agent (default: high)",
    )
    parser.add_argument(
        "--coder-verbosity",
        choices=VERBOSITY_CHOICES,
        default="low",
        help="Verbosity for the round-0 kernel-coder agent (default: low)",
    )
    parser.add_argument(
        "--designer-max-turns", type=int, default=80,
        help="Max LLM turns for the round-0 kernel-designer agent (default: 80)",
    )
    parser.add_argument(
        "--designer-reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default="high",
        help="Reasoning effort for the round-0 kernel-designer agent (default: high)",
    )
    parser.add_argument(
        "--designer-verbosity",
        choices=VERBOSITY_CHOICES,
        default="low",
        help="Verbosity for the round-0 kernel-designer agent (default: low)",
    )
    parser.add_argument(
        "--planner-reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default="high",
        help="Reasoning effort for the kernel-planner agent (default: high)",
    )
    parser.add_argument(
        "--optimizer-reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default="high",
        help="Reasoning effort for the kernel-optimizer agent (default: high)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    prompt_dump = None
    if args.dump_prompts or args.dump_prompts_only:
        dump_all_prompts(
            output_dir=PROJECT_ROOT / "prompts",
            model=args.model,
            coder_model=args.coder_model,
            designer_model=args.designer_model,
            quality_profile=args.quality_profile,
            codex_worker_mode=args.codex_worker_mode,
            codex_worker_model=args.codex_worker_model,
            codex_worker_reasoning_effort=args.codex_worker_reasoning_effort,
            coder_reasoning_effort=args.coder_reasoning_effort,
            coder_verbosity=args.coder_verbosity,
            designer_reasoning_effort=args.designer_reasoning_effort,
            designer_verbosity=args.designer_verbosity,
            planner_reasoning_effort=args.planner_reasoning_effort,
            optimizer_reasoning_effort=args.optimizer_reasoning_effort,
        )
        if args.dump_prompts:
            prompt_dump = _make_prompt_dump_config(PROJECT_ROOT / "prompts")
    if args.dump_prompts_only:
        return

    asyncio.run(run_loop(
        num_rounds=args.num_rounds,
        model=args.model,
        coder_model=args.coder_model,
        designer_model=args.designer_model,
        round0_mode=args.round0_mode,
        quality_profile=args.quality_profile,
        codex_worker_mode=args.codex_worker_mode,
        codex_worker_model=args.codex_worker_model,
        codex_worker_reasoning_effort=args.codex_worker_reasoning_effort,
        resume=args.resume,
        skip_designer=args.skip_designer,
        verbose=args.verbose,
        coder_max_turns=args.coder_max_turns,
        coder_reasoning_effort=args.coder_reasoning_effort,
        coder_verbosity=args.coder_verbosity,
        designer_max_turns=args.designer_max_turns,
        designer_reasoning_effort=args.designer_reasoning_effort,
        designer_verbosity=args.designer_verbosity,
        planner_reasoning_effort=args.planner_reasoning_effort,
        optimizer_reasoning_effort=args.optimizer_reasoning_effort,
        prompt_dump=prompt_dump,
    ))


if __name__ == "__main__":
    main()
