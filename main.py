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
from kernel_agents.kernel_stage_fixer import make_round0_stage_fixer
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
DEFAULT_ROUND0_ENTRY_STAGE = "S0"
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
        "round0_stage_history": _dump_round0_stage_history(ctx.round0_stage_history),
        "round0_review_history": _dump_round0_review_history(ctx.round0_review_history),
    }
    path.write_text(json.dumps(state, indent=2))
    logger.info("State saved to %s", path)

def load_state(ctx: SharedContext, path: Path) -> int:
    """Load loop state from JSON. Returns the last completed round number."""
    state = json.loads(path.read_text())
    ctx.best_latency_ms = state["best_latency_ms"]
    ctx.best_round = state["best_round"]
    ctx.history = [RoundRecord(**record) for record in state["history"]]
    ctx.codex_thread_id_coder_engineer = state["codex_thread_id_coder_engineer"]
    ctx.codex_thread_id_optimizer_engineer = state["codex_thread_id_optimizer_engineer"]
    ctx.round0_mode = state["round0_mode"]
    ctx.round0_stage_history = _load_round0_stage_history(state["round0_stage_history"])
    ctx.round0_review_history = _load_round0_review_history(state["round0_review_history"])
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

def round0_plan_path(solution_dir: Path) -> Path:
    """Return the canonical path for the staged round-0 human plan."""
    return solution_dir / "kernel_0_plan.md"


def _slugify_stage_id(stage_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_id).strip("_") or "stage"


def _project_relative_path(path: Path, *, project_root: Path = PROJECT_ROOT) -> str:
    """Return *path* relative to the project root when possible."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def round0_stage_kernel_snapshot_path(solution_dir: Path, stage_id: str, attempt: int) -> Path:
    """Return the per-attempt staged kernel snapshot path."""
    return (
        round0_artifacts_dir(solution_dir)
        / f"{_slugify_stage_id(stage_id)}.attempt_{attempt:02d}.kernel_0.py"
    )


def _stage_attempt_number_at_history_index(
    history: list[Round0StageResult],
    index: int,
) -> int:
    """Return the 1-based attempt number for history[index]."""
    target_stage_id = history[index].stage_id
    return sum(1 for item in history[: index + 1] if item.stage_id == target_stage_id)


def _ensure_round0_kernel_snapshot(
    solution_dir: Path,
    *,
    stage_id: str,
    attempt: int,
) -> Path:
    """Persist the current kernel_0.py as the canonical snapshot for this attempt."""
    source = solution_dir / "kernel_0.py"
    destination = round0_stage_kernel_snapshot_path(solution_dir, stage_id, attempt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination


def _previous_round0_kernel_snapshot_path(
    ctx: SharedContext,
    solution_dir: Path,
) -> Path | None:
    """Return the snapshot path for the immediately previous staged kernel attempt."""
    if len(ctx.round0_stage_history) < 2:
        return None
    previous_index = len(ctx.round0_stage_history) - 2
    previous_stage = ctx.round0_stage_history[previous_index]
    previous_attempt = _stage_attempt_number_at_history_index(
        ctx.round0_stage_history,
        previous_index,
    )
    return round0_stage_kernel_snapshot_path(
        solution_dir,
        previous_stage.stage_id,
        previous_attempt,
    )


@dataclass
class Round0DerivedProgress:
    """Derived staged round-0 progress reconstructed from append-only histories."""

    attempt_counts: dict[str, int]
    latest_stage_result: Round0StageResult | None
    pending_stage_result: Round0StageResult | None
    pending_review: bool
    next_stage_id: str | None
    latest_review: StageReviewResult | None
    complete: bool


def validate_stage_review_result(
    *,
    stage_result: Round0StageResult,
    review: StageReviewResult,
) -> None:
    """Validate reviewer routing against the latest stage outcome."""
    if review.stage_id != stage_result.stage_id:
        raise ValueError(
            "round0 stage/review history is out of sync: "
            f"stage result is {stage_result.stage_id!r}, review is {review.stage_id!r}."
        )

    if review.action == "retry_same_stage":
        if review.next_stage != stage_result.stage_id:
            raise ValueError(
                f"retry_same_stage for {stage_result.stage_id!r} requires next_stage to match."
            )
        return

    if review.action == "revise_design_then_retry":
        if review.next_stage is not None:
            raise ValueError("revise_design_then_retry requires next_stage to be null.")
        return

    if review.action != "continue_next_stage":
        raise ValueError(f"Unknown round0 review action {review.action!r}")

    if stage_result.status != "success":
        raise ValueError(
            f"Reviewer approved stage {stage_result.stage_id!r} even though "
            f"status={stage_result.status!r}."
        )
    if not stage_result.stage_output_verified:
        raise ValueError(
            f"Reviewer approved stage {stage_result.stage_id!r} without passing stage validation."
        )

    if review.next_stage is None:
        if not stage_result.final_correctness_verified:
            raise ValueError(
                f"Stage {stage_result.stage_id!r} cannot terminate round-0 without correctness."
            )


def _derive_round0_progress(ctx: SharedContext) -> Round0DerivedProgress:
    if len(ctx.round0_review_history) > len(ctx.round0_stage_history):
        raise ValueError("round0 review history cannot be longer than stage history.")
    if len(ctx.round0_stage_history) > len(ctx.round0_review_history) + 1:
        raise ValueError("round0 stage history can have at most one unreviewed stage attempt.")

    attempt_counts: dict[str, int] = {}
    latest_stage_result: Round0StageResult | None = None
    latest_review: StageReviewResult | None = None
    complete = False

    for stage_result in ctx.round0_stage_history:
        latest_stage_result = stage_result
        attempt_counts[stage_result.stage_id] = attempt_counts.get(stage_result.stage_id, 0) + 1

    reviewed_attempt_count = len(ctx.round0_review_history)
    for index in range(reviewed_attempt_count):
        stage_result = ctx.round0_stage_history[index]
        review = ctx.round0_review_history[index]
        validate_stage_review_result(stage_result=stage_result, review=review)
        latest_review = review

        if review.action == "continue_next_stage":
            if review.next_stage is None:
                complete = True
        elif review.action == "retry_same_stage":
            continue
        elif review.action == "revise_design_then_retry":
            # Histories are cleared after a revision, so this branch is only reached
            # defensively (e.g., a corrupted state file). Treat as a full reset.
            latest_stage_result = None
            attempt_counts = {}
            complete = False

    pending_stage_result = None
    pending_review = False
    if len(ctx.round0_stage_history) > reviewed_attempt_count:
        pending_stage_result = ctx.round0_stage_history[-1]
        pending_review = True

    next_stage_id = None
    if not complete:
        if pending_stage_result is not None:
            next_stage_id = pending_stage_result.stage_id
        elif latest_review is None:
            next_stage_id = DEFAULT_ROUND0_ENTRY_STAGE
        elif latest_review.action in {"continue_next_stage", "retry_same_stage"}:
            next_stage_id = latest_review.next_stage

    return Round0DerivedProgress(
        attempt_counts=attempt_counts,
        latest_stage_result=latest_stage_result,
        pending_stage_result=pending_stage_result,
        pending_review=pending_review,
        next_stage_id=next_stage_id,
        latest_review=latest_review,
        complete=complete,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _display_path(path: Path) -> str:
    """Return a repo-relative path when possible, otherwise an absolute path."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


async def _run_staged_designer(
    *,
    starting_agent: object,
    initial_input: str,
    context: SharedContext,
    max_turns: int,
    verbose: bool,
    solution_dir: Path,
    stage_label: str = "kernel-designer-staged",
    max_rate_limit_retries: int = STAGE_RATE_LIMIT_MAX_RETRIES,
) -> tuple[object, DesignerResult, float]:
    """Run the staged designer with stage-local rate-limit recovery."""
    started_at = time.perf_counter()
    session = SQLiteSession(f"{stage_label}-{time.time_ns()}")
    current_input: str | list[dict[str, Any]] = initial_input
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
            continue

        designer_out = require_structured_output(result, DesignerResult)
        if designer_out.status != "success":
            return result, designer_out, time.perf_counter() - started_at

        plan_path = round0_plan_path(solution_dir)
        if not plan_path.exists():
            raise ValueError(
                "staged kernel-designer reported success but did not write "
                f"{_display_path(plan_path)}."
            )

        return result, designer_out, time.perf_counter() - started_at


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
        static_dir=output_dir / "static",
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
    reviewer_verbosity: str,
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
        "kernel_stage_fixer": make_round0_stage_fixer(
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
            verbosity=reviewer_verbosity,  # type: ignore[arg-type]
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
    reviewer_verbosity: str,
    planner_reasoning_effort: str,
    optimizer_reasoning_effort: str,
) -> None:
    """Resolve and write each agent's composed system prompt under `output_dir`.

    Builds a SharedContext identical to what `run_loop` would construct, wires
    up the round-0 and optimization agents with the same knobs, and writes each resolved prompt
    (dynamic callables invoked) to `output_dir/static/<agent_name>.md`.
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
        reviewer_verbosity=reviewer_verbosity,
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


def _round0_stage_executor_role_name(stage_attempt: int) -> str:
    """Return the staged executor role name for a 1-based attempt count."""
    return "coder" if stage_attempt <= 1 else "fixer"


def _render_json_section(title: str, payload: Any) -> str:
    """Render a markdown section containing JSON payload data."""
    return f"## {title}\n```json\n{json.dumps(payload, indent=2)}\n```"


def _render_markdown_section(title: str, markdown_text: str) -> str:
    """Render a markdown section that preserves embedded markdown content."""
    body = markdown_text.strip()
    if not body:
        return f"## {title}\n"
    return f"## {title}\n\n{body}"


def _trim_stage_history(
    history: list[Round0StageResult],
) -> list[dict[str, Any]]:
    """Project stage results down to the fields the reviewer needs."""
    return [
        {
            "stage_id": record.stage_id,
            "stage_output_validation_report": record.stage_output_validation_report,
            "correctness_check_report": record.correctness_check_report,
            "message": record.message,
        }
        for record in history
    ]


def _trim_review_history(
    history: list[StageReviewResult],
) -> list[dict[str, Any]]:
    """Project prior reviews down to the fields the reviewer needs."""
    return [
        {
            "action": record.action,
            "next_stage": record.next_stage,
            "message": record.message,
        }
        for record in history
    ]


def _trim_last_review(review: StageReviewResult | None) -> dict[str, Any] | None:
    """Project a single review into the payload form the fixer/designer consume."""
    if review is None:
        return None
    return {
        "action": review.action,
        "next_stage": review.next_stage,
        "message": review.message,
    }


def _build_designer_revision_input(review: StageReviewResult) -> str:
    """Build the designer's initial input when re-invoked after a revise_design_then_retry review."""
    trimmed = _trim_last_review(review)
    feedback_json = json.dumps(trimmed, indent=2)
    return (
        "The staged round-0 reviewer flagged the current design as unworkable and "
        "requested a revised plan. Produce a fresh `DesignerResult` that addresses "
        "the reviewer's critique below. Rewrite solution/dsa_attention/kernel_0_plan.md "
        "in place and return a corrected design result.\n\n"
        "## Trimmed Last Reviewer Feedback\n"
        f"```json\n{feedback_json}\n```"
    )


def _apply_stage_coder_prompt_sections(
    ctx: SharedContext,
    payload: dict[str, Any],
) -> None:
    """Populate transient dynamic prompt sections for the stage coder."""
    ctx.round0_stage_coder_prompt_sections = {
        "assigned_stage": _render_markdown_section(
            "Assigned Stage",
            f"`{payload['assigned_stage']}`",
        ),
        "full_plan": _render_markdown_section(
            "Full kernel_0_plan.md",
            payload["full_plan"],
        ),
    }


def _apply_stage_fixer_prompt_sections(
    ctx: SharedContext,
    payload: dict[str, Any],
) -> None:
    """Populate transient dynamic prompt sections for the stage fixer."""
    ctx.round0_stage_fixer_prompt_sections = {
        "stage_to_fix": _render_markdown_section(
            "Stage To Fix",
            f"`{payload['stage_to_fix']}`",
        ),
        "full_plan": _render_markdown_section(
            "Full kernel_0_plan.md",
            payload["full_plan"],
        ),
        "last_review": _render_json_section(
            "Last Reviewer Feedback",
            payload["last_review"],
        ),
    }


def _apply_stage_reviewer_prompt_sections(
    ctx: SharedContext,
    payload: dict[str, Any],
) -> None:
    """Populate transient dynamic prompt sections for the stage reviewer."""
    ctx.round0_stage_reviewer_prompt_sections = {
        "stage_under_review": _render_markdown_section(
            "Stage Under Review",
            f"`{payload['stage_under_review']}`",
        ),
        "full_plan": _render_markdown_section(
            "Full kernel_0_plan.md",
            payload["full_plan"],
        ),
        "kernel_snapshots": _render_json_section(
            "Kernel Snapshots",
            payload["kernel_snapshots"],
        ),
        "stage_results_history": _render_json_section(
            "Trimmed Round0StageResult history",
            payload["stage_results_history"],
        ),
        "review_history": _render_json_section(
            "Trimmed StageReviewResult history",
            payload["review_history"],
        ),
        "recovery_context": _render_markdown_section(
            "Recovery Task",
            payload["recovery_context"],
        ),
    }


def _build_stage_coder_payload(
    *,
    plan_text: str,
    stage_id: str,
) -> dict[str, Any]:
    return {
        "assigned_stage": stage_id,
        "full_plan": plan_text,
    }


def _build_stage_fixer_payload(
    *,
    plan_text: str,
    stage_id: str,
    last_review: StageReviewResult,
) -> dict[str, Any]:
    return {
        "stage_to_fix": stage_id,
        "full_plan": plan_text,
        "last_review": _trim_last_review(last_review),
    }


def _require_same_stage_retry_review(
    *,
    stage_id: str,
    review: StageReviewResult | None,
) -> StageReviewResult:
    """Return the retry review required for a fixer attempt."""
    if review is None:
        raise ValueError(
            f"fixer attempt for stage {stage_id!r} is missing reviewer feedback."
        )
    if review.stage_id != stage_id:
        raise ValueError(
            "fixer attempt expected reviewer feedback for the same stage: "
            f"{review.stage_id!r} vs {stage_id!r}."
        )
    if review.action != "retry_same_stage":
        raise ValueError(
            "fixer attempt requires reviewer action 'retry_same_stage', got "
            f"{review.action!r} for stage {stage_id!r}."
        )
    return review


def _build_stage_review_payload(
    *,
    ctx: SharedContext,
    solution_dir: Path,
    plan_text: str,
    stage_result: Round0StageResult,
    stage_attempt: int,
) -> dict[str, Any]:
    project_root = Path(ctx.project_root)

    def _payload_path(path: Path) -> str:
        return _project_relative_path(path, project_root=project_root)

    current_snapshot = _ensure_round0_kernel_snapshot(
        solution_dir,
        stage_id=stage_result.stage_id,
        attempt=stage_attempt,
    )
    previous_snapshot = _previous_round0_kernel_snapshot_path(ctx, solution_dir)
    if previous_snapshot is not None and not previous_snapshot.exists():
        previous_snapshot = None
    return {
        "stage_under_review": stage_result.stage_id,
        "full_plan": plan_text,
        "kernel_snapshots": {
            "current_kernel_path": _payload_path(solution_dir / "kernel_0.py"),
            "current_attempt_snapshot": _payload_path(current_snapshot),
            "previous_attempt_snapshot": (
                _payload_path(previous_snapshot)
                if previous_snapshot is not None
                else None
            ),
        },
        "stage_results_history": _trim_stage_history(ctx.round0_stage_history),
        "review_history": _trim_review_history(ctx.round0_review_history),
        "recovery_context": "",
    }


def _build_stage_recovery_payload(
    *,
    ctx: SharedContext,
    plan_text: str,
) -> dict[str, Any]:
    project_root = Path(ctx.project_root)
    solution_dir = Path(ctx.solution_dir)
    if not solution_dir.is_absolute():
        solution_dir = project_root / solution_dir

    return {
        "stage_under_review": "Recovery",
        "full_plan": plan_text,
        "kernel_snapshots": {
            "current_kernel_path": _project_relative_path(
                solution_dir / "kernel_0.py",
                project_root=project_root,
            ),
        },
        "stage_results_history": _trim_stage_history(ctx.round0_stage_history),
        "review_history": _trim_review_history(ctx.round0_review_history),
        "recovery_context": (
            "There is no fresh stage result to review. Inspect the current `kernel_0.py`, "
            "the full plan, and the trimmed histories, then choose the stage the next "
            "coder pass should execute. Return `action=\"continue_next_stage\"` and set "
            "`next_stage` to that concrete stage ID. Do not return null in recovery mode."
        ),
    }


async def _recover_round0_next_stage_with_reviewer(
    *,
    ctx: SharedContext,
    plan_text: str,
    stage_reviewer: object,
    reviewer_max_turns: int,
    reviewer_verbose: bool,
    prompt_dump: PromptDumpConfig | None,
) -> StageReviewResult:
    """Ask the reviewer to infer the next coder stage from the current kernel and histories."""
    recovery_payload = _build_stage_recovery_payload(ctx=ctx, plan_text=plan_text)
    _apply_stage_reviewer_prompt_sections(ctx, recovery_payload)
    _dump_stage_runtime_prompt_artifacts(
        prompt_dump=prompt_dump,
        agent=stage_reviewer,
        ctx=ctx,
        stage_id="recovery",
        attempt=0,
        role_name="reviewer",
        payload=recovery_payload,
    )
    ctx.current_agent_role = "reviewer"
    recovery_result_raw, _ = await _run_agent(
        stage_label="kernel-stage-reviewer-recovery",
        starting_agent=stage_reviewer,
        input=(
            "Recover staged round-0 progress from the current kernel implementation and "
            "choose the next stage the coder should execute."
        ),
        context=ctx,
        max_turns=reviewer_max_turns,
        verbose=reviewer_verbose,
    )
    recovery_out = require_structured_output(recovery_result_raw, StageReviewResult)
    if recovery_out.action != "continue_next_stage" or recovery_out.next_stage is None:
        raise ValueError(
            "stage recovery reviewer must return action='continue_next_stage' with a "
            "concrete next_stage."
        )
    return recovery_out


def _final_round0_result(
    *,
    solution_dir: Path,
    stage_result: Round0StageResult,
) -> CoderResult:
    generated = [str((solution_dir / "kernel_0.py").relative_to(PROJECT_ROOT))]
    return CoderResult(
        generated=generated,
        correctness_verified=stage_result.final_correctness_verified,
        status="success" if stage_result.final_correctness_verified else "validation_failed",
        message=stage_result.message,
        reflection="",
    )


async def _run_round0_staged(
    *,
    ctx: SharedContext,
    solution_dir: Path,
    state_path: Path,
    designer: object,
    stage_coder: object,
    stage_fixer: object,
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

    plan_text: str

    if resume:
        if not plan_path.exists():
            print(
                "FATAL: --resume was set but "
                f"{plan_path.relative_to(PROJECT_ROOT)} does not exist."
            )
            sys.exit(1)
        plan_text = plan_path.read_text(encoding="utf-8")
    elif skip_designer:
        if not plan_path.exists():
            print(
                "FATAL: --skip-designer was set but "
                f"{plan_path.relative_to(PROJECT_ROOT)} does not exist. "
                "Run kernel-designer first, or drop --skip-designer."
            )
            sys.exit(1)
        plan_text = plan_path.read_text(encoding="utf-8")
        print("=" * 60)
        print("ROUND 0a: SKIPPED — reusing existing staged plan")
        print("=" * 60)
        print(f"  Design plan: {plan_path.relative_to(PROJECT_ROOT)}")
    else:
        print("=" * 60)
        print("ROUND 0a: Design kernel_0_plan.md")
        print("=" * 60)
        ctx.current_agent_role = "designer"
        try:
            designer_result, designer_out, designer_elapsed_s = (
                await _run_staged_designer(
                    stage_label="kernel-designer-staged",
                    starting_agent=designer,
                    initial_input=(
                        "Design the staged round-0 kernel architecture. Read the CuTeDSL "
                        "references and Blackwell kernel examples, then write "
                        "solution/dsa_attention/kernel_0_plan.md."
                    ),
                    context=ctx,
                    max_turns=designer_max_turns,
                    verbose=designer_verbose,
                    solution_dir=solution_dir,
                )
            )
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
            print("FATAL: staged kernel-designer failed.")
            if designer_out.plan_file:
                print(f"  Design plan: {designer_out.plan_file}")
            sys.exit(1)

        plan_text = plan_path.read_text(encoding="utf-8")
        print(
            f"  {format_run_telemetry('kernel-designer', designer_elapsed_s, designer_result)}"
        )
        print(f"  Design plan: {designer_out.plan_file}")

    save_state(ctx, state_path)

    while True:
        progress = _derive_round0_progress(ctx)

        if progress.complete:
            final_stage_result = progress.latest_stage_result
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
            )

        current_stage_id = progress.next_stage_id
        recovered_stage_id = None
        if current_stage_id is None:
            if resume and not progress.pending_review:
                try:
                    recovery_review = await _recover_round0_next_stage_with_reviewer(
                        ctx=ctx,
                        plan_text=plan_text,
                        stage_reviewer=stage_reviewer,
                        reviewer_max_turns=designer_max_turns,
                        reviewer_verbose=designer_verbose,
                        prompt_dump=prompt_dump,
                    )
                except MaxTurnsExceeded:
                    print("FATAL: staged kernel-stage-reviewer recovery hit max turns.")
                    sys.exit(1)
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        print(
                            "FATAL: staged kernel-stage-reviewer recovery exhausted "
                            f"stage-local rate-limit retries while resuming saved context: {exc}"
                        )
                    else:
                        print(
                            "FATAL: staged kernel-stage-reviewer recovery returned an invalid "
                            f"structured result: {exc}"
                        )
                    sys.exit(1)
                current_stage_id = recovery_review.next_stage
                recovered_stage_id = current_stage_id
            else:
                print("FATAL: staged round-0 has no remaining stage to execute.")
                sys.exit(1)

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
            executor_role_name = _round0_stage_executor_role_name(stage_attempt)
            if executor_role_name == "coder":
                executor_agent = stage_coder
                executor_payload = _build_stage_coder_payload(
                    plan_text=plan_text,
                    stage_id=current_stage_id,
                )
                _apply_stage_coder_prompt_sections(ctx, executor_payload)
            else:
                try:
                    retry_review = _require_same_stage_retry_review(
                        stage_id=current_stage_id,
                        review=progress.latest_review,
                    )
                except ValueError as exc:
                    print(f"FATAL: {exc}")
                    sys.exit(1)
                executor_agent = stage_fixer
                executor_payload = _build_stage_fixer_payload(
                    plan_text=plan_text,
                    stage_id=current_stage_id,
                    last_review=retry_review,
                )
                _apply_stage_fixer_prompt_sections(ctx, executor_payload)
            _dump_stage_runtime_prompt_artifacts(
                prompt_dump=prompt_dump,
                agent=executor_agent,
                ctx=ctx,
                stage_id=current_stage_id,
                attempt=stage_attempt,
                role_name=executor_role_name,
                payload=executor_payload,
                reuse_if_present=True,
            )
        else:
            stage_attempt += 1
            executor_role_name = (
                "coder"
                if recovered_stage_id == current_stage_id
                else _round0_stage_executor_role_name(stage_attempt)
            )
            executor_display_name = f"kernel-stage-{executor_role_name}"
            if executor_role_name == "coder":
                executor_agent = stage_coder
                executor_payload = _build_stage_coder_payload(
                    plan_text=plan_text,
                    stage_id=current_stage_id,
                )
                _apply_stage_coder_prompt_sections(ctx, executor_payload)
            else:
                try:
                    retry_review = _require_same_stage_retry_review(
                        stage_id=current_stage_id,
                        review=progress.latest_review,
                    )
                except ValueError as exc:
                    print(f"FATAL: {exc}")
                    sys.exit(1)
                executor_agent = stage_fixer
                executor_payload = _build_stage_fixer_payload(
                    plan_text=plan_text,
                    stage_id=current_stage_id,
                    last_review=retry_review,
                )
                _apply_stage_fixer_prompt_sections(ctx, executor_payload)
            _dump_stage_runtime_prompt_artifacts(
                prompt_dump=prompt_dump,
                agent=executor_agent,
                ctx=ctx,
                stage_id=current_stage_id,
                attempt=stage_attempt,
                role_name=executor_role_name,
                payload=executor_payload,
            )
            ctx.current_agent_role = "coder"
            try:
                stage_result_raw, stage_elapsed_s = await _run_agent(
                    stage_label=f"{executor_display_name}-{_slugify_stage_id(current_stage_id)}",
                    starting_agent=executor_agent,
                    input="Implement the assigned staged round-0 kernel slice.",
                    context=ctx,
                    max_turns=coder_max_turns,
                    verbose=coder_verbose,
                )
                stage_out = require_structured_output(stage_result_raw, Round0StageResult)
            except MaxTurnsExceeded:
                print(
                    f"FATAL: staged {executor_display_name} hit max turns on stage "
                    f"{current_stage_id}."
                )
                sys.exit(1)
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    print(
                        f"FATAL: staged {executor_display_name} exhausted stage-local "
                        "rate-limit retries "
                        f"while resuming saved context: {exc}"
                    )
                else:
                    print(
                        f"FATAL: staged {executor_display_name} returned an invalid "
                        f"structured result: {exc}"
                    )
                sys.exit(1)

            if stage_out.stage_id != current_stage_id:
                print(
                    f"FATAL: staged {executor_display_name} returned stage_id={stage_out.stage_id!r}, "
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
                f"  {format_run_telemetry(executor_display_name, stage_elapsed_s, stage_result_raw)}"
            )
            print(f"  Stage result: {stage_result_path.relative_to(PROJECT_ROOT)}")
            print(f"  {stage_out.message}")
            save_state(ctx, state_path)

        ctx.current_agent_role = "reviewer"
        review_payload = _build_stage_review_payload(
            ctx=ctx,
            solution_dir=solution_dir,
            plan_text=plan_text,
            stage_result=stage_out,
            stage_attempt=stage_attempt,
        )
        _apply_stage_reviewer_prompt_sections(ctx, review_payload)
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
                input=(
                    "Review the latest staged round-0 kernel implementation attempt and decide "
                    "the next action."
                ),
                context=ctx,
                max_turns=designer_max_turns,
                verbose=designer_verbose,
                )
            review_out = require_structured_output(review_result_raw, StageReviewResult)
            validate_stage_review_result(
                stage_result=stage_out,
                review=review_out,
            )
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
        print(f"  Review next stage: {review_out.next_stage}")
        print(f"  {review_out.message}")

        if review_out.action == "continue_next_stage":
            pass
        elif review_out.action == "retry_same_stage":
            pass
        elif review_out.action == "revise_design_then_retry":
            ctx.current_agent_role = "designer"
            revision_input = _build_designer_revision_input(review_out)
            try:
                designer_result, designer_out, designer_elapsed_s = (
                    await _run_staged_designer(
                        stage_label="kernel-designer-staged-revise",
                        starting_agent=designer,
                        initial_input=revision_input,
                        context=ctx,
                        max_turns=designer_max_turns,
                        verbose=designer_verbose,
                        solution_dir=solution_dir,
                    )
                )
            except MaxTurnsExceeded:
                print("FATAL: staged kernel-designer revision hit max turns.")
                sys.exit(1)
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    print(
                        "FATAL: staged kernel-designer revision exhausted stage-local rate-limit "
                        f"retries while resuming saved context: {exc}"
                    )
                else:
                    print(
                        "FATAL: staged kernel-designer revision returned an invalid structured "
                        f"result: {exc}"
                    )
                sys.exit(1)

            if designer_out.status != "success":
                print("FATAL: staged kernel-designer revision failed.")
                if designer_out.plan_file:
                    print(f"  Design plan: {designer_out.plan_file}")
                sys.exit(1)

            plan_text = plan_path.read_text(encoding="utf-8")
            ctx.round0_stage_history.clear()
            ctx.round0_review_history.clear()
            print(
                f"  {format_run_telemetry('kernel-designer', designer_elapsed_s, designer_result)}"
            )
            print(f"  Revised design plan: {designer_out.plan_file}")
            print("  Round-0 stage/review histories cleared for fresh restart.")
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
    reviewer_verbosity: str = "low",
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
    stage_fixer = make_round0_stage_fixer(
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
        verbosity=reviewer_verbosity,
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
        try:
            last_round = load_state(ctx, state_path)
        except ValueError as exc:
            print(f"FATAL: {exc}")
            sys.exit(1)
        round0_mode = ctx.round0_mode
        if round0_mode == "staged" and last_round == 0:
            plan_path = round0_plan_path(solution_dir)
            if plan_path.exists():
                try:
                    progress = _derive_round0_progress(ctx)
                except ValueError as exc:
                    print(f"FATAL: resume found invalid staged round-0 state: {exc}")
                    sys.exit(1)
                if progress.complete:
                    start_round = 1
                    print(f"Resuming from round {start_round}")
                else:
                    start_round = 0
                    current_stage = progress.next_stage_id or "(reviewer recovery)"
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
                stage_fixer=stage_fixer,
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
                    print("FATAL: kernel-designer failed.")
                    if designer_out.plan_file:
                        print(f"  Design plan: {designer_out.plan_file}")
                    sys.exit(1)

                print(f"  {format_run_telemetry('kernel-designer', designer_elapsed_s, designer_result)}")
                print(f"  Design plan: {designer_out.plan_file}")

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
                        "run_correctness_check for the 23/23 gate. If the Modal log is "
                        "trimmed, inspect last_shell_dump.txt with read_file or "
                        "grep_search.\n"
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
            "solution/dsa_attention/kernel_0_plan.md and its staged implementation outline; "
            "in legacy mode it reuses only the plan. "
            "Ignored when --resume successfully loads saved state."
        ),
    )
    parser.add_argument(
        "--dump-prompts", action="store_true",
        help=(
            "Resolve and write the static agent prompts to prompts/static/, then continue "
            "running and capture fully materialized staged round-0 coder/fixer/reviewer "
            "system prompts and caller payloads under prompts/round0_runtime/."
        ),
    )
    parser.add_argument(
        "--dump-prompts-only", action="store_true",
        help=(
            "Resolve and write the static agent prompts to prompts/static/ and exit without "
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
        "--reviewer-verbosity",
        choices=VERBOSITY_CHOICES,
        default="low",
        help="Verbosity for the round-0 kernel-stage-reviewer agent (default: low)",
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
            reviewer_verbosity=args.reviewer_verbosity,
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
        reviewer_verbosity=args.reviewer_verbosity,
        planner_reasoning_effort=args.planner_reasoning_effort,
        optimizer_reasoning_effort=args.optimizer_reasoning_effort,
        prompt_dump=prompt_dump,
    ))


if __name__ == "__main__":
    main()
