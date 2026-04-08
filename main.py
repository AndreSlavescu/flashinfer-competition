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
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from agents import ModelRetrySettings, ModelSettings, RunConfig, Runner, retry_policies
from agents.exceptions import MaxTurnsExceeded
from agents.run_config import CallModelData, ModelInputData

from kernel_agents.context import (
    CODEX_WORKER_MODE_CHOICES,
    CODEX_WORKER_REASONING_EFFORT_CHOICES,
    QUALITY_PROFILE_CHOICES,
    REASONING_EFFORT_CHOICES,
    VERBOSITY_CHOICES,
    CoderResult,
    CodexWorkerMode,
    CodexWorkerReasoningEffort,
    DesignerResult,
    OptimizerResult,
    PlannerResult,
    QualityProfile,
    RoundRecord,
    SharedContext,
    tool_limits_for_profile,
)
from kernel_agents.kernel_coder import make_kernel_coder
from kernel_agents.kernel_designer import make_kernel_designer
from kernel_agents.kernel_optimizer import make_kernel_optimizer
from kernel_agents.kernel_planner import make_kernel_planner
from kernel_agents.stream_logging import consume_streamed_run

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()
PUBLIC_CODEX_COMPACTION_THRESHOLD = 200_000


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

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


async def _run_agent(
    *,
    starting_agent: object,
    input: str,
    context: SharedContext,
    max_turns: int,
    verbose: bool,
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
        )
        return result, time.perf_counter() - started_at

    result = Runner.run_streamed(
        starting_agent=starting_agent,
        input=input,
        context=context,
        max_turns=max_turns,
        run_config=run_config,
    )
    return await consume_streamed_run(result), time.perf_counter() - started_at


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_loop(
    num_rounds: int = 5,
    model: str = "gpt-5.4",
    coder_model: str = "gpt-5.4",
    designer_model: str = "gpt-5.4",
    quality_profile: QualityProfile = "public_codex",
    codex_worker_mode: CodexWorkerMode = "off",
    codex_worker_model: str = "gpt-5-codex",
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort = "high",
    resume: bool = False,
    verbose: bool = False,
    coder_max_turns: int = 300,
    coder_reasoning_effort: str = "xhigh",
    coder_verbosity: str = "low",
    designer_max_turns: int = 80,
    designer_reasoning_effort: str = "xhigh",
    designer_verbosity: str = "low",
    planner_reasoning_effort: str = "high",
    optimizer_reasoning_effort: str = "high",
    coder_extra: str = "",
    designer_extra: str = "",
    planner_extra: str = "",
    optimizer_extra: str = "",
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
    )

    # Build agents
    designer = make_kernel_designer(
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
        start_round = load_state(ctx, state_path) + 1
        print(f"Resuming from round {start_round}")
    else:
        # ── Round 0a: Design kernel architecture ─────────────────────────
        print("=" * 60)
        print("ROUND 0a: Design kernel_0_plan.md")
        print("=" * 60)

        ctx.current_round = 0
        try:
            designer_result, designer_elapsed_s = await _run_agent(
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

        try:
            result, elapsed_s = await _run_agent(
                starting_agent=coder,
                input=(
                    "Implement kernel_0.py based on the design plan at "
                    f"{designer_out.plan_file}. Read the plan first, "
                    "then implement, debug, and validate until ALL 23 workloads pass "
                    "correctness. Do not submit a PyTorch fallback; if the CuTeDSL compute "
                    "path is not working, return validation_failed."
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

        try:
            planner_result, planner_elapsed_s = await _run_agent(
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
            print(f"FATAL: kernel-planner returned an invalid structured result in round {i}: {exc}")
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

        try:
            optimizer_result, optimizer_elapsed_s = await _run_agent(
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
            print(f"FATAL: kernel-optimizer returned an invalid structured result in round {i}: {exc}")
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
        default="xhigh",
        help="Reasoning effort for the round-0 kernel-coder agent (default: xhigh)",
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
        default="xhigh",
        help="Reasoning effort for the round-0 kernel-designer agent (default: xhigh)",
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

    asyncio.run(run_loop(
        num_rounds=args.num_rounds,
        model=args.model,
        coder_model=args.coder_model,
        designer_model=args.designer_model,
        quality_profile=args.quality_profile,
        codex_worker_mode=args.codex_worker_mode,
        codex_worker_model=args.codex_worker_model,
        codex_worker_reasoning_effort=args.codex_worker_reasoning_effort,
        resume=args.resume,
        verbose=args.verbose,
        coder_max_turns=args.coder_max_turns,
        coder_reasoning_effort=args.coder_reasoning_effort,
        coder_verbosity=args.coder_verbosity,
        designer_max_turns=args.designer_max_turns,
        designer_reasoning_effort=args.designer_reasoning_effort,
        designer_verbosity=args.designer_verbosity,
        planner_reasoning_effort=args.planner_reasoning_effort,
        optimizer_reasoning_effort=args.optimizer_reasoning_effort,
    ))


if __name__ == "__main__":
    main()
