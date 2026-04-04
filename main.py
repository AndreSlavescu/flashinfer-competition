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

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY (and any other vars) from .env

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from agents import Runner
from agents.exceptions import MaxTurnsExceeded

from kernel_agents.context import (
    CoderResult,
    OptimizerResult,
    PlannerResult,
    RoundRecord,
    SharedContext,
)
from kernel_agents.kernel_coder import (
    REASONING_EFFORT_CHOICES,
    VERBOSITY_CHOICES,
    make_kernel_coder,
)
from kernel_agents.kernel_optimizer import make_kernel_optimizer
from kernel_agents.kernel_planner import make_kernel_planner
from kernel_agents.stream_logging import consume_streamed_run

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()


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
        "history": [asdict(r) for r in ctx.history],
    }
    path.write_text(json.dumps(state, indent=2))
    logger.info("State saved to %s", path)


def load_state(ctx: SharedContext, path: Path) -> int:
    """Load loop state from JSON. Returns the last completed round number."""
    state = json.loads(path.read_text())
    ctx.best_latency_ms = state["best_latency_ms"]
    ctx.best_round = state["best_round"]
    ctx.history = [RoundRecord(**r) for r in state["history"]]
    last_round = state["current_round"]
    logger.info("Resumed from round %d (best=%.3fms @ round %d)",
                last_round, ctx.best_latency_ms, ctx.best_round)
    return last_round


# ---------------------------------------------------------------------------
# History formatting
# ---------------------------------------------------------------------------

def format_history(history: list[RoundRecord]) -> str:
    """Build a human-readable history summary for agent prompts."""
    if not history:
        return "(No prior rounds — this is the first optimization.)"
    lines = []
    for r in history:
        status = "CORRECT" if r.correctness else "FAILED"
        lat = f"{r.latency_ms:.3f}ms" if r.latency_ms is not None else "N/A"
        lines.append(
            f"  Round {r.round_num}: [{status}] latency={lat} | "
            f"bottleneck='{r.bottleneck}' | strategy='{r.strategy_summary}'"
        )
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


def parse_coder_result(raw_output: object) -> CoderResult:
    """Parse the kernel-coder's plain-text JSON response into a CoderResult."""
    if isinstance(raw_output, CoderResult):
        return raw_output

    text = raw_output if isinstance(raw_output, str) else str(raw_output)
    candidates: list[str] = []

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if "\n" in block:
                _, remainder = block.split("\n", 1)
                candidates.append(remainder.strip())
            else:
                candidates.append(block.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1].strip())

    last_error: Exception | None = None
    parsed: dict[str, object] | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            maybe = json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(maybe, dict):
            parsed = maybe
            break
        last_error = ValueError("kernel-coder output was valid JSON but not an object")

    if parsed is None:
        detail = f" ({last_error})" if last_error else ""
        raise ValueError(f"Failed to parse kernel-coder JSON output{detail}")

    generated = parsed.get("generated")
    if isinstance(generated, str):
        generated = [generated]
    if not isinstance(generated, list) or not all(isinstance(x, str) for x in generated):
        raise ValueError("kernel-coder output field 'generated' must be a list of strings")

    correctness_verified = parsed.get("correctness_verified")
    if not isinstance(correctness_verified, bool):
        raise ValueError("kernel-coder output field 'correctness_verified' must be a bool")

    status = parsed.get("status")
    if not isinstance(status, str):
        raise ValueError("kernel-coder output field 'status' must be a string")

    message = parsed.get("message")
    if not isinstance(message, str):
        raise ValueError("kernel-coder output field 'message' must be a string")

    reflection = parsed.get("reflection", "")
    if not isinstance(reflection, str):
        reflection = str(reflection)

    return CoderResult(
        generated=list(generated),
        correctness_verified=correctness_verified,
        status=status,
        message=message,
        reflection=reflection,
    )


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
    if not verbose:
        result = await Runner.run(
            starting_agent=starting_agent,
            input=input,
            context=context,
            max_turns=max_turns,
        )
        return result, time.perf_counter() - started_at

    result = Runner.run_streamed(
        starting_agent=starting_agent,
        input=input,
        context=context,
        max_turns=max_turns,
    )
    return await consume_streamed_run(result), time.perf_counter() - started_at


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_loop(
    num_rounds: int = 5,
    model: str = "gpt-5.4",
    coder_model: str = "gpt-5.4",
    resume: bool = False,
    verbose: bool = False,
    coder_max_turns: int = 300,
    coder_reasoning_effort: str = "xhigh",
    coder_verbosity: str = "low",
    coder_extra: str = "",
    planner_extra: str = "",
    optimizer_extra: str = "",
) -> None:
    """Run the full kernel generation loop."""

    solution_dir = PROJECT_ROOT / "solution" / "dsa_attention"
    notes_dir = PROJECT_ROOT / "notes" / "dsa_attention"
    state_path = solution_dir / "loop_state.json"

    # Ensure directories exist
    notes_dir.mkdir(parents=True, exist_ok=True)

    # Initialize context
    ctx = SharedContext(
        project_root=str(PROJECT_ROOT),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
        model_name=model,
    )

    # Build agents
    coder = make_kernel_coder(
        model=coder_model,
        reasoning_effort=coder_reasoning_effort,
        verbosity=coder_verbosity,
        extra_instructions=coder_extra,
    )
    planner = make_kernel_planner(model=model, extra_instructions=planner_extra)
    optimizer = make_kernel_optimizer(model=model, extra_instructions=optimizer_extra)

    # Resume handling
    start_round = 0
    if resume and state_path.exists():
        start_round = load_state(ctx, state_path) + 1
        print(f"Resuming from round {start_round}")
    else:
        # ── Round 0: Generate initial kernel ────────────────────────────
        print("=" * 60)
        print("ROUND 0: Generate kernel_0.py")
        print("=" * 60)

        ctx.current_round = 0
        raw_coder_output: object | None = None

        try:
            result, elapsed_s = await _run_agent(
                starting_agent=coder,
                input=(
                    "Generate kernel_0.py. Follow your instructions — first write "
                    "solution/dsa_attention/kernel_0_plan.md with the final CuTeDSL design, "
                    "then implement, debug, and validate until ALL 23 workloads pass "
                    "correctness. Do not submit a PyTorch fallback; if the CuTeDSL compute "
                    "path is not working, return validation_failed."
                ),
                context=ctx,
                max_turns=coder_max_turns,
                verbose=verbose,
            )
            raw_coder_output = result.final_output
            coder_out = parse_coder_result(raw_coder_output)
        except MaxTurnsExceeded:
            print("FATAL: kernel-coder hit max turns without producing a result.")
            sys.exit(1)
        except Exception as exc:
            print(f"FATAL: kernel-coder returned an unparsable result: {exc}")
            if raw_coder_output is not None:
                print(f"Raw output:\n{raw_coder_output}")
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
        history_str = format_history(ctx.history)

        # ── Planner ────────────────────────────────────────────────────
        print(f"\n--- Planner (round {i}) ---")
        planner_input = (
            f"Round {i}. The current kernel.py was copied from {last_correct.name}.\n\n"
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
            pr: PlannerResult = planner_result.final_output
        except MaxTurnsExceeded:
            print(f"WARNING: Planner hit max turns in round {i}. Skipping this round.")
            ctx.history.append(RoundRecord(
                round_num=i, kernel_file=f"kernel_{i}.py",
                latency_ms=None, correctness=False,
                strategy_summary="planner_timeout", bottleneck="unknown",
            ))
            save_state(ctx, state_path)
            continue

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
            f"Validate with --correctness-only. You have up to 5 attempts."
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
            opt: OptimizerResult = optimizer_result.final_output
        except MaxTurnsExceeded:
            print(f"WARNING: Optimizer hit max turns in round {i}.")
            opt = OptimizerResult(
                kernel_file=f"kernel_{i}.py",
                correctness_verified=False,
                status="timeout",
                message="Hit max_turns limit",
            )

        status_icon = "OK" if opt.correctness_verified else "FAIL"
        print(f"  [{status_icon}] {opt.kernel_file}: {opt.message}")

        # Record round
        ctx.history.append(RoundRecord(
            round_num=i,
            kernel_file=f"kernel_{i}.py",
            latency_ms=pr.latency_ms,  # latency of kernel_{i-1}
            correctness=opt.correctness_verified,
            strategy_summary=pr.strategy_summary,
            bottleneck=pr.bottleneck,
        ))

        save_state(ctx, state_path)

    # ── Epilogue: Bench the last kernel for best_kernel tracking ───────
    last_kernel = f"kernel_{num_rounds}.py"
    if (solution_dir / last_kernel).exists():
        print()
        print("=" * 60)
        print(f"EPILOGUE: Benchmark {last_kernel}")
        print("=" * 60)

        shutil.copy2(solution_dir / last_kernel, solution_dir / "kernel.py")

        try:
            epilogue_result, epilogue_elapsed_s = await _run_agent(
                starting_agent=planner,
                input=(
                    f"Epilogue round. Benchmark kernel.py (copied from {last_kernel}). "
                    f"Report the latency. You do NOT need to write a strategy file — "
                    f"just benchmark and report the PlannerResult."
                ),
                context=ctx,
                max_turns=20,
                verbose=verbose,
            )
            print(
                f"  {format_run_telemetry('epilogue-planner', epilogue_elapsed_s, epilogue_result)}"
            )
            ep: PlannerResult = epilogue_result.final_output
            print(f"  Last kernel latency: {ep.latency_ms:.3f}ms")

            if ep.latency_ms < ctx.best_latency_ms:
                ctx.best_latency_ms = ep.latency_ms
                ctx.best_round = num_rounds
                shutil.copy2(solution_dir / "kernel.py", solution_dir / "best_kernel.py")
                print(f"  NEW BEST: {ep.latency_ms:.3f}ms (from round {num_rounds})")
        except MaxTurnsExceeded:
            print("WARNING: Epilogue planner hit max turns.")

    # ── Final benchmark ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("FINAL BENCHMARK")
    print("=" * 60)
    print(f"Best kernel: round {ctx.best_round} ({ctx.best_latency_ms:.3f}ms)")

    if (solution_dir / "best_kernel.py").exists():
        # bench.py auto-selects best_kernel.py for full perf benchmark
        result = subprocess.run(
            [
                sys.executable, "-m", "modal", "run",
                "scripts/bench.py",
                "--track", "dsa_attention",
                "--solution-dir", "solution/dsa_attention",
            ],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"Final benchmark exited with code {result.returncode}")
    else:
        print("No best_kernel.py found — skipping final benchmark.")

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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    asyncio.run(run_loop(
        num_rounds=args.num_rounds,
        model=args.model,
        coder_model=args.coder_model,
        resume=args.resume,
        verbose=args.verbose,
        coder_max_turns=args.coder_max_turns,
        coder_reasoning_effort=args.coder_reasoning_effort,
        coder_verbosity=args.coder_verbosity,
    ))


if __name__ == "__main__":
    main()
