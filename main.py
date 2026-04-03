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
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import json as _json

from agents import Runner
from agents.exceptions import MaxTurnsExceeded
from agents.lifecycle import RunHooksBase
from agents import Agent as _Agent
from typing import Any as _Any


class _VerboseHooks(RunHooksBase[_Any, _Agent]):
    """Print each tool call + a result preview as they happen."""

    async def on_llm_end(self, context, agent, response) -> None:
        """Called after each LLM turn — response.output has tool calls with args."""
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                fn_name = getattr(item, "name", "?")
                args_str = getattr(item, "arguments", "") or ""
                try:
                    args = _json.loads(args_str)
                    key = next(
                        (k for k in ("file_path", "command", "query", "url", "pattern", "path")
                         if k in args), None
                    )
                    hint = f" {key}={str(args[key])[:80]}" if key else f" {args_str[:60]}"
                except Exception:
                    hint = f" {args_str[:60]}"
                print(f"  [{agent.name}] {fn_name}{hint}")
            elif item_type == "shell_call":
                cmds = getattr(getattr(item, "action", None), "commands", []) or []
                for cmd in cmds[:3]:
                    print(f"  [{agent.name}] shell: {cmd[:120]}")

    async def on_tool_end(self, context, agent, tool, result: str) -> None:
        """Called after a tool returns — show a short preview of the result."""
        preview = (result or "").replace("\n", " ")[:100]
        print(f"  [{agent.name}]   ✓ {preview}")


VERBOSE_HOOKS = _VerboseHooks()

from kernel_agents.context import (
    CoderResult,
    OptimizerResult,
    PlannerResult,
    RoundRecord,
    SharedContext,
)
from kernel_agents.kernel_coder import make_kernel_coder
from kernel_agents.kernel_optimizer import make_kernel_optimizer
from kernel_agents.kernel_planner import make_kernel_planner

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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_loop(
    num_rounds: int = 5,
    model: str = "gpt-5.4",
    resume: bool = False,
    verbose: bool = False,
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
    coder = make_kernel_coder(model=model, extra_instructions=coder_extra)
    planner = make_kernel_planner(model=model, extra_instructions=planner_extra)
    optimizer = make_kernel_optimizer(model=model, extra_instructions=optimizer_extra)

    # Hooks for live tool-call visibility (activated by --verbose)
    hooks = VERBOSE_HOOKS if verbose else None  # type: ignore[assignment]

    # Resume handling
    start_round = 0
    if resume and state_path.exists():
        start_round = load_state(ctx, state_path) + 1
        print(f"Resuming from round {start_round}")
    else:
        # ── Round 0: Bootstrap ──────────────────────────────────────────
        print("=" * 60)
        print("ROUND 0: Bootstrap kernel_0.py")
        print("=" * 60)

        ctx.current_round = 0

        try:
            result = await Runner.run(
                starting_agent=coder,
                input=(
                    "Bootstrap kernel_0.py. Follow your instructions — design, implement, "
                    "debug, and validate until ALL 23 workloads pass correctness."
                ),
                context=ctx,
                max_turns=300,
                hooks=hooks,
            )
            coder_out: CoderResult = result.final_output
        except MaxTurnsExceeded:
            print("FATAL: kernel-coder hit max turns without producing a result.")
            sys.exit(1)

        if not coder_out.correctness_verified:
            print(f"FATAL: kernel-coder failed to produce a correct kernel: {coder_out.message}")
            sys.exit(1)

        print(f"Round 0 complete: {coder_out.kernel_file} [{coder_out.status}]")
        print(f"  {coder_out.message}")

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
            planner_result = await Runner.run(
                starting_agent=planner,
                input=planner_input,
                context=ctx,
                max_turns=40,
                hooks=hooks,
            )
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
            optimizer_result = await Runner.run(
                starting_agent=optimizer,
                input=optimizer_input,
                context=ctx,
                max_turns=60,
                hooks=hooks,
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
            epilogue_result = await Runner.run(
                starting_agent=planner,
                input=(
                    f"Epilogue round. Benchmark kernel.py (copied from {last_kernel}). "
                    f"Report the latency. You do NOT need to write a strategy file — "
                    f"just benchmark and report the PlannerResult."
                ),
                context=ctx,
                max_turns=20,
                hooks=hooks,
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
        help="LLM model for all agents (default: gpt-5.4)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint (loop_state.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    asyncio.run(run_loop(
        num_rounds=args.num_rounds,
        model=args.model,
        resume=args.resume,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
