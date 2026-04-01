"""Shared context and structured output types for the kernel generation agents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoundRecord:
    """Record of a single optimization round."""

    round_num: int
    kernel_file: str  # e.g. "kernel_1.py"
    latency_ms: float | None  # None if not benchmarked
    correctness: bool
    strategy_summary: str  # 1-line summary
    bottleneck: str  # what was identified


@dataclass
class SharedContext:
    """Mutable state threaded through all agent runs."""

    project_root: str  # /home/mark123/projects/FlashMLA
    solution_dir: str  # solution/dsa_attention (relative to project_root)
    notes_dir: str  # notes/dsa_attention (relative to project_root)
    current_round: int = 0
    best_latency_ms: float = float("inf")
    best_round: int = -1
    history: list[RoundRecord] = field(default_factory=list)
    model_name: str = "gpt-4.5"


# ---------------------------------------------------------------------------
# Structured output types — used as Agent.output_type so we get programmatic
# access to results instead of parsing free text.
# ---------------------------------------------------------------------------


@dataclass
class CoderResult:
    """Returned by kernel-coder after round 0 bootstrap."""

    kernel_file: str  # path to kernel_0.py
    correctness_verified: bool
    status: str  # "success" | "compile_error" | "validation_failed"
    message: str  # human-readable summary


@dataclass
class PlannerResult:
    """Returned by kernel-planner after profiling + strategy writing."""

    latency_ms: float  # measured avg latency of the current kernel
    bottleneck: str  # identified bottleneck
    strategy_summary: str  # 1-line summary of proposed optimization
    strategy_file: str  # path to strategy_{i}.md
    is_new_best: bool  # whether this kernel beat the previous best


@dataclass
class OptimizerResult:
    """Returned by kernel-optimizer after implementing + validating."""

    kernel_file: str  # path to kernel_{i}.py
    correctness_verified: bool  # passed --correctness-only
    status: str  # "success" | "compile_error" | "validation_failed" | "timeout"
    message: str  # human-readable summary
