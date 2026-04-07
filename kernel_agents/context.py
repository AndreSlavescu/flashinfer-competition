"""Shared context and structured output types for the kernel generation agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared type aliases used by multiple agents (designer, coder)
# ---------------------------------------------------------------------------

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]
Verbosity = Literal["low", "medium", "high"]
QualityProfile = Literal["legacy", "public_codex"]
CodexWorkerMode = Literal["off", "coder_optimizer"]
CodexWorkerReasoningEffort = Literal["low", "medium", "high", "xhigh"]

REASONING_EFFORT_CHOICES: tuple[ReasoningEffort, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
)
VERBOSITY_CHOICES: tuple[Verbosity, ...] = ("low", "medium", "high")
QUALITY_PROFILE_CHOICES: tuple[QualityProfile, ...] = ("legacy", "public_codex")
CODEX_WORKER_MODE_CHOICES: tuple[CodexWorkerMode, ...] = ("off", "coder_optimizer")
CODEX_WORKER_REASONING_EFFORT_CHOICES: tuple[CodexWorkerReasoningEffort, ...] = (
    "low",
    "medium",
    "high",
    "xhigh",
)


@dataclass(frozen=True)
class ToolLimitSettings:
    """Per-profile tool output and listing limits."""

    shell_output_limit_chars: int
    search_output_limit_chars: int
    web_fetch_limit_chars: int
    read_file_limit_chars: int
    diff_limit_chars: int
    search_match_limit: int
    glob_match_limit: int
    list_directory_cap: int
    reference_read_window_lines: int
    grep_max_columns: int


LEGACY_TOOL_LIMITS = ToolLimitSettings(
    shell_output_limit_chars=20_000,
    search_output_limit_chars=20_000,
    web_fetch_limit_chars=20_000,
    read_file_limit_chars=40_000,
    diff_limit_chars=40_000,
    search_match_limit=100,
    glob_match_limit=200,
    list_directory_cap=500,
    reference_read_window_lines=400,
    grep_max_columns=200,
)

PUBLIC_CODEX_TOOL_LIMITS = ToolLimitSettings(
    shell_output_limit_chars=100_000,
    search_output_limit_chars=100_000,
    web_fetch_limit_chars=100_000,
    read_file_limit_chars=200_000,
    diff_limit_chars=200_000,
    search_match_limit=200,
    glob_match_limit=500,
    list_directory_cap=500,
    reference_read_window_lines=1200,
    grep_max_columns=400,
)


def tool_limits_for_profile(profile: QualityProfile) -> ToolLimitSettings:
    """Resolve the configured tool-limit profile."""
    if profile == "public_codex":
        return PUBLIC_CODEX_TOOL_LIMITS
    return LEGACY_TOOL_LIMITS


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

    project_root: str  # /home/mark123/projects/word2kernel
    solution_dir: str  # solution/dsa_attention (relative to project_root)
    notes_dir: str  # notes/dsa_attention (relative to project_root)
    current_round: int = 0
    best_latency_ms: float = float("inf")
    best_round: int = -1
    history: list[RoundRecord] = field(default_factory=list)
    model_name: str = "gpt-5.4"
    quality_profile: QualityProfile = "public_codex"
    tool_limits: ToolLimitSettings = field(default_factory=lambda: PUBLIC_CODEX_TOOL_LIMITS)
    codex_worker_mode: CodexWorkerMode = "off"
    codex_thread_id_coder_engineer: str | None = None
    codex_thread_id_optimizer_engineer: str | None = None


# ---------------------------------------------------------------------------
# Structured output models — used as Agent.output_type so we get programmatic
# access to results instead of parsing free text.
# ---------------------------------------------------------------------------


class StructuredResult(BaseModel):
    """Base model for agent final outputs with strict top-level schemas."""

    model_config = ConfigDict(extra="forbid")


class DesignerResult(StructuredResult):
    """Returned by kernel-designer after producing the design plan."""

    plan_file: str = Field(
        description="Path to the design plan file written by the agent.",
    )
    status: Literal["success", "error"] = Field(
        description="Whether the designer successfully produced the design plan.",
    )
    message: str = Field(
        description="Brief human-readable summary of the design approach or failure.",
    )


class CoderResult(StructuredResult):
    """Returned by kernel-coder after round 0 bootstrap."""

    generated: list[str] = Field(
        description="Paths to the generated kernel artifacts produced by the agent.",
    )
    correctness_verified: bool = Field(
        description="Whether the generated kernel passed the full correctness check.",
    )
    status: Literal["success", "compile_error", "validation_failed"] = Field(
        description="Final implementation outcome for the bootstrap kernel.",
    )
    message: str = Field(
        description="Brief human-readable summary of the implementation result.",
    )
    reflection: str = Field(
        description=(
            "Reflection covering task difficulty, encountered bugs, helpful resources, "
            "and hindsight design changes."
        ),
    )


class PlannerResult(StructuredResult):
    """Returned by kernel-planner after profiling + strategy writing."""

    latency_ms: float = Field(
        description="Measured average latency in milliseconds for the current kernel.",
    )
    bottleneck: str = Field(
        description="Single highest-impact bottleneck identified in the current kernel.",
    )
    strategy_summary: str = Field(
        description="One-line summary of the proposed optimization for the next round.",
    )
    strategy_file: str = Field(
        description="Path to the strategy markdown file written by the planner.",
    )
    is_new_best: bool = Field(
        description="Whether the benchmarked kernel is faster than the previous best.",
    )


class OptimizerResult(StructuredResult):
    """Returned by kernel-optimizer after implementing + validating."""

    kernel_file: str = Field(
        description="Path to the optimized kernel file written by the agent.",
    )
    correctness_verified: bool = Field(
        description="Whether the optimized kernel passed correctness validation.",
    )
    status: Literal["success", "compile_error", "validation_failed", "timeout"] = Field(
        description="Final implementation outcome for the optimization round.",
    )
    message: str = Field(
        description="Brief human-readable summary of the optimization result.",
    )
