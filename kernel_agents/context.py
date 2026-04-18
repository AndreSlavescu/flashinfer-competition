"""Shared context and structured output types for the kernel generation agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Shared type aliases used by multiple agents (designer, coder)
# ---------------------------------------------------------------------------

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]
Verbosity = Literal["low", "medium", "high"]
QualityProfile = Literal["legacy", "public_codex"]
CodexWorkerMode = Literal["off", "coder_optimizer"]
CodexWorkerReasoningEffort = Literal["low", "medium", "high", "xhigh"]
Round0Mode = Literal["staged", "legacy"]
Round0StageStatus = Literal["success", "compile_error", "validation_failed"]
StageReviewAction = Literal[
    "continue_next_stage",
    "retry_same_stage",
    "revise_design_then_retry",
]
AsyncPipelineType = Literal[
    "PipelineAsync",
    "PipelineCpAsync",
    "PipelineTmaAsync",
    "PipelineTmaUmma",
    "PipelineAsyncUmma",
    "PipelineUmmaAsync",
    "PipelineClcFetchAsync",
]

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
ROUND0_MODE_CHOICES: tuple[Round0Mode, ...] = ("staged", "legacy")


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
    failure_reason: str = ""  # why this round failed, empty if success
    strategy_file: str = ""  # path to the strategy_N.md file
    ncu_metrics: dict[str, float] | None = None  # full NCU metrics snapshot


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
    current_agent_role: str = ""
    round0_mode: Round0Mode = "staged"
    round0_impl_graph_file: str = ""
    round0_stage_history: list[Round0StageResult] = field(default_factory=list)
    round0_review_history: list[StageReviewResult] = field(default_factory=list)
    round0_stage_coder_prompt_sections: dict[str, str] = field(default_factory=dict)
    round0_stage_fixer_prompt_sections: dict[str, str] = field(default_factory=dict)
    round0_stage_reviewer_prompt_sections: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structured output models — used as Agent.output_type so we get programmatic
# access to results instead of parsing free text.
# ---------------------------------------------------------------------------


class StructuredResult(BaseModel):
    """Base model for agent final outputs with strict top-level schemas."""

    model_config = ConfigDict(extra="forbid")


class AsyncPipelineSpec(BaseModel):
    """Machine-readable structural view of an async pipeline from the design plan."""

    model_config = ConfigDict(extra="forbid")

    pipeline_id: str = Field(description="Stable identifier such as P0/P1.")
    type: AsyncPipelineType = Field(
        description=(
            "Exact single-producer/single-consumer pipeline class name exported by "
            "references/cutlass/python/CuTeDSL/cutlass/pipeline/__init__.py."
        ),
    )
    producer_warp: str = Field(
        description="Single producer warp/group name responsible for the pipeline payload.",
    )
    consumer_warp: str = Field(
        description="Single consumer warp/group name responsible for the pipeline payload.",
    )
    num_stages: int = Field(description="Pipeline depth.")
    payload: str = Field(description="Short description of the payload being handed off.")

    @model_validator(mode="after")
    def _validate_non_empty_pipeline_strings(self) -> "AsyncPipelineSpec":
        scalar_fields = {
            "pipeline_id": self.pipeline_id,
            "type": self.type,
            "producer_warp": self.producer_warp,
            "consumer_warp": self.consumer_warp,
            "payload": self.payload,
        }
        for field_name, value in scalar_fields.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank.")
        return self


StageOutputScope = Literal["gmem", "rmem", "smem", "host"]


class StageOutput(BaseModel):
    """Single validated output from a stage, paired with its CuTe memory scope."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "Output tensor identifier. Used verbatim as the <field> token in "
            "'[PyTorch Val] <field>: BEGIN' / '[CuTe Val] <field>: BEGIN' tags "
            "or '[CuTe Host] <field>: BEGIN' for host-scope outputs."
        ),
    )
    scope: StageOutputScope = Field(
        description=(
            "Validation scope metadata for the output. 'gmem', 'rmem', and 'smem' are "
            "runtime compared through matching [PyTorch Val] / [CuTe Val] blocks; "
            "'host' is inspection-only and uses a [CuTe Host] block."
        ),
    )

    @model_validator(mode="after")
    def _validate_non_empty_name(self) -> "StageOutput":
        if not self.name.strip():
            raise ValueError("name must not be blank.")
        return self


class StageSpec(BaseModel):
    """Single implementation stage derived from the design plan."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(description="Stable stage identifier.")
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Stage IDs that must be completed and validated before this stage runs.",
    )
    outputs: list[StageOutput] = Field(
        min_length=1,
        description=(
            "Stage outputs that MUST be logged for validation or inspection. Each entry provides the output name and its validation scope metadata."
        ),
    )
    relevant_helpers: list[str] = Field(
        min_length=1,
        description=(
            "CuTeDSL APIs, abstractions, snippets, or examples that are especially "
            "relevant when implementing this stage."
        ),
    )
    plan_excerpt: str = Field(
        description=(
            "Exact markdown excerpt copied from kernel_0_plan.md that describes "
            "this stage's intended design."
        ),
    )

    @model_validator(mode="after")
    def _validate_non_empty_required_strings(self) -> "StageSpec":
        scalar_fields = {
            "stage_id": self.stage_id,
            "plan_excerpt": self.plan_excerpt,
        }
        for field_name, value in scalar_fields.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank.")

        if any(not value.strip() for value in self.relevant_helpers):
            raise ValueError("relevant_helpers must not contain blank entries.")

        names = [output.name for output in self.outputs]
        if len(names) != len(set(names)):
            raise ValueError("outputs entries must have unique names.")

        return self

class ImplementationGraph(BaseModel):
    """Machine contract for staged round-0 implementation."""

    model_config = ConfigDict(extra="forbid")

    async_pipelines: list[AsyncPipelineSpec] = Field(
        description="Architectural async pipeline definitions derived from the plan.",
    )
    stages: list[StageSpec] = Field(
        description="Ordered stage DAG used by the staged round-0 coder loop.",
    )


class DesignerResult(StructuredResult):
    """Returned by kernel-designer after producing the design plan."""

    plan_file: str = Field(
        description="Path to the design plan file written by the agent.",
    )
    impl_graph: ImplementationGraph = Field(
        description="Staged implementation graph derived from the plan.",
    )
    status: Literal["success", "error"] = Field(
        description="Whether the designer successfully produced the design plan.",
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


class StageValidationReport(BaseModel):
    """Validation report for a completed round-0 stage."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(description="Stage ID that was validated.")
    report: str = Field(description="Concise report returned by the validation workflow.")


class Round0StageResult(StructuredResult):
    """Returned by the staged round-0 coder after a single stage attempt."""

    stage_id: str = Field(description="Stage ID that was implemented.")
    stage_output_validation_report: str = Field(
        description="Parsed report for the stage output validation of this stage.",
    )
    stage_output_verified: bool = Field(
        description="Whether the stage output validation of this stage passed.",
    )
    final_correctness_verified: bool = Field(
        description="Whether the final round-0 correctness validation for this stage passed.",
    )
    correctness_check_report: str = Field(
        description="Parsed correctness-only report for the final stage.",
    )
    status: Round0StageStatus = Field(
        description="Final outcome for the stage attempt.",
    )
    message: str = Field(
        description="Brief summary of the stage attempt outcome.",
    )


class StageReviewResult(StructuredResult):
    """Returned by the dedicated reviewer for a round-0 stage attempt."""

    stage_id: str = Field(description="Stage ID under review.")
    action: StageReviewAction = Field(
        description="Judge action for the next orchestrator step.",
    )
    message: str = Field(
        description="Brief explanation for the selected action.",
    )


class NCUMetrics(BaseModel):
    """Structured NCU profiling metrics matching KEY_METRICS in tools/ncu/ncu_modal.py.

    Field names are short aliases for the canonical NCU metric names.
    All values default to 0.0 so the planner only needs to fill in what NCU returned.
    """

    model_config = ConfigDict(extra="forbid")

    # Throughput (pct of peak sustained)
    dram_throughput_pct: float = Field(default=0.0, description="dram__throughput.avg.pct_of_peak_sustained_elapsed")
    sm_throughput_pct: float = Field(default=0.0, description="sm__throughput.avg.pct_of_peak_sustained_elapsed")
    # Cache hit rates
    l1_hit_rate_pct: float = Field(default=0.0, description="l1tex__t_sector_hit_rate.pct")
    l2_hit_rate_pct: float = Field(default=0.0, description="lts__t_sector_hit_rate.pct")
    # Occupancy
    occupancy_pct: float = Field(default=0.0, description="sm__warps_active.avg.pct_of_peak_sustained_active")
    # Issue utilization
    issue_active_pct: float = Field(default=0.0, description="smsp__issue_active.avg.pct_of_peak_sustained_active")
    # Warp stall reasons (per issue active)
    stall_barrier: float = Field(default=0.0, description="smsp__average_warps_issue_stalled_barrier_per_issue_active")
    stall_membar: float = Field(default=0.0, description="smsp__average_warps_issue_stalled_membar_per_issue_active")
    stall_long_scoreboard: float = Field(default=0.0, description="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active")
    stall_short_scoreboard: float = Field(default=0.0, description="smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active")
    stall_wait: float = Field(default=0.0, description="smsp__average_warps_issue_stalled_wait_per_issue_active")
    stall_math_pipe_throttle: float = Field(default=0.0, description="smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active")
    # Pipe utilization (pct of peak sustained active)
    pipe_tensor_pct: float = Field(default=0.0, description="smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active")
    pipe_lsu_pct: float = Field(default=0.0, description="smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active")
    # Memory traffic (bytes)
    dram_bytes_read: float = Field(default=0.0, description="dram__bytes_read.sum")
    dram_bytes_write: float = Field(default=0.0, description="dram__bytes_write.sum")
    l2_bytes_hit: float = Field(default=0.0, description="lts__t_bytes_lookup_hit.sum")
    l2_bytes_miss: float = Field(default=0.0, description="lts__t_bytes_lookup_miss.sum")
    # Instructions / cycles
    inst_executed: float = Field(default=0.0, description="smsp__inst_executed.sum")
    cycles_elapsed: float = Field(default=0.0, description="sm__cycles_elapsed.avg")


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
    ncu_metrics: NCUMetrics | None = Field(
        default=None,
        description=(
            "NCU profiling metrics from run_ncu_profile. Fill in the values returned by "
            "the profiler. null if NCU profiling was not run or failed."
        ),
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
    reflection: str = Field(
        default="",
        description=(
            "Brief reflection: what worked, what didn't, and what the next round "
            "should try differently. Empty string if no insight."
        ),
    )
