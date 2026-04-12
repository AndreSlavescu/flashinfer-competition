"""Designer-as-judge agent for staged round-0 kernel bootstrap."""

from __future__ import annotations

from agents import Agent, ModelSettings
from openai.types.shared import Reasoning

from kernel_agents.context import (
    ReasoningEffort,
    SharedContext,
    StageReviewResult,
    Verbosity,
)
from kernel_agents.prompting import build_agent_instructions
from kernel_agents.tools import build_tools_for_role

REVIEWER_BODY = """\
You are kernel-stage-reviewer. You are the kernel-designer acting as the judge for staged \
round-0 bootstrap attempts.

## Task

Review the current stage attempt against:
- the human design plan
- the staged implementation graph
- the current kernel_0.py implementation
- the current cumulative frontier validation report
- approved frontier summaries for previously completed stages

Return one of exactly three actions:
1. `continue_next_stage`
2. `retry_same_stage`
3. `revise_design_then_retry`

## Decision Policy

- Choose `continue_next_stage` only when the current cumulative frontier is coherent, the current stage is implemented well enough, and the provided approved prefixes remain consistent with the new frontier.
- Choose `retry_same_stage` when the current design is still sound and the coder should keep iterating with guidance.
- Choose `revise_design_then_retry` only when the design or staged graph itself is flawed.

## Revision Rules

- If you choose `revise_design_then_retry`, update solution/dsa_attention/kernel_0_plan.md so the human plan stays in sync.
- When revising the design, return a full replacement implementation graph in `replacement_impl_graph`.
- `restart_from_stage_id` must name the earliest stage in the replacement graph that must be redone.
- All stages before `restart_from_stage_id` should be preserved with stable stage IDs.
- Prefer preserving stable stage IDs for unaffected stages.

## Output Format

Return structured output matching `StageReviewResult`. No markdown fences or extra prose.
"""


def make_round0_stage_reviewer(
    context: SharedContext,
    model: str = "gpt-5.4",
    reasoning_effort: ReasoningEffort = "high",
    verbosity: Verbosity = "low",
    extra_instructions: str = "",
) -> Agent[SharedContext]:
    """Create the staged round-0 design reviewer."""

    tools = build_tools_for_role("designer", codex_worker_mode=context.codex_worker_mode)
    return Agent[SharedContext](
        name="kernel-stage-reviewer",
        instructions=build_agent_instructions(
            body=REVIEWER_BODY,
            tools=tools,
            extra_instructions=extra_instructions,
        ),
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=StageReviewResult,
    )
