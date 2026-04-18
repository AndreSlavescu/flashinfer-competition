"""Dedicated reviewer agent for staged round-0 kernel bootstrap."""

from __future__ import annotations

from agents import Agent, ModelSettings, RunContextWrapper
from openai.types.shared import Reasoning

from kernel_agents.context import (
    ReasoningEffort,
    SharedContext,
    StageReviewResult,
    Verbosity,
)
from kernel_agents.prompting import build_agent_instructions, render_prompt_template
from kernel_agents.tools import build_tools_for_role

NEW_STAGE_REVIEWER_BODY = """\
You are kernel-stage-reviewer, a strict CuTeDSL and Deepseek Sparse Attention kernel correctness judge.

## Workflow
1. Read through 'kernel_0_plan.md' and the current stage of 'kernel_0_impl_graph.json' to understand the kernel design
2. Analyze the implementation of the current stage from the latest staged implementation attempt in kernel_0.py
3. Use the stage validation summary, raw tagged validation blocks, and any compile/runtime errors in 'last_shell_dump.txt' together with the trimmed stage-result and review histories to understand the failure.
4. Report at most ONE critical correctness issue in the last stage, including any deviations from the plan and issues with the stage output validation harness. Be precise.
5. Report at most ONE critical correctness issue with the design 'kernel_0_plan.md' and 'kernel_0_impl_graph.json'

## Rules
- It's a LOT more likely that the implementation is wrong, and not the plan. Be absolutely sure when blaming the design.
- When there are multiple issues, report the most critical one
- For each current-stage output with scope `gmem`/`rmem`/`smem`, missing, malformed, duplicated, or mismatched `[PyTorch Val]` / `[CuTe Val]` blocks are implementation issues.
- For each current-stage output with scope `host`, missing or malformed `[CuTe Host]` blocks are implementation issues. These rows are informational but could still reveal logical issues.
- Treat `plan_excerpt` as the source of truth for stage ownership and stage-local responsibilities.
- The latest entry of the trimmed stage-result history is the current attempt under review.

{..## Current Stage Specifications..}
{..## File Diffs..}
{..## Trimmed Round0StageResult history..}
{..## Trimmed StageReviewResult history..}
"""

STAGE_REVIEWER_OUTPUT_REMINDER = """\
## Output
Return structured output matching `StageReviewResult`.
Set `action="retry_same_stage"` for implementation issues, `action="revise_design_then_retry"` for true design flaws, and `action="continue_next_stage"` only when there is no blocking issue and the stage passed its required checks.
Populate the remaining schema fields consistently with the chosen action.
"""


def _compose_stage_reviewer_body(ctx: SharedContext) -> str:
    """Render the stage-reviewer body from the active transient prompt sections."""
    prompt_sections = ctx.round0_stage_reviewer_prompt_sections
    body = render_prompt_template(
        NEW_STAGE_REVIEWER_BODY.strip(),
        {
            "{..## Current Stage Specifications..}": prompt_sections.get("current_stage", ""),
            "{..## File Diffs..}": prompt_sections.get("file_diffs", ""),
            "{..## Trimmed Round0StageResult history..}": prompt_sections.get(
                "stage_results_history",
                "",
            ),
            "{..## Trimmed StageReviewResult history..}": prompt_sections.get(
                "review_history",
                "",
            ),
        },
    )

    parts = [body]
    attempt_metadata = prompt_sections.get("attempt_metadata", "").strip()
    if attempt_metadata:
        parts.append(attempt_metadata)
    parts.append(STAGE_REVIEWER_OUTPUT_REMINDER.strip())
    return "\n\n".join(parts) + "\n"


def make_round0_stage_reviewer(
    context: SharedContext,
    model: str = "gpt-5.4",
    reasoning_effort: ReasoningEffort = "high",
    verbosity: Verbosity = "low",
    extra_instructions: str = "",
) -> Agent[SharedContext]:
    """Create the staged round-0 kernel reviewer."""

    tools = build_tools_for_role("reviewer", codex_worker_mode=context.codex_worker_mode)

    def dynamic_instructions(
        run_ctx: RunContextWrapper[SharedContext],
        agent: Agent[SharedContext],
    ) -> str:
        del agent
        body = _compose_stage_reviewer_body(run_ctx.context)
        return build_agent_instructions(
            role="reviewer",
            body=body,
            tools=tools,
            extra_instructions=extra_instructions,
            include_hardware_spec=True,
        )

    return Agent[SharedContext](
        name="kernel-stage-reviewer",
        instructions=dynamic_instructions,
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=StageReviewResult,
    )
