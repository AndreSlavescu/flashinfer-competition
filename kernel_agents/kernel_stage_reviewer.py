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
1. Read the full kernel design plan, focusing on the assigned stage's responsibilities, dependencies, validation outputs, and helper APIs.
2. Analyze the latest implementation attempt in `kernel_0.py` for the assigned stage.
3. Use `grep_search` and `read_file` on `last_shell_dump.txt` to inspect the raw tagged validation blocks and any compile/runtime errors. Start with targeted searches for `[PyTorch] <name>` / `[CuTeDSL] <name>` markers before broad reads.
5. Use the raw tagged validation blocks, kernel snapshots, and trimmed stage-result/review histories together to understand the failure.
6. Report at most ONE critical correctness issue in the current stage, including any deviations from the plan and issues with the stage output validation harness. Be precise.
7. Report at most ONE critical correctness issue with the design plan.

## Rules
- It's a LOT more likely that the implementation is wrong, and not the plan. Be absolutely sure when blaming the design.
- When there are multiple issues, report the most critical one
- Missing or mismatched validation outputs for the current stage (`[PyTorch]` or `[CuTeDSL]` logs) are implementation issues.
- The latest entry of the trimmed stage-result history is the current attempt under review.

{..## Full kernel_0_plan.md..}
{..## Kernel Snapshots..}
{..## Trimmed Round0StageResult history..}
{..## Trimmed StageReviewResult history..}
{..## Stage Under Review..}
{..## Recovery Task..}
"""

STAGE_REVIEWER_OUTPUT_REMINDER = """\
"""


def _compose_stage_reviewer_body(ctx: SharedContext) -> str:
    """Render the stage-reviewer body from the active transient prompt sections."""
    prompt_sections = ctx.round0_stage_reviewer_prompt_sections
    body = render_prompt_template(
        NEW_STAGE_REVIEWER_BODY.strip(),
        {
            "{..## Stage Under Review..}": prompt_sections.get("stage_under_review", ""),
            "{..## Full kernel_0_plan.md..}": prompt_sections.get("full_plan", ""),
            "{..## Kernel Snapshots..}": prompt_sections.get("kernel_snapshots", ""),
            "{..## Trimmed Round0StageResult history..}": prompt_sections.get(
                "stage_results_history",
                "",
            ),
            "{..## Trimmed StageReviewResult history..}": prompt_sections.get(
                "review_history",
                "",
            ),
            "{..## Recovery Task..}": prompt_sections.get("recovery_context", ""),
        },
    )

    parts = [body, STAGE_REVIEWER_OUTPUT_REMINDER.strip()]
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
