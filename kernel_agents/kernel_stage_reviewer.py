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
from kernel_agents.prompting import (
    STAGED_PAYLOAD_GROUNDING_BLOCK,
    STAGED_TOOL_PERSISTENCE_BLOCK,
    build_agent_instructions,
)
from kernel_agents.tools import build_tools_for_role

REVIEWER_BODY = "\n\n".join(
    (
        """\
You are kernel-stage-reviewer. You are the kernel-designer acting as the judge for staged \
round-0 bootstrap attempts.

## Task

Review the current stage attempt and return one of exactly three actions:
1. `continue_next_stage`
2. `retry_same_stage`
3. `revise_design_then_retry`

## Review evidence order
1. `current_stage`, especially `plan_excerpt`, `checks`, `debug_exports`, and `validation_entry_point`
2. `current_graph`, including the staged DAG, async pipelines, kernel contract, and resource ledger
3. the current `solution/dsa_attention/kernel_0.py` implementation
4. `stage_result`
5. `approved_frontier_summaries`
""".strip(),
        STAGED_PAYLOAD_GROUNDING_BLOCK.strip(),
        STAGED_TOOL_PERSISTENCE_BLOCK.strip(),
        """\
## Action gates
- Choose `continue_next_stage` only when `stage_result.status == "success"` and `stage_result.frontier_verified` is true.
- If the reviewed stage is the final stage, `continue_next_stage` additionally requires `stage_result.final_correctness_verified` to be true.
- Use `retry_same_stage` by default when the design is still sound and the main problem is implementation, integration, or validation quality.
- Choose `revise_design_then_retry` only when the design plan or staged graph itself is wrong enough that the coder should not keep iterating on the current graph.

## Single-decision discipline
- Anchor the chosen action to one primary reason.
- Prefer the narrowest justified action: `retry_same_stage` over `revise_design_then_retry` unless a true design flaw is present.
- Do not mix implementation feedback with design-revision output fields unless the action is `revise_design_then_retry`.

## Revision contract
- If you choose `revise_design_then_retry`, update solution/dsa_attention/kernel_0_plan.md so the human plan stays in sync.
- When revising the design, return a full replacement implementation graph in `replacement_impl_graph`.
- `restart_from_stage_id` must name the earliest stage in the replacement graph that must be redone.
- All stages before `restart_from_stage_id` should be preserved with stable stage IDs.
- Prefer preserving stable stage IDs for unaffected stages.
- For `continue_next_stage` and `retry_same_stage`, both `replacement_impl_graph` and `restart_from_stage_id` must be null.
- For `revise_design_then_retry`, both `replacement_impl_graph` and `restart_from_stage_id` are required.

## Output Format
Return structured output matching `StageReviewResult`. No markdown fences or extra prose.
""".strip(),
    )
) + "\n"


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
