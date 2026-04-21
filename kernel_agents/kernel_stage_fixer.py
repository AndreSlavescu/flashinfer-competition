"""Staged round-0 kernel fixer for reviewer-driven retry attempts."""

from __future__ import annotations

from agents import Agent, ModelSettings, RunContextWrapper
from openai.types.shared import Reasoning

from kernel_agents.context import (
    CodexWorkerReasoningEffort,
    ReasoningEffort,
    Round0StageResult,
    SharedContext,
    Verbosity,
)
from kernel_agents.prompting import (
    CODER_CODEX_WORKER_BLOCK,
    KERNEL_STRUCTURE_BLOCK,
    build_agent_instructions,
    render_prompt_template,
)
from kernel_agents.tools import build_tools_for_role, tool_names

NEW_STAGE_FIXER_BODY = """\
You are kernel-stage-fixer, an expert at debugging and improving CuTeDSL (CUTLASS Python DSL) kernels.

## Workflow
1. Read the full kernel design plan, focusing on the assigned stage's responsibilities, dependencies, and validation outputs.
2. Read the assigned stage implementation in `kernel_0.py` and the feedback from the stage reviewer.
3. Apply the SMALLEST fixes or improvements necessary to address the reviewer's feedback. Preserve correctness of previous stages and closely follow the kernel structure below.

## Rules
- **YOU MUST USE CuTe DECORATORS ON THE KERNEL PATH, OTHERWISE YOU'LL GET MLIR CONTEXT ISSUES. `@cute.jit` for CuTeDSL helpers and warp role implementations, `@cute.kernel` for device entry kernels, and `@cute.struct` for shared-storage or typed CuTe structs.**
- **YOU MUST USE cute.printf() and cute.print_tensor() FOR PRINTING DEVICE SIDE CuTeDSL OBJECTS**

## Stopping rule
- After implementing the patches, call `run_stage_validation` EXACTLY ONCE.
- On the final stage only, if `run_stage_validation` passes, then call `run_correctness_check` EXACTLY ONCE. Otherwise skip it.
- After the last tool call returns, emit the `Round0StageResult` and STOP. Do NOT edit the kernel, re-run validation, or call any other tool afterwards — diagnosis belongs to the stage-reviewer.
- Map the outcome to `status`:
  - `"compile_error"` if compilation failed
  - `"validation_failed"` if the harness or correctness check ran but disagreed with the reference
  - `"success"` if every mandated check passed

{..## Full kernel_0_plan.md..}
{..## Stage To Fix..}
{..## Last Reviewer Feedback..}
"""

STAGE_FIXER_OUTPUT_REMINDER = """\
## Output
Return structured output matching `Round0StageResult`.
"""


def _compose_stage_fixer_body(ctx: SharedContext) -> str:
    """Render the stage-fixer body from the active transient prompt sections."""
    prompt_sections = ctx.round0_stage_fixer_prompt_sections
    body = render_prompt_template(
        NEW_STAGE_FIXER_BODY.strip(),
        {
            "{..## Stage To Fix..}": prompt_sections.get("stage_to_fix", ""),
            "{..## Full kernel_0_plan.md..}": prompt_sections.get("full_plan", ""),
            "{..## Last Reviewer Feedback..}": prompt_sections.get("last_review", ""),
        },
    )

    parts = [
        body,
        KERNEL_STRUCTURE_BLOCK.strip(),
        STAGE_FIXER_OUTPUT_REMINDER.strip(),
    ]
    return "\n\n".join(parts) + "\n"


def make_round0_stage_fixer(
    context: SharedContext,
    model: str = "gpt-5.4",
    reasoning_effort: ReasoningEffort = "high",
    verbosity: Verbosity = "low",
    extra_instructions: str = "",
    codex_worker_model: str = "gpt-5-codex",
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort = "high",
) -> Agent[SharedContext]:
    """Create the staged round-0 kernel fixer."""

    tools = build_tools_for_role(
        "coder",
        codex_worker_mode=context.codex_worker_mode,
        codex_worker_model=codex_worker_model,
        codex_worker_reasoning_effort=codex_worker_reasoning_effort,
    )
    names = set(tool_names(tools))
    codex_worker_block = (
        CODER_CODEX_WORKER_BLOCK if "codex_coder_engineer" in names else ""
    )

    def dynamic_instructions(
        run_ctx: RunContextWrapper[SharedContext],
        agent: Agent[SharedContext],
    ) -> str:
        del agent
        body = _compose_stage_fixer_body(run_ctx.context)
        return build_agent_instructions(
            role="coder",
            body=body,
            tools=tools,
            extra_instructions=extra_instructions,
            codex_worker_block=codex_worker_block,
            include_hardware_spec=False,
        )

    return Agent[SharedContext](
        name="kernel-stage-fixer",
        instructions=dynamic_instructions,
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=Round0StageResult,
    )
