"""Staged round-0 kernel coder for plan-derived bootstrap execution."""

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

NEW_STAGE_CODER_BODY = """\
You are kernel-stage-coder, an expert at CuTeDSL (CUTLASS Python DSL) programming.

## Workflow
1. Read the full kernel design plan, focusing on the assigned stage's responsibilities, dependencies, validation outputs, and helper APIs 
2. Read the existing implementation in `kernel_0.py` to identify the pre-requisite code paths this stage should extend.
3. Find kernel snippets that use the CuTeDSL abstractions and APIs under the `Key CuTeDSL helpers` section. Plan out how to adapt them for this stage.
4. Implement the assigned kernel stage in CuTeDSL, including all required pipelining, barrier, and buffer logic from pre-requisite stages. Write the computed output validation values to temporary buffers passed into the kernel.
5. Implement the assigned kernel stage in naive PyTorch, computing the same output validation fields.
6. Build/extend `prefix_validation_harness` to launch the CuTeDSL and PyTorch paths on synthetic input. It should compare the output validation fields using the assigned mode (`exact` or `allclose`)

## Rules
- **YOU MUST FOLLOW THE KERNEL STRUCTURE BELOW AS CLOSELY AS POSSIBLE**
- **YOU MUST USE CuTe DECORATORS ON THE KERNEL PATH, OTHERWISE YOU'LL GET MLIR CONTEXT ISSUES. `@cute.jit` for CuTeDSL helpers and warp role implementations, `@cute.kernel` for device entry kernels, and `@cute.struct` for shared-storage or typed CuTe structs.**
- **YOU MUST EXTEND UPON THE PREVIOUS COMPLETED CuTeDSL and PyTorch STAGES. THERE MUST ONLY BE ONE CuTeDSL AND ONE PyTorch PATH**
- **YOU MUST USE cute.printf() and cute.print_tensor() AGGRESSIVELY FOR EXPOSING CuTe objects (layouts, MMA atoms, copy atoms, tiled objects, tensors, fragments, pipelines, barriers etc.) TO THE REVIEWER.**
- Use `cutlass.Constexpr` and type annotations extensively for CuTeDSL JIT compiler

## Stopping rule
- After implementing the stage and prefix validations, call `run_stage_validation` EXACTLY ONCE.
- On the final stage only, if `run_stage_validation` passes, then call `run_correctness_check` EXACTLY ONCE. Otherwise skip it.
- After the last tool call returns, emit the `Round0StageResult` and STOP. Do NOT edit the kernel, re-run validation, or call any other tool afterwards — diagnosis belongs to the stage-reviewer.
- Map the outcome to `status`:
  - `"compile_error"` if compilation failed
  - `"validation_failed"` if the harness or correctness check ran but disagreed with the reference
  - `"success"` if every mandated check passed

{..## Full kernel_0_plan.md..}
{..## Assigned Stage..}
"""

STAGE_CODER_OUTPUT_REMINDER = """"""


def _compose_stage_coder_body(ctx: SharedContext) -> str:
    """Render the stage-coder body from the active transient prompt sections."""
    prompt_sections = ctx.round0_stage_coder_prompt_sections
    body = render_prompt_template(
        NEW_STAGE_CODER_BODY.strip(),
        {
            "{..## Assigned Stage..}": prompt_sections.get("assigned_stage", ""),
            "{..## Full kernel_0_plan.md..}": prompt_sections.get("full_plan", ""),
        },
    )

    parts = [
        body,
        KERNEL_STRUCTURE_BLOCK.strip(),
        STAGE_CODER_OUTPUT_REMINDER.strip(),
    ]
    return "\n\n".join(parts) + "\n"


def make_round0_stage_coder(
    context: SharedContext,
    model: str = "gpt-5.4",
    reasoning_effort: ReasoningEffort = "high",
    verbosity: Verbosity = "low",
    extra_instructions: str = "",
    codex_worker_model: str = "gpt-5-codex",
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort = "high",
) -> Agent[SharedContext]:
    """Create the staged round-0 kernel coder."""

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
        body = _compose_stage_coder_body(run_ctx.context)
        return build_agent_instructions(
            role="coder",
            body=body,
            tools=tools,
            extra_instructions=extra_instructions,
            codex_worker_block=codex_worker_block,
            include_hardware_spec=False,
        )

    return Agent[SharedContext](
        name="kernel-stage-coder",
        instructions=dynamic_instructions,
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=Round0StageResult,
    )
