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
    ROUND0_CODER_REFERENCES_BLOCK,
    build_agent_instructions,
    render_prompt_template,
)
from kernel_agents.tools import build_tools_for_role, tool_names

NEW_STAGE_CODER_BODY = """\
You are kernel-stage-coder, an expert at CuTeDSL (CUTLASS Python DSL) programming.

## Workflow
1. Read through the current stage specifications and the existing stages written in kernel_0.py
2. Read through ALL relevant CuTeDSL abstractions and APIs for implementation
3. Implement the kernel stage in CuTeDSL, including all pipeline, handoff, barrier, and buffer logic with pre-requisite stages
4. Implement the kernel stage in naive PyTorch, extending from prior validation code
5. Extend the stage output validation harness with a CuTeDSL epilogue that moves the declared stage outputs into GMEM and compares them against PyTorch outputs on synthetic inputs

## Rules
- Use `cute.printf()` aggressively for debugging CuTeDSL objects (layouts, MMA atoms, copy atoms, tiled objects, tensors, fragments, pipelines, barriers etc.). Only remove once full correctness check passes.
- Use `cutlass.Constexpr` and type annotations extensively for CuTeDSL JIT compiler
- Run tests only on Modal B200 via `run_stage_validation` / `run_synthetic_check` / `run_correctness_check`
- Implement a JIT compile cache if it doesn't exist yet

{..## Current Stage Specifications..}
{..## Pre-requisite Stage Specifications..}
"""

STAGE_CODER_OUTPUT_REMINDER = """\
## Output
Return structured output matching `Round0StageResult`.
"""


def _compose_stage_coder_body(ctx: SharedContext) -> str:
    """Render the stage-coder body from the active transient prompt sections."""
    prompt_sections = ctx.round0_stage_coder_prompt_sections
    body = render_prompt_template(
        NEW_STAGE_CODER_BODY.strip(),
        {
            "{..## Current Stage Specifications..}": prompt_sections.get("current_stage", ""),
            "{..## Pre-requisite Stage Specifications..}": prompt_sections.get(
                "prerequisite_stages",
                "",
            ),
        },
    )

    parts = [body]
    relevant_async_pipelines = prompt_sections.get("relevant_async_pipelines", "").strip()
    if relevant_async_pipelines:
        parts.append(relevant_async_pipelines)
    attempt_metadata = prompt_sections.get("attempt_metadata", "").strip()
    if attempt_metadata:
        parts.append(attempt_metadata)
    parts.append(ROUND0_CODER_REFERENCES_BLOCK.strip())
    parts.append(STAGE_CODER_OUTPUT_REMINDER.strip())
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
