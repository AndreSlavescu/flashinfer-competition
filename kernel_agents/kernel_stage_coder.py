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
1. Read through the current stage specifications and the pre-requisite stage specifications
2. Read through the pre-requisite stage implementations in kernel_0.py to find areas the current stage and validations should extend upon
3. Find kernel snippets that use the CuTeDSL abstractions and APIs under 'relevant_helpers'. Plan out how to adapt them for the current stage.
4. Implement the kernel stage in CuTeDSL, including all pipeline, handoff, barrier, and buffer logic with pre-requisite stages
5. Create or extend `prefix_validation_outputs_cute` so it runs ALL completed prefix stages plus the current stage in CuTeDSL and exposes the declared stage contract under `outputs`
6. Create or extend `prefix_validation_outputs_torch` so it runs the same prefix through the naive PyTorch implementation and exposes the same `outputs`
7. Create or extend `prefix_validation_harness` so it synthesizes inputs, calls both helpers, and compares the declared `outputs`

## Rules
- Use `cutlass.Constexpr` and type annotations extensively for CuTeDSL JIT compiler
- Run tests only on Modal B200 via `run_stage_validation` / `run_synthetic_check` / `run_correctness_check`
- Implement a JIT compile cache if it doesn't exist yet
- `run_stage_validation` always executes `kernel_0.py::prefix_validation_harness`
- The prefix validation helpers are cumulative. Extend them after each stage; do not replace them with stage-local standalone validators.
- The values exposed under `outputs` must correspond to the current stage's `StageSpec.outputs` contract.

## Stopping rule
- After implementing the stage and prefix validations, call `run_stage_validation` EXACTLY ONCE.
- On the final stage only, if `run_stage_validation` passes, then call `run_correctness_check` EXACTLY ONCE. Otherwise skip it.
- After the last tool call returns, emit the `Round0StageResult` and STOP. Do NOT edit the kernel, re-run validation, or call any other tool afterwards — diagnosis belongs to the stage-reviewer.
- Map the outcome to `status`:
  - `"compile_error"` if compilation failed
  - `"validation_failed"` if the harness or correctness check ran but disagreed with the reference
  - `"success"` if every mandated check passed

{..## Current Stage Specifications..}
{..## Pre-requisite Stage Specifications..}
{..## Last Reviewer Feedback..}
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
            "{..## Last Reviewer Feedback..}": prompt_sections.get("last_review", ""),
        },
    )

    parts = [body]
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
