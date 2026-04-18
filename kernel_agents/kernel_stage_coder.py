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
4. Implement the assigned kernel stage in CuTeDSL, including all required pipeline, handoff, barrier, and buffer logic with pre-requisite stages
5. Add stage validation prints to each field listed under the `Validation outputs` section.
   - For each non-`host` output, emit one matching `[PyTorch Val] <name>: BEGIN/END` block and one matching `[CuTe Val] <name>: BEGIN/END` block.
   - Numeric validation block bodies must be comma-separated numeric values. Signed ints, decimals, and scientific notation are all allowed.
   - For each `host` output, emit one `[CuTe Host] <name>: BEGIN/END` block.
6. Create `prefix_validation_harness` so it launches the CuTeDSL and PyTorch validation paths on synthetic input and prints the required blocks.

## Rules
- Use `cutlass.Constexpr` and type annotations extensively for CuTeDSL JIT compiler
- Run tests only on Modal B200 via `run_stage_validation` / `run_synthetic_check` / `run_correctness_check`
- Implement a JIT compile cache if it doesn't exist yet
- `run_stage_validation` always executes `kernel_0.py::prefix_validation_harness` in dedicated single-case stage-validation mode and appends a `=== Stage Validation Summary ===` block with per-field PASS/FAIL to stdout. Raw logs remain in `last_shell_dump.txt`.
- Every assigned-stage validation output with scope `gmem`/`rmem`/`smem` must produce exactly one `[PyTorch Val] <name>: BEGIN/END` block AND one `[CuTe Val] <name>: BEGIN/END` block with matching `<name>`, and both numeric bodies must be comma-separated values. Missing-pair fields fail validation.
- Every assigned-stage validation output with scope `host` must produce exactly one `[CuTe Host] <name>: BEGIN/END` block. Host blocks surface as `[INFO]` rows in the summary and do not gate PASS/FAIL.

## Stopping rule
- After implementing the stage and prefix validations, call `run_stage_validation` EXACTLY ONCE.
- On the final stage only, if `run_stage_validation` passes, then call `run_correctness_check` EXACTLY ONCE. Otherwise skip it.
- After the last tool call returns, emit the `Round0StageResult` and STOP. Do NOT edit the kernel, re-run validation, or call any other tool afterwards — diagnosis belongs to the stage-reviewer.
- Map the outcome to `status`:
  - `"compile_error"` if compilation failed
  - `"validation_failed"` if the harness or correctness check ran but disagreed with the reference
  - `"success"` if every mandated check passed

{..## Assigned Stage..}
{..## Full kernel_0_plan.md..}
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
            "{..## Assigned Stage..}": prompt_sections.get("assigned_stage", ""),
            "{..## Full kernel_0_plan.md..}": prompt_sections.get("full_plan", ""),
        },
    )

    parts = [body, STAGE_CODER_OUTPUT_REMINDER.strip()]
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
