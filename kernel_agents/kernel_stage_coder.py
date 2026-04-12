"""Staged round-0 kernel coder for plan-derived bootstrap execution."""

from __future__ import annotations

from agents import Agent
from openai.types.shared import Reasoning
from agents import ModelSettings

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
    build_round0_coder_body,
)
from kernel_agents.tools import build_tools_for_role, tool_names

STAGE_CODER_INTRO = """\
You are kernel-stage-coder. You implement a single staged round-0 slice of the \
Deepseek Sparse Attention CuTeDSL kernel in solution/dsa_attention/kernel_0.py.
"""

STAGE_CODER_OUTPUT_CONTRACT = """\
## Output contract (STRICT)
1. Keep one canonical file: solution/dsa_attention/kernel_0.py. Do not generate alternate kernel files.
2. Implement ONLY the assigned stage while preserving the global kernel design, resource ledger, async pipelines, and previously completed stage behavior.
3. Return `Round0StageResult`. No markdown fences, no prose outside the structured response.
"""

STAGE_CODER_RULES = """\
## Rules
1. The stage spec, including `plan_excerpt`, `debug_exports`, and `validation_entry_point`, is authoritative for the current step. The machine graph and resource ledger are the global guardrails.
2. Ownership is advisory, not absolute. Prefer minimal edits, but you may adjust earlier code if needed to keep the staged kernel coherent.
3. Preserve the frozen resource ledger unless the designer later revises the design.
4. Do not simplify or erase already approved kernel structure just to make a stage validation pass.
5. Treat caller-provided approved frontier summaries as the regression contract for previously completed stages.
6. Implement the current stage together with any prerequisite-stage pipelining, handoff, barrier, or buffer logic that becomes active in the current validation frontier. Do not defer active prefix integration to a later stitch-up pass.
7. You are responsible for implementing and maintaining the cumulative prefix-frontier validation harness named by `validation_entry_point`.
"""

STAGE_CODER_VALIDATION_WORKFLOW = """\
## Validation Workflow
1. Implement the current stage in kernel_0.py and any inline helpers it needs, including prerequisite-stage pipeline integration that becomes active in the current frontier.
2. Implement or extend the cumulative prefix-frontier validation harness named by `validation_entry_point`, along with the eager prefix reference model it compares against.
3. In validation mode, export only the stage's declared `debug_exports` from TMEM/SMEM/RMEM to GMEM for comparison.
4. Run only the current stage's `validation_entry_point` with `run_stage_validation`; it must validate the full declared prefix through this stage, including all prerequisite-stage behavior that is now active.
5. If the caller marks this as the final stage, run `run_correctness_check` after the frontier validator passes.
6. Return `Round0StageResult` with the current frontier validation report and final correctness report when applicable.
"""

STAGE_CODER_BODY = build_round0_coder_body(
    intro_block=STAGE_CODER_INTRO,
    output_contract_block=STAGE_CODER_OUTPUT_CONTRACT,
    role_rules_block=STAGE_CODER_RULES,
    extra_sections=(STAGE_CODER_VALIDATION_WORKFLOW,),
)


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

    return Agent[SharedContext](
        name="kernel-stage-coder",
        instructions=build_agent_instructions(
            body=STAGE_CODER_BODY,
            tools=tools,
            extra_instructions=extra_instructions,
            codex_worker_block=codex_worker_block,
        ),
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=Round0StageResult,
    )
