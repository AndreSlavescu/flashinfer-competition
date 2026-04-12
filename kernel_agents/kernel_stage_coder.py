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
    STAGED_DIFF_FIRST_CODE_DISCIPLINE_BLOCK,
    STAGED_PAYLOAD_GROUNDING_BLOCK,
    STAGED_TOOL_PERSISTENCE_BLOCK,
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

STAGE_CODER_CALLER_PAYLOAD_CONTRACT = """\
## Caller payload contract
- `current_stage` defines the active stage contract. Its `plan_excerpt`, `outputs`, `relevant_helpers`, and `validation_entry_point` govern this attempt.
- `relevant_async_pipelines` names the async pipelines that matter to this stage. Only activate or extend pipeline behavior that is required for the current cumulative frontier.
- `resource_ledger` is the frozen kernel-wide budget for warps, TMEM, SMEM, registers, barriers, and pipeline depth unless the designer later revises the graph. Stage-local resource pressure usually comes from `current_stage`, `relevant_async_pipelines`, and the stage plan excerpt rather than a separate per-stage ledger.
- `kernel_contract` is the stable kernel interface and workspace contract.
- `approved_frontier_summaries` are the regression contract for previously completed prefixes. Preserve them while extending the frontier.
- `is_final_stage` only changes whether you must run the full correctness gate after frontier validation passes.
"""

STAGE_CODER_RULES = """\
## Scope and invariants
1. Ownership is advisory, not absolute. Prefer minimal edits, but you may adjust earlier code if needed to keep the cumulative kernel coherent.
2. Do not simplify, erase, or collapse already approved kernel structure just to make the current frontier pass.
3. Implement the current stage together with any prerequisite-stage pipelining, handoff, barrier, or buffer logic that becomes active in the current validation frontier.
4. You are responsible for implementing and maintaining the cumulative prefix-frontier validation harness named by `validation_entry_point`.
"""

STAGE_CODER_WORKFLOW = """\
## Stage workflow
1. Implement the active frontier in `kernel_0.py`, including any prerequisite integration that is now live.
2. During staged bring-up, use temporary `cute.printf()` instrumentation aggressively across the active CuTeDSL objects that are relevant to the frontier: layouts, MMA atoms, copy atoms, tiled objects, tensors, fragments, pipelines, barriers, and similar kernel-state objects.
3. Keep those temporary `cute.printf()` calls in place for every non-final stage and through the final stage's frontier/synthetic validation pass.
4. Implement or extend the cumulative prefix-frontier validation harness named by `validation_entry_point`, along with the eager prefix reference model it compares against.
5. Derive the minimal validation-only GMEM exports from the stage `outputs`; materialize only what the harness needs to validate the declared frontier.
6. Run only the current stage's `validation_entry_point` with `run_stage_validation`; it must validate the full declared prefix through this stage, including prerequisite behavior that is now active.
7. If `is_final_stage` is true and the frontier validator passes, remove the temporary `cute.printf()` instrumentation before running `run_correctness_check`.
"""

STAGE_CODER_STATUS_RUBRIC = """\
## `Round0StageResult` status rubric
- Emit `status="success"` only when the required frontier validation passes, the active stage is integrated coherently, and the final correctness gate also passes when `is_final_stage` is true.
- For non-final stages, emit `status="success"` only when `frontier_verified` is true. Leave `final_correctness_verified` false unless a real final correctness run happened.
- Emit `status="compile_error"` when a compile/import/runtime failure blocks a trustworthy frontier validation result.
- Emit `status="validation_failed"` when the code runs but the frontier check fails, the final correctness gate fails, or a plan-mandated stage contract is still missing.
- `message` should name the highest-signal outcome for this attempt.
- `reflection` should summarize the concrete root cause, the smallest next fix, and any regression risk to approved prefixes.
- Never report success when the frontier validator fails, when the final stage skips the correctness gate, when final-stage correctness still runs with temporary `cute.printf()` debug instrumentation enabled, or when the returned fields disagree with the actual tool evidence.
"""

STAGE_CODER_BODY = build_round0_coder_body(
    intro_block=STAGE_CODER_INTRO,
    output_contract_block=STAGE_CODER_OUTPUT_CONTRACT,
    role_rules_block=STAGE_CODER_RULES,
    extra_sections=(
        STAGE_CODER_CALLER_PAYLOAD_CONTRACT,
        STAGED_PAYLOAD_GROUNDING_BLOCK,
        STAGED_TOOL_PERSISTENCE_BLOCK,
        STAGED_DIFF_FIRST_CODE_DISCIPLINE_BLOCK,
        STAGE_CODER_WORKFLOW,
        STAGE_CODER_STATUS_RUBRIC,
    ),
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
            include_hardware_spec=False,
        ),
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=Round0StageResult,
    )
