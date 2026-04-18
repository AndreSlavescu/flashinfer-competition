"""kernel-coder agent — implements the kernel_0.py from the design plan in round 0."""

from __future__ import annotations

from pathlib import Path

from agents import Agent, ModelSettings, RunContextWrapper
from openai.types.shared import Reasoning

from kernel_agents.context import (
    CoderResult,
    CodexWorkerReasoningEffort,
    ReasoningEffort,
    SharedContext,
    Verbosity,
)
from kernel_agents.prompting import (
    CODER_CODEX_WORKER_BLOCK,
    build_round0_coder_body,
    build_agent_instructions,
)
from kernel_agents.tools import build_tools_for_role, tool_names

PLAN_FILENAME = "kernel_0_plan.md"

CODER_INTRO = """\
You are kernel-coder. You implement a CuTeDSL Deepseek Sparse Attention kernel for \
Blackwell B200 (sm100a) into solution/dsa_attention/kernel_0.py, EXACTLY per the design \
plan embedded below in these instructions.
"""

CODER_OUTPUT_CONTRACT = """\
## Output contract (STRICT)
1. Single file: solution/dsa_attention/kernel_0.py. Inline every helper.
2. Every numbered section of the plan must be reflected in kernel_0.py — work \
partition, warp specialization, memory flow, async pipelines, SMEM plan, TMEM plan, \
synchronization, API map. Omitting or collapsing a section (replacing warp \
specialization with a single loop, replacing a split-attention + reduction design \
with one kernel, replacing tcgen05 UMMA with torch ops, etc.) is a failure — return \
status="validation_failed" with the missing section named in `reflection`.
3. Return `CoderResult`. No markdown fences, no prose outside the structured response.
"""

CODER_RULES = """\
## Rules
- Follow the plan VERBATIM. Do not simplify decomposition to make correctness pass.
- If the kernel already reflects the plan structure, apply the smallest fix on \
failure. If a plan-mandated component is missing, adding it is not a 'rewrite' — it \
is required work.
- Two-gate acceptance for status="success":
  - Gate 1 (adherence): every numbered section of the plan is reflected in the file.
  - Gate 2 (correctness): all 23/23 correctness workloads pass.
"""


def _resolve_plan_path(ctx: SharedContext) -> Path:
    """Return the absolute path to kernel_0_plan.md for the current context."""
    root = Path(ctx.project_root)
    solution_dir = Path(ctx.solution_dir)
    if not solution_dir.is_absolute():
        solution_dir = root / solution_dir
    return solution_dir / PLAN_FILENAME


def _compose_coder_body_with_plan(ctx: SharedContext) -> str:
    """Append the verbatim plan file to `CODER_BODY` as a dedicated section.

    Raises FileNotFoundError if the plan is missing — by the time the coder
    runs, round 0a must have produced it.
    """
    plan_path = _resolve_plan_path(ctx)
    if not plan_path.exists():
        raise FileNotFoundError(
            f"kernel-coder instructions require {plan_path}, but it does not exist. "
            "Run kernel-designer (round 0a) before invoking the coder."
        )
    plan_text = plan_path.read_text()
    return build_round0_coder_body(
        intro_block=CODER_INTRO,
        output_contract_block=CODER_OUTPUT_CONTRACT,
        role_rules_block=CODER_RULES,
        extra_sections=(
            f"## Design plan (VERBATIM — implement this)\n\n{plan_text.strip()}",
        ),
    )


def make_kernel_coder(
    context: SharedContext,
    model: str = "gpt-5.4",
    reasoning_effort: ReasoningEffort = "high",
    verbosity: Verbosity = "low",
    extra_instructions: str = "",
    codex_worker_model: str = "gpt-5-codex",
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort = "high",
) -> Agent[SharedContext]:
    """Create the kernel-coder agent with the given model.

    Instructions are resolved dynamically at run time so the latest
    kernel_0_plan.md content is always inlined into the system prompt. The
    agent can be constructed before round 0a runs; the callable only fires
    once Runner.run is invoked for the coder.
    """
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
        body = _compose_coder_body_with_plan(run_ctx.context)
        return build_agent_instructions(
            role="coder",
            body=body,
            tools=tools,
            extra_instructions=extra_instructions,
            codex_worker_block=codex_worker_block,
            include_hardware_spec=False,
        )

    return Agent[SharedContext](
        name="kernel-coder",
        instructions=dynamic_instructions,
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=CoderResult,
    )
