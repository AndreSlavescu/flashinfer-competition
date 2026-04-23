"""kernel-designer agent — designs the kernel architecture and writes kernel_0_plan.md."""

from __future__ import annotations

from agents import Agent, ModelSettings
from openai.types.shared import Reasoning

from kernel_agents.context import (
    DesignerResult,
    ReasoningEffort,
    SharedContext,
    Verbosity,
)
from kernel_agents.prompting import build_agent_instructions
from kernel_agents.tools import build_tools_for_role

DESIGNER_BODY = """\
You are kernel-designer, an expert in GPU kernel development for the B200 (sm100a) architecture.

## Tasks

- Design a Deepseek Sparse Attention kernel in CuTeDSL (Python CUTLASS DSL) for B200 GPUs.
- Write a design document to solution/dsa_attention/kernel_0_plan.md

BASELINE KERNEL (FOR LOGICAL REFERENCE ONLY): \
references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py


## Workflow

1. Research CuTeDSL kernel examples to write a design plan for the algorithm. Use the PyTorch baseline only to confirm semantics and edge cases.
2. Write the design plan kernel_0_plan.md, with sections:
   - **Problem Specification**: inputs/outputs shapes and layouts, data types, constraints, correctness criteria, edge cases
   - **Launch Configurations**: cluster/grid configs, work partition, tile sizes, persistent CTA scheduling
   - **Warp Schedule**: which warps handle which stage (page table loads, KV TMA, QK MMA, softmax, PV MMA, combine partials, writeback etc.), warp register budgets
   - **Async pipelining**:
     - Analyze warp roles for overlapping opportunities, and organize them into pipelines (PipelineCpAsync, PipelineTmaAsync, PipelineTmaUmma, PipelineAsyncUmma, PipelineUmmaAsync etc.)
     - Custom named barrier and fence synchronization (TMEM, exchange etc.)
     - For all pipelines, determine producer/consumer warps, stage count for every pipeline, tx count (if applicable), barriers
     - For all warps, determine thread counts based on pipeline type for cooperative groups (leader only, full warp etc.)
     - Prefetch strategy
   - **Infrastructure**: MMA atoms, SMEM layouts, TMEM layouts, TMA atoms/descriptors, SharedStorage struct fields (barriers, staged tiles, aux buffers etc.)
3. Break down the plan into small implementation stages, each centering around ONE section and ONE instruction/operation (ex. define MMA atoms and layouts, implement TMA warp role etc.) 
4. For each implementation stage, include:
   - A stable IDs such as `S0`, `S1`...
   - dependencies on other stages and how they are connected (ex. QK mma consumes initial TMA loads)
   - The exact output fields that should be validated for correctness, including dtype/shape expectations, capture location (`GMEM`, `SMEM`, `RMEM`, or `host`), and comparison mode (`exact` or `allclose`)
   - the key CuTeDSL helpers or APIs the stage coder agent should consult
"""


def make_kernel_designer(
    context: SharedContext,
    model: str = "gpt-5.4",
    reasoning_effort: ReasoningEffort = "high",
    verbosity: Verbosity = "low",
    extra_instructions: str = "",
) -> Agent[SharedContext]:
    """Create the kernel-designer agent with the given model."""
    tools = build_tools_for_role("designer", codex_worker_mode=context.codex_worker_mode)
    return Agent[SharedContext](
        name="kernel-designer",
        instructions=build_agent_instructions(
            role="designer",
            body=DESIGNER_BODY,
            tools=tools,
            extra_instructions=extra_instructions,
            include_hardware_spec=True,
        ),
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=DesignerResult,
    )
