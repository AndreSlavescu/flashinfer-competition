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
- Write the design document to solution/dsa_attention/kernel_0_plan.md
- Return a staged implementation graph aligned with the `DesignerResult` schema.

BASELINE KERNEL (FOR LOGICAL REFERENCE ONLY): \
references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py


## Workflow

1. Read CuTeDSL kernel examples to brainstorm a design for the algorithm. Use the PyTorch baseline only to confirm semantics and edge cases.
2. Write the design plan kernel_0_plan.md, covering:
   - **Work partition**: distributing Qs and topk KVs across CTAs
   - **Warp specialization**: which warps handle each stage (sparse KV loading, QK MMA, softmax, PV MMA, combine partials etc.)
   - **Memory flow**: how each tensor (Q, K, V, P, O etc.) moves between TMEM, RMEM, SMEM, GMEM
   - **Async pipelining**:
     - Overlap opportunities (e.g., gather KV -> QK MMA)
     - Pipeline types (TmaAsync, TmaUmma, AsyncUmma, UmmaAsync, TmaStore etc.)
     - For each pipeline: producer/consumer warps, payload, num_stages (pipeline depth)
     - For each pipeline and warp: SMEM, TMEM, and register budgets for occupancy limits
     - Prefetch strategy
   - **Shared memory plan**: buffer layouts for tensors, total SMEM requirement (pipeline stages)
   - **Tensor memory plan**: column assignments, layouts for tcgen05 mma and ld/st
   - **Synchronization**: barriers and fences at async pipeline, SMEM, TMEM boundaries
3. For each stage of the plan, find ALL CuTeDSL abstractions and APIs that can help with the implementation (pipelining and synchronization, building tma/mma atoms, tiling, creating memory layouts/descriptors etc.)
4. Derive the comprehensive staged implementation graph from kernel_0_plan.md
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
