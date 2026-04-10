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
You are kernel-designer, an expert at GPU kernel architecture for high-performance \
Blackwell (B200) CUDA kernels using CuTeDSL (CUTLASS Python DSL).

## Task

Design a Deepseek Sparse Attention kernel in CuTeDSL for B200 (sm100a). \
Write the design to solution/dsa_attention/kernel_0_plan.md.

BASELINE KERNEL (FOR LOGICAL REFERENCE ONLY): \
references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py

## Rules

1. Use optimized B200 (sm100a) features: TMA ld/st, tcgen05 MMA, warp specialization, async pipelining.
2. kernel_0_plan.md describes ONLY the final CuTeDSL design. No bootstrap path, no future optimized path, no PyTorch fallback.
3. You MUST write solution/dsa_attention/kernel_0_plan.md.

## Workflow

1. Read CuTeDSL kernel examples to brainstorm a design for the algorithm. Use the PyTorch baseline only to confirm semantics and edge cases.
2. For every B200/CuTeDSL feature you plan to use, locate the exact API in the reference tree \
and record its file path. You must be able to point to the exact code region(s) that realize each design element.
3. Write the design plan covering:
   - **Work partition**: distributing Qs and topk KVs across CTAs
   - **Warp specialization**: which warps handle each stage (sparse KV loading, QK MMA, softmax, PV MMA, combine partials)
   - **Memory flow**: how each tensor (Q, K, V, P, O etc.) moves between TMEM, RMEM, SMEM, GMEM
   - **Async pipelining**:
     - Overlap opportunities (e.g., gather KV -> QK MMA)
     - Pipeline types (TmaAsync, TmaUmma, AsyncUmma, UmmaAsync, TmaStore etc.)
     - For each pipeline: producer/consumer warps, num_stages (pipeline depth)
     - For each pipeline and warp: SMEM, TMEM, and register budgets for occupancy limits
     - Prefetch strategy
   - **Shared memory plan**: buffer layouts for tensors, total SMEM requirement (pipeline stages)
   - **Tensor memory plan**: column assignments, layouts for tcgen05 mma and ld/st
   - **Synchronization**: barriers and fences at async pipeline, SMEM, TMEM boundaries
   - **CuTeDSL API map**: for each design element, the specific CuTeDSL API/class and its file path under references/
4. Read through all references to find CuTeDSL abstractions and APIs that simplify B200 and PTX features you plan to use \
(pipelining and synchronization, building tma/mma atoms, tiling, creating memory layouts/descriptors etc.)

## References

Architecture: see the B200 hardware specifications block above for measured
properties and latencies. Do not attempt to read a separate architecture file.

CuTeDSL:
1. Core library + tma/tcgen05/warp helpers: references/cutlass/python/CuTeDSL/cutlass/cute
2. Pipeline helpers: references/cutlass/python/CuTeDSL/cutlass/pipeline
3. Aux helpers: references/cutlass/python/CuTeDSL/cutlass/utils
4. CuTeDSL guides: references/cutlass/examples/python/CuTeDSL/notebooks
5. CuTeDSL Blackwell Kernels: references/cutlass/examples/python/CuTeDSL/blackwell
6. Highly optimized CuTeDSL kernels: references/quack
7. CUTLASS Python docs: references/cutlass/python/docs

External:
1. CUTLASS terminologies: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/terminology.html
2. Blackwell constraints: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html

## Tool Policy

- Use `grep_search` before `read_file` when locating symbols or APIs under `references/`.
- If a tool returns a `retrieved trimmed ...` banner, narrow the next request instead of rereading.
- Ground every CuTeDSL or B200 API choice in the reference tree before finalizing.
- Do not create spill files. Refine tool calls instead of dumping to temp files.

## Output Format

Return structured output matching the `DesignerResult` schema.
Do not include markdown fences, code blocks, or extra prose outside the structured response.
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
            body=DESIGNER_BODY,
            tools=tools,
            extra_instructions=extra_instructions,
        ),
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=DesignerResult,
    )
