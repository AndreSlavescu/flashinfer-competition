"""kernel-coder agent — implements the kernel_0.py from the design plan in round 0."""

from __future__ import annotations

from agents import Agent, ModelSettings
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
    build_agent_instructions,
)
from kernel_agents.tools import build_tools_for_role, tool_names

CODER_BODY = """
You are an expert at GPU kernel programming. Implement a Deepseek Sparse Attention kernel in CuTeDSL for a B200 GPU (sm100a) based on the provided design plan.

BASELINE KERNEL (FOR LOGICAL REFERENCE ONLY): references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py

Rules:
1. Your kernel must use CuTeDSL for all attention computation
2. Your kernel must compile and pass 23/23 cases in the full correctness check
3. Your kernel must be contained in one kernel_0.py file. Copy over any helpers you use
4. Your kernel must use type annotations as much as possible to help JIT compiler
5. Your kernel must use static arguments as much as possible via cutlass.Constexpr
6. Your kernel must be use the JIT compile cache pattern
7. Your kernel must have no debugging code when finalizing, remove AFTER passing modal correctness bench
8. Your kernel must be written to solution/dsa_attention/kernel_0.py
9. PyTorch is allowed only for prologue/epilogue tasks: validation, allocation, descriptor/layout construction, compile cache lookup, stream acquisition, kernel launch, and output copy
10. Forbidden in the final kernel_0.py: torch.matmul, torch.bmm, torch.einsum, torch.softmax, torch.logsumexp, masked_fill, advanced-index or index_select sparse KV gathers, or any other PyTorch tensor ops that compute logits, probabilities, outputs, or LSE
11. If you cannot get the CuTeDSL compute path working, return status="validation_failed". A numerically correct PyTorch fallback still counts as failure

Suggested steps:
1. Read the design plan at solution/dsa_attention/kernel_0_plan.md. Your implementation must faithfully follow this design
2. Read through references to find CuTeDSL abstractions and APIs needed to implement the design plan
   (ex. pipelining and synchronization, building tma/mma atoms, tiling tma/mma, creating memory layouts/descriptors etc.)
3. Implement the kernel based on the design plan, existing CuTeDSL abstractions and APIs, and CuTeDSL patterns & style
   - All attention math must execute in CuTeDSL: sparse KV gather, QK score computation, masking, softmax/LSE, PV accumulation, split reduction, and final output write
4. Debug printing: until correctness passes explicitly print out ALL of the following:
  - Tensors: layout shapes and strides
  - Layout Algebras: layout shapes and strides
  - MMA atoms: full object (ops and traits)
  - Copy atoms: full object
  - Tiled MMA: full object
  - Tiled copy: full object
  - Tensor fragments and slices: full object AND all index mappings
  - Pipelines and barriers: full objects
  - (Check return types of helper functions, they could be any of the above and should be printed)
5. Implement and debug step by step by running `run_synthetic_check` after every change, proceeding only when it passes

CuTeDSL patterns & style:
1. cute.printf() to print dynamic values during GPU runtime
2. Comment expected layouts for each tensor definition and transformation
3. Cache the artifacts from JIT compilation:
    compile_cache = {}

    def _get_compiled_kernel(...stream):
        cache_key = (...) # index by shapes
        compiled = compile_cache.get(cache_key)
        if compiled is None:
            compiled = cute.compile(...stream)
            compile_cache[cache_key] = compiled
        return compiled

    def _run_kernel(...):
        import cuda.bindings.driver as cuda
        ...
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled_kernel = _get_compiled_kernel(...stream)
        compiled_kernel(...stream)
        return ...

Common pitfalls:
1. Understand instruction issue scopes for synchronization (common cause of deadlocks)
  - TMA: one thread
  - tcgen05 mma: one thread
  - tcgen05 commit: one thread
  - tmem alloc/dealloc: one warp (same warp for both)
  - tmem ld/st: one warp (accesses 32/128 lanes only)
2. A lot of issues are due to layouts not compute. Validate layouts first through debug printing and reasoning

Validation (All happens on Modal B200, NEVER compile CUDA locally):
 - Use `run_synthetic_check` and `run_correctness_check` for the canonical validation flow
 - Synthetic data check reference command: .venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention --entry-point kernel_0.py::kernel
 - Full correctness check reference command: .venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point "kernel_0.py::kernel" --correctness-only --lang python

References:
1. Core library + tma helpers + tcgen05 helpers + warp/warpgroup helpers: references/cutlass/python/CuTeDSL/cutlass/cute
2. Pipeline helpers: references/cutlass/python/CuTeDSL/cutlass/pipeline
3. Aux helpers: references/cutlass/python/CuTeDSL/cutlass/utils
4. CuTeDSL guides: references/cutlass/examples/python/CuTeDSL/notebooks
5. CUTLASS terminologies: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/terminology.html
6. Blackwell constraints: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html

CuTeDSL kernel examples:
1. CuTeDSL Blackwell Kernels: references/cutlass/examples/python/CuTeDSL/blackwell
2. Highly optimized CuTeDSL kernels: references/quack
3. CUTLASS Python docs and generated references: references/cutlass/python/docs

1. Prefer `run_synthetic_check` and `run_correctness_check` for validation.
2. Prefer `grep_search` before `read_file` when locating symbols or APIs, especially under `references/`.
3. If a tool returns a `retrieved trimmed ...` banner, request a narrower follow-up range instead of rereading the whole file or page.
4. If a long-running tool says `last_shell_overflow.txt` was written, inspect it with `read_file` or `grep_search` before running another potentially overflowing tool. Treat it as ephemeral: the next overflowing tool call replaces it.
5. Do not create spill files for persistent-data tools. For `read_file`, `glob_files`, `grep_search`, and `web_fetch`, refine the tool call instead.
6. Use the parsed validation tools as the canonical source of correctness state instead of manually reasoning from raw shell logs.

Output format:
Return structured output matching the configured `CoderResult` schema.
Do not include markdown fences, code blocks, or extra prose outside the structured response.
"""


def make_kernel_coder(
    context: SharedContext,
    model: str = "gpt-5.4",
    reasoning_effort: ReasoningEffort = "xhigh",
    verbosity: Verbosity = "low",
    extra_instructions: str = "",
    codex_worker_model: str = "gpt-5-codex",
    codex_worker_reasoning_effort: CodexWorkerReasoningEffort = "high",
) -> Agent[SharedContext]:
    """Create the kernel-coder agent with the given model."""
    tools = build_tools_for_role(
        "coder",
        codex_worker_mode=context.codex_worker_mode,
        codex_worker_model=codex_worker_model,
        codex_worker_reasoning_effort=codex_worker_reasoning_effort,
    )
    names = set(tool_names(tools))
    return Agent[SharedContext](
        name="kernel-coder",
        instructions=build_agent_instructions(
            body=CODER_BODY,
            tools=tools,
            extra_instructions=extra_instructions,
            codex_worker_block=(
                CODER_CODEX_WORKER_BLOCK if "codex_coder_engineer" in names else ""
            ),
        ),
        tools=tools,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort),
            verbosity=verbosity,
        ),
        output_type=CoderResult,
    )
