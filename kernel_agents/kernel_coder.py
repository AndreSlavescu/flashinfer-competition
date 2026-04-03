"""kernel-coder agent — generates the initial kernel_0.py in round 0."""

from __future__ import annotations

from agents import Agent, ModelSettings
from openai.types.shared import Reasoning

from kernel_agents.context import SharedContext
from kernel_agents.tools import ALL_TOOLS

INSTRUCTIONS = """
You are an expert at GPU kernel programming. Generate a Deepseek Sparse Attention kernel in CuTeDSL for a B200 GPU (sm100a).

BASELINE KERNEL (FOR LOGICAL REFERENCE ONLY): references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py

Rules:
1. The final kernel_0.py must use CuTeDSL for all attention compute
2. Your kernel must compile and pass 23/23 cases in the full correctness check
3. Your kernel must be contained in one kernel_0.py file. Copy over any helpers you use
4. Your kernel must use type annotations as much as possible to help JIT compiler
5. Your kernel must use static arguments as much as possible via cutlass.Constexpr
6. Your kernel must be use the JIT compile cache pattern
7. Your kernel must use optimized B200 (sm100a) features as much as possible (ex. TMA ld/st, tcgen05 mma etc.)
8. Your kernel must have no debugging code when finalizing, remove AFTER passing modal bench
9. Your kernel must be written to solution/dsa_attention/kernel_0.py
10. PyTorch is allowed only for prologue/epilogue tasks: validation, allocation, descriptor/layout construction, compile cache lookup, stream acquisition, kernel launch, and output copy
11. Forbidden in the final kernel_0.py: torch.matmul, torch.bmm, torch.einsum, torch.softmax, torch.logsumexp, masked_fill, advanced-index or index_select sparse KV gathers, or any other PyTorch tensor ops that compute logits, probabilities, outputs, or LSE
12. kernel_0_plan.md must describe only the final CuTeDSL design. Do not describe a bootstrap path, a future optimized path, or a PyTorch fallback
13. If you cannot get the CuTeDSL compute path working, return status="validation_failed". A numerically correct PyTorch fallback still counts as failure

Suggested steps:
1. Read CuTeDSL kernel examples first to brainstorm a design for the algorithm. Use the PyTorch baseline only to confirm semantics and edge cases
2. Write a design plan in kernel_0_plan.md, outlining the following:
  - Work partition: How to distribute multiple Qs and topk KVs per Q across all CTAs 
  - Warp specialization: Which warps handle stages like sparse KV loading, QK MMA, softmax, PV MMA, combine partials etc.
  - Flow of memory: How should each tensor (like Q, K, V, P, O etc.) be moved between memory subsystems (TMEM, RMEM, SMEM, GMEM)
  - Asynchronous pipelining:
    - What types of work can be overlapped (ex. gather KV -> QK MMA)
    - What types of pipelines should it use (ex. TmaAsync, TmaUmma, AsyncUmma, UmmaAsync, TmaStore etc.)
    - For each pipeline, which warps are the producer/consumer, what are the num stages (pipeline depth)
    - For each pipeline and warp, how should SMEM, TMEM, and register be budgeted for occupancy limits
    - How could resources be prefetched
  - Shared memory plan: Buffer layouts for tensors, total smem requirement (pipeline stages)
  - Tensor memory plan: Column assignments for tensors, layouts for tcgen05 mma and ld/st
  - Synchronization: How should barriers and fences be placed at async pipeline, SMEM, TMEM boundaries
3. Read through all references to find CuTeDSL abstractions and APIs that simplify any B200 and PTX features you plan to use
   (ex. pipelining and synchronization, building tma/mma atoms, tiling tma/mma, creating memory layouts/descriptors etc.)
4. Implement the kernel based on the design plan, existing CuTeDSL abstractions and APIs, and CuTeDSL patterns & style
   - All attention math must execute in CuTeDSL: sparse KV gather, QK score computation, masking, softmax/LSE, PV accumulation, split reduction, and final output write
5. Debug printing: until correctness passes explicitly print out ALL of the following:
  - Tensors: layout shapes and strides
  - Layout Algebras: layout shapes and strides
  - MMA atoms: full object (ops and traits)
  - Copy atoms: full object
  - Tiled MMA: full object
  - Tiled copy: full object
  - Tensor fragments and slices: full object AND all index mappings
  - Pipelines and barriers: full objects
  - (Check return types of helper functions, they could be any of the above and should be printed)
6. Implement and debug step by step by running the synthetic data check after every change, proceeding only when it passes 

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
 - Synthetic data check: python3 scripts/bench_synthetic.py --solution-dir solution/dsa_attention --entry-point kernel_0.py::kernel
 - Full correctness check: modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point "kernel_0.py::kernel" --correctness-only

References:
1. Core library + tma helpers + tcgen05 helpers + warp/warpgroup helpers: csrc/cutlass/python/CuTeDSL/cutlass/cute
2. Pipeline helpers: csrc/cutlass/python/CuTeDSL/cutlass/pipeline
3. Aux helpers: csrc/cutlass/python/CuTeDSL/cutlass/utils
4. CuTeDSL guides: csrc/cutlass/examples/python/CuTeDSL/notebooks
5. CUTLASS terminologies: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/terminology.html
6. Blackwell constraints: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html

CuTeDSL kernel examples:
1. CuTeDSL Blackwell Kernels: csrc/cutlass/examples/python/CuTeDSL/blackwell
2. Highly optimized CuTeDSL kernels: references/CuTeDSL-kernels/quack 
3. Minimal CuTe C++ examples: csrc/cutlass/examples/cute/tutorial/blackwell

Tools You Have:
1. shell: Execute bash commands (use for `modal run`, compilation, git, etc.). Commands run from the project root.
2. apply_patch: Create, update, or delete files via unified diffs.
3. web_search: Search the web for documentation, examples, CUDA forums, PTX ISA specs.
4. read_file: Read any file with line numbers. Supports range reads (start_line, end_line).
5. glob_files: Find files by pattern (e.g. '**/*.py', 'solution/**/*.cu').
6. grep_search: Search file contents with regex (e.g. 'def kernel', 'tcgen05').
7. web_fetch: Fetch content from a specific URL.

Output format:
Return ONLY a single valid JSON object. Do not include markdown fences or any extra prose.
The JSON object must contain exactly these keys:
1. generated: array of paths to generated design plan and DSA kernel
2. correctness_verified: true if all 23 workloads passed the full correctness check
3. status: "success" or "compile_error" or "validation_failed"
4. message: brief summary of what you did and the result
5. reflection: a single string containing:
   - Rank each task in the design plan by difficulty
   - What bugs did you encounter and how did you fix them? Was there any you couldn't fix?
   - What information did you wish you had upfront? What resources were most helpful?
   - What design decisions would you change in hindsight?
{extra_instructions}
"""


def make_kernel_coder(
    model: str = "gpt-5.4-pro",
    extra_instructions: str = "",
) -> Agent[SharedContext]:
    """Create the kernel-coder agent with the given model."""
    return Agent[SharedContext](
        name="kernel-coder",
        instructions=INSTRUCTIONS.replace("{extra_instructions}", extra_instructions),
        tools=ALL_TOOLS,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="xhigh"),
            verbosity="high",
        ),
        output_type=str,
    )
