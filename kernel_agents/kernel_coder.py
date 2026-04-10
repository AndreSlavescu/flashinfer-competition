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
    build_agent_instructions,
)
from kernel_agents.tools import build_tools_for_role, tool_names

PLAN_FILENAME = "kernel_0_plan.md"

CODER_BODY = """\
You are kernel-coder. You implement a CuTeDSL Deepseek Sparse Attention kernel for \
Blackwell B200 (sm100a) into solution/dsa_attention/kernel_0.py, EXACTLY per the design \
plan embedded below in these instructions.

## Kernel interface
    def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
- q_nope [T,16,512] bf16, q_pe [T,16,64] bf16, ckv_cache [P,64,512] bf16,
  kpe_cache [P,64,64] bf16, sparse_indices [T,2048] int32 (-1 = padding), sm_scale float
- output [T,16,512] bf16, lse [T,16] fp32 (base-2) — pre-allocated, in-place writes
- Logical reference only: references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py

## Output contract (STRICT)
1. Single file: solution/dsa_attention/kernel_0.py. Inline every helper.
2. Implement every section of the plan VERBATIM — work partition, warp specialization, \
memory flow, async pipelines, SMEM plan, TMEM plan, synchronization. A scalar, \
warp-per-head, or single-kernel fallback that only passes correctness is a failure: \
return status="validation_failed" instead.
3. All attention math in CuTeDSL. Allowed PyTorch surface: validation, allocation, \
descriptor/layout construction, compile-cache lookup, stream acquisition, kernel launch, \
output copy. Nothing else.
4. FORBIDDEN: torch.matmul/bmm/einsum/softmax/logsumexp/masked_fill, advanced-index or \
index_select sparse KV gathers, any torch op computing logits/probs/outputs/LSE.
5. Return `CoderResult`. No markdown fences, no prose outside the structured response.

## Rules
- Follow the plan VERBATIM. Do not simplify decomposition to make correctness pass.
- Use `cutlass.Constexpr` for static shapes; annotate types for the JIT.
- Use the JIT compile cache pattern below.
- Compile only on Modal B200 via `run_synthetic_check` / `run_correctness_check` — \
never locally. Trust the parsed summaries, not raw shell logs.
- `grep_search` before `read_file` under `references/`.
- On failure: apply the SMALLEST fix. Do not rewrite working code.
- Gate: all 23/23 correctness workloads must pass before returning status="success".

## JIT compile cache pattern
```
compile_cache = {}

def _get_compiled_kernel(..., stream):
    cache_key = (...)  # index by shapes
    compiled = compile_cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(..., stream)
        compile_cache[cache_key] = compiled
    return compiled
```

## References (follow the plan's own API map for exact call sites)
- references/cutlass/python/CuTeDSL/cutlass/cute         — core, tma, tcgen05, warp helpers
- references/cutlass/python/CuTeDSL/cutlass/pipeline     — PipelineTma*/PipelineAsync*/PipelineUmma*
- references/cutlass/python/CuTeDSL/cutlass/utils        — blackwell_helpers, smem/tmem allocators
- references/cutlass/examples/python/CuTeDSL/blackwell   — warp-specialized B200 kernels (MLA)
- references/quack                                        — optimized CuTeDSL kernels

## Tool policy
- `run_synthetic_check` / `run_correctness_check` are the canonical correctness source.
- If a tool returns a `retrieved trimmed ...` banner, narrow the next request.
- If `last_shell_overflow.txt` is written, inspect it before the next overflowing call.
- Do not create spill files. Refine tool calls instead.
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
    return (
        f"{CODER_BODY.rstrip()}\n\n"
        f"## Design plan (VERBATIM — implement this)\n\n"
        f"{plan_text.strip()}\n"
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
            body=body,
            tools=tools,
            extra_instructions=extra_instructions,
            codex_worker_block=codex_worker_block,
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
