"""Shared prompt blocks and tool rendering for kernel agents."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

from kernel_agents.tools import tool_names

B200_HARDWARE_SPEC_BLOCK = """
## NVIDIA B200 (sm100a) Hardware Specifications

- SMs: 148 (8 GPCs), 4 sub-cores per SM (warp_id % 4 mapping)
- HBM3e: 178 GB, 7.67 TB/s peak bandwidth (bus width 7680-bit, mem clock 3996 MHz)
- L2 Cache: 126.5 MB
- Shared Memory per SM: 228 KB (48 KB default per block, up to 228 KB with opt-in)
- TMEM per SM: 512 columns x 128 lanes x 32-bit = 256 KB; alloc granularity 32 cols
- Register File per SM: 256 KB (65536 x 32-bit), max 256 per thread
- Warps per SM: up to 64, max 1024 threads per block
- SM clock: ~1.965 GHz boost (~1.844 GHz sustained under thermal load)
"""

# NOTE: Roofline reference points intentionally omitted — NCU profiling tools
# already return throughput percentages, hit rates, stall reasons, and
# memory-bound classification. Agents should reason from actual NCU output.

CODER_CODEX_WORKER_BLOCK = """
## Write-Capable Codex Worker

- `codex_coder_engineer` is a write-capable Codex worker for bounded coding subtasks.
- Use it when a focused Codex edit or repo investigation pass would speed up progress, but keep the scope concrete.
- Keep validation ownership in this parent agent: run the repo validation tools (`run_stage_validation`,
  `run_synthetic_check`, `run_correctness_check`) yourself as appropriate, inspect their parsed
  summaries yourself, and return the final structured result yourself.
"""

OPTIMIZER_CODEX_WORKER_BLOCK = """
## Write-Capable Codex Worker

- `codex_optimizer_engineer` is a write-capable Codex worker for bounded optimization subtasks.
- Use it when a focused Codex edit or refactor pass would speed up progress, but keep the scope concrete.
- Keep validation ownership in this parent agent: run `run_synthetic_check` and `run_correctness_check` yourself and return the final structured result yourself.
"""

ROUND0_CODER_KERNEL_INTERFACE_BLOCK = """\
## Kernel interface
    def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
- q_nope [T,16,512] bf16, q_pe [T,16,64] bf16, ckv_cache [P,64,512] bf16,
  kpe_cache [P,64,64] bf16, sparse_indices [T,2048] int32 (-1 = padding), sm_scale float
- output [T,16,512] bf16, lse [T,16] fp32 (base-2) — pre-allocated, in-place writes
- Logical reference only: references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py
"""

ROUND0_CODER_SUPPORT_BLOCK = """\
## Shared implementation rules
- All attention math in CuTeDSL. Allowed PyTorch surface: validation, allocation,
  descriptor/layout construction, compile-cache lookup, stream acquisition, kernel launch,
  output copy. Nothing else.
- FORBIDDEN: torch.matmul/bmm/einsum/softmax/logsumexp/masked_fill, advanced-index or
  index_select sparse KV gathers, any torch op computing logits/probs/outputs/LSE.
- Use `cutlass.Constexpr` for static shapes; annotate types for the JIT.
- Use the JIT compile cache pattern below.
- Compile only on Modal B200 via `run_synthetic_check` / `run_correctness_check` —
  never locally. Trust the parsed summaries, not raw shell logs.
- `grep_search` before `read_file` under `references/`.
- Do not create spill files. Refine tool calls instead.
"""

ROUND0_CODER_JIT_CACHE_BLOCK = """\
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
"""

ROUND0_CODER_REFERENCES_BLOCK = """\
## References

- Core library + tma/tcgen05/warp helpers: references/cutlass/python/CuTeDSL/cutlass/cute
- Pipeline helpers: references/cutlass/python/CuTeDSL/cutlass/pipeline
- Aux helpers: references/cutlass/python/CuTeDSL/cutlass/utils
- CuTeDSL guides: references/cutlass/examples/python/CuTeDSL/notebooks
- CuTeDSL Blackwell Kernels: references/cutlass/examples/python/CuTeDSL/blackwell
- Highly optimized CuTeDSL kernels: references/quack
"""

ROUND0_CODER_TOOL_POLICY_BLOCK = """\
## Tool policy
- The repo's parsed validation tools (`run_stage_validation`, `run_synthetic_check`,
  `run_correctness_check`) are the canonical correctness source.
- If a tool returns a `retrieved trimmed ...` banner, narrow the next request.
- If `last_shell_overflow.txt` is written, inspect it before the next overflowing call.
- Do not create spill files. Refine tool calls instead.
"""

def build_round0_coder_body(
    *,
    intro_block: str,
    output_contract_block: str,
    role_rules_block: str,
    extra_sections: Sequence[str] = (),
) -> str:
    """Assemble a round-0 coder prompt from shared and role-specific sections."""
    parts = [
        intro_block.strip(),
        ROUND0_CODER_KERNEL_INTERFACE_BLOCK.strip(),
        output_contract_block.strip(),
        role_rules_block.strip(),
        ROUND0_CODER_SUPPORT_BLOCK.strip(),
        ROUND0_CODER_JIT_CACHE_BLOCK.strip(),
        ROUND0_CODER_REFERENCES_BLOCK.strip(),
        ROUND0_CODER_TOOL_POLICY_BLOCK.strip(),
    ]
    parts.extend(section.strip() for section in extra_sections if section.strip())
    return "\n\n".join(parts) + "\n"


def render_prompt_template(body: str, sections: Mapping[str, str]) -> str:
    """Replace known placeholders in *body*, leaving missing sections intact."""
    rendered = body
    for placeholder, replacement in sections.items():
        if replacement:
            rendered = rendered.replace(placeholder, replacement.strip())
    return rendered

TOOL_PROMPT_DESCRIPTIONS: dict[str, str] = {
    "apply_patch": "Create, update, or delete files via SDK apply-patch diffs.",
    "web_search": "Search the web for documentation, examples, PTX ISA notes, and CUDA/CuTeDSL references.",
    "web_fetch": "Fetch content from a specific URL when you already know the page to inspect.",
    "codex_kernel_assist": "Experimental read-only Codex helper for bounded repo investigation only. Do not use it for edits.",
    "codex_coder_engineer": "Write-capable Codex worker for bounded kernel-coder subtasks. Keep final validation in this agent.",
    "codex_optimizer_engineer": "Write-capable Codex worker for bounded kernel-optimizer subtasks. Keep final validation in this agent.",
    "read_file": "Read any file with line numbers. Use range reads for large files.",
    "glob_files": "Find files by pattern. Prefer scoping with `directory` instead of embedding long prefixes in the pattern.",
    "grep_search": "Search file contents with regex. Prefer this before broad file reads when locating symbols or APIs.",
    "list_directory": "List files and directories at a given path.",
    "diff_files": "Compare two files with a unified diff.",
    "run_ncu_profile": "Run NCU profiling on Modal B200 and return hardware utilization metrics.",
    "run_sass_analysis": "Run SASS analysis on a CuTeDSL kernel and return opcode and pipeline analysis.",
    "run_full_benchmark": "Run the full Modal benchmark and return parsed performance results.",
    "run_stage_validation": "Run a cumulative frontier synthetic validation entry point and return a concise parsed summary.",
    "run_synthetic_check": "Run the fast synthetic correctness sweep and return a concise parsed summary.",
    "run_correctness_check": "Run the full Modal correctness check and return a concise parsed summary.",
}


def build_tools_section(tools: Sequence[object]) -> str:
    """Render the actual registered tools into prompt text."""
    names = tool_names(tools)
    lines = ["## Tools You Have"]
    for index, name in enumerate(names, start=1):
        description = TOOL_PROMPT_DESCRIPTIONS.get(name, "Available tool.")
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def build_agent_instructions(
    *,
    body: str,
    tools: Sequence[object],
    extra_instructions: str = "",
    codex_worker_block: str = "",
    include_hardware_spec: bool = False,
) -> str:
    """Compose a final agent prompt from role-specific and shared blocks.

    Assembly order: body (identity + instructions + workflow) -> hardware specs
    (context) -> codex worker block -> tools section -> extra_instructions.
    """
    parts = [body.strip()]
    if include_hardware_spec:
        parts.append(B200_HARDWARE_SPEC_BLOCK.strip())
    if codex_worker_block.strip():
        parts.append(codex_worker_block.strip())
    parts.append(build_tools_section(tools))
    if extra_instructions.strip():
        parts.append(extra_instructions.strip())
    return "\n\n".join(parts) + "\n"
