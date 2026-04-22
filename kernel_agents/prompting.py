"""Shared prompt blocks and tool rendering for kernel agents."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

from kernel_agents.scoping import AGENT_SCOPES
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

CODER_CODEX_WORKER_BLOCK = """"""

OPTIMIZER_CODEX_WORKER_BLOCK = """"""

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
  never locally. For long Modal logs, inspect `last_shell_dump.txt` with `read_file` or
  `grep_search`.
- `grep_search` before `read_file` under `references/`.
- Do not create spill files. Refine tool calls instead.
"""

ROUND0_CODER_JIT_CACHE_BLOCK = """"""

ROUND0_CODER_TOOL_POLICY_BLOCK = """"""

KERNEL_STRUCTURE_BLOCK = """
## Kernel structure reference for agentic code + validate workflow

```
import cutlass.cute as cute
...

class BlackwellStyleKernel:
    def __init__(...):
        # static configurations (dimensions, scheduler policy, warp roles, register budgets, barriers, tilers, iteration counts etc.)
        ...

    @cute.jit
    def __call__(...):
        # launch-time orchestration (operand layouts, tiled MMA objects, TMA atoms / descriptors / sizes, pipeline configs, scheduler params, grid / cluster shapes, shared-storage structs etc.)
        
        ...
        @cute.struct
        class SharedStorage:
            ...

        self.kernel1_impl(...).launch(...)
        # more kernel launches if needed

    @cute.kernel
    def kernel1_impl(...): ...
    # more kernel implementations if needed

    @staticmethod
    def _compute_grid(...): ...
    # more static config helpers 

    @cute.jit
    def mma(...): ...
    @cute.jit
    def load_tma(...): ...
    @cute.jit
    def epilogue(...): ...
    # more warp role implementations

def torch_reference(...):
    ...

program_compile_cache = dict()

def run(...):
    torch.manual_seed(...)
    def create_synthetic_data(...):
        ...

    program = BlackwellStyleKernel(...)
    cache_key = (...)
    if not (cache_key in program_compile_cache):
        program_compile_cache[cache_key] = cute.compile(program...)
    
    compiled_program = program_compile_cache[cache_key]
    ...

    # dumps output validations in `last_shell_dump.txt` for reviewer agent inspection
    compiled_program(...)
    torch.cuda.synchronize()
    o_ref, lse_ref = torch_reference(...)
```

**YOU MUST USE CuTe DECORATORS ON THE KERNEL PATH, OTHERWISE YOU'LL GET MLIR CONTEXT ISSUES. `@cute.jit` for CuTeDSL helpers and warp role implementations, `@cute.kernel` for device entry kernels, and `@cute.struct` for shared-storage or typed CuTe structs.**
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
        KERNEL_STRUCTURE_BLOCK.strip(),
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

def build_tools_section(tools: Sequence[object]) -> str:
    """Render the actual registered tools into prompt text."""
    names = tool_names(tools)
    lines = ["## Tools You Have"]
    for name in names:
        lines.append(f"- {name}")
    return "\n".join(lines)


def build_scope_references_section(role: str) -> str:
    """Render the role's readable path scope in allowlist order."""
    lines = ["## References"]
    for path in AGENT_SCOPES[role].read_allow:
        lines.append(f"- `{path}`")
    return "\n".join(lines)


def build_agent_instructions(
    *,
    role: str,
    body: str,
    tools: Sequence[object],
    extra_instructions: str = "",
    codex_worker_block: str = "",
    include_hardware_spec: bool = False,
) -> str:
    """Compose a final agent prompt from role-specific and shared blocks.

    Assembly order: body (identity + instructions + workflow) -> hardware specs
    (context) -> scoped references -> codex worker block -> tools section ->
    extra_instructions.
    """
    parts = [body.strip()]
    if include_hardware_spec:
        parts.append(B200_HARDWARE_SPEC_BLOCK.strip())
    parts.append(build_scope_references_section(role))
    if codex_worker_block.strip():
        parts.append(codex_worker_block.strip())
    parts.append(build_tools_section(tools))
    if extra_instructions.strip():
        parts.append(extra_instructions.strip())
    return "\n\n".join(parts) + "\n"
