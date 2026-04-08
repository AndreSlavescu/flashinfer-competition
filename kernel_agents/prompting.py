"""Shared prompt blocks and tool rendering for kernel agents."""

from __future__ import annotations

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

## Key Measured Latencies

- L1 hit: ~36 cycles (18 ns) | L2 hit: ~300 cycles (153 ns) | HBM cold: ~707 cycles (360 ns)
- TMA load: ~200-400 cycles | tcgen05 MMA: ~32 cycles
- L1 bypass penalty: 7-8x for working sets < 32 KB, zero beyond L2
"""

# NOTE: Roofline reference points intentionally omitted — NCU profiling tools
# already return throughput percentages, hit rates, stall reasons, and
# memory-bound classification. Agents should reason from actual NCU output.

CODER_CODEX_WORKER_BLOCK = """
## Write-Capable Codex Worker

- `codex_coder_engineer` is a write-capable Codex worker for bounded coding subtasks.
- Use it when a focused Codex edit or repo investigation pass would speed up progress, but keep the scope concrete.
- Keep validation ownership in this parent agent: run `run_synthetic_check` and `run_correctness_check` yourself, inspect their parsed summaries yourself, and return the final structured result yourself.
"""

OPTIMIZER_CODEX_WORKER_BLOCK = """
## Write-Capable Codex Worker

- `codex_optimizer_engineer` is a write-capable Codex worker for bounded optimization subtasks.
- Use it when a focused Codex edit or refactor pass would speed up progress, but keep the scope concrete.
- Keep validation ownership in this parent agent: run `run_synthetic_check` and `run_correctness_check` yourself and return the final structured result yourself.
"""

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
    "run_synthetic_check": "Run the fast synthetic correctness sweep and return a concise parsed summary.",
    "run_correctness_check": "Run the full Modal correctness check and return a concise parsed summary.",
}


def build_tools_section(tools: Sequence[object]) -> str:
    """Render the actual registered tools into prompt text."""
    names = tool_names(tools)
    lines = ["## Tools You Have"]
    for index, name in enumerate(names, start=1):
        description = TOOL_PROMPT_DESCRIPTIONS.get(name, "Available tool.")
        lines.append(f"{index}. {name}: {description}")
    return "\n".join(lines)


def build_agent_instructions(
    *,
    body: str,
    tools: Sequence[object],
    extra_instructions: str = "",
    codex_worker_block: str = "",
) -> str:
    """Compose a final agent prompt from role-specific and shared blocks.

    Assembly order: body (identity + instructions + workflow) -> hardware specs
    (context) -> codex worker block -> tools section -> extra_instructions.
    """
    parts = [body.strip()]
    parts.append(B200_HARDWARE_SPEC_BLOCK.strip())
    if codex_worker_block.strip():
        parts.append(codex_worker_block.strip())
    parts.append(build_tools_section(tools))
    if extra_instructions.strip():
        parts.append(extra_instructions.strip())
    return "\n\n".join(parts) + "\n"
