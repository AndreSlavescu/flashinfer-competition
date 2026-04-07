"""Shared prompt blocks and tool rendering for kernel agents."""

from __future__ import annotations

from collections.abc import Sequence

from kernel_agents.tools import tool_names

SHARED_QUALITY_BLOCK = """
## Codex-Style Quality Bar

- Autonomy and persistence: Continue until the current subtask is complete or a real blocker remains.
- Dependency checks: Do prerequisite lookup, reading, and comparison work before taking action.
- Terminal and tool hygiene: Prefer repo tools and `apply_patch`; do not emulate tools in shell.
- Parsed validation first: Prefer structured repo validation tools over ad hoc shell parsing whenever a parsed tool exists.
- Verification loop: Before finalizing, check correctness, grounding, and formatting.
"""

COMMON_TOOL_POLICY_BLOCK = """
## Tool Policy

- Prefer `grep_search`, `glob_files`, `list_directory`, and `read_file` to ground repo facts before acting.
- If a tool returns a trimmed banner, narrow the next request instead of rerunning the same broad query.
- Treat `last_shell_overflow.txt` as ephemeral: inspect it before another overflowing shell-like workflow call because the next overflow replaces it.
"""

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
    """Compose a final agent prompt from shared and role-specific blocks."""
    parts = [
        body.strip(),
        SHARED_QUALITY_BLOCK.strip(),
        COMMON_TOOL_POLICY_BLOCK.strip(),
    ]
    if codex_worker_block.strip():
        parts.append(codex_worker_block.strip())
    parts.append(build_tools_section(tools))
    if extra_instructions.strip():
        parts.append(extra_instructions.strip())
    return "\n\n".join(parts) + "\n"
