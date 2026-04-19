# TOOL_SCOPE_FLOW.md

This document describes how tool scoping works for the kernel agents in
`word2kernel` after the current scope-hardening pass.

## Overview

The repository uses three different scope mechanisms:

1. function-tool read guardrails
2. `apply_patch` write enforcement
3. role-specific `codex_kernel_assist` workspace narrowing

The high-level idea is:

- file-reading tools are checked against a per-role read allowlist
- file-writing through `apply_patch` is checked against a per-role write allowlist
- the read-only Codex helper is given a role-specific workspace
- write-capable Codex workers are still a known limitation

## Scope Sources

The canonical allowlists live in `kernel_agents/scoping.py` as `AGENT_SCOPES`.

```text
designer
  read:  references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py
         references/cutlass/examples/python/CuTeDSL/blackwell/mla/
         references/cutlass/python/CuTeDSL/cutlass/cute/
         references/cutlass/python/CuTeDSL/cutlass/pipeline/
         references/cutlass/python/CuTeDSL/cutlass/utils/
         solution/dsa_attention/
  write: solution/dsa_attention/kernel_0_plan.md

reviewer
  read:  references/cutlass/examples/python/CuTeDSL/blackwell/mla/
         references/CuTeGen_guidelines.md
         references/cutlass/python/CuTeDSL/cutlass/cute/
         references/cutlass/python/CuTeDSL/cutlass/pipeline/
         references/cutlass/python/CuTeDSL/cutlass/utils/
         solution/dsa_attention/
         last_shell_dump.txt
  write: (none)

coder
  read:  references/cutlass/examples/python/CuTeDSL/blackwell/mla/
         references/CuTeGen_guidelines.md
         references/cutlass/python/CuTeDSL/cutlass/cute/
         references/cutlass/python/CuTeDSL/cutlass/pipeline/
         references/cutlass/python/CuTeDSL/cutlass/utils/
         solution/dsa_attention/
         last_shell_dump.txt
  write: solution/dsa_attention/

planner
  read:  references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py
         references/cutlass/examples/python/CuTeDSL/blackwell/mla/
         references/cutlass/python/CuTeDSL/cutlass/cute/
         references/cutlass/python/CuTeDSL/cutlass/pipeline/
         references/cutlass/python/CuTeDSL/cutlass/utils/
         solution/dsa_attention/
         notes/dsa_attention/
         last_shell_dump.txt
  write: notes/dsa_attention/

optimizer
  read:  references/
         solution/dsa_attention/
         notes/dsa_attention/
         last_shell_dump.txt
  write: solution/dsa_attention/
```

Directory entries end with `/` and match everything under that directory.
Single-file entries must match exactly.

## Enforcement Flow

### 1. File-access function tools

`read_file`, `glob_files`, `grep_search`, `list_directory`, and `diff_files`
use a `ToolInputGuardrail`.

```text
Agent
  |
  v
Function tool call
  |
  v
make_scope_guardrail(role)
  |
  +--> resolve explicit path args against project_root
  |
  +--> check_path_allowed(resolved_path, project_root, role.read_allow)
  |
  +--> allow or BLOCKED
  |
  v
Tool implementation
```

Notes:

- `glob_files` guardrail checks `directory`, not `pattern`
- `glob_files` also resolves every match and post-filters each returned file
  against `role.read_allow`, so patterns like `../*.py` cannot escape
- `glob_files` requires a valid `current_agent_role` in run context and fails
  closed if it is missing

### 2. `apply_patch`

`apply_patch` does not use a function-tool input guardrail. Instead it uses a
role-specific `ApplyPatchTool` backed by `ScopedWorkspaceEditor`.

```text
Agent
  |
  v
apply_patch
  |
  v
ScopedWorkspaceEditor(role)
  |
  +--> resolve operation.path against project_root
  |
  +--> check_path_allowed(resolved_path, project_root, role.write_allow)
  |
  +--> allow or BLOCKED
  |
  v
WorkspaceEditor(create/update/delete)
```

### 3. `codex_kernel_assist`

`codex_kernel_assist` is created per role inside `build_tools_for_role(...)`.
It is read-only and its workspace is narrowed to that role's allowed directory
roots.

```text
build_tools_for_role(role)
  |
  v
_build_codex_kernel_assist(role)
  |
  +--> choose primary directory as working_directory
  |
  +--> put other allowed directories into additional_directories
  |
  +--> create read-only codex_tool(...)
```

The Codex helper uses directory scope only. Single-file exceptions like
`last_shell_dump.txt` are not exposed through `codex_kernel_assist`.

### 4. Write-capable Codex workers

`codex_coder_engineer` and `codex_optimizer_engineer` are still a known
exception.

```text
coder/optimizer
  |
  v
codex_*_engineer
  |
  v
workspace-write Codex session
  |
  +--> working_directory = repo root
  +--> no per-path read/write allowlist enforcement
```

This is intentional for now because the current Codex SDK/CLI surface does not
offer a clean read-root/write-root split for write-capable sessions.

## Tool Categories

### Strictly read-scoped

These tools are checked against `read_allow`:

- `read_file`
- `glob_files`
- `grep_search`
- `list_directory`
- `diff_files`

### Strictly write-scoped

These writes are checked against `write_allow`:

- `apply_patch`

### Directory-scoped read-only Codex

- `codex_kernel_assist`

### Role-assigned, but not path-guarded by `AGENT_SCOPES`

- `run_stage_validation`
- `run_synthetic_check`
- `run_correctness_check`
- `run_full_benchmark`
- `run_ncu_profile`
- `run_sass_analysis`

These tools are part of the role-specific tool surface, but they are not
enforced by the same path-prefix guardrail/editor mechanism as the file tools.

### Not currently assigned to any role

- `web_fetch`

The helper exists in `kernel_agents/tools.py` and is tested directly, but it is
not exposed through `build_tools_for_role(...)`.

### Known scoping limitation

- `codex_coder_engineer`
- `codex_optimizer_engineer`

## Per-Role Tool Surface

### Designer

Used by `kernel-designer`.

| Tool | Scope |
|---|---|
| `apply_patch` | Can only create/update/delete `solution/dsa_attention/kernel_0_plan.md`. |
| `codex_kernel_assist` | Read-only Codex workspace. `working_directory=references/`, `additional_directories` mirrors the other designer-readable directories. |
| `read_file` | Can read only the designer `read_allow` set. |
| `glob_files` | Can glob only inside an explicit scoped directory. Returned matches are post-filtered to the designer read allowlist. |
| `grep_search` | Can search only within an explicit scoped `path` under the designer read allowlist. |
| `list_directory` | Can list only an explicit scoped directory under the designer read allowlist. |
| `diff_files` | Both `file_a` and `file_b` must be within the designer read allowlist. |

### Reviewer

Used by `kernel-stage-reviewer`.

| Tool | Scope |
|---|---|
| `apply_patch` | No writable paths. All writes are blocked. |
| `codex_kernel_assist` | Read-only Codex workspace. `working_directory=solution/dsa_attention/`, `additional_directories` mirrors the reviewer-readable directories. |
| `read_file` | Can read the reviewer `read_allow` set, including `last_shell_dump.txt`. |
| `glob_files` | Can glob only inside an explicit scoped directory. Returned matches are post-filtered to the reviewer read allowlist. |
| `grep_search` | Can search only within an explicit scoped `path` under the reviewer read allowlist. |
| `list_directory` | Can list only an explicit scoped directory under the reviewer read allowlist. |
| `diff_files` | Both files must be within the reviewer read allowlist. |

### Coder

Used by `kernel-coder`, `kernel-stage-coder`, and `kernel-stage-fixer`.

| Tool | Scope |
|---|---|
| `apply_patch` | Can only write under `solution/dsa_attention/`. |
| `codex_kernel_assist` | Read-only Codex workspace. `working_directory=solution/dsa_attention/`, `additional_directories` mirrors the other coder-readable directories. |
| `read_file` | Can read the coder `read_allow` set, including `last_shell_dump.txt`. |
| `glob_files` | Can glob only inside an explicit scoped directory. Returned matches are post-filtered to the coder read allowlist. |
| `grep_search` | Can search only within an explicit scoped `path` under the coder read allowlist. |
| `list_directory` | Can list only an explicit scoped directory under the coder read allowlist. |
| `diff_files` | Both files must be within the coder read allowlist. |
| `run_stage_validation` | Inspection-oriented single-case validation tool. Not enforced by `AGENT_SCOPES` path guardrails. |
| `run_synthetic_check` | Role-available validation tool. Not enforced by `AGENT_SCOPES` path guardrails. |
| `run_correctness_check` | Role-available validation tool. Not enforced by `AGENT_SCOPES` path guardrails. |
| `codex_coder_engineer` | Optional write-capable Codex worker. `workspace-write` at repo root. Known scoping limitation: not constrained by `AGENT_SCOPES`. |

### Planner

Used by `kernel-planner`.

| Tool | Scope |
|---|---|
| `apply_patch` | Can only write under `notes/dsa_attention/`. |
| `codex_kernel_assist` | Read-only Codex workspace. `working_directory=solution/dsa_attention/`, `additional_directories` mirrors the other planner-readable directories. |
| `read_file` | Can read the planner `read_allow` set, including `last_shell_dump.txt`. |
| `glob_files` | Can glob only inside an explicit scoped directory. Returned matches are post-filtered to the planner read allowlist. |
| `grep_search` | Can search only within an explicit scoped `path` under the planner read allowlist. |
| `list_directory` | Can list only an explicit scoped directory under the planner read allowlist. |
| `diff_files` | Both files must be within the planner read allowlist. |
| `run_synthetic_check` | Role-available validation tool. Not enforced by `AGENT_SCOPES` path guardrails. |
| `run_correctness_check` | Role-available validation tool. Not enforced by `AGENT_SCOPES` path guardrails. |
| `run_ncu_profile` | Planner-only profiling tool. No `AGENT_SCOPES` path guardrail. |
| `run_sass_analysis` | Planner-only SASS analysis tool. Accepts `kernel_file`, but this path is not guarded by the file-tool guardrail mechanism. |
| `run_full_benchmark` | Planner-only benchmark tool. No `AGENT_SCOPES` path guardrail. |

### Optimizer

Used by `kernel-optimizer`.

| Tool | Scope |
|---|---|
| `apply_patch` | Can only write under `solution/dsa_attention/`. |
| `codex_kernel_assist` | Read-only Codex workspace. `working_directory=solution/dsa_attention/`, `additional_directories` mirrors the other optimizer-readable directories. |
| `read_file` | Can read the optimizer `read_allow` set, including `last_shell_dump.txt`. |
| `glob_files` | Can glob only inside an explicit scoped directory. Returned matches are post-filtered to the optimizer read allowlist. |
| `grep_search` | Can search only within an explicit scoped `path` under the optimizer read allowlist. |
| `list_directory` | Can list only an explicit scoped directory under the optimizer read allowlist. |
| `diff_files` | Both files must be within the optimizer read allowlist. |
| `run_synthetic_check` | Role-available validation tool. Not enforced by `AGENT_SCOPES` path guardrails. |
| `run_correctness_check` | Role-available validation tool. Not enforced by `AGENT_SCOPES` path guardrails. |
| `codex_optimizer_engineer` | Optional write-capable Codex worker. `workspace-write` at repo root. Known scoping limitation: not constrained by `AGENT_SCOPES`. |

## End-to-End Build Flow

```text
main.py
  |
  +--> set ctx.current_agent_role
  |
  +--> make agent with build_tools_for_role(role)
          |
          +--> role-scoped apply_patch
          +--> role-scoped file tools
          +--> role-scoped codex_kernel_assist
          +--> role-specific workflow tools
          +--> optional write-capable Codex worker for coder/optimizer
```

## Practical Reading Guide

If you want to know whether a tool is scoped, use this rule:

```text
If it is a file tool or apply_patch:
  check AGENT_SCOPES

If it is codex_kernel_assist:
  check working_directory + additional_directories for that role

If it is a benchmark/profile/validation tool:
  it is role-assigned, but not enforced by the AGENT_SCOPES guardrail/editor mechanism

If it is a write-capable Codex worker:
  treat it as a known exception
```
