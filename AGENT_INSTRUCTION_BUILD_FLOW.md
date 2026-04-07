# AGENT_INSTRUCTION_BUILD_FLOW.md

## Purpose

This document explains how the repo builds each SDK agent's initial `instructions` string and how runtime configuration changes the tool surface available to that agent.

The important distinction is:

- The SDK agent instruction string is assembled inside `kernel_agents/`.
- Repo-root [AGENTS.md](/home/mark123/projects/word2kernel/AGENTS.md) is ambient guidance for Codex-style tooling and humans, but it is not concatenated directly into `Agent.instructions` for the four main SDK agents.

## Build Flow

### 1. CLI and runtime defaults

`main.py` parses the runtime configuration and constructs `SharedContext`.

Relevant knobs:

- `--quality-profile`
- `--codex-worker-mode`
- `--codex-worker-model`
- `--codex-worker-reasoning-effort`
- `--model`
- `--coder-model`
- `--designer-model`
- per-role reasoning and verbosity flags
- per-role `*_extra` prompt suffixes

Current defaults:

- `quality_profile=public_codex`
- `codex_worker_mode=off`
- `codex_worker_model=gpt-5-codex`
- `codex_worker_reasoning_effort=high`

### 2. SharedContext is created

`run_loop()` builds `SharedContext` with:

- the active `quality_profile`
- `tool_limits=tool_limits_for_profile(quality_profile)`
- `codex_worker_mode`
- persisted Codex thread ID slots for coder and optimizer workers

This happens before any agent is instantiated, so the factories can use the same context.

### 3. Each agent factory chooses a role body

Each role has a fixed role-specific body:

- `kernel_designer.DESIGNER_BODY`
- `kernel_coder.CODER_BODY`
- `kernel_planner.PLANNER_BODY`
- `kernel_optimizer.OPTIMIZER_BODY`

These bodies contain the role’s mission, constraints, and workflow guidance.

### 4. Each factory builds the actual tool list

Each factory calls `build_tools_for_role(...)` in [kernel_agents/tools.py](/home/mark123/projects/word2kernel/kernel_agents/tools.py#L1232).

Inputs that affect the tool list:

- role: `designer`, `coder`, `planner`, `optimizer`
- `codex_worker_mode`
- `codex_worker_model`
- `codex_worker_reasoning_effort`

Inputs that do not change the tool list:

- `quality_profile`
- main agent model name such as `gpt-5.4`
- main agent reasoning effort / verbosity
- retry settings

`quality_profile` changes limits and run config, not tool membership.

### 5. Prompt assembly happens in prompting.py

Each factory calls `build_agent_instructions(...)` from [kernel_agents/prompting.py](/home/mark123/projects/word2kernel/kernel_agents/prompting.py#L73).

The final instruction string is assembled in this order:

1. role-specific body
2. shared Codex-style quality block
3. shared tool-policy block
4. optional Codex-worker block
5. generated "Tools You Have" section based on the real registered tools
6. optional caller-provided extra instructions

The "Tools You Have" section is generated from the actual tool objects via `tool_names(tools)`, so the prompt stays aligned with the runtime tool surface.

### 6. Optional Codex-worker guidance is injected only when relevant

- `kernel_coder` includes the Codex-worker guidance block only if `codex_coder_engineer` is present.
- `kernel_optimizer` includes the Codex-worker guidance block only if `codex_optimizer_engineer` is present.
- `kernel_designer` and `kernel_planner` never receive a write-capable Codex-worker block in the current implementation.

### 7. Model settings are attached separately

After the prompt string is built, the factory creates the `Agent(...)` with:

- `instructions=<assembled string>`
- `tools=<actual role tool list>`
- `model=<role model>`
- `model_settings=<reasoning/verbosity for that agent, where applicable>`
- `output_type=<structured result model>`

This means:

- the instruction text and the model choice are configured separately
- swapping `gpt-5.4` for another main model does not automatically change tool access
- tool access is driven by role plus Codex-worker mode

## Tool Access Matrix

### Base repo tools

All roles start from this base set:

- `apply_patch`
- `web_search`
- `web_fetch`
- `codex_kernel_assist`
- `read_file`
- `glob_files`
- `grep_search`
- `list_directory`

### Performance tools

Coder and optimizer receive:

- `diff_files`
- `run_synthetic_check`
- `run_correctness_check`

Planner receives:

- `diff_files`
- `run_synthetic_check`
- `run_correctness_check`
- `run_ncu_profile`
- `run_sass_analysis`
- `run_full_benchmark`

### Write-capable Codex worker tools

These are added only when `codex_worker_mode=coder_optimizer`:

- coder gets `codex_coder_engineer`
- optimizer gets `codex_optimizer_engineer`

Designer and planner do not get a write-capable Codex worker in the current implementation.

## Config Combinations

### Tool access by role and codex_worker_mode

`designer` + `off`
- base repo tools only

`designer` + `coder_optimizer`
- base repo tools only

`coder` + `off`
- base repo tools
- `diff_files`
- `run_synthetic_check`
- `run_correctness_check`

`coder` + `coder_optimizer`
- base repo tools
- `diff_files`
- `run_synthetic_check`
- `run_correctness_check`
- `codex_coder_engineer`

`planner` + `off`
- base repo tools
- `diff_files`
- `run_synthetic_check`
- `run_correctness_check`
- `run_ncu_profile`
- `run_sass_analysis`
- `run_full_benchmark`

`planner` + `coder_optimizer`
- base repo tools
- `diff_files`
- `run_synthetic_check`
- `run_correctness_check`
- `run_ncu_profile`
- `run_sass_analysis`
- `run_full_benchmark`

`optimizer` + `off`
- base repo tools
- `diff_files`
- `run_synthetic_check`
- `run_correctness_check`

`optimizer` + `coder_optimizer`
- base repo tools
- `diff_files`
- `run_synthetic_check`
- `run_correctness_check`
- `codex_optimizer_engineer`

Only `planner` currently receives `run_full_benchmark`.

### What quality_profile changes

`quality_profile=legacy`
- legacy tool caps
- no automatic run-level compaction unless explicitly requested via `make_run_config("legacy")` override behavior

`quality_profile=public_codex`
- larger Codex-like tool caps
- `parallel_tool_calls=False`
- `truncation="auto"`
- `store=False`
- `extra_args={"context_management": [{"type": "compaction", "compact_threshold": 200000}]}`
- `call_model_input_filter=public_codex_input_filter`

Important: `quality_profile` changes runtime behavior and limits, but not which tools the role can access.

## Codex Worker Tool Configuration

When a write-capable Codex worker is enabled, it is built with:

- `sandbox_mode="workspace-write"`
- `working_directory=PROJECT_ROOT`
- `default_thread_options.model=<codex_worker_model>`
- `default_thread_options.model_reasoning_effort=<codex_worker_reasoning_effort>`
- `default_thread_options.network_access_enabled=False`
- `default_thread_options.web_search_mode="disabled"`
- `default_thread_options.approval_policy="never"`
- `default_turn_options.idle_timeout_seconds=180`
- `persist_session=True`
- `use_run_context_thread_id=True`
- a role-specific `run_context_thread_id_key`

Current role-specific keys:

- coder: `codex_thread_id_coder_engineer`
- optimizer: `codex_thread_id_optimizer_engineer`

## Why the prompt/tool alignment matters

The repo used to maintain large static prompt tool lists. That is easy to drift out of sync with the actual tool registration.

The current flow fixes that by:

- building the tool list first
- rendering the prompt tool section from the real tool objects
- injecting Codex-worker guidance only when the worker is actually registered

That keeps the instruction text, tool schema, and runtime behavior aligned.
