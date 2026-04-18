# AGENT_INSTRUCTION_BUILD_FLOW.md

## Purpose

This document explains how the repo builds each SDK agent's `instructions`
string and how runtime configuration changes the actual tool surface exposed to
that agent.

The important distinction is:

- The SDK agent instruction strings are assembled inside `kernel_agents/`.
- Repo-root [AGENTS.md](/home/mark123/projects/word2kernel/AGENTS.md) is ambient
  guidance for Codex and humans, but it is not concatenated directly into
  `Agent.instructions`.

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
- transient prompt sections for staged round-0 coder, fixer, and reviewer runs

### 3. Agent factories choose a role body

The repo has both primary round agents and staged round-0 agents.

Primary bodies:

- `kernel_designer.DESIGNER_BODY`
- `kernel_coder.CODER_*`
- `kernel_planner.PLANNER_BODY`
- `kernel_optimizer.OPTIMIZER_BODY`

Staged round-0 bodies:

- `kernel_stage_coder.NEW_STAGE_CODER_BODY`
- `kernel_stage_fixer.NEW_STAGE_FIXER_BODY`
- `kernel_stage_reviewer.NEW_STAGE_REVIEWER_BODY`

These bodies hold the role mission, constraints, workflow guidance, and any
runtime placeholders that `main.py` later fills in.

### 4. Each factory builds the actual tool list

Each factory calls `build_tools_for_role(...)` in
[kernel_agents/tools.py](/home/mark123/projects/word2kernel/kernel_agents/tools.py).

Inputs that affect the tool list:

- role: `designer`, `reviewer`, `coder`, `planner`, `optimizer`
- `codex_worker_mode`
- `codex_worker_model`
- `codex_worker_reasoning_effort`

Inputs that do not change tool membership:

- `quality_profile`
- main agent model such as `gpt-5.4`
- main agent reasoning effort / verbosity
- retry settings

`quality_profile` changes limits and run config, not tool membership.

### 5. Prompt assembly happens in prompting.py

Each factory calls `build_agent_instructions(...)` from
[kernel_agents/prompting.py](/home/mark123/projects/word2kernel/kernel_agents/prompting.py).

The final instruction string is assembled in this order:

1. role-specific body
2. optional hardware-spec block
3. role-scoped references section
4. optional Codex-worker guidance block
5. generated `## Tools You Have` section based on the real registered tools
6. optional caller-provided extra instructions

The tools section is generated from the real tool objects via `tool_names(tools)`,
so prompt text stays aligned with runtime tool registration.

### 6. Dynamic staged prompts are resolved at runtime

The staged round-0 agents use placeholders that `main.py` resolves right before
execution:

- stage coder gets the assigned stage and full plan
- stage fixer gets the stage, full plan, and last reviewer feedback
- stage reviewer gets the stage under review, full plan, kernel snapshot paths,
  trimmed histories, and any recovery context

Those fully materialized prompts can optionally be dumped under `prompts/` via
`--dump-prompts`.

### 7. Model settings are attached separately

After the prompt string is built, the factory creates the `Agent(...)` with:

- `instructions=<assembled string>`
- `tools=<actual role tool list>`
- `model=<role model>`
- `model_settings=<reasoning/verbosity for that agent, where applicable>`
- `output_type=<structured result model>`

Swapping `gpt-5.4` for another main model does not automatically change tool
access. Tool access is driven by role plus `codex_worker_mode`.

## Live Tool Access Matrix

### Shared inspection tools

Every role currently receives:

- `apply_patch`
- `codex_kernel_assist`
- `read_file`
- `glob_files`
- `grep_search`
- `list_directory`
- `diff_files`

The repo still defines standalone helpers like `web_fetch`, but they are not
wired into any agent role through `build_tools_for_role(...)`.

### Role-specific workflow tools

`designer`
- no extra workflow tools beyond the shared inspection tools

`reviewer`
- no extra workflow tools beyond the shared inspection tools

`coder`
- `run_synthetic_check`
- `run_correctness_check`
- `run_stage_validation`

`planner`
- `run_synthetic_check`
- `run_correctness_check`
- `run_ncu_profile`
- `run_sass_analysis`
- `run_full_benchmark`

`optimizer`
- `run_synthetic_check`
- `run_correctness_check`

### Staged round-0 role mapping

- `kernel-stage-coder` uses the `coder` tool surface
- `kernel-stage-fixer` uses the `coder` tool surface
- `kernel-stage-reviewer` uses the `reviewer` tool surface

### Write-capable Codex worker tools

These are added only when `codex_worker_mode=coder_optimizer`:

- `coder` gets `codex_coder_engineer`
- `optimizer` gets `codex_optimizer_engineer`

No other role gets a write-capable Codex worker in the current implementation.

## Config Combinations

### Tool access by role and codex_worker_mode

`designer` + `off`
- shared inspection tools only

`designer` + `coder_optimizer`
- shared inspection tools only

`reviewer` + `off`
- shared inspection tools only

`reviewer` + `coder_optimizer`
- shared inspection tools only

`coder` + `off`
- shared inspection tools
- `run_synthetic_check`
- `run_correctness_check`
- `run_stage_validation`

`coder` + `coder_optimizer`
- shared inspection tools
- `run_synthetic_check`
- `run_correctness_check`
- `run_stage_validation`
- `codex_coder_engineer`

`planner` + `off`
- shared inspection tools
- `run_synthetic_check`
- `run_correctness_check`
- `run_ncu_profile`
- `run_sass_analysis`
- `run_full_benchmark`

`planner` + `coder_optimizer`
- shared inspection tools
- `run_synthetic_check`
- `run_correctness_check`
- `run_ncu_profile`
- `run_sass_analysis`
- `run_full_benchmark`

`optimizer` + `off`
- shared inspection tools
- `run_synthetic_check`
- `run_correctness_check`

`optimizer` + `coder_optimizer`
- shared inspection tools
- `run_synthetic_check`
- `run_correctness_check`
- `codex_optimizer_engineer`

Only `planner` currently receives `run_full_benchmark`.

### What quality_profile changes

`quality_profile=legacy`
- legacy tool caps
- no automatic public-Codex compaction profile

`quality_profile=public_codex`
- larger Codex-like tool caps
- `parallel_tool_calls=False`
- `truncation="auto"`
- `store=False`
- `extra_args={"context_management": [{"type": "compaction", "compact_threshold": 200000}]}`
- `call_model_input_filter=public_codex_input_filter`

Important: `quality_profile` changes runtime behavior and limits, not which
tools a role can access.

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

## Why prompt/tool alignment matters

The repo used to maintain larger static tool lists in prose. That is easy to
drift out of sync with runtime registration.

The current flow avoids that by:

- building the tool list first
- rendering the prompt tool section from the actual tool objects
- injecting Codex-worker guidance only when the worker is actually registered

That keeps instruction text, tool schema, and runtime behavior aligned.
