# Kernel Coder Context

This document explains exactly how model context is managed for the `kernel-coder` agent when you run:

```bash
python main.py --num-rounds 0 --verbose --coder-max-turns 1000
```

The focus here is **model-visible context**: what is actually sent to the kernel-coder model, how that context grows across turns, and how tool results re-enter the transcript.

It does **not** treat `SharedContext` as model context. In this codebase, `SharedContext` is runtime metadata passed to agent code and tools, not prompt content.

## Short answer

For this command, the kernel-coder gets a **fresh single-run conversation** with:

1. A static system prompt from `kernel_agents/kernel_coder.py`
2. One user message from `main.py` telling it to implement `kernel_0.py` from the designer's plan file
3. The full tool surface in `kernel_agents/tools.py`
4. A strict structured output schema (`CoderResult`)
5. Model settings: coder model, reasoning effort, and verbosity

After that, context grows turn by turn by appending:

1. The model's prior messages
2. Tool calls
3. Tool outputs
4. Reasoning items produced by the SDK/model path

There is **no application-side context trimming**, **no session-based persistence**, and **no automatic inclusion of file contents** like the design plan or generated kernel source. If the coder wants file contents in context, it must read them through tools.

## Execution path for this command

### 1. `main.py` creates shared runtime metadata

`run_loop()` builds a `SharedContext` with:

- `project_root`
- `solution_dir`
- `notes_dir`
- `model_name`
- bookkeeping fields like `current_round`, `history`, and best-latency metadata

This happens before any agents run, but that object is **not automatically sent to the model**.

Relevant code:

- `/home/mark123/projects/word2kernel/main.py`
- `/home/mark123/projects/word2kernel/kernel_agents/context.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_context.py`

The SDK's own docs and source are explicit: run context is available to your code, tools, hooks, etc., but **not passed to the LLM**.

### 2. Round 0 runs the designer first

For a fresh run, `main.py` does:

1. Run `kernel-designer`
2. Run `kernel-coder`
3. Skip the optimizer loop because `--num-rounds 0`
4. Run the planner epilogue benchmark if `kernel_0.py` exists

The designer produces a plan file, but the plan's **contents are not injected into the kernel-coder's initial context**. The coder only receives a path to that file in its user message.

### 3. The kernel-coder receives one explicit user input

`main.py` invokes the coder with this message:

```text
Implement kernel_0.py based on the design plan at {designer_out.plan_file}. Read the plan first, then implement, debug, and validate until ALL 23 workloads pass correctness. Do not submit a PyTorch fallback; if the CuTeDSL compute path is not working, return validation_failed.
```

That is the only explicit user input supplied to the coder at run start.

Relevant code:

- `/home/mark123/projects/word2kernel/main.py`

## What counts as kernel-coder model context

For this run, the model-visible context is built from five pieces.

### 1. Static instructions

The coder agent is created in `make_kernel_coder()` with a large static instruction string:

- role and objective
- required workflow
- constraints
- output contract expectations
- optional `extra_instructions` interpolation

For the command shown here, `extra_instructions` is empty unless passed elsewhere by code changes.

Relevant code:

- `/home/mark123/projects/word2kernel/kernel_agents/kernel_coder.py`

### 2. The initial user message

The coder's first-turn user content is only the message from `main.py`. It includes the plan file path, but not the file contents.

### 3. Tool definitions

The coder gets `ALL_TOOLS` from `kernel_agents/tools.py`. In the current code, that tool surface is:

1. `shell`
2. `apply_patch`
3. `web_search`
4. `web_fetch`
5. `codex_kernel_assist`
6. `read_file`
7. `glob_files`
8. `grep_search`
9. `run_synthetic_check`
10. `run_correctness_check`
11. `run_full_benchmark`

These tools are not just runtime capabilities; their schemas are part of the model call. They shape what the model believes it can do next.

Relevant code:

- `/home/mark123/projects/word2kernel/kernel_agents/tools.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/models/openai_responses.py`

### 4. Structured output schema

The coder has `output_type=CoderResult`. The SDK converts that to a strict JSON schema for final output. The current schema requires:

- `generated`
- `correctness_verified`
- `status`
- `message`
- `reflection`

and constrains `status` to:

- `success`
- `compile_error`
- `validation_failed`

That schema is part of the model request. So the final answer is not free-form prose; it must conform to the `CoderResult` shape.

Relevant code:

- `/home/mark123/projects/word2kernel/kernel_agents/context.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/agent.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/models/openai_responses.py`

### 5. Model settings

The coder agent is created with:

- `model=<coder model>`
- `reasoning=Reasoning(effort=<coder reasoning effort>)`
- `verbosity=<coder verbosity>`

For the command in question, the defaults are:

- model: `gpt-5.4`
- reasoning effort: `high`
- verbosity: `low`

`main.py` also creates a `RunConfig`, but in this path it only adds retry settings. It does **not** add a model input filter, session, conversation id, or previous response id.

Relevant code:

- `/home/mark123/projects/word2kernel/kernel_agents/kernel_coder.py`
- `/home/mark123/projects/word2kernel/main.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/model_settings.py`

### 6. Responses-model request shape

In the current SDK path, this agent is sent through the OpenAI **Responses** model implementation rather than Chat Completions.

For the kernel-coder's first turn, the request is conceptually assembled from:

- `instructions=<kernel-coder system prompt>`
- `model=<coder model>`
- `input=<initial user message>`
- `tools=<serialized tool definitions>`
- `text.format=<strict CoderResult schema>`
- `text.verbosity=<coder verbosity>`
- `reasoning=<coder reasoning settings>`

And notably it does **not** include app-managed conversation state such as:

- `previous_response_id`
- `conversation`
- `session`

Relevant code:

- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/models/openai_provider.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/models/openai_responses.py`

## What does *not* enter model context automatically

This is the important negative space.

### `SharedContext` does not become prompt text

`SharedContext` is available as `ctx.context` inside tools and other runtime code, but it is not automatically serialized into the model transcript.

That includes:

- `project_root`
- `solution_dir`
- `notes_dir`
- `current_round`
- `best_latency_ms`
- `best_round`
- `history`
- `model_name`

This matters because it means the coder does not automatically "know" those values unless:

1. They are hardcoded into instructions
2. They appear in the initial user message
3. A tool result mentions them

### The design plan content is not preloaded

The first user message points to the designer's plan path, but the plan body itself is not inserted into the model call.

So the coder must explicitly fetch it using a tool such as:

- `read_file`
- `shell`
- `codex_kernel_assist`

### Existing code and generated files are not preloaded

Nothing automatically injects:

- reference kernels
- prior notes
- benchmark outputs
- `kernel_0.py`
- validation logs

Those only enter context when tools return them.

## The exact runtime style for `--verbose`

`--verbose` changes the execution path from `Runner.run()` to `Runner.run_streamed()`.

That affects how progress is surfaced to the terminal, but **does not materially change what the model sees**. The streamed path still accumulates turn history in the same conversation run.

Relevant code:

- `/home/mark123/projects/word2kernel/main.py`
- `/home/mark123/projects/word2kernel/kernel_agents/stream_logging.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/run_loop.py`

## How the SDK builds context turn by turn

The OpenAI Agents SDK runs the agent in a loop. On each turn, it prepares the next model input from:

1. The original caller input
2. The generated items from previous turns

In this run, there is no server-managed conversation state configured, so the SDK uses locally accumulated items.

Relevant code:

- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/run_loop.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/items.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/turn_preparation.py`

### First turn

On turn 1, the kernel-coder model call contains:

1. The static kernel-coder instructions
2. The one user message from `main.py`
3. The tool definitions
4. The structured output schema
5. Model settings

There is no prior transcript yet.

### Later turns

After the first response, later model calls include the original user input plus all accumulated generated items for the current run. That includes:

1. Prior assistant messages
2. Tool call items
3. Tool output items
4. Reasoning items that the SDK preserves

So the coder's working context is effectively a growing transcript for the duration of that one `Runner.run_streamed(...)` call.

### Tool results are looped back into the model by default

The agent is not configured to stop after tool execution. It uses the SDK's default tool behavior, which is effectively "run the LLM again" after tools complete.

That means a typical cycle is:

1. model decides to call a tool
2. tool runs
3. tool output is converted into transcript items
4. model gets another turn with that new information in context

There are also no handoffs configured for the kernel-coder, so context stays inside the same agent run rather than being transferred to another agent transcript.

Relevant code:

- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/agent.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/tool_execution.py`

## There is no application-side context trimming

This repository currently does **not** install a model input filter for the coder run.

`make_run_config()` only sets retry behavior. The relevant fields remain unset:

- `call_model_input_filter=None`
- `session_settings=None`
- no `conversation_id`
- no `previous_response_id`

That means the app does not summarize, trim, or rewrite the kernel-coder's accumulated history before the next turn.

The practical consequence is:

- with `--coder-max-turns 1000`, the run can in principle accumulate a very large transcript
- any eventual token-window management would be model/provider behavior, not custom logic in this repo

Relevant code:

- `/home/mark123/projects/word2kernel/main.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/turn_preparation.py`

## There is no persistent conversation state across runs

For this command, the coder run is a fresh conversation.

The app does not configure:

- a reusable session
- a server conversation id
- `previous_response_id`
- `auto_previous_response_id`

So once the coder run finishes, that model transcript is gone unless your application explicitly carries information forward in some other form. In this codebase, it does not carry the coder transcript forward.

The next process invocation of `python main.py ...` starts over with a fresh coder conversation.

## How tool calls feed context back into the model

The tool surface is where most real working context enters the coder's transcript.

### `read_file`

This is the main path for pulling file contents into model context.

Important behavior:

- it can add line numbers
- it can slice by line range
- it trims long content in some cases
- for reference-heavy files, it may return excerpts instead of the whole file

So file contents become model context only when the coder asks for them, and often in a bounded form.

### `glob_files` and `grep_search`

These tools inject directory structure and search results into the transcript. They are compact context-gathering tools rather than full-content loaders.

### `shell`

The local shell tool returns command output back into the transcript. If output is too large, the workflow writes overflow to `last_shell_overflow.txt` and returns a message telling the model where to inspect the overflow.

That means large logs do not automatically flood context. Instead, the model sees:

1. a summarized or truncated tool result
2. a path it can inspect later if needed

The shell environment also advertises a local shell skill:

- `kernel-workbench`

This skill gives the model a repo-specific command playbook for validation and benchmarking workflows.

Relevant code:

- `/home/mark123/projects/word2kernel/kernel_agents/tools.py`
- `/home/mark123/projects/word2kernel/kernel_agents/skills/kernel-workbench/SKILL.md`

### `apply_patch`

`apply_patch` changes files on disk, but the patch content is not itself a substitute for file context. The model usually needs to re-read files if it wants the latest source text reflected in context.

In other words:

- patch action affects the workspace
- file text only affects model context once a read-style tool returns it

### Validation and benchmark tools

These tools are especially important for coder context management because they do not just dump raw subprocess output back into the model. They return curated workflow reports.

The current tools are:

1. `run_synthetic_check`
2. `run_correctness_check`
3. `run_full_benchmark`

These return concise summaries parsed from command output, which is a very important context-control mechanism: the model sees a compact report rather than an uncontrolled terminal log in every turn.

### `web_search` and `web_fetch`

These can bring external information into context. They are available to the coder, but they only affect context if called.

### `codex_kernel_assist`

This tool is a nested Codex-based helper. It is configured with:

- `sandbox_mode="read-only"`
- `working_directory=<repo root>`
- `persist_session=True`
- a default internal model configuration

The key subtlety is:

- the nested Codex tool can persist its **own** session across multiple calls to that tool
- that is **separate** from the kernel-coder's own model transcript

So there are really two context systems here:

1. the kernel-coder conversation managed by the OpenAI Agents SDK
2. the nested Codex helper's private persistent thread

Only the nested tool's returned result flows back into the kernel-coder transcript.

Relevant code:

- `/home/mark123/projects/word2kernel/kernel_agents/tools.py`

## Why `SharedContext` still matters even though it is not model context

It is still operationally important because tools use it to resolve workspace paths and execution defaults.

Examples:

- `project_root` determines where shell commands run
- `solution_dir` can be used as a default location for validation tools
- round metadata can be consumed by code even if the model never sees it

So `SharedContext` shapes tool behavior, and tool behavior shapes what can later enter model context, but `SharedContext` itself is not prompt content.

This distinction is the right one for this codebase:

- `SharedContext` = runtime metadata for code
- model context = prompt, transcript, tool schemas, tool outputs, and output schema

## Command-specific implications for `--num-rounds 0 --verbose --coder-max-turns 1000`

### `--num-rounds 0`

This means:

- designer runs
- coder runs
- optimizer loop does not run

So the coder is a one-shot implementation/debugging conversation for round 0. There is no later optimizer round feeding revised prompts back into the coder.

### `--verbose`

This enables streamed execution and terminal event printing, but not a different context policy.

### `--coder-max-turns 1000`

This only raises the SDK turn cap for the coder run. It does not itself enlarge the model context window; it just allows more iterative turns before the SDK stops the run.

Because there is no app-side trimming, this can produce a long growing conversation transcript.

## Practical mental model

When you think about kernel-coder context in this repo, the most accurate model is:

1. Start with a static system prompt plus a single user task
2. Give the model a toolbox
3. Let it pull files, logs, and reports into context on demand
4. Append all those interactions to one growing per-run transcript
5. End the run with a strict `CoderResult`

That means the coder's real working memory is mostly **self-constructed through tool use**, not preloaded through `SharedContext` or large up-front prompt assembly.

## Most important conclusions

### 1. The kernel-coder does not start with the plan contents

It starts with the plan path and must explicitly read the file.

### 2. `SharedContext` is runtime metadata, not LLM context

This is true both by repo behavior and by the SDK design.

### 3. Tool outputs are the main mechanism for bringing working state into context

File reads, validation summaries, shell output, and searches are what actually populate the coder's transcript.

### 4. There is no repo-level transcript trimming or summarization before model calls

So context grows naturally across turns until the run ends.

### 5. The coder run is ephemeral per process invocation

No session or previous-response chaining is configured for the coder across separate `main.py` runs.

### 6. The nested Codex helper has its own separate persistent session

That persistence belongs to the helper tool, not to the kernel-coder conversation itself.

## Source map

Primary code paths:

- `/home/mark123/projects/word2kernel/main.py`
- `/home/mark123/projects/word2kernel/kernel_agents/kernel_coder.py`
- `/home/mark123/projects/word2kernel/kernel_agents/context.py`
- `/home/mark123/projects/word2kernel/kernel_agents/tools.py`
- `/home/mark123/projects/word2kernel/kernel_agents/stream_logging.py`

SDK internals:

- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_context.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/agent.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/model_settings.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/models/openai_provider.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/models/openai_responses.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/run_loop.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/items.py`
- `/home/mark123/projects/word2kernel/openai-agents-python/src/agents/run_internal/turn_preparation.py`

Repo-specific shell guidance exposed through the shell tool:

- `/home/mark123/projects/word2kernel/kernel_agents/skills/kernel-workbench/SKILL.md`

Official docs:

- https://openai.github.io/openai-agents-python/context/
- https://openai.github.io/openai-agents-python/agents/
- https://openai.github.io/openai-agents-python/tools/
- https://openai.github.io/openai-agents-python/models/
- https://openai.github.io/openai-agents-python/running_agents/
