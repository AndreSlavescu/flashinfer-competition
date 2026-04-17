# Modal Tools Output Flow

How agent tools that invoke Modal process their outputs before returning
results to the LLM context window.

---

## Overview

Five agent tools execute work on remote Modal B200 GPUs. Each follows a
common pattern: **pack a local payload, invoke a Modal function, receive a
structured result dict, then summarize and trim the output** before it
enters the agent's context window.

```
 Agent Tool (tools.py)          Modal Remote (scripts/ or tools/)
 ========================       ==================================

 run_synthetic_check ---------> bench_synthetic_service.py  (deployed service)
 run_stage_validation --------> bench_synthetic_service.py  (deployed service)
 run_correctness_check -------> bench.py::run_benchmark_remote  (modal run)
 run_full_benchmark ----------> bench.py::run_benchmark_remote  (modal run)
 run_ncu_profile -------------> tools/ncu/ncu_modal.py          (modal run)
 run_sass_analysis -----------> tools/sass/dump_sass_modal.py   (modal run)
```

---

## 1. Synthetic Check / Stage Validation

The fastest feedback loop. Uses a **pre-deployed** Modal service so
there is no cold-start penalty.

```
 LOCAL (agent machine)                          MODAL B200 (deployed service)
 =====================                          ============================

 run_synthetic_check()                          run_dsa_synthetic_fast()
        |                                              |
        |  1. Resolve solution_dir                     |
        |  2. Build command:                           |
        |     .venv/bin/python                         |
        |       scripts/bench_synthetic.py             |
        |       --solution-dir ...                     |
        |       --entry-point kernel.py::kernel        |
        |                                              |
        v                                              |
 +------------------+                                  |
 | _run_command()   |  <-- subprocess with timeout     |
 +------------------+                                  |
        |                                              |
        | bench_synthetic.py (local launcher):         |
        |   a. Resolve solution dir                    |
        |   b. Collect dependency closure              |
        |      (AST-walk imports -> base64 encode)     |
        |   c. fn = modal.Function.from_name(...)      |
        |   d. result = fn.remote(                     |
        |        raw_files={...},        ------------> |
        |        entry_point="...",                     |
        |        rebuild_fixture=False)                 |
        |                                              v
        |                                 +---------------------------+
        |                                 | _ensure_fixture_bundle()  |
        |                                 |  - Check memory cache     |
        |                                 |  - Check volume cache     |
        |                                 |  - Build if needed:       |
        |                                 |    generate inputs +      |
        |                                 |    reference outputs      |
        |                                 |    with seeded RNG        |
        |                                 +---------------------------+
        |                                              |
        |                                 +---------------------------+
        |                                 | _ensure_solution_module() |
        |                                 |  - Hash payload           |
        |                                 |  - Check memory cache     |
        |                                 |  - Write files to /tmp    |
        |                                 |  - importlib load         |
        |                                 +---------------------------+
        |                                              |
        |                                 +---------------------------+
        |                                 | _run_synthetic_sweep()    |
        |                                 |  For each num_tokens in   |
        |                                 |  (1, 2, 6, 7, 8):        |
        |                                 |    - Run solution kernel  |
        |                                 |    - Compare vs reference |
        |                                 |    - Record PASSED/FAILED |
        |                                 |      + abs errors         |
        |                                 +---------------------------+
        |                                              |
        |                     <------------------------+
        |                     dict:
        |                       success: True
        |                       results: [{status, num_tokens, ...}]
        |                       fixture_version, fixture_source
        |                       solution_source, payload_hash
        |                       elapsed_s
        |
        |  bench_synthetic.py prints structured text:
        |    "SYNTHETIC RESULTS: 5/5 cases passed"
        |    + per-case table
        |    + failure details (if any)
        |
        v
 stdout captured by _run_command()
```

### Output Processing Pipeline

```
 Raw stdout + stderr from subprocess
        |
        v
 +----------------------------------+
 | _run_command()                    |
 |  - Capture stdout/stderr bytes   |
 |  - Decode UTF-8                  |
 |  - Check for timeout             |
 |  - Build ShellCommandOutput      |
 +----------------------------------+
        |
        v
 +----------------------------------+
 | _write_shell_dump_for_outputs()  |
 |  - Build full transcript         |
 |  - Write to shell dump file      |    last_shell_dump.txt
 +----------------------------------+
        |
        v
 +--------------------------------------+
 | _format_modal_shell_output()        |
 |  - Render Command/Exit/STDOUT/      |
 |    STDERR transcript block          |
 |  - Keep latest 200 lines            |
 |  - Apply char cap after tailing     |
 |  - Prepend dump guidance if trimmed |
 +--------------------------------------+
```

**What the agent sees** (success case):

```text
retrieved trimmed shell output to the last 200 lines; full transcript saved to last_shell_dump.txt.
Use read_file or grep_search on last_shell_dump.txt if you need more detail.
Command: .venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention --entry-point kernel.py::kernel
Exit code: 0
STDOUT:
...
SYNTHETIC RESULTS: 5/5 cases passed
```

**What the agent sees** (failure + trimmed dump context):

```text
retrieved trimmed shell output to the last 200 lines; full transcript saved to
last_shell_dump.txt ...
Command: .venv/bin/python scripts/bench_synthetic.py ...
Exit code: 1
STDOUT:
...
FAILED synthetic cases (
...
```

**What the agent sees** (timeout / deadlock):

```
Command: .venv/bin/python scripts/bench_synthetic.py ...
Status: timed out after 120s
WARNING: A timeout at this length is MOST LIKELY A DEADLOCK in the kernel
(e.g. producer/consumer pipeline stall, unmet mbarrier arrival count,
missing async fence/commit, warp specialization hang). Investigate the
synchronization logic before retrying.
```

---

## 2. Correctness Check / Full Benchmark

Uses `modal run` (ephemeral container) rather than a deployed service.
The kernel code is packed and sent to Modal, where `flashinfer-bench`
runs the official evaluation.

```
 LOCAL                                         MODAL B200
 =====                                         =========

 run_correctness_check()                       run_benchmark_remote()
 run_full_benchmark()                          (same remote fn, different config)
        |                                              |
        | Build command:                               |
        |   .venv/bin/modal run scripts/bench.py       |
        |   --track dsa_attention                      |
        |   --solution-dir ...                         |
        |   --entry-point kernel.py::kernel            |
        |   [--correctness-only]  <-- only for         |
        |                         correctness check    |
        v                                              |
 +------------------+                                  |
 | _run_command()   |                                  |
 +------------------+                                  |
        |                                              |
        | bench.py local_entrypoint():                 |
        |   a. Collect solution files                  |
        |      (rglob *.py -> base64)                  |
        |   b. Build pack_args dict                    |
        |   c. run_benchmark_remote.remote(            |
        |        raw_files={...},       -------------> |
        |        pack_args={...},                      |
        |        correctness_only=...)                  |
        |                                              v
        |                                 +---------------------------+
        |                                 | Remote packing:           |
        |                                 |  - Write files to tmpdir  |
        |                                 |  - Build flashinfer-bench |
        |                                 |    Solution object        |
        |                                 +---------------------------+
        |                                              |
        |                                 +---------------------------+
        |                                 | flashinfer-bench:         |
        |                                 |  - Load TraceSet from     |
        |                                 |    Modal volume           |
        |                                 |  - Configure BenchmarkCfg |
        |                                 |    correctness:           |
        |                                 |      warmup=0, iter=1     |
        |                                 |    full:                  |
        |                                 |      warmup=3, iter=100   |
        |                                 |  - benchmark.run_all()    |
        |                                 |  - Collect per-workload:  |
        |                                 |    status, latency_ms,    |
        |                                 |    ref_latency_ms,        |
        |                                 |    speedup, abs/rel err   |
        |                                 +---------------------------+
        |                                              |
        |                     <------------------------+
        |                     dict:
        |                       success: True
        |                       results: [{status, latency_ms,
        |                                  speedup, ...}, ...]
        |
        |  bench.py prints structured text:
        |    "RESULTS: 23/23 workloads passed"
        |    "Average speedup: 1.42x"
        |    + per-workload table
        |    + failure details
        |
        v
 stdout captured by _run_command()
```

### Output Processing Pipeline

```
 Raw stdout + stderr
        |
        v
 +----------------------------------+
 | _run_command()                    |
 |  (same as synthetic, 120s or     |
 |   300s timeout)                  |
 +----------------------------------+
        |
        v
 +----------------------------------+
 | _write_shell_dump_for_outputs()  |
 |  (always writes for Modal tools) |
 +----------------------------------+
        |
        v
 +----------------------------------------+
 | _format_modal_shell_output()          |
 |  - Build raw command/stdout/stderr    |
 |    transcript block                   |
 |  - Keep latest 200 lines in context   |
 |  - Add dump-search notice if trimmed  |
 |  - Apply char cap after tailing       |
 +----------------------------------------+
        |
        v
 +--------------------------------------+
 | _prepend_notice() + return to agent  |
 +--------------------------------------+
```

**What the agent sees** (full benchmark, success):

```
retrieved trimmed shell output to the last 200 lines; full transcript saved to last_shell_dump.txt.
Use read_file or grep_search on last_shell_dump.txt if you need more detail.
Command: .venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point kernel.py::kernel
Exit code: 0
STDOUT:
...
  RESULTS: 23/23 workloads passed
...
```

---

## 3. NCU Profiling

Runs Nsight Compute on Modal to collect GPU hardware counters.

```
 LOCAL                                         MODAL B200
 =====                                         =========

 run_ncu_profile()
        |
        | Build command:
        |   .venv/bin/modal run
        |     tools/ncu/ncu_modal.py
        |
        v
 +------------------+
 | _run_command()   |
 |  timeout: 1800s  |
 +------------------+
        |
        | ncu_modal.py local_entrypoint():
        |   profile_kernel_on_modal.remote()
        |                            ----------------> |
        |                                              v
        |                                 +---------------------------+
        |                                 | 1. Find ncu binary        |
        |                                 | 2. nvidia-smi GPU info    |
        |                                 | 3. Run kernel (correctness|
        |                                 |    check first)           |
        |                                 | 4. Run with NCU:          |
        |                                 |    ncu --set full --csv   |
        |                                 |    -k <pattern>           |
        |                                 |    -c <num_captures>      |
        |                                 | 5. Parse CSV metrics      |
        |                                 +---------------------------+
        |                                              |
        |                     <------------------------+
        |                     dict: success, csv,
        |                       ncu_version,
        |                       normal_run_stdout
        |
        |  ncu_modal.py prints to stdout
        |
        v
 stdout captured by _run_command()
```

### Output Processing Pipeline

```
 Raw stdout + stderr
        |
        v
 +------------------------------------+
 | _run_command()                      |
 |  (1800s timeout, shell dump check)  |
 +------------------------------------+
        |
        v
 +------------------------------------+
 | Direct formatting (no summarizer): |
 |  lines = [                         |
 |    "Command: ...",                  |
 |    "Exit code: N" / "timed out",   |
 |    stdout or stderr or "(no output)"|
 |  ]                                 |
 |  _truncate_output(                 |
 |    report,                         |
 |    limit=shell_output_limit_chars  |   <-- 20k default
 |  )                                 |
 +------------------------------------+
        |
        v
 +--------------------------------------+
 | _prepend_notice() + return to agent  |
 +--------------------------------------+
```

NCU output is **not regex-summarized** like benchmarks. The agent
receives the raw printed output (GPU info, NCU metrics, CSV data)
truncated to the shell output limit.

---

## 4. SASS Analysis

Disassembles kernel SASS instructions for pipeline analysis.

```
 LOCAL                                         MODAL B200
 =====                                         =========

 run_sass_analysis()
        |
        | Resolve kernel_file
        |  (default: solution/.../kernel.py)
        |
        | Build command:
        |   .venv/bin/modal run
        |     tools/sass/dump_sass_modal.py
        |     --cutedsl <kernel_file>
        |
        v
 +------------------+
 | _run_command()   |
 |  timeout: 1200s  |
 +------------------+
        |
        | dump_sass_modal.py local_entrypoint():
        |   Reads kernel source locally, then
        |   analyze_cutedsl_kernel.remote(source)
        |                            ----------------> |
        |                                              v
        |                                 +---------------------------+
        |                                 | 1. Write kernel to /root  |
        |                                 | 2. CuTeDSL JIT compile:   |
        |                                 |    Python -> IR -> MLIR   |
        |                                 |    -> PTX -> cubin        |
        |                                 | 3. Find .cubin artifact   |
        |                                 |    (or compile .ptx)      |
        |                                 | 4. nvdisasm -g            |
        |                                 | 5. Extract opcodes:       |
        |                                 |    regex on disassembly   |
        |                                 | 6. classify_opcodes()     |
        |                                 | 7. estimate_pipeline_cost |
        |                                 | 8. format_analysis()      |
        |                                 +---------------------------+
        |                                              |
        |                     <------------------------+
        |                     dict: success,
        |                       analysis_report,
        |                       opcode_counts,
        |                       classification,
        |                       pipeline_costs,
        |                       disassembly (partial),
        |                       ptx, kernels, ...
        |
        |  dump_sass_modal.py prints:
        |    PTX (first 2000 chars)
        |    SASS Pipeline Analysis report
        |    Disassembly (first 5000 chars)
        |
        v
 stdout captured by _run_command()
```

### Output Processing Pipeline

Same as NCU -- raw shell context formatting, no regex summarizer:

```
 Raw stdout + stderr
        |
        v
 +------------------------------------+
 | _run_command()                      |
 |  (1200s timeout, shell dump check)  |
 +------------------------------------+
        |
        v
 +------------------------------------+
 | Direct formatting:                 |
 |  "Command: ..."                    |
 |  "Exit code: N"                    |
 |  stdout or stderr                  |
 |  _truncate_output(limit=20k)       |
 +------------------------------------+
        |
        v
 +--------------------------------------+
 | _prepend_notice() + return to agent  |
 +--------------------------------------+
```

---

## Common Infrastructure

### `_run_command()` -- The Shared Subprocess Runner

All Modal tools go through `_run_command()`, which runs the `modal run`
or `python` subprocess locally and captures output.

```
 _run_command(command, timeout_s, source_tool, limits)
        |
        v
 asyncio.create_subprocess_exec(
   *command,
   cwd=project_root,
   stdout=PIPE, stderr=PIPE
 )
        |
        +-- asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        |       |
        |       +-- Success: decode stdout/stderr as UTF-8
        |       |
        |       +-- TimeoutError:
        |               proc.kill()
        |               timed_out = True
        |
        v
 ShellCommandOutput(command, stdout, stderr, outcome)
        |
        v
_write_shell_dump_for_outputs()
   |
   +-- Build transcript: source_tool + cwd + command + stdout + stderr
   +-- Write to last_shell_dump.txt
   |
   +-- Otherwise: return None
        |
        v
 CommandRunResult(returncode, stdout, stderr, timed_out, output, dump_notice)
```

### Shell Dump Mechanism

When raw output exceeds the configured limit, the full transcript is
saved to a file the agent can read later with targeted queries.

```
 +------------------------------------------+
 |          Agent Context Window             |
 |                                           |
 |  "retrieved trimmed shell output to the   |
 |   last 200 lines; full transcript saved   |
 |   to last_shell_dump.txt ..."             |
 |                                           |
 |  Command: .venv/bin/modal run ...         |
 |  Exit code: 0                             |
 |  STDOUT:                                  |
 |  ...                                      |
 +------------------------------------------+
                    |
                    | Agent can use read_file or
                    | grep_search to inspect
                    v
 +------------------------------------------+
 |      last_shell_dump.txt                  |
 |                                           |
 |  Source tool: run_full_benchmark          |
 |  Working directory: /path/to/project      |
 |  [Command 1]                              |
 |  Command: .venv/bin/modal run ...         |
 |  Exit code: 0                             |
 |  STDOUT:                                  |
 |  (full untruncated output)                |
 |  STDERR:                                  |
 |  (full untruncated stderr)                |
 +------------------------------------------+
```

### Unified Modal Output Formatting

All six Modal-backed workflow tools now use the same output path.

```text
_run_command(command, timeout_s, source_tool)
    -> captures stdout/stderr into ShellCommandOutput
    -> writes full transcript to last_shell_dump.txt
    -> returns ShellCommandOutput

_format_modal_shell_output(output, limits)
    -> renders Command / Exit code / STDOUT / STDERR block
    -> keeps latest 200 lines
    -> applies char cap after tailing
    -> prepends dump guidance when trimming occurs
```

### Timeout Configuration

```
 Tool                    Subprocess    Modal Function
                         Timeout       Timeout
 =====================   ===========   ==============
 run_synthetic_check       120s          3600s
 run_stage_validation      120s          3600s
 run_correctness_check     120s          3600s
 run_full_benchmark        300s          3600s
 run_ncu_profile          1800s          1800s
 run_sass_analysis        1200s          1200s
```

The subprocess timeout is the local `_run_command()` watchdog. If this
fires, it kills the local `modal run` / `python` process. The Modal
function timeout is the remote container watchdog and is set
independently in the `@app.function()` decorator.

### Tool Output Limits (Legacy Profile)

```
 Limit                         Value     Affects
 ============================  ========  ============================
 shell_output_limit_chars      20,000    Returned shell context cap after
                                         Modal tailing / synthetic trimming
 read_file_limit_chars         40,000    Agent reading shell dump file
 search_output_limit_chars     20,000    Agent grepping shell dump file
```

### End-to-End Data Flow (Complete Picture)

```
 +-----------+     base64 files     +----------------+     run kernel      +----------+
 |  Agent    | ------------------> | Modal Function  | -----------------> | B200 GPU |
 |  (tools.  |     via subprocess  | (bench.py /     |     torch.cuda     |          |
 |   py)     |     + modal SDK     |  service.py /   |                    |          |
 |           |                     |  ncu_modal.py / |                    |          |
 |           |                     |  dump_sass.py)  | <----------------- |          |
 |           | <------------------ |                 |     results dict   +----------+
 |           |     structured      +----------------+
 |           |     stdout text
 |           |
 |           |     Dump + Tail
 |           |     ===========
 |           |     1. _run_command() captures raw stdout/stderr
 |           |     2. Write shell dump artifact on every call
 |           |     3. _format_modal_shell_output() keeps recent context
 |           |     4. _truncate_output() caps returned shell context if needed
 |           |     5. _prepend_notice() adds dump guidance when needed
 |           |
 |           |     Result: recent raw shell context for LLM context
 +-----------+
```
