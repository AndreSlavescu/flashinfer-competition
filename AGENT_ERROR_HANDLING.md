# Agent Error Handling: Astra, CudaForge, KernelBench

A comparative analysis of how each project detects, captures, truncates, and feeds back kernel errors through their agent infrastructure.

---

## 1. Astra

**Repo:** `Astra/` — OpenAI Agents SDK, multi-agent CUDA optimization loop.

### Error Detection & Classification

Astra has **no custom exception types**. All errors are caught as generic `Exception` and returned as plain strings to the agent.

| Error Source | File & Lines | Detection Method |
|---|---|---|
| Kernel save | `cuda_kernel_optimizer_multi.py:137-184` | `try/except Exception` around file write |
| Compilation | `cuda_kernel_optimizer_multi.py:1427-1487` | Threaded `torch.utils.cpp_extension.load()` with **60s timeout** |
| Correctness | `test.py:272-293` | `try/except` around `torch.allclose()` check |
| Benchmarking | `test.py:192-253` | `try/except` around timed forward passes |
| Test generation | `cuda_kernel_optimizer_multi.py:236-352` | `json.JSONDecodeError` for malformed LLM specs |

Compilation uses a **threaded timeout pattern** — `torch.utils.cpp_extension.load()` runs in a daemon thread with `thread.join(timeout=60)`. Three distinct failure modes:
- Timeout → `"Error: Compilation timed out after 60 seconds"`
- Build error → `"Error: Compilation failed: {exception}"`
- Missing export → `"Compiled module missing export func '{name}'"`

Correctness testing flags **kernel launch failures** for automatic test removal (`test.py:287-291`):
```python
is_config_error = any(kw in error_msg.lower() for kw in [
    "kernel launch failed", "kernel failed"
])
if is_config_error:
    result.should_remove = True
```

### Shell Output Processing

Astra does **not use subprocess** for compilation. It calls `torch.utils.cpp_extension.load()` directly in-process, so there is no stdout/stderr capture — errors surface as Python exceptions.

**Output truncation is minimal** and only applies in `format.py`:
- Error messages truncated to **120 characters** in benchmark summaries (`format.py:34`)
- Only **5 error samples** shown when all benchmarks fail (`format.py:36`)
- Detailed correctness results (pass/fail/error per test) are **NOT truncated**

### Prompt-Level Error Instructions

The prompts (`prompts.py`) give **minimal explicit error-handling guidance**:

- **Code generator** (`prompts.py:35`): "repeating compilation (with updated code) until it succeeds" — relies on the agent's own judgment to retry.
- **Correctness tester / benchmarker**: No instructions on what to do when verification fails. The prompt just says to call the tool and results are "automatically stored."

There is **no structured error repair prompt** — the agent receives raw error strings and must self-direct fixes.

### Retry / Feedback Loop

- **No code-level retry logic.** Each iteration makes a single attempt per kernel version.
- If code generation or compilation fails, the iteration returns `False` and the loop moves to the next iteration with the same base version (`cuda_kernel_optimizer_multi.py:1050`).
- **Benchmarking proceeds regardless of correctness failure** (documented behavior).
- Fixed iteration count (default 5). No adaptive retry based on failure patterns.

### Agent-to-Agent Error Context

Results flow between agents via the `optimization_state` dict:
1. Correctness results formatted via `format_test_results_for_suggestions()` — shows all failures/errors untruncated, passed tests in scientific notation.
2. Performance comparison formatted via `format_comparison_summary()`.
3. Only a **brief summary message** sent to the orchestrator agent ("Completed correctness verification for vN").
4. The suggestion agent receives full formatted context with both correctness and benchmark data.

---

## 2. CudaForge

**Repo:** `CudaForge/` — Multi-LLM kernel optimization with subprocess isolation and NCU profiling.

### Error Detection & Classification

CudaForge defines **two custom exception types** (`utils/compile_and_run.py:39-48`):

```python
class CompilationError(RuntimeError):
    """First argument is the full build log (Python + ninja/nvcc)."""

class AccuracyError(RuntimeError):
    """Raised when outputs do not meet accuracy tolerance."""
```

The parent process classifies errors into three categories when receiving from the subprocess (`main.py:224-233`):
- `CompilationError` — build/link/import failures
- `AccuracyError` — `torch.allclose()` failures
- Generic `RuntimeError` — everything else (kernel launch, OOM, etc.)

Error metrics are stored per kernel (`main.py:338-346`):
```python
ind.metrics = {
    "runnable": False,
    "phase": phase,
    "error_type": err_type,
    "message": message,
}
```

### Shell Output Processing

CudaForge has the most sophisticated output capture of the three projects:

**Dual-level compilation capture** (`utils/compile_and_run.py:76-112`):
1. **Python-level**: `io.StringIO()` with `contextlib.redirect_stdout/stderr`
2. **OS-level**: `os.dup2()` file descriptor redirection to capture `ninja`/`nvcc` subprocess output

Both are concatenated into the `CompilationError` message:
```python
full_log = "".join([py_buf.getvalue(), subproc_log, str(exc)]).strip()
raise CompilationError(full_log) from None
```

**Two-stage output sanitization** before passing to the LLM:

1. **Pybind tensor stripping** (`main.py:28-33`): Removes large tensor printouts by splitting on `"Invoked with:"`.
2. **Line truncation** (`main.py:84-86`): Keeps only the **last 150 lines** of error output:
   ```python
   def _last_n_lines(text: str, n: int = 150) -> str:
       lines = text.splitlines()
       return "\n".join(lines[-n:]) if len(lines) > n else text
   ```

**Subprocess isolation** (`main.py:275-286`): Kernels are compiled and benchmarked in **spawned subprocesses** (`multiprocessing.get_context("spawn")`) to isolate CUDA context. Results/errors are communicated via `Pipe`.

### Prompt-Level Error Instructions

CudaForge uses **structured, multi-stage error prompts**:

**Stage 1 — Problem identification** (`prompts/judger_repair.py:23-46`): A "correctness auditor" LLM identifies exactly one critical issue as structured JSON:
```json
{
  "critical_issue": "<max 20 words>",
  "why_it_matters": "<max 35 words>",
  "minimal_fix_hint": "<max 20 words>"
}
```

**Stage 2 — Error repair** (`prompts/error.py:20-54`): The repair prompt is constructed with three sections:
1. `$ERROR_LOG` — the sanitized/truncated error output
2. `$OLD_CODE` — the broken kernel code
3. `$Problem` — the structured JSON from stage 1

The prompt gives strict output rules (imports, source, `load_inline` call, `ModelNew` class) and forbids testing code.

**Problem formatting** (`prompts/error.py:63-75`) converts the JSON into a readable string:
```
critical_issue: ...
why_it_matters: ...
minimal_fix_hint: ...
```

### Retry / Feedback Loop

The main loop (`main.py:493-596`) branches on `metrics["runnable"]`:

```
Round 0: SEED (generate initial kernel)
Round 1+:
  if NOT runnable → REPAIR path (error → identify problem → fix)
  if runnable     → OPTIMIZE path (profile with NCU → judge bottleneck → optimize)
```

- **No retry within a single round** — each round produces one kernel.
- **Implicit retry across rounds**: If a kernel fails, the next round enters the REPAIR path with the error context.
- With default `--round 10`, there are up to **9 repair attempts** if all generations fail.
- **Best kernel tracking**: The best successful kernel is saved to `test_kernel_{id}.py`. Error rounds are plotted with 'x' markers on score curves.

### Error Context Flow

```
Error → _sanitize_error_message() → _last_n_lines(150)
  → correctness auditor LLM → critical_issue JSON
  → build_error_prompt(old_code + error_log + problem)
  → repair LLM → new kernel → _bench_and_score()
```

The error message, old code, and diagnosed problem are all packaged into a single repair prompt. The LLM sees both what went wrong (error log) and why (critical issue analysis).

---

## 3. KernelBench

**Repo:** `KernelBench/` — Benchmark framework for evaluating LLM-generated CUDA kernels. Single-shot generation, no iterative agent loop.

### Error Detection & Classification

KernelBench has the most granular error classification, tracking errors through a rich metadata dict:

| Error Type | Detection | Metadata Fields |
|---|---|---|
| Compilation | `try/except` around `load_custom_model()` | `compilation_error`, `compilation_error_name` |
| Lock file | String match on `"lock"` or `"No such file"` | Returns `None` (retry signal) |
| Runtime | `RuntimeError` during model instantiation | `runtime_error`, `runtime_error_name` |
| Shape mismatch | `output.shape != ref.shape` comparison | `correctness_issue` |
| Value mismatch | `torch.allclose()` with precision-aware tolerance | `max_difference[]`, `avg_difference[]` |
| Runtime exception | `try/except` during forward pass | `runtime_error_traceback` |
| Excessive speedup | `speedup > 10x` threshold | `excessive_speedup: True` |

**Error name extraction** (`eval.py:35-39`): Uses `get_error_name(e)` to extract the exception class name for categorization.

**Error message truncation** (`eval.py:702-724`): `register_and_format_exception()` truncates long error messages to **200 characters** for metadata storage.

### Shell Output Processing

**Two capture approaches**:

1. **In-process** (`eval.py:310-315`): `StringIO` + `redirect_stdout/redirect_stderr` for `torch.utils.cpp_extension.load()`.
2. **Subprocess** (`eval.py:330-367` / `compile.py`): `subprocess.Popen` with `stdout=PIPE, stderr=PIPE` for isolated compilation, returning `(success, stdout, stderr)` tuples.

**Metadata serialization** (`eval.py:851-906`): Recursive JSON serialization ensures all metadata (including exception objects) is serializable — non-serializable objects are converted to strings.

### Prompt-Level Error Instructions

KernelBench prompts are **minimal** (`prompts/prompts.toml:16-18`):
```
Optimize the architecture named Model with custom {backend}! 
Name your optimized output architecture ModelNew. ...
Please generate real code, NOT pseudocode, make sure the code compiles 
and is fully functional. Just output the new model code, no other text, 
and NO testing code!
```

Error prevention is primarily handled through **static analysis** rather than prompt instructions.

### Static Checking (Pre-Runtime Validation)

`kernel_static_checker.py` performs regex-based validation to catch known error patterns BEFORE runtime:

| Check | What it Detects | Lines |
|---|---|---|
| Bypass hacks | Try-except fallback, `pass` inheritance | 38-73 |
| PyTorch wrapping | High-level ops like `nn.Linear`, `nn.Conv2d` | 84-100 |
| Stream injection | Non-default CUDA streams (timing manipulation) | 300-334 |
| Thread injection | `threading`, `multiprocessing`, `concurrent.futures` | 337-378 |
| Lazy evaluation | `_make_subclass`, custom tensor subclasses | 381-414 |
| Timing monkey-patch | Overwriting `torch.cuda.Event`, `synchronize` | 417-449 |
| Precision downgrade | FP32→FP16 casts (`__float2half`, `tl.astype`) | 452-499 |

### Retry / Feedback Loop

KernelBench is fundamentally a **single-shot evaluation framework** — there is no iterative repair loop:

1. LLM generates kernel code
2. Static check (optional, `run_and_check.py:286-294`)
3. Compile and evaluate
4. Log results

**Compilation retry**: Timeout-based retry in batch compilation (`compile.py:130-146`) — if a task times out, the cache is cleared and **one retry** is attempted.

**Multiple correctness trials**: Runs kernel against `num_correct_trials` different random inputs. **All trials must pass** for correctness success (no partial credit).

**No error feedback to LLM** — error context is recorded in metadata but not passed back for iterative improvement.

---

## Cross-Project Comparison

| Aspect | Astra | CudaForge | KernelBench |
|---|---|---|---|
| **Custom exception types** | None | `CompilationError`, `AccuracyError` | None (uses metadata dict) |
| **Compilation capture** | In-process (`torch.cpp_extension.load`) | Dual-level (Python + OS FD redirect) | In-process + subprocess |
| **Output truncation** | 120 chars + 5 samples in summaries | Strip pybind tensors, last 150 lines | 200 chars for metadata |
| **Subprocess isolation** | No | Yes (spawned process + Pipe) | Yes (for batch compile) |
| **Error → LLM prompt** | Raw error strings, no structure | Structured: error_log + critical_issue JSON + old_code | Not fed back (single-shot) |
| **Problem diagnosis** | Agent self-directs | Separate "correctness auditor" LLM call | Static regex checker |
| **Retry logic** | None (skip to next iteration) | Implicit across rounds (REPAIR path) | Timeout-based, one retry |
| **Max repair attempts** | 0 per version (fail-forward) | Up to `--round - 1` across loop | 1 (timeout retry only) |
| **Correctness gating** | Benchmarking proceeds regardless | Must be runnable to enter OPTIMIZE path | All trials must pass |
| **Error classification** | Untyped strings | 3 categories via exception type | Granular metadata fields |
| **Adversarial detection** | Test removal for kernel launch failures | None | Excessive speedup flag + static checks |

## Key Patterns

### Error Capture
- **Astra** and **KernelBench** use `torch.utils.cpp_extension.load()` in-process, catching Python exceptions.
- **CudaForge** captures both Python-level and OS-level (file descriptor redirect) output, yielding the richest error logs for the LLM.

### Output Truncation for Agent Context
- **Astra**: Minimal truncation (120 chars in formatted summaries only). Full errors stored internally.
- **CudaForge**: Two-stage pipeline — strip pybind tensor dumps, then keep last 150 lines. This is the most deliberate approach to preventing context window overflow.
- **KernelBench**: 200-char truncation in metadata. Not designed for agent consumption.

### Prompt Design for Error Recovery
- **CudaForge** is the only project with a dedicated error repair prompt flow: diagnose (auditor LLM → JSON) then fix (repair LLM with error + diagnosis + code). This two-stage approach separates problem identification from solution generation.
- **Astra** relies on the agent's own reasoning — the prompt says "repeat until it succeeds" without structured guidance.
- **KernelBench** doesn't feed errors back at all.

### Iterative Repair
- **CudaForge**: Explicit REPAIR vs OPTIMIZE branching based on `runnable` flag. Each failed round triggers a targeted repair cycle.
- **Astra**: Fail-forward — failed versions are skipped, next iteration starts from the last good version.
- **KernelBench**: No iteration. Evaluation only.
