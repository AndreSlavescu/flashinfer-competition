# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

word2kernel is a multi-agent system that automatically generates and optimizes CuTeDSL (CUTLASS Python DSL) kernels for Deepseek Sparse Attention on NVIDIA B200 GPUs. It uses the OpenAI Agents SDK to coordinate four specialized agents in an iterative optimization loop.

## Commands

```bash
# Install dependencies (virtualenv assumed at .venv/)
.venv/bin/python -m pip install -r requirements-agent-loop.txt

# Environment setup
cp .env.example .env   # then populate OPENAI_API_KEY

# Run agent loop
.venv/bin/python main.py --num-rounds 5
.venv/bin/python main.py --num-rounds 1              # quick test
.venv/bin/python main.py --num-rounds 5 --resume     # resume from checkpoint

# Run tests
pytest tests/ -v
pytest tests/test_structured_outputs.py -v
pytest tests/test_output_trimming.py -v
pytest tests/test_output_trimming.py::test_read_file_keeps_full_project_files -v  # single test

# Benchmarking (requires Modal + B200)
.venv/bin/modal deploy scripts/bench_synthetic_service.py           # deploy synthetic service (once)
.venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention  # fast synthetic check
.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --lang python  # full benchmark
.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --correctness-only  # correctness only
```

## Architecture

### Agent Loop (`main.py`)

The orchestrator runs this flow:

1. **Round 0a - Designer** (`kernel_designer.py`): Designs kernel architecture from reference implementations. Writes `solution/dsa_attention/kernel_0_plan.md`.
2. **Round 0b - Coder** (`kernel_coder.py`): Implements the design as `kernel_0.py`. Must pass all 23/23 correctness workloads.
3. **Rounds 1..N** (repeating):
   - **Planner** (`kernel_planner.py`): Benchmarks current kernel, profiles with NCU, identifies bottleneck, writes `notes/dsa_attention/strategy_N.md`.
   - **Optimizer** (`kernel_optimizer.py`): Implements the strategy as `kernel_N.py`, validates correctness.

State is persisted to `solution/dsa_attention/loop_state.json` for resumability (`--resume`). Best kernel is tracked and copied to `best_kernel.py`.

### Agent Modules (`kernel_agents/`)

- **`context.py`** - Shared types: `SharedContext` (mutable state threaded through agents), `RoundRecord`, and Pydantic structured output models (`DesignerResult`, `CoderResult`, `PlannerResult`, `OptimizerResult`).
- **`tools.py`** - All agent tools: file I/O, shell execution, validation (`run_synthetic_check`, `run_correctness_check`, `run_full_benchmark`), web access. Enforces output trimming limits (shell: 20k chars, read_file: 40k, search: 20k). Modal-backed tools write full transcripts to `last_shell_dump.txt` and return recent raw shell context. Designer gets read-only `DESIGNER_TOOLS`; coder/planner/optimizer get `ALL_TOOLS`.
- **`stream_logging.py`** - Streaming progress display for agent runs.

### Key Constraints

- **CuTeDSL only**: Kernels must use CuTeDSL exclusively. PyTorch ops (`torch.matmul`, `torch.softmax`, etc.) are forbidden in the kernel compute path (allowed only in prologue/epilogue).
- **B200 target**: Kernels must target sm100a with B200-specific features (TMA, tcgen05, warp specialization, async pipelining).
- **Correctness gating**: No kernel proceeds to optimization until it passes all 23 correctness workloads.

### Output Directories

- `solution/dsa_attention/` - Generated kernels (`kernel_N.py`), best kernel, loop state
- `notes/dsa_attention/` - Strategy documents from planner

### Reference Material (`references/`)

- `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py` - Baseline PyTorch reference kernel
- `cutlass/` (submodule) - CUTLASS library with CuTeDSL APIs and B200 examples
- `quack/` (submodule) - Optimized CuTeDSL kernel examples and design notes

### Benchmarking (`scripts/`)

- `bench.py` - Full Modal-based benchmark and correctness runner
- `bench_synthetic.py` - Fast synthetic correctness launcher (requires deployed service)
- `bench_synthetic_service.py` - Modal service deployment for synthetic checks

## Notes

- `dsa_attention` is the primary track; `dsa_indexer` is a placeholder scaffold only.
- `openai-agents-python/` is an in-repo reference copy of the SDK, not used as an import path.
- Large reference files are auto-windowed to 400 lines on first agent read; specific ranges can be requested.
