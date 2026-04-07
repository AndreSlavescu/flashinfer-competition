# AGENTS.md

## Project

`word2kernel` is a multi-agent kernel workbench that designs, implements, validates, and optimizes CuTeDSL kernels for Deepseek Sparse Attention on NVIDIA B200 (`sm100a`) GPUs.

Success means:
- the generated kernel uses CuTeDSL for the attention compute path
- correctness is proven through the repo’s parsed validation tools
- optimization rounds preserve correctness while improving measured latency

## Canonical Commands

- Install deps: `.venv/bin/python -m pip install -r requirements-agent-loop.txt`
- Run one round: `.venv/bin/python main.py --num-rounds 1`
- Resume loop: `.venv/bin/python main.py --num-rounds 5 --resume`
- Deploy synthetic service: `.venv/bin/modal deploy scripts/bench_synthetic_service.py`
- Fast synthetic sweep: `.venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention --entry-point kernel.py::kernel`
- Correctness only: `.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point kernel.py::kernel --correctness-only --lang python`
- Full benchmark: `.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point kernel.py::kernel --lang python`
- NCU profile: `.venv/bin/modal run tools/ncu/ncu_modal.py`
- Tests: `pytest tests/ -v`

## Kernel Rules

- CuTeDSL only for the attention compute path. PyTorch is allowed only for prologue and epilogue duties such as allocation, descriptor construction, compile-cache lookup, stream acquisition, launch, and copying outputs.
- Final kernels must not rely on `torch.matmul`, `torch.bmm`, `torch.einsum`, `torch.softmax`, `torch.logsumexp`, sparse gather-by-advanced-indexing, or equivalent PyTorch tensor math for logits, probabilities, outputs, or LSE.
- Target B200 / `sm100a` features directly when justified: TMA, `tcgen05`, warp specialization, async pipelining, and Blackwell-aware memory movement.
- Never compile CUDA locally. Canonical correctness and performance checks run through the repo’s Modal flows.

## Working Rules

- Keep edits scoped to the task. Do not revert unrelated user changes.
- Prefer repo-native tools for file reads, search, diffs, validation, and profiling.
- Prefer parsed repo validation tools over ad hoc shell parsing when both exist.
- If a tool reports trimmed output, narrow the next request instead of rerunning the same broad read.
- If a shell-like workflow overflows, inspect `last_shell_overflow.txt` before another large shell-like call. The next overflow replaces it.

## File Ownership

- `solution/dsa_attention/`: generated kernels, working kernel, design plan, best kernel, loop state
- `notes/dsa_attention/`: planner strategy documents
- `references/`: read-only external reference material
- `kernel_agents/`: agent prompts, tool wiring, shared context, and orchestration helpers

## Important References

- Baseline semantics: `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py`
- CuTeDSL core helpers: `references/cutlass/python/CuTeDSL/cutlass/cute`
- Pipeline helpers: `references/cutlass/python/CuTeDSL/cutlass/pipeline`
- Utility helpers: `references/cutlass/python/CuTeDSL/cutlass/utils`
- Blackwell examples: `references/cutlass/examples/python/CuTeDSL/blackwell`
- Additional optimized examples: `references/quack`
