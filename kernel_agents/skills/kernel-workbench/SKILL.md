# Kernel Workbench

Use this local shell skill for the canonical repo workflows in `word2kernel`.

Start from the repo root. Prefer these commands:

- Bootstrap local Python deps: `.venv/bin/python -m pip install -r requirements-agent-loop.txt`
- Run the loop: `.venv/bin/python main.py --num-rounds 1`
- Deploy the fast synthetic service: `.venv/bin/modal deploy scripts/bench_synthetic_service.py`
- Run the fast synthetic sweep: `.venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention --entry-point kernel.py::kernel`
- Run correctness only: `.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point kernel.py::kernel --correctness-only --lang python`
- Run the full benchmark: `.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point kernel.py::kernel --lang python`
- Run NCU profiling: `.venv/bin/modal run tools/ncu/ncu_modal.py`

If a shell-like tool overflows, it writes the full transcript to `last_shell_overflow.txt` at repo root and replaces that file on the next overflowing shell-like call. Inspect it with `read_file` or `grep_search` before launching another large shell command.

Reference paths that exist in this trimmed repo:

- `references/cutlass/python/CuTeDSL/cutlass/cute`
- `references/cutlass/python/CuTeDSL/cutlass/pipeline`
- `references/cutlass/python/CuTeDSL/cutlass/utils`
- `references/cutlass/examples/python/CuTeDSL/blackwell`
- `references/cutlass/examples/python/CuTeDSL/notebooks`
- `references/quack`
