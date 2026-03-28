# Kernel Coder Memory

## Toolchain Notes

### 2026-03-27 — Modal correctness run failed before upload because local packer dependency was missing

- Error:
  `scripts/bench.py` failed locally during `pack_solution_local(...)` with `No module named 'flashinfer_bench'`.
- Root cause:
  The Modal benchmark job itself runs remotely, but the solution packaging step happens on the client machine first and imports `flashinfer_bench` locally.
- Fix:
  Run the benchmark from the repo's `.venv`, where `modal`, `flashinfer_bench`, and `torch` are installed. The command path should be `.venv/bin/modal ...` moving forward.

### 2026-03-27 — CuTeDSL Modal smoke compiler only receives a single source file

- Error:
  Not a hard failure, but an important compile constraint discovered during the Modal smoke setup.
- Root cause:
  `tools/sass/dump_sass_modal.py --cutedsl ...` writes only the target `kernel.py` into the Modal container before executing it.
- Fix:
  Keep the CuTeDSL smoke path in `solution/dsa_attention/kernel.py` self-contained and avoid repo-local imports for the compile-smoke entrypoint.
