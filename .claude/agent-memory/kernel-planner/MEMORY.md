# Kernel Planner Agent Memory

## Project
- [project_v1_design_decisions.md](project_v1_design_decisions.md) — V1 attention: M=64 padded, split-KV 32 chunks, 8 warps, PyTorch combine; indexer: PyTorch reference
- [../../../notes/kernel_v1_as_built.md](../../../notes/kernel_v1_as_built.md) — Canonical as-built baseline for `solution/dsa_attention/kernel_v1.py`: vectorized split-KV PyTorch runtime plus a self-contained CuTeDSL ckv QK experiment

## Planning Notes

### 2026-03-27 — Bench and SASS should refer to the same CuTeDSL target

- Rule:
  For final performance analysis, the benchmark path and the SASS path should target the same CuTeDSL kernel, not a CuTe smoke kernel plus a different runtime path.
- Why:
  Otherwise planner conclusions can be polluted by analyzing one kernel and benchmarking another.

### 2026-03-27 — Bench improvement alone does not prove the intended CuTe kernel is live

- Rule:
  Do not infer "the custom kernel is working" from benchmark gains alone.
- Why:
  Runtime-path changes outside the CuTe kernel can improve results, so planner should verify what code path actually ran before forming architecture conclusions.

### 2026-03-27 — Keep these files as primary references for SM100 CuTeDSL planning

- Rule:
  For staged SMEM / MMA / TMEM / copy-atom structure on SM100, start from these references before proposing detailed layout or pipeline changes:
  - `references/flash-attention/flash_attn/cute/flash_fwd_sm100.py`
  - `references/CuTeDSL-kernels/quack/gemm_sm100.py`
  - `references/flash-attention/flash_attn/cute/copy_utils.py`
  - `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/blackwell_helpers.py`
  - `tools/sass/dump_sass_modal.py`
  - `scripts/bench.py`
- Why:
  These files answered the hardest questions from this run: SM100 staging layout, fragment indexing, TMEM/T2R structure, SASS harness constraints, and Modal packaging/runtime behavior.
