---
name: include migration checklist
description: When recommending file moves or consolidations, always audit ALL consumers of the target file — .cu, .cuh, and run_modal.py — before proposing the change
type: feedback
---

When proposing any refactoring that moves or renames a header file (e.g., extracting `common.cuh` to `microbenchmarks/common/benchmark_common.cuh`), the refactor report must include a complete consumer audit as part of the recommendation — not just the obvious entry points.

**Why:** During the `common.cuh` → `benchmark_common.cuh` extraction (2026-03-17), only `main.cu` includes were updated initially. Several `.cuh` files (`gather4_kernels.cuh`, `index_patterns.cuh`, `tensor_map_utils.cuh`, `dual_tma_kernels.cuh`) also included `common.cuh` and were missed, causing compile failures. The `run_modal.py` files also hardcode local file paths via `.add_local_file(...)` and needed updating too — this was also missed initially.

**How to apply:** For every file-move or header-consolidation recommendation, include two extra steps in the report's "Proposed Next Steps":

1. **Include audit**: Run `grep -r '#include "common.cuh"'` (or the relevant filename) across all `*.cu` and `*.cuh` files in `microbenchmarks/` — not just `main.cu`. List every file that needs updating.
2. **Modal audit**: Search all `run_modal.py` files for `.add_local_file(...)` calls that reference the old path. These are hardcoded and will cause `FileNotFoundError` at Modal startup if not updated alongside the include changes.

Both steps should appear as explicit checklist items in the report before any "verify compilation" step.
