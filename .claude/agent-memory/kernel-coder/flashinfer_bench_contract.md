---
name: flashinfer-bench submission contract
description: kernel.py must be self-contained (no local imports), packed via flashinfer_bench, runs on Modal B200 with entry_point="kernel.py::kernel"
type: reference
---

## flashinfer-bench Submission Contract

### File requirements
- `solution/dsa_attention/kernel.py` must be **self-contained** — no imports from other local files
- All files in the solution directory get packed into the Solution JSON via `flashinfer_bench.agents.pack_solution_from_files`
- The entry point is `kernel.py::kernel` (DPS-style: writes output/lse in place)
- Language is `triton` (despite using PyTorch — this is the default)

### What gets packed
- All `.py` files in the solution directory are included in `sources`
- But the kernel.py entry point is loaded standalone on the remote — **do not rely on local imports**

### How to run
```bash
# Correctness only (fast):
.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --correctness-only

# Full benchmark:
.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention

# Must use .venv/bin/modal (not just modal) for local flashinfer_bench dependency
```

### COMPILE_ERROR means
- Import failed (most common: local imports not found)
- Syntax error
- Runtime crash during kernel execution

### Benchmark config
- Correctness: warmup=0, iterations=1, trials=1
- Full: warmup=3, iterations=100, trials=5
- 23 workloads for attention, num_tokens=[1,2,6,7,8]
