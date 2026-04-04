# Playbook

## Validation

Synthetic:

```bash
.venv/bin/python scripts/bench_synthetic.py --solution-dir solution/dsa_attention --entry-point kernel.py::kernel
```

Full correctness:

```bash
.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point kernel.py::kernel --correctness-only --lang python
```

Benchmark:

```bash
.venv/bin/modal run scripts/bench.py --track dsa_attention --solution-dir solution/dsa_attention --entry-point kernel.py::kernel --lang python
```

## References

- Baseline semantics: `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py`
- CuTeDSL runtime: `references/cutlass/python/CuTeDSL/cutlass/cute`
- CuTeDSL pipelines: `references/cutlass/python/CuTeDSL/cutlass/pipeline`
- CuTeDSL helpers: `references/cutlass/python/CuTeDSL/cutlass/utils`
- Blackwell examples: `references/cutlass/examples/python/CuTeDSL/blackwell`
- Quack kernels: `references/quack`
