---
name: CuTeDSL sQrow column-major vs row-major bug
description: sQrow staging buffer must be row-major (stride N_BLOCK,1) not default column-major (stride 1,N_BLOCK) to match gather4 write order and _make_rowmajor_mn_view strides
type: reference
---

## Bug: sQrow default column-major layout transposes data

### Root cause

`smem.allocate_tensor(layout=cute.make_layout((M_BLOCK, N_BLOCK, 1)))` creates column-major strides `(1, M_BLOCK)`. But:

1. **gather4** writes data at address `row * N_BLOCK + col` (row-major in SMEM)
2. **`_make_rowmajor_mn_view`** computes strides `(b*d, a*b*d, 1, b)` which assume `src[M,K]` at address `M*K_total + K` (row-major)

With column-major sQrow, `sQrow[m, n]` is at address `m + n*M_BLOCK`, but gather4 puts data at `m*N_BLOCK + n`. This causes a transposition: the M and K (or N and K) dimensions are swapped.

### Fix

Allocate sQrow with explicit row-major strides:

```python
sQrow = smem.allocate_tensor(
    element_type=BFloat16,
    layout=cute.make_layout(
        (M_BLOCK, N_BLOCK, 1),
        stride=(N_BLOCK, 1, M_BLOCK * N_BLOCK),
    ),
    byte_alignment=16,
)
```

### Impact

- **B operand (K staging)**: With column-major sQrow, the copy puts K^T into sK instead of K. GEMM computes Q@K instead of Q@K^T. Scores are wrong.
- **A operand (P staging)**: With column-major sQrow, the copy transposes softmax output. PV GEMM input is wrong.
- **A operand (Q staging)**: Bypassed by TMA, not affected.

### Status (2026-03-30)

Fix applied but fused kernel still has remaining numerical errors (~30-60 LSE difference). The row-major fix improved some heads but others still show large discrepancies. Further debugging needed on the `_make_swizzled_mn_view` / copy function correctness.
