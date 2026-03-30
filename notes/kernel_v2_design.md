# Kernel v2 Design

## Hypothesis

Replace the hybrid `CuTe QK + PyTorch everything else` path with a single CuTeDSL kernel that performs the full per-split attention body:

- sparse KV gather
- QK (ckv + kpe)
- mask + softmax
- SV
- partial output/LSE writeback

Keep only the cross-split combine outside the kernel for now.

This should remove the dominant fixed overhead in the current v3 path. The goal of v2 is not to match FlashAttention 4 or FlashMLA immediately; the goal is to land the first correct, end-to-end fused CuTe baseline on the real runtime path.

## Why The Previous v2 Design Needed Correction

The earlier `kernel_v2_design.md` was directionally right about fusion, but it bundled too many advanced choices into the first fused iteration:

- it assumed a FlashMLA-style TMA-first kernel from day one
- it put both Q and P in TMEM, which makes the first implementation much harder
- it treated microbenchmark latencies as if they would compose directly into end-to-end kernel time
- it predicted implausibly large speedups for a first CuTe baseline

That combination made the design much closer to a later optimized kernel than to a practical v2 baseline.

## Evidence From Current Results

From the current CuTe QK-only path in [kernel_v3_cutedsl.py](/home/mark123/projects/FlashMLA/solution/dsa_attention/kernel_v3_cutedsl.py):

- the benchmarked runtime still spends most of its work outside CuTe
- the CuTe path only handles QK, while gather, softmax, SV, and combine stay in PyTorch
- the runtime pads, expands, permutes, launches CuTe, synchronizes, then returns to PyTorch
- correctness-only results are nearly flat at about `123-137 ms` across workloads, which is a strong sign of large fixed runtime overhead rather than useful scaling

So the next step should not be "a better QK kernel." It should be "move the whole split attention body into one CuTe kernel."

## Scope Of v2

### What v2 must fuse

For one `(token, split)` pair:

1. gather 64 sparse KV rows
2. compute `Q_nope @ Kc^T`
3. compute `Q_pe @ Kp^T`
4. accumulate the two score tensors
5. apply `sm_scale`
6. apply the split-local mask
7. run softmax
8. compute `P @ V`
9. write `partial_o[split, token, head, dim]`
10. write `partial_lse[split, token, head]` in log2 base

### What v2 explicitly does not fuse

- the final reduction across the 32 splits
- persistent scheduling
- TMA gather4
- split-P arrive
- correction warps
- multi-CTA MMA
- TMEM-stored Q or TMEM-stored P

Those are valid later optimizations, but they should not be prerequisites for the first fused baseline.

## Kernel Shape

- Grid: `(num_tokens, 32, 1)`
- One CTA per `(token, split)`
- Block size: `256` threads / `8` warps
- `M = 64`
  - only the first 16 rows are real query heads
  - rows 16..63 are zero-padded
- `N = 64`
  - one sparse chunk per CTA

This keeps the same split-KV decomposition already used by the current implementation and leaves the outer combine step unchanged.

## Core Design Decision

The first fused CuTe baseline should use:

- **SMEM for Q**
- **SMEM for gathered K/V**
- **TMEM only for accumulators**
- **SMEM for softmax output P**

This is the key simplification.

It avoids the hardest parts of the earlier design:

- no UTCCP requirement for Q
- no TMEM rewrite of P
- no need to reinterpret one TMEM region as both FP32 scores and packed BF16 probabilities
- no TMA descriptor choreography for the first iteration

The baseline still uses CuTeDSL and tcgen05 MMA for the heavy math, but it does not force the whole FlashMLA memory hierarchy into v2.

## Memory Plan

### Shared memory

Allocate one CTA-local workspace:

- `sQ_nope[64, 512]` BF16
- `sQ_pe[64, 64]` BF16
- `sKc[64, 512]` BF16
- `sKp[64, 64]` BF16
- `sP[64, 64]` BF16
- small metadata area for masks/barriers

Approximate footprint:

- `sQ_nope`: 64 KB
- `sQ_pe`: 8 KB
- `sKc`: 64 KB
- `sKp`: 8 KB
- `sP`: 8 KB
- metadata/barriers: a few KB

Total is comfortably within SM100 shared-memory limits for a non-persistent baseline.

### TMEM

Use TMEM only for accumulator tiles:

- `S` region: QK accumulator, `64 x 64` FP32
- `O` region: SV accumulator, `64 x 512` FP32, materialized as two `64 x 256` tiles

Column plan:

- `S`: cols `0-63`
- `O`: cols `64-319`

Total TMEM requirement is about `320` columns, which is much simpler than the previous `Q + P + O + S` layout.

## Data Loading Strategy

### Q loading

Do not use TMA for Q in v2.

Instead:

- cooperative global loads fill the first 16 rows of `sQ_nope` and `sQ_pe`
- the remaining rows are zero-filled in shared memory

This matches the real input layout directly and avoids forcing a 16-row tensor into a TMA/TMEM path designed for larger tiles.

### Sparse KV loading

Do not require TMA gather4 in v2.

Instead:

- one load warp reads the 64 sparse token indices for the current split
- load warps cooperatively gather rows from `ckv_cache` and `kpe_cache` into `sKc` and `sKp`
- invalid `-1` indices produce zero-filled rows and a per-row validity mask

This is slower than FlashMLA's optimized TMA gather path, but it is much easier to implement and still gives us the main win: the whole attention body stays inside one CuTe kernel instead of bouncing back to PyTorch.

TMA gather4 belongs in the next optimization iteration after this fused baseline is working.

## MMA Strategy

### QK

Run two MMA phases that accumulate into the same TMEM score tile:

1. `Q_nope @ Kc^T`
2. `Q_pe @ Kp^T`

Use SMEM-backed operands for both phases and accumulate into the TMEM `S` region.

This is exactly the kind of work CuTeDSL is already good at in the existing v3 QK-only kernel, but now it happens inside the full fused kernel.

### Softmax

Softmax warps:

- read `S` from TMEM into registers
- apply `sm_scale`
- apply the split-local validity mask
- compute row max and row sum in FP32
- compute `partial_lse` in log2 base
- convert normalized probabilities to BF16
- write those probabilities to `sP`

For v2, this is the correct place to materialize P:

- **SMEM, not TMEM**

That makes the following SV step much simpler.

### SV

Run `P @ Kc` using:

- `P` from `sP`
- `V` from `sKc`
- result accumulated into TMEM `O`

Use two `N=256` output tiles to cover the full 512 output dimension.

This is the baseline version of the fused SV path. It is intentionally simpler than the FlashAttention 4 / FlashMLA TMEM-P path.

## Warp Specialization

Use 8 warps:

- warps `0-3`: softmax + epilogue
- warp `4`: MMA warp
- warp `5`: ckv gather/load
- warp `6`: kpe gather/load
- warp `7`: sparse index handling + Q loading + misc coordination

This keeps the same overall decomposition that the earlier v2 note wanted, but the responsibilities are simpler because the load path is simpler.

## Synchronization

Use a minimal barrier structure:

- barrier after Q/K loads are complete
- barrier after QK accumulation is complete
- barrier after `sP` is ready
- barrier after SV accumulation is complete

The baseline should prefer obvious, correct synchronization over aggressive pipelining tricks.

## Runtime Contract

The benchmarked runtime path should be CuTe-only for the fused kernel path:

- no silent eager fallback on the submission path
- if the fused CuTe kernel is unavailable or fails, the benchmark path should fail loudly

Reference or debug implementations can still exist, but they should not be the competition entrypoint once v2 is enabled.

## Expected Outcome

The previous v2 note promised an unrealistic `75-150x` improvement over the vectorized baseline. That should not be the planning target.

The realistic v2 success criteria are:

1. the kernel is fully fused for the split attention body
2. the runtime path used by bench is the same CuTe path analyzed by SASS
3. the result is dramatically faster than the current QK-only CuTe v3 path
4. the result is at least in the same order of magnitude as the vectorized PyTorch split-KV baseline

If v2 lands as a correct fused kernel with real tensor-core SASS and no PyTorch inner-loop fallback, it is a successful baseline even if it is still far from FlashMLA-class performance.

## Future Optimization Path

Once this fused baseline works, the next optimizations should be considered in this order:

1. replace manual sparse KV loads with TMA gather4
2. improve register budgeting with `setmaxnreg`
3. consider moving P from SMEM to TMEM for TS-mode SV
4. overlap load and compute with deeper pipelines
5. consider more FlashMLA-like scheduling only after the simple fused version is stable

## References

These are the right references for this design, but v2 should borrow the structure selectively rather than copying the full optimization stack:

- [flash_fwd_sm100.py](/home/mark123/projects/FlashMLA/references/flash-attention/flash_attn/cute/flash_fwd_sm100.py)
- [flashmla_implementation.md](/home/mark123/projects/FlashMLA/notes/benchmark-reviews/flashmla_implementation.md)
- [kernel_v3_cutedsl.py](/home/mark123/projects/FlashMLA/solution/dsa_attention/kernel_v3_cutedsl.py)
- CUTLASS Blackwell FMHA example:
  - https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

The important planning lesson is:

- **FlashMLA and CUTLASS FMHA are references for where we want to go**
- **v2 should be the simplest fully fused CuTe baseline that gets onto the real runtime path**
