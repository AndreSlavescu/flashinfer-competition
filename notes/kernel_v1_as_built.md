# Kernel v1 As-Built Design

This note captures the implementation currently shipped in `solution/dsa_attention/kernel_v1.py`.

Use this as the canonical baseline when planning follow-on kernels for `kernel_v1.py`.
The older `notes/kernel_v1_design.md` is the original pre-implementation proposal and no longer fully matches the current code.

## Goal

Keep the public competition entrypoints correct while removing the original host-side nested-loop overhead from the split-KV baseline. The current file also carries a self-contained CuTeDSL QK experiment so Modal SASS tooling can compile and inspect a real SM100 kernel from the same source file.

## Public contract

- `run(...) -> (output, lse)`
- `kernel(..., output, lse)` writes in place
- Inputs:
  - `q_nope`: `[num_tokens, 16, 512]`
  - `q_pe`: `[num_tokens, 16, 64]`
  - `ckv_cache`: `[num_pages, 64, 512]`
  - `kpe_cache`: `[num_pages, 64, 64]`
  - `sparse_indices`: `[num_tokens, 2048]`
- Outputs:
  - `output`: `[num_tokens, 16, 512]` in `bfloat16`
  - `lse`: `[num_tokens, 16]` in `float32`, stored in log2 base

## High-level algorithm

The runtime is a split-KV implementation with `topk=2048` and `chunk_size=64`, so each token is processed as `32` independent sparse chunks.

For each token:
1. Gather `32 x 64` sparse KV rows from the paged caches.
2. Reshape gathered KV into chunked tensors:
   - `kc_chunks`: `[T, 32, 64, 512]`
   - `kp_chunks`: `[T, 32, 64, 64]`
3. Compute per-split ckv logits from `q_nope`.
4. Compute per-split kpe logits from `q_pe`.
5. Add the two logits tensors and apply `sm_scale`.
6. Run masked softmax independently inside each 64-token split.
7. Produce:
   - `partial_o`: `[32, T, 16, 512]`
   - `partial_lse`: `[32, T, 16]` in log2 base
8. Combine the 32 split partials with a stable `exp2` reduction.

## Current runtime architecture

### 1. Vectorized gather and masking

- The page caches are flattened so `sparse_indices` can directly index them as token offsets.
- `-1` entries are treated as invalid and tracked with `valid_mask`.
- Invalid gathered rows are zero-filled before math so later batched tensor ops can stay dense.

This is the main architectural shift from the older Python-loop baseline: the runtime materializes dense split tensors once and then uses batched PyTorch ops instead of iterating in Python over tokens and splits.

### 2. Vectorized split math in PyTorch

The default runtime path casts Q/K tensors to `float32` and computes the split math with batched `torch.einsum`:

- ckv logits:
  - `torch.einsum("thd,tsnd->tshn", q_nope_f, kc_f)`
- kpe logits:
  - `torch.einsum("thd,tsnd->tshn", q_pe_f, kp_f)`
- partial output:
  - `torch.einsum("tshn,tsnd->tshd", attn, kc_f)`

Softmax is done independently per split after masking invalid positions. Empty splits are handled explicitly so they contribute zero output and `-inf` LSE.

### 3. Split combine in PyTorch

The combine step remains a batched PyTorch reduction:

- take the per-head max over the 32 split LSE values
- form `exp2(partial_lse - max_lse)` weights
- compute the weighted sum of `partial_o`
- divide by the summed weights
- keep final `lse` in log2 base

This matches the split-KV contract expected by the benchmark harness while avoiding a second custom kernel.

## Experimental CuTeDSL QK path

The file contains a self-contained CuTeDSL SM100 kernel for only one subproblem: the ckv QK matmul.

### Scope

- Computes one padded `64 x 64 x 512` QK product per CTA
- Uses `256` threads
- Writes a `64 x 64` `float32` accumulator tile back to global memory
- Is not a full end-to-end attention kernel

### How it is integrated

- The runtime helper `_compute_qk_ckv_cutedsl(...)` pads `16` real query heads up to `m_block=64`
- It expands Q across all `32` splits and flattens K chunks so the launcher sees a batch of independent `64 x 64 x 512` problems
- After launch, only the first `16` rows are kept

### Important current behavior

- The CuTe runtime branch is gated by `DSA_ATTENTION_ENABLE_CUTEDSL_QK=1`
- If CuTe is disabled, unavailable, or throws during launch, the runtime falls back to the vectorized PyTorch ckv einsum path
- The file also exposes a `compile_only` path through `__main__` so `dump_sass_modal.py --cutedsl` can compile the real CuTe launcher from this file

This means the file serves two purposes today:
- benchmark/runtime baseline: vectorized split-KV PyTorch path
- SASS/compile target: self-contained CuTeDSL QK subkernel

## Why this implementation existed

- The biggest immediate win was deleting Python orchestration overhead from the original split loop.
- Keeping the CuTe kernel self-contained in the same file made Modal SASS tooling work without introducing extra source-file dependencies.
- Narrowing the CuTe scope to the ckv QK subproblem let the file carry a real SM100 kernel while leaving the larger attention flow in a stable vectorized implementation.

## As-built limitations

- The default benchmark path is not a full CuTeDSL attention kernel.
- Only the ckv QK subproblem has a CuTe candidate.
- kpe QK, softmax, SV accumulation, and split combine remain in PyTorch.
- Gathered sparse KV rows are materialized into dense chunk tensors, which can be a meaningful memory-traffic cost.
- SASS analysis and benchmark runtime are only fully coupled if a future iteration makes the CuTe path the default runtime path.

## Planning guidance for future iterations

- Treat `solution/dsa_attention/kernel_v1.py` as a vectorized split-KV PyTorch baseline with an embedded CuTeDSL QK experiment.
- When writing new design docs, state explicitly which layer is being optimized:
  - the vectorized PyTorch split path
  - the CuTe QK subkernel
  - the full runtime path
- Do not rely on `notes/kernel_v1_design.md` alone to infer the current implementation structure.
- If performance gains appear, verify which runtime path actually executed before attributing the win to the CuTe kernel.

## Code landmarks

- `DSASparseAttentionConfig` defines the fixed workload geometry.
- `_make_cutedsl_qk_launcher()` builds the self-contained CuTeDSL QK kernel and its `compile_only` hook.
- `_gather_kv_chunks()` flattens paged caches and materializes dense split tensors.
- `_compute_qk_ckv_cutedsl()` handles Q padding, split expansion, CuTe launch, and result slicing.
- `_compute_split_partials()` runs the vectorized split attention math.
- `_combine_splits()` merges the 32 split results into final output and log2 LSE.
