---
name: Fused 3-warpgroup kernel hang fix — defer_sync + cluster
description: Adding defer_sync=True to pipeline creates and cluster=(1,1,1) to launch fixed GPU deadlock in the fused kernel
type: reference
---

## Bug: Fused kernel hangs on GPU (TIMEOUT in benchmark)

### Symptoms

- Kernel compiles successfully (~1.8s)
- Kernel launch returns
- `torch.cuda.synchronize()` hangs forever
- Benchmark reports TIMEOUT, worker marked unhealthy

### Root cause (likely)

Multiple `PipelineAsync.create()` and `PipelineUmmaAsync.create()` calls without `defer_sync=True` each internally call `mbarrier_init_fence()` + `__syncthreads()`. With 4 pipeline creates in sequence, this generates 4 redundant sync points that may interact badly with the manual mbarrier init for `gather_mbar` and `q_tma_mbar`.

Additionally, launching without `cluster=(1,1,1)` when the kernel uses `cute.arch.block_idx_in_cluster()` and `cluster_layout_vmnk` may cause undefined behavior on SM100.

### Fix

1. Add `defer_sync=True` to all 4 pipeline creates
2. Add explicit `cute.arch.mbarrier_init_fence()` + `cute.arch.sync_threads()` after all pipeline creates
3. Add `cluster=(1, 1, 1)` to the kernel launch call

### Verified

- Fused kernel with these fixes: compiles (1.8s), launches, synchronizes (0.0s), produces output (no NaN)
- Without these fixes: hangs at synchronize
