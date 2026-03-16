# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Microbenchmark **all performance-critical PTX instructions** on the B200 GPU (sm100a / Blackwell architecture) to inform writing an optimized DeepSeek Sparse Attention (DSA) decode kernel AND indexer kernel for a competition.

The target instructions are those used in the FlashMLA sm100 decode kernel pipeline: TMA gather4, UTCMMA (tcgen05.mma), TMEM load/store, vectorized f32x2 math, wide global loads/stores, transactional barriers, tcgen05 fences, bulk stores, cache policy creation, and TMA bulk reduce.

PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/#instruction-set

## Competition Workload

**Task**: Optimize a batched sparse decode attention kernel for B200 (sm100a).

**Evaluation parameters**:
- `num_tokens` = 1-2 (pure decode, tiny batch)
- `num_pages` = 8462 (~541K KV tokens, page_size=64)
- `topk` = 2048 (sparse selection per query token)
- `h_q` = 16 query heads
- `ckv` = 512d BF16 (head_dim_ckv)
- `kpe` = 64d BF16 (head_dim_kpe)
- `sm_scale` ~ 0.135
- **Separate** `ckv_cache [num_pages, 64, 512]` and `kpe_cache [num_pages, 64, 64]` — NOT FP8, NOT interleaved
- `sparse_indices [num_tokens, 2048]` — -1 indicates invalid entries
- Output: `[num_tokens, 16, 512]` BF16
- LSE: `[num_tokens, 16]` float32 in **log2 base**

**Key differences from FlashMLA decode kernel**:
| Aspect | Competition | FlashMLA Decode |
|---|---|---|
| KV precision | **BF16** | FP8 (float8_e4m3) |
| KV layout | **Separate** ckv_cache + kpe_cache | Single interleaved 656B/token |
| Dequantization | **None needed** | FP8 -> BF16 with per-tile scales |
| Batch size | **1-2 tokens** | Typically larger batches |
| sparse_indices shape | `[num_tokens, topk]` | `[b, s_q, topk]` |
| LSE base | **log2** | natural log (kernel uses log2 internally) |

**Why this matters for kernel design**:
1. Skip FP8 dequantization entirely — WG2 (dequant warpgroup) in FlashMLA is unnecessary
2. Split-KV is essential — topk=2048, B_TOPK=64 means 32 blocks per query. With only 1-2 tokens, you MUST split across SMs for parallelism
3. Memory-bound — 2048 tokens x (512+64) x 2 bytes = ~2.3 MB of KV per query token, scattered across ~8462 pages. TMA gather efficiency is critical
4. Two separate TMA gather streams — one for ckv_cache (512d), one for kpe_cache (64d)
5. Output is Kc (NoPE only) — SV GEMM uses ckv_cache values (512d), same data already gathered for QK

## Build & Run Microbenchmarks

All microbenchmarks run on **Modal B200** (remote). No local GPU needed.

### TMA gather4 benchmark
```bash
cd microbenchmarks/tma_gather4
modal run run_modal.py
```
Build flags: `-std=c++20 -gencode arch=compute_100a,code=sm_100a -O3 -lcuda`

### Visualize results
```bash
cd microbenchmarks/tma_gather4
python3 -m http.server 8765   # open http://localhost:8765/visualize.html
```
Paste CSV into the UI. Charts are generated dynamically from `experiment` and `x_var` CSV fields.

## B200 (SM100) Hardware Specs

### Datasheet specs
| Spec | Value | Source |
|---|---|---|
| SMs | 148 (across 8 GPCs, dual-die via NV-HBI) | arxiv:2512.02189 |
| HBM3e capacity | 192 GB | arxiv:2512.02189 |
| HBM3e peak BW | 8 TB/s | NVIDIA product page |
| L2 cache | ~64 MB (4 partitions, 2x Hopper) | empirical L2 cliff + arxiv:2512.02189 |
| Shared memory / SM | up to 228 KB configurable (default 48 KB static) | cudaFuncSetAttribute testing |
| TMEM (Tensor Memory) / SM | 256 KB (512 cols x 128 lanes x 32-bit) | arxiv:2512.02189 |
| Register file / SM | 256 KB (65,536 x 32-bit registers) | arxiv:2507.10789 |
| Max threads / SM | 2,048 (64 warps) | arxiv:2512.02189 |
| Warp schedulers / SM | 4 (sub-cores/partitions) | arxiv:2507.10789 |
| CUDA cores / SM | 128 FP32 (unified INT32/FP32) | arxiv:2507.10789 |
| Process | TSMC 4NP, 208B transistors (dual-die) | NVIDIA |
| NVLink | 5th gen, 1.8 TB/s GPU-to-GPU | NVIDIA |
| Compute capability | sm_100a (datacenter, required for cta_group::1) | NVIDIA |
| CUDA version | 12.9+ required for SM100 kernels | FlashMLA README |

### Microbenchmark-measured performance
| Metric | Value | Source |
|---|---|---|
| HBM sustained BW (STREAM Triad) | ~7.48 TB/s (93.5% of peak) | arxiv:2512.02189 |
| Global mem latency (cache miss) | ~420 cycles (~200 ns at ~2.1 GHz) | arxiv:2512.02189 |
| L1 cache hit latency | 30-40 cycles | arxiv:2507.10789 |
| Shared memory latency | ~20-30 cycles (low warp counts) | arxiv:2507.10789 |
| TMEM read BW | ~16 TB/s per SM | arxiv:2512.02189 |
| TMEM write BW | ~8 TB/s per SM | arxiv:2512.02189 |
| Optimal TMEM tile | 64x64 elements | arxiv:2512.02189 |
| UTCMMA (tcgen05.mma) latency | 11.0-11.4 cycles (constant across tile sizes) | arxiv:2512.02189 |
| BF16 tensor core throughput | 1,926 TFLOPS (96.3% of peak) | arxiv:2512.02189 |
| FP16 tensor core throughput | 1,930 TFLOPS (96.5% of peak) | arxiv:2512.02189 |
| FP8 tensor core throughput | 3,851 TFLOPS | arxiv:2512.02189 |
| FP64 throughput | 44.8 TFLOPS | arxiv:2512.02189 |

### UTCMMA latency by tile size (arxiv:2512.02189)
| Tile Shape | Latency (cycles) |
|---|---|
| m64n64k16 | 11.0 |
| m128n128k16 | 11.3 |
| m256n256k16 | 11.4 |

Key: unlike Hopper's wgmma (32-128 cycles scaling linearly with tile width), SM100 UTCMMA is near-constant ~11 cycles.

## SM100 PTX Intrinsics Catalog (Performance-Relevant)

All wrappers from `csrc/kerutils/include/kerutils/device/`. These are the instructions to microbenchmark for the competition kernel.

### Data Movement — TMA (highest impact, memory-bound bottleneck)

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `cp.async.bulk.tensor.2d...tile::gather4...cta_group::1.L2::cache_hint` | `tma_gather4()` | sm100/intrinsics.cuh:10 | **Critical** — loads 4 arbitrary rows from 2D tensor map into smem |
| `cp.async.bulk.prefetch.tensor.2d.L2...tile::gather4.L2::cache_hint` | `tma_gather4_prefetch()` | sm100/intrinsics.cuh:26 | **High** — L2 prefetch for gather4 |
| `cp.async.bulk.tensor.2d...tile::gather4...cta_group::2.L2::cache_hint` | `tma_gather4_cta_group_2()` | sm100/intrinsics.cuh:38 | **Medium** — cluster variant for cross-CTA sync |

### Data Movement — Global Load/Store

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `ld.global[.nc].L1::[hint].L2::[hint].v4.u64` (256-bit) | `KU_LDG_256` macro | sm100/intrinsics.cuh:183 | **Medium** — index/metadata loading (Warp 7). Supports `.nc` (non-coherent), L1/L2 cache hints (evict_first/normal/last/unchanged/no_allocate), L2 prefetch sizes (64B/128B/256B) |
| `st.global.L1::[hint].L2::[hint].v4.u64` (256-bit) | `KU_STG_256` macro | sm100/intrinsics.cuh:199 | **Medium** — output store with L1/L2 cache hints |
| `st.weak.shared::cta.b128` | `st_128b` lambda | kernel.cuh:772 | **Medium** — 128-bit shared memory store, avoids bank conflicts in K data path |
| `st.bulk.weak.shared::cta` | `st_bulk()` | sm100/intrinsics.cuh:102 | **Low-Medium** — bulk shared memory zero-fill |

### Data Movement — TMA Bulk Reduce & Async Copy

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` | `tma_bulk_reduce_add()` | sm90/intrinsics.cuh:41 | **Medium** — shared-to-global atomic reduce for output accumulation |
| `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64` | `st_async<T>()` | sm90/intrinsics.cuh:10 | **Medium** — 128-bit async store to peer CTA shared memory (DSM crossover) |
| `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes` | `cp_async_bulk_shared_cta_to_shared_cluster()` | sm90/intrinsics.cuh:96 | **Medium** — intra-cluster shared-to-shared async copy |

### Compute — UTCMMA (Tensor Core)

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `tcgen05.mma.ws.cta_group::1.kind::f16` (TMEM x Shared) | `SM100_MMA_F16BF16_WS_TS_NOELECT` | gemm.cuh:38 | **High** — used for QK^T GEMM: Q (TMEM) x K (smem) |
| `tcgen05.mma.ws.cta_group::1.kind::f16` (Shared x Shared) | `SM100_MMA_F16BF16_WS_SS_NOELECT` | gemm.cuh:135 | **High** — used for SV GEMM: S (smem) x V (smem) |
| `tcgen05.mma.cta_group::1.kind::f16` (TMEM x Shared, no .ws) | `SM100_MMA_F16BF16_TS_NOELECT` | gemm.cuh:447 | **Medium** — non-warp-specialized TS variant |
| `tcgen05.mma.cta_group::1.kind::f16` (Shared x Shared, no .ws) | `SM100_MMA_F16BF16_SS_NOELECT` | gemm.cuh:554 | **Medium** — non-warp-specialized SS variant |
| `tcgen05.mma.cta_group::2.kind::f16` (2-CTA TS) | `SM100_MMA_F16BF16_2x1SM_TS_NOELECT` | gemm.cuh:231 | **Low** — 2-CTA cluster MMA variant |
| `tcgen05.mma.cta_group::2.kind::f16` (2-CTA SS) | `SM100_MMA_F16BF16_2x1SM_SS_NOELECT` | gemm.cuh:340 | **Low** — 2-CTA cluster SS variant |

UTCMMA helper functions: `utcmma_ss()` (helpers.cuh:18) and `utcmma_ts()` (helpers.cuh:54) handle descriptor creation, K-dimension unrolling, and accumulator control.

UMMA swizzle layout helpers: `make_umma_canonical_k_major_layout<MN,K,SWIZZLE,T>()` and `make_umma_canonical_mn_major_layout<>()` — construct CuTe layouts with INTER/SW32/SW64/SW128 swizzle atoms for UTCMMA smem descriptors.

### Compute — Vectorized FP32 (softmax loop)

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `add.f32x2` | `float2_add()` | sm100/intrinsics.cuh:58 | **Medium** — 2-wide f32 add (softmax) |
| `mul.f32x2` | `float2_mul()` | sm100/intrinsics.cuh:69 | **Medium** — 2-wide f32 mul (scaling) |
| `fma.rn.f32x2` | `float2_fma()` | sm100/intrinsics.cuh:81 | **Medium** — 2-wide f32 FMA (rescaling O) |
| `(neg via mul by -1)` | `float2_neg()` | sm100/intrinsics.cuh:95 | **Low** — uses float2_mul |

### Synchronization — Barriers & Fences (pipeline bubbles)

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `tcgen05.commit.cta_group::1.mbarrier::arrive::one` | `umma_arrive_noelect()` | sm100/intrinsics.cuh:218 | **High** — commits UTCMMA output, signals barrier |
| `tcgen05.commit.cta_group::1...multicast::cluster` | `umma_arrive_multicast_noelect()` | sm100/intrinsics.cuh:229 | **Medium** — multicast variant for cluster |
| `tcgen05.commit.cta_group::2...` | `umma_arrive_2x1SM_noelect()` | sm100/intrinsics.cuh:240 | **Low** — 2-CTA commit variant |
| `tcgen05.fence::before_thread_sync` | `tcgen05_before_thread_sync()` | sm100/intrinsics.cuh:261 | **Medium** — required fence before __syncthreads in TMEM-heavy kernels |
| `tcgen05.fence::after_thread_sync` | `tcgen05_after_thread_sync()` | sm100/intrinsics.cuh:266 | **Medium** — required fence after __syncthreads in TMEM-heavy kernels |
| `barrier.cluster.arrive.release` | `barrier_cluster_arrive_release()` | sm90/intrinsics.cuh:51 | **Medium** — cluster-level barrier with release semantics |
| `barrier.cluster.wait.acquire` | `barrier_cluster_wait_acquire()` | sm90/intrinsics.cuh:62 | **Medium** — cluster-level barrier with acquire semantics |
| `mbarrier.arrive.relaxed.cluster` | `mbarrier_arrive_relaxed_cluster()` | sm90/intrinsics.cuh:69 | **Medium** — relaxed mbarrier arrive for cluster sync |
| Standard mbarrier ops (init, expect_tx, arrive, try_wait.parity) | via CuTe `transac_bar_t` | common.h | **High** — per-iteration overhead |

### TMEM Access

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `tcgen05.ld.32dp32bNx` (N=1..128) | `tmem_ld_32dp32bNx<N>()` | sm100/intrinsics.cuh:272 | **High** — P extraction for softmax, O readback |
| `tcgen05.ld.16dp128bNx` (N=1..64) | `tmem_ld_16dp128bNx<N>()` | sm100/intrinsics.cuh:301 | **Medium** — alternative TMEM load layout |
| `tcgen05.ld.16dp256bNx` (N=1..32) | `tmem_ld_16dp256bNx<N>()` | sm100/intrinsics.cuh:328 | **Medium** — wider TMEM load |
| `tcgen05.st.32dp32bNx` (N=1..128) | `tmem_st_32dp32bNx<N>()` | sm100/intrinsics.cuh:354 | **High** — O rescaling writeback, Q loading |

### Cache Policy

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `createpolicy.fractional.L2::evict_last.b64` | `createpolicy_evict_last()` | sm90/helpers.h:25 | **Medium** — L2 cache eviction policy |
| `createpolicy.fractional.L2::evict_first.b64` | `createpolicy_evict_first()` | sm90/helpers.h:35 | **Medium** — alternative eviction policy |
| `createpolicy.fractional.L2::[primary].L2::[secondary].b64` | `create_fraction_based_cache_policy<P,S>()` | sm80/intrinsics.cuh:38 | **Medium** — parameterized policy with fraction |

### Cluster Launch Control (CLC)

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `clusterlaunchcontrol.try_cancel.async...b128` | `issue_clc_query()` | sm100/intrinsics.cuh:124 | **Low** — persistent kernel scheduling |
| `clusterlaunchcontrol.query_cancel.is_canceled...` | `get_clc_query_response<>()` | sm100/intrinsics.cuh:150 | **Low** — CLC response parsing |

### Type Conversion (sm100/helpers.h)

| Function | Purpose |
|---|---|
| `fp8x2_to_bf16x2_with_scale()` | FP8 dequant (not needed for competition, but used in FlashMLA reference kernel) |

### Atomic Operations (sm90/intrinsics.cuh)

| PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|
| `red.relaxed.gpu.global.add.L2::cache_hint.v4.f32` | `atomicadd_f32x4_with_policy_and_pred()` | sm90/intrinsics.cuh:81 | **Medium** — vectorized atomic add with cache hint and predicate, for output accumulation |

## Microbenchmarking Research

- Dissecting the NVIDIA Hopper Architecture through Microbenchmarking: https://arxiv.org/pdf/2501.12084
- Microbenchmarking NVIDIA's Blackwell Architecture (Dec 2025): https://arxiv.org/html/2512.02189v1
- Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks (Jul 2025): https://arxiv.org/html/2507.10789v2
- Dissecting the NVidia Turing T4 GPU via Microbenchmarking: https://arxiv.org/pdf/1903.07486
- Dissecting GPU Memory Hierarchy through Microbenchmarking: https://arxiv.org/pdf/1509.02308
- Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking: https://arxiv.org/pdf/1804.06826

## Reference Benchmark Suites

- references/gpu-benches — L1/L2/shared/global bandwidth, latency, cache, roofline, stream, strides
- references/NVIDIA-Hopper-Benchmark — TMA (1D/2D/3D throughput, latency), DSM, DPX, tensor cores, ALU, memory hierarchy
- references/mem_benchmarks.cu — standalone memory benchmark
- references/pchase.cu — pointer-chase latency benchmark
- references/wmma_benchmarks.cu — WMMA (tensor core) benchmark

## My Benchmarks

- notes/tma_gather4_plan.md — full microbenchmark plan with experiment designs, PTX catalog, and implementation details
- microbenchmarks/tma_gather4/ — standalone TMA gather4 benchmark suite (main.cu, gather4_kernels.cuh, tensor_map_utils.cuh, index_patterns.cuh, common.cuh, visualize.html)
- notes/tma_gather4_analysis.md — benchmark results analysis vs B200 specs
- microbenchmarks/tma_gather4/results/tma-gather4-ver*.csv — successive benchmark runs on Modal B200

## DeepSeek Sparse Attention Notes

- notes/deepseek_sparse_attention_algorithm.md — DSA algorithm from MLA basics to sparse indexing
- notes/flashmla_implementation.md — FlashMLA kernel implementation notes
- docs/20250929-hopper-fp8-sparse-deep-dive.md — FlashMLA FP8 sparse decode deep dive (crossover technique using DSM)

## FlashMLA Reference Code (for kernel design, not for running)

### Key kernel files (priority reading order for competition)
1. csrc/sm100/decode/head64/kernel.cuh — **THE main kernel**: online softmax loop, TMA gather pattern, split-KV accumulation, O rescaling (skip FP8 dequant sections)
2. csrc/sm100/decode/head64/config.h — tile sizes (B_H=64, B_TOPK=64), TMEM column assignments, barrier setup, shared memory plan
3. csrc/sm100/decode/head64/kernel.h — kernel launch + TMA tensor map construction for gather operations
4. csrc/sm100/prefill/sparse/common_subroutine.h — softmax building blocks: `load_indices_and_generate_mask()`, `retrieve_mask_and_reduce_p()`, `rescale_O()`, `get_max()`, `get_s_from_p()`
5. csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu — how topk=2048 blocks are partitioned across SMs
6. csrc/smxx/decode/combine/combine.cu — how partial (O_accum, LSE_accum) from splits are merged
7. csrc/params.h — `SparseAttnDecodeParams`, `DecodingSchedMeta`, `CombineParams`
8. csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh — all sm100 PTX wrappers
9. csrc/kerutils/include/kerutils/device/sm100/gemm.cuh — UTCMMA wrappers and MMA traits
10. csrc/kerutils/include/kerutils/device/sm100/helpers.cuh — UTCMMA helper functions (utcmma_ss, utcmma_ts) and UMMA layout utilities

### Build (for reference only)
```bash
git submodule update --init --recursive
pip install -v .          # requires SM90/SM100 GPU, CUDA 12.8+
```

### Tests (for reference only)
```bash
python tests/test_flash_mla_sparse_decoding.py
python tests/test_flash_mla_dense_decoding.py
python tests/test_flash_mla_sparse_prefill.py
python tests/test_fmha_sm100.py
```

## Reference Attention Kernel

```python
import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        # For page_size=64, indices encode (page_idx * 64 + offset)
        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]
        qn = q_nope[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[t].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[t] = out.to(torch.bfloat16)

    return output, lse
```

## Reference Indexer Kernel

```python
import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm format.

    Input: [num_pages, page_size, 1, 132] int8 (interpreted as uint8)
           Memory layout (per page): [fp8_data (page_size * 128 bytes), scales (page_size * 4 bytes)]
           After view to [num_pages, page_size, 1, 132]: NOT directly indexable as [fp8, scale] per token!
    Output: [num_pages, page_size, 128] float32
    """
    # View as uint8 for correct byte interpretation
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4  # 128

    # Go back to flat format to reverse the packing
    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)

    # FP8 part: first page_size * head_dim bytes
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    # Scale part: last page_size * 4 bytes -> page_size float32 values
    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [num_pages, page_size, 1]

    return fp8_float * scale


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    # Check constants
    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    # Dequantize inputs
    q = q_index_fp8.to(torch.float32)  # [batch, heads, head_dim]
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, page_size, head_dim]

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    max_num_pages = block_table.shape[1]

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())

        if seq_len == 0:
            continue

        # Get pages for this sequence
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        # Gather K from pages
        K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, head_dim]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, head_dim]

        # Query for this batch element
        q_b = q[b]  # [num_heads, head_dim]

        # Compute attention scores
        scores = q_b @ K.T  # [num_heads, seq_len]

        # Apply ReLU (deep_gemm uses ReLU activation)
        scores_relu = torch.relu(scores)  # [num_heads, seq_len]

        # Apply learned weights and sum across heads
        w = weights[b]  # [num_heads]
        weighted_scores = scores_relu * w[:, None]  # [num_heads, seq_len]
        final_scores = weighted_scores.sum(dim=0)  # [seq_len]

        # Select top-K
        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        # Convert to global token indices
        # Token index = page_idx * page_size + offset_in_page
        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)
```
