# Competition Kernel: DSA Sparse Decode on B200 / SM100

## Context

**Competition task**: Optimize a batched sparse decode attention kernel for B200 (sm100a).

**Workload characteristics** (from competition evaluation configs):
- `num_tokens` = 1–2 (pure decode, tiny batch)
- `num_pages` = 8462 (~541K KV tokens, page_size=64)
- `topk` = 2048 (sparse selection per query token)
- `h_q` = 16, `ckv` = 512d, `kpe` = 64d
- **BF16 KV** — separate `ckv_cache` and `kpe_cache` tensors (NOT FP8, NOT concatenated)
- `sm_scale` ≈ 0.135

**Reference impl**: `output = softmax((q_nope @ Kc.T + q_pe @ Kp.T) * scale) @ Kc`

### Key differences from FlashMLA decode kernel

| Aspect | Competition | FlashMLA Decode |
|---|---|---|
| KV precision | **BF16** | FP8 (float8_e4m3) |
| KV layout | **Separate** ckv_cache + kpe_cache | Single interleaved 656B/token |
| Dequantization | **None needed** | FP8→BF16 with per-tile scales |
| Batch size | **1–2 tokens** | Typically larger batches |
| sparse_indices shape | `[num_tokens, topk]` | `[b, s_q, topk]` |
| LSE base | **log2** | natural log (kernel uses log2 internally) |

### What this means for your kernel

1. **Skip FP8 dequantization entirely** — WG2 (dequant warpgroup) in FlashMLA is unnecessary
2. **Split-KV is essential** — with topk=2048 and B_TOPK=64, that's 32 blocks per query. With only 1–2 tokens, you MUST split across SMs for parallelism
3. **Memory-bound** — 2048 tokens × (512+64) × 2 bytes = ~2.3 MB of KV per query token, scattered across ~541K pages. TMA gather efficiency is critical
4. **Two separate TMA gather streams** — one for ckv_cache (512d), one for kpe_cache (64d)
5. **Online softmax + split-KV combine** — standard pattern from FlashMLA

---

## Architecture for Competition Kernel

### Execution Model

```
Phase 1: Schedule    — partition 2048 topk indices across SM splits
Phase 2: Kernel      — each SM processes its KV block range → partial (O_accum, LSE_accum)
Phase 3: Combine     — merge partial results using LSE-based rescaling
```

With num_tokens=1, topk=2048, B_TOPK=64 → 32 blocks. B200 has ~160 SMs. You could use ~32 splits per token, each processing ~1 block of 64 KV tokens.

### Thread Block Design (adapted from FlashMLA head64 decode)

With no FP8 dequant needed, you can simplify to 2 warpgroups (256 threads):

| Warpgroup | Role | Details |
|---|---|---|
| WG0 (128 threads) | **Scale & Exp** | Online softmax: load P from TMEM, mask invalids, max/exp, rescale O, write LSE |
| WG1 (128 threads) | **MMA + Data Movement** | Warp 4: UTCMMA (QK and SV GEMMs). Warp 5: TMA gather ckv_cache. Warp 6: TMA gather kpe_cache. Warp 7: index loading + validity masks |

### Per-Block Processing Loop (B_TOPK=64 KV tokens per iteration)

```
for each KV block assigned to this SM:
  1. Warp 7: Load 64 sparse_indices, compute TMA coords, validity mask
  2. Warp 5: TMA gather 64 tokens from ckv_cache [64, 512] → shared mem
  3. Warp 6: TMA gather 64 tokens from kpe_cache [64, 64] → shared mem
  4. Warp 4: P = Q_nope @ Kc.T (UTCMMA_TS: [16,512] × [512,64] → [16,64])
  5. Warp 4: P += Q_pe @ Kp.T (UTCMMA_TS: [16,64] × [64,64] → [16,64])
  6. WG0: mask invalid, online softmax (max, exp, rescale O)
  7. WG0: store S to shared mem
  8. Warp 4: O += S @ Kc (UTCMMA_SS: [16,64] × [64,512] → [16,512])
```

### TMEM Layout (adapted)

| Region | Columns | Size | Purpose |
|---|---|---|---|
| O | 0–255 | 16 heads × 512d / 2 | Output accumulator (float32) |
| Q_nope | 256–383 | 16 × 512 / 2 | Query NoPE part |
| Q_pe | 384–399 | 16 × 64 / 2 | Query RoPE part |
| P | 400–431 | 16 × 64 / 2 | QK^T attention scores |

---

## Files to Study (Competition-Focused, Priority Order)

### Must-Read: Core Decode Kernel Mechanics

| # | File | Why | Focus On |
|---|---|---|---|
| 1 | [csrc/sm100/decode/head64/kernel.cuh](csrc/sm100/decode/head64/kernel.cuh) | **THE main kernel to study** | Online softmax loop, TMA gather pattern, split-KV accumulation, O rescaling. Ignore FP8 dequant code (WG2). |
| 2 | [csrc/sm100/decode/head64/config.h](csrc/sm100/decode/head64/config.h) | Tile sizes and TMEM layout | B_H=64, B_TOPK=64, TMEM column assignments, barrier setup, shared memory plan |
| 3 | [csrc/sm100/decode/head64/kernel.h](csrc/sm100/decode/head64/kernel.h) | Kernel launch + TMA setup | How TMA tensor maps are constructed for gather operations, grid dimensions |
| 4 | [csrc/sm100/prefill/sparse/common_subroutine.h](csrc/sm100/prefill/sparse/common_subroutine.h) | Softmax building blocks | `load_indices_and_generate_mask()`, `retrieve_mask_and_reduce_p()`, `rescale_O()`, `get_max()`, `get_s_from_p()` |

### Must-Read: Split-KV Infrastructure

| # | File | Why |
|---|---|---|
| 5 | [csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu](csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu) | How topk=2048 blocks are partitioned across SMs |
| 6 | [csrc/smxx/decode/combine/combine.cu](csrc/smxx/decode/combine/combine.cu) | How partial (O_accum, LSE_accum) from splits are merged into final output |
| 7 | [csrc/params.h](csrc/params.h) | `SparseAttnDecodeParams` (L63), `DecodingSchedMeta` (L10), `CombineParams` (L105) |

### Must-Read: SM100 Hardware Primitives

| # | File | Why |
|---|---|---|
| 8 | [csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh) | `tma_gather4()`, TMEM load/store (`tmem_ld_32dp32bNx`), TCGen05 sync |
| 9 | [csrc/kerutils/include/kerutils/device/sm100/gemm.cuh](csrc/kerutils/include/kerutils/device/sm100/gemm.cuh) | `utcmma_ts()` (for QK^T), `utcmma_ss()` (for SV), UMMA layout helpers |
| 10 | [csrc/sm100/helpers.h](csrc/sm100/helpers.h) | Type conversion utilities (bf16↔fp32) |

### Reference & Validation

| # | File | Why |
|---|---|---|
| 11 | [tests/ref.py](tests/ref.py) | `ref_sparse_attn_decode()` — validate your understanding of the math |
| 12 | [csrc/api/sparse_decode.h](csrc/api/sparse_decode.h) | How the 3-phase pipeline (schedule→kernel→combine) is orchestrated |

### Can Skip / Low Priority

| File | Why Skip |
|---|---|
| All `csrc/sm90/` files | Wrong architecture |
| `csrc/sm100/decode/head64/*/dequant.h` | No FP8 in competition |
| `csrc/sm100/prefill/` kernels | Competition is decode-only |
| `tests/quant.py` | FP8 quantization not needed |
| `fwd_for_small_topk/` | Head128 variant, competition uses h_q=16 |

---

## Reading Order (10 files, focused)

1. **[tests/ref.py:55-103](tests/ref.py)** → decode math, understand split + combine
2. **[csrc/params.h:10-16,63-103](csrc/params.h)** → `DecodingSchedMeta`, `SparseAttnDecodeParams`
3. **[csrc/sm100/decode/head64/config.h](csrc/sm100/decode/head64/config.h)** → tile sizes, TMEM layout, shared mem
4. **[common_subroutine.h](csrc/sm100/prefill/sparse/common_subroutine.h)** → softmax primitives
5. **[csrc/sm100/decode/head64/kernel.cuh](csrc/sm100/decode/head64/kernel.cuh)** → **the main kernel** (skip FP8 dequant sections)
6. **[csrc/sm100/decode/head64/kernel.h](csrc/sm100/decode/head64/kernel.h)** → TMA tensor map setup, launch config
7. **[get_decoding_sched_meta.cu](csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu)** → split-KV scheduling
8. **[combine.cu](csrc/smxx/decode/combine/combine.cu)** → split-KV combine
9. **[intrinsics.cuh](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh)** → TMA gather, TMEM ops
10. **[gemm.cuh](csrc/kerutils/include/kerutils/device/sm100/gemm.cuh)** → UTCMMA wrappers

## Key Optimization Considerations

1. **This is memory-bound**: 2048 sparse KV tokens × 576d × 2B = 2.36 MB scattered reads per query. With num_tokens=1–2, you have very few queries to amortize over.

2. **TMA gather is the bottleneck**: Each B_TOPK=64 block requires gathering 64 tokens from random pages. Maximize TMA gather throughput and overlap with compute.

3. **Split-KV granularity matters**: 32 blocks per query × 1–2 queries. Consider how many SMs per query gives best latency. Too many splits = combine overhead. Too few = underutilized SMs.

4. **Two separate gathers**: Unlike FlashMLA (single interleaved cache), you need two TMA streams — ckv_cache(512d) and kpe_cache(64d). The kpe gather is 8× smaller, consider combining or pipelining differently.

5. **No dequant = simpler pipeline**: Without FP8, you can potentially use 2 warpgroups instead of 3, freeing registers and shared memory.

6. **Output is Kc (NoPE only)**: The SV GEMM uses ckv_cache values (512d) — same data already gathered for QK. No need for a separate V gather.

---

## SM100 Microbenchmarking Strategy

### Why Microbenchmark

SM100 (Blackwell datacenter) microarchitecture is largely undocumented. Two recent papers provide baseline numbers but leave gaps specifically relevant to sparse attention:
- [Microbenchmarking NVIDIA's Blackwell Architecture (Dec 2025)](https://arxiv.org/html/2512.02189v1)
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks (Jul 2025)](https://arxiv.org/html/2507.10789v2)

The Citadel Volta methodology ([arXiv:1804.06826](https://arxiv.org/abs/1804.06826)) provides the template: use targeted SASS-level microbenchmarks to measure latency, throughput, and contention for specific hardware units.

### Known B200 Numbers (from existing research)

| Metric | B200 (SM100) Value | Source |
|---|---|---|
| UTCMMA (tcgen05.mma) latency | **11.0–11.4 cycles** (constant across tile sizes) | arxiv:2512.02189 |
| FP16 tensor core throughput | **1,929.2 TFLOPS** | arxiv:2512.02189 |
| TMEM read bandwidth | **16 TB/s per SM** | arxiv:2512.02189 |
| TMEM write bandwidth | **8 TB/s per SM** | arxiv:2512.02189 |
| HBM3e sustained bandwidth | **7.48 TB/s** (93.5% of peak) | arxiv:2512.02189 |
| Global memory latency (cache miss) | **~420 cycles** (58% lower than Hopper) | arxiv:2512.02189 |
| L1 cache hit latency | **30–40 cycles** | arxiv:2507.10789 |
| Shared memory capacity | **128 KB per SM** | arxiv:2507.10789 |
| Optimal TMEM tile | **64×64 elements** (4KB for FP8) | arxiv:2512.02189 |

### Gaps to Fill — What the Papers DON'T Cover

These are the measurements most relevant to our sparse decode kernel that existing research hasn't published:

#### Priority 1: TMA Gather (the bottleneck)

| Experiment | What to Measure | Why It Matters |
|---|---|---|
| **TMA gather4 latency** | Cycles from `cp.async.bulk.tensor.2d...gather4` issue to barrier completion | Determines minimum NUM_BUFS for double/triple buffering |
| **TMA gather4 throughput** | Max bytes/cycle with back-to-back gathers (vary row count, data size) | Sets absolute ceiling for our kernel |
| **TMA gather4 in-flight capacity** | How many concurrent gather4 ops before stalling | Whether ckv (512d) and kpe (64d) gathers compete for TMA unit |
| **TMA gather4 with random indices** | Throughput when row_idxs are scattered vs sequential | L2 cache impact on scattered page access (our pattern: ~8462 pages) |
| **TMA gather4 vs data size** | Latency/throughput for 64B, 128B, 256B, 512B, 1024B columns | Optimal chunk size for ckv(512d=1024B) vs kpe(64d=128B) |
| **L2 cache hint effectiveness** | Compare `evict_last`, `evict_first`, `no_allocate` for scattered gathers | Whether cache hints help when indices are random |
| **TMA gather4 prefetch benefit** | Throughput with/without `cp.async.bulk.prefetch.tensor.2d.L2...gather4` issued 1-2 iterations ahead | Whether L2 prefetch hides gather latency for scattered indices |
| **L2 cache policy creation** | Cost of `createpolicy.fractional.L2::evict_last/first` + impact on subsequent gathers | Whether evict_last keeps useful KV pages in L2 across split-KV SMs |

**Method**: Create a kernel with a single warp issuing gather4 in a loop. Vary: data size, index pattern (sequential/random/strided), number of concurrent requests (via multiple barriers). Measure cycles using `clock64()`.

**PTX instructions to test** (from [intrinsics.cuh](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh)):
- `cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint` (line 14)
- `cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint` (line 28)
- `createpolicy.fractional.L2::evict_last.b64` (from [sm90/helpers.h:27](csrc/sm90/helpers.h#L27))

#### Priority 2: UTCMMA Pipeline

| Experiment | What to Measure | Why It Matters |
|---|---|---|
| **UTCMMA_TS issue-to-completion** | Cycles for `tcgen05.mma.ws.cta_group::1.kind::f16` (tensor×shared) | Known ~11 cycles, verify for our tile sizes |
| **UTCMMA_SS issue-to-completion** | Cycles for shared×shared variant | May differ from TS |
| **UTCMMA + barrier overhead** | Cost of `tcgen05.commit` + barrier wait in pipeline | Pipeline bubble between QK and SV GEMMs |
| **UTCMMA dual-issue** | Can two UTCMMA ops overlap? (P and O in adjacent iterations) | Whether we need to separate QK and SV into different pipeline stages |
| **UTCMMA with TMEM read interleaving** | Throughput when WG0 reads P from TMEM while UTCMMA writes it | Whether softmax TMEM reads conflict with MMA writes |

**Method**: Kernel with tight UTCMMA loop, measure cycles per MMA. Then add tcgen05.commit + barrier, measure overhead. Test overlapping MMA with TMEM reads.

**PTX instructions to test** (from [gemm.cuh](csrc/kerutils/include/kerutils/device/sm100/gemm.cuh)):
- `tcgen05.mma.ws.cta_group::1.kind::f16` (line 38 — TS variant)
- `tcgen05.mma.ws.cta_group::1.kind::f16` (SS variant, same opcode, different descriptor)
- `tcgen05.commit.cta_group::1.mbarrier::arrive::one` (from [intrinsics.cuh:220](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L220))

#### Priority 3: Transactional Barriers

| Experiment | What to Measure | Why It Matters |
|---|---|---|
| **Barrier arrive cost** | Cycles for `mbarrier::arrive` + `expect_tx` | Per-iteration overhead |
| **Barrier wait latency** | Cycles from `mbar.wait` to resume when data is ready vs not ready | Whether to spin-wait or do useful work |
| **Barrier vs __syncthreads** | Cost comparison for full-block sync | Whether named barriers are cheaper |
| **tcgen05.fence overhead** | Cost of `fence::before_thread_sync` + `fence::after_thread_sync` around syncthreads | Hidden cost in TMEM-heavy kernels |

**Method**: Tight loops with barrier operations, measure with clock64(). Compare ping-pong between two warps using barriers vs polling.

**PTX instructions to test**:
- `tcgen05.fence::before_thread_sync` / `tcgen05.fence::after_thread_sync` (from [intrinsics.cuh:262-267](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L262))
- Standard mbarrier ops (arrive, wait, expect_tx)

#### Priority 4: Shared Memory & TMEM Interaction

| Experiment | What to Measure | Why It Matters |
|---|---|---|
| **SMEM bank conflict rate** | Throughput under different access patterns (stride-1, stride-4, stride-32) | Swizzle mode selection (SW0/16/32/64/128) |
| **TMEM→register bandwidth** | Throughput of `tmem_ld_32dp32bNx` for N=4,8,16,32 | How fast WG0 can extract P for softmax |
| **Register→TMEM bandwidth** | Throughput of `tmem_st_32dp32bNx` | How fast O rescaling writes back |
| **SMEM→TMEM (UTCCP)** | Throughput of `SM100_UTCCP_128dp256bit_1cta::copy` | Q loading speed |
| **STS.128 (st.weak.shared.b128)** | Throughput of 128-bit shared mem stores vs 64-bit or 32-bit | Used in K data path ([kernel.cuh:772](csrc/sm100/decode/head64/kernel.cuh#L772)). Forces wide store to avoid bank conflicts |
| **f32x2 vectorized throughput** | IPC of `add.f32x2` / `mul.f32x2` / `fma.rn.f32x2` vs scalar f32 | Used throughout softmax loop ([intrinsics.cuh:58-91](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L58)). If 2x throughput, keep vectorized; if same, not worth the register pressure |
| **LDG.256 throughput** | Sustained throughput of `ld.global.v4.u64` with different cache hints | Index/metadata loading on Warp 7 ([intrinsics.cuh:188](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L188)). Compare `.nc` (non-coherent) vs coherent, and L1 hint variants |

#### Priority 5: Contention & Concurrency

| Experiment | What to Measure | Why It Matters |
|---|---|---|
| **TMA gather + UTCMMA overlap** | Does issuing gather4 while UTCMMA runs degrade either? | Whether pipeline overlap actually works |
| **Multiple warps issuing gather4** | Throughput when 2 warps (ckv + kpe) vs 1 warp issues gathers | Whether to merge or separate the two gather streams |
| **L2 contention across SMs** | Throughput when many SMs all do scattered gathers simultaneously | Models the actual split-KV scenario (32 SMs × scattered reads) |

### Complete PTX Instruction Catalog (Performance-Relevant Only)

From the full codebase scan (48 total inline PTX sites), these are the **18 instructions with non-negligible performance impact** for the competition kernel:

**Data Movement (memory-bound = highest impact):**

| # | PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|---|
| 1 | `cp.async.bulk.tensor.2d...tile::gather4...cta_group::1` | `tma_gather4()` | [sm100/intrinsics.cuh:13](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L13) | **Critical** |
| 2 | `cp.async.bulk.prefetch.tensor.2d.L2...tile::gather4` | `tma_gather4_prefetch()` | [sm100/intrinsics.cuh:27](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L27) | **High** |
| 3 | `cp.async.bulk.tensor.2d...tile::gather4...cta_group::2` | `tma_gather4_cta_group_2()` | [sm100/intrinsics.cuh:44](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L44) | **Medium** (cluster variant) |
| 4 | `ld.global[.nc].L1::[hint].L2::[hint].v4.u64` | `KU_LDG_256` | [sm100/intrinsics.cuh:188](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L188) | **Medium** |
| 5 | `st.global.L1::[hint].L2::[hint].v4.u64` | `KU_STG_256` | [sm100/intrinsics.cuh:204](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L204) | **Medium** |
| 6 | `st.weak.shared::cta.b128` | `st_128b` lambda | [kernel.cuh:772](csrc/sm100/decode/head64/kernel.cuh#L772) | **Medium** |
| 7 | `st.bulk.weak.shared::cta` | `st_bulk()` | [sm100/intrinsics.cuh:104](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L104) | **Low-Medium** |
| 8 | `cp.reduce.async.bulk.global.shared::cta...add.f32` | `tma_bulk_reduce_add()` | [sm90/intrinsics.cuh:43](csrc/kerutils/include/kerutils/device/sm90/intrinsics.cuh#L43) | **Medium** (output accum) |

**Compute (softmax loop):**

| # | PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|---|
| 9 | `tcgen05.mma.ws.cta_group::1.kind::f16` | `SM100_MMA_F16BF16_WS_TS/SS_NOELECT` | [gemm.cuh:38](csrc/kerutils/include/kerutils/device/sm100/gemm.cuh#L38) | **High** |
| 10 | `add.f32x2` | `float2_add()` | [sm100/intrinsics.cuh:58](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L58) | **Medium** |
| 11 | `mul.f32x2` | `float2_mul()` | [sm100/intrinsics.cuh:71](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L71) | **Medium** |
| 12 | `fma.rn.f32x2` | `float2_fma()` | [sm100/intrinsics.cuh:84](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L84) | **Medium** |

**Synchronization (pipeline bubbles):**

| # | PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|---|
| 13 | `tcgen05.commit.cta_group::1.mbarrier::arrive::one` | `umma_arrive_noelect()` | [sm100/intrinsics.cuh:220](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L220) | **High** |
| 14 | `tcgen05.fence::before_thread_sync` | `tcgen05_before_thread_sync()` | [sm100/intrinsics.cuh:262](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L262) | **Medium** |
| 15 | `tcgen05.fence::after_thread_sync` | `tcgen05_after_thread_sync()` | [sm100/intrinsics.cuh:267](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L267) | **Medium** |

**TMEM Access:**

| # | PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|---|
| 16 | `tcgen05.ld.32dp32bNx` | `tmem_ld_32dp32bNx<N>()` | [sm100/intrinsics.cuh:274](csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh#L274) | **High** |
| 17 | `tcgen05.st.32dp32bNx` | `tmem_st_32dp32bNx<N>()` | (via CuTe TMEM_STORE) | **High** |

**Cache Policy:**

| # | PTX Instruction | Wrapper | Source | Impact |
|---|---|---|---|---|
| 18 | `createpolicy.fractional.L2::evict_last.b64` | `createpolicy_evict_last()` | [sm90/helpers.h:27](csrc/sm90/helpers.h#L27) | **Medium** |

### Microbenchmark Implementation on Godbolt

Since Godbolt supports sm_100a, you can:

1. **Write minimal CUDA kernels** with inline PTX for each operation being measured
2. **Inspect the SASS** to verify:
   - No unexpected register spills (`STL`/`LDL` in SASS)
   - Instructions aren't reordered by compiler (use `asm volatile`)
   - Correct dependency barriers assigned by ptxas
3. **Run locally** on a B200 for timing (Godbolt only shows SASS, not execution)

Example Godbolt skeleton for TMA gather4 latency:
```cuda
__global__ void bench_tma_gather4(const void* desc, void* smem_base, int* indices, long long* cycles_out) {
    __shared__ __align__(128) char smem[64*1024];
    __shared__ transac_bar_t bar;

    // Init barrier
    if (threadIdx.x == 0) { bar.init(1, expected_bytes); }
    __syncthreads();

    int4 row_idxs = make_int4(indices[0], indices[1], indices[2], indices[3]);

    long long start = clock64();
    // Issue gather4
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                 ".mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2,%3,%4,%5,%6}], [%7];"
                 :: ...);
    // Wait
    bar.wait(0);
    long long end = clock64();

    if (threadIdx.x == 0) cycles_out[0] = end - start;
}
```
