# Blackwell (SM100a / B200) Architecture Reference

Central ground-truth document for the kernel-optimizer agent. All values are either from NVIDIA specs, CUTLASS source code, or directly measured on Modal B200 hardware.

---

## 1. Device Properties (deviceQuery on Modal B200)

| Property | Value | Notes |
|---|---|---|
| GPU | NVIDIA B200 | Datacenter, dual-die via NV-HBI |
| Compute Capability | sm_100a | `a` suffix = datacenter features (cta_group, CLC) |
| SMs | 148 | Across 8 GPCs |
| Sub-cores per SM | 4 | Warp scheduler partitions, `warp_id % 4` mapping |
| Max threads per SM | 2,048 | 64 warps |
| Warp size | 32 | |
| Register file per SM | 256 KB | 65,536 × 32-bit registers |
| TMEM per SM | 256 KB | 512 columns × 128 lanes × 32-bit |
| Shared memory per SM | 228 KB | 233,472 bytes total configurable |
| Shared memory per block | 48 KB default | Up to 228 KB with opt-in |
| L2 cache | **126.5 MB** | 132,644,864 bytes — 2× many published claims of 64 MB |
| Total global memory | 178.351 GB | HBM3e |
| SM clock (boost) | 1,965 MHz | 1.965 GHz |
| SM clock (sustained, measured) | ~1,844–1,965 MHz | Varies with thermal; benchmarks measure 1.844–1.965 GHz |
| Memory clock | 3,996 MHz | |
| Memory bus width | 7,680 bits | |
| Peak memory BW (theoretical) | 7.67 TB/s | Formula: 2 × 3,996 MHz × 7,680 / 8 |
| Async engines | 4 | |
| CUDA version | 12.9+ required | For SM100 kernels |

---

## 2. Tensor Memory (TMEM) Architecture

### 2.1 Physical Organization

```
TMEM = 128 data-paths (rows/lanes) × 512 columns × 32-bit cells = 256 KB per SM
```

- **Addressing**: 32-bit TMEM addresses with 2D structure
  - Bits [31:16] = lane ID (data path)
  - Bits [15:0] = column offset
- **DP stride** (bit-addressing): `DP_b = 2^21 = 2,097,152 bits`
- **Column mask**: `0x0000FFFF`
- **Allocation granularity**: 32 columns minimum (power-of-2, from 32 to 512)
- **Allocation instructions**: `tcgen05.alloc.cta_group::{1|2}.sync.aligned`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`
- **Allocation constraint**: Must be issued by a **single fully active warp** of the CTA; same warp must issue all allocations; base address written to shared memory

### 2.2 Thread-to-TMEM Lane Mapping

Each warp accesses 32 lanes (from Colfax):
- Warp 0 → lanes 0–31
- Warp 1 → lanes 32–63
- Warp 2 → lanes 64–95
- Warp 3 → lanes 96–127

A full 128-lane readout requires a warpgroup (4 warps). This is why epilogue stages use warpgroups for TMEM→register transfer.

### 2.3 TMEM Column Allocation in FlashMLA Decode

From `csrc/sm100/decode/head64/config.h`:
```
Column 0–255:   Output accumulator (O)      — 256 cols for M=64, N=256 FP32
Column 256–399: Query (Q)                   — B_H×D_NOPE/2/128 cols (BF16)
Column 400–463: Attention weights (P)       — 64 cols for B_TOPK
                                             — Total: 464 cols of 512 used
```

### 2.4 TMEM Load/Store Modes

Three addressing modes for `tcgen05.ld.sync.aligned` / `tcgen05.st.sync.aligned`:

| Mode | Data Paths | Pattern Width | Registers per Thread | Use Case |
|---|---|---|---|---|
| `32dp32b×N` | 32 DP | 32-bit × N | N | General purpose, P extraction, Q loading |
| `16dp128b×N` | 16 DP | 128-bit × N | N×2 | Wider access, half the lanes |
| `16dp256b×N` | 16 DP | 256-bit × N | N×4 | Widest mode, UTCCP |

**Valid N values**:
- 32dp32b: 1, 2, 4, 8, 16, 32, 64, 128
- 16dp128b: 1, 2, 4, 8, 16, 32, 64
- 16dp256b: 1, 2, 4, 8, 16, 32

**Constraint**: `lanes × bits × N ≤ 128 Kb` (16 KB max per warp = ~128 registers/thread)

**Thread participation**: Warp-collective (`.sync.aligned` suffix = all 32 threads in warp execute)

### 2.5 Measured TMEM Performance

| Metric | Value | Source |
|---|---|---|
| TMEM read BW (per SM, arxiv) | ~16 TB/s | arxiv:2512.02189 |
| TMEM write BW (per SM, arxiv) | ~8 TB/s | arxiv:2512.02189 |
| Optimal TMEM tile | 64×64 elements | arxiv:2512.02189 |
| 32dp32b×1 RAW latency | 1.80 cy (0.98 ns) | utcmma-ver8 measured |
| 32dp32b×64 RAW latency | 3.73 cy (2.02 ns) | utcmma-ver8 measured |
| 16dp128b×32 RAW latency | 5.73 cy (3.11 ns) | utcmma-ver8 measured |
| 16dp256b×16 RAW latency | 9.74 cy (5.28 ns) | utcmma-ver8 measured |
| TMEM ld model (32dp32b) | `cy = 1.74 + 0.031 × WIDTH` | utcmma-ver8 linear fit (R²≈0.999) |
| TMEM single-warp BW | 126.7 GB/s | 256 bytes / 2.02 ns (from 32dp32b×64) |
| UTCCP startup overhead | ~168 cycles | utcmma-ver5 (128dp256b, N=1) |
| UTCCP marginal cost | ~68 cy/copy (~37 ns) | utcmma-ver5 (from copy 2 onward) |

---

## 3. UTCMMA (tcgen05.mma) — 5th Generation Tensor Cores

### 3.1 Instruction Syntax

```
tcgen05.mma[.ws].cta_group::{1|2}.kind::{f16|tf32|f8f6f4|s8|boolean}
    [tmem_d], {[tmem_a]|desc_a}, desc_b, idesc, {mask[0..N]}, pred;
```

**Key modifiers**:
- `.ws` (warp-specialized): MMA runs asynchronously, decoupled from issuing thread. Requires commit+mbarrier for completion notification.
- No `.ws`: MMA still async but uses elect_one_sync() pattern (CuTe default). FlashMLA NOELECT variants remove this.
- `.cta_group::1`: Single-SM MMA
- `.cta_group::2`: Two paired SMs cooperate (TMEM shared across SM pair)

### 3.2 Thread Participation

**tcgen05.mma is issued by 1 thread per CTA** (not warp-collective). All data lives in CTA-shared spaces (TMEM + SMEM), so no register consumption for operands. This is a fundamental difference from Hopper's wgmma.

### 3.3 Operand Modes

| Mode | Operand A | Operand B | Accumulator D | PTX Pattern |
|---|---|---|---|---|
| **TS** (TMEM×Shared) | TMEM address (uint32_t) | SMEM descriptor (uint64_t) | TMEM address (uint32_t) | `[tmem_d], [tmem_a], desc_b, ...` |
| **SS** (Shared×Shared) | SMEM descriptor (uint64_t) | SMEM descriptor (uint64_t) | TMEM address (uint32_t) | `[tmem_d], desc_a, desc_b, ...` |

**TS constraint**: Operand A from TMEM must be **K-major** (`a_major == UMMA::Major::K`); TMEM operands cannot be transposed.

### 3.4 Valid Tile Shapes (FP16/BF16, kind::f16)

**K dimension**: Always **256 bits** per instruction = 16 elements for BF16/FP16.

**1SM (cta_group::1)**:

| M | N range | N step | Notes |
|---|---|---|---|
| 64 | 8–256 | 8 | Full N flexibility |
| 128 | 16–256 | 16 | Coarser N granularity |

**2SM (cta_group::2)**:

| M | N range | N step | Notes |
|---|---|---|---|
| 128 | 16–256 | 16 | Each SM handles M/2=64 rows |
| 256 | 16–256 | 16 | Each SM handles M/2=128 rows |

**Static asserts from FlashMLA gemm.cuh confirm these ranges.**

### 3.5 Instruction Descriptor (idesc) — 32-bit

From `cute/arch/mma_sm100_desc.hpp`, union `InstrDescriptor`:

```
Bit [0,2)   sparse_id2     Sparse metadata id2
Bit [2,3)   sparse_flag    0=dense, 1=sparse
Bit [3,4)   saturate       0=no, 1=saturate (S8 only)
Bit [4,6)   c_format       0=F16, 1=F32, 2=S32
Bit [6,7)   (reserved)
Bit [7,10)  a_format       F16F32: 0=F16,1=BF16,2=TF32 | MXF8F6F4: 0=E4M3,1=E5M2,3=E2M3,4=E3M2,5=E2M1
Bit [10,13) b_format       (same encoding as a_format)
Bit [13,14) a_negate       0=no, 1=negate A
Bit [14,15) b_negate       0=no, 1=negate B
Bit [15,16) a_major        0=K-major, 1=MN-major
Bit [16,17) b_major        0=K-major, 1=MN-major
Bit [17,23) n_dim          N/8 (3 LSBs excluded). Range: 1 (N=8) to 32 (N=256)
Bit [23,24) (reserved)
Bit [24,29) m_dim          M/16 (4 LSBs excluded). Values: 4 (M=64), 8 (M=128), 16 (M=256)
Bit [29,30) (reserved)
Bit [30,32) max_shift      0=none, 1=shift8, 2=shift16, 3=shift32
```

**Passed as upper 32 bits of uint64_t idescE**: `idescE = (uint64_t(idesc) << 32) | tmem_e_sparse`

### 3.6 Shared Memory Descriptor (SmemDescriptor) — 64-bit

From `cute/arch/mma_sm100_desc.hpp`, union `SmemDescriptor`:

```
Bit [0,14)   start_address         SMEM base >> 4 (4 LSBs excluded)
Bit [14,16)  (padding)
Bit [16,30)  leading_byte_offset   MN-dim stride in uint128_t units (4 LSBs excluded)
Bit [30,32)  (padding)
Bit [32,46)  stride_byte_offset    K-dim stride in uint128_t units (4 LSBs excluded)
Bit [46,48)  version               Must be 1 for Blackwell
Bit [48,49)  (padding)
Bit [49,52)  base_offset           Usually 0
Bit [52,53)  lbo_mode              0=legacy, 1=next-buffer mode
Bit [53,56)  (padding)
Bit [56,61)  (padding)
Bit [61,64)  layout_type           Swizzle encoding (see below)
```

**Layout type encoding** (bits [61,64)):

| Value | Enum | CuTe Swizzle |
|---|---|---|
| 0 | SWIZZLE_NONE | `Swizzle<0,4,3>` (INTER) |
| 1 | SWIZZLE_128B_BASE32B | `Swizzle<2,5,2>` |
| 2 | SWIZZLE_128B | `Swizzle<3,4,3>` (SW128) |
| 4 | SWIZZLE_64B | `Swizzle<2,4,3>` (SW64) |
| 6 | SWIZZLE_32B | `Swizzle<1,4,3>` (SW32) |

### 3.7 Accumulate/Zero-Clear Semantics

- `scaleC` predicate: `p=1` → D = A×B + D (accumulate); `p=0` → D = A×B (zero-clear)
- In CuTe: `UMMA::ScaleOut::One` = accumulate, `UMMA::ScaleOut::Zero` = clear
- FlashMLA pattern: First K-tile uses `clear_accum=true`, subsequent K-tiles use `clear_accum=false`

### 3.8 Mask Array

- **cta_group::1**: `uint32_t mask[4]` — used for sparse MMA metadata (all zeros for dense)
- **cta_group::2**: `uint32_t mask[8]` — encodes peer SM destination for cross-SM sync

### 3.9 Measured UTCMMA Performance

| Config | K_DEPTH | cy/gemm | ns/gemm | cy/K-tile | Source |
|---|---|---|---|---|---|
| WS-TS m64n64 k=1 | 1 | 173.0 | 93.8 | 173.0 | utcmma-ver5 |
| WS-TS m64n128 k=1 | 1 | 173.0 | 93.8 | 173.0 | utcmma-ver5 |
| WS-TS m64n256 k=1 | 1 | 238.5 | 129.5 | 238.5 | utcmma-ver5 |
| WS-TS m64n128 k=4 | 4 | 334.0 | 181.1 | 83.5 | utcmma-ver5 |
| WS-TS m64n128 k=8 | 8 | 551.0 | 299.0 | 68.9 | utcmma-ver5 |
| WS-TS m64n128 k=16 | 16 | 1,013 | 549.5 | 63.3 | utcmma-ver5 |
| WS-TS m64n128 k=32 | 32 | 1,925 | 1,043.8 | 60.2 | utcmma-ver5 |
| SS m64n256 k=4 | 4 | 539 | 292 | 135 | utcmma-ver5 |
| Non-WS TS any N≤64 k=1 | 1 | 173.0 | — | 173.0 | utcmma-ver5 |
| Non-WS SS any N≤64 k=1 | 1 | 175.3 | — | 175.3 | utcmma-ver5 |

**Amortization model (WS-TS M64N128)**: `Total cycles = 56.4 × K_DEPTH + 118.5`
- Per-K-tile execution: 56.4 cy
- Fixed overhead (commit + mbarrier.wait + fence): 118.5 cy
- At K=32 (competition scale): 60.2 cy/tile = 32.6 ns/tile

**Key findings**:
- **N-independence**: Completion latency flat for all N ≤ 128 at K=1 (173 cy). Step up at N=256 (238.5 cy)
- **Swizzle-independence**: INTER/SW32/SW64/SW128 all identical performance
- **Multi-accumulator NULL**: N_ACC=1..5 all identical at K=32 — single-thread issuance serializes everything
- **M=32 vs M=64**: Only 6% faster at K=32 (1,814 vs 1,924 cy) → 1.6% end-to-end benefit, not worth complexity

---

## 4. TMA (Tensor Memory Access) — cp.async.bulk.tensor

### 4.1 Gather4 Mode

```
cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes
    .cta_group::{1|2}.L2::cache_hint [smem], [desc, {col, row0, row1, row2, row3}], [mbar], hint;
```

- Loads **4 arbitrary rows** from a 2D tensor map into contiguous SMEM
- Row indices are `int32_t` (watch for overflow with large tensors)
- **Prefetch variant**: `cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint` (no smem/mbar, just L2)
- **cta_group::2 variant**: Signals mbarrier across CTA pair. Uses `Sm100MmaPeerBitMask = 0xFEFFFFFF` (clears bit 20) for peer-SM mbarrier address

### 4.2 Measured TMA Gather4 Performance (Competition-Realistic, HBM-Cold)

Data from `tma-gather4-ver9.csv`, 1024 MB working set, random scatter:

| Config | Payload | N=1 (ns) | N=4 (ns) | N=16 (ns) | N=16 (GB/s) |
|---|---|---|---|---|---|
| ckv (int64, 512d) | 4,096 B | 650.9 | 311.0 | 234.5 | 17.47 |
| kpe (bf16, 64d) | 512 B | 609.9 | 298.0 | 229.3 | 2.23 |

**Key findings**:
- kpe latency ≈ ckv despite 8× smaller payload → **TMA is pipeline-overhead-dominated, not data-size-dominated**
- Deep pipelining N=1→16 gives **2.77× speedup** for random access
- Per-block (16 pipelined calls): **~3,752 ns** — the critical path bottleneck

### 4.3 TMA / GEMM Overlap (Competition Timing Model)

| Stage | ns | Source |
|---|---|---|
| TMA gather4 (16 ckv + 16 kpe, pipelined N=16) | **3,752** | tma-gather4-ver9 |
| QK ckv GEMM (M64N64, K=32) | 1,044 | utcmma-ver5 |
| QK kpe GEMM (M64N64, K=4) | 181 | utcmma-ver5 |
| SV GEMM ×2 (M64N256, SS, K=4 each) | 584 | utcmma-ver5 |
| **Total GEMM** | **1,809** | |
| **Per-block serial** | **5,561** | TMA + GEMM |
| **Pipelined steady-state** | **~3,752** | TMA-bound (TMA/GEMM = 2.07×) |

---

## 5. Synchronization — Fences, Commits, Barriers

### 5.1 tcgen05.fence

```
tcgen05.fence::before_thread_sync;   // Per-thread. Must precede __syncthreads() in TMEM-heavy code
tcgen05.fence::after_thread_sync;    // Per-thread. Must follow __syncthreads() in TMEM-heavy code
```

**When required**: Any time `__syncthreads()` is called while TMEM operations may be in flight. The fences ensure TMEM consistency across the barrier. Omitting them causes silent data corruption.

### 5.2 tcgen05.commit

```
tcgen05.commit.cta_group::{1|2}.mbarrier::arrive::one.shared::cluster[.multicast::cluster] [mbar_addr];
```

- Issued by MMA-producing thread after `tcgen05.mma`
- Signals that all pending MMA operations have committed their results to TMEM
- Consumer waits on the mbarrier to know TMEM data is valid
- **Multicast variant**: Signals mbarriers across multiple CTAs in cluster

### 5.3 Barrier Pattern (FlashMLA Decode)

```
Named barriers:
  main_loop_sync = 0   (128 threads)
  wg0_sync = 1         (warpgroup 0)
  wg0_warp02_sync = 2
  wg0_warp13_sync = 3
  everyone_sync = 4

Transactional barriers (mbarriers):
  bar_nope_ready[2]     — KV NoPE data ready in smem
  bar_rope_ready[2]     — KV RoPE data ready in smem
  bar_qk_done[2]        — QK GEMM complete (P in TMEM)
  bar_so_ready[2]       — S (softmax output) ready in smem
  bar_sv_done[2]        — SV GEMM complete (O in TMEM)
  bar_raw_ready[2], bar_raw_free[2] — Raw FP8 buffer double-buffering
  bar_valid_coord_scale_ready[4], bar_valid_coord_scale_free[4] — Index quad-buffering
```

### 5.4 Measured Sync Overhead

- UTCMMA completion sync overhead (commit + mbarrier.wait + fence): **~118.5 cycles** fixed cost
- `bulk_wait_group()` pipeline flush: **~500 cycles** at end of bulk S2G copy chain

---

## 6. Memory Hierarchy — Measured Latencies and Bandwidths

### 6.1 Cache Hierarchy (from p-chase sweep, ldg-pchase-sweep-ver1.csv)

| Working Set | Cycles | ns | Level |
|---|---|---|---|
| 4 KB | 36.0 | 18.3 | **L1 hit** |
| 8 KB | 36.8 | 18.7 | L1 hit |
| 16 KB | 40.2 | 20.4 | L1 spilling |
| 32 KB | 46.8 | 23.9 | L1/L2 boundary |
| 64 KB | 60.3 | 30.7 | L1→L2 transition |
| 128 KB | 87.3 | 44.5 | L1→L2 transition |
| 256 KB | 257 | 131 | L2 partial hit |
| 512 KB | 299 | 152 | **L2 steady-state** |
| 1 MB–32 MB | ~300 | ~153 | L2 plateau |
| 64 MB | 327 | 167 | L2 capacity pressure |
| 128 MB | 535 | 273 | L2→HBM transition |
| 256 MB | 707 | 360 | **HBM deep cold** |

### 6.2 L1 Bypass Penalty

| Working Set | Normal (cy) | L1-bypass (na_ef, cy) | Penalty |
|---|---|---|---|
| 4 KB | 36.0 | 278.5 | 7.7× |
| 8 KB | 36.8 | 298.6 | 8.1× |
| 32 KB | 46.9 | 278.0 | 5.9× |
| 512 KB+ | ~299 | ~299 | 1.0× |

**Finding**: L1-bypass only penalizes when working set fits in L1 (≤ ~32 KB). Beyond L1, zero difference.

### 6.3 Non-Coherent (.nc) Load Flag

`.nc` flag shows **zero performance difference** at ALL working set sizes (L1, L2, HBM). On B200, coherent loads are already optimized.

### 6.4 LDG/STG Cache Hints

From `ldg-hints-ver4.csv`:
- **Sequential 256 MB, 32 threads**: All 18 hint combinations → **1,070 ± 1 GB/s** (prefetcher dominates)
- **Competition 16.5 MB, 32 threads**: All hints → **1,564–1,568 GB/s** (L2-resident, hints irrelevant)
- **Bottom line**: For the competition workload, cache hints provide <0.3% variance. The data is L2-resident.

### 6.5 Bulk Copy S2G (cp.async.bulk Shared→Global)

From `bulk-copy-s2g-ver1.csv`:

| Size | Copy (ns) | Copy (GB/s) | Reduce_Add overhead |
|---|---|---|---|
| 512 B | 34.2 | 14.96 | +18.6% |
| 2 KB | 58.7 | 34.91 | +18.6% |
| 32 KB | 547.3 | 59.88 | +24.3% |
| 128 KB | 2,110.8 | 62.09 | +24.8% |

- **Fixed startup**: ~34 ns at 512B
- **Marginal BW**: ~15.9 GB/s single-CTA DMA ceiling
- **Async overlap**: Copy hides fully behind ≥300 cycle compute

### 6.6 Dual TMA Stream Interference

From `dual-tma-stream-ver1.csv`:
- Sequential 1024 MB: ckv +0%, kpe +7% interference → **minimal at HBM scale**
- Random 256 MB: kpe latency 2.85× when concurrent → **significant at L2-miss scale**

---

## 7. Sub-core Architecture and Warp Scheduling

### 7.1 Sub-core Partitioning (Measured)

From `warp-scheduler-ver1.csv`:

- **4 sub-cores per SM**, each with its own CUDA core pipeline
- **Mapping**: `scheduler_id = warp_id % 4`
- **Warp assignment** (8-warp CTA):
  - Sub-core 0: warps 0, 4
  - Sub-core 1: warps 1, 5
  - Sub-core 2: warps 2, 6
  - Sub-core 3: warps 3, 7

### 7.2 Scaling Behavior

| Active Warps | Sub-cores | Scaling | Efficiency |
|---|---|---|---|
| 1 | 1 | 1.00× | 100% |
| 2 (different sub-cores) | 2 | 2.00× | 100% |
| 4 (all sub-cores) | 4 | 4.00× | 100% |
| 8 (2 per sub-core) | 4 | 6.13× | 76.7% |
| 2 (same sub-core) | 1 | 1.53× | 76.7% |

**Key**: Perfect linear scaling up to 4 warps (one per sub-core). Two warps sharing a sub-core achieve only 1.53× (not 2×) due to pipeline interleaving.

### 7.3 FFMA Baseline

- Single-warp FFMA: **23.0 cycles/iteration** (baseline for 100K iterations)
- Same sub-core contention factor: **1.304×**

---

## 8. Warp Specialization Pattern (FlashMLA Decode)

From `config.h`: `NUM_THREADS = 128*3 = 384` (3 warpgroups, 12 warps)

**Warp Role Assignment** (inferred from kernel):

| Warp ID | Sub-core | Role |
|---|---|---|
| 0–3 | 0,1,2,3 | Warpgroup 0: Softmax / P extract / O rescale (128 threads) |
| 4 | 0 | MMA issuer (1 thread launches tcgen05.mma) |
| 5 | 1 | Raw KV producer (TMA load) |
| 6 | 2 | RoPE/dequant producer |
| 7 | 3 | Index + scale + valid_mask producer (LDG) |
| 8–11 | 0,1,2,3 | Warpgroup 2: FP8 dequantization (128 threads) |

**CUTLASS GEMM pattern** (from learn-cuda/02e_matmul_sm100):
```
Warp 0–3:   A/B TMA loaders + epilogue warps
Warp 4:     MMA computation warp (tcgen05.mma issuer)
Warp 5+1:   Epilogue loader
Warp 7:     Tile scheduler
```

---

## 9. Shared Memory Layout and Swizzle Atoms

### 9.1 UMMA Layout Atoms

CuTe provides canonical layout atoms for UMMA SMEM descriptors:

| Atom | Swizzle | Bytes | CuTe |
|---|---|---|---|
| `Layout_K_INTER_Atom<T>` | `Swizzle<0,4,3>` | 16B (no swizzle) | K-major interleaved |
| `Layout_K_SW32_Atom<T>` | `Swizzle<1,4,3>` | 32B | K-major 32B swizzle |
| `Layout_K_SW64_Atom<T>` | `Swizzle<2,4,3>` | 64B | K-major 64B swizzle |
| `Layout_K_SW128_Atom<T>` | `Swizzle<3,4,3>` | 128B | K-major 128B swizzle |
| `Layout_MN_INTER_Atom<T>` | `Swizzle<0,4,3>` | 16B | MN-major interleaved |
| `Layout_MN_SW128_Atom<T>` | `Swizzle<3,4,3>` | 128B | MN-major 128B swizzle |

**Swizzle K-atom divisibility rule**: SW128 requires K-dimension ≥ 64 elements for BF16 (128 bytes / 2 bytes per element). If K < 64, use SW64 or SW32.

### 9.2 FlashMLA Smem Layouts

From `config.h`:
- **Q (NoPE)**: `Layout_K_SW128_Atom<bf16>`, tiled to `Shape<B_H=64, D_NOPE_SW128=512>`
- **K (NoPE)**: `Layout_K_SW128_Atom<bf16>`, tiled to `Shape<B_H=64, 64×NUM_TILES>`
- **K (RoPE)**: `Layout_K_SW64_Atom<bf16>` (V32 mode) or `Layout_K_SW128_Atom<bf16>` (MODEL1)
- **S (softmax)**: `Layout_K_INTER_Atom<bf16>`, tiled to `Shape<B_H=64, B_TOPK=64>` — INTER because K=64 < SW128 threshold
- **O buffer**: `Layout_K_SW128_Atom<bf16>`, tiled to `Shape<B_H=64, D_V=512>`
- **O accum**: Custom stride=520 layout (avoids bank conflict: 520 vs 512 eliminates stride-of-power-of-2 conflicts)

### 9.3 make_umma_desc Construction

`make_umma_desc<Major>()` constructs a 64-bit `SmemDescriptor` from a CuTe smem tensor:
1. Recast tensor to `uint128_t` units
2. Set `version_ = 1` (Blackwell)
3. Detect layout type from swizzle pattern
4. Extract `start_address_ = smem_ptr >> 4`
5. Compute `stride_byte_offset_` and `leading_byte_offset_` from canonical layout strides

**Descriptor iteration**: `DescriptorIterator` advances descriptors by adding offset to lower 32 bits only (start address), preserving upper 32 bits (stride/layout metadata).

---

## 10. Cache Policies

### 10.1 Policy Creation

```
createpolicy.fractional.L2::evict_last.b64   reg, fraction;   // Keep data in L2
createpolicy.fractional.L2::evict_first.b64  reg, fraction;   // Evict data from L2 early
```

- `fraction`: Float in [0,1] controlling L2 residency priority
- Used as `L2::cache_hint` parameter in TMA and LDG/STG instructions

### 10.2 Competition Relevance

From LDG benchmark: All 18 hint combinations produce identical throughput for the competition working set (16.5 MB ≪ 126.5 MB L2). Cache hints are **not performance-differentiating** for this workload. Focus optimization effort elsewhere.

---

## 11. Vectorized FP32 Operations (SM100-Specific)

### 11.1 Packed f32x2 Instructions

| PTX | Operation | Operand Type |
|---|---|---|
| `add.f32x2 d, a, b` | d = a + b | uint64_t (packed 2×f32) |
| `mul.f32x2 d, a, b` | d = a × b | uint64_t (packed 2×f32) |
| `fma.rn.f32x2 d, a, b, c` | d = a×b + c | uint64_t (packed 2×f32) |

These are SM100-new. They pack two `float` values into a `uint64_t` register and execute both operations simultaneously. Used in the softmax loop for P scaling, O rescaling, and accumulator updates.

---

## 12. Cluster Launch Control (CLC) — Persistent Kernels

```
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [resp], [mbar];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, result;
clusterlaunchcontrol.query_cancel.get_first_ctaid::{x|y|z}.b32.b128 reg, result;
```

- Used for **persistent kernel** tile scheduling (FlashMLA decode uses this)
- CTA queries for new work tiles without re-launching kernels
- Response is 128-bit opaque value read from shared memory
- `.multicast::cluster::all` variant broadcasts to all CTAs in cluster

---

## 13. Key Architectural Constants

| Constant | Value | Source |
|---|---|---|
| `Sm100MmaPeerBitMask` | `0xFEFFFFFF` | Clears bit 20 for peer-SM mbarrier addressing |
| `Sm100MemDescDefault` | `0x1000000000000000` | Default TMA descriptor |
| TMEM `ColumnsPerAllocationSlice` | 32 | Minimum allocation unit |
| TMEM `Sm100TmemCapacityColumns` | 512 | Maximum columns |
| TMEM `DP_b` stride | `2^21` bits | Bit-addressing DP stride |
| SmemDescriptor `version_` | 1 | Blackwell (was 0 for Hopper) |
| MMA K extent | 256 bits | Fixed for all precisions |
| BF16 elements per K-tile | 16 | 256 bits / 16 bits |

---

## 14. PTX ISA Section References

| Topic | PTX Section |
|---|---|
| tcgen05.mma | 9.7.16.10.9 |
| MMA matrix layout | 9.7.16.10.2 |
| MMA swizzling | 9.7.16.10.6 |
| tcgen05.ld / tcgen05.st | 9.7.16.8.3 / 9.7.16.8.4 |
| tcgen05.fence | 9.7.16.11.1 |
| tcgen05.commit | 9.7.16.12.1 |
| tcgen05.alloc / dealloc | 9.7.16.7.1 |
| cp.async.bulk.tensor (TMA) | 9.7.9.25.5.2 |
| cp.reduce.async.bulk | 9.7.9.25.4.2 |
| createpolicy | 9.7.9.18 |
| Block scaling (MXF8F6F4) | 9.7.16.10.7 |
| Sparse MMA | 9.7.16.10.8 |

---

## 15. Colfax Research Tutorials

| Title | Key Content |
|---|---|
| [GEMM with Tensor Memory](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) | TMEM architecture, TS/SS modes, warp→lane mapping, descriptor construction, pipeline stages |
| [GEMM with Thread Block Clusters](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/) | cta_group::2, cluster shapes, DSM multicast, 2-SM MMA coordination |
| [Sub-byte GEMM](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/) | FP4/FP6/FP8 on Blackwell, runtime idesc, SMEM padding for sub-byte types |
| [Block-scaling](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/) | Block-scaled MMA with scale factors in TMEM |

---

## 16. Competition-Critical Derived Values

| Metric | Value | Derivation |
|---|---|---|
| KV data per query token | 2,048 × (512+64) × 2 B = **2.36 MB** | topk × (ckv+kpe) × sizeof(bf16) |
| Pages accessed per token | ~2,048 / 64 ≈ 32 | topk / page_size (assuming dense) |
| Blocks per token (B_TOPK=64) | 2,048 / 64 = **32** | topk / B_TOPK |
| Total blocks (num_tokens=8) | 8 × 32 = **256** | |
| Blocks per SM | 256 / 148 ≈ **1.73** | Not enough for full occupancy |
| Per-block latency (pipelined) | **~3,752 ns** | TMA-bound |
| Per-SM latency (2 blocks) | **~7,504 ns** | 2 × per-block |
| Kernel latency estimate (8 tokens) | **~7.5–9.5 μs** | Depends on load balancing |
| sparse_indices load (L2-resident) | **~36 cy = 18 ns** per 256-bit LDG | L1-hit path |
