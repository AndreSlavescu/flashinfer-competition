# `kernel_utcmma_latency_ts` — Line-by-Line Walkthrough

## What it's measuring

**TS = TMEM × Shared → TMEM**. This is the QK GEMM path in FlashMLA: A (query) lives in TMEM, B (key) lives in shared memory, C (output scores) lives in TMEM. We measure how long one `tcgen05.mma` takes end-to-end.

---

## Kernel signature

```cpp
template<int M, int N>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_latency_ts(BenchResult* result, int iters)
```

- `M`, `N` — tile dimensions (e.g., M=64, N=64 or N=128). K is always 16 (one bf16 MMA tile = 256 bits).
- `__launch_bounds__(128, 1)` — exactly 128 threads per block, 1 block per SM. Prevents the compiler from spilling registers.
- Only **1 block** is launched; only **thread 0** does the actual MMA work.

---

## Step 1 — Shared memory setup

```cpp
static constexpr int K = 16;
extern __shared__ char smem_raw[];
__shared__ __align__(16) uint32_t smem_tmem_addr;
```

`smem_raw` is the raw shared memory buffer. `smem_tmem_addr` will hold the TMEM base pointer (a `uint32_t` TMEM column address) written by the allocator.

```cpp
auto layout_B = ku::make_umma_canonical_k_major_layout<N, K, 32, bf16>();
bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);
```

This creates the **CuTe layout** for the B operand (K matrix) in shared memory.

### What is a CuTe layout?

A CuTe layout is `(shape, stride)` — it maps a logical (row, col) index to a flat memory offset. Here the logical shape is `[N × K]`.

`make_umma_canonical_k_major_layout<N=64, K=16, SW=32, bf16>()` produces a **K-major** layout with **SW32 swizzle**.

**Why K-major?** The UMMA hardware expects B in K-major form (K is the contiguous dimension, i.e., row-major if you think of `[N, K]`). This matches how you'd store transposed keys.

**Why SW32 swizzle?** The swizzle prevents shared memory bank conflicts when the tensor core reads 128-byte chunks. SW32 has a K-atom of 16 elements (32 bytes / 2 bytes per bf16), which exactly fits `K=16`. SW64 or SW128 need K≥32 or K≥64.

```
B in shared memory: logical [N=64, K=16] bf16
                    physical layout with SW32 swizzle

  K dim (contiguous) ─────────────────────────►
  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐  row 0
  │                                               │
  ├──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┤  row 1
  │                    XOR swizzled               │
  ├──────────────────────────────────────────────...┤
  │                   (N rows total)              │
  └──────────────────────────────────────────────...┘  row N-1

  Each row = 16 bf16 = 32 bytes = 1 "K-atom"
  SW32 swizzle XORs the row index into the column offset
  to rotate which bank each row starts at → no conflicts
```

---

## Step 2 — TMEM allocation

```cpp
if (threadIdx.x < 32) {
    TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
    TMEM::Allocator1Sm().release_allocation_lock();
}
__syncthreads();
```

TMEM (Tensor Memory) is a **per-SM** scratchpad separate from shared memory. It has 512 columns × 128 lanes × 32 bits = 256 KB per SM. The allocator requires a full warp (32 threads) to call the PTX allocation instruction.

After this, `smem_tmem_addr` contains `0` — the base TMEM column address. The kernel uses:

```
TMEM layout (512 columns total):
col 0                   col 255  col 256          col 383   col 384  col 511
├────── C (output) ─────────────┤──── A (input) ──────────┤─ unused ─────────┤
  256 cols for [M×N] accumulators   128 cols for [M×K] Q tile
```

---

## Step 3 — TiledMMA construction

```cpp
using MMA_Atom = SM100_MMA_F16BF16_WS_TS_NOELECT<
    bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>;
auto tiled_mma = make_tiled_mma(MMA_Atom{});
```

`SM100_MMA_F16BF16_WS_TS_NOELECT` is a **compile-time descriptor** of the `tcgen05.mma.ws.cta_group::1.kind::f16` PTX instruction for the **warp-specialized** (WS) TS variant.

- `bf16, bf16` — A and B element types
- `float` — accumulator type (C in TMEM)
- `M, N` — tile dimensions
- `UMMA::Major::K, UMMA::Major::K` — both A and B are K-major

`make_tiled_mma` wraps the atom into a `TiledMMA` object that knows how to partition work across threads. In the WS (warp-specialized) variant, there's only one "math warp" — thread 0 issues all the MMA instructions.

```
TiledMMA for M=64, N=64, K=16 (TS):

  Logical problem:         After tiling:
  A: [M=64, K=16] TMEM    → 1 MMA atom (hardware does full 64×16 at once)
  B: [N=64, K=16] smem    → 1 MMA atom
  C: [M=64, N=64] TMEM    → 1 accumulator tile
```

---

## Step 4 — Smem descriptor for B

```cpp
auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);
UMMA::SmemDescriptor desc_B = UMMA::make_umma_desc<UMMA::Major::K>(sB);
```

`make_tensor` wraps the raw `bf16*` pointer with the CuTe layout to make a **typed tensor view** — you can now index it as `sB(row, col)`.

`make_umma_desc` converts this tensor into a 64-bit hardware descriptor that encodes:
- The smem address
- The matrix dimensions
- The swizzle mode

This descriptor is passed directly to the `tcgen05.mma` instruction in the register file.

---

## Step 5 — Fragment creation

```cpp
auto thr_mma = tiled_mma.get_slice(_0{});
```

`get_slice(_0{})` gets the "per-thread view" for thread index 0. Since this is warp-specialized with a single math warp, thread 0 owns the whole problem.

```cpp
auto tA_frag = thr_mma.make_fragment_A(
    partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
tA_frag.data().get() = TMEM_COL_A;   // = 256
```

`tA_frag` is the **A fragment** — a logical view of A's data. For TS MMA, A lives in TMEM, so the "fragment" is really just a **TMEM column address** (not actual register data). Setting `.data().get() = 256` tells the hardware "read A starting at TMEM column 256."

```
tA_frag: not registers, just a TMEM column pointer
          ┌─────────────┐
          │ col_addr=256 │  → hardware reads M=64 rows × K=16 cols from TMEM
          └─────────────┘
```

```cpp
auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
tC_frag.data().get() = TMEM_COL_C;   // = 0
```

Same idea for C (the output accumulator) — stored in TMEM starting at column 0.

```cpp
auto sB_frag = thr_mma.partition_fragment_B(sB);
```

B is in shared memory, so `sB_frag` holds the **smem descriptor** (a 64-bit value encoding the smem address + swizzle) in registers. This is what gets passed to `tcgen05.mma` as the B operand.

---

## Step 6 — Accumulate mode

```cpp
tiled_mma.accumulate_ = UMMA::ScaleOut::One;
```

`ScaleOut::One` means **C += A×B** (accumulate into existing C).
`ScaleOut::Zero` means **C = A×B** (initialize C, discarding old values).

Setting `One` here creates a **RAW (Read-After-Write) dependency** on the TMEM accumulator: each MMA must wait for the previous one to finish writing C before it can read C back. This is exactly what we want for **latency measurement** — chained dependent ops, not parallel pipelined ones.

```
Iteration 0:  C = A×B                    (if ScaleOut::Zero)
Iteration 1:  C += A×B  ← depends on iter 0's C write
Iteration 2:  C += A×B  ← depends on iter 1's C write
...
```

---

## Step 7 — Warmup + timed loop

```cpp
for (int i = 0; i < 100; i++) {
    gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
}
```

100 warmup iterations to:
- Bring B into L1/L2 cache
- Let the GPU reach steady-state clock frequency
- Allow the hardware scheduler to "learn" the instruction pattern

The `(_, _, 0)` indexing selects K-tile index 0. Since K=16 and there's only 1 K-tile, this is the only tile.

```cpp
uint64_t c_start = clock64_start();
for (int i = 0; i < iters; i++) {
    gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
}
uint64_t c_end = clock64_stop();
```

`clock64` reads the SM's cycle counter. The timed loop runs `iters` chained MMAs. Reported latency = `(c_end - c_start) / iters` cycles per MMA.

Each `gemm(...)` call compiles down to a single `tcgen05.mma.ws.cta_group::1.kind::f16` PTX instruction.

---

## Step 8 — Result and cleanup

```cpp
result->total_cycles = c_end - c_start;
result->gt_start_ns  = gt_start;
result->gt_end_ns    = gt_end;
```

Writes raw cycles + wall-clock ns to host-visible memory. Host divides by `iters` to get per-op latency.

```cpp
if (threadIdx.x < 32) {
    TMEM::Allocator1Sm().free(0, 512);
}
```

TMEM must be explicitly freed (full-warp call) or it leaks for the lifetime of the SM context.

---

## Full data flow diagram

```
Host memory
    │
    │  result* (write-only)
    ▼
┌─────────────────────────────────────────────────────┐
│                    SM (1 CTA, 128 threads)           │
│                                                      │
│  Shared Memory                                       │
│  ┌──────────────────────────────┐                   │
│  │ smem_B: [N×K] bf16, SW32    │ ← B (keys)        │
│  │  (never written in kernel,   │                   │
│  │   pre-filled or garbage,     │                   │
│  │   we only time the MMA)      │                   │
│  └──────────────────────────────┘                   │
│            │ SmemDescriptor (64-bit)                 │
│            │ in register of thread 0                 │
│            ▼                                         │
│  ┌──────────────────────────────────────────────┐   │
│  │         tcgen05.mma  (UTCMMA unit)           │   │
│  │   A: TMEM[col=256..383] → [M×K] bf16        │   │
│  │   B: smem_B descriptor  → [N×K] bf16        │   │
│  │   C: TMEM[col=0..255]   → [M×N] float32     │   │
│  │                                              │   │
│  │   C += A × Bᵀ  (one ~11-cycle operation)    │   │
│  └──────────────────────────────────────────────┘   │
│            │                                         │
│            ▼                                         │
│  TMEM (per-SM, 256KB)                               │
│  ┌──col 0────────col 255──col 256────col 383──┐      │
│  │  C accumulator [M×N]  │  A input [M×K]    │      │
│  │  float32              │  bf16              │      │
│  └───────────────────────┴───────────────────┘      │
│                                                      │
│  thread 0 reads clock64 before/after iters MMAs     │
│  writes (total_cycles, gt_start_ns, gt_end_ns)      │
│  to result*                                          │
└─────────────────────────────────────────────────────┘
```

---

## Key takeaways

| Concept | Detail |
|---|---|
| Only thread 0 does MMA | WS (warp-specialized) MMA is issued by a single thread; other 127 threads are just there for the TMEM allocator |
| A/C are TMEM column pointers | Not registers — the fragment `.data().get()` is literally a column index into the TMEM address space |
| B is a smem descriptor | A 64-bit register encoding address + swizzle, passed directly to `tcgen05.mma` |
| `accumulate_ = One` forces serial chain | Each MMA depends on previous output — measures RAW latency (~11 cycles on B200) not throughput |
| SW32 swizzle for K=16 | SW64/SW128 need bigger K-atoms; SW32 fits exactly one K-tile of 16 bf16 elements |
