# `kernel_utcmma_kdepth_ts` — Deep Dive: MMA_Atom, TiledMMA, Fragments, K-tiling

## What's different from `latency_ts`

`latency_ts` benchmarks the **RAW latency** of a *single* MMA tile (K=16, one `tcgen05.mma` instruction), using `ScaleOut::One` to chain ops and force serial dependency.

`kdepth_ts` benchmarks **real GEMM cost across K depth**: `K_TOTAL = K_DEPTH * 16`. For the competition kernel, `K_DEPTH=32` gives `K_TOTAL=512`, matching the ckv head dimension. This is what the kernel actually executes per iteration — not one MMA but K_DEPTH sequential MMAs reducing into one C.

---

## MMA_Atom — a compile-time type tag (no data)

```cpp
using MMA_Atom = SM100_MMA_F16BF16_WS_TS_NOELECT<
    bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>;
```

`MMA_Atom` is a **C++ type, not a value**. It carries zero runtime data. It encodes at compile time:
- Which PTX instruction to emit: `tcgen05.mma.ws.cta_group::1.kind::f16`
- Operand element types: bf16 A, bf16 B, float accumulator C
- Tile shape: one hardware tile is M×N (output), with K=16 consumed per tile
- Memory source for A: TMEM (T in TS)
- Memory source for B: shared memory (S in TS)
- Major order: both A and B are K-major

The CuTe framework templates on this type to generate the exact PTX at compile time — no branches, no virtual dispatch. The `MMA_Atom{}` expression constructs a zero-size instance just to pass the type to `make_tiled_mma`.

---

## TiledMMA — work partitioner

```cpp
auto tiled_mma = make_tiled_mma(MMA_Atom{});
```

`TiledMMA` answers: *given a larger M×N×K problem and multiple threads, who does what?*

For a WS (warp-specialized) atom in a single-CTA non-cluster kernel:
- The atom itself covers the full M×N tile
- There is exactly one "math" participant (thread 0)
- The tiling is trivial — one atom, one thread

For non-WS atoms tiled across 4 warps (like standard WMMA), the math would be different. But here `TiledMMA` mostly exists to provide the `partition_*` functions that carve up A, B, C consistently.

`tiled_mma.accumulate_` controls whether each `gemm` call does C=A×B (`Zero`) or C+=A×B (`One`).

---

## The three fragments — why they're created so differently

```
Fragment     | Source    | How created                          | Has K-tile dim?
-------------|-----------|--------------------------------------|----------------
tA_frag      | TMEM      | make_fragment_A(partition_shape_A()) | YES
tC_frag      | TMEM      | partition_fragment_C()               | NO
sB_frag      | smem      | partition_fragment_B(sB)             | YES
```

### `tC_frag` — simplest: just a TMEM column address

```cpp
auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
tC_frag.data().get() = TMEM_COL_C;  // = 0
```

C has **no K dimension** — it's the M×N output that every K-tile reduces *into*. Its shape is just `(M, N)` in MMA atom units. For TMEM, the "data" is a 32-bit TMEM column address. `TMEM_COL_C=0` means "accumulator starts at column 0."

This fragment is passed unchanged to every `gemm` call in the K loop — the hardware always writes to the same TMEM columns.

```
tC_frag shape: (M_per_atom, N_per_atom)   ← no k dimension
                [TMEM col addr = 0]
```

### `sB_frag` — smem-backed: K-tile descriptors precomputed from tensor

```cpp
auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);  // [N × K_TOTAL] in smem
auto sB_frag = thr_mma.partition_fragment_B(sB);
```

B lives in shared memory at a **real pointer**. `partition_fragment_B` takes the actual `sB` tensor and walks it: for each K-tile `k`, it computes the 64-bit smem descriptor encoding `smem_B + k * K_PER_TILE` with the correct swizzle embedded. All K_DEPTH descriptors are precomputed into registers before the loop.

```
sB_frag shape: (N_per_atom, 1, K_DEPTH)
               [desc_k0, desc_k1, ..., desc_kN]   ← all in registers

sB_frag(_, _, 0) → 64-bit descriptor pointing to smem_B[0..16]
sB_frag(_, _, 1) → 64-bit descriptor pointing to smem_B[16..32]
...
sB_frag(_, _, k) → 64-bit descriptor pointing to smem_B[k*16..(k+1)*16]
```

This is why `partition_fragment_B` takes the tensor: it needs the smem address to precompute K-tile descriptors.

### `tA_frag` — TMEM-backed: column address with K-tile stride

```cpp
auto tA_frag = thr_mma.make_fragment_A(
    partition_shape_A(tiled_mma, Shape<Int<M>, Int<K_TOTAL>>{}));
tA_frag.data().get() = TMEM_COL_A;  // = 256
```

A lives in TMEM, which has no pointer — it's addressed by **column number**. There's no smem tensor to pass to `partition_fragment_A`, so the API is different:

1. `partition_shape_A(tiled_mma, Shape<M, K_TOTAL>{})` — computes the logical shape of A as seen by thread 0 (the full M×K_TOTAL for WS mode), in units of MMA atoms: `(M_per_atom, K_per_atom, K_DEPTH)`.
2. `make_fragment_A(...)` — allocates a fragment of that shape, but with *uninitialized* data (the TMEM address).
3. `.data().get() = TMEM_COL_A` — sets the **base** TMEM column. CuTe knows the column stride per K-tile from the MMA atom's TMEM layout. When you access `tA_frag(_, _, k)`, it returns a fragment with TMEM address = `TMEM_COL_A + k * stride_cols`.

```
tA_frag shape: (M_per_atom, 1, K_DEPTH)
               [TMEM col 256, TMEM col 256+s, ...]   ← column addresses with stride s

tA_frag(_, _, 0) → reads TMEM cols 256..256+s-1
tA_frag(_, _, 1) → reads TMEM cols 256+s..256+2s-1
...
```

**Why the asymmetry vs sB?** TMEM is not pointer-addressed. You can't call `make_smem_ptr` for TMEM. The fragment is a placeholder that the hardware resolves to TMEM column addresses at execution time.

---

## `(_, _, k)` indexing in CuTe

CuTe tensors support multi-mode indexing. `_` (a.k.a. `cute::Underscore`) means "all elements in this mode," like Python's `:`. So:

```
tA_frag(_, _, k)   → mode-0: all M elements
                     mode-1: all K-in-atom elements
                     mode-2: k-th K-tile
                     result: a sub-tensor (the k-th K-tile slice of A)

sB_frag(_, _, k)   → same idea: k-th K-tile slice of B

tC_frag            → has no third mode, passed whole
```

These slices are what get passed to `cute::gemm` as its A, B, C arguments. CuTe then emits the `tcgen05.mma` PTX with the TMEM column address from the A slice and the smem descriptor from the B slice.

---

## The K-reduction loop

```cpp
tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;   // ← C = A×B (initialize)
CUTE_UNROLL
for (int k = 0; k < K_DEPTH; ++k) {
    gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One; // ← C += A×B (accumulate) for all remaining k
}
```

- k=0: `ScaleOut::Zero` initializes C (zeroes old accumulator)
- k=1..K_DEPTH-1: `ScaleOut::One` accumulates into the same C

This contrasts with `latency_ts` which always used `ScaleOut::One` to create artificial RAW chains. Here `ScaleOut::Zero` on k=0 means the first tile *doesn't* depend on old C, but subsequent k's do depend on k-1's C write. This creates a realistic RAW dependency chain that matches what the actual kernel does.

```
Per outer iteration (iters loop):
  k=0: C_TMEM = A[0..16) × B[0..16)           ← fresh write (Zero)
  k=1: C_TMEM += A[16..32) × B[16..32)         ← waits for k=0 write (One)
  k=2: C_TMEM += A[32..48) × B[32..48)         ← waits for k=1 write
  ...
  k=31: C_TMEM += A[496..512) × B[496..512)
```

This measures: does latency scale linearly with K_DEPTH? If ~11 cycles/tile, K_DEPTH=32 should take ~352 cycles total.

---

## Swizzle auto-selection for larger K_TOTAL

```cpp
constexpr int SWIZZLE_B = (K_TOTAL >= 64) ? 128 : (K_TOTAL >= 32 ? 64 : 32);
```

The swizzle atom must evenly divide the K_TOTAL. SW128 has K-atom=64 (64 bf16 = 128 bytes), SW64→32 elems, SW32→16 elems. Choosing the largest swizzle that fits maximizes bank conflict avoidance:

```
K_TOTAL=16  → SW32  (K-atom=16 fits exactly once)
K_TOTAL=32  → SW64  (K-atom=32 fits exactly once)
K_TOTAL=64+ → SW128 (K-atom=64, tiles K_TOTAL/64 times)
```

---

## Memory picture with K_DEPTH=4

```
Shared memory (B operand: [N=128, K_TOTAL=64] bf16, SW128 layout):

  K=0..15       K=16..31      K=32..47      K=48..63
  ┌───────────┬──────────────┬─────────────┬────────────┐
  │  k=0 tile │   k=1 tile   │  k=2 tile   │  k=3 tile  │ row 0
  │  (16 bf16)│              │             │            │
  ├───────────┴──────────────┴─────────────┴────────────┤
  │            N=128 rows, SW128 swizzled               │
  └──────────────────────────────────────────────────────┘

sB_frag precomputes 4 smem descriptors (one per K-tile) into registers.


TMEM (A operand: [M=64, K_TOTAL=64] bf16, columns 256..255+stride*4):

  col 256     col 256+s     col 256+2s    col 256+3s
  ┌──────────┬─────────────┬─────────────┬─────────────┐
  │ A k=0    │ A k=1       │ A k=2       │ A k=3       │ lane 0
  │ (M=64    │             │             │             │ ...
  │  rows)   │             │             │             │ lane 127
  └──────────┴─────────────┴─────────────┴─────────────┘

tA_frag base = col 256, CuTe adds k*s per tile index.


TMEM (C accumulator: [M=64, N=128] float32, columns 0..255):

  ┌──────────────────────────────────────────────────────┐
  │ C[M=64, N=128] in float32                           │
  │ Accumulated across all 4 K-tiles                    │
  └──────────────────────────────────────────────────────┘
```

---

## Bug (FIXED): swizzle label mismatch in CSV output

**`run_utcmma_latency_ts`** hardcoded `"SW128"` in the CSV, but `kernel_utcmma_latency_ts` always uses `SW=32` (K=16 requires K-atom=16, SW32 is the only valid choice). Fixed to `"SW32"`.

**`run_utcmma_kdepth_ts`** hardcoded `"SW128"` for all k_depth values, but the kernel auto-selects:
```
k_depth=1  (K_TOTAL=16):  SW32  ← was mislabeled SW128
k_depth=2  (K_TOTAL=32):  SW64  ← was mislabeled SW128
k_depth≥4  (K_TOTAL≥64):  SW128 ← was correct
```

Fixed by computing the swizzle label at runtime in main.cu to match the kernel's selection logic exactly. Cycle counts were always correct — only the CSV metadata was wrong — but incorrect metadata corrupts downstream analysis and agent conclusions.

---

## Relationship to `utcmma_ts` in helpers.cuh

The kdepth kernel's inner loop is essentially what `kerutils::utcmma_ts()` does:

```cpp
// helpers.cuh — the production helper used in the real FlashMLA kernel
void utcmma_ts(tiled_mma, tA_frag, sB, tC_frag, clear_accum) {
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        tiled_mma.accumulate_ = One;
    }
}
```

The benchmark inlines this logic and adds the clock64 timing around it. The `size<2>(tA_frag)` is exactly `K_DEPTH` — the number of K-tiles in the fragment's third mode.

---

## Key takeaways

| Concept | Meaning |
|---|---|
| `MMA_Atom` | Compile-time type tag — encodes which PTX instruction and operand layout. Zero runtime cost. |
| `TiledMMA` | Wraps the atom + knows how to partition A/B/C across threads. For WS mode, thread 0 owns everything. |
| `tC_frag` | TMEM column pointer for M×N output. No K dimension — same C address for all K-tiles. |
| `sB_frag` | K_DEPTH smem descriptors in registers, one per K-tile. Precomputed from real smem pointer. |
| `tA_frag` | TMEM column pointer with K-tile stride. Created from shape (not tensor) because TMEM isn't pointer-addressed. |
| `(_, _, k)` | CuTe slice: "all values in modes 0,1, k-th tile in mode 2." Selects one K-tile from a fragment. |
| `ScaleOut::Zero` on k=0 | Initializes C fresh each outer iteration. Subsequent K-tiles use `One` to accumulate. |
