# MLA_CUTE_BREAKDOWN.md

A detailed walk through every CuTe layout transformation, tiler, tiled MMA, and TMA atom used in

`references/cutlass/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py`

(4373 lines, Blackwell SM100 / B200, FP16 Multi-Head Latent Attention decode with paged KV-cache).

All line numbers below refer to that file.

---

## 0. How to read this doc

### 0.1 Pinned worked-example config

Whenever you see a number in `[square brackets]`, it is the value under this config:

| Knob | Value | Defined at |
|------|-------|------------|
| `mma_qk_tiler_mn` | `(128, 256)` | constructor arg, e.g. line 4336 in the test harness |
| `mma_pv_tiler_mn` | `(128, 256)` | constructor arg |
| `page_size` | `64` | constructor arg |
| `latent_dim` | `512` (fixed) | line 172 |
| `rope_dim` | `64` (fixed) | line 173 |
| `cluster_shape_mnk` | `(2, 1, 1)` (fixed) | line 184 |
| `q_dtype` / `k_dtype` / `v_dtype` | `Float16` | kernel is FP16 everywhere |
| `acc_dtype` | `Float32` | line 174 |

### 0.2 CuTe layout primer — with concrete numbers

**A layout is a pair `(shape, stride)`**. It's a function from a coordinate tuple to a linear offset:

```
layout = (shape=(4, 8), stride=(8, 1))
   coord (i, j) → offset = i*8 + j*1

element (0, 0) → offset 0
element (0, 7) → offset 7
element (1, 0) → offset 8
element (3, 7) → offset 31
```

This is a plain 4×8 row-major matrix. `size(layout)` = 4·8 = 32 elements. `cosize(layout)` = max offset + 1 = 32 bytes of contiguous backing memory.

**Nested shapes.** A single mode can itself be a tuple. The layout

```
shape  = ((2, 2), 8)
stride = ((16, 8), 1)
```

is **still** a 4×8 matrix (the outer shape is `4·1=4` by `8`), but the first dim is viewed as `(inner, outer) = (2, 2)` with distinct strides `(16, 8)`. So:

```
coord ((a, b), j) → offset = a*16 + b*8 + j*1
((0, 0), 0) → 0
((1, 0), 0) → 16    ← inner index a steps by 16
((0, 1), 0) → 8     ← outer index b steps by 8
```

Nested modes exist precisely so that you can say "I'm partitioning the 4 rows into 2 groups of 2, and I want independent stride for each". Every `*_divide` operation produces nested modes.

**Column-major vs row-major** is just the stride:
- Row-major `M×N`: `shape=(M, N)`, `stride=(N, 1)`.
- Column-major `M×N`: `shape=(M, N)`, `stride=(1, M)`.

**Tensor = Iterator (pointer) + Layout.** `cute.make_tensor(ptr, layout)` is just bundling the two.

### 0.3 CuTe tiler primer

A **tiler** is a shape tuple you hand to one of the `_divide` operations to cut a big layout into sub-tiles. Example: apply tiler `(2, 4)` to a layout of shape `(4, 8)`:

- `cute.logical_divide((4, 8), (2, 4))` → `((2, 2), (4, 2))` — each mode is split into `(tile_size, rest)`.
- `cute.tiled_divide((4, 8), (2, 4))` → `((2, 4), (2, 2))` — tiles grouped first, rest after.
- `cute.flat_divide((4, 8), (2, 4))` → `(2, 4, 2, 2)` — zipped form flattened.
- `cute.zipped_divide((4, 8), (2, 4))` → `((2, 4), (2, 2))` — same structure as tiled_divide.

Section 4 below gives a concrete worked example of each, with strides.

### 0.4 What are "MMA atoms" and "TMA atoms"?

- **MMA atom** = one tcgen05 matrix-multiply instruction. It has a fixed `(M, N, K)` per issue and a built-in thread/warp layout (`thr_id`).
- **TMA atom** = one bulk tensor-memory-accelerator copy descriptor. It has a tile shape and a swizzle.
- **Tiled MMA / Tiled TMA** = the atom plus a cluster/thread layout that says how to dispatch multiple atom issues across the cluster.

---

## 1. Static knobs of this kernel (lines 172–272)

### 1.1 Fixed constants

| Field | Value | Line | Meaning |
|-------|-------|------|---------|
| `latent_dim` | `512` | 172 | MLA latent head dim |
| `rope_dim` | `64` | 173 | RoPE head dim (also used as QK MMA's K tile) |
| `cluster_shape_mnk` | `(2, 1, 1)` | 184 | 2-CTA cluster splitting the M dim |
| `use_2cta_instrs` | `True` | 185 | enables `CtaGroup.TWO` tcgen05 MMAs |
| `warps_in_n` | `2` | 188 | see §3 |
| `num_compute_warps` | `4` | 189 | warps doing softmax |
| `threads_per_warp` | `32` | 190 | always 32 |

### 1.2 Warp roles (lines 213–230)

12 warps × 32 threads = **384 threads per CTA**:

```
warp  0-3  → compute      (softmax + P cast-down)
warp  4-7  → correction   (rescale + O writeback)
warp  8    → mma          (issues every tcgen05 gemm)
warp  9    → load_tma     (Q/K/V TMA loads)
warp 10    → load_pt      (page-table async-copy)
warp 11    → empty        (keeps register budget free for other warps)
```

Register budgets (233–235): compute=192, correction=208, everyone else=96.

### 1.3 Pipeline stage counts (line 261–267)

| Stage | Value | Purpose |
|-------|-------|---------|
| `load_q_stage` | 1 | Q TMA buffer (one-shot at prologue) |
| `load_kv_stage` | **15** | K/V TMA depth — hides variable paged-KV latency |
| `mma_s_stage` | 2 | S ping-pong TMEM staging |
| `p_mma_stage` | 2 | P ping-pong SMEM staging |
| `p_cor_stage` | 2 | softmax-stats ping-pong |
| `mma_o_stage` | 1 | single O staging |
| `load_pt_stage` | 4 | page-table index prefetch |

### 1.4 TMEM offsets (269–272)

A single TMEM pointer is carved into 3 regions:

```python
tmem_o_offset            = mma_s_stage * mma_qk_tiler[1] // warps_in_n  # [= 2*256/2 = 256]
correction_factor_offset = tmem_o_offset + latent_dim // warps_in_n     # [= 256+256 = 512]
```

```
tmem_ptr
  ├─ [0,   256)   S (QK result)      : 2 stages × 128 cols
  ├─ [256, 512)   O (PV result)      : 512/2 = 256 cols
  └─ [512, …)     correction metadata: row_max, row_sum, correction_factor
```

---

## 2. All the derived variables, defined

This section exists because the kernel pre-computes a dozen tile/iteration variables in `__init__` and `_setup_attributes` and then uses them everywhere. Memorize what each one *is* before trying to read the layout code.

### 2.1 Input variables (from constructor)

| Name | Value [pinned] | Meaning |
|------|----------------|---------|
| `mma_qk_tiler_mn` | `(128, 256)` | user-supplied (M, N) of the QK MMA tile |
| `mma_pv_tiler_mn` | `(128, 256)` | user-supplied (M, N) of the PV MMA tile |
| `page_size` | `64` | elements per KV-cache page |

### 2.2 Derived MMA-tile shapes (lines 191–206)

```python
mma_qk_tiler_k  = rope_dim                              # [= 64]         (line 191)
mma_qk_tiler    = (mma_qk_tiler_mn[0],
                   mma_qk_tiler_mn[1],
                   mma_qk_tiler_k)                      # [= (128, 256, 64)]   (192-196)
mma_qk_rope_tiler = (mma_qk_tiler_mn[0],
                     mma_qk_tiler_mn[1],
                     rope_dim)                          # [= (128, 256, 64)]   (197-201)
mma_pv_tiler    = (mma_pv_tiler_mn[0],
                   mma_pv_tiler_mn[1],
                   mma_qk_tiler[1] * mma_qk_tiler[2]
                       // mma_pv_tiler_mn[1])           # [= (128, 256, 64)]   (202-206)
```

**Why the weird formula for `mma_pv_tiler[2]`?** It is `(qk_N * qk_K) / pv_N` = `(256 * 64) / 256` = `64` under the pinned config. The design invariant is

```
    pv_K * pv_N  ==  qk_N * qk_K
```

i.e. the "accumulator footprint" of the PV MMA matches that of the QK MMA. This is what lets the S accumulator in TMEM be consumed as the P input to PV without reshuffling — S covers `qk_M × qk_N`, and after softmax it is re-tiled as `(pv_M, pv_K)` chunks fed through `iterations_pv_k` issues.

### 2.3 Derived iteration counts (lines 207–211)

```python
iterations_qk_latent = latent_dim    // mma_qk_tiler[2]   # [= 512 / 64 = 8]
iterations_qk_rope   = mma_qk_tiler_k // mma_qk_tiler[2]  # [= 64 / 64 = 1]
iterations_qk        = iterations_qk_latent + iterations_qk_rope   # [= 9]
iterations_pv_k      = mma_qk_tiler[1] // mma_pv_tiler[2] # [= 256 / 64 = 4]
iterations_pv_n      = latent_dim      // mma_pv_tiler[1] # [= 512 / 256 = 2]
```

**Plain English:** the QK GEMM walks a 576-wide K dim (512 latent + 64 rope) in 9 MMA issues of K=64 each. The PV GEMM walks qk_N=256 as K in 4 issues of pv_K=64, and covers latent_dim=512 as the output dimension in 2 N-blocks of pv_N=256.

### 2.4 CTA tilers — actual code

The CTA tiler is what one CTA of the 2-CTA cluster owns. The cluster splits only the M dim, not N or K:

```python
# lines 2483–2487
cta_qk_tiler = (
    self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],   # 128 / 2 = 64
    self.mma_qk_tiler[1],                                # 256
    self.mma_qk_tiler[2],                                # 64
)                                                        # [= (64, 256, 64)]

# lines 2730–2734
cta_pv_tiler = (
    self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],   # 128 / 2 = 64
    self.mma_pv_tiler[1],                                # 256
    self.mma_pv_tiler[2],                                # 64
)                                                        # [= (64, 256, 64)]
```

So: CTA0 owns M rows `[0, 64)` of the MMA tile, CTA1 owns M rows `[64, 128)`; both see the full N and K.

### 2.5 `thr_id.shape` — what it is, concretely

Every tiled MMA object carries a layout `thr_id` that encodes **how threads/CTAs cooperate to produce one MMA atom issue**. For B200 2-CTA tcgen05 atoms:

- `qk_tiled_mma.thr_id.shape == 2` (under `CtaGroup.TWO`)
- `cute.size(qk_tiled_mma.thr_id.shape) == 2`

This value equals `cluster_shape_mnk[0]` here, because the cluster's 2 CTAs cooperate on one MMA tile, and each contributes 1 leader thread that issues the instruction. The hardware (tensor cores + TMEM) handles the rest.

You can verify by tracing line 429:

```python
kc_page_tile_size = min(
    self.page_size,                                      # 64
    qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape,   # 128 / 2 = 64
)
```

and line 1622:

```python
cta_n = self.mma_pv_tiler[1] // v_params.tiled_mma_pv.thr_id.shape  # 256 / 2 = 128
```

So `cta_m = mma_qk_tiler[0] // thr_id.shape = 64` and `cta_n = mma_pv_tiler[1] // thr_id.shape = 128`. These values show up inside `make_paged_tiled_tma_atom` (677) and the TMA smem-layout builders (434, 461).

### 2.6 Named variables reference card

| Name | Formula | Value [pinned] | First use |
|------|---------|----------------|-----------|
| `mma_m` (QK) | `mma_qk_tiler[0]` | 128 | line 193 |
| `mma_n` (QK) | `mma_qk_tiler[1]` | 256 | line 194 |
| `mma_k` (QK) | `mma_qk_tiler[2]` | 64 | line 195 |
| `cta_m` (QK) | `mma_qk_tiler[0] // thr_id.shape` | 64 | line 429 / 434 |
| `cta_n` (PV) | `mma_pv_tiler[1] // thr_id.shape` | 128 | line 1622 |
| `kc_page_tile_size` | `min(page_size, cta_m)` | 64 | line 428 |
| `vc_page_tile_size` | `min(page_size, mma_pv_tiler[2])` | 64 | line 458 |
| `iterations_qk_latent` | `latent_dim // mma_k` | 8 | line 207 |
| `iterations_qk` | `8 + 1` | 9 | line 209 |
| `iterations_pv_k` | `mma_n / pv_K` | 4 | line 210 |
| `iterations_pv_n` | `latent_dim / pv_N` | 2 | line 211 |

---

## 3. `warps_in_n` and the 4-compute-warp layout

This is the most confusing piece of this kernel, so it gets its own section.

### 3.1 What the comment says (lines 186–188)

Verbatim from the source:

```
# When using 2 CTAs with m=128: warps 0-1 handle accumulation for first half [0, n/2),
# while warps 2-3 handle accumulation for second half [n/2, n)
self.warps_in_n = 2
```

So within a single CTA of the cluster:

- **Warps 0 and 1** both cover **N in [0, n/2)** (for pinned `n = 256` → N columns `[0, 128)`).
- **Warps 2 and 3** both cover **N in [n/2, n)** (N columns `[128, 256)`).
- Warps `(0, 2)` hold the **same M rows**, and together cover the full N.
- Warps `(1, 3)` hold **a different set of M rows**, and together cover the full N.

### 3.2 What the code shows (lines 2560–2568)

The softmax row-max reduction confirms the pairing:

```python
if cutlass.const_expr(self.warps_in_n == 2):
    common_params.smem_exchange[tidx] = row_max_new
    self.softmax_exchange_sync_bar.wait()
    row_max_new = cute.arch.fmax(
        row_max_new,
        common_params.smem_exchange[
            (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
        ],
    )
```

`tidx` runs 0..127 (4 warps × 32 threads). Thread `tidx` reads from thread `tidx + 64`. Since warps are laid out contiguously (warp 0 = tidx [0,32), warp 1 = [32,64), warp 2 = [64,96), warp 3 = [96,128)), the exchange pairs:

- warp 0 ↔ warp 2 (each thread with the same-column thread 64 tidx later)
- warp 1 ↔ warp 3

This is exactly how you exchange a partial row-max computed over half-N with the partial from the other half-N, to get the full-N row-max.

### 3.3 Corrected ownership diagram

For one QK MMA tile of shape `(mma_m=128, mma_n=256, mma_k=64)`:

```
                ┌────────────────────────────────────────────┐
                │    MMA tile: M=128, N=256                 │
                └────────────────────────────────────────────┘
                               │  cluster_shape_mnk = (2, 1, 1)
                ┌──────────────┴──────────────┐
                ▼                             ▼
           CTA 0                          CTA 1
           rows M[0,   64)                rows M[64, 128)

         Inside one CTA (say CTA 0, rows [0, 64)):

           N[0, 128)              N[128, 256)
         ┌───────────┬────────────┬───────────┐
         │ warp 0    │ warp 1     │            │     <- rows M subset A  (warp 0 cols N[0,128), warp 2 cols N[128,256))
         │           │            │   warp 2   │
         │ rows      │ rows       │            │     <- rows M subset B  (warp 1 cols N[0,128), warp 3 cols N[128,256))
         │ M subset A│ M subset B │   warp 3   │
         └───────────┴────────────┴───────────┘

         softmax row-max exchange (tidx ↔ tidx+64):
           warp 0 <-> warp 2         (combine N halves over subset A)
           warp 1 <-> warp 3         (combine N halves over subset B)
         via  softmax_exchange_sync_bar  (barrier_id=2, line 244)
```

The exact M-subset split between warps 0 and 1 (and between warps 2 and 3) is determined by the MMA atom's own thread→coord mapping and is *not* something the kernel code chooses directly.

---

## 4. Every layout primitive, with numerical examples

Each subsection below shows: the CuTe signature, a toy input layout with explicit shape+stride, the output, and where it appears in `mla_decode_fp16.py`.

For toy inputs I use a layout named `A = (shape=(4, 8), stride=(8, 1))` — row-major 4×8 matrix, 32 elements — so you can eyeball the arithmetic.

### 4.1 `cute.make_layout(shape, stride=...)`

Raw constructor. If `stride` is omitted, it defaults to column-major compact strides.

```
cute.make_layout((4, 8))                → shape=(4,8), stride=(1,4)   column-major
cute.make_layout((4, 8), stride=(8, 1)) → shape=(4,8), stride=(8,1)   row-major
cute.make_layout(16)                    → shape=(16,), stride=(1,)    1D
cute.make_layout((2, 1, 1))             → shape=(2,1,1), stride=(1,0,0)   cluster layout (line 399)
```

Used at: lines 399 (cluster layout), 905 / 909 / 912 (SMEM page-table layouts), 1305 / 1314 / 1357 (register buffers), 1949–1951 (O iteration layout), 2605–2614 (P matrix 2-mode view).

### 4.2 `cute.select(layout, mode=[...])`

Pick / reorder / drop modes.

Input: `A` with `shape=(4, 8)`, `stride=(8, 1)`.

```
cute.select(A, mode=[0])    → shape=(4,),  stride=(8,)
cute.select(A, mode=[1])    → shape=(8,),  stride=(1,)
cute.select(A, mode=[1, 0]) → shape=(8, 4), stride=(1, 8)   ← TRANSPOSED
```

Used at:
- **Line 367** `cute.select(c_latent.layout, mode=[1, 0, 2])` — transposes the latent tensor to build `c_latent_transpose` without copying memory, just reinterpreting the strides.
- **Lines 475, 484, 536, 539** `mode=[0, 1, 2]` — drop the trailing stage mode of a 4-mode staged layout before handing it to a TMA atom.
- **Lines 494, 512** `mode=[0]` — keep only the first partition after a `tiled_divide`.
- **Lines 1935, 1943, 2474, 2488, 2736** `mode=[0, 1]` — reduce a `(M, N, K)` tiler to `(M, N)`.
- **Line 679** `mode=[0, 2]` — inside `make_paged_tiled_tma_atom`, keeps the CTA and rest-N axes after `flat_divide`.

### 4.3 `cute.logical_divide(layout, tiler)`

Partitions *each mode independently* and keeps mode structure. For 1D: divides one axis into `(tile, rest)`. Use `None` to leave a mode alone.

Input `A = (shape=(4, 8), stride=(8, 1))`:

```
cute.logical_divide(A, (2, 4))
    → shape  = ((2, 2), (4, 2))      first-mode split: 4 = 2*2,  second-mode split: 8 = 4*2
      stride = ((8, 16), (1, 4))

cute.logical_divide(A, (None, 4))
    → shape  = (4, (4, 2))           leave first mode alone
      stride = (8, (1, 4))

cute.logical_divide(A, (None, None, None, 8))
    → error — A has only 2 modes; this would need a 4-mode input
```

The 4-mode form is how the kernel uses it at lines **411** and **448**:

```python
q_latent_smem_layout_staged = cute.logical_divide(
    q_latent_smem_layout_staged,
    (None, None, None, self.iterations_qk_latent),   # [= (None, None, None, 8)]
)
```

Here `q_latent_smem_layout_staged` already has 4 modes `(M, K, stage, ?)` of total stage count `iterations_qk_latent * load_q_stage = 8 * 1 = 8`. The `None, None, None, 8` leaves the first 3 modes untouched and splits the 4th mode of size 8 into `(8, 1)` — i.e. re-labels it as `(latent_iteration, stage_within_iteration)`.

**Concrete before/after** with some made-up strides (the actual ones are set by `make_smem_layout_a`, we don't need them to see the shape transformation):

```
before:   shape = (128, 64, 1, 8)           stride = (64, 1, ?, 8192)
after:    shape = (128, 64, 1, (8, 1))      stride = (64, 1, ?, (8192, 0))
```

The 4th mode is now nested so you can index `layout[..., (latent_it, stage_it)]`.

### 4.4 `cute.tiled_divide(layout, tiler)`

Like `logical_divide` but groups all tile modes together at the front and all rest modes at the back.

Input `A = (shape=(4, 8), stride=(8, 1))`, tiler `(2, 4)`:

```
cute.tiled_divide(A, (2, 4))
    → shape  = ((2, 4), 2, 2)
      stride = ((8, 1), 16, 4)

Structure: ((tile_m, tile_n), rest_m, rest_n)
```

Contrast with `logical_divide(A, (2, 4))` which gave `((2, 2), (4, 2))` — same information, different nesting.

Used at:
- **Line 398** `cute.tiled_divide(cute.make_layout(cluster_shape_mnk), (thr_id.shape,))` — the `cluster_shape_mnk = (2, 1, 1)` layout is divided by `thr_id.shape = 2` to produce the `cta_layout_vmnk` (virtual-MNK) layout used by every pipeline factory.
  ```
  before: shape=(2, 1, 1), stride=(1, 0, 0)
  after:  shape=((2,), 1, 1, 1), stride=((1,), 0, 0, 0)
  ```
- **Lines 438–440** — splits the KC TMA-flavored SMEM layout by `(page_tile_size=64, mma_k=64)`.
- **Lines 465–470** — splits the VC TMA-flavored layout by `(atom_n, vc_page_tile_size)`.

### 4.5 `cute.flat_divide(layout, tiler)`

Same as `tiled_divide` but with the tile-mode tuple **flattened**. Handy when you want a clean N-dimensional coordinate space instead of a nested tile/rest pair.

Input `A = (shape=(4, 8), stride=(8, 1))`, tiler `(2, 4)`:

```
cute.flat_divide(A, (2, 4))
    → shape  = (2, 4, 2, 2)
      stride = (8, 1, 16, 4)

Structure: (tile_m, tile_n, rest_m, rest_n)
```

Used at:
- **Line 678** inside `make_paged_tiled_tma_atom`. Given `g_tile = cute.composition(identity, (N, K))` (shape `(N, K, rest_m)` = e.g. `(256, 64, …)`), then `cute.flat_divide(g_tile, (cta_mn,))` partitions the first mode by `cta_mn = 64`, giving `(64, 4, K, rest_m)` — per-CTA strips.
- **Line 1538** `cute.flat_divide(mQL, mma_qk_tiler_mk)` — partition Q-latent gmem by `(M, K) = (128, 64)`.
- **Line 1621** `cute.flat_divide(mCLT, (mma_pv_tiler[1], page_tile_size))` — partition V gmem by `(N=256, page=64)`.

### 4.6 `cute.zipped_divide(layout, tiler)`

Identical to `tiled_divide` for a 2-mode input: groups tile modes into one tuple, rest modes into another. The difference only matters when the *tiler* itself has nested modes that need to be paired with specific axes.

Input `A = (shape=(4, 8), stride=(8, 1))`, tiler `(2, 4)`:

```
cute.zipped_divide(A, (2, 4))
    → shape  = ((2, 4), (2, 2))
      stride = ((8, 1), (16, 4))
```

Used **exactly once** in the file, at **line 685**, to finish the paged-TMA layout setup. After line 679 we have `cta_v_map` with shape `(cta_mn, rest_N, rest_M)` = e.g. `(64, 4, …)`. Then:

```python
cta_v_map = cute.zipped_divide(
    cta_v_map,
    (page_tile_size, mma_tiler[1])           # (64, 256)
)
```

pairs the `page_tile_size=64` with the first mode (cta_mn=64) and `mma_tiler[1]=256` with the second/third modes, producing a layout whose outer tuple is `(page_tile, mma_n)` and inner is "which page" — exactly what the TMA descriptor wants.

### 4.7 `cute.composition(outer, inner)`

Functionally composes two layouts: `composition(f, g)` is the layout whose coord-to-offset map is `f(g(coord))`.

The important case here is when `outer` is an **identity layout** and `inner` is a **tiler tuple**. That "lays a tiler over an identity", producing a coordinate layout grouped by the tiler.

Input `I = make_identity_layout((16, 16))` (16×16 coords), tiler `(4, 4)`:

```
cute.composition(I, (4, 4))
    → shape  = ((4, 4), (4, 4))
      maps  (i, j) → ((i % 4, j % 4), (i // 4, j // 4))
```

Used at **line 676** as step 1 of `make_paged_tiled_tma_atom`:

```python
ident  = cute.make_identity_layout(gmem.shape)
g_tile = cute.composition(ident, mma_tiler)
```

This turns the gmem coordinate space into one that is explicitly "tile-major", ready for `flat_divide` to chop further.

### 4.8 `cute.make_composed_layout(outer, offset, inner)` + swizzles

`make_composed_layout` is the "compose a swizzle with a layout" operation. The `outer` argument is typically a `cute.Swizzle`, and `inner` is the base layout.

A **swizzle** `cute.make_swizzle(B, M, S)` is a bit-XOR transformation of linear offsets that scrambles addresses to avoid SMEM bank conflicts. Conceptually:

```
Given offset o, view its bits:  [upper | S-bit segment | B-bit segment | lower M bits]
Output: XOR the B-bit segment into the S-position segment.
```

Used **exactly once** at **lines 2622–2625** to re-swizzle the P matrix for PV MMA reads:

```python
sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
sP_mk_view = cute.make_tensor(
    sP_wo_swizzle_iter,
    cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
)
```

Under the pinned config, `swizzle_bits = log2(64*16/8/32) + 1 = log2(4) + 1 = 3`, `swizzle_base = 3` (FP16). The `0` is an additive offset in bytes (zero here).

### 4.9 `cute.group_modes(layout, start, end)`

Flattens a *contiguous range* of modes into a single nested tuple. Inverse of "split this mode back out".

Input `shape=(2, 3, 4, 5)`:

```
cute.group_modes(layout, 0, 3)
    → shape = ((2, 3, 4), 5)         group modes [0..3), leave mode 3 alone
    → cosize unchanged; just re-buckets the coord tuple.
```

Used at **lines 1584, 1585, 1592, 1593** to prepare operands for `cpasync.tma_partition`:

```python
tQsQ, tQLgQL_mkl = cpasync.tma_partition(
    qk_params.tma_atom_q_latent,
    0,
    cute.make_layout(1),
    cute.group_modes(qk_params.sQ, 0, 3),   # 4-mode sQ → 2-mode ((a,b,c), stage)
    cute.group_modes(tSgQL,       0, 3),
)
```

`tma_partition` expects its tensor arguments in the form `(atom_payload, rest)` — a 2-mode tuple. The `group_modes(…, 0, 3)` collapses the 3 "atom payload" modes into one tuple, leaving the stage mode as-is.

### 4.10 `cute.append(layout, extra)`

Append a new trailing mode.

```
cute.append((shape=(4,), stride=(1,)), make_layout(8))
    → shape = (4, 8), stride = (1, ?)
```

Used at:
- **Line 1938** `cute.append(tStS_shape, mma_s_stage)` — tack a 2-stage dim onto the S TMEM layout.
- **Lines 1947–1953** — append an `iteration` mode of size `latent_dim // pv_N = 2` to the O TMEM layout, with **stride = `pv_N // warps_in_n = 128`**. This is how iterating the outer `acc_stage` loop (lines 2163) advances to the next 128-column block of O in TMEM.
- **Lines 2477, 2708** — same pattern, re-indexing S / O for different consumer passes.

### 4.11 `cute.local_tile(tensor, cta_tiler, coord)`

Slice out the per-CTA tile of a global tensor at a given `(M_tile, N_tile, batch)` coord.

Concept:

```
gO = cute.local_tile(mO, (64, 256), (m_idx, n_idx, batch_idx))
   → equivalent to mO[m_idx*64:(m_idx+1)*64, n_idx*256:(n_idx+1)*256, batch_idx]
   → but returns a layout view, no copy
```

Used at:
- **Lines 2740–2779** — carve out the CTA-local O tile from `mAccO` (split-KV intermediate) and from `mO` (final output) for the epilogue.
- **Lines 2998–3033** — same for LSE writeback.

### 4.12 Worked end-to-end: paged TMA layout setup (lines 666–704)

Let me walk `make_paged_tiled_tma_atom(..., mma_tiler=(256, 64), is_k_load=True)` with actual numbers. Input: gmem `c_latent` of shape `(SeqLen_K, latent_dim=512, batch)`. We want a TMA descriptor that fetches one page of K-latent (64 rows × 64 cols) per issue.

```python
# (a) line 675 — start from an identity layout over gmem
ident  = cute.make_identity_layout(c_latent.shape)
       # shape = (SeqLen_K, 512, batch)   stride = (1, SeqLen_K, SeqLen_K*512)

# (b) line 676 — compose with the MMA tiler (256, 64)
g_tile = cute.composition(ident, (256, 64))
       # Coords are now "tile-major": the first 2 modes iterate inside one (256,64) tile,
       # the rest iterate over which tile you're on.

# (c) line 677 — cta_mn = 256 / 2 = 128 (N for V would be different; here N=256)
cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape     # 256 / 2 = 128

# (d) line 678 — flat_divide the first mode by cta_mn
cta_v_map = cute.flat_divide(g_tile, (128,))
          # shape becomes (128, 2, 64, rest_latent, batch)
          # First 128: rows inside a per-CTA strip
          # Next 2:    which of the 2 CTAs owns this strip
          # Then 64:   K cols
          # Then rest: which (256,64) tile you're on, batch

# (e) line 679 — select modes [0, 2], dropping the "which CTA" and "which tile" axes
cta_v_map = cute.select(cta_v_map, mode=[0, 2])
          # shape becomes (128, 64)   — one CTA-strip of one MMA tile

# (f) lines 680–683 — page tile size
page_tile_size = min(page_size, cta_mn) = min(64, 128) = 64

# (g) lines 685–688 — zipped_divide so we can index (page_id, within_page)
cta_v_map = cute.zipped_divide(cta_v_map, (64, 256))
          # shape = ((64, 256), (2, 1))   — pairs (page_tile, N) with (num_pages, rest)
          # Nesting is essential: the outer tuple is "one atomic TMA tile",
          # the inner tuple is "how many such tiles make up the full data".

# (h) line 689 — keep only the atomic-tile mode
cta_v_map = cute.select(cta_v_map, mode=[0])
          # shape = (64, 256)   — the "non-exec" template the TMA descriptor wires up

# (i) lines 692–698 — MLIR lowering into a non-exec TMA descriptor
```

The final object is a TMA atom whose gmem-side tile shape is `(64, 256)` but whose *base pointer* is not fixed at descriptor-creation time. At runtime the load warp fills the free coord with a page index (from the page-table), turning one descriptor into a per-page gather.

---

## 5. Derivation rules and design patterns — *why* each operation exists

Up to here this doc has been "here are the operations, here's what each one looks like." This section explains **how you would have derived these layouts yourself** — the mental model, the decision rules, and the canonical patterns. Sources: NVIDIA's `media/docs/cpp/cute/02_layout_algebra.md`, `03_tensor.md`, `0t_mma_atom.md`, `0x_gemm_tutorial.md`, and Lei Mao's "CuTe Layout Algebra" write-up.

### 5.1 The mathematical foundation: a layout *is* a function

> Layouts are functions from integers to integers.
> — `02_layout_algebra.md`, §Coalesce

That one sentence is the whole model. `layout(coord) = offset` and that's it. A shape+stride pair is one way to represent such a function; CuTe also supports nested `((shape), (stride))` representations, which just make the same function cheaper to store when it has structure. Two layouts that produce the same `(coord → offset)` map are considered identical.

Consequences:

- Composition `A ∘ B` is just `A(B(coord))`.
- Any divide / tile / partition is *also* just a composition dressed in specific arguments.
- The `cosize` (maximum offset + 1) tells you how much backing memory to allocate; the `size` (domain cardinality) tells you how many logical elements exist.

### 5.2 Composition is the only primitive you really need

`02_layout_algebra.md` defines all higher-level ops from three primitives:

- **Composition** `composition(A, B)` = `A ∘ B`.
- **Concatenation** `(A, B)` = stack two layouts as separate modes.
- **Complement** `complement(A, M)` = find a layout that fills the holes in `A` up to cosize `M`.

Everything else is sugar:

```
logical_divide(A, B) = composition(A, (B, complement(B, size(A))))
logical_product(A, B) = (A, composition(complement(A, size(A)*cosize(B)), B))
```

So if you ever forget what `logical_divide` does: it composes `A` with a layout whose first mode picks out the `B`-tile and whose second mode walks through "everything else." The first mode is literally `composition(A, B)` and the second mode is "the repetition of B across `A`'s domain."

### 5.3 Unpacking `logical_divide` with the MLA-Q example

Take Q SMEM layout staging (`q_latent_smem_layout_staged`, line 411):

```python
before: shape = (128, 64, 1, 8)                            # 4 modes: (M, K, load_q_stage, iters)
tiler:  (None, None, None, 8)                              # "leave modes 0-2 alone; split mode 3 by 8"
after:  shape = (128, 64, 1, (8, 1))                       # mode 3 is now nested (tile, rest)
```

What just happened mathematically? CuTe took mode 3 (size 8) and composed it with `(8:1, complement(8:1, 8)) = (8:1, 1:0)`. The tile `8` pulls out the full range, and `complement` says "there's nothing left" (only 1 rest, with stride 0). Mode 3 is now a 2-tuple `(iter, stage)` where you can index separately. The elements in memory are unmoved.

**Why do this at all?** Because later, line 2093 does

```python
qk_params.tSrQ[None, None, k_block, q_stage]              # indexes the latent-iteration axis
```

If mode 3 were still a flat `8`, you couldn't name "the latent iteration axis" independently of the stage axis — they'd be a single combined integer.

### 5.4 The 4 divide variants: same operation, just re-nested

From `02_layout_algebra.md`:

```
Input Layout Shape : (M, N, L, ...)
Input Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM, RestM), (TileN, RestN), L, ...)
zipped_divide  : ((TileM, TileN), (RestM, RestN, L, ...))
tiled_divide   : ((TileM, TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

They are **all the same underlying function**. The difference is purely how the modes are grouped. The decision tree:

```
Do you want the "Rest" modes grouped together as one coord?
   yes → zipped_divide                    (→ use for local_tile / local_partition)
   no  → do you want Tile modes grouped together as one coord?
          yes → tiled_divide              (→ use when Rest has named axes you index separately)
          no  → do you want 0 grouping at all?
                  yes → flat_divide       (→ use right before tma_partition / when you want a clean N-D grid)
                  no  → logical_divide    (→ use when each mode's (tile, rest) semantics matters independently)
```

**Rule of thumb**: 
- `logical_divide` preserves semantics (mode-M is still mode-M).
- `zipped_divide` is for the inner/outer partition trick (§5.6). `local_tile` and `local_partition` are defined on top of it.
- `tiled_divide` is the workhorse when your tensor's rest modes are meaningful (batch, k_tile iteration).
- `flat_divide` is for "I just want an N-D array of tiles, no nesting."

### 5.5 When to reach for which divide — concrete rules from this kernel

| Intent | Pick | From this file |
|--------|------|----------------|
| "Split mode M, keep M-semantics" | `logical_divide(A, (None, …, tile, …))` | Line 411 (Q staging split), 448 (P staging split) |
| "Split for `local_tile` / `local_partition`" (one tile per CTA / per thread) | `zipped_divide` (usually via `local_tile`) | Line 2740 (O gmem tile), implicit in `local_tile` calls |
| "Split, and Rest modes need to stay indexable" | `tiled_divide` | Line 398 (cluster → CTAs), 438 (KC layout → pages), 465 (VC layout → pages) |
| "Flatten everything into a plain N-D grid of tiles" | `flat_divide` | Line 678 (paged TMA gmem tiling), 1538 (Q gmem tiling), 1621 (V gmem tiling) |

Diagnostic question: **after the divide, how will the result be indexed?** If the next line is a single `[coord, _]` slice pulling out "one tile," you probably wanted zipped/tiled. If the next line is `[None, None, None, iter]` naming one axis, you wanted `logical_divide`. If the next line feeds a `tma_partition`, you often want `flat_divide`.

### 5.6 The partitioning pattern — inner vs outer vs TV

From `03_tensor.md`, §Partitioning a Tensor:

> Partitioning is tiling and/or composition followed by slicing.

After `zipped_divide(A, tiler)` → `((tile), (rest))`:

- **inner_partition** = slice into `rest` → keep **one tile's worth of elements** (used when many parallel agents each own one tile, e.g. CTAs). Alias: `local_tile`.
- **outer_partition** = slice into `tile` → keep **one element from every tile** (used when many parallel agents each own one element per tile, e.g. threads within a CTA). Alias: `local_partition`.
- **TV-partition** = use a hand-crafted `(thread, value) → (coord)` layout as the *composition* target; slice on the thread axis. Used for MMA atoms with irregular thread-data mappings.

**This is the design pattern** behind everything in a CUTLASS kernel:

```
CTA level:   local_tile(global_tensor, cta_tiler, cta_coord)
Thread level: local_partition(cta_tile, thread_layout, threadIdx)  OR
              thr_copy.partition_S / thr_copy.partition_D  (for structured copies)
              thr_mma.partition_A / .partition_B / .partition_C   (for MMA atoms)
```

MLA uses all three: `local_tile` for the O tile (2740), `cpasync.tma_partition` for TMA (1580–1638), and `tiled_mma.make_fragment_A/B/C` (1928–1932) for MMA.

### 5.7 When you would `append` vs `logical_divide` vs `group_modes`

These three are the "reshape-without-touching-memory" ops. They cover disjoint needs:

- **`cute.append(layout, extra)`** — add a *new* mode that didn't exist before. You need it when you want to replicate a layout into a staging buffer or a loop iteration. Example (line 1938):
  ```python
  tStS_staged_fake = tiled_mma_qk.make_fragment_C(
      cute.append(tStS_shape, self.mma_s_stage)   # [append a 2-stage axis]
  )
  ```
  Intuition: the MMA atom's C fragment shape has no notion of "which ping-pong stage." You *need* a new axis, not a split of an existing one.

- **`cute.logical_divide(layout, (…, tile, …))`** — split an *existing* mode into `(tile, rest)`. Use when the existing mode is already sized right but you want to name sub-axes. Example: Q staging (line 411) already has 8 stages worth of memory — we split that size-8 mode into `(iter=8, stage=1)` tuple.

- **`cute.group_modes(layout, start, end)`** — *collapse* a range of existing modes into a single nested tuple. Use when a downstream function wants a 2-mode `(atom, rest)` form but your layout has more modes. Example: `tma_partition` takes exactly 2 modes, so we call `group_modes(sQ, 0, 3)` at line 1584 to fold 4 modes into 2.

Mnemonic:

```
append       — I need a new axis that doesn't exist yet.
logical_divide — I need to name sub-axes of an existing axis.
group_modes  — I need to hide axes by bundling them into a tuple.
```

### 5.8 The canonical GEMM layout flow (and how MLA deviates)

From `0x_gemm_tutorial.md`, here is the pattern every CuTe GEMM follows. Memorize this — every flash-attention / MLA / sparse-attention kernel is a variation of it.

```
STEP                                            CuTe ops
-----------------------------------------------  --------------------------------------------------
1. Define problem shape and strides              make_tensor(ptr, (M,N,K), strides)
2. Define CTA tiler                              make_shape(bM, bN, bK)
3. Slice the CTA's gmem tile                     local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>)
                                                   → (bM, bK, k_iter)     # the k_iter mode is the
                                                                          #  loop variable in the mainloop
4. Declare SMEM layouts                          make_layout((bM, bK), stride=...)   # with the
                                                                                     # right swizzle
5. Allocate SMEM with cosize(smem_layout)        __shared__ TA smem[cosize_v<Layout>]
6. Wrap SMEM as a tensor                         make_tensor(make_smem_ptr(smem), smem_layout)
7. Build TiledCopy / TiledMMA                    make_tiled_copy(atom, thr_layout, val_layout)
                                                 make_tiled_mma(atom, warp_layout, tiler)
8. Get per-thread views                          thr_copy.partition_S(gA)  → (CPY, M_cpy, K_cpy, k_iter)
                                                 thr_copy.partition_D(sA)  → (CPY, M_cpy, K_cpy)
                                                 thr_mma.partition_A(sA)   → (MMA, M_mma, K_mma)
                                                 thr_mma.partition_B(sB)   → (MMA, N_mma, K_mma)
                                                 thr_mma.partition_C(gC)   → (MMA, M_mma, N_mma)
9. Allocate the C accumulator fragment           make_fragment_C(tCgC)       → in registers
10. Mainloop                                     for k in range(k_iter):
                                                     cute.copy(copy_atom, tAgA(_,_,_,k), tAsA)
                                                     cute.gemm(mma, tCsA, tCsB, tCrC)
11. Epilogue                                     cute.copy(tCrC, tCgC)
```

**Where MLA deviates:**

- *Step 4*: SMEM layouts are **staged** (extra dim for pipelining) and **multi-iteration** (extra dim for latent × 8, etc.). The `logical_divide` trick at lines 411 and 448 is how you add the iteration axis on top of a staged buffer that `make_smem_layout_a` produced as a single flat stage axis.
- *Step 7*: `TiledMMA` comes from `sm100_utils.make_trivial_tiled_mma`, which already bakes in the B200 tcgen05 atom and the 2-CTA cluster layout.
- *Step 9*: `make_fragment_C` is retargeted to **TMEM** via `make_tensor(tmem_ptr, fragment.layout)` (line 1941, 1954). This is Blackwell-specific — on Hopper and earlier, accumulators are in registers.
- *Step 10*: the loop is warp-specialized; the "copy" step is a TMA descriptor built via the paged template (§4.12), and `cute.copy` is called with a runtime-substituted gmem coord.

### 5.9 Reading the naming convention

From `0x_gemm_tutorial.md`:

> The naming convention `tAsA` is pretty typical across CuTe and CUTLASS. This is read as "Partitioning pattern `tA` applied to tensor `sA`".

All those cryptic names in `mla_decode_fp16.py` become pronounceable once you know this:

| Symbol | Decoding |
|--------|----------|
| `sQ`, `sKC`, `sVC`, `sP` | **S**MEM tensors for Q, K-C, V-C, P |
| `gA`, `gO` | **G**MEM tensors |
| `tQsQ` | TMA-**Q** partition applied to **s**MEM-**Q** |
| `tQLgQL` | TMA-**Q**-**L**atent partition applied to **g**MEM-**Q**-**L**atent |
| `tSrQ` | MMA-**S** partition applied to **r**egister-**Q** (fragment from SMEM-Q) |
| `tOrP`, `tOrVC` | MMA-**O** partition applied to **r**egister-**P** / **r**egister-**V**-**C** |
| `tStS` | MMA-**S** partition applied to **t**mem-**S** |
| `tOtO` | MMA-**O** partition applied to **t**mem-**O** |
| `tTR_tS` | **T**MEM-load **R** partition applied to **t**mem-**S** |
| `tTR_rAcc` | **T**MEM-load **R** partition applied to **r**egister-accumulator |
| `rCor` | **r**egister correction-factor |

Rule: **first letter block** = the partitioning pattern (which copy/MMA), **lowercase letter after** = memory space (`s`=smem, `g`=gmem, `r`=register, `t`=tmem), **capital suffix** = tensor identity.

### 5.10 A derivation workflow — how would you build this from scratch?

If you were writing your own Blackwell attention kernel using MLA as the template, you'd work in this order. This is the same order the code appears in `__call__` and `_setup_attributes`:

```
1. Pick your MMA atom(s).
   - Choose M, N, K from the tcgen05 atom's valid shape space.
   - Note the atom's ThrID (tells you cta_m = atom_m // thr_id.shape).

2. Pick your CTA tiler.
   - cta_qk_tiler = mma_qk_tiler with M divided by cluster_shape_mnk[0].
   - Enforce relations between QK and PV tilers so accumulators can be reused
     (the pv_K · pv_N == qk_K · qk_N invariant in §2.2).

3. Decide iteration counts.
   - iterations_qk_latent = latent_dim / mma_k, etc.
   - These drive the number of gemm issues inside each warp's inner loop.

4. Decide pipeline stages.
   - Long-latency loads (paged KV) → deep pipeline (15).
   - Short-latency MMA ping-pong → shallow pipeline (1 or 2).
   - The MMA warp is the consumer; the TMA warp is the producer.

5. Build staged SMEM layouts.
   - Call make_smem_layout_a / _b with the atom and total-stage-count.
   - If you have nested iterations (e.g. latent × stage), use logical_divide
     to split the flat stage axis into (iter, stage).

6. Build TMA-flavored SMEM layouts (only if the TMA tile != MMA tile).
   - Smaller per-issue tile (e.g. page-sized) gets its own layout,
     then tiled_divide by the ratio to match the MMA buffer.

7. Build TMA atoms.
   - Dense case: make_tiled_tma_atom_A (or _B).
   - Paged case: custom function that does
       composition(identity, mma_tiler) →
       flat_divide(_, (cta_mn,)) →
       select(mode=[0, 2]) →
       zipped_divide(_, page_tile_sizes) →
       select(mode=[0]) →
       MLIR lower into a non-exec tma atom.

8. Build TMEM accumulator tensors.
   - partition_shape_C(mma_tiler_mn) gives per-thread C shape.
   - append(staging_count) to add a ping-pong axis.
   - make_tensor(tmem_ptr + offset, layout) to land it in TMEM.

9. Write the load warp.
   - tma_partition(atom, 0, make_layout(1), group_modes(smem, 0, N),
                   group_modes(gmem, 0, N)) → (tXsX, tXgX).
   - Mainloop: cute.copy(atom, tXgX[... page_idx ...], tXsX[stage]).

10. Write the MMA warp.
    - make_fragment_A / _B from SMEM tensors.
    - Loop: cute.gemm(mma, C, A_frag, B_frag, C); accumulate via
      tcgen05.Field.ACCUMULATE.

11. Write the compute/softmax warp.
    - Load C from TMEM via a tmem_load_atom tiled copy.
    - Do the math in registers.
    - Cast-down and write P back to SMEM (with re-swizzle if needed).

12. Write the correction/epilogue warp.
    - Load O from TMEM.
    - Apply split-KV correction factor.
    - Write final output to gmem via local_tile + cute.copy.
```

The key design decisions happen at steps 1–4. Once those are fixed, steps 5–8 are mechanical applications of the patterns in §5.5–5.7.

### 5.11 Intuition cheat sheet for each operation

| Operation | When you would reach for it |
|-----------|----------------------------|
| `composition(A, B)` | "I want to view `A` through the access pattern of `B`." Used to build TV-layouts, re-swizzle, and lay tilers over identity layouts. |
| `logical_divide(A, tiler)` | "I want to name sub-axes of existing axes without copying data." Perfect when each mode's semantics should stay distinct. |
| `zipped_divide(A, tiler)` | "I want `(tile, rest)` as two top-level axes so I can slice one and keep the other." Building block of `local_tile` and `local_partition`. |
| `tiled_divide(A, tiler)` | "I want tiles grouped into one axis but my `rest` has multiple named dimensions I'll index separately." |
| `flat_divide(A, tiler)` | "I want a plain N-D coord space with no tuple-nesting." Frequently right before a `tma_partition`. |
| `cute.append(L, M)` | "I need a *new* outer axis for staging, iteration, or replication." Memory arrangement unchanged, domain grows. |
| `cute.group_modes(L, s, e)` | "A downstream function wants fewer modes; bundle these into a tuple." Use sparingly — prefer to design with the right rank from the start. |
| `cute.select(L, mode=[…])` | "Permute / drop modes." Free transpose: `select(L, mode=[1, 0])`. |
| `cute.make_composed_layout(swz, 0, L)` | "Apply a swizzle on top of an existing layout to avoid SMEM bank conflicts." |
| `cute.local_tile(T, tiler, coord)` | "Give me this CTA's slab of the global tensor." Equivalent to `zipped_divide` + slice-into-rest. |
| `cute.local_partition(T, thr_layout, tid)` | "Give me this thread's strided subset of the CTA's slab." Equivalent to `zipped_divide` + slice-into-tile. |
| `thr_mma.partition_A/B/C(T)` | "Give me this thread's MMA operand/accumulator view of `T`." Uses the MMA atom's built-in TV layout. |
| `cute.make_tensor(ptr, layout)` | "Rebind this layout onto a specific pointer (tmem, smem, gmem, registers)." |
| `cute.make_fragment_like(src, dtype)` | "Give me a register tensor with the same shape as `src` but a different dtype." Common for F32→F16 cast-down. |
| `cute.recast_ptr(iter, swizzle_=None)` | "Strip or replace the swizzle on a pointer before re-composing." |

### 5.12 Recommended reading, in order

1. **`references/cutlass/media/docs/cpp/cute/01_layout.md`** — the 20-minute intro to shape/stride.
2. **`references/cutlass/media/docs/cpp/cute/02_layout_algebra.md`** — the *only* place that formally explains why logical_divide is `composition(A, (B, complement(B)))`. Read this once and it all clicks.
3. **`references/cutlass/media/docs/cpp/cute/03_tensor.md`** — partitioning (inner/outer/TV), slicing with `_`.
4. **`references/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.md`** — the canonical GEMM flow from §5.8. Pair this with `references/cutlass/examples/cute/tutorial/sgemm_1.cu`.
5. **`references/cutlass/media/docs/cpp/cute/0t_mma_atom.md`** — how `MMA_Traits::ThrID`, `ALayout`, `BLayout`, `CLayout` encode the (thread, value) → (coord) mapping.
6. **`references/cutlass/media/docs/cpp/cute/0z_tma_tensors.md`** — TMA descriptor construction (what `make_tiled_tma_atom_A` does under the hood).
7. **`references/cutlass/media/docs/cpp/cute/0y_predication.md`** — how to handle non-divisible tile sizes with identity tensors.
8. **Lei Mao, "CuTe Layout Algebra"** — `leimao.github.io/article/CuTe-Layout-Algebra/` — worked examples beyond the CUTLASS docs.
9. **Jay Shah (Colfax), "A note on the algebra of CuTe Layouts"** — `research.colfax-intl.com/wp-content/uploads/2024/01/layout_algebra.pdf` — formal treatment of composition, complement, divide.
10. **simonb, "An applied introduction to CuTeDSL"** — `veitner.bearblog.dev/an-applied-introduction-to-cutedsl/` — Python-DSL walkthrough with concrete examples.

Reading these in order takes a weekend and will make every layout line in `mla_decode_fp16.py` obvious.

---

## 6. SMEM layouts and swizzles

Every operand has **two SMEM layouts** sharing the same underlying buffer:

- **Staged layout** — what the MMA atom eats. Has all stages + nested iteration modes.
- **TMA-flavored layout** — the page-sized view a single TMA issue writes.

### 6.1 Q latent SMEM (lines 405–413)

```python
q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
    qk_tiled_mma,
    self.mma_qk_tiler,                            # (128, 256, 64)
    self.q_dtype,                                 # Float16
    self.iterations_qk_latent * self.load_q_stage,  # [8 * 1 = 8 copies]
)
q_latent_smem_layout_staged = cute.logical_divide(
    q_latent_smem_layout_staged,
    (None, None, None, self.iterations_qk_latent) # [insert 4th-mode split of 8 → (8, 1)]
)
```

`sm100_utils.make_smem_layout_a` builds the canonical B200 SMEM layout for MMA operand A: shape `(cta_m, mma_k, stage)` with an internal swizzle matching K-major FP16. Final shape after `logical_divide`: `(cta_m, mma_k, stage_within, latent_iteration)` = `(128, 64, 1, 8)` under pinned.

### 6.2 Q rope SMEM (414–419)

Same pattern without `logical_divide` (only 1 rope iteration). Shape: `(128, 64, 1)`.

### 6.3 KC staged vs KC TMA (422–440)

K latent and K rope share one SMEM buffer. The MMA-side and TMA-side layouts:

```python
# MMA-side — 15-deep staged buffer                                 # 422–427
kc_smem_layout_staged = sm100_utils.make_smem_layout_b(
    qk_tiled_mma, self.mma_qk_tiler, self.k_dtype, self.load_kv_stage)
# shape ~ (cta_m, mma_n, mma_k, load_kv_stage) with MN-major swizzle for B operand.

# TMA-side — per-page view                                         # 432–440
kc_smem_layout_for_tma = sm100_utils.make_smem_layout(
    OperandMajorMode.K,
    (mma_qk_tiler[0] // thr_id.shape, mma_qk_tiler[2]),  # (cta_m=64, mma_k=64)
    k_dtype, load_kv_stage)
kc_smem_layout_for_tma = cute.tiled_divide(
    kc_smem_layout_for_tma, (kc_page_tile_size, mma_qk_tiler[2]))   # (64, 64)
```

The `tiled_divide` at line 438 gives you `((64, 64), rest)` — the outer tuple is "one TMA atomic transfer" and the `rest` mode indexes pages.

### 6.4 P SMEM (442–450) and VC SMEM (452–471)

Analogous to Q-latent / KC. P has `iterations_pv_k * p_mma_stage = 4*2 = 8` total staging copies, re-split into `(p_mma_stage=2, iterations_pv_k=4)` by `logical_divide` at line 448. VC is the B-operand of PV MMA, MN-major, with its TMA-side layout paged in the N dimension.

### 6.5 P re-swizzle for PV MMA (2617–2626)

The SMEM buffer already has one swizzle from `make_smem_layout_a`. For PV MMA to read P broadcast-efficiently, the code composes a second swizzle on a `recast_ptr`-stripped pointer:

```python
sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
swizzle_bits = int(math.log2(self.mma_pv_tiler[2] * self.q_dtype.width // 8 // 32)) + 1
# [= log2(64 * 16 / 8 / 32) + 1 = log2(4) + 1 = 3]
swizzle_base = 3 if self.q_dtype.width == 16 else 4        # [= 3 for FP16]
sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
sP_mk_view = cute.make_tensor(
    sP_wo_swizzle_iter,
    cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
)
```

Under pinned config: **`B=3`**, **`M=3`**, **`S=3`**. The net effect is what CUTLASS calls a "PISL" pattern that keeps LDSM reads bank-conflict-free.

---

## 7. Tiled MMA objects

### 7.1 The two tiled MMAs (lines 381–396)

```python
# QK MMA
qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
    self.q_dtype,                 # Float16
    OperandMajorMode.K,           # Q is K-major
    OperandMajorMode.K,           # K is K-major
    self.acc_dtype,               # Float32
    CtaGroup.TWO,                 # 2-CTA tcgen05
    self.mma_qk_tiler[:2],        # (128, 256)  -- atom knows K=64 itself
)

# PV MMA
pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
    self.v_dtype, OperandMajorMode.K, OperandMajorMode.MN,
    self.acc_dtype, CtaGroup.TWO,
    self.mma_pv_tiler[:2],        # (128, 256)
)
```

`[:2]` drops the K element because the atom supplies its own K.

### 7.2 Fragments — SMEM → MMA view (lines 1928–1932)

```python
tSrQ      = tiled_mma_qk.make_fragment_A(qk_params.sQ)       # Q SMEM  → Q MMA frag
tSrQ_rope = tiled_mma_qk.make_fragment_A(qk_params.sQ_rope)
tSrKC     = tiled_mma_qk.make_fragment_B(qk_params.sKC)
tOrP      = tiled_mma_pv.make_fragment_A(pv_params.sP)
tOrVC     = tiled_mma_pv.make_fragment_B(pv_params.sVC)
```

`tSrQ.shape[2]` is the per-atom K-block count (iterated at line 2089).

### 7.3 TMEM-backed accumulators (lines 1934–1956)

`partition_shape_C` asks the MMA atom "what's the per-thread shape of C?". We `append` a staging mode and then `make_tensor(tmem_ptr, layout)` to land the fragment in TMEM instead of registers.

```python
tStS_shape       = tiled_mma_qk.partition_shape_C(cute.select(mma_qk_tiler, mode=[0, 1]))
tStS_staged_fake = tiled_mma_qk.make_fragment_C(cute.append(tStS_shape, mma_s_stage))
tStS_staged     = cute.make_tensor(tmem_ptr, tStS_staged_fake.layout)

tOtO_shape  = tiled_mma_pv.partition_shape_C(cute.select(mma_pv_tiler, mode=[0, 1]))
tOtO        = tiled_mma_pv.make_fragment_C(tOtO_shape)
tOtO_layout = cute.append(
    tOtO.layout,
    cute.make_layout(L // mma_pv_tiler[1],                   # [= latent_dim / pv_N = 2]
                     stride=mma_pv_tiler[1] // warps_in_n),  # [= pv_N / 2 = 128]
)
tOtO_staged = cute.make_tensor(tStS_staged.iterator + tmem_o_offset, tOtO_layout)
```

The appended mode of size 2 (with stride 128) is what lets the PV loop at line 2163 walk the two halves of O in TMEM.

### 7.4 The gemm loops (2086–2113 for QK, 2161–2183 for PV)

**QK**: 8 latent issues + 1 rope issue, all accumulating into the same S in TMEM:

```python
tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)             # line 2080
for q_stage in range(iterations_qk_latent):                   # [8]
    for k_block in range(cute.size(tSrQ.shape[2])):
        cute.gemm(tiled_mma_qk, tStS,
                  tSrQ[...,k_block,q_stage], tSrKC[...,k_block,kc_stage], tStS)
        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
for q_stage in range(iterations_qk_rope):                     # [1]
    ...
```

**PV**: nested loop over K-blocks and output N-blocks:

```python
for p_stage in range(iterations_pv_k):              # [4]
    for acc_stage in range(iterations_pv_n):        # [2]
        tOtO = tOtO_staged[..., acc_stage]          # walks the appended stride=128 axis
        for k_block in range(tOrP.shape[2]):
            cute.gemm(tiled_mma_pv, tOtO,
                      tOrP[...,k_block,(p_stage, p_idx)],
                      tOrVC[...,k_block,vc_stage], tOtO)
```

---

## 8. TMA atoms and descriptors

All five TMA atoms start from one bulk copy op:

```python
tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(CtaGroup.TWO)   # line 473
```

### 8.1 Five TMA atoms

| Line | Atom | Call | Page? |
|------|------|------|-------|
| 476 | `tma_atom_q_latent`          | `make_tiled_tma_atom_A`          | no |
| 485 | `tma_atom_q_rope`            | `make_tiled_tma_atom_A`          | no |
| 495 | `tma_atom_c_latent`          | `make_paged_tiled_tma_atom(is_k_load=True)` | **yes** |
| 503 | `tma_atom_c_rope`            | `make_paged_tiled_tma_atom(is_k_load=True)` | **yes** |
| 513 | `tma_atom_c_latent_transpose`| `make_paged_tiled_tma_atom(is_k_load=False)`| **yes** |

Q atoms are dense; KV atoms use the paged template described in §4.12.

### 8.2 `tma_partition` at the load warp (lines 1580–1638)

`cpasync.tma_partition(atom, axis=0, warp_layout, smem_view, gmem_view)` produces per-thread tensors `(tXsX, tXgX)` that `cute.copy(atom, gmem_view, smem_view)` consumes. Shapes, per the comment at line 1578:

```
smem:  ((atom_v, rest_v), STAGE)
gmem:  ((atom_v, rest_v), RestM, RestK, RestL)
```

The `group_modes(..., 0, 3)` at lines 1584–1593 is what produces the `(atom_v, rest_v)` tuple.

### 8.3 Page-table indirection (lines 1705–1808, 1836–1872)

At runtime, the load warp:

1. Reads page indices out of the page-table SMEM buffer into Int32 registers via `cute.make_rmem_tensor(Int32, page_per_tile)` (line 1744).
2. For each page index `p`, issues `cute.copy(tma_atom_c_latent, tCLgCL[..., p], tKCsKC[..., stage])`. The `[..., p]` fills in the one "free" coord of the non-exec descriptor.

### 8.4 Descriptor prefetch (lines 829–833) and per-stage byte counts (lines 524–546)

5 `cpasync.prefetch_descriptor(...)` calls in the MMA warp prologue warm the TMA descriptor cache. The byte counts computed via `cute.size_in_bytes(dtype, smem_layout) * cute.size(thr_id.shape) * iterations_*` feed `tx_count` to the pipeline factory at line 3085.

---

## 9. End-to-end flow for one K-tile

```
warp 10 (load_pt)                       ← async page-table copy
   │
   ▼
warp  9 (load_tma)                      ← reads page indices from SMEM
   │       for i in range(page_per_tile):
   │           cute.copy(tma_atom_c_latent,           tCLgCL [..., pgidx[i]],  tKCsKC [stage])
   │           cute.copy(tma_atom_c_rope,             tKRgKR [..., pgidx[i]],  tKCsKC [stage])
   │           cute.copy(tma_atom_c_latent_transpose, tCLTgCLT[..., pgidx[i]], tVCsVC [stage])
   │
   ▼
warp  8 (mma)
   │   QK gemm: 8 latent + 1 rope issues, accumulate into S (TMEM)
   │   PV gemm: 4 K-blocks × 2 N-blocks, accumulate into O (TMEM)
   │
   ▼
warps 0-3 (compute)
   │   load S from TMEM, mask, row_max (warp-pair exchange), exp2, row_sum,
   │   cast F32→F16, write P into SMEM via recast_ptr + make_swizzle
   │
   ▼
warps 4-7 (correction)
       load O from TMEM, multiply by correction_factor;
       on last tile: O *= output_scale * rcp_approx(row_sum);
       store O and LSE to gmem via cute.copy and cute.local_tile.
```

---

## 10. Pipeline summary

| Pipeline | Class | Stages | Producer | Consumer | Factory line |
|----------|-------|--------|----------|----------|--------------|
| `load_pt_pipeline` | `PipelineCpAsync`   | 4  | warp 10 (32 thr) | warp 9 (32 thr) | 3077 |
| `load_q_pipeline`  | `PipelineTmaUmma`   | 1  | warp 9 (1 thr)   | warp 8 (1 thr)  | 3108 |
| `load_kv_pipeline` | `PipelineTmaUmma`   | 15 | warp 9 (1 thr)   | warp 8 (1 thr)  | 3108 |
| `mma_s_pipeline`   | `PipelineUmmaAsync` | 2  | warp 8 (1 thr)   | warps 0-3 × cluster | 3144 |
| `p_mma_pipeline`   | `PipelineAsyncUmma` | 2  | warps 0-3 × cluster | warp 8 (1 thr)   | 3179 |
| `p_cor_pipeline`   | `PipelineAsync`     | 2  | warps 0-3        | warps 4-7       | 3209 |
| `mma_o_pipeline`   | `PipelineUmmaAsync` | 1  | warp 8 (1 thr)   | warps 0-3 × cluster | 3243 |

---

## 11. Layout-op cheat sheet

| If you want to… | Use | Output structure |
|-----------------|-----|------------------|
| rename / reorder / drop modes | `cute.select(l, mode=[...])` | re-modes of `l` |
| split each mode into `(tile, rest)` keeping structure | `cute.logical_divide(l, tiler)` | `((t0, r0), (t1, r1), …)` |
| group all tile modes first, all rest modes after | `cute.tiled_divide(l, tiler)` | `((t0, t1, …), r0, r1, …)` |
| like `tiled_divide` but flatten tile-tuple | `cute.flat_divide(l, tiler)` | `(t0, t1, …, r0, r1, …)` |
| like `tiled_divide` but nest rest too | `cute.zipped_divide(l, tiler)` | `((t0, t1, …), (r0, r1, …))` |
| overlay a tiler on an identity | `cute.composition(identity, tiler)` | tile-major coord layout |
| apply a swizzle | `cute.make_composed_layout(sw, 0, l)` | swizzled layout |
| flatten a range of modes into one tuple | `cute.group_modes(l, start, end)` | fewer-mode layout |
| append a trailing mode | `cute.append(l, m)` | one more mode |
| slice CTA-local gmem tile | `cute.local_tile(t, cta_tiler, coord)` | per-CTA view |
| build predication coords | `cute.make_identity_tensor(shape)` | coord tensor |
| reinterpret dtype | `cute.recast_tensor(t, new_dtype)` | same shape, new dtype |
| strip/replace swizzle on a pointer | `cute.recast_ptr(iter, swizzle_=…)` | re-swizzled iter |

---

## 12. How to verify this doc

1. Open the source next to this doc and click through the cited lines — especially the two `make_trivial_tiled_mma` calls (381, 389), the 5 TMA atoms (476, 485, 495, 503, 513), the `make_paged_tiled_tma_atom` walkthrough (666–704), and the QK+PV gemm loops (2086, 2161).
2. Re-derive the pinned numbers: `iterations_qk_latent = 512/64 = 8`, `iterations_pv_n = 512/256 = 2`, `iterations_pv_k = 256/64 = 4`, `tmem_o_offset = 2·256/2 = 256`, `cta_m = 128/2 = 64`, `cta_n = 256/2 = 128`.
3. For §4's numerical partition examples, manually compute `coord → offset` on paper and confirm against `shape × stride` arithmetic.
