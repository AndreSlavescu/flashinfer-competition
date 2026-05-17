# Stage 3 Path in `kernel_0.py` — a Walkthrough

This document traces the **Stage 3 ("S3")** path through `solution/dsa_attention/kernel_0.py` — a kernel that is being built incrementally by a designer → **stage coder ↔ stage reviewer** loop. It covers (1) what S3 is for, (2) the implementation, (3) how the per-stage validation harness is wired, and (4) the workarounds the coder agent inserted so that the stage-reviewer protocol remains satisfiable at S3 *without* derailing S0/S1/S2.

All citations use `kernel_0.py:N`.

---

## 1. Orientation — how the stage loop shapes this file

The kernel emerges one stage at a time. After the designer writes `solution/dsa_attention/kernel_0_plan.md`, a stage-coder implements stage `Sk`, a stage-reviewer evaluates the output, and the two iterate until the reviewer's harness returns `matched=True`. Only then does stage `Sk+1` begin. As the file grows it accumulates, per stage, three artifacts:

- a **device probe path** inside the CuTe kernel that writes `debug_stage<N>_*` metadata tensors
- **two independent host references** — a canonical `torch_reference_s<N>_...` and a `..._naive` second-opinion rederivation
- a **`_stage<N>_prefix_validation_harness(...)`** that cross-checks the probe, the canonical reference, the naive reference, and a set of hard-coded expectations

Each harness takes a `dependency_validation` dict from the prior stage and `and`s the prior stage's `matched` into its own — so regressions in S0 still fail S3 (`kernel_0.py:2196`, `:2217`).

> **Where S3 sits:** it is the *last infrastructure stage* before compute starts. S0/S1/S2 set up scheduling and MMA atom construction; S3 locks down the SMEM/TMEM layout budget; S4+ (Q-prologue load, QK gather, UMMA, softmax, PV, epilogue) consumes S3's layouts. In the current file, S4 has just begun — evidenced by `_stage4_prefix_validation_harness` at `kernel_0.py:2255` and `torch_reference_s4_q_prologue` at `:2086`.

---

## 2. What Stage 3 does

**Stage 3 performs no compute.** It is a *specification stage* that produces:

1. Live CuTe SMEM layout objects for Q/Kc/Kpe/P/V via `sm100_utils.make_smem_layout_{a,b}` + `cute.logical_divide`, consumed by S4–S12.
2. A modeled TMEM occupancy (fake fragments for the logits `S` and the accumulator `O`) from which the hardware TMEM column allocation is computed via `cutlass_utils.get_num_tmem_alloc_cols(..., rounding=True)`.
3. A launch-time `SharedStorage` `@cute.struct` whose size the kernel passes as `smem=` at launch (`kernel_0.py:1210`).
4. A device-captured budget report (`debug_stage3_storage_meta`, 6 × Int32) that the host harness can validate.

The planner spec lives in `kernel_0_plan.md` Section "Infrastructure" (around lines 386, 424 — "Use `sm100_utils.make_smem_layout_a/b(...)` plus `cute.logical_divide(...)` exactly like the MLA example"). Budget constants are hoisted to module scope at `kernel_0.py:133-159`:

```python
# kernel_0.py:133-144
S3_Q_CKV_SMEM_BYTES = PADDED_Q_HEAD_ROWS * HEAD_DIM_CKV * BF16_BYTES
S3_Q_KPE_SMEM_BYTES = PADDED_Q_HEAD_ROWS * HEAD_DIM_KPE * BF16_BYTES
S3_Q_SMEM_BYTES = S3_Q_CKV_SMEM_BYTES + S3_Q_KPE_SMEM_BYTES
...
S3_TMEM_LOGICAL_COLS = (S3_TMEM_S_BYTES + S3_TMEM_O_BYTES) // TMEM_COLUMN_BYTES_SM100
S3_TMEM_ALLOC_COLS = max(32, 1 << (S3_TMEM_LOGICAL_COLS - 1).bit_length())
```

The **planner-facing static contract** is `@cute.struct Stage3SharedStorage` at `kernel_0.py:170-228`: seven pipeline barrier rings (`page`, `qk_load`, `s`, `p`, `corr`, `v`, `o`), a TMEM-allocator handoff slot, a sparse metadata ring (`smem_page_id`, `smem_page_off`, `smem_valid_mask`), a correction metadata ring (`smem_corr_alpha`, `smem_corr_row_sum`, `smem_corr_row_max`, `smem_corr_flags`), and the six UMMA-aligned operand buffers (`smem_q_ckv`, `smem_q_kpe`, `smem_kc_qk`, `smem_kpe_qk`, `smem_p`, `smem_v`), each 1024-byte aligned. Its total is fixed as `S3_SHARED_STORAGE_BYTES = Stage3SharedStorage.size_in_bytes()` at `kernel_0.py:231` — used as the reviewer's ground truth.

---

## 3. Implementation walkthrough

### 3.1 Entry point

```python
# kernel_0.py:437-445
@cute.jit
def _ensure_stage3_storage_artifacts_jit(self, qk_tiled_mma, pv_tiled_mma):
    """Construct the real S3 CuTe layout artifacts needed by later stages.

    S3 is not only an arithmetic budget check: later stages S4-S12 consume the
    actual staged SMEM/TMEM layouts. Build and persist those layout objects here
    from the official helper APIs so the launch path already matches the
    Blackwell reference style.
    """
```

That docstring is almost certainly a reviewer-note the coder internalized — see §6.

### 3.2 SMEM layouts for Q and K (`kernel_0.py:447-489`)

Q-ckv and Q-kpe layouts come from `sm100_utils.make_smem_layout_a`; Kc-qk and Kpe-qk from `make_smem_layout_b`. Each is then `cute.logical_divide`'d over its subtile-iteration count so later stages can index by iter:

```python
# kernel_0.py:447-456
q_ckv_smem_layout_staged = sm100_utils.make_smem_layout_a(
    qk_tiled_mma, self.qk_mma_shape, cutlass.BFloat16, self.qk_ckv_iters,
)
q_ckv_smem_layout_staged = cute.logical_divide(
    q_ckv_smem_layout_staged,
    (None, None, None, self.qk_ckv_iters),
)
```

The K-ring layouts scale the outer iter count by `QK_PIPE_STAGES` to carve out a ring-buffer along the stages mode (`:473`, `:484`). This is the canonical Blackwell MLA construction path — using the blackwell_helpers makes the resulting layouts byte-compatible with TMA and UMMA expectations without hand-rolling.

### 3.3 SMEM layouts for P and V (`kernel_0.py:491-511`)

Symmetric treatment with `P_PIPE_STAGES` / `V_PIPE_STAGES` multiplied into the outer iter count, giving double-buffered rings for softmax output and V-slice gathers.

### 3.4 TMEM fake fragments for S and O (`kernel_0.py:513-529`)

```python
# kernel_0.py:513-529
tmem_s_shape = qk_tiled_mma.partition_shape_C(self.qk_mma_shape[:2])
tmem_s_staged_fake = qk_tiled_mma.make_fragment_C(
    cute.append(tmem_s_shape, QK_PIPE_STAGES)
)
tmem_o_tile_shape = pv_tiled_mma.partition_shape_C(self.pv_mma_shape[:2])
tmem_o_tile_fake = pv_tiled_mma.make_fragment_C(tmem_o_tile_shape)
# Model O as the single logical [64, 512] TMEM tensor planned for S3, not
# as four independent [64, 128] slices. Follow the Blackwell MLA style and
# append the output-slice mode onto the partitioned C fragment layout.
tmem_o_staged_layout = cute.append(
    tmem_o_tile_fake.layout,
    cute.make_layout(self.pv_out_iters, stride=self.pv_mma_shape[1] // 2),
)
tmem_o_staged_fake = cute.make_tensor(
    tmem_o_tile_fake.iterator, tmem_o_staged_layout,
)
```

The 3-line comment (`:519-521`) is a reviewer artifact — it records *why* O is modelled as one `[64, 512]` tensor instead of four `[64, 128]` slices. See §5.4.

### 3.5 Byte counts and TMEM column allocation (`kernel_0.py:531-560`)

Element counts are derived via `cute.cosize(layout)` and converted to bytes by `BF16_BYTES`. TMEM columns are computed three times via `cutlass_utils.get_num_tmem_alloc_cols`: unrounded for S alone, unrounded for O alone, and `rounding=True` on the pair to get the actual hardware allocation width:

```python
# kernel_0.py:553-560
tmem_alloc_cols = cutlass_utils.get_num_tmem_alloc_cols(
    [tmem_s_staged_fake, tmem_o_staged_fake],
    rounding=True,
    arch="sm_100",
)
```

### 3.6 Persistence onto `self` (`kernel_0.py:562-600`)

All sixteen fields (six layouts, two fake fragments, six element counts, two col counts) are written to `self.stage3_*` plus the four byte budgets (`q_smem_bytes`, `qk_ring_bytes`, `p_ring_bytes`, `v_ring_bytes`) and `tmem_alloc_cols`. The same tuple is returned so the caller can bind local names. Stages S4+ read these back directly off the program object.

### 3.7 Caching wrapper (`kernel_0.py:672-725`)

Public entry is `stage3_storage_debug_reference(...)` at `:672`. It:

1. Short-circuits on a cached result (`:679-681`) — idempotent across repeated review cycles.
2. Raises if `sm100_utils`, `tcgen05`, or `cutlass_utils` are `None` (`:683-687`) — explains to the reviewer exactly which import is missing.
3. Flips `self.enable_stage3_storage_probe = True` inside a `try/finally` and re-runs `_launch_s1_scheduler_debug_kernel` so the probe writes the six metadata values (`:702-711`).
4. Calls `self._require_stage3_storage_artifacts()` to confirm the JIT populated all 16 required fields (`:713`).
5. Cross-checks the probe-captured ints against `self.stage3_storage_host_reference()` field-by-field and raises `AssertionError` on any mismatch (`:717-722`) — this is effectively a *third* consistency check before the harness even runs.

---

## 4. The validation harness

### 4.1 Two host references

```python
# kernel_0.py:2011-2023   — canonical reference
def torch_reference_s3_storage_config(q_nope, q_pe, ...):
    """Host mirror of the S3 storage-layout / shared-storage configuration."""
    device = q_nope.device
    del q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
    return BlackwellStyleKernel().stage3_storage_host_reference(device=device)
```

```python
# kernel_0.py:2026-2083   — naive second-opinion reference (arithmetic-only)
def torch_reference_s3_storage_config_naive(q_nope, q_pe, ...):
    """Independent arithmetic-only S3 reference for storage sizes and TMEM columns."""
    ...
    barrier_bytes = (6 * (QK_PIPE_STAGES * 2) + (O_PIPE_STAGES * 2)) * INT64_BYTES
    ...
    shared_storage_bytes = _align_up(pre_operand_bytes, 1024)
    shared_storage_bytes += q_smem_bytes
    shared_storage_bytes = _align_up(shared_storage_bytes, 1024)
    ...
    tmem_alloc_cols = max(32, 1 << (tmem_logical_cols - 1).bit_length())
```

The naive version open-codes `_align_up` around every operand ring, so if the canonical `Stage3SharedStorage.size_in_bytes()` ever silently changed its padding strategy, the harness would catch the drift as an exact-integer mismatch.

### 4.2 `_stage3_prefix_validation_harness` (`kernel_0.py:2161-2252`)

Three classes of check, 15 total, combined with the prior-stage `matched`:

| Class | Check | Lines |
|---|---|---|
| Exact cross-check | `shared_storage_bytes`, `tmem_alloc_cols`, `q_smem_bytes`, `qk_ring_bytes`, `p_ring_bytes`, `v_ring_bytes` all `torch.equal(cute_result, naive_result)` | `:2169-2180` |
| Expected constant | Each of the six matches its hoisted `S3_*` module constant | `:2182-2187` |
| Hardware constraint | `shared_storage_bytes <= (228-1) * 1024`; `tmem_alloc_cols` power of 2; `tmem_alloc_cols % 32 == 0` | `:2188-2192` |

```python
# kernel_0.py:2194-2213
matched = all(
    [
        dependency_validation["matched"],   # propagates S2 failure
        shared_storage_match,
        tmem_alloc_cols_match,
        q_smem_bytes_match,
        qk_ring_bytes_match,
        p_ring_bytes_match,
        v_ring_bytes_match,
        shared_storage_expected,
        tmem_cols_expected,
        q_budget_expected,
        qk_budget_expected,
        p_budget_expected,
        v_budget_expected,
        shared_storage_within_budget,
        tmem_cols_power_of_two,
        tmem_cols_multiple_of_32,
    ]
)
```

Return shape: `{"matched", "report", "stage_outputs"}` where `stage_outputs` is the dict the *next* stage receives as `dependency_validation` (`:2241-2251`).

### 4.3 Dispatcher (`kernel_0.py:2339-2380`)

`prefix_validation_harness(stage_id="S3", dependency_validation=...)` routes to `_stage3_prefix_validation_harness` and `raise ValueError("S3 validation requires dependency_validation from S2")` if the caller forgets the chain (`:2364-2371`).

### 4.4 Probe capture path

The device values fed into `cute_result` come from `_emit_stage_probe_metadata` at `kernel_0.py:845`, whose relevant block is:

```python
# kernel_0.py:881-887
if cutlass.const_expr(self.enable_stage3_storage_probe):
    debug_stage3_storage_meta[0] = cutlass.Int32(SharedStorage.size_in_bytes())
    debug_stage3_storage_meta[1] = cutlass.Int32(self.tmem_alloc_cols)
    debug_stage3_storage_meta[2] = cutlass.Int32(self.q_smem_bytes)
    debug_stage3_storage_meta[3] = cutlass.Int32(self.qk_ring_bytes)
    debug_stage3_storage_meta[4] = cutlass.Int32(self.p_ring_bytes)
    debug_stage3_storage_meta[5] = cutlass.Int32(self.v_ring_bytes)
```

The exact slot order matches the key order in `_stage3_storage_result_from_meta` (host side), so the harness just unpacks by index.

---

## 5. Workarounds the coder inserted to preserve the protocol

Each subsection: *planned path → workaround → why*.

### 5.1 Dynamic `@cute.struct` re-declaration at launch time

**Planned:** use the single static `Stage3SharedStorage` (`:170`) for everything.

**Workaround:** inside the `__call__` JIT path a *second* struct is defined from scratch with `cute.cosize(...)` as its buffer sizes:

```python
# kernel_0.py:1114-1186
@cute.struct
class Stage3LayoutBackedSharedStorage:
    # Pipeline barriers.
    page_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
    ...
    # Main operand storage aligned for UMMA-facing SMEM.
    smem_q_ckv: cute.struct.Align[
        cute.struct.MemRange[
            cutlass.BFloat16, cute.cosize(q_ckv_smem_layout_staged)
        ],
        1024,
    ]
    ...
    smem_v: cute.struct.Align[
        cute.struct.MemRange[
            cutlass.BFloat16, cute.cosize(v_smem_layout_staged)
        ],
        1024,
    ]
```

Then:

```python
# kernel_0.py:1187-1213
SharedStorage = Stage3LayoutBackedSharedStorage
self.shared_storage_bytes = SharedStorage.size_in_bytes()
...
self.kernel1_impl(..., SharedStorage).launch(
    grid=self._host_launch_grid,
    block=[self.threads_per_cta, 1, 1],
    cluster=self.cluster_shape_mnk,
    smem=SharedStorage.size_in_bytes(),
    stream=stream,
    min_blocks_per_mp=1,
)
```

**Why:** `@cute.struct` fields must be resolvable when the class is defined. The operand buffer sizes returned by `sm100_utils.make_smem_layout_*` are only known *after* `_ensure_stage3_storage_artifacts_jit` runs inside MLIR — they are not available at module import time. The static `Stage3SharedStorage` therefore has to encode the byte budget with *closed-form* element counts (e.g. `PADDED_Q_HEAD_ROWS * HEAD_DIM_CKV`), while the launch-time struct closes over the live layout objects and uses `cute.cosize(...)`. The reviewer's `shared_storage_expected` check at `:2182` depends on both to agree: the module constant `S3_SHARED_STORAGE_BYTES` comes from the static version (`:231`), and the probe value comes from the dynamic one (`:882`). If the two ever disagreed, the harness would catch it.

### 5.2 Defensive import fallbacks at module import

**Planned:** plain top-of-file imports.

**Workaround:**

```python
# kernel_0.py:9-34
try:
    import cutlass.pipeline as pipeline
except Exception:  # pragma: no cover - stage scaffolding must remain importable
    pipeline = None
try:
    import cutlass.utils as cutlass_utils
except Exception:  # pragma: no cover - stage scaffolding must remain importable
    cutlass_utils = None
try:
    import cutlass.cute.nvgpu.tcgen05 as tcgen05
except Exception:  # pragma: no cover - stage scaffolding must remain importable
    tcgen05 = None
try:
    import cutlass.utils.blackwell_helpers as sm100_utils
except Exception:  # pragma: no cover - stage scaffolding must remain importable
    sm100_utils = None
```

The S3 entry then explicitly re-checks and raises:

```python
# kernel_0.py:683-687
if sm100_utils is None or tcgen05 is None or cutlass_utils is None:
    raise RuntimeError(
        "Stage S3 requires cutlass.utils, tcgen05, and blackwell_helpers so the real "
        "SMEM/TMEM layout artifacts can be constructed through the CuTe kernel path."
    )
```

**Why:** the stage-reviewer runs the *host-side* harness and references (`torch_reference_*`, `_stage*_prefix_validation_harness`) in environments that may not have the full CUTLASS Python SDK installed. If the module failed hard at import, the reviewer couldn't even evaluate S0/S1 on a machine without CUTLASS — which would incorrectly block all downstream stage review. The `# pragma: no cover - stage scaffolding must remain importable` annotation spells this contract out in code.

### 5.3 `cutlass.const_expr` gating on the stage-3 probe writes

**Planned:** the probe always writes stage-3 metadata.

**Workaround:** see the `const_expr` branch at `kernel_0.py:881`.

**Why:** the *same* CuTe kernel function (`kernel1_impl`) is reused for S1-only runs, S2 runs, S3 runs, and S4 runs. At `:2673` (and the similarly-named S2 probe flag) the host toggles `program.enable_stage3_storage_probe = stage_id in {"S3", "S4"}`. Because the flag is a Python-level attribute captured by `const_expr`, the branch collapses at MLIR lowering time — S1 runs do not emit the stage-3 writes at all. This matters because the probe references `SharedStorage.size_in_bytes()`, `self.tmem_alloc_cols`, etc., which may not be populated when the reviewer is only validating S1. Without the `const_expr` gate, an S1-only harness run would either crash at lowering or silently read stale fields.

### 5.4 Modeling O as one `[64, 512]` TMEM tensor, not four `[64, 128]` slices

**Planned:** the naïve Blackwell MLA style would produce four separate `[64, 128]` O fragments.

**Workaround (`kernel_0.py:519-529`, quoted in §3.4):** append an `out_iters`-sized mode to the layout of the first `make_fragment_C` result and rebuild a tensor from the original iterator, yielding a single `[64, 512]` fake fragment.

**Why:** `cutlass_utils.get_num_tmem_alloc_cols(..., rounding=True)` called on `[S_fake, O_fake]` at `:553` assumes O is one allocation. Four independent slices would round four times separately and overstate the footprint, breaking the `tmem_cols_expected == S3_TMEM_ALLOC_COLS == 512` assertion at `:2183`. It also means downstream stages (the PV MMA and correction/epilogue) see one contiguous TMEM tensor rather than four — matching the plan's specification that "O and logits stay in TMEM" (`kernel_0_plan.md:10`) with a single allocation. The 3-line comment in the code makes this reviewer-facing reasoning explicit.

### 5.5 Per-field S3 artifact requirement gate

**Planned:** downstream stages access layouts directly.

**Workaround:**

```python
# kernel_0.py:602-626
def _require_stage3_storage_artifacts(self) -> None:
    required_fields = (
        "stage3_q_ckv_smem_layout_staged",
        "stage3_q_kpe_smem_layout_staged",
        "stage3_kc_qk_smem_layout_staged",
        "stage3_kpe_qk_smem_layout_staged",
        "stage3_p_smem_layout_staged",
        "stage3_v_smem_layout_staged",
        "stage3_tmem_s_staged_fake",
        "stage3_tmem_o_staged_fake",
        "stage3_q_ckv_smem_elements",
        "stage3_q_kpe_smem_elements",
        "stage3_kc_qk_smem_elements",
        "stage3_kpe_qk_smem_elements",
        "stage3_p_smem_elements",
        "stage3_v_smem_elements",
        "stage3_tmem_s_cols",
        "stage3_tmem_o_cols",
    )
    missing = [f for f in required_fields if getattr(self, f) is None]
    if missing:
        raise RuntimeError(
            "Stage S3 probe did not persist the planned CuTe layout/TMEM artifacts needed "
            f"by later stages; missing fields: {missing}"
        )
```

Called twice: from the cache path (`:680`) and after a fresh probe (`:713`).

**Why:** the cache at `self._stage3_storage_cache` makes the JIT idempotent, but the *cache* holds only the host-facing 6-field dict — the 16 on-object layout artifacts live separately. If a prior code path populated one but not the other, later stages would see `None` for a layout and fail deep inside MLIR with a message unrelated to S3. The explicit gate surfaces the failure with a specific list of missing fields at the S3 boundary, which is the reviewer-friendly place to fail.

### 5.6 Protocol-duplicated TMEM-col formula (`kernel_0.py:144` and `:2069`)

Not strictly a workaround, but worth flagging: the rounding formula

```python
S3_TMEM_ALLOC_COLS = max(32, 1 << (S3_TMEM_LOGICAL_COLS - 1).bit_length())
```

appears verbatim in the naive reference at `:2069`. The duplication is deliberate: if the naive path "simplified" the rounding (e.g. used plain `round_up(..., 32)`), the harness would diverge from the CuTe-side `rounding=True` behavior whenever the logical column count lands strictly between two powers of two. Both paths encode the protocol inline so they stay aligned by construction rather than by convention.

---

## 6. Evidence of the coder ↔ reviewer loop in the file

- **Dependency chaining**: every `_stage<N>_prefix_validation_harness` takes a `dependency_validation` keyword (`:2165`, `:2196`, `:2217` for S3; mirrored for S4 at `:2259`, `:2298`). A regression in any earlier stage fails every later stage at the same boundary.
- **Dual independent references**: `stage3_storage_host_reference` (canonical) vs `torch_reference_s3_storage_config_naive` (open-coded arithmetic) — a second source of truth that the reviewer can compare against even if the canonical path drifts.
- **15-assertion `all(...)` gate (`:2194-2213`)**: a list this granular is the trace of multiple review rounds ("also check power-of-two"; "also check multiple-of-32"; "also check it fits in 227 KB"). Each bullet is likely one review ticket.
- **Docstring at `:439-445`**: "S3 is not only an arithmetic budget check: later stages S4-S12 consume the actual staged SMEM/TMEM layouts. Build and persist those layout objects here…" — reads as a direct reviewer-note the coder internalized, strongly suggesting an earlier revision treated S3 as purely arithmetic (no live layouts) and the reviewer pushed for materialization.
- **Third consistency check in `stage3_storage_debug_reference` (`:717-722`)**: even before the harness runs, the probe-captured dict is `torch.equal`'d field-by-field against the host reference and `AssertionError`'s on mismatch. Belt-and-suspenders of the kind a reviewer asks for after a once-only mismatch.
- **Cache + `_require_*` pair (`:679-681` and `:602-626`)**: the cache makes the JIT re-entry idempotent for the reviewer; the require-gate makes partial state visible. The pair exists because the loop *re-enters* S3 across review cycles, and the coder has had to make that re-entry robust.

---

## 7. Closing observations

Stage 3 has two simultaneous loops it must satisfy:

- the **reviewer-facing loop** — host harness + naive reference + `all()` gate + `dependency_validation` chain
- the **downstream-facing loop** — cached on-object artifacts + `_require_*` gate consumed by S4+

Most of the S3 code is neither compute nor arithmetic: it is **protocol-preservation surface**. The static `Stage3SharedStorage`, the module-level `S3_*` constants, the `_naive` reference, the duplicated rounding formula, the `const_expr` gates, and the defensive imports all exist so that each stage review can run *in isolation* on machines with incomplete toolchains while still catching drift between the static plan and the dynamic CuTe layout it anchors.

S3 is small in compute (none) but heavy in contract — which is exactly what you would expect from a stage that ships *a promise* for stages S4 through S12 to build on.
