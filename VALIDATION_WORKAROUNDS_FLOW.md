# Validation-Only Scaffolding in `kernel_0.py`

`solution/dsa_attention/kernel_0.py` is **3023 lines**; by my count **~2500 of those lines** exist *only* to satisfy the stage-reviewer harness and will disappear once S4–S12 are implemented. This doc enumerates all of that scaffolding and states why each piece is load-bearing *for review* even though it has no place in the final kernel.

---

## TL;DR — the seven categories

| # | Category | Approx lines | Keep in final? |
|---|---|---|---|
| 1 | Single-thread `@cute.jit` runners replacing warp-specialized paths | ~200 | **drop** |
| 2 | Debug kernel + probe flags + `debug_*_meta` tensor plumbing | ~300 | **drop** |
| 3 | Static `Stage{0,3}SharedStorage` + module-level `S3_*` budget constants | ~100 | **drop** (keep only `Stage3LayoutBackedSharedStorage`) |
| 4 | Probe-driven per-stage `*_debug_reference` methods + `_require_*` gate + cache | ~250 | **drop** |
| 5 | Ten host-reference functions (five `torch_reference_s<N>_*` pairs) | ~600 | **drop naive twins**, promote one pair as correctness ref |
| 6 | Five `_stage<N>_prefix_validation_harness` functions + dispatcher | ~550 | **drop** (move to a separate test file if kept) |
| 7 | `run()` stage-dispatch driver, fixtures, printers, `_launch_s1_scheduler_debug_kernel` | ~500 | **drop** (extract into tests/) |

What actually stays: `BlackwellStyleKernel.__call__`, `_ensure_stage3_storage_artifacts_jit`, `Stage3LayoutBackedSharedStorage`, the (future) real warp-specialized `@cute.kernel`, clean imports, and **one** correctness reference.

---

## 1. Single-thread runners (the user's concern)

| Function | Lines | What it does |
|---|---|---|
| `_run_scheduler_single_thread` | `kernel_0.py:890-958` | Wraps the whole scheduler loop in `if tidx == 0:`. Walks the persistent tile scheduler, emits `cute.printf` logs, and writes `debug_token_order[bidz, iter]` / `debug_token_done_count[token]` one element at a time. |
| `_run_stage4_q_prologue_single_thread` | `kernel_0.py:960-1065` | Again `if tidx == 0:`. Reads Q rows one-by-one, emits `cute.printf`, writes `debug_q_ckv_padded` / `debug_q_kpe_padded` serially. |

**Why we need them for validation.** The reviewer needs a *deterministic, captured* snapshot of what the scheduler visited and what the Q-prologue wrote. The final kernel distributes this work across warps W0–W7 concurrently (per `kernel_0_plan.md:227-234`), which makes the output order observable only in aggregate — you can't diff a race-correct multi-warp trace against a Python reference. Single-thread serialization gives a stable `debug_token_order` matrix that `_stage1_prefix_validation_harness` at `:1742` can `torch.equal`-compare against `torch_reference_s1_scheduler_naive`. The `cute.printf` spam also lets the reviewer read the CuTe trace when the harness fails.

**Why they're throwaway.** The final kernel has no `tidx == 0` scheduler loop — every warp pulls work via `StaticPersistentTileScheduler` and produces output in TMEM, not in Python-visible debug tensors.

## 2. Debug kernel wrappers + probe metadata

| Identifier | Line | Role |
|---|---|---|
| `self.enable_stage2_mma_probe` / `enable_stage3_storage_probe` / `enable_stage4_q_prologue_probe` | `:321-323` | Per-stage on/off switches toggled by `run(stage_id=...)` at `:2717-2719`. |
| `_emit_stage_probe_metadata` | `:844-888` | Writes 18 Int32 MMA fields + 6 Int32 storage fields to `debug_stage{2,3}_*_meta` under `cutlass.const_expr(...)` guards. |
| `persistent_token_scheduler_debug` | `:1067-1120` | The "kernel" the debug path actually runs. Calls probe emit + single-thread runners instead of the real compute. |
| `kernel1_impl` (`@cute.kernel`) | `:1261-1290` | Thin `@cute.kernel` entry that forwards to `persistent_token_scheduler_debug`. |
| `debug_token_order`, `debug_token_done_count`, `debug_stage2_mma_meta`, `debug_stage3_storage_meta`, `debug_q_ckv_padded`, `debug_q_kpe_padded` (kernel parameters) | `:1266-1271` and plumbed through `:1119-1286` | Six extra tensor parameters the kernel carries only to emit probe data. |

**Why we need them for validation.** The reviewer's only channel for inspecting what *really happens inside the MLIR-lowered kernel* is these tensors. Without them the harness would have to trust the host-side `stage3_storage_host_reference` alone, defeating the whole point of a second opinion. `cutlass.const_expr(self.enable_*_probe)` lets the reviewer run S1-only without dragging in S3's tcgen05 references (see prior doc §5.3).

**Why they're throwaway.** In the final kernel: no six debug tensors in the signature, no probe flags on `self`, `kernel1_impl` body is replaced by the real warp-specialized `@cute.kernel` described in `kernel_0_plan.md:222-234`.

## 3. Static `SharedStorage` stand-ins and budget constants

| Identifier | Line | Role |
|---|---|---|
| `Stage0SharedStorage` | `:167-171` | Empty placeholder: `placeholder: cutlass.Int32`. Pure scaffold. |
| `Stage3SharedStorage` (static) | `:170-228` | Hand-written `@cute.struct` with constant element counts. Its `size_in_bytes()` seeds `S3_SHARED_STORAGE_BYTES`. |
| `S3_Q_SMEM_BYTES`, `S3_QK_RING_BYTES`, `S3_P_RING_BYTES`, `S3_V_RING_BYTES`, `S3_TMEM_ALLOC_COLS`, `S3_SHARED_STORAGE_BYTES`, `_align_up` helper | `:133-159`, `:231`, `:133` | Closed-form byte/column budgets referenced only by the harness and the naive reference. |

**Why we need them for validation.** The *dynamic* `Stage3LayoutBackedSharedStorage` built inside `__call__` at `:1114-1186` depends on `cute.cosize(...)` of JIT-produced layouts, so it can't be known at module import. The harness needs *some* compile-time ground truth to assert against, so the coder mirrors the budget by hand. Every `S3_*` constant appears in the expected-value checks at `:2220-2225`.

**Why they're throwaway.** `Stage3LayoutBackedSharedStorage` is the real struct the launch uses (`:1187`, `:1255`). `Stage0SharedStorage` is only referenced by the round-0 scaffold. All the `S3_*_BYTES` constants are duplicated in the naive reference at `:2062-2120` and have no consumer in compute-path code.

## 4. Probe-driven reference methods on the program

| Method | Line | Role |
|---|---|---|
| `scheduler_debug_reference` | `:789-842` | Host-side Python model of the scheduler (nested loop over CTAs and iters). |
| `stage2_mma_debug_reference` | `:727-787` | Toggles `enable_stage2_mma_probe`, launches debug kernel, reconstructs MMA config from `debug_stage2_mma_meta`. |
| `stage3_storage_debug_reference` | `:672-725` | Same pattern for S3. Also does a field-by-field `torch.equal` against `stage3_storage_host_reference` before returning. |
| `stage3_storage_host_reference` | `:633-657` | Host mirror of the 6 S3 byte/col outputs, built from fields on `self`. |
| `_stage3_storage_result_from_meta` | `:659-670` | Unpacks the 6-Int32 probe tensor back into a dict keyed like the host reference. |
| `_require_stage3_storage_artifacts` | `:602-626` | 16-field presence gate; raises with the list of missing fields. |
| `self._stage3_storage_cache` | `:327`, used `:679-681`, `:724` | Memoizes probe result across the multi-step `run()` orchestration. |

**Why we need them for validation.** They give the harness three independent snapshots (device probe, host model, naive arithmetic) of the same facts, making drift detectable no matter which path regresses.

**Why they're throwaway.** In a finished kernel `self` fields *are* the ground truth — no three-way cross-check needed, no probe-to-host comparison, no per-stage cache.

## 5. Ten host-reference functions — the dual-reference pattern

Every stage has exactly two references: a canonical one that may share code with the program object, and a `..._naive` rederivation.

| Stage | Canonical | Naive |
|---|---|---|
| full output (S0) | `torch_reference` `:1414` | `torch_reference_naive` `:1459` |
| scheduler (S1) | `torch_reference_s1_scheduler` `:1636` | `torch_reference_s1_scheduler_naive` `:1694` |
| MMA atoms (S2) | `torch_reference_s2_mma_config` `:1828` | `torch_reference_s2_mma_config_naive` `:1847` |
| storage (S3) | `torch_reference_s3_storage_config` `:2047` | `torch_reference_s3_storage_config_naive` `:2062` |
| Q prologue (S4) | `torch_reference_s4_q_prologue` `:2122` | `torch_reference_s4_q_prologue_naive` `:2155` |

**Why we need them for validation.** If the canonical reference uses *any* code path that also populates the device result (same arithmetic helpers, same struct definitions), they can be wrong in the same way. The naive twin re-derives the value from first principles (explicit per-element Python loops in `torch_reference_naive`, manual `_align_up` in `torch_reference_s3_storage_config_naive`) to catch co-drift. S3 even re-duplicates the TMEM-col rounding formula verbatim at `:2069`.

**Why they're throwaway.** Only *one* reference survives into tests — the naive per-element one, because it's the implementation-independent oracle. The canonical variant exists solely because the reviewer compares three-way (device / canonical / naive) during a review round; post-review that's noise.

## 6. Per-stage harnesses and the dispatcher

| Function | Line | Asserts |
|---|---|---|
| `_stage0_prefix_validation_harness` | `:1544` | Full-output (output, lse) exact vs naive, plus shape/dtype/fixture shape. |
| `_stage1_prefix_validation_harness` | `:1742` | Scheduler `debug_token_order` + `debug_token_done_count` exact match and all tokens scheduled exactly once. |
| `_stage2_prefix_validation_harness` | `:1892` | MMA atom `shape_mnk`, iter counts, major-mode flags, "atoms were populated" sentinel. |
| `_stage3_prefix_validation_harness` | `:2197` | 15 checks (see prior doc §4.2). |
| `_stage4_prefix_validation_harness` | `:2291` | Padded Q rows match, trailing rows are zero, live rows nonzero, shape/dtype. |
| `prefix_validation_harness` dispatcher | `:2375-2421` | Routes by `stage_id`, raises if `dependency_validation` missing for S1+. |

**Why we need them for validation.** This *is* the reviewer's pass/fail signal. Each harness `and`s its own checks with the prior stage's `matched`, so a regression anywhere earlier fails the current review too.

**Why they're throwaway.** Tests live in `tests/`, not in the kernel file. In the final shape of the system these would move out and be called from the main harness in `kernel_agents/`.

## 7. `run()` dispatch driver, fixtures, tensor plumbing, helpers

| Identifier | Line | Role |
|---|---|---|
| `SchedulerDebugResult` dataclass | `:238-242` | Wraps scheduler debug tensors. |
| `SyntheticFixture` dataclass + `create_synthetic_data` | `:1289`, `:1344-1413` | Builds the canonical 3-token deterministic fixture with edge cases (all-valid, mixed, all-invalid `-1` sparse indices). |
| `_align_up`, `_format_int_tensor`, `_summarize_tensor` | `:133`, `:1540`, `:1528` | Naive-reference math helper and two pretty-printers used in report strings. |
| `_launch_s1_scheduler_debug_kernel` | `:2419-2628` | Allocates all six `debug_*` host tensors, wraps them via `cute.runtime.from_dlpack(...).mark_layout_dynamic(...)`, launches the debug kernel, returns the captured dict. ~200 lines. |
| `run(stage_id=...)` | `:2631-3015` | The orchestrator: picks fixture, sets probe flags (`:2717-2719`), launches debug kernel, builds 2×5 = 10 host references, calls 1–5 harnesses (cumulative), returns a dict with `matched` / `report` / per-stage outputs. ~380 lines. |
| `kernel(*args, stage_id=...)` | `:3016-3023` | Alias wrapper. |
| `try/except` imports with `# pragma: no cover - stage scaffolding must remain importable` | `:9-34` | Guards against missing CUTLASS pieces so the harness evaluates on CPU-only machines. |

**Why we need them for validation.** `run()` is the entry point the `kernel_agents/tools.py` harness calls — it must accept `stage_id` for all five stages, ship probes, chain harnesses, and emit a single report string. The defensive imports keep that contract honourable in environments without a B200 SDK.

**Why they're throwaway.** The final kernel file exports a single compute path, not a five-way stage switch. Fixtures and launchers move to `tests/` or `scripts/bench.py`. Imports become plain `import`s once the runtime guarantees the SDK is present.

---

## What the final implementation actually keeps

From the current 3023 lines, roughly **500 lines** survive into the final kernel:

- Imports (clean, no `try/except`): replacement for `:9-34`
- Module constants that truly describe the compute (shapes, pipe stages, dtype sizes): a subset of `:40-132`
- `BlackwellStyleKernel.__init__` without the three `enable_*_probe` fields or `_stage3_storage_cache`: trimmed `:277-380`
- `_ensure_stage3_storage_artifacts_jit`: `:437-600` (the real layout construction)
- `Stage3LayoutBackedSharedStorage` (promoted out of the `__call__` closure if possible): `:1114-1186`
- `BlackwellStyleKernel.__call__` trimmed to take real tensors + launch the real kernel: `:1115` truncated
- A new `@cute.kernel` replacing `kernel1_impl` body — warp-specialized, pipelined, following the W0-W7 plan in `kernel_0_plan.md`
- Possibly `torch_reference_naive` as a correctness test reference, moved to `tests/`

Every other artifact in this document is stage-coder ↔ reviewer protocol surface. The "single-thread runners" you flagged are the most visible symptom: they replace whole parallel warp graphs with a single `if tidx == 0` so that the reviewer can diff a device tensor against a Python loop. Useful now, gone at S12.

---

## Updates / corrections

Three clarifications after a second read.

### U1. Q-prologue warps — W3–W7, not W0–W7

The original writeup said the final kernel "distributes Q-prologue work across warps W0–W7." That's wrong. Per `kernel_0_plan.md:240` the prologue is a **W3–W7 cooperative load**:

> `W3-W7` cooperatively load `Q_nope` and `Q_pe` for the token

The kernel is warp-specialized — each warp has a dedicated steady-state role (W0 softmax, W1 correction/epilogue, W2 UMMA, W3 page decode, W4/W5 Kc/Kpe gather, W6/W7 V gather; see the role table at `kernel_0_plan.md:225-234`). The prologue just borrows W3–W7 for a one-shot bulk load before they take up their steady-state jobs; W0/W1/W2 are busy with TMEM allocation and stat init (`plan:242-243`) during that window.

### U2. Single-thread runners are not actually required by the harness

The original writeup justified `_run_scheduler_single_thread` (`:890-958`) and `_run_stage4_q_prologue_single_thread` (`:960-1065`) by claiming the reviewer needs "deterministic, captured per-element snapshots." On re-reading the S4 harness at `kernel_0.py:2297-2323`, that's not what the harness actually checks:

- `torch.equal(cute_result["debug_q_ckv_padded"], naive_result["debug_q_ckv_padded"])`
- `torch.equal(cute_result["debug_q_kpe_padded"], naive_result["debug_q_kpe_padded"])`
- shape/dtype invariants, trailing rows are zero, live rows are nonzero

The harness inspects only the **final** contents of `debug_q_ckv_padded` / `debug_q_kpe_padded`. It does not inspect execution order or `cute.printf` output. A correct W3–W7 cooperative load that commits the same final deterministic bytes would satisfy S4 identically.

So the real reason the single-thread runners exist is simpler: **the cooperative multi-warp load hasn't been written yet**, and `if tidx == 0` + per-element writes is the shortest path to a passing harness. The `cute.printf` spam is debugging aid for when the harness *fails*, never part of the oracle. Once the real cooperative `autovec_copy`-based load exists (using `cute.autovec_copy`, `pipeline.agent_sync`, `cute.arch.setmaxregister_decrease` per `plan:644`), both `_single_thread` runners are pure dead code — not retained scaffolding.

Reclassification for the §1 table: "drop" with the stronger reason that they are **not** load-bearing even now — they are placeholder compute, not validation scaffolding.

### U3. `Stage3LayoutBackedSharedStorage` is redundant — static `Stage3SharedStorage` suffices

Original §3 argued the dynamic `Stage3LayoutBackedSharedStorage` at `:1114-1186` is needed because `cute.cosize(layout)` values can't appear in a module-scope `@cute.struct`. That's a correct statement about CuTeDSL but the wrong conclusion for *this* kernel.

The harness at `:2181` already asserts

```python
torch.equal(cute_result["shared_storage_bytes"], naive_result["shared_storage_bytes"])
```

and it passes. That means `cute.cosize(q_ckv_smem_layout_staged)` already equals `PADDED_Q_HEAD_ROWS * HEAD_DIM_CKV`, and the same for the other five operand buffers. So the static struct's `cute.struct.MemRange[cutlass.BFloat16, PADDED_Q_HEAD_ROWS * HEAD_DIM_CKV]` (`:201-204`) already allocates the exact same byte count as the dynamic cosize-based one.

The `@cute.struct` only owns **bytes + alignment**; the swizzle structure lives on the layout object and is passed separately at `:1103-1109`:

```python
sQ_ckv_staged = storage.smem_q_ckv.get_tensor(
    self.stage3_q_ckv_smem_layout_staged.outer,
    swizzle=self.stage3_q_ckv_smem_layout_staged.inner,
)
```

So `storage.smem_q_ckv` (a plain `MemRange`) + the live layout passed to `get_tensor(...)` gives the identical tiled tensor as the dynamic struct would. The dynamic variant buys nothing the harness observes.

Move `Stage3LayoutBackedSharedStorage` (`:1114-1186`, ~75 lines) + the `SharedStorage = Stage3LayoutBackedSharedStorage` rebind (`:1187`) to the drop list. Launch directly against module-scope `Stage3SharedStorage` with `smem=S3_SHARED_STORAGE_BYTES`.

The only defensive reason to keep the dynamic version is insurance against a future `sm100_utils.make_smem_layout_*` change that introduces swizzle/alignment padding beyond the naive product. But the harness already catches that case — `shared_storage_match` flips to `False` on the next validation run — so the drift is observable, not silent. Adjust the static struct then; don't pay the duplication now.

### Revised "what survives into final impl"

The list at the end of §7 becomes:

- Clean imports (no `try/except`)
- Real compute-path module constants (subset of `:40-132`)
- Trimmed `BlackwellStyleKernel.__init__` (no probe flags, no caches)
- `_ensure_stage3_storage_artifacts_jit` at `:437-600`
- **Static `Stage3SharedStorage` at `:170-228`** — promoted as the launch struct, no dynamic duplicate
- `BlackwellStyleKernel.__call__` stripped of the `@cute.struct` redefinition block
- A new warp-specialized `@cute.kernel` replacing `kernel1_impl` body, following the W0–W2 compute + W3–W7 load specialization from `kernel_0_plan.md:225-234` (prologue: W3–W7 cooperative Q load; steady state: the 7-step pipeline at `plan:247-253`)
- `torch_reference_naive` moved to `tests/` as the single correctness oracle
