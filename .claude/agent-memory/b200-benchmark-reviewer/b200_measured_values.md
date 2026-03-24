---
name: B200 Confirmed Measured Performance Values
description: Empirically confirmed B200 performance values from real Modal hardware runs, with measurement context
type: project
---

Confirmed B200 (sm_100a) performance from tma_gather4 benchmark, ver5 CSV, CUDA 13.0, 148 SMs:

**TMA gather4 latency (single CTA, pipeline N=1)**
- L2-RESIDENT total: ~225 ns (includes barrier overhead ~28 ns)
- L2-RESIDENT net (minus barrier): ~197 ns (ckv_int64, 4096B), ~182-190 ns (kpe_bf16/ckv_bf16, 512B)
- Context: 1024-iteration loop, 16MB unique data, all L2-resident after warmup
- HBM-COLD net: **546 ns (ckv_int64, 4096B)**, **492 ns (kpe_bf16, 512B)** — ver5 NEW MEASUREMENT
- HBM-cold context: sequential non-repeating access of entire 1024MB working set
- IMPORTANT: The ~197 ns L2-resident value does NOT match the published 420-cycle HBM miss latency. That 420-cycle figure is for ld.global, not TMA. TMA HBM-miss latency is ~1100 cycles (546 ns at 2.1 GHz) — 2.77× higher than ld.global because TMA goes through a DMA pipeline with additional overhead.
- Null barrier overhead (expect_tx + arrive + try_wait): ~28 ns, stable across configs

**TMA gather4 pipeline N=2 speedup (ver5 confirmed)**
- N=1: 225 ns/gather4
- N=2: 174 ns/gather4 (1.29× improvement)
- N=4..32: 168-174 ns/gather4 (diminishing, <4% more)
- Conclusion: N=2 pipeline stages is optimal for single-CTA TMA

**TMA gather4 aggregate throughput (76 CTAs, ckv_int64 4096B/gather4, ver5)**
- L2-resident (≤64MB): ~900-904 GB/s
- HBM-bound (256MB+): ~390-405 GB/s (random)
- competition_realistic/pt at 256MB: ~897-908 GB/s — MISLEADING, only 2048 unique rows = 2MB unique data, all L2-resident regardless of declared working set
- Single CTA: ~5.68 GB/s HBM-bound

**TMA gather4 multi-CTA saturation (ver5, extended to 296 CTAs)**
- 76 CTAs: 405 GB/s; 152 CTAs: 788 GB/s; 176: 906; 224: 1151; 256: 1309; 296: 1511 GB/s
- Still NO HBM saturation at 296 CTAs (= 2 CTAs/SM for all 148 SMs)
- 1511 GB/s = 18.9% of 8 TB/s HBM peak — serial-issue TMA is latency-bound, not BW-bound
- Slight superlinear efficiency improvement at 176+ CTAs (memory controller operates more efficiently under higher concurrency)

**B200 L2 cache size (empirical, confirmed ver5 + deviceQuery)**
- Cliff between 64MB (902 GB/s) and 128MB (540 GB/s) in TMA gather4 benchmark
- deviceQuery on Modal B200 confirms: **L2 = 126.5 MB** (132,644,864 bytes) — corrects earlier ~64 MB estimate
- The 64/128 MB cliff is consistent: 64 MB working set fits in L2, 128 MB does not (126.5 MB capacity)
- Results: `microbenchmarks/device_query/results/device-query-b200-ver1.txt`

**L2 promotion size effect (ver5 NEW)**
- L2_PROMOTION_L2_128B vs L2_PROMOTION_L2_64B: ~0% difference at all working set sizes
- Cliff location identical for both. Use L2_128B as default; L2_64B offers no benefit.

**TMA box size effect on throughput (ver5 confirmed)**
- 256B/gather4 → 289 GB/s; 512B → 330; 1024B → 371; 2048B → 402; 4096B → 405 (HBM-bound, 76 CTAs, 256MB)
- Larger boxes amortize per-barrier overhead. Data type (INT64 vs BF16) irrelevant at same bytes/gather4.

**kpe TMA (512B/gather4, bf16 dim0=64)**
- ~53 GB/s HBM-bound at 76 CTAs
- L2-hot latency: ~182-190 ns net (single CTA)
- HBM-cold latency: ~492 ns net (NEW ver5)
- Much slower aggregate throughput than ckv because smaller box = more barriers per unit data

**Cache hints (ver5 confirmed)**
- evict_last vs evict_normal vs none: essentially identical (< 5% difference) at HBM-bound scale
- evict_first: -48% for sequential L2-resident at 64MB. Avoid for cached data.
- Competition recommendation: use evict_last (never worse, best for L2-resident cases)

**TMA gather4 prefetch effectiveness (ver6 NEW)**
- `tma_gather4_prefetch()` (`cp.async.bulk.prefetch.tensor.2d.L2...tile::gather4`) tested at DIST=0,1,2,4
- DIST=0 baseline: 537.7 ns (1.2% faster than latency_cold due to spurious same-row prefetch — negligible artifact)
- DIST=1: ~380 ns (29% latency reduction)
- DIST=2: ~324 ns (40% latency reduction) — **saturation point**
- DIST=4: ~324 ns (no improvement over DIST=2)
- Conclusion: issue prefetch for block i+2 while computing on block i; 2 outstanding requests saturate the TMA prefetch pipeline

**32-CTA competition-scale TMA throughput (ver6 NEW)**
- At 32 CTAs (competition workload: 1 CTA per split-KV block × 32 blocks):
  - 64MB (L2-resident): ~393 GB/s
  - 256MB (HBM-bound): ~174 GB/s
  - 1024MB (HBM-bound): ~166 GB/s
- 2.32× slower than 76-CTA results. L2 cliff at same 64-128MB location.

**FRESH competition scatter throughput (ver6 NEW)**
- FRESH_COMPETITION_REALISTIC at 1024MB: **385.73 GB/s** (within 1.1% of pure random 389.95 GB/s)
- competition_realistic at 1024MB: ~905 GB/s (L2-warm artifact — only 8MB unique data)
- B_TOPK=64 sort structure adds zero measurable benefit in HBM-bound regime
- The 906 GB/s figure is NOT the real competition bandwidth — use 386 GB/s for planning

**Cache hints — kpe update (ver6 NEW)**
- kpe_bf16 `evict_first` at 64MB (L2 boundary, sequential): **65 GB/s — 47% penalty** vs evict_last (122 GB/s)
- kpe_bf16 `evict_first` at HBM-bound (256MB+): **+5% faster** than evict_last
- Updated recommendation: ckv=evict_last (keeps data in L2), kpe=evict_first (frees L2 for ckv, marginal HBM gain)

**Dual TMA stream interference (dual_tma_stream ver1 NEW)**
- HBM-cold (sequential 1024MB), BOTH mode: ckv **7.58 GB/s isolated = 7.58 GB/s concurrent** (0% interference)
- L2-warm (random 256MB), BOTH mode: kpe 201 ns isolated → **595 ns concurrent** (≈ HBM cold, 3× slowdown)
- Root cause: ckv's 256MB working set evicts kpe's 8MB data from L2 → kpe serializes to HBM
- In production (8MB KV fits in L2), ckv will dominate L2 — kpe must be budgeted at HBM-cold latency (~595 ns)

**Why:** These values are from actual B200 hardware runs on Modal, ver5/ver6 and dual_tma_stream ver1. Use these over CLAUDE.md specs which are literature estimates.
**How to apply:** When analyzing future benchmarks or designing the competition kernel, use these as ground truth for B200 TMA performance.

**TMA gather4 competition-realistic RANDOM SCATTER (ver9 NEW — 2026-03-23)**
- Pattern: fresh_competition_realistic, 1024 MB, single CTA
- ckv_int64 (4096B/gather4) N=1: **650.9 ns** (vs 546 ns sequential — +19% random scatter overhead)
- kpe_bf16 (512B/gather4) N=1: **609.9 ns** (vs 492 ns sequential — +24% random scatter overhead)
- ckv N=16 pipelining: 234.5 ns (2.77× speedup vs N=1); 17.47 GB/s throughput
- kpe N=16 pipelining: 229.3 ns (2.66× speedup vs N=1); 2.23 GB/s throughput
- Pipeline saturation for random scatter does NOT plateau at N=2 (unlike sequential); continues to N=16
- 16-call ckv block (N=16): **~3,752 ns** (vs 3,690 ns ver8 sequential — similar but random is ~1.7% slower)
- 16-call kpe block (N=16): **~3,673 ns** (extrapolated)
- kpe latency ≈ ckv latency at N=1 despite 8× smaller payload: confirms TMA overhead-dominated regime
- Throughput ratio ckv:kpe = 7.84× ≈ bytes_per_call ratio (8×): throughput scales linearly with box size
- UPDATED COMPETITION BASELINE: Use N=16 pipelining with random scatter. Competition TMA-to-GEMM ratio = 2.49× (revised from 3.6-6.8×)

---

## UTCMMA and TMEM Performance (utcmma-ver3.csv, 2026-03-19)

**Sustained clock rate**: 1.854 GHz (stable across ver1 and ver3; 18% below 2.25 GHz advertised boost).

**UTCMMA RAW latency (single accumulator, dependency chain)**

| Tile | cy/K-tile | ns/K-tile |
|---|---|---|
| M=64 N=64 k16 | 54.5 | 29.4 |
| M=64 N=128 k16 | 54.5 | 29.4 |
| M=64 N=256 k16 | 76.0 | 41.0 |
| M=128 N=128 k16 | 74.0 | 39.9 |
| M=128 N=256 k16 | 138.0 | 74.5 |
- K-depth linearity: ±2% from k=1 to k=32 (M=64 N=128)
- K_DEPTH = number of K-tiles (each 16 BF16 elements) along the GEMM reduction dimension
- QK GEMM (M=64 N=128 k_depth=32): **1,719 cycles = 928 ns** (competition QK is N=64, but per-K-tile cost is identical to N=128)
- SV GEMM single tile (M=64 N=256 k_depth=4 SS): **320 cycles = 173 ns**; competition SV needs 2 tiles (N=256×2=512 output) = **640 cy = 346 ns**

**CRITICAL: The arxiv:2512.02189 figure of ~11 cycles is for independent-accumulator throughput
(initiation interval), NOT the RAW dependency chain. Competition kernel serialized RAW chain
is ~54 cycles/K-tile — 5× slower than the paper's figure.**

**UTCMMA multi-accumulator throughput** (N_ACC=1..4, all identical):
- Rotating accumulator TMEM columns provides ZERO speedup for N_ACC ≤ 4.
- Structural initiation interval appears to be ~54 cycles.
- N_ACC ≥ 5 still untested; N_ACC=8 (M=64 N=64) needed for definitive conclusion.

**WS vs non-WS**:
- M=64 N=64: identical (54.5 cy both)
- M=64 N=128: WS=54.5 cy, non-WS=64.0 cy (+17%). Always use .ws variants for N≥128.

**TMEM load RAW-serialized latency (utcmma-ver8, 2026-03-23 — confirmed valid)**:
- Model (32dp32b mode): **cy = 1.74 + 0.031 × WIDTH** (R² ≈ 0.999), where WIDTH = number of 32-bit replications per call
- 32dp32b×1: 1.80 cy (near loop floor, effectively free); 32dp32b×64: 3.73 cy; 16dp256b×16: 9.74 cy
- Cross-mode consistency: 32dp32b×W and 16dp128b×(W/2) produce identical cycle counts at equal bit volumes (validation of measurement correctness)
- 16dp256b×1 (2.23 cy) > 16dp128b×2 (1.98 cy) at equal 256-bit output: 16dp256b has higher per-instruction overhead (reads 2 TMEM columns per replication vs 1)
- Competition: large accumulator reads (32dp32b×128 = 4096 bits) cost ~5.72 cy (interpolated); dominated by surrounding MMA latency (173+ cy)
- Prior "~1.8 cycles floor" remains correct for small W; the model now explains the scaling behavior

**TMEM store latency** (N=64 + fence): **49.8 cycles = 26.8 ns** (linear: 12 + 0.59×N cy)

**Fence costs**:
- Warp-collective TMEM wait pair (load+store fence, nothing in flight): **12.83 cycles**
- Per-thread tcgen05 fence pair (before/after_thread_sync): **2.83 cycles**

**Full O-rescale roundtrip** (isolated measurement):
- 1 chunk (N=64 ld + 32×float2_mul + N=64 st): **168.1 cycles = 85.6 ns**
- 4 chunks (full D=512 width): **622.1 cycles = 316.6 ns**
- NOTE: O-rescale applies only to online softmax designs where one SM processes multiple KV chunks sequentially. In split-KV (competition design), each SM processes one B_TOPK=64 chunk with a single softmax pass — no O-rescale needed.

**Synchronization costs**:
- named_bar.sync(128 threads): **20.67 cycles = 10.5 ns**
- named_bar.sync(32 threads): **14.65 cycles = 7.5 ns**
- mbarrier arrive+wait (expect_tx=0): **88.51 cycles = 45.0 ns**
- Full MMA + commit + wait (M=64 N=64): **173.03 cycles = 93.4 ns**
- tcgen05.commit overhead: ~30 cycles
- NOTE: In a pipelined kernel, mbarrier waits synchronize TMA completion — they're part of TMA latency, not additive overhead. Per-SM mbarrier count = ~32 waits (one per gather4 call: 16 ckv + 16 kpe).

**UTCCP (smem→TMEM copy) latency**:
- 1 copy (128×16 bf16 tile): 168.3 cy = 85.6 ns (startup cost)
- Marginal cost at 16 copies: ~74.7 cy/copy = 38.0 ns/copy
- Competition Q load (32 copies extrapolated): ~2,276 cycles = ~1,228 ns

**UTCMMA M=32 WS-TS latency (ver6 NEW — 2026-03-23)**:
- M=32 N=64 k=1: **173.0 cy = 93.9 ns** (identical to M=64 N=64)
- M=32 N=128 k=1: **173.0 cy = 93.9 ns** (38% faster than M=64 N=128 at 238.5 cy)
- M=32 N=256 k=1: **238.5 cy = 129.5 ns** (identical to M=64 N=256)
- M=32 N=64 k=4: 310.4 cy = 168.5 ns; 77.59 cy/K-tile
- M=32 N=64 k=32: **1,814 cy = 984 ns**; 56.69 cy/K-tile (vs M=64: 60.16 cy/tile — 5.8% faster)
- Clock check: 173.0 / 93.9e-9 = 1.842 GHz (consistent with prior 1.844 GHz)
- N=64 and N=128 have identical completion at M=32 (same output element count: 32×64 = 32×64... wait, they differ. M=32 N=64=2048 ops, M=32 N=128=4096 ops — yet same latency. Confirmed N-independence holds for N≤128 at M=32.)
- M=32 saves only 110 cy (59 ns) vs M=64 for k=32. With TMA at 3,752 ns, saving is negligible (1.6% end-to-end).

**cp.async.bulk S2G performance (bulk-copy-s2g-ver1 NEW — 2026-03-23)**:
- Plain copy latency: 67.3 cy (512B) to 4,147 cy (128KB); startup ~34 ns; marginal BW ~15.9 GB/s
- reduce_add latency: +18–25% vs plain copy across all sizes (read-modify-write overhead)
- Single-CTA throughput ceiling: **~62.6 GB/s** (32KB, N_PIPE=16 — essentially identical to N_PIPE=1 at 60.3 GB/s)
- Pipelining N_PIPE 1→16 gives only **+3.8%** — cp.async.bulk.global does NOT benefit from pipelining (unlike TMA gather)
- True async: copy hides completely within compute with only ~11 cy issue overhead when compute >> copy time
- At c1000 FMA iters: +1,029 ns overhead from bulk_wait_group flushing memory pipeline (even if copy is done)
- Conclusion: Issue S2G reduce_add early; 18-25% cost is fixed and must be accepted for accumulation

**LDG.256 Memory Hierarchy — absolute-pointer p-chase, iters=max(num_slots,20000) (ldg-pchase-sweep-ver1 — 2026-03-23)**:

SUPERSEDES ldg-hints-ver1 latency values which had ~160 cy of ALU overhead (% and & on critical path)
AND wrong iters (too few to traverse full WS ring — measured L1/L2 cached subset, not declared WS).

Corrected B200 memory hierarchy (en_en_128B = coherent, L1+L2 normal caching):

| WS       | cy    | ns    | Level                                |
|----------|-------|-------|--------------------------------------|
| 4–8 KB   | ~36   | ~18   | **L1 hit** (floor)                   |
| 16 KB    | 40    | 20    | L1 spilling                          |
| 32 KB    | 47    | 24    | L1 partial miss                      |
| 48 KB    | 54    | 27    | L1 boundary region                   |
| 64 KB    | 60    | 31    | L1/L2 mix                            |
| 128 KB   | 87    | 45    | L1→L2 transition                     |
| 256 KB   | 255   | 130   | L2 hit (approaching steady-state)    |
| 512KB–32MB | ~300 | ~153  | **L2 steady-state** (plateau)        |
| 64 MB    | 327   | 167   | L2 capacity pressure                 |
| 128 MB   | 535   | 273   | **L2→HBM transition**                |
| 192 MB   | 640–655 | 326–333 | HBM                               |
| 256 MB   | **689–707** | **351–360** | **HBM deep cold** (two independent Modal runs: ver1=707 cy, ver4=689 cy, ~2.5% variation) |

Clock validation: cy/ns = 1.961–1.967 GHz (matches deviceQuery 1965 MHz). ✓

L1-bypass (na_ef_128B) penalty:
- At 4 KB: 36 cy → 279 cy (7.7× penalty)
- At 32 KB: 47 cy → 278 cy (5.9× penalty)
- At 512 KB+: IDENTICAL to en_en (L2/HBM regime — hint irrelevant)

nc_en_en_128B = en_en_128B at ALL sizes (0% difference on B200).

Competition recommendation: use `en_en_128B` for all index loads. NEVER `na_ef`.
- sparse_indices per block (256 B – 8 KB): pure L1 hit, 36 cy / 18 ns
- Full topk=2048 indices (8 KB): L1 hit, 36 cy vs 279 cy with na_ef (7.7× worse)

Cross-reference vs arxiv:2512.02189 "~420 cy global memory latency": Their figure is L2 latency (WS fits L2);
our L2 = ~300 cy (~153 ns); their 420 cy/200 ns at 2.1 GHz likely reflects ~256-512 KB WS, not deep HBM.
Our HBM = 689–707 cy / 351–360 ns across two independent Modal runs — more rigorous (random Hamiltonian p-chase,
full ring traversal, no prefetcher). Range represents B200 instance-to-instance variation (~2.5%).

**LDG.256 single-CTA L2 streaming BW (32-thread sequential, 256 MB, ver2/ver3)**: ~1,049–1,072 GB/s.
All 18 hints identical for sequential streaming (prefetcher dominates, hints irrelevant).
For HBM-bound hint differentiation at streaming, need multi-block loads saturating HBM.

**Ver5 UTCMMA completion latency (full dataset, corrected sync):**
- WS-TS M64N256 k=1: 238.5 cy = 129.3 ns
- WS-TS M128N128 k=1: 238.5 cy = 129.3 ns
- WS-TS M128N256 k=1: 370.5 cy = 200.9 ns
- WS-SS M64N128 k=1: 175.3 cy = 95.0 ns
- WS-SS M64N256 k=4 (competition SV): **539 cy = 292 ns**
- Non-WS TS M64N64 k=1: 173.0 cy (identical to WS-TS)
- Non-WS TS M64N128 k=1: 238.5 cy (+38% vs WS-TS M64N64)
- Non-WS SS M64N64 k=1: 175.3 cy; M64N128 k=1: 262.8 cy; M64N256 k=4: 730.7 cy
- Fine-N sweep (N=8..64, k=1): ALL produce IDENTICAL 173.0 cy (TS) / 175.3 cy (SS) — N-independent latency
- UTCCP 128dp256bit 16-copy: 74.7 cy/copy; UTCCP 128dp128bit 32-copy: 51.3 cy/copy
- GPU clock inferred from ver5: 173.0 cy / 93.8 ns = **1.844 GHz** (6.2% below 1.965 GHz boost)
- Spread: 0.0% on all MMA experiments — extremely stable

**N_ACC multi-accumulator (ver5, definitive with commit+wait):**
- N_ACC=1..5 at M64N64 k_depth=32: all identical **60.16 cy/tile** (structural floor confirmed)
- Cannot achieve paper's 11 cy/tile with N_ACC ≤ 5; likely requires N_ACC ≥ latency/initiation ≈ 173/11 ≈ 16

**Competition kernel per-SM timing model (split-KV, B_TOPK=64, ver3 corrected)**:

TMA per block (16 gather4 calls each for ckv and kpe):
- NOTE: all TMA latency values above (546/595 ns) are PER-GATHER4-CALL, not per-block total
- **Competition baseline = HBM-cold** (evaluation runs on separate machine, no guaranteed cache warmth)
- HBM-cold sequential (no pipelining): 16 × 595 = ~9,520 ns per stream
- HBM-cold N=16 batch-issued (ver8 MEASURED for ckv): **3,690 ns** per 16-call block
- ckv and kpe run in parallel; HBM-cold concurrent interference: +0% ckv, +7% kpe (ver1 confirmed)
- **Per-block TMA: ~3,690 ns** (ckv ver8 measured; kpe N=16 unconfirmed — extrapolated)

GEMM per block (serialized RAW chain, all values measured):
- QK ckv (M=64 N=64 K_DEPTH=32): ~1,719 cy = ~928 ns
- QK kpe (M=64 N=64 K_DEPTH=4): ~218 cy = ~118 ns
- SV (2 × M=64 N=256 K_DEPTH=4): 2 × 320 = ~640 cy = ~346 ns
- **Per-block GEMM total: ~2,577 cy = ~1,392 ns**

With N=64 WS tiles (split-KV, 1 block/SM): GEMM needs all 64 KV tokens loaded → TMA and GEMM sequential.
**Per-block total = TMA + GEMM = ~6,400-10,900 ns. Kernel is TMA-BOUND (~3.6-6.8×).**
