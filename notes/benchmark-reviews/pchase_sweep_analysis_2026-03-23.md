# B200 P-Chase Sweep Analysis — 2026-03-23

## Benchmark: ldg_pchase_sweep (ver1)

Results: `microbenchmarks/ldg_hints/results/ldg-pchase-sweep-ver1.csv`

**Methodology**: Absolute-pointer Hamiltonian-cycle p-chase. Each 32-byte slot stores an absolute device
pointer to the next slot in a random permutation. Chase step = `ptr = (const uint64_t*)buf[0]` — zero ALU
overhead on the critical path. Following arxiv:1509.02308 and the gpu-benches implementation.

**Key fix over ver3**: `iters = max(num_slots, 20000)` ensures at least one full ring traversal. Previous
version (ver3) used fixed iters=500-50000, so at large WS only `iters*32` bytes were visited (L1-resident),
making all WS > 512 KB look identical at ~277 cy (which was actually the L1/L2 hit latency, not HBM).

---

## B200 Memory Hierarchy (LDG.256, absolute-pointer p-chase)

### Staircase (en_en_128B = coherent, L1+L2 normal)

| Working Set | Cycles | ns    | Cache Level                         |
|-------------|--------|-------|-------------------------------------|
| 4 KB        | 36.0   | 18.3  | **L1 hit** (floor)                  |
| 8 KB        | 36.8   | 18.7  | L1 hit                              |
| 16 KB       | 40.2   | 20.4  | L1 spilling (above L1 hot region)   |
| 32 KB       | 46.8   | 23.9  | L1 partial miss                     |
| 48 KB       | 53.5   | 27.3  | L1 boundary region                  |
| 64 KB       | 60.3   | 30.7  | L1/L2 mix                           |
| 128 KB      | 87.3   | 44.5  | L1→L2 transition                    |
| 256 KB      | 255.4  | 130.0 | L2 hit (approaching steady-state)   |
| 512 KB      | 299.0  | 152.2 | **L2 steady-state** (~300 cy)       |
| 1 MB        | 300.3  | 152.9 | L2 plateau                          |
| 4 MB        | 300.4  | 152.9 | L2 plateau                          |
| 16 MB       | 300.4  | 152.9 | L2 plateau                          |
| 32 MB       | 300.4  | 152.9 | L2 plateau                          |
| 64 MB       | 327.3  | 166.6 | L2 capacity pressure (L2 = 126.5MB) |
| 128 MB      | 535.3  | 272.5 | **L2→HBM transition**               |
| 192 MB      | 654.6  | 333.3 | HBM                                 |
| 256 MB      | 706.9  | 359.9 | **HBM** (deep cold)                 |

**Clock validation**: cycles/ns at all sizes = 1.964–1.967 GHz. Matches deviceQuery `1965 MHz`. ✓

---

## na_ef_128B (L1-bypass) vs en_en_128B

| Working Set | en_en cy | na_ef cy | Penalty  | Explanation                  |
|-------------|----------|----------|----------|------------------------------|
| 4 KB        | 36.0     | 278.5    | **7.7×** | L1 bypass forces L2 hit      |
| 32 KB       | 46.8     | 277.6    | **5.9×** | L1 bypass forces L2 hit      |
| 128 KB      | 87.3     | 299.1    | 3.4×     | L1 miss anyway, still L2     |
| 512 KB      | 299.0    | 299.9    | ≈1.0×    | Both L2 → identical          |
| 32 MB       | 300.4    | 300.5    | ≈1.0×    | Both L2 → identical          |
| 128 MB      | 535.3    | 536.6    | ≈1.0×    | Both HBM → identical         |
| 256 MB      | 706.9    | 707.6    | ≈1.0×    | Both HBM → identical         |

**Key finding**: `na_ef` only hurts when the working set fits in L1 (≤ ~64 KB). Beyond L1, the hint makes
no latency difference for random-access p-chase.

**Competition implication**: sparse_indices for one block = 64 tokens × 4 bytes = 256 bytes (L1). For full
topk=2048 indices (8 KB), L1 hit at 36 cy. Use `en_en_128B`. Never `na_ef` for index loads.

---

## nc (non-coherent) vs coherent

`nc_en_en_128B` = `en_en_128B` at ALL working set sizes (0% difference). The `.nc` flag on B200 does not
change latency for this access pattern — confirmed at L1, L2, and HBM.

---

## Cross-Reference with Prior Work

| Source | Metric | Their Value | Our Measurement | Notes |
|--------|--------|-------------|-----------------|-------|
| arxiv:2512.02189 (B200) | Global mem latency | ~420 cy (~200 ns @ 2.1 GHz) | 707 cy, 360 ns @ 1.965 GHz | See note below |
| arxiv:2507.10789 (GB203) | L1 latency | 30–40 cy | **36 cy** ✓ | Matches exactly |
| arxiv:2507.10789 (GB203) | HBM latency | ~877 cy | 707 cy | B200 faster than RTX 5080 |

### Note on arxiv:2512.02189 discrepancy

Their "~420 cycles" is most likely measuring L2 latency (not HBM), or uses a hardware-prefetcher-friendly
sequential access pattern. Our random Hamiltonian chase at 256 MB (full HBM) gives 707 cy. Possible
reconciliations:
- At L2 steady-state we measure ~300 cy; at 2.1 GHz that's 143 ns (~their 200 ns with L2 working set)
- Their working set may have fit mostly in L2
- Sequential stride (prefetcher exploitable) would show lower apparent latency

Our measurement is more rigorous for random-access latency representative of sparse_indices loading.

---

## Memory Hierarchy Summary (B200, measured via LDG.256 p-chase)

| Level | Latency (cy) | Latency (ns) | Effective Boundary |
|-------|-------------|-------------|-------------------|
| L1 hit | ~36 cy | ~18 ns | ≤ ~32 KB (random-access) |
| L1/L2 mix | 40–87 cy | 20–45 ns | 32–128 KB transition |
| L2 hit | ~300 cy | ~153 ns | 128 KB – 64 MB plateau |
| L2→HBM | 327–535 cy | 167–273 ns | 64–128 MB transition |
| HBM | ~655–707 cy | ~333–360 ns | > 128 MB |

---

## LDG.256 Hint Recommendation for Competition

| Use Case | WS Size | Recommendation | Reason |
|----------|---------|----------------|--------|
| sparse_indices per block (B_TOPK=64) | 256 B | `en_en_128B` | Pure L1 (36 cy) |
| Full topk=2048 indices | 8 KB | `en_en_128B` | L1 hit (36 cy); na_ef = 279 cy (7.7×) |
| Index buffer full pass | 64–128 KB | `en_en_128B` | L1 partial (60–87 cy); na_ef = 299 cy |
| **Never use for indices** | any | `na_ef_*` | Bypasses L1; 5–8× penalty at L1-fittable |

---

## Throughput Experiment (256 MB, 32 threads)

All 18 hints identical at ~1,072 GB/s for sequential streaming. Expected: sequential access is
prefetcher-friendly → all hints converge. Single-warp L2 streaming BW = ~1,072 GB/s.
To differentiate hints for HBM throughput, need multi-block loads.

---

## Methodology Notes

1. **CRITICAL**: iters must be `>= num_slots = WS_bytes / 32`. Less = measuring L1/L2 cached subset.
2. **Absolute pointer p-chase**: `ptr = (const uint64_t*)buf[0]` eliminates `%` and `&` ALU overhead.
   Old formula added ~170+ cycles overhead (old 256 MB latency = 456 cy was wrong; true = 707 cy).
3. **Hamiltonian cycle**: Random permutation visits every slot exactly once per pass.
4. **Warmup = 200 iters**: Warms 200 slots (6.4 KB). For WS >> 6.4 KB, timing loop is cold.
5. **Reference**: gpu-benches uses `iters = max(LEN, 100000)`; NVIDIA-Hopper uses `max(LEN, 1000000)`.
   Our `max(num_slots, 20000)` is slightly more conservative but correct.
