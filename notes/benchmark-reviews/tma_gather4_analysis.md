Now let me analyze the results against B200 specs.

Results Analysis
B200 Reference Specs (from arxiv 2512.02189)
Spec	Value
Peak HBM3e BW	8 TB/s
Sustained HBM (STREAM Triad)	~4.14 TB/s (51.8%)
L2 cache	~64 MB
SMs	148
Global mem latency (miss)	~420 cycles (~200ns at 2.1 GHz)
Sanity Checks
1. L2 cliff matches spec. Your L2 sweep (Exp 5) shows 462 GB/s at 64MB, drops to 272 GB/s at 128MB. The cliff at ~64-128MB is consistent with the ~64MB L2.

2. Latency matches. Net gather4 latency is ~191-195ns regardless of L2 hit/miss. At ~2.1 GHz that's ~400-410 cycles, closely matching the published ~420 cycle HBM miss latency. The near-identical L2-hot vs L2-cold numbers mean TMA pipeline overhead dominates — the TMA issue→DMA→barrier→completion chain has ~190ns constant cost.

3. Aggregate throughput is plausible but low. Peak 409 GB/s at 152 CTAs = only ~5% of peak HBM. This is expected for random scatter-gather — DRAM row buffer misses kill bandwidth. The benchmark uses 1 thread per CTA issuing serial gather4 calls, so per-CTA TMA utilization is minimal.

4. Linear CTA scaling (not saturated). 1→152 CTAs scales nearly linearly, meaning the HBM system has massive headroom. The real kernel (with pipelined multi-warp TMA + compute overlap) will achieve much higher throughput.

Key Findings for Competition
BF16 > INT64 packing (2x throughput): BF16 box=256 gets 410 GB/s vs INT64 box=64 at 210 GB/s for the same 256MB working set. BF16 issues fewer gather4 calls with more col_steps, amortizing the ~50ns barrier overhead per call.

Index sorting is huge: competition_realistic at 256MB hits 463 GB/s vs 210 GB/s for random — a 2.2x improvement. Sorting sparse indices within B_TOPK=64 blocks creates L2-friendly locality. This is an easy, high-impact optimization for the competition kernel.

Cache hints don't matter for random scattered access above L2 capacity. All hints give ~210 GB/s at 256MB random. evict_first actually hurts sequential patterns at 64MB (253 vs 475 GB/s).

Pipeline depth 2+ is sufficient. N=1→N=2 gives 22% improvement (224→173 ns/gather4), then diminishing returns to N=32 (169 ns). Barrier overhead ≈ 50ns, amortized by N≥2.

Swizzle has zero impact on gather4 throughput. BF16 box=64 NONE vs 128B: 54.4 vs 54.3 GB/s. Swizzle is applied by TMA during smem write — doesn't affect HBM→L2 fetch speed. Choose swizzle mode based on the UTCMMA consumer, not TMA.

Latency concern
The latency experiment shows nearly identical L2-hot vs HBM numbers (191 vs 195 ns). This is because with num_iters=1024 and a 256MB address space, we only touch 1024 * 4 * 512B = 2MB of unique data, which fits in L2 after first access. Consider increasing num_iters or using a larger working set to get a true HBM-miss latency measurement.

The CSV field fixes are done — run_throughput now reports num_iters and avg_latency_ns, run_latency reports elapsed_ms and throughput_GBps, and run_pipeline reports throughput_GBps and num_iters. Re-run on Modal to get the complete CSV.

The B200 specs agent confirmed the numbers I used in the analysis above. The results are consistent with the hardware specs — no anomalies to flag beyond the latency caveat I mentioned.
