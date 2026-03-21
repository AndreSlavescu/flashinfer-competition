# Agent Memory Index

## Performance Values
- [b200_measured_values.md](b200_measured_values.md) — Empirically confirmed B200 TMA gather4 performance (ver5/ver6 + dual_tma_stream ver1): latency, throughput, L2 size, cache hints, pipeline depth, prefetch effectiveness, 32-CTA competition-scale BW, dual-stream interference

## Methodology
- [benchmark_antipatterns.md](benchmark_antipatterns.md) — Known sm_100a benchmark pitfalls (10 patterns): L2-warming confounds latency, competition_realistic caps unique addresses, cudaFuncSetAttribute pointer failure, wrong dim0, swizzle mismatch, cold-latency warmup pre-warming, page-table pattern still L2-warm, false CRITICAL on valid mbarrier ordering, concurrent stream L2 serialization assumption, CSV metadata labels not matching kernel execution

## Design Guidance
- [competition_kernel_insights.md](competition_kernel_insights.md) — Competition kernel recommendations: TMA config, pipeline depth, prefetch (DIST=2), cache hints (ckv=evict_last/kpe=evict_first), dual-stream findings, updated timing model, unknowns requiring UTCMMA + mbarrier benchmarks

## Reference
- [reference_local_sm100.md](reference_local_sm100.md) — Local ground-truth sm100 patterns from references/learn-cuda/02e_matmul_sm100/ and arxiv paper access status
