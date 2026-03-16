# Agent Memory Index

## Performance Values
- [b200_measured_values.md](b200_measured_values.md) — Empirically confirmed B200 TMA gather4 performance from Modal hardware (ver4 CSV): latency, throughput, L2 size, cache hint effects, pipeline depth

## Methodology
- [benchmark_antipatterns.md](benchmark_antipatterns.md) — Known sm_100a benchmark pitfalls: L2-warming confounds latency measurement, competition_realistic pattern caps unique addresses, cudaFuncSetAttribute with function pointer fails, wrong dim0 for competition config

## Design Guidance
- [competition_kernel_insights.md](competition_kernel_insights.md) — Specific competition kernel recommendations from tma_gather4 results: TMA config, pipeline depth, index sorting, cache hints, working set analysis
