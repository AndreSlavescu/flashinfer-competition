# Kernel Optimizer Agent Memory

## References
- [CuTeDSL FMHA architecture](reference_cutedsl_fmha_architecture.md) — Blackwell FMHA warp roles, pipelines, TMEM layout, mainloop, adaptation notes for competition
- [TMEM architecture](reference_tmem_architecture.md) — Tensor Memory architecture, constraints, UMMA interaction, allocation patterns

## Competition Workload (AUTHORITATIVE)

From `competition-dataset/workloads/dsa_paged/`:

**Attention** (`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl`):
- 23 workloads, **num_tokens=[1,2,6,7,8]** — NOT just 1-2
- num_pages=8462, sm_scale≈0.1352, sparse_indices are real safetensors (not random)
- At num_tokens≥5: 32×N > 148 SMs → multi-block-per-SM unavoidable → inter-block pipelining pays off

**Indexer** (`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl`):
- 128 workloads, **batch_size up to 31** ([1,2,3,4,6,7,8,11,12,14,15,16,25,26,27,29,30,31])

## B200 Hardware Ground Truth (deviceQuery 2026-03-21)
- L2 cache: **126.5 MB** (132,644,864 bytes) — NOT ~64 MB
- SM clock: 1,965 MHz | warp_id%4 sub-core mapping confirmed
- Results: `microbenchmarks/device_query/results/device-query-b200-ver1.txt`

## Local CuTeDSL References (Phase 1 priority)
- `references/flash-attention/flash_attn/cute/flash_fwd_sm100.py` — FA4 forward, nearest architectural match
- `references/flash-attention/flash_attn/cute/blackwell_helpers.py` + `mma_sm100_desc.py`
- `references/CuTeDSL-kernels/quack/gemm_sm100.py` — production sm100 GEMM
- `references/learn-cuda/07_attention/` — attention v1–v5 progressive implementations
- `references/learn-cuda/02e_matmul_sm100/` — ground-truth tcgen05/TMEM/warp-spec patterns
