---
name: v1 kernel design decisions
description: Key architectural decisions for v1 CuTeDSL attention kernel — M=64 padding, split-KV 32 chunks, 8-warp specialization, PyTorch combine, PyTorch indexer
type: project
---

## V1 Attention Kernel Architecture

- M=64 (padded from 16 query heads): UTCMMA requires M=64 or M=128 minimum. Only 16 of 64 rows carry real data. 75% MMA compute wasted.
- Split-KV with 32 chunks (B_TOPK=64 per chunk, topk=2048): Each CTA processes ONE chunk. No online softmax or O-rescale within CTA.
- Grid: (num_tokens, 32, 1). For num_tokens=8: 256 CTAs on 148 SMs.
- 8 warps (256 threads): WG0 (warps 0-3) = softmax+epi, warp 4 = MMA, warp 5 = TMA ckv, warp 6 = TMA kpe, warp 7 = index prep
- TMEM: S(64), O(256), Q_nope(128), Q_pe(16) = 464 cols total
- Smem: union(sQ,sO)=16KB, sKc×2=128KB, sKp×2=16KB, sP=8KB, meta=2KB = ~170KB
- Combine kernel: PyTorch (temporary, to be replaced with CuTeDSL in v2)

## V1 Indexer: PyTorch reference pass-through

## M-dimension Problem for Future
- V2 opportunity: pack 4 tokens into M=64 (16 heads × 4 tokens) when num_tokens >= 4
- V3 opportunity: use M=128 with cta_group::2 (pair SMs)

## TMA Notes
- ckv gather4: 16 calls per chunk, box=[1024B,4], SW128, evict_last
- kpe gather4: 16 calls per chunk, box=[128B,4], SW128, evict_first
- sparse_indices → TMA coord: token_index is already the flat index into [num_pages*64, dim]

**Why:** These decisions trade performance for simplicity in v1. Each one has a known optimization path for v2+.
**How to apply:** Start from this v1 baseline; measure before changing any architectural decision.
