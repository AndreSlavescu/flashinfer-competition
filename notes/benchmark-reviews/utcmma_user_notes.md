 ---
  Q1: RAW chain — per-matmul or across matmuls? Can GEMMs pipeline?

  The ~54 cy/K-tile RAW chain is WITHIN each GEMM, not across them. Each K-tile
  MMA accumulates into the same TMEM columns, so MMA[i+1] must wait for MMA[i] to
   drain its TMEM write before the accumulator is readable again. This gives a
  ~54-cycle floor per tile, giving:
  - QK_ckv: 32 K-tiles × 54 cy = 1,719 cy
  - QK_kpe: 4 K-tiles × 54 cy = 218 cy
  - SV: 2 × (4 K-tiles × 80 cy) = 640 cy

  Across the three GEMMs, you CAN overlap QK_ckv and QK_kpe, because they write
  to independent TMEM columns (independent accumulators). SV cannot start until
  you've computed softmax(QK_ckv + QK_kpe), which is a true semantic dependency.
  So the critical path is:

  QK_ckv (1719 cy) ─┐
                     ├─ add + softmax ─ SV (640 cy)
  QK_kpe (218 cy)  ─┘

  With parallelism: max(1719, 218) + softmax + 640 ≈ 1719 + ~100 + 640 = ~2,460
  cy instead of 2,577 cy (sequential). A minor win but essentially QK_kpe is
  "free" relative to QK_ckv.

  ---
  Q2: O-rescale — what is it and is it needed?

  O-rescale is the online softmax correction factor applied when one SM iterates
  over KV chunks sequentially. For each new chunk with logits that may exceed the
   previous running max, O must be rescaled: O = O × exp(old_max − new_max)
  before accumulating the new partial attention. This requires reading O out of
  TMEM, multiplying by the correction, and writing back — that's the 168–622 cy
  cost measured.

  In split-KV (competition design):
  - Each SM gets exactly one chunk of B_TOPK=64 tokens
  - It computes softmax over those 64 tokens in a single pass — no running max
  update needed
  - It writes (O_partial, LSE_partial) out, and a separate combine kernel does
  the final merge using LSE values

  No O-rescale is needed within each SM's kernel. The combine kernel does a
  simple weighted sum using exp(LSE_partial_i - LSE_global) — but that's scalar
  per-head, not TMEM round-trip.

  ---
  Q3: 32 mbarriers and sequential TMA timing

  You're right on both counts.

  Mbarrier count: The "32 barriers" note was over-counting based on a naive
  1-barrier-per-gather4 design. You can absolutely issue all 16 ckv gather4s + 16
   kpe gather4s pointing to the same mbarrier with expect_tx = 16×4096 + 16×512 =
   73,728 bytes, then wait once. The TMA DMA engine queues outstanding requests
  and pipelines them. That's the whole point.

  For double-buffering (N=2 pipeline), you'd need 2 mbarriers (ping/pong), not
  32.

  Sequential timing model: The 16 × 595 ns = 9,520 ns estimate is the worst case
  (wait after each gather4 individually). The realistic model depends on the TMA
  pipelining efficiency when all 16 gather4s are issued at once — which is the
  top-priority unmeasured quantity. From the single-gather4 loop benchmark: N=1 →
   546 ns, N=2 (with DIST=2 prefetch) → 324 ns. If 16 outstanding requests
  pipeline similarly, total could be as low as 546 + 15 × throughput_gap, which
  is much less than 9,520 ns. That experiment needs to be run.

  ---
  Q4: The 11-cycle paper claim

  You're probably right. Our data: ~54 cycles RAW chain for N=64/128, ~76 for
  N=256. The paper's 11 cycles is almost certainly the initiation interval (MMA
  issue rate with independent accumulators), not the RAW latency with TMEM
  readback. This is the standard microbenchmark ambiguity — measuring throughput
  vs. latency.

  For the competition kernel where every K-tile accumulates into the same TMEM
  columns, you get the full ~54 cycle RAW latency. The paper's 11-cycle figure is
   irrelevant for this use case.

  On TMA + UTCMMA overlap by further K-tiling: Yes, that's exactly what it means.
   If instead of loading all 64 KV tokens and then computing, you load 16 at a
  time:

  Issue TMA for tokens 0-15 → Wait → Compute QK partial (with running max)
  Issue TMA for tokens 16-31 (concurrent with above compute) → Wait → Compute QK
  partial + O-rescale
  ...
  After 4 sub-tiles: softmax, SV

  This enables TMA/GEMM overlap within the 64-token block, at the cost of 3
  O-rescale operations (~3 × 168 cy = 504 cy overhead). Whether it's a net win
  depends on whether TMA is truly the bottleneck (the 5,000–9,500 ns vs 1,392 cy
  GEMM suggests TMA dominates so heavily that overlap would be very valuable).

  ---
  Q5: The FlashMLA pipeline — does it apply to split-KV?

  FlashMLA uses warp specialization:
  - Producer warpgroup (WG0): continuously issues TMA gather4s into smem
  ping/pong buffers
  - Consumer warpgroups (WG1, WG2): compute QK + SV on already-loaded data
  - TMA and GEMM overlap across KV blocks

  This pipeline requires multiple KV blocks per SM to keep both WGs busy. In
  FlashMLA's standard (non-split) decode, one SM processes all B_TOPK=2048 tokens
   sequentially in 32 blocks of 64 — plenty of blocks to pipeline.

  In split-KV (competition), each SM processes only 1 block (64 tokens). The
  cross-block producer/consumer pipeline has nothing to pipeline into. WG0 would
  issue 16 gather4s, and then there's no next block. So FlashMLA's WS pipeline
  design doesn't directly apply to single-block split-KV.

  To get TMA/GEMM overlap you'd need the intra-block sub-tiling approach from Q4
  above. The relevant question is whether the overlap gain (TMA latency hidden
  behind compute) justifies the O-rescale overhead. Given the timing model shows
  TMA is ~3.6–6.8× the bottleneck, hiding even half the TMA cost would roughly
  halve total kernel time — so yes, it's almost certainly worth benchmarking.

  The summary timing model should be:
  Naive:       TMA(5,000-9,500 ns) + GEMM(1,392 ns) = 6,400-10,900 ns
  With overlap: max(TMA, GEMM) + O-rescale ≈ ?  ← needs measurement

  The next benchmark to build is 16-consecutive-gather4 pipelining to measure
  actual throughput vs the sequential estimate, and then TMA+UTCMMA concurrent
  execution with sub-tiling
  