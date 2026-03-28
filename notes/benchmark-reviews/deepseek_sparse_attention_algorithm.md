# DeepSeek Sparse Attention Algorithm

## Notation

Throughout this document, tensor shapes are written with explicit labels:

```
  [num_tokens, 16 heads, 512 elements]
       ^          ^          ^
     axis 0    axis 1     axis 2
```

"Elements" means the raw dimension size (floats/bf16s). When a dimension
represents something specific (heads, tokens, pages), it's always labeled.

## Prerequisites: From Standard Attention to MLA

### Standard Multi-Head Attention (MHA)

In a standard transformer, each attention head independently projects Q, K, V:

```
  For each head h (e.g. 16 heads, each with 128-element head dim):
    Q_h = x @ W_Q_h     [seq_len, 128 elements]
    K_h = x @ W_K_h     [seq_len, 128 elements]
    V_h = x @ W_V_h     [seq_len, 128 elements]

    scores = Q_h @ K_h.T / sqrt(128)   [seq_len, seq_len]
    attn   = softmax(scores)            [seq_len, seq_len]
    out_h  = attn @ V_h                 [seq_len, 128 elements]

  KV cache per token: K[16 heads, 128 elements] + V[16 heads, 128 elements]
                    = 16 * 128 * 2 (K+V) * 2 bytes = 8,192 bytes/token
```

### Grouped Query Attention (GQA)

GQA reduces KV cost by sharing K/V across groups of query heads:

```
  Q heads:  h0  h1  h2  h3  h4  h5  h6  h7    (8 query heads)
             \  /    \  /    \  /    \  /
  KV heads:  kv0     kv1     kv2     kv3        (4 KV heads, 2:1 ratio)

  KV cache per token: K[4 heads, 128 elements] + V[4 heads, 128 elements]
                    = 4 * 128 * 2 * 2 = 2,048 bytes/token  (4x smaller)
```

Each KV head serves a group of query heads. KV cache shrinks by the group ratio.

### DeepSeek Multi-head Latent Attention (MLA)

MLA pushes this idea to the extreme: instead of multiple KV heads, compress
ALL KV information into a **single latent vector** per token with **no head
dimension at all**.

```
  GQA (still per-head KV):          MLA (single latent, no KV head dim):

  Token --+--> W_K_h0 --> K_h0       Token ----> W_DKV ----> c_kv [512 elements]
          +--> W_K_h1 --> K_h1                                 |
          +--> W_V_h0 --> V_h0                           (one vector per token,
          +--> W_V_h1 --> V_h1                            used as BOTH key and value)

  KV cache: K[kv_heads, dim]          KV cache: c_kv[512 elements]  <-- no head dim!
          + V[kv_heads, dim]                  + k_pe[64 elements]
```

**The key MLA innovations:**

1. **c_kv is both key AND value.** The same 512-element latent serves as
   K_nope for computing attention scores AND as V for computing output.
   No separate V projection or cache needed.

2. **No head dimension on KV.** The KV cache stores just one vector per token.
   All 16 query heads attend to this same single representation. Head
   differentiation happens entirely on the query side (16 different Q projections
   that each read the same KV).

3. **Separate positional encoding.** RoPE can't be applied to the compressed
   latent (it would destroy the learned structure). So position info lives in
   a separate small tensor:

```
  MLA's KV cache per token:
    c_kv  [512 elements]   -- compressed latent (serves as BOTH K_nope and V)
    k_pe  [64 elements]    -- positional key (RoPE applied here)
    Total: 576 * 2 bytes = 1,152 bytes/token

  vs. MHA: 8,192 bytes/token  (7x reduction!)
  vs. GQA with 4 heads: 2,048 bytes/token  (1.8x reduction)
```

**How Q interacts with KV — the split matmul:**

The query also has two parts, matching the two KV parts:

```
  Query:  q_nope [16 heads, 512 elements]   -- content query (no position info)
          q_pe   [16 heads, 64 elements]    -- positional query (RoPE applied)

  KV:     c_kv   [512 elements]   -- single vector, no head dim
          k_pe   [64 elements]    -- single vector, no head dim

  Attention score for each of the 16 query heads independently:
    score = q_nope_h @ c_kv.T   +   q_pe_h @ k_pe.T
            [512] @ [512]           [64] @ [64]          = scalar per (head, kv_token)
            ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^
            content matching         position matching

  Over 2048 KV tokens, for all 16 heads at once:
    logits = q_nope @ Kc.T  +  q_pe @ Kp.T
             [16 heads, 512] @ [512, 2048 tokens]  +  [16 heads, 64] @ [64, 2048 tokens]
           = [16 heads, 2048 tokens]
```

Each of the 16 query heads computes its own scores against the SAME set of
KV tokens. The heads differ only because they have different learned Q
projections — the K/V data is shared.

**Why two separate matmuls?** The two KV parts (c_kv and k_pe) are stored in
separate paged caches. Splitting the matmul avoids concatenating them:

```
  Conceptually equivalent (if K were one tensor):
    Q_full = [q_nope | q_pe]     [16 heads, 576 elements]
    K_full = [c_kv   | k_pe]     [2048 tokens, 576 elements]
    logits = Q_full @ K_full.T   [16 heads, 2048 tokens]

  But physically, c_kv and k_pe are in different caches, so we compute:
    logits = (q_nope @ Kc.T) + (q_pe @ Kp.T)     (two matmuls, then add)
```


## The Competition's Sparse Attention Pipeline

The competition implements a decode-time pipeline with two separate kernels.
Each kernel uses **its own set of projections** — the indexer and the attention
kernel operate on different representations of the same underlying tokens.

```
  DeepSeek model produces (during prefill/encoding):

  Per token, TWO separate representations are cached:

    Indexer cache (for cheap scoring):
      k_index  [128 elements, FP8]     Separate projection for token selection
      (+ 1 float32 scale per token)

    Attention cache (for actual attention):
      c_kv     [512 elements, BF16]    Compressed latent (key + value)
      k_pe     [64 elements, BF16]     Positional key

  Per query token, the model also produces:
    q_index  [64 heads, 128 elements]  Index query (separate projection)
    q_nope   [16 heads, 512 elements]  Attention query (NoPE part)
    q_pe     [16 heads, 64 elements]   Attention query (RoPE part)
    weights  [64 elements]             Per-head importance weights for indexer

  These come from DIFFERENT learned projections in the model:
    q_index =/= q_nope, q_pe          (different weight matrices, different dims)
    k_index =/= c_kv, k_pe            (different weight matrices, different dims)
```

The two kernels then operate on these different representations:

```
  +---------------------+  sparse_indices  +---------------------+
  |  TOP-K INDEXER      | ---[2048 tok]--> |  SPARSE ATTENTION   |
  |  (Kernel 1)         |                  |  (Kernel 2)         |
  +---------------------+                  +---------------------+
  Uses: q_index, k_index                   Uses: q_nope, q_pe,
        weights                                  ckv_cache, kpe_cache
  64 index heads, 128 elem                 16 attn heads, 512+64 elem
  FP8, ReLU scoring                        BF16, softmax attention
  Cheap, scores ALL tokens                 Expensive, only 2048 tokens
```

### Kernel 1: Top-K Indexer

Reference: `references/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.py`

**Purpose**: Efficiently score all ~541K KV tokens and select the 2048 most
relevant. Uses a cheap scoring mechanism — NOT real attention.

```
  Inputs:
    q_index         [batch, 64 heads, 128 elements]            FP8
    k_index_cache   [num_pages, 64 tokens/page, 1 head, 132]  FP8 (128 data + 4 scale bytes)
    weights         [batch, 64 elements]                       FP32
    seq_lens        [batch]                                    INT32
    block_table     [batch, max_pages]                         INT32
```

**What are `weights`?**

`weights` is a vector of 64 learned scalars — one per index head.
It has a batch dimension `[batch, 64]` because different sequences
in the batch may have different weights (likely computed by a small
MLP from the current query token during model inference). Think of
it as "how much does each index head's opinion matter when deciding
which tokens are important":

```
  head 0 says token X has score 0.8, weight = 0.1  -->  contributes 0.08
  head 1 says token X has score 0.3, weight = 0.9  -->  contributes 0.27
  ...                                                          sum = 0.35
  head 0 says token Y has score 0.2, weight = 0.1  -->  contributes 0.02
  head 1 says token Y has score 0.7, weight = 0.9  -->  contributes 0.63
  ...                                                          sum = 0.65

  Token Y wins (0.65 > 0.35), mostly because head 1 (high weight) liked it.
```

**Algorithm** (per batch element b):

```
  +-------------------------------------------------------------------+
  | 1. Dequantize FP8 key cache                                       |
  |    K = dequant(k_index_cache)        [seq_len tokens, 128 elems]  |
  |    (FP8 values * per-token float32 scale)                         |
  |                                                                   |
  | 2. Score all tokens per index head                                |
  |    scores = q_index[b] @ K.T                                      |
  |             [64 heads, 128] @ [128, seq_len] = [64 heads, seq_len]|
  |                                                                   |
  | 3. ReLU activation (NOT softmax — this is scoring, not attention) |
  |    scores = relu(scores)             [64 heads, seq_len tokens]   |
  |    Negative scores -> 0 (token is irrelevant to this head)        |
  |                                                                   |
  | 4. Weight by per-head importance and sum across heads             |
  |    weighted = scores * weights[b, :]                              |
  |              [64 heads, seq_len] * [64 heads, 1]  (broadcast)     |
  |    final = sum(weighted, dim=heads)  [seq_len tokens]             |
  |                                                                   |
  |    Each token now has a single importance score.                   |
  |                                                                   |
  | 5. Select top 2048 tokens                                         |
  |    topk_indices = topk(final, k=2048)  [2048 indices]             |
  |                                                                   |
  | 6. Convert to global token indices via page table                 |
  |    page   = topk_indices // 64  (tokens/page)                     |
  |    offset = topk_indices % 64                                     |
  |    global = block_table[page] * 64 + offset                       |
  +-------------------------------------------------------------------+

  Output:
    sparse_indices [batch, 2048 indices]   Global token indices into paged KV cache
```

**Why this design is cheap:**
- 128-element head dim vs 512+64 in attention (fewer FLOPs per dot product)
- FP8 vs BF16 (halves memory bandwidth, which is the bottleneck)
- ReLU instead of softmax (no exp/log, no normalization)
- No output matmul (just scores -> topk selection)
- The weighted sum collapses 64 heads to 1 score (no per-head output needed)


### Kernel 2: Sparse Attention (This Is What You're Building)

Reference: `references/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py`

**Purpose**: Compute full attention over only the 2048 tokens selected by the indexer.

```
  Inputs:
    q_nope          [num_tokens, 16 heads, 512 elements]  BF16
    q_pe            [num_tokens, 16 heads, 64 elements]   BF16
    ckv_cache       [num_pages, 64 tokens/page, 512 elements]  BF16
    kpe_cache       [num_pages, 64 tokens/page, 64 elements]   BF16
    sparse_indices  [num_tokens, 2048 indices]            INT32
    sm_scale        float (~0.135)

  Key shapes to note:
    - q_nope/q_pe have a HEAD dimension (16 heads)
    - ckv_cache/kpe_cache have NO head dimension (single latent per token)
    - All 16 query heads attend to the SAME KV data

  Constants:
    num_tokens = 1-2     (pure decode, single token generation)
    num_pages  = 8462    (~541K total KV tokens at 64 tokens/page)
    topk       = 2048    (sparse selection from indexer)
```

**The full algorithm, step by step:**

```
  For each query token t (num_tokens = 1 or 2):

  STEP 1: Gather KV from sparse indices
  ======================================

    indices = sparse_indices[t]               [2048 indices]

    The indices are GLOBAL token indices encoding page + offset:
      index value = page_idx * 64 (tokens/page) + offset_in_page

    To gather one token at index i:
      page   = i // 64
      offset = i % 64
      ckv    = ckv_cache[page, offset, :]     [512 elements]
      kpe    = kpe_cache[page, offset, :]     [64 elements]

    After gathering all 2048 selected tokens:
      Kc = gathered ckv tokens                [2048 tokens, 512 elements]
      Kp = gathered kpe tokens                [2048 tokens, 64 elements]

    NOTE: these gathers are SCATTERED across ~8462 pages (random access):

    ckv_cache in HBM:
    +------+------+------+------+------+------+------+
    |page 0|page 1|page 2| .... |page N|      |      |
    +------+------+------+------+------+------+------+
       ^           ^       ^                ^
       |           |       |                |    Scattered reads!
     idx[0]     idx[3]  idx[1]           idx[7]  (random pages)

    2048 tokens * 512 elements * 2 bytes/elem = 2.1 MB (ckv)
    2048 tokens * 64 elements  * 2 bytes/elem = 0.26 MB (kpe)
    Total: ~2.36 MB per query token, scattered across ~8462 pages
    This scattered memory access is the kernel's bottleneck.


  STEP 2: Compute attention logits (split matmul)
  =================================================

    All 16 query heads compute scores against the SAME 2048 KV tokens.
    There is no KV head dimension — just one shared KV representation.

    logits = (q_nope[t] @ Kc.T) + (q_pe[t] @ Kp.T)

    Breaking this down:

      NoPE scores (content matching — "what is this token about?"):
        score_nope = q_nope[t] @ Kc.T
        shapes:      [16 heads, 512 elements] @ [512 elements, 2048 tokens]
        result:      [16 heads, 2048 tokens]

      RoPE scores (position matching — "where is this token?"):
        score_rope = q_pe[t] @ Kp.T
        shapes:      [16 heads, 64 elements] @ [64 elements, 2048 tokens]
        result:      [16 heads, 2048 tokens]

      Combined:
        logits = score_nope + score_rope    [16 heads, 2048 tokens]

    Each head gets its own [2048] score vector, but they all scored
    against the same KV tokens. The heads differ because each has its
    own learned q_nope and q_pe projection.


  STEP 3: Scale
  ==============

    logits_scaled = logits * sm_scale        [16 heads, 2048 tokens]

    sm_scale ~ 1/sqrt(576) ~ 0.0417, but competition uses ~0.135
    (model-specific, provided as input)


  STEP 4: Handle invalid indices
  ===============================

    Some of the 2048 indices may be -1 (padding / fewer valid tokens):

    for each index position i in 0..2047:
      if sparse_indices[t, i] == -1:
        logits_scaled[all 16 heads, i] = -infinity

    After softmax, -inf logits become zero attention weight.


  STEP 5: Softmax + LSE
  =======================

    Per-head softmax over the 2048 token dimension:

      attn[t] = softmax(logits_scaled, dim=tokens)   [16 heads, 2048 tokens]

    Also compute LSE (Log-Sum-Exp) in log-base-2 for the split-KV combine:

      lse[t] = logsumexp(logits_scaled, dim=tokens) / log(2)   [16 heads]
             = log2( sum( exp(logits_scaled), over 2048 tokens ) )


  STEP 6: Compute output (S @ V, but V = Kc!)
  ==============================================

    THE KEY MLA INSIGHT: Value IS the compressed latent (ckv).
    We reuse the SAME Kc data that we already gathered for the QK matmul.

    output[t] = attn @ Kc
    shapes:     [16 heads, 2048 tokens] @ [2048 tokens, 512 elements]
    result:     [16 heads, 512 elements]

    No separate V gather needed! The ckv data serves double duty.

    Data flow showing Kc used twice:

      q_nope ----+
                 |---> NoPE scores -+
      Kc --------+                  |
      |                             +--> logits --> softmax --> attn weights
      |   kpe ---+                  |                              |
      |          |---> RoPE scores -+                              |
      q_pe -----+                                                  |
                                                                   |
      Kc -----+                    (same tensor, reused!)          |
              |---> output = attn @ Kc  <--------------------------+
              |
      (V=Kc)--+
```

### Why "Sparse"?

```
  Full attention:     541,248 tokens * 576 elements * 2 bytes = ~594 MB per query
  Sparse attention:     2,048 tokens * 576 elements * 2 bytes = ~2.3 MB per query
                                                                 ~~~~
                                                                 258x less memory!

  Quality loss is minimal because attention weights are naturally concentrated
  on a small subset of tokens. The indexer identifies that subset cheaply.
```

## Online Softmax (For Tiled Computation)

The kernel can't compute softmax over all 2048 tokens at once — it processes
them in tiles of 64 tokens (B_TOPK=64). This requires **online softmax**,
which maintains a running max and sum that get corrected as new tiles arrive.

The key challenge: softmax requires knowing the global max across ALL tokens
to compute exp(x - max) without overflow. But we only see 64 tokens at a time.

**Solution**: Track a running max. When a new tile has a larger max, rescale
all previous work by the correction factor exp(old_max - new_max).

```
  Per head (all 16 heads do this independently):

  Initialize:
    mi = -inf        (running max of scaled logits)
    li = 0           (running sum of exp(logits - mi))
    O  = [0]*512     (running output accumulator, 512 elements)

  For each tile of 64 KV tokens:

    1. Compute logits for this tile:
       P = q_nope @ K_tile_c.T + q_pe @ K_tile_p.T    [64 tokens]
       P_scaled = P * sm_scale                          [64 tokens]

    2. Find tile max (per head):
       tile_max = max(P_scaled)

    3. Decide whether to rescale:
       new_mi = max(mi, tile_max)
       need_rescale = (new_mi > mi)  ?

    4. If rescale needed — correct ALL previous work:
       scale = exp(mi - new_mi)     (< 1.0, shrinks old values)
       O  *= scale                  (rescale all 512 elements of output)
       li *= scale                  (rescale running sum)

    5. Compute tile softmax values (relative to current max):
       S = exp(P_scaled - new_mi)   [64 tokens]

    6. Update running sum:
       li += sum(S over 64 tokens)

    7. Accumulate output (V = Kc, same data gathered for QK):
       O += S @ K_tile_c            [512 elements]

    8. Update max:
       mi = new_mi

  After all 32 tiles (2048/64 = 32 tiles):
    output = O / li                           (normalize)
    lse    = log2(li) + mi                    (log-sum-exp in log2 base)

  Worked example (showing state after each of 3 tiles, for one head):

  After tile 0:             After tile 1:              After tile 2:
  mi=-inf -> 3.2            mi=3.2 -> 3.5              mi=3.5 (unchanged)
  li=0 -> 45.1              li=45.1 -> 73.8            li=73.8 -> 102.3
                                |                         |
                                v                         v
  tile_max=3.2               tile_max=3.5              tile_max=3.1
  3.2 > -inf: rescale!      3.5 > 3.2: rescale!       3.1 < 3.5: no rescale
  scale=exp(-inf-3.2)=0     scale=exp(3.2-3.5)=0.74   scale=1.0
  O *= 0 (zeroed out)       O *= 0.74 (shrunk!)       O unchanged
  li *= 0 (zeroed)          li = 45.1*0.74 = 33.4     li stays 73.8
  S = exp(P - 3.2)          S = exp(P - 3.5)          S = exp(P - 3.5)
  li = 0 + 45.1 = 45.1      li = 33.4 + 40.4 = 73.8  li = 73.8 + 28.5 = 102.3
  O = S @ V_tile0            O += S @ V_tile1          O += S @ V_tile2
```

## Split-KV: Parallelizing Across SMs

With only 1-2 query tokens and B200's ~160 SMs, a single SM processing
all 32 tiles (2048 tokens / 64 per tile) would leave most SMs idle.
**Split-KV** distributes tiles across multiple SMs:

```
  2048 tokens / 64 per tile = 32 tiles per query token

  Without split-KV (1 SM per query):
  +------+------+------+------+------+     +-------+
  |tile 0|tile 1|tile 2|tile 3| .... |     |tile 31|   SM #0 does all 32 (slow)
  +------+------+------+------+------+     +-------+
  SMs 1-159: idle!

  With split-KV (distribute across SMs):
  SM #0:   [tile 0, tile 1, tile 2, tile 3 ]  -> partial (O_0, LSE_0)
  SM #1:   [tile 4, tile 5, tile 6, tile 7 ]  -> partial (O_1, LSE_1)
  SM #2:   [tile 8, tile 9, tile 10, tile 11]  -> partial (O_2, LSE_2)
  ...
  SM #7:   [tile 28, tile 29, tile 30, tile 31] -> partial (O_7, LSE_7)
  SMs 8-159: handle query token 2 (if present), or idle

  Each SM runs online softmax over its subset of tiles.
  Then a COMBINE kernel merges the partial results.
```

### The Combine Algorithm

Each SM produces partial output with its own LSE (log-sum-exp). These
can't simply be averaged — softmax is non-linear. Instead, the partials
are combined using LSE-based rescaling:

```
  Given N partial results (one per SM split):
    (O_0, lse_0), (O_1, lse_1), ..., (O_{N-1}, lse_{N-1})
    where lse_i is in log2 base
    and O_i is already divided by its local li (normalized partial output)

  For each of the 16 heads independently:

  1. Find global max LSE:
     max_lse = max(lse_0, lse_1, ..., lse_{N-1})

  2. Compute total sum in log2 space:
     sum_exp = sum( exp2(lse_i - max_lse) for i in 0..N-1 )

  3. Global LSE:
     global_lse = log2(sum_exp) + max_lse

  4. Weighted combination of partial outputs:
     O_final = zeros [512 elements]
     for each split i:
       scale_i = exp2(lse_i - global_lse)     (this split's fraction of total mass)
       O_final += scale_i * O_i

  Why this works:
    lse_i represents log2(sum of exp(scores) in split i)
    scale_i = 2^(lse_i) / 2^(global_lse)
            = (probability mass in split i) / (total probability mass)
    So O_final is a weighted average, with weights = fractional probability mass.
```

## Summary: The Complete Decode Pipeline

```
  +-------------------+
  | All ~541K tokens  |
  | in paged KV cache |
  +---------+---------+
            |
            v
  +----------------------+
  | TOP-K INDEXER        |<-- q_index  [batch, 64 index heads, 128 elements] (FP8)
  | (Kernel 1)           |<-- k_index  [num_pages, 64 tok/page, 128 elements] (FP8)
  |                      |<-- weights  [batch, 64 per-head weights] (FP32)
  | Separate projections |
  | from attention Q/K!  |    score = relu(q @ K.T) * w    (per index head)
  |                      |    final = sum(score, over heads) (one score/token)
  | Score ALL tokens     |    indices = topk(final, k=2048)
  | cheaply (FP8, ReLU)  |
  +----------+-----------+
             |
             | sparse_indices [num_tokens, 2048 indices]
             v
  +-----------------------+
  | SPARSE ATTENTION      |<-- q_nope    [num_tokens, 16 attn heads, 512 elements] (BF16)
  | (Kernel 2)            |<-- q_pe      [num_tokens, 16 attn heads, 64 elements]  (BF16)
  |                       |<-- ckv_cache [num_pages, 64 tok/page, 512 elements]    (BF16)
  | Different projections |<-- kpe_cache [num_pages, 64 tok/page, 64 elements]     (BF16)
  | from indexer Q/K!     |
  |                       |    KV has NO head dimension - all 16 Q heads share it
  | Phase 1: Schedule     |    Partition 2048 indices across SMs
  | Phase 2: Kernel       |    Per-SM: gather KV, QK, softmax, SV
  | Phase 3: Combine      |    Merge partial (O, LSE) across SMs
  +-----------+-----------+
              |
              v
  output [num_tokens, 16 attn heads, 512 elements] (BF16)
  lse    [num_tokens, 16 attn heads]                (FP32, log2 base)
```
