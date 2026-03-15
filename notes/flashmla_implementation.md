# FlashMLA Implementation: Line-by-Line Mapping

This document maps the DeepSeek sparse attention algorithm to the FlashMLA
CUDA implementation on SM100 (B200 Blackwell).

## Architecture Overview

```
  +------------------------------------------------------------------+
  |                    HOST (Python / C++ API)                        |
  |  csrc/api/sparse_decode.h                                        |
  |                                                                  |
  |  Phase 1: Schedule --> get_decoding_sched_meta_kernel             |
  |  Phase 2: Kernel   --> flash_fwd_splitkv_mla_fp8_sparse_kernel    |
  |  Phase 3: Combine  --> flash_fwd_mla_combine_kernel               |
  +------------------------------------------------------------------+

  +------------------------------------------------------------------+
  |                   PHASE 2: MAIN KERNEL                           |
  |  csrc/sm100/decode/head64/kernel.cuh                             |
  |                                                                  |
  |  Grid: (s_q, num_sm_parts)    Block: 384 threads                  |
  |                                                                  |
  |  +----------------------------+  +-----------------------------+ |
  |  | WG0: Softmax (128 threads) |  | WG1: GEMM+Data (128 thds)  | |
  |  | Warps 0-3                  |  | Warp 4: UTCMMA (QK, SV)    | |
  |  |                            |  | Warp 5: TMA gather NoPE    | |
  |  | Load P from TMEM           |  | Warp 6: TMA gather RoPE    | |
  |  | Reduce dual-gemm halves    |  | Warp 7: Index transform    | |
  |  | Mask invalid tokens        |  +-----------------------------+ |
  |  | Online softmax (max, exp)  |  +-----------------------------+ |
  |  | Rescale O accumulator      |  | WG2: Dequant (128 threads)  | |
  |  | Write S to shmem           |  | FP8 -> BF16 conversion      | |
  |  | Store final O / LSE        |  | (NOT needed for competition) | |
  |  +----------------------------+  +-----------------------------+ |
  +------------------------------------------------------------------+
```

## Phase 1: Scheduling

**File**: [get_decoding_sched_meta.cu](csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu)

**Purpose**: Partition the topk blocks across SM partitions for split-KV parallelism.

```
  With topk=2048, B_TOPK=64:  32 blocks per query token
  With num_tokens=1, num_sms=160:  up to 160 SM partitions

  The scheduler distributes blocks to balance load:

  SM Part 0:  blocks [0, 4)     for batch 0
  SM Part 1:  blocks [4, 8)     for batch 0
  ...
  SM Part 7:  blocks [28, 32)   for batch 0
  SM Part 8:  blocks [0, 4)     for batch 1    (if num_tokens=2)
  ...
  SM Parts 16-159: empty (begin_req_idx >= b, early exit)
```

**Key data structure** ([params.h:10-16](csrc/params.h#L10)):
```c
struct DecodingSchedMeta {
    int begin_req_idx, end_req_idx;     // Batch item range [inclusive, inclusive]
    int begin_block_idx, end_block_idx; // Block range [inclusive, exclusive)
    int begin_split_idx;                // Split index for combine kernel
    int is_first_req_splitted;          // Is this batch item split across SMs?
    int is_last_req_splitted;
};
```

**Algorithm** (lines 61-99, single-threaded on thread 0):
```
payload = ceil(total_blocks / num_sm_parts) + fixed_overhead

for each SM partition i:
  Greedily assign blocks from current batch item
  If payload exhausted mid-batch: record split point, move to next SM
  If batch item completed: move to next batch item

Output: DecodingSchedMeta[num_sm_parts] + num_splits[batch_size+1]
```

The `num_splits` array is a prefix sum: `num_splits[b]` = total splits for
batches 0..b-1. The combine kernel uses this to find all partials for batch b.


## Phase 2: Main Kernel

**File**: [kernel.cuh](csrc/sm100/decode/head64/kernel.cuh)

### Configuration

**File**: [config.h](csrc/sm100/decode/head64/config.h)

Key constants ([config.h:33-50](csrc/sm100/decode/head64/config.h#L33)):
```
D_Q     = 576 (V32) or 512 (MODEL1)    Query/Key total dimension
D_V     = 512                           Value dimension (= output dim)
D_NOPE  = 512 (V32) or 448 (MODEL1)    NoPE (compressed latent) dimension
D_ROPE  = 64                            RoPE (positional) dimension
B_H     = 64                            Heads processed per block
B_TOPK  = 64                            KV tokens per tile
NUM_BUFS     = 2                        Double-buffering for KV
NUM_INDEX_BUFS = 4                      Quad-buffering for indices
NUM_THREADS  = 384                      3 warpgroups x 128 threads
```

### Memory Hierarchy

```
  +------------------+
  | HBM (Global)     |    Q, KV cache (paged), indices, output
  | 8 TB/s bandwidth |    ~2.3 MB of scattered KV reads per query
  +--------+---------+
           | TMA gather (async, DMA engine)
           v
  +------------------+
  | Shared Memory    |    ~200 KB per SM
  | (SMEM)           |    KV tiles, Q, softmax scores, indices, barriers
  +--------+---------+
           | UTCCP (async copy), explicit loads
           v
  +------------------+
  | Tensor Memory    |    512 columns, on-chip per-SM
  | (TMEM)           |    O accumulator, Q, P (attention scores)
  +------------------+    Only accessible by tensor core (UTCMMA) + tmem_ld/st
           |
           v
  +------------------+
  | Tensor Core      |    UTCMMA instructions
  | (TCGen05)        |    QK^T matmul, SV matmul
  +------------------+
```

TMEM column layout ([config.h:72-80](csrc/sm100/decode/head64/config.h#L72)):
```
  TMEM Columns:    0          256       400    464    512
                   |           |         |      |      |
                   +----O------+----Q----+--P---+      |
                   | 64h x 512d| 64h x   |64h x |      |
                   | fp32 accum| D_NOPE  |64 tok|      |
                   |  (output) | (query) |(QK^T)|      |
                   +-----------+---------+------+------+

  O:  columns 0-255    Output accumulator [B_H=64 heads, D_V/2=256]
                       (fp32, 2 elements packed per column)
  Q:  columns 256-399  Query tensor loaded via UTCCP from shared memory
  P:  columns 400-463  QK^T attention scores (dual-gemm layout, see below)
```

Shared memory layout ([config.h:159-194](csrc/sm100/decode/head64/config.h#L159)):
```
  SharedMemoryPlan (~200 KB):

  +-- Union 1: QO / KV (mutually exclusive in time) ----+
  |                                                       |
  |  QO mode (during Q load):                             |
  |    q[B_H * D_Q_SW128]     Query (SW128 layout)        |
  |    q_sw64[B_H * D_Q_SW64] Query tail (SW64, V32 only) |
  |    o_buf / o_accum_buf     Output (bf16 or fp32)       |
  |                                                       |
  |  KV mode (during main loop):                          |
  |    dequant[2]:             Double-buffered             |
  |      nope[B_H * D_NOPE]   Dequantized NoPE (bf16)    |
  |      rope[B_H * D_ROPE]   Dequantized RoPE (bf16)    |
  |    raw_nope[2]:            Raw FP8 NoPE data          |
  |                                                       |
  +-------------------------------------------------------+
  |  s_p:                                                  |
  |    p_exchange_buf[4][...]  For reducing P across warps |
  |    s[B_H * B_TOPK]        Softmax scores (bf16)       |
  +-------------------------------------------------------+
  |  rowwise_max_buf[128]      For exchanging row maxima   |
  |  is_token_valid[4][8]      Validity bitmasks           |
  |  tma_coord[4][64]          TMA gather coordinates      |
  |  scales[4][64][N]          FP8 dequant scales          |
  |  barriers (16+)            Transactional barriers      |
  +-------------------------------------------------------+
```

### Initialization (Lines 24-68)

```
  Line 25-30:  Extract block/thread indices
               s_q_idx = blockIdx.x        (query position)
               partition_idx = blockIdx.y   (SM partition for split-KV)

  Line 35-41:  Prefetch TMA descriptors (warp 0, thread 0)
               These describe the global tensor layouts for TMA ops

  Line 43-66:  Initialize transactional barriers (warp 0, thread 0)
               Each barrier tracks expected bytes for async operations

  Line 63-65:  Allocate 512 TMEM columns (for O, Q, P)
```

### Main Loop Dispatcher (Lines 77-118)

The `run_main_loop` lambda handles the split-KV scheduling:

```
  Line 81-88:  Load DecodingSchedMeta for this partition (256-bit global load)

  Line 90-92:  Early exit if no work assigned (partition beyond batch size)

  Line 96-117: Loop over batch items assigned to this partition
               For each batch item:
                 - Compute block range [start_block_idx, end_block_idx)
                 - Determine if this is a split (partial) or full computation
                 - Package into MainLoopArgs and call the warpgroup-specific handler
                 - Sync all threads between batch items (NamedBarrier everyone_sync)
```

### Ring State (Lines 120-132)

Double-buffering state machine for pipelining KV loads with compute:

```
  buf_idx:       0 or 1         Which KV buffer is active
  bar_phase:     0 or 1         Which barrier phase to wait on
  index_buf_idx: 0,1,2,3        Which index buffer (quad-buffered)

  rs.update() advances all indices and flips phases at wrap-around.

  The pipeline:
    Iteration i:   Load KV[buf=0], Compute on KV[buf=1]
    Iteration i+1: Load KV[buf=1], Compute on KV[buf=0]
```

### WG0: Softmax Warpgroup (Lines 134-422)

This is the online softmax consumer. 128 threads (warps 0-3), 224 registers each.

**Per-block loop** (lines 159-299) — this is the heart of the softmax:

```
  Line 155-157:  Initialize running state
                 mi = -1e30 (running max, not -inf to avoid NaN from -inf - -inf)
                 li = 0     (running sum of exp)
                 real_mi = -inf  (true running max for detecting lonely queries)

  For each block (B_TOPK=64 KV tokens):

    WAIT FOR DATA:
    Line 161:    Sync WG0 (ensure intermediate buffers are free)
    Line 162:    Wait for validity mask from Warp 7 (bar_valid_coord_scale_ready)
    Line 163:    Wait for QK^T result from Warp 4 (bar_qk_done)
    Line 164:    TMEM fence (ensure TMEM writes from UTCMMA are visible)

    LOAD P FROM TMEM (lines 167-175):
    +-----------+
    | P in TMEM |    P is in "dual-gemm" layout: two 32-column halves
    | cols 400  |    Each warp owns either left or right half
    | to 463    |    tmem_ld reads 32 fp32 values per thread
    +-----------+

    The dual-gemm layout means P is split across warps like this:
    +--------+--------+
    |        |        |
    | Warp 0 | Warp 2 |   32 heads x 32 tokens each
    |        |        |
    +--------+--------+
    |        |        |
    | Warp 1 | Warp 3 |
    |        |        |
    +--------+--------+
    64 heads x 64 tokens total

    Lines 168-174: Warps 0,1 load left half as "p" and right as "p_peer"
                   Warps 2,3 load right half as "p" and left as "p_peer"

    REDUCE P HALVES (lines 178-195):
    The dual-gemm computes P in two halves that must be summed.
    Each warp stores its "peer" half to shared memory exchange buffer,
    then loads the other warp's data and adds element-wise.
    After reduction: each thread has the full P value for its head/token.

      Warp 0 stores p_peer to exchange_buf[2]  (for warp 2)
      Warp 2 stores p_peer to exchange_buf[0]  (for warp 0)
      barrier (warp 0 <-> warp 2 sync)
      Warp 0 loads exchange_buf[0], adds to p
      Warp 2 loads exchange_buf[2], adds to p

    MASK INVALID TOKENS (lines 210-216):
    Line 211:    Load 32-bit validity bitmask from is_token_valid buffer
    Lines 213-215: For each of B_TOPK/2 tokens, if bit is 0: p[i] = -inf

    COMPUTE ROW-WISE MAX (lines 218-230):
    Lines 219-223: Local max across B_TOPK/2 elements (unrolled)
    Line 224:      Multiply by sm_scale_div_log2 (scale factor / log2)
    Line 226:      Store to shared memory rowwise_max_buf
    Line 227:      Sync WG0
    Line 229:      Load peer thread's max (thread XOR 64), take max
                   Now cur_pi_max is the full row max for this head

    ONLINE SOFTMAX UPDATE (lines 231-260):

    Line 231:    should_scale_o = any thread has (cur_pi_max - mi > 6.0)?
                 Threshold of 6.0 is ~exp(6)=403x error tolerance.
                 Avoids rescaling O for small max changes (optimization).

    Lines 237-245: Compute scale factor for old accumulator
                   If should_scale_o:
                     new_max = max(cur_pi_max, mi)
                     scale_for_old = exp2(mi - new_max)    (< 1.0, shrinks old)
                   Else:
                     scale_for_old = 1.0  (no rescaling needed)
                     new_max = mi

    Lines 248-259: Compute softmax S values and running sum
                   For each pair of P values:
                     d = fma(p, scale, -new_max)    (scale and shift in log2 space)
                     d = exp2(d)                     (exponentiate)
                     cur_sum += d                    (accumulate sum)
                     s = convert_to_bf16(d)          (for S@V matmul)
    Line 260:      li = fma(li, scale_for_old, cur_sum.x + cur_sum.y)
                   Update running sum: old_sum * scale + new_sum

    WRITE S TO SHARED MEMORY (lines 262-266):
    S values (bf16) are written to shmem in the layout expected by UTCMMA_SS.

    RESCALE O IN TMEM (lines 268-291):
    If should_scale_o and not first block:
      Load O from TMEM in chunks of 64 fp32 values
      Multiply each by scale_for_old
      Store back to TMEM
    This corrects the running output for the new max value.

    SIGNAL S READY (line 294):
    bar_so_ready.arrive() — tells Warp 4 that S is in shared memory

  END OF MAIN LOOP (lines 299-325):

  Line 301-306:  Handle "lonely queries" (no valid tokens):
                 If real_mi is still -inf, set li=0 and mi=-inf

  Lines 308-311: Exchange li across thread pairs (idx XOR 64)
                 Each thread had half the columns; now has full sum

  Lines 314-325: Store LSE to global memory
                 If no-split (single SM handled entire query):
                   lse = mi * ln(2) + log(li)    Convert log2 -> natural log
                 If split (partial result):
                   lse = log2(li) + mi            Keep in log2 for combine kernel
```

**Output epilogue** (lines 327-421):

```
  Line 327-328:  Wait for final SV GEMM to complete (bar_sv_done)

  NO-SPLIT PATH (lines 335-385): Direct output to global memory
    Line 346:    o_scale = 1/(li + exp2(attn_sink - mi))
                 Incorporates attention sink into the normalization
    Lines 350-384: Load O from TMEM, scale, convert fp32->bf16,
                   write to shared memory, then TMA store to global

  SPLIT PATH (lines 386-421): Output to o_accum buffer for combine
    Line 387:    o_scale = 1/li   (no attn_sink here; combine handles it)
    Lines 390-406: Load O from TMEM, scale, store to shmem (fp32)
    Lines 409-419: Bulk copy from shmem to global o_accum
```

### WG1: GEMM + Data Movement (Lines 427-746)

128 threads, 72 registers each (saves registers for WG0/WG2).

#### Warp 4: UTCMMA GEMM Producer (Lines 431-585)

This warp runs all three matrix multiplications.

**Q Loading** (lines 438-527):

```
  Lines 438-461: TMA load Q from global to shared memory
                 Two loads: Q_SW128 (first 512d) + Q_SW64 (last 64d, V32 only)

  Lines 464-513: UTCCP: Copy Q from shared memory to TMEM
                 SM100_UTCCP_128dp256bit_1cta::copy() — async copy
                 Q ends up in TMEM columns 256-399

  Line 526:      Wait for UTCCP to complete (bar_q_utccp)

  Lines 517-523: Allocate TMEM tensor views for P and O
                 P: TMEM column 400 (for QK^T output)
                 O: TMEM column 0   (for output accumulator)
```

**Main compute loop** (lines 530-584):

For each KV block of B_TOPK=64 tokens:

```
  GEMM 1: QK^T — Compute attention scores P

    V32 model (lines 532-552):
      Step 1: P  = Q_rope @ K_rope.T    (lines 535-542)
              [64h, 64d] x [64d, 64tok] -> [64h, 64tok] in TMEM
              utcmma_ts: TMEM(Q_rope) x SMEM(K_rope) -> TMEM(P)
              zero_init = true (first matmul clears P)

      Step 2: P += Q_nope @ K_nope.T    (lines 545-552)
              [64h, 512d] x [512d, 64tok] -> [64h, 64tok] in TMEM
              utcmma_ts: TMEM(Q_nope) x SMEM(K_nope) -> TMEM(P)
              zero_init = false (accumulate onto P from step 1)

    MODEL1 (lines 553-571):
      Single dual-GEMM: P = Q @ K.T     (lines 566-571)
              [64h, 512d] x [512d, 64tok] -> [64h, 64tok]
              Dual-gemm processes NoPE+RoPE together

    Line 573: Signal P ready: umma_arrive(bar_qk_done)

  Wait for S (line 576):
    bar_so_ready.wait() — WG0 has computed softmax S from P

  GEMM 2: SV — Compute output

    Lines 578-581:
      O += S @ V      where V = K_nope (same data!)
      [64h, 64tok] x [64tok, 512d] -> [64h, 512d] in TMEM
      utcmma_ss: SMEM(S) x SMEM(V_nope) -> TMEM(O)

      V reuses the NoPE shared memory buffer (dequant[].nope)!
      zero_init = true only for first block (start fresh per batch)

    Line 581: Signal O updated: umma_arrive(bar_sv_done)
    Line 583: Advance ring buffer state

  Data flow through TMEM:

    TMEM cols:  0-------256------400---464
                |   O   |   Q   | P  |
                |       |       |    |
    UTCMMA #1:  |       |---Q-->|-P->|   Q_rope x K_rope -> P
    UTCMMA #2:  |       |---Q-->|-P->|   Q_nope x K_nope -> P (accum)
                |       |       |    |
    WG0 reads:  |       |       |<-P-|   tmem_ld P for softmax
    WG0 writes: |<--O---|       |    |   tmem_st O for rescaling
                |       |       |    |
    UTCMMA #3:  |<--O---|       |    |   S x V -> O (accum)
```

#### Warp 5: TMA Gather NoPE (Lines 586-615)

Gathers the NoPE (512d) portion of KV from scattered page locations.

```
  For each block of B_TOPK=64 tokens:

    Line 593:  Wait for TMA coordinates from Warp 7
    Line 594:  Wait for previous raw buffer to be consumed by WG2

    Lines 595-610: Issue TMA gather4 operations
      Each gather4 loads 4 tokens at a time from global memory
      Loop: 64 tokens / 4 per gather = 16 gather4 calls

      tma_gather4(
        tensor_map_kv_nope,     // TMA descriptor for NoPE cache layout
        bar_raw_ready[buf],     // Signal when data arrives
        raw_nope[buf] + offset, // Destination in shared memory
        col_idx=0,              // Start at column 0
        row_indices,            // 4 TMA coordinates from Warp 7
        EVICT_LAST              // L2 cache hint: keep in cache
      )

    Line 611:  Arrive with expected bytes: B_TOPK * D_NOPE * sizeof(e4m3)
    Line 612:  Release index buffer for Warp 7 to reuse

  Memory access pattern:
    Global Memory (HBM):
    +------+------+------+------+------+
    |Page 0|Page 1|Page 2| ...  |Pg 8461|
    +------+------+------+------+------+
       ^           ^    ^           ^
       |           |    |           |    4 tokens per TMA gather4
       +-----------+----+-----------+    (scattered across pages!)
       gather4 #0    gather4 #1

    Each page: [64 tokens, 512d] = 64 KB (NoPE only)
    16 gather4s x 4 tokens x 512B = 32 KB per block
```

#### Warp 6: TMA Gather RoPE (Lines 616-652)

Same pattern as Warp 5 but for the RoPE (64d) portion — 8x smaller.

```
  Key differences from Warp 5:
    - Uses tensor_map_kv_rope (64d layout)
    - Data goes directly to dequant[buf].rope (not raw buffer)
    - In V32 mode: waits for bar_qk_done (previous QK done)
      In MODEL1 mode: waits for bar_sv_done (previous SV done)
      This is because V32 processes RoPE separately (first), while
      MODEL1 processes NoPE+RoPE together.

    - V32: 1 gather4 loop (64d = 128B, fits in one TMA column)
    - MODEL1: 2 gather4 loops (64d split across two 64B columns)

  Expected bytes: B_TOPK * D_ROPE * sizeof(bf16)
  (RoPE is always BF16, even when NoPE is FP8)
```

#### Warp 7: Index Transform (Lines 653-743)

Converts sparse indices to TMA coordinates, extracts FP8 scales, generates validity masks.

```
  Line 657-658:  Compute TMA coordinate scaling factors
                 tma_coords_step_per_token = 656/TMA_K_STRIDE  (V32)
                 tma_coords_step_per_block = stride / TMA_K_STRIDE

  For each block of B_TOPK=64 tokens (lines 734-742):
    32 threads, each handles 2 tokens (lane_idx * 2):

    Line 686:    Load 2 indices from global memory (int2 coalesced load)
                 my_indices[0..1] = indices[block_idx*64 + lane_idx*2]

    Line 691:    Wait for index buffer to be free (bar_valid_coord_scale_free)

    For each of 2 tokens (lines 697-717):
      Line 699:  block_idx = index / page_block_size
      Line 700:  idx_in_block = index % page_block_size

      Line 701:  Validity check: is_valid = (index != -1) && (pos < topk_length)
      Line 703:  TMA coord = block_idx * steps_per_block + idx_in_block * steps_per_token
                 If invalid: TMA coord = -1 (TMA will return zeros)

      Lines 704-716: Extract FP8 scale factors from KV cache header
                 V32: 4 float32 scales -> converted to e8m0 format
                 MODEL1: 8 e8m0 scales loaded directly

    Lines 718-720: Shuffle validity bits across 4 lanes to build 8-bit mask
                   valid_mask = 8 bits packed (2 per lane, 4 lanes)

    Lines 722-728: Store TMA coords, scales, validity mask to shared memory
    Line 730:    Signal data ready: bar_valid_coord_scale_ready.arrive()
```

### WG2: Dequantization (Lines 747-843)

128 threads, 208 registers. **This entire warpgroup is unnecessary for the competition**
since the competition uses BF16 KV cache (no FP8).

```
  Purpose: Convert FP8 NoPE data to BF16 with per-tile scale factors

  Organization: 8 threads per token, 16 token groups
    GROUP_SIZE = 8, NUM_GROUPS = 16
    Each group handles B_TOPK/16 = 4 rows (tokens)

  For each block (lines 764-841):
    Wait for: indices ready, raw FP8 data ready, previous SV done

    For each token in group (lines 783-805 for V32):
      Load 4 e8m0 scale factors -> convert to bf16
      For each 8-byte FP8 chunk:
        Load 8 FP8 values (uint64_t)
        fp8x2_to_bf16x2_with_scale: convert pairs with scale
        st_128b: write 16 bytes of bf16 to shmem (128-bit store)

    Line 837:  Signal: bar_nope_ready.arrive() — dequant complete
    Line 838:  Signal: bar_raw_free.arrive() — raw buffer can be reused
```

## Phase 3: Combine Kernel

**File**: [combine.cu](csrc/smxx/decode/combine/combine.cu)

**Grid**: `(batch_size, s_q, h_q/8)`. Each warp handles one head.

```
  Line 35-37:  Load split range for this batch item
               [start_split, end_split) = num_splits[batch_idx..batch_idx+1]

  Line 38-39:  If only 1 split: no combining needed, return
               (Main kernel already wrote final output)

  Lines 70-78:  Each warp loads all LSE values for its head
                local_lse[i] = lse_accum[split_i, head]

  Lines 80-86:  Warp-wide reduction to find max_lse
                (shuffle-based parallel reduction)

  Lines 89-95:  Compute sum_exp = sum(exp2(lse_i - max_lse))
                (another warp-wide reduction)

  Line 97:      global_lse = log2(sum_exp) + max_lse

  Line 99:      Store final LSE (converted back to natural log)

  Lines 101-112: Incorporate attention sink (if present)
                 global_lse += log2(1 + exp2(attn_sink - global_lse))

  Lines 114-117: Compute per-split scale factors
                 scale[i] = exp2(lse_i - global_lse)
                 Store to shared memory for the accumulation phase

  Lines 129-144: Weighted accumulation of output vectors
                 For each split:
                   result += scale[split] * o_accum[split]
                 Each thread handles HEAD_DIM_V/(32*4) = 4 float4 chunks

  Lines 146-159: Convert fp32 result to bf16, store to final output
```

## Barrier Synchronization Map

The kernel uses transactional barriers for producer-consumer coordination.
Here's the complete flow:

```
  Warp 7               Warp 5              Warp 6              WG2                 Warp 4              WG0
  (Indices)            (NoPE gather)        (RoPE gather)       (Dequant)           (GEMM)              (Softmax)
    |                    |                    |                    |                    |                    |
    |  valid_coord_      |                    |                    |                    |                    |
    |  scale_ready ----->|                    |                    |                    |                    |
    |        |---------->|                    |                    |                    |                    |
    |        |           |                    |                    |                    |                    |
    |        |           |  raw_ready ------->|                    |                    |                    |
    |        |           |                    |                    |                    |                    |
    |        |           |                    |  nope_ready ------>|                    |                    |
    |        |           |                    |                    |                    |                    |
    |        |           |  rope_ready ------>|                    |                    |                    |
    |        |           |                    |------------------->|                    |                    |
    |        |           |                    |                    |                    |                    |
    |        |           |                    |                    |  qk_done --------->|                    |
    |        |           |                    |                    |                    |                    |
    |        |           |                    |                    |                    |  so_ready -------->|
    |        |           |                    |                    |                    |                    |
    |        |           |                    |                    |  sv_done --------->|                    |
    |        |           |                    |                    |                    |                    |
    |  valid_coord_      |                    |                    |                    |                    |
    |  scale_free <------|                    |                    |                    |                    |
    |        ^-----------|                    |                    |                    |                    |
    |        ^-----------|--------------------|--------------------|                    |                    |
```

Key barrier dependencies:
```
  Warp 7 produces --> Warp 5/6 consume: TMA coordinates, validity masks
  Warp 5 produces --> WG2 consumes:     Raw FP8 NoPE data
  Warp 6 produces --> Warp 4 consumes:  RoPE data (BF16, no dequant needed)
  WG2 produces    --> Warp 4 consumes:  Dequantized NoPE data (BF16)
  Warp 4 produces --> WG0 consumes:     P = QK^T scores (in TMEM)
  WG0 produces    --> Warp 4 consumes:  S = softmax(P) scores (in SMEM)
  Warp 4 produces --> WG0 consumes:     O += S@V (in TMEM)
```

## Pipeline Timeline (Simplified)

```
  Time --->

  Warp 7:   [idx 0][idx 1][idx 2][idx 3][idx 4][idx 5]...
             ^^^^   ^^^^   ^^^^   ^^^^
             4 index buffers (quad-buffered, runs ahead)

  Warp 5:        [NoPE gather 0][NoPE gather 1][NoPE gather 2]...
                  ^^^^^^^^^^^^^^
                  Waits for idx, then TMA gathers 64 tokens

  Warp 6:        [RoPE gather 0][RoPE gather 1][RoPE gather 2]...

  WG2:                [dequant 0  ][dequant 1  ][dequant 2  ]...
                       ^^^^^^^^^^^
                       FP8->BF16, waits for raw NoPE data

  Warp 4:                  [QK^T 0][SV 0][QK^T 1][SV 1][QK^T 2]...
                            ^^^^^^  ^^^^
                            Waits    Waits
                            nope+    softmax
                            rope     scores

  WG0:                          [softmax 0][softmax 1][softmax 2]...
                                 ^^^^^^^^^^
                                 Waits for QK^T,
                                 then rescales O,
                                 writes S for SV

  Double-buffering: while Warp 4 computes on buf[0], Warps 5/6 load buf[1]
```

## Competition Adaptations

For the competition (BF16 KV, separate caches), the key changes from FlashMLA:

```
  FlashMLA:                          Competition:
  +----------+                       +----------+
  | WG0: 128 |  Softmax              | WG0: 128 |  Softmax (same)
  +----------+                       +----------+
  | WG1: 128 |  GEMM + TMA           | WG1: 128 |  GEMM + TMA
  |  W4: MMA |                       |  W4: MMA |  (same)
  |  W5: NoPE| TMA gather FP8        |  W5: NoPE|  TMA gather BF16
  |  W6: RoPE| TMA gather BF16       |  W6: RoPE|  TMA gather BF16
  |  W7: Idx | + FP8 scale extract   |  W7: Idx |  NO scale extraction
  +----------+                       +----------+
  | WG2: 128 |  FP8->BF16 dequant    | (REMOVED)|  No dequant needed!
  +----------+                       +----------+

  Total: 384 threads                 Total: 256 threads
  Registers: split 3 ways            Registers: split 2 ways (more per WG!)
```

Removing WG2 means:
- 128 fewer threads competing for SM resources
- More registers available for WG0 and WG1
- Simpler barrier graph (no raw_ready/raw_free/nope_ready chain)
- Warp 5 writes directly to dequant[].nope buffer (BF16 from TMA)
- No FP8 scale extraction in Warp 7

The TMA gather now loads BF16 directly (2 bytes/element instead of 1),
so memory bandwidth requirements double for NoPE. But B200's 8 TB/s HBM
bandwidth should handle the 2.3 MB scattered read workload.
