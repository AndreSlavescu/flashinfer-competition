---
name: B200 TMEM architecture and constraints
description: Tensor Memory architecture on Blackwell — allocation, access patterns, UMMA interaction, key constraints for kernel design
type: reference
---

## Source
Colfax Research: "CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA Blackwell GPUs"

## TMEM Architecture

- **Size**: 256 KB per SM
- **Organization**: 512 columns × 128 lanes × 32-bit cells
- **Addressing**: 32-bit — bits 31-16 = lane ID, bits 15-0 = column
- **Allocation granularity**: powers of 2 columns, minimum 32
- **Warp access**: each warp can only access 32 of 128 lanes → need full warpgroup (4 warps) for epilogue

## TMEM Allocation/Deallocation

- `tcgen05.alloc` allocates columns, stores base address to shared memory
- `tcgen05.dealloc` frees columns
- **CRITICAL**: both alloc and dealloc must be called from the SAME warp
- In CuTeDSL: `cute.arch.alloc_tmem(cols, holding_buf)` + `cute.arch.retrieve_tmem_ptr()`

## UMMA (tcgen05.mma) Interaction

- Operand A: TMEM or SMEM
- Operand B: SMEM only
- Accumulator D: TMEM only (transparent row-major format)
- Only one thread launches UMMA per CTA (single-thread issue)
- Uses shared memory descriptors (like WGMMA)
- Expects K-major tiles with predefined swizzling

## Supported UMMA Shapes

- 64×N×16 (N multiple of 8, N ≤ 256)
- 128×N×16 (N multiple of 16, N ≤ 256)
- Largest atom: 128×256×16 — consumes exactly half of TMEM

## Data Movement: tcgen05.ld

- `tcgen05.ld.sync.aligned.shape.num.b32 r, [taddr]`
- Shapes: 16×64b, 16×128b, 16×256b, 32×32b ({lanes}×{bits})
- Num: x1, x2, x4... x128 (repeat pattern)
- Each warp load ≤ 16KB
- Warp-wide instruction — all threads must execute same instruction

## Key Constraint (CRITICAL for kernel design)

**No operations access TMEM except UMMA and tcgen05.ld/st.**
All pre-processing must happen BEFORE data enters TMEM.
All post-processing must happen AFTER data is extracted via tcgen05.ld.

This creates hard stage boundaries in the kernel pipeline.

## Register Pressure Advantage

UMMA requires NO registers for input data — data stays in TMEM. Combined with single-thread launch, this decouples MMA from the CTA's main execution, enabling deep pipelining without register contention.
