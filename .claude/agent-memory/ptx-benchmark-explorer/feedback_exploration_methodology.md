---
name: PTX exploration must use both ISA docs and FlashMLA codebase
description: Always live-fetch PTX ISA documentation for the full modifier/argument space AND cross-reference FlashMLA for competition-relevant filtering — never rely on training data alone for ISA coverage
type: feedback
---

Always use BOTH sources when exploring PTX intrinsics and argument combinations — not just one.

**Why:** Prior sessions relied primarily on the FlashMLA codebase (csrc/kerutils/) plus PTX ISA knowledge from training data. This risks missing: modifier combinations added in CUDA 12.9+, full cross-products of L1+L2 cache hint enumerations, and sm100a-specific instruction families with incomplete training coverage. The user explicitly called this out and wants the approach corrected.

**How to apply:**
1. **Live-fetch the PTX ISA docs first** for the instruction family being explored. Use WebFetch on the relevant section of https://docs.nvidia.com/cuda/parallel-thread-execution/#instruction-set to get the authoritative, complete modifier space before proposing any experiments. Do not rely on training data for the ISA — fetch it.
2. **Then cross-reference FlashMLA** (csrc/kerutils/include/kerutils/device/sm100/, sm90/, csrc/sm100/) to identify which forms appear in competition-relevant code paths and filter suggestions accordingly.

The ISA docs give completeness (don't miss variants); the FlashMLA codebase gives relevance (don't propose benchmarks for dead code paths). Both are required — neither alone is sufficient.
