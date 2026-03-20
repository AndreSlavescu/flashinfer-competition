"""
SASS instruction set analysis for B200 (sm_100a) competition kernels.

Three modes of operation:
  1. Full ISA extraction — mine nvdisasm for complete sm_100a instruction docs + latencies
  2. Kernel disassembly — disassemble a .cubin and produce opcode frequency / pipeline analysis
  3. CuTeDSL analysis  — JIT-compile a CuTeDSL kernel, extract cubin, disassemble, and classify

ISA extraction uses memcpy interception on nvdisasm to capture instruction metadata strings.
Kernel analysis classifies opcodes by pipeline stage, extracts scheduling control words
(stall counts, yield bits, barrier deps), and identifies async pipeline patterns.

All runs on Modal B200.

Usage:
    modal run tools/sass/dump_sass_modal.py                              # full ISA docs
    modal run tools/sass/dump_sass_modal.py --cubin kernel.cubin         # disassemble cubin
    modal run tools/sass/dump_sass_modal.py --cutedsl kernel.py          # CuTeDSL → SASS
"""

import modal
import re
from collections import Counter

app = modal.App("sass-analyzer")

# ---------------------------------------------------------------------------
# sm100a SASS opcode classification
# ---------------------------------------------------------------------------
# Known Blackwell SASS opcodes grouped by pipeline stage. Used to map raw
# disassembly into a competition-kernel-relevant breakdown.

SM100A_OPCODE_CLASSES = {
    # Tensor core / TMEM
    "tensor_mma": [
        "UTCHMMA",       # tcgen05.mma — tensor core MMA via TMEM
        "HMMA",          # legacy wgmma path (should not appear on sm100a optimal code)
    ],
    "tmem": [
        "UTMALDG",       # TMA async global load (4D gather)
        "UTMASTG",       # TMA async global store
        "UTMACCTL",      # TMA cache control / prefetch
        "UTMACMDFLUSH",  # TMA command flush
        "UTMALDL",       # TMA async local load (if any)
        "TCGEN05",       # tcgen05.ld / tcgen05.st / tcgen05.alloc / tcgen05.dealloc
    ],
    # Barriers & synchronization
    "barrier": [
        "UTCBAR",        # tensor core barrier (multicast)
        "UTCATOMSWS",    # tensor core atomic shared-memory ops
        "BAR",           # standard barrier
        "MBAR",          # mbarrier
        "DEPBAR",        # dependency barrier
        "WARPSYNC",      # warp-level sync
        "FENCE",         # memory fence
        "MEMBAR",        # memory barrier
    ],
    # Global / shared memory
    "memory": [
        "LDG",           # global load
        "STG",           # global store
        "LDS",           # shared load
        "STS",           # shared store
        "ATOMS",         # shared atomic
        "ATOMG",         # global atomic
        "RED",           # reduction
        "LDSM",          # shared → register matrix load
        "STSM",          # register → shared matrix store
    ],
    # Compute (FP/INT)
    "compute_fp": [
        "FADD", "FMUL", "FFMA", "FMNMX", "FSET", "FSETP",
        "HADD2", "HMUL2", "HFMA2", "HSET2", "HSETP2",
        "MUFU",          # multi-function unit (exp, log, rcp, rsq, sin, cos)
        "DFMA", "DADD", "DMUL",
    ],
    "compute_int": [
        "IADD3", "IMAD", "ISETP", "IMNMX",
        "LEA", "LOP3", "SHF", "SHL", "SHR",
        "POPC", "FLO", "BREV",
    ],
    # Control flow
    "control": [
        "BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT",
        "BREAK", "CONT", "BSSY", "BSYNC", "YIELD",
        "SETMAXNREG",   # setmaxregister (warp specialization register management)
    ],
    # Data movement
    "move": [
        "MOV", "PRMT", "SEL", "SHFL",
        "S2R",           # special register read (threadIdx, blockIdx, clock, etc.)
        "CS2R",          # constant special register
        "R2P", "P2R",    # predicate ↔ register
        "I2F", "I2I", "F2I", "F2F",  # type conversion
    ],
}

# Opcodes critical for the competition kernel — flag if missing or low count
COMPETITION_CRITICAL_OPCODES = {
    "UTCHMMA":    "tcgen05.mma (tensor core MMA via TMEM) — the core compute instruction",
    "UTMALDG":    "TMA async global load — used for gather4 of ckv/kpe from HBM",
    "UTMASTG":    "TMA async global store — output write",
    "UTMACCTL":   "TMA cache control / prefetch — L2 prefetch for gather locality",
    "UTCBAR":     "tensor core barrier — warp specialization synchronization",
    "TCGEN05":    "tcgen05.ld/st/alloc/dealloc — TMEM data movement",
    "SETMAXNREG": "setmaxregister — per-warp register budget (warp specialization)",
    "MUFU":       "multi-function unit — exp2 for softmax",
}

# Synchronous instruction issue latencies (cycles) for sm100a.
#
# ONLY synchronous, deterministic instructions are listed here. Async
# instructions (TMA, MMA commits, mbarrier waits) do NOT have a fixed
# latency — their completion time depends on pipeline depth, memory
# contention, cache state, and overlap with other work. Treating async
# ops as having a "cycle count" is fundamentally misleading.
#
# For async ops, we track ISSUE COST (cycles to dispatch the instruction)
# separately from COMPLETION (which is non-deterministic and measured
# empirically via our benchmarks, not assigned a fixed number).
#
# Sources marked per entry. [est] = carried from Hopper, not confirmed on B200.
SM100A_SYNC_LATENCIES = {
    # Compute — arxiv:2507.10789 confirms standard 4-cycle FP/INT pipeline on Blackwell
    "FFMA":       (4,  "FP32 fused multiply-add (arxiv:2507.10789)"),
    "FADD":       (4,  "FP32 add (arxiv:2507.10789)"),
    "FMUL":       (4,  "FP32 multiply (arxiv:2507.10789)"),
    "HFMA2":      (4,  "FP16x2 fused multiply-add (arxiv:2507.10789)"),
    "MUFU":       (8,  "multi-function unit — exp, rcp, rsq; varies by fn [est]"),
    "IMAD":       (4,  "integer multiply-add (arxiv:2507.10789)"),
    "IADD3":      (4,  "3-input integer add (arxiv:2507.10789)"),
    "LOP3":       (4,  "3-input logic op [est]"),
    # Shared memory — synchronous from the warp's perspective
    # arxiv:2507.10789: smem latency ~20-30 cycles
    "LDS":        (25,  "shared memory load — 20-30 cyc (arxiv:2507.10789)"),
    "STS":        (25,  "shared memory store (arxiv:2507.10789)"),
    # Data movement — synchronous register ops
    "MOV":        (4,   "register move [est]"),
    "SHFL":       (4,   "warp shuffle [est]"),
    "S2R":        (20,  "special register read [est]"),
}

# Async instruction ISSUE costs (cycles to dispatch, NOT completion time).
# Completion is non-deterministic — depends on memory hierarchy, contention,
# pipeline depth, and overlap. Use our benchmark data for completion estimates.
SM100A_ASYNC_ISSUE_COSTS = {
    "UTCHMMA":    (1,  "tcgen05.mma — single-thread issue, async completion via TMEM"),
    "UTMALDG":    (1,  "TMA gather — async issue, completion varies (our benchmarks: 197-546ns)"),
    "UTMASTG":    (1,  "TMA store — async issue"),
    "UTMACCTL":   (1,  "TMA prefetch — async issue, fire-and-forget"),
    "LDG":        (1,  "global load — async issue, data arrives later via scoreboard"),
    "STG":        (1,  "global store — async issue"),
    "MBAR":       (1,  "mbarrier arrive — async issue"),
    "BAR":        (20, "barrier — sync, blocks until all threads arrive [est]"),
}


def classify_opcodes(opcode_counts: dict) -> dict:
    """Classify raw opcode counts into pipeline-stage groups."""
    classified = {}
    unclassified = {}

    for opcode, count in opcode_counts.items():
        found = False
        for cls_name, prefixes in SM100A_OPCODE_CLASSES.items():
            for prefix in prefixes:
                if opcode.startswith(prefix):
                    classified.setdefault(cls_name, {})[opcode] = count
                    found = True
                    break
            if found:
                break
        if not found:
            unclassified[opcode] = count

    return {"classified": classified, "unclassified": unclassified}


def extract_scheduling_info(disassembly: str) -> dict:
    """
    Extract scheduling control words from SASS disassembly.

    nvdisasm --print-instruction-encoding shows control words that encode:
    - Stall count (cycles to wait before next instruction can issue)
    - Yield flag (hints scheduler to switch to another warp)
    - Write/read barrier dependencies
    - Reuse flags (register cache hints)

    This gives us a picture of async pipeline pressure and scheduling slack.
    """
    # Pattern: instructions with encoding show hex control words
    # Format varies, but stall counts appear in the control word
    stall_counts = []

    # Look for control word patterns in encoded disassembly
    # The control word is typically the first 8 bytes, encoding stalls + barriers
    # We also look for scheduling info comments that nvdisasm can produce
    sched_pattern = re.compile(
        r"^\s*/\*[0-9a-f]+\*/\s+(\S+).*?;\s*/\*\s*([^*]+)\*/\s*$",
        re.MULTILINE,
    )

    for m in sched_pattern.finditer(disassembly):
        opcode = m.group(1)
        comment = m.group(2).strip()
        if "stall" in comment.lower() or "yield" in comment.lower():
            stall_counts.append((opcode, comment))

    return {
        "stall_annotations": stall_counts,
        "total_annotated": len(stall_counts),
    }


def estimate_pipeline_cost(opcode_counts: dict) -> dict:
    """
    Estimate synchronous cycle cost by pipeline stage.

    Only synchronous instructions get cycle estimates. Async instructions
    (TMA, MMA, global loads) are counted by issue frequency only — their
    actual cost depends on pipeline overlap and is not a fixed number.
    """
    stage_costs = {}

    for opcode, count in opcode_counts.items():
        base_op = opcode.split(".")[0]

        # Check sync latency first, then async issue cost
        lat = None
        is_async = False
        for prefix, (l, _desc) in SM100A_SYNC_LATENCIES.items():
            if base_op.startswith(prefix):
                lat = l
                break
        if lat is None:
            for prefix, (l, _desc) in SM100A_ASYNC_ISSUE_COSTS.items():
                if base_op.startswith(prefix):
                    lat = l
                    is_async = True
                    break
        if lat is None:
            continue

        raw_cycles = lat * count

        # Determine pipeline stage
        stage = "unknown"
        for cls_name, prefixes in SM100A_OPCODE_CLASSES.items():
            for prefix in prefixes:
                if base_op.startswith(prefix):
                    stage = cls_name
                    break

        stage_costs.setdefault(stage, {
            "sync_cycles": 0, "async_issue_count": 0, "ops": [],
        })
        if is_async:
            stage_costs[stage]["async_issue_count"] += count
        else:
            stage_costs[stage]["sync_cycles"] += raw_cycles
        stage_costs[stage]["ops"].append((opcode, count, lat, raw_cycles, is_async))

    return stage_costs


def format_analysis(opcode_counts: dict, total: int, disassembly: str = "") -> str:
    """Format opcode analysis into a readable report."""
    result = classify_opcodes(opcode_counts)
    lines = []

    lines.append(f"Total instructions: {total}")
    lines.append("")

    # Per-class summary
    lines.append("Pipeline stage breakdown:")
    for cls_name in SM100A_OPCODE_CLASSES:
        cls_ops = result["classified"].get(cls_name, {})
        cls_total = sum(cls_ops.values())
        pct = (cls_total / total * 100) if total > 0 else 0
        lines.append(f"  {cls_name:15s}  {cls_total:6d}  ({pct:5.1f}%)")
        for op, count in sorted(cls_ops.items(), key=lambda x: -x[1]):
            lines.append(f"    {op:25s}  {count:6d}")

    if result["unclassified"]:
        unc_total = sum(result["unclassified"].values())
        pct = (unc_total / total * 100) if total > 0 else 0
        lines.append(f"  {'unclassified':15s}  {unc_total:6d}  ({pct:5.1f}%)")
        for op, count in sorted(result["unclassified"].items(), key=lambda x: -x[1])[:20]:
            lines.append(f"    {op:25s}  {count:6d}")

    # Competition-critical check
    lines.append("")
    lines.append("Competition-critical opcodes:")
    for prefix, desc in COMPETITION_CRITICAL_OPCODES.items():
        matching = {k: v for k, v in opcode_counts.items() if k.startswith(prefix)}
        if matching:
            count = sum(matching.values())
            lines.append(f"  OK   {prefix:15s}  {count:6d}  -- {desc}")
        else:
            lines.append(f"  MISS {prefix:15s}       0  -- {desc}")

    # Pipeline cost estimation (sync only — async has no fixed cost)
    lines.append("")
    lines.append("Synchronous cycle cost by pipeline stage:")
    lines.append("  (async ops listed by issue count only — completion is non-deterministic)")
    costs = estimate_pipeline_cost(opcode_counts)
    total_sync = sum(s["sync_cycles"] for s in costs.values())
    total_async_issues = sum(s["async_issue_count"] for s in costs.values())

    for stage, info in sorted(costs.items(), key=lambda x: -x[1]["sync_cycles"]):
        sync_c = info["sync_cycles"]
        async_n = info["async_issue_count"]
        pct = (sync_c / total_sync * 100) if total_sync > 0 else 0
        parts = []
        if sync_c:
            parts.append(f"sync={sync_c} cyc ({pct:.1f}%)")
        if async_n:
            parts.append(f"async_issues={async_n}")
        lines.append(f"  {stage:15s}  {', '.join(parts)}")
        for op, count, lat, cyc, is_async in sorted(info["ops"], key=lambda x: -x[3])[:5]:
            if is_async:
                lines.append(f"    {op:25s}  {count:5d} issues  (completion depends on pipeline)")
            else:
                lines.append(f"    {op:25s}  {count:5d} x {lat:4d}cyc = {cyc:8d}")

    if total_async_issues > 0:
        lines.append("")
        lines.append(f"  Total sync cost: {total_sync} cycles (the schedulable work)")
        lines.append(f"  Total async issues: {total_async_issues} (overlap determines actual wall time)")
        lines.append(f"  Kernel is {'async-dominated' if total_async_issues > total_sync / 4 else 'sync-dominated'}"
                     f" — profile with NCU for actual bottleneck")

    # Async pipeline pattern detection
    if disassembly:
        lines.append("")
        lines.append("Async pipeline patterns detected:")
        # Look for TMA → barrier → MMA sequences
        tma_count = sum(v for k, v in opcode_counts.items() if k.startswith("UTMALDG"))
        mma_count = sum(v for k, v in opcode_counts.items() if k.startswith("UTCHMMA"))
        bar_count = sum(v for k, v in opcode_counts.items()
                       if k.startswith("MBAR") or k.startswith("BAR") or k.startswith("UTCBAR"))
        setmax_count = sum(v for k, v in opcode_counts.items() if k.startswith("SETMAXNREG"))

        if tma_count and mma_count:
            ratio = tma_count / mma_count
            lines.append(f"  TMA:MMA ratio = {tma_count}:{mma_count} = {ratio:.2f}")
            if ratio > 2:
                lines.append(f"    -> memory-heavy: {ratio:.1f}x more TMA than MMA (likely memory-bound)")
            elif ratio < 0.5:
                lines.append(f"    -> compute-heavy: {1/ratio:.1f}x more MMA than TMA (likely compute-bound)")
            else:
                lines.append(f"    -> balanced TMA/MMA pipeline")

        if bar_count:
            lines.append(f"  Barrier instructions: {bar_count} (sync overhead in pipeline)")
            if tma_count and bar_count > tma_count * 2:
                lines.append(f"    -> high barrier:TMA ratio ({bar_count/tma_count:.1f}x) — possible over-synchronization")

        if setmax_count:
            lines.append(f"  SETMAXNREG count: {setmax_count} (warp specialization register transitions)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
sass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-13-0 || apt-get install -y cuda-toolkit-12-8",
    )
    .env({
        "PATH": "/usr/local/cuda/bin:${PATH}",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
    })
    # Local ISA extraction tools (intercept.c adapted from
    # https://github.com/0xD0GF00D/DocumentSASS — memcpy interception technique)
    .add_local_file("tools/sass/intercept.c", "/root/sass_tools/intercept.c", copy=True)
    .add_local_file("tools/sass/extract_isa.py", "/root/sass_tools/extract_isa.py", copy=True)
    .run_commands("gcc -shared -fPIC -o /root/sass_tools/intercept.so /root/sass_tools/intercept.c -ldl")
    .pip_install("nvidia-cutlass-dsl", "torch")
)


# ---------------------------------------------------------------------------
# 1. Full ISA extraction
# ---------------------------------------------------------------------------
@app.function(image=sass_image, gpu="B200", timeout=1800)
def extract_isa(target_arch: str = "sm_100a") -> dict:
    """
    Extract complete SASS instruction set documentation for sm_100a.

    Pipeline:
      1. Compile a minimal stub kernel to .cubin for sm_100a
      2. Run nvdisasm under LD_PRELOAD=intercept.so to capture all internal
         memcpy traffic (instruction description + latency strings)
      3. Parse the raw output to extract 'ARCHITECTURE' (instruction docs)
         and 'OPERATION SETS' (scheduling latencies)

    ISA extraction technique adapted from:
      https://github.com/0xD0GF00D/DocumentSASS
    """
    import subprocess
    import os
    import sys

    tools_dir = "/root/sass_tools"
    results = {
        "success": False,
        "arch": target_arch,
        "instructions": "",
        "latencies": "",
        "nvdisasm_version": "",
        "error": "",
    }

    # Check tools
    for tool in ["nvcc", "nvdisasm"]:
        check = subprocess.run([tool, "--version"], capture_output=True, text=True)
        if check.returncode != 0:
            results["error"] = f"{tool} not found"
            return results
        ver = check.stdout.strip().splitlines()[-1]
        print(f"{tool}: {ver}")
        if tool == "nvdisasm":
            results["nvdisasm_version"] = ver

    interceptor = os.path.join(tools_dir, "intercept.so")
    if not os.path.exists(interceptor):
        results["error"] = "intercept.so not found — image build may have failed"
        return results

    # 1. Compile stub kernel — deliberately minimal so nvdisasm loads the
    #    full ISA metadata tables for the target arch without code clutter
    stub_cu = "/root/stub.cu"
    stub_cubin = f"/root/stub_{target_arch}.cubin"
    with open(stub_cu, "w") as f:
        f.write("__global__ void _stub(float *o) { o[0] = 0; }\n")

    comp = subprocess.run(
        ["nvcc", "-cubin", f"-arch={target_arch}", "-o", stub_cubin, stub_cu],
        capture_output=True, text=True,
    )
    if comp.returncode != 0:
        results["error"] = f"Stub compilation failed: {comp.stderr}"
        return results
    print(f"Compiled stub for {target_arch}")

    # 2. Run nvdisasm under LD_PRELOAD to intercept memcpy
    #    (adapted from https://github.com/0xD0GF00D/DocumentSASS)
    print("Running nvdisasm with memcpy interception...")
    intercept_run = subprocess.run(
        ["nvdisasm", stub_cubin],
        capture_output=True, text=True,
        env={**os.environ, "LD_PRELOAD": interceptor},
        timeout=120,
    )

    raw_output = intercept_run.stdout
    print(f"Captured {len(raw_output)} bytes of intercepted output")

    # 3. Parse with our extraction module
    sys.path.insert(0, tools_dir)
    from extract_isa import extract_from_raw

    parsed = extract_from_raw(raw_output)
    results["instructions"] = parsed["instructions"]
    results["latencies"] = parsed["latencies"]

    if results["instructions"]:
        print(f"Instructions: {len(results['instructions'])} bytes")
    if results["latencies"]:
        print(f"Latencies: {len(results['latencies'])} bytes")

    results["success"] = bool(results["instructions"] or results["latencies"])
    if not results["success"]:
        results["error"] = (
            "ISA extraction produced no output. "
            "The nvdisasm internal string format may have changed in this CUDA version. "
            "Kernel disassembly (--cubin/--cutedsl) still works independently."
        )
    return results


# ---------------------------------------------------------------------------
# 2. Kernel disassembly
# ---------------------------------------------------------------------------
@app.function(image=sass_image, gpu="B200", timeout=900)
def disassemble_cubin(cubin_bytes: bytes, filename: str = "kernel.cubin") -> dict:
    """
    Disassemble a cubin and return classified SASS analysis.

    Returns opcode counts classified by pipeline stage, competition-critical
    opcode check, estimated pipeline cost breakdown, async pattern detection,
    and raw disassembly text.
    """
    import subprocess

    cubin_path = f"/root/{filename}"
    with open(cubin_path, "wb") as f:
        f.write(cubin_bytes)

    # Basic disassembly
    disasm = subprocess.run(
        ["nvdisasm", "-g", cubin_path],
        capture_output=True, text=True,
    )
    if disasm.returncode != 0:
        return {"success": False, "error": f"nvdisasm failed: {disasm.stderr}"}

    # With instruction encoding (shows control words for scheduling analysis)
    enc = subprocess.run(
        ["nvdisasm", "--print-instruction-encoding", "-g", cubin_path],
        capture_output=True, text=True,
    )

    # With control flow graph
    cfg = subprocess.run(
        ["nvdisasm", "-cfg", cubin_path],
        capture_output=True, text=True,
    )

    # Extract per-kernel sections
    kernel_pattern = re.compile(r"\.text\.(\S+)")
    kernels_found = kernel_pattern.findall(disasm.stdout)

    # Extract all opcodes
    opcode_pattern = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(\S+)", re.MULTILINE)
    all_opcodes = opcode_pattern.findall(disasm.stdout)
    opcode_counts = dict(Counter(all_opcodes))
    total = len(all_opcodes)

    # Classify, estimate costs, and format
    analysis = format_analysis(opcode_counts, total, disasm.stdout)
    classification = classify_opcodes(opcode_counts)
    pipeline_costs = estimate_pipeline_cost(opcode_counts)
    scheduling = extract_scheduling_info(enc.stdout if enc.returncode == 0 else "")

    return {
        "success": True,
        "kernels": kernels_found,
        "disassembly": disasm.stdout,
        "disassembly_with_encoding": enc.stdout if enc.returncode == 0 else "",
        "control_flow_graph": cfg.stdout if cfg.returncode == 0 else "",
        "opcode_counts": opcode_counts,
        "classification": classification,
        "pipeline_costs": {k: {kk: vv for kk, vv in v.items() if kk != "ops"} for k, v in pipeline_costs.items()},  # strip verbose op lists
        "scheduling": scheduling,
        "analysis_report": analysis,
        "total_instructions": total,
        "unique_opcodes": sorted(set(all_opcodes)),
    }


# ---------------------------------------------------------------------------
# 3. CuTeDSL kernel → SASS analysis
# ---------------------------------------------------------------------------
@app.function(image=sass_image, gpu="B200", timeout=1200)
def analyze_cutedsl_kernel(kernel_source: str, kernel_filename: str = "kernel.py") -> dict:
    """
    JIT-compile a CuTeDSL kernel, extract the cubin, and disassemble.

    The CuTeDSL JIT pipeline: Python → IR → MLIR → PTX → cubin.
    We capture the cubin after compilation and feed it to nvdisasm for
    full SASS analysis including pipeline cost estimation.
    """
    import subprocess
    import os
    import glob

    results = {
        "success": False,
        "ptx": "",
        "cubin_path": "",
        "error": "",
    }

    # Write kernel source
    kernel_path = f"/root/{kernel_filename}"
    with open(kernel_path, "w") as f:
        f.write(kernel_source)

    # Set env to capture compilation artifacts
    cache_dir = "/root/cutlass_cache"
    os.makedirs(cache_dir, exist_ok=True)
    env = os.environ.copy()
    env["CUTLASS_DSL_CACHE_DIR"] = cache_dir
    env["CUTLASS_DSL_DUMP_PTX"] = "1"

    print(f"Compiling CuTeDSL kernel: {kernel_filename}")
    run = subprocess.run(
        ["python3", kernel_path],
        capture_output=True, text=True,
        timeout=300, env=env,
    )

    if run.returncode != 0:
        results["error"] = f"CuTeDSL compilation failed:\nstdout: {run.stdout[:1000]}\nstderr: {run.stderr[:1000]}"
        return results

    print(f"Compilation output:\n{run.stdout[:500]}")

    # Find generated cubin files
    cubins = glob.glob(f"{cache_dir}/**/*.cubin", recursive=True)
    cubins += glob.glob("/tmp/**/*.cubin", recursive=True)
    cubins += glob.glob("/root/**/*.cubin", recursive=True)

    if not cubins:
        # Try to find .ptx files instead and compile them
        ptx_files = glob.glob(f"{cache_dir}/**/*.ptx", recursive=True)
        ptx_files += glob.glob("/tmp/**/*.ptx", recursive=True)

        if ptx_files:
            ptx_path = ptx_files[0]
            with open(ptx_path, "r") as f:
                results["ptx"] = f.read()
            print(f"Found PTX: {ptx_path} ({len(results['ptx'])} bytes)")

            # Compile PTX to cubin
            cubin_path = "/root/compiled_kernel.cubin"
            ptxas = subprocess.run(
                ["ptxas", "-arch=sm_100a", "-o", cubin_path, ptx_path],
                capture_output=True, text=True,
            )
            if ptxas.returncode == 0:
                cubins = [cubin_path]
            else:
                results["error"] = f"ptxas failed: {ptxas.stderr[:500]}"
                results["success"] = True  # PTX still available
                return results
        else:
            results["error"] = (
                "No .cubin or .ptx found after CuTeDSL compilation. "
                "The kernel may not have triggered JIT, or artifacts are "
                f"in an unexpected location. Cache dir: {cache_dir}"
            )
            results["compilation_stdout"] = run.stdout
            results["compilation_stderr"] = run.stderr
            return results

    cubin_path = cubins[0]
    results["cubin_path"] = cubin_path
    print(f"Found cubin: {cubin_path}")

    # Also grab PTX if we haven't already
    if not results["ptx"]:
        ptx_files = glob.glob(f"{cache_dir}/**/*.ptx", recursive=True)
        if ptx_files:
            with open(ptx_files[0], "r") as f:
                results["ptx"] = f.read()

    # Disassemble
    disasm = subprocess.run(
        ["nvdisasm", "-g", cubin_path],
        capture_output=True, text=True,
    )
    if disasm.returncode != 0:
        results["error"] = f"nvdisasm failed: {disasm.stderr}"
        results["success"] = True  # PTX still available
        return results

    enc = subprocess.run(
        ["nvdisasm", "--print-instruction-encoding", "-g", cubin_path],
        capture_output=True, text=True,
    )

    # Extract opcodes
    opcode_pattern = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(\S+)", re.MULTILINE)
    all_opcodes = opcode_pattern.findall(disasm.stdout)
    opcode_counts = dict(Counter(all_opcodes))
    total = len(all_opcodes)

    analysis = format_analysis(opcode_counts, total, disasm.stdout)
    classification = classify_opcodes(opcode_counts)
    pipeline_costs = estimate_pipeline_cost(opcode_counts)
    scheduling = extract_scheduling_info(enc.stdout if enc.returncode == 0 else "")

    kernel_names = re.compile(r"\.text\.(\S+)").findall(disasm.stdout)

    results.update({
        "success": True,
        "kernels": kernel_names,
        "disassembly": disasm.stdout,
        "disassembly_with_encoding": enc.stdout if enc.returncode == 0 else "",
        "opcode_counts": opcode_counts,
        "classification": classification,
        "pipeline_costs": {k: {kk: vv for kk, vv in v.items() if kk != "ops"} for k, v in pipeline_costs.items()},  # strip verbose op lists
        "scheduling": scheduling,
        "analysis_report": analysis,
        "total_instructions": total,
        "unique_opcodes": sorted(set(all_opcodes)),
    })
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(cubin: str = "", cutedsl: str = ""):
    """
    SASS analysis for sm_100a kernels.

    Usage:
        modal run tools/sass/dump_sass_modal.py                          # full ISA extraction
        modal run tools/sass/dump_sass_modal.py --cubin kernel.cubin     # disassemble cubin
        modal run tools/sass/dump_sass_modal.py --cutedsl kernel.py      # CuTeDSL → SASS
    """
    if cutedsl:
        print(f"Analyzing CuTeDSL kernel: {cutedsl}")
        with open(cutedsl, "r") as f:
            source = f.read()
        result = analyze_cutedsl_kernel.remote(source, cutedsl.split("/")[-1])

        if result["success"]:
            if result.get("ptx"):
                print(f"\n{'='*60}")
                print(f"PTX ({len(result['ptx'])} bytes, first 2000 chars)")
                print(f"{'='*60}")
                print(result["ptx"][:2000])

            if result.get("analysis_report"):
                print(f"\n{'='*60}")
                print("SASS Pipeline Analysis")
                print(f"{'='*60}")
                print(result["analysis_report"])

            if result.get("kernels"):
                print(f"\nKernels found: {', '.join(result['kernels'])}")

            if result.get("disassembly"):
                print(f"\n{'='*60}")
                print(f"Disassembly (first 5000 chars of {len(result['disassembly'])} total)")
                print(f"{'='*60}")
                print(result["disassembly"][:5000])
        else:
            print(f"Failed: {result.get('error', 'unknown')}")

    elif cubin:
        print(f"Disassembling: {cubin}")
        with open(cubin, "rb") as f:
            cubin_bytes = f.read()
        result = disassemble_cubin.remote(cubin_bytes, cubin.split("/")[-1])

        if result["success"]:
            print(f"\n{'='*60}")
            print("SASS Pipeline Analysis")
            print(f"{'='*60}")
            print(result["analysis_report"])

            if result.get("kernels"):
                print(f"\nKernels: {', '.join(result['kernels'])}")

            print(f"\n{'='*60}")
            print(f"Disassembly (first 5000 chars of {len(result['disassembly'])} total)")
            print(f"{'='*60}")
            print(result["disassembly"][:5000])
        else:
            print(f"Failed: {result.get('error', 'unknown')}")

    else:
        print("Extracting full sm_100a ISA documentation...")
        result = extract_isa.remote("sm_100a")

        if result["success"]:
            print(f"\nnvdisasm: {result['nvdisasm_version']}")

            if result["instructions"]:
                print(f"\n{'='*60}")
                print(f"Instructions ({len(result['instructions'])} bytes)")
                print(f"{'='*60}")
                print(result["instructions"])

            if result["latencies"]:
                print(f"\n{'='*60}")
                print(f"Latencies ({len(result['latencies'])} bytes)")
                print(f"{'='*60}")
                print(result["latencies"])
        else:
            print(f"Failed: {result.get('error', 'unknown')}")
