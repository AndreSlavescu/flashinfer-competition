"""
NCU (Nsight Compute) profiling integration for Modal B200.

Runs ncu CLI on compiled CUDA binaries, parses results into structured
metrics, and provides convenience accessors for memory-bound kernel analysis.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap_runtime import require_dependencies

require_dependencies(
    {"modal": "modal"},
    entrypoint="tools/ncu/ncu_modal.py",
)

import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import modal

app = modal.App("ncu-profiler")

# ---------------------------------------------------------------------------
# Metrics of interest for memory-bound kernel optimization
# ---------------------------------------------------------------------------
KEY_METRICS = [
    # Throughput
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # Cache hit rates
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Issue utilization
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    # Warp stall reasons
    "smsp__average_warps_issue_stalled_barrier_per_issue_active",
    "smsp__average_warps_issue_stalled_membar_per_issue_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active",
    "smsp__average_warps_issue_stalled_wait_per_issue_active",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active",
    # Pipe utilization
    "smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    # Memory
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes_lookup_hit.sum",
    "lts__t_bytes_lookup_miss.sum",
    # Instructions
    "smsp__inst_executed.sum",
    "sm__cycles_elapsed.avg",
]


@dataclass
class NCUMetric:
    """A single metric from NCU output."""
    name: str
    value: float
    unit: str = ""


@dataclass
class NCUKernelResult:
    """Profiling result for a single kernel invocation."""
    kernel_name: str
    metrics: Dict[str, NCUMetric] = field(default_factory=dict)

    def get(self, metric_name: str, default: float = 0.0) -> float:
        """Get a metric value by name."""
        m = self.metrics.get(metric_name)
        return m.value if m else default

    @property
    def dram_throughput_pct(self) -> float:
        return self.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")

    @property
    def sm_throughput_pct(self) -> float:
        return self.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")

    @property
    def l1_hit_rate(self) -> float:
        return self.get("l1tex__t_sector_hit_rate.pct")

    @property
    def l2_hit_rate(self) -> float:
        return self.get("lts__t_sector_hit_rate.pct")

    @property
    def occupancy_pct(self) -> float:
        return self.get("sm__warps_active.avg.pct_of_peak_sustained_active")

    @property
    def is_memory_bound(self) -> bool:
        """Heuristic: kernel is memory-bound if DRAM% >> SM%."""
        return self.dram_throughput_pct > self.sm_throughput_pct * 1.5

    def top_stall_reasons(self, n: int = 3) -> List[Tuple[str, float]]:
        """Return top N warp stall reasons by magnitude."""
        stall_metrics = [
            (k, m.value) for k, m in self.metrics.items()
            if "stalled" in k and m.value > 0
        ]
        stall_metrics.sort(key=lambda x: x[1], reverse=True)
        return stall_metrics[:n]

    def summary(self) -> str:
        lines = [f"Kernel: {self.kernel_name}"]
        lines.append(f"  DRAM throughput: {self.dram_throughput_pct:.1f}%")
        lines.append(f"  SM throughput:   {self.sm_throughput_pct:.1f}%")
        lines.append(f"  L1 hit rate:     {self.l1_hit_rate:.1f}%")
        lines.append(f"  L2 hit rate:     {self.l2_hit_rate:.1f}%")
        lines.append(f"  Occupancy:       {self.occupancy_pct:.1f}%")
        lines.append(f"  Memory-bound:    {self.is_memory_bound}")
        stalls = self.top_stall_reasons()
        if stalls:
            lines.append("  Top stall reasons:")
            for name, val in stalls:
                short = name.split("stalled_")[-1].split("_per_")[0]
                lines.append(f"    {short}: {val:.2f}")
        return "\n".join(lines)


@dataclass
class NCUResult:
    """Complete NCU profiling result."""
    success: bool
    kernels: List[NCUKernelResult] = field(default_factory=list)
    raw_csv: str = ""
    ncu_version: str = ""
    error_message: Optional[str] = None
    fallback: bool = False

    def summary(self) -> str:
        if not self.success:
            return f"NCU profiling failed: {self.error_message}"
        lines = [f"NCU version: {self.ncu_version}", f"Kernels profiled: {len(self.kernels)}"]
        for k in self.kernels:
            lines.append(k.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------
def parse_ncu_csv(csv_text: str) -> List[NCUKernelResult]:
    """
    Parse NCU --csv output into structured kernel results.

    NCU CSV format has header rows starting with == that we skip,
    then a proper CSV with columns like:
    "ID","Process ID","Process Name","Host Name","Kernel Name","...","Metric Name","Metric Unit","Metric Value"
    """
    # Strip NCU header lines (start with ==)
    lines = []
    for line in csv_text.strip().split("\n"):
        if not line.startswith("==") and line.strip():
            lines.append(line)

    if not lines:
        return []

    csv_content = "\n".join(lines)
    reader = csv.DictReader(io.StringIO(csv_content))

    # Group metrics by kernel name
    kernel_metrics: Dict[str, Dict[str, NCUMetric]] = {}

    for row in reader:
        kernel_name = row.get("Kernel Name", "unknown")
        metric_name = row.get("Metric Name", "")
        metric_unit = row.get("Metric Unit", "")
        metric_value_str = row.get("Metric Value", "0")

        if not metric_name:
            continue

        try:
            metric_value = float(metric_value_str.replace(",", ""))
        except (ValueError, TypeError):
            metric_value = 0.0

        if kernel_name not in kernel_metrics:
            kernel_metrics[kernel_name] = {}

        kernel_metrics[kernel_name][metric_name] = NCUMetric(
            name=metric_name, value=metric_value, unit=metric_unit,
        )

    return [
        NCUKernelResult(kernel_name=name, metrics=metrics)
        for name, metrics in kernel_metrics.items()
    ]


# ---------------------------------------------------------------------------
# Modal image with NCU support
# ---------------------------------------------------------------------------
ncu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        # CUDA toolkit (includes ncu)
        "apt-get install -y cuda-toolkit-13-0 || apt-get install -y cuda-toolkit-12-8",
        # Explicit nsight-compute install as fallback
        "apt-get install -y nsight-compute-2026.1 || apt-get install -y nsight-compute-2025.3 || true",
    )
    .env({
        "PATH": "/usr/local/cuda/bin:/opt/nvidia/nsight-compute/current:${PATH}",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
    })
    .pip_install("pandas")
)


# ---------------------------------------------------------------------------
# Modal profiling function
# ---------------------------------------------------------------------------
@app.function(image=ncu_image, gpu="B200", timeout=1800)
def profile_kernel_on_modal(
    binary_path: str = "/root/bench/kernel",
    kernel_name_pattern: str = ".*",
    num_captures: int = 1,
    metric_set: str = "full",
    run_args: list = None,
) -> dict:
    """
    Run NCU profiling on a compiled CUDA binary on Modal B200.

    Args:
        binary_path: Path to compiled CUDA binary inside the container.
        kernel_name_pattern: Regex for kernel name filtering (-k flag).
        num_captures: Number of kernel launches to profile (-c flag).
        metric_set: NCU metric set (basic, detailed, full).
        run_args: Additional arguments to pass to the binary.

    Returns:
        Dict with success, csv, ncu_version, parsed metrics, or error info.
    """
    import subprocess

    if run_args is None:
        run_args = []

    # 1. Check NCU availability
    ncu_bin = None
    for candidate in ["ncu", "/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/current/ncu"]:
        check = subprocess.run([candidate, "--version"], capture_output=True, text=True)
        if check.returncode == 0:
            ncu_bin = candidate
            ncu_version = check.stdout.strip()
            break

    if ncu_bin is None:
        return {
            "success": False,
            "fallback": True,
            "error": "NCU not found on this Modal instance",
        }

    # 2. GPU info
    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"GPU: {info.stdout.strip()}")
    print(f"NCU: {ncu_version}")

    # 3. Run kernel normally first (correctness check)
    print(f"\nRunning kernel for correctness: {binary_path}")
    normal_run = subprocess.run(
        [binary_path] + run_args,
        capture_output=True, text=True, timeout=300,
    )
    if normal_run.returncode != 0:
        return {
            "success": False,
            "error": f"Kernel execution failed: {normal_run.stderr}",
        }
    print("Kernel executed successfully")

    # 4. Run with NCU for profiling
    ncu_cmd = [
        ncu_bin,
        "--set", metric_set,
        "--csv",
        "-k", kernel_name_pattern,
        "-c", str(num_captures),
        "--target-processes", "all",
        binary_path,
    ] + run_args

    print(f"\nRunning NCU: {' '.join(ncu_cmd)}")
    ncu_result = subprocess.run(
        ncu_cmd, capture_output=True, text=True, timeout=600,
    )

    if ncu_result.returncode != 0:
        # NCU may fail due to permissions — return fallback
        return {
            "success": False,
            "fallback": True,
            "error": f"NCU failed (may need elevated privileges): {ncu_result.stderr[:500]}",
            "ncu_version": ncu_version,
        }

    print("NCU profiling complete")

    # 5. Optionally capture .ncu-rep
    rep_path = "/root/profile_output.ncu-rep"
    rep_cmd = [
        ncu_bin,
        "--set", metric_set,
        "-o", rep_path.replace(".ncu-rep", ""),  # ncu adds .ncu-rep automatically
        "-k", kernel_name_pattern,
        "-c", str(num_captures),
        binary_path,
    ] + run_args
    subprocess.run(rep_cmd, capture_output=True, text=True, timeout=600)

    return {
        "success": True,
        "csv": ncu_result.stdout,
        "ncu_stderr": ncu_result.stderr,
        "ncu_version": ncu_version,
        "normal_run_stdout": normal_run.stdout,
    }


@app.local_entrypoint()
def main():
    """Smoke test: check if NCU is available on Modal B200."""
    print("Checking NCU availability on Modal B200...")
    result = profile_kernel_on_modal.remote(
        binary_path="/usr/local/cuda/bin/nvcc",  # just check ncu itself
        kernel_name_pattern=".*",
        num_captures=0,
    )
    if result.get("success"):
        print(f"NCU available: {result.get('ncu_version')}")
    elif result.get("fallback"):
        print(f"NCU not available (fallback mode): {result.get('error')}")
    else:
        print(f"Error: {result.get('error')}")
