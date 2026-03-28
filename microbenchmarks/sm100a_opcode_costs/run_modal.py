import re
import subprocess
import sys
from pathlib import Path

import modal


app = modal.App("sm100a-opcode-costs")


cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-13-0 || apt-get install -y cuda-toolkit-12-8",
    )
    .env(
        {
            "PATH": "/usr/local/cuda/bin:${PATH}",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
        }
    )
    .add_local_file("microbenchmarks/common/benchmark_common.cuh", "/root/common/benchmark_common.cuh", copy=True)
    .add_local_file("microbenchmarks/tma_gather4/tensor_map_utils.cuh", "/root/bench/tensor_map_utils.cuh", copy=True)
    .add_local_file("microbenchmarks/sm100a_opcode_costs/opcode_kernels.cuh", "/root/bench/opcode_kernels.cuh", copy=True)
    .add_local_file("microbenchmarks/sm100a_opcode_costs/main.cu", "/root/bench/main.cu", copy=True)
    .add_local_dir("csrc/cutlass/include", "/root/bench/include", copy=True)
    .add_local_dir("csrc/kerutils/include", "/root/bench/kerutils_include", copy=True)
)


NVCC_FLAGS = [
    "nvcc",
    "-std=c++20",
    "-gencode",
    "arch=compute_100a,code=sm_100a",
    "-O3",
    "--expt-relaxed-constexpr",
    "-Xptxas",
    "-O3",
    "-Xptxas",
    "--allow-expensive-optimizations=true",
    "-Xptxas",
    "-v",
    "-lineinfo",
    "--ftz=true",
    "-I/root/common",
    "-I/root/bench",
    "-I/root/bench/include",
    "-I/root/bench/kerutils_include",
    "-o",
    "/root/bench/sm100a_opcode_costs_bench",
    "/root/bench/main.cu",
    "-lcuda",
]


@app.function(image=cuda_image, gpu="B200", timeout=1800)
def run_benchmark(suite: str = "all", op: str = "all", smoke: bool = False) -> dict:
    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    nvcc_ver = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)

    compile_result = subprocess.run(NVCC_FLAGS, capture_output=True, text=True, timeout=600)
    if compile_result.returncode != 0:
        return {
            "success": False,
            "phase": "compile",
            "error": compile_result.stderr,
            "gpu": info.stdout.strip(),
            "nvcc": nvcc_ver.stdout.strip(),
            "compile_log": compile_result.stderr,
            "runtime_log": "",
            "csv": "",
            "sass": "",
            "sass_error": "",
        }

    run_cmd = ["/root/bench/sm100a_opcode_costs_bench", f"--suite={suite}", f"--op={op}"]
    if smoke:
        run_cmd.append("--smoke")
    run_cmd.append("--verbose")

    run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=1200)
    sass_result = subprocess.run(
        ["cuobjdump", "--dump-sass", "/root/bench/sm100a_opcode_costs_bench"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if run_result.returncode != 0:
        return {
            "success": False,
            "phase": "runtime",
            "error": run_result.stderr,
            "gpu": info.stdout.strip(),
            "nvcc": nvcc_ver.stdout.strip(),
            "compile_log": compile_result.stderr,
            "runtime_log": run_result.stderr,
            "csv": run_result.stdout,
            "sass": sass_result.stdout if sass_result.returncode == 0 else "",
            "sass_error": sass_result.stderr if sass_result.returncode != 0 else "",
        }

    return {
        "success": True,
        "phase": "done",
        "gpu": info.stdout.strip(),
        "nvcc": nvcc_ver.stdout.strip(),
        "compile_log": compile_result.stderr,
        "runtime_log": run_result.stderr,
        "csv": run_result.stdout,
        "sass": sass_result.stdout if sass_result.returncode == 0 else "",
        "sass_error": sass_result.stderr if sass_result.returncode != 0 else "",
    }


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "all"


@app.local_entrypoint()
def main(
    suite: str = "all",
    op: str = "all",
    smoke: bool = False,
    update: bool = True,
    update_policy: str = "confirmed",
) -> None:
    if update_policy not in {"confirmed", "measured"}:
        raise SystemExit(f"Unsupported update policy: {update_policy}")

    print(f"Launching sm100a opcode cost benchmark on B200 (suite={suite}, op={op}, smoke={smoke})...")
    result = run_benchmark.remote(suite=suite, op=op, smoke=smoke)

    stamp = f"{_slugify(suite)}-{_slugify(op)}"
    if smoke:
        stamp += "-smoke"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    measurements_path = results_dir / f"{stamp}-measurements.csv"
    sass_path = results_dir / f"{stamp}-disassembly.sass"
    summary_path = results_dir / f"{stamp}-summary.csv"
    report_path = results_dir / f"{stamp}-report.md"
    manifest_path = results_dir / f"{stamp}-manifest.json"
    log_path = results_dir / f"{stamp}-run.log"

    measurements_path.write_text(result.get("csv", ""))
    sass_path.write_text(result.get("sass", ""))
    log_path.write_text(
        "\n".join(
            [
                f"Success: {result.get('success', False)}",
                f"Phase: {result.get('phase', 'unknown')}",
                f"GPU: {result['gpu']}",
                f"NVCC: {result['nvcc']}",
                "",
                "--- compile log ---",
                result.get("compile_log", ""),
                "",
                "--- runtime log ---",
                result.get("runtime_log", ""),
                "",
                "--- error ---",
                result.get("error", ""),
                "",
                "--- sass error ---",
                result.get("sass_error", ""),
            ]
        )
        + "\n"
    )

    if not result["success"]:
        print("Benchmark failed.")
        print(result.get("error", "unknown error"))
        print(f"Wrote partial measurements to {measurements_path}")
        print(f"Wrote disassembly to {sass_path}")
        print(f"Wrote run log to {log_path}")

    verify_script = Path(__file__).resolve().parent / "verify_and_update.py"
    verify_cmd = [
        sys.executable,
        str(verify_script),
        "--measurements",
        str(measurements_path),
        "--disassembly",
        str(sass_path),
        "--summary-out",
        str(summary_path),
        "--report-out",
        str(report_path),
        "--manifest-out",
        str(manifest_path),
        "--update-policy",
        update_policy,
    ]
    if not update:
        verify_cmd.append("--no-update")

    if measurements_path.read_text().strip() and sass_path.read_text().strip():
        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
        if verify_result.returncode != 0:
            print("Post-processing failed.")
            print(verify_result.stderr or verify_result.stdout)
            return
    else:
        summary_path.write_text("")
        report_path.write_text("# SM100A Opcode Cost Confirmation\n\n- No measurements were produced.\n")
        manifest_path.write_text("{}\n")

    print(f"Wrote measurements to {measurements_path}")
    print(f"Wrote disassembly to {sass_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote report to {report_path}")
    print(f"Wrote manifest to {manifest_path}")
    if result["success"] and update:
        print(f"Updated tools/sass/dump_sass_modal.py using update policy: {update_policy}.")
