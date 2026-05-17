"""Trace the PyTorch computation graph of the DSA sparse attention reference on Modal B200.

Captures FX graphs and graph breaks via ``torch._dynamo.explain`` and (optionally) a flat
``torch.jit.trace`` view, then writes artifacts to a local notes directory.

Usage:
    .venv/bin/modal run scripts/trace_dsa_graph.py --num-tokens 4 --num-pages 64 --mode dynamo
    .venv/bin/modal run scripts/trace_dsa_graph.py --mode both
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_SOURCE_PATH = SCRIPT_DIR / "bench_synthetic_common.py"
REFERENCE_FILENAME = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.py"
LOCAL_REFERENCE_SOURCE_PATH = SCRIPT_DIR.parent / "references" / REFERENCE_FILENAME
REMOTE_COMMON_PATH = "/root/bench_synthetic_common.py"
REMOTE_REFERENCE_DIR = "/root/references"
REMOTE_REFERENCE_PATH = f"{REMOTE_REFERENCE_DIR}/{REFERENCE_FILENAME}"

app = modal.App("dsa-graph-trace")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .add_local_file(str(COMMON_SOURCE_PATH), remote_path=REMOTE_COMMON_PATH)
    .add_local_file(str(LOCAL_REFERENCE_SOURCE_PATH), remote_path=REMOTE_REFERENCE_PATH)
)


@app.function(gpu="B200:1", image=image, timeout=600)
def trace_graph(num_tokens: int, num_pages: int, mode: str) -> dict:
    import importlib.util
    import io
    from contextlib import redirect_stdout

    import torch

    sys.path.insert(0, "/root")
    from bench_synthetic_common import (
        FIXTURE_SHARED_CACHE_SEED,
        SYNTHETIC_CKV_DIM,
        SYNTHETIC_KPE_DIM,
        SYNTHETIC_NUM_QO_HEADS,
        SYNTHETIC_PAGE_SIZE,
        SYNTHETIC_SM_SCALE,
        SYNTHETIC_TOPK,
    )

    spec = importlib.util.spec_from_file_location("dsa_reference", REMOTE_REFERENCE_PATH)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    run = ref_mod.run

    torch.manual_seed(FIXTURE_SHARED_CACHE_SEED)
    torch.cuda.manual_seed(FIXTURE_SHARED_CACHE_SEED)

    total_kv = num_pages * SYNTHETIC_PAGE_SIZE
    q_nope = torch.randn(num_tokens, SYNTHETIC_NUM_QO_HEADS, SYNTHETIC_CKV_DIM,
                         dtype=torch.bfloat16, device="cuda") / 10.0
    q_pe = torch.randn(num_tokens, SYNTHETIC_NUM_QO_HEADS, SYNTHETIC_KPE_DIM,
                       dtype=torch.bfloat16, device="cuda") / 10.0
    ckv_cache = torch.randn(num_pages, SYNTHETIC_PAGE_SIZE, SYNTHETIC_CKV_DIM,
                            dtype=torch.bfloat16, device="cuda") / 10.0
    kpe_cache = torch.randn(num_pages, SYNTHETIC_PAGE_SIZE, SYNTHETIC_KPE_DIM,
                            dtype=torch.bfloat16, device="cuda") / 10.0
    sparse_indices = torch.randint(0, total_kv, (num_tokens, SYNTHETIC_TOPK),
                                   dtype=torch.int32, device="cuda")
    invalid_mask = torch.rand(num_tokens, SYNTHETIC_TOPK, device="cuda") < 0.05
    sparse_indices[invalid_mask] = -1
    sm_scale = torch.tensor(SYNTHETIC_SM_SCALE, dtype=torch.float32, device="cuda")

    args = (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

    out: dict = {
        "graphs": [],
        "graph_break_log": "",
        "jit_graph": None,
        "jit_code": None,
        "summary": {},
    }

    if mode in ("dynamo", "both"):
        torch._dynamo.reset()
        torch._dynamo.config.capture_scalar_outputs = True

        explain = torch._dynamo.explain(run)(*args)

        graphs_data = []
        for i, gm in enumerate(explain.graphs):
            tab_buf = io.StringIO()
            with redirect_stdout(tab_buf):
                gm.graph.print_tabular()
            graphs_data.append({
                "idx": i,
                "tabular": tab_buf.getvalue(),
                "code": gm.code,
                "num_nodes": len(gm.graph.nodes),
            })

        out["graphs"] = graphs_data
        out["graph_break_log"] = str(explain)
        out["summary"]["graph_count"] = explain.graph_count
        out["summary"]["graph_break_count"] = explain.graph_break_count
        out["summary"]["op_count"] = explain.op_count

        torch._dynamo.reset()
        eager_out, eager_lse = run(*args)
        compiled_out, compiled_lse = torch.compile(run, dynamic=False)(*args)
        out["summary"]["verify_output_max_abs"] = float(
            (eager_out.float() - compiled_out.float()).abs().max()
        )
        out["summary"]["verify_lse_max_abs"] = float(
            (eager_lse.float() - compiled_lse.float()).abs().max()
        )

    if mode in ("jit", "both"):
        traced = torch.jit.trace(run, args, strict=False, check_trace=False)
        out["jit_graph"] = str(traced.graph)
        out["jit_code"] = traced.code

    return out


@app.local_entrypoint()
def main(
    num_tokens: int = 4,
    num_pages: int = 64,
    mode: str = "dynamo",
    out_dir: str = "notes/dsa_attention/traces",
):
    if mode not in ("dynamo", "jit", "both"):
        raise SystemExit(f"mode must be one of dynamo|jit|both, got {mode!r}")

    result = trace_graph.remote(num_tokens, num_pages, mode)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for graph in result["graphs"]:
        (out_path / f"graph_{graph['idx']}.code.py").write_text(graph["code"])
        (out_path / f"graph_{graph['idx']}.tabular.txt").write_text(graph["tabular"])

    if result["graph_break_log"]:
        (out_path / "graph_breaks.log").write_text(result["graph_break_log"])

    if result["jit_graph"]:
        (out_path / "jit_graph.txt").write_text(result["jit_graph"])
        (out_path / "jit_code.py").write_text(result["jit_code"])

    s = result["summary"]
    total_nodes = sum(g["num_nodes"] for g in result["graphs"])
    print(f"Wrote {len(result['graphs'])} FX graph(s) ({total_nodes} nodes) to {out_path}")
    if s:
        print(
            f"  graph_count={s.get('graph_count')} "
            f"graph_breaks={s.get('graph_break_count')} "
            f"op_count={s.get('op_count')}"
        )
        if "verify_output_max_abs" in s:
            print(
                f"  verify: out_max_abs={s['verify_output_max_abs']:.3e} "
                f"lse_max_abs={s['verify_lse_max_abs']:.3e}"
            )
