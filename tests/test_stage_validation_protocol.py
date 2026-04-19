from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bench_synthetic_common import STAGE_VALIDATION_INSPECTION_FOOTER  # noqa: E402


def test_stage_validation_footer_describes_inspection_protocol() -> None:
    assert "inspection-only" in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "[PyTorch]" in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "[CuTeDSL]" in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "last_shell_dump.txt" in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "grep_search" in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "read_file" in STAGE_VALIDATION_INSPECTION_FOOTER


def test_stage_validation_footer_no_longer_mentions_parser_summary() -> None:
    assert "Stage Validation Summary" not in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "parser" not in STAGE_VALIDATION_INSPECTION_FOOTER.lower()


def test_stage_validation_bundle_uses_page_dense_single_case(
    monkeypatch,
) -> None:
    import torch

    monkeypatch.setenv("MODAL_TASK_ID", "1")
    sys.modules.pop("bench_synthetic_service", None)
    svc = importlib.import_module("bench_synthetic_service")

    num_pages = 8
    case = {
        "num_tokens": svc.STAGE_VALIDATION_NUM_TOKENS,
        "q_nope": torch.zeros(
            (svc.STAGE_VALIDATION_NUM_TOKENS, svc.SYNTHETIC_NUM_QO_HEADS, svc.SYNTHETIC_CKV_DIM),
            dtype=torch.bfloat16,
        ),
        "q_pe": torch.zeros(
            (svc.STAGE_VALIDATION_NUM_TOKENS, svc.SYNTHETIC_NUM_QO_HEADS, svc.SYNTHETIC_KPE_DIM),
            dtype=torch.bfloat16,
        ),
        "ckv_cache": torch.zeros(
            (num_pages, svc.SYNTHETIC_PAGE_SIZE, svc.SYNTHETIC_CKV_DIM),
            dtype=torch.bfloat16,
        ),
        "kpe_cache": torch.zeros(
            (num_pages, svc.SYNTHETIC_PAGE_SIZE, svc.SYNTHETIC_KPE_DIM),
            dtype=torch.bfloat16,
        ),
        "sparse_indices": torch.zeros(
            (svc.STAGE_VALIDATION_NUM_TOKENS, svc.SYNTHETIC_TOPK),
            dtype=torch.int32,
        ),
        "sm_scale": torch.tensor(svc.SYNTHETIC_SM_SCALE, dtype=torch.float32),
    }
    bundle = {
        "cases": [case],
        "reference_outputs": [
            (
                torch.zeros(
                    (svc.STAGE_VALIDATION_NUM_TOKENS, svc.SYNTHETIC_NUM_QO_HEADS, svc.SYNTHETIC_CKV_DIM),
                    dtype=torch.bfloat16,
                ),
                torch.zeros(
                    (svc.STAGE_VALIDATION_NUM_TOKENS, svc.SYNTHETIC_NUM_QO_HEADS),
                    dtype=torch.float32,
                ),
            )
        ],
    }

    captured: dict[str, torch.Tensor] = {}

    class _ReferenceModule:
        def run(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
            del q_pe, ckv_cache, kpe_cache, sm_scale
            captured["sparse_indices"] = sparse_indices.clone()
            return (
                torch.zeros(
                    (q_nope.shape[0], svc.SYNTHETIC_NUM_QO_HEADS, svc.SYNTHETIC_CKV_DIM),
                    dtype=torch.bfloat16,
                    device=q_nope.device,
                ),
                torch.zeros(
                    (q_nope.shape[0], svc.SYNTHETIC_NUM_QO_HEADS),
                    dtype=torch.float32,
                    device=q_nope.device,
                ),
            )

    monkeypatch.setattr(svc, "_load_reference_module", lambda: _ReferenceModule())
    monkeypatch.setattr(svc.torch.cuda, "synchronize", lambda: None)

    stage_bundle = svc._build_stage_validation_bundle(bundle)
    stage_case = stage_bundle["cases"][0]

    expected = torch.full_like(case["sparse_indices"], -1)
    for slot, page_id in enumerate((1, 3, 5)):
        start = slot * svc.SYNTHETIC_PAGE_SIZE
        stop = start + svc.SYNTHETIC_PAGE_SIZE
        expected[:, start:stop] = torch.arange(
            page_id * svc.SYNTHETIC_PAGE_SIZE,
            (page_id + 1) * svc.SYNTHETIC_PAGE_SIZE,
            dtype=expected.dtype,
        )

    assert torch.equal(stage_case["sparse_indices"], expected)
    assert torch.equal(captured["sparse_indices"], expected)
