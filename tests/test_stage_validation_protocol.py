from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bench_synthetic_common import (  # noqa: E402
    RUN_RESULT_FILE_NAME,
    STAGE_VALIDATION_INSPECTION_FOOTER,
    print_stage_validation_logs,
)


def _import_stage_validation_service(monkeypatch):
    monkeypatch.setenv("MODAL_TASK_ID", "1")
    sys.modules.pop("bench_synthetic_service", None)
    return importlib.import_module("bench_synthetic_service")


def test_stage_validation_footer_describes_free_form_result_protocol() -> None:
    assert "last_run_res.txt" in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "last_shell_dump.txt" in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "runtime CuTe object prints" in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "stage_validation_report.json" not in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "coerce" not in STAGE_VALIDATION_INSPECTION_FOOTER.lower()
    assert "[PyTorch]" not in STAGE_VALIDATION_INSPECTION_FOOTER
    assert "[CuTeDSL]" not in STAGE_VALIDATION_INSPECTION_FOOTER


def test_stage_validation_logs_print_passed_cases(capsys) -> None:
    print_stage_validation_logs(
        [
            {
                "num_tokens": 1,
                "status": "PASSED",
                "log": "verbose cute.printf output should remain in last_shell_dump",
            }
        ]
    )

    out = capsys.readouterr().out
    assert "STAGE VALIDATION LOGS (1)" in out
    assert "--- num_tokens=1 status=PASSED ---" in out
    assert "verbose cute.printf output should remain in last_shell_dump" in out


def test_stage_validation_logs_print_only_failing_cases(capsys) -> None:
    print_stage_validation_logs(
        [
            {
                "num_tokens": 1,
                "status": "FAILED",
                "log": "Traceback:\nvalidation crash",
            }
        ]
    )

    out = capsys.readouterr().out
    assert "STAGE VALIDATION LOGS (1)" in out
    assert "--- num_tokens=1 status=FAILED ---" in out
    assert "validation crash" in out


def test_stage_validation_bundle_uses_page_dense_single_case(
    monkeypatch,
) -> None:
    import torch

    svc = _import_stage_validation_service(monkeypatch)

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


def test_stage_validation_service_returns_free_form_payload(
    monkeypatch,
) -> None:
    import torch

    svc = _import_stage_validation_service(monkeypatch)
    monkeypatch.setattr(svc.torch.cuda, "synchronize", lambda: None)

    case = {
        "num_tokens": 1,
        "num_pages": 3,
        "q_nope": torch.zeros((1, 1), dtype=torch.bfloat16),
        "q_pe": torch.zeros((1, 1), dtype=torch.bfloat16),
        "ckv_cache": torch.zeros((1, 1, 1), dtype=torch.bfloat16),
        "kpe_cache": torch.zeros((1, 1, 1), dtype=torch.bfloat16),
        "sparse_indices": torch.zeros((1, 1), dtype=torch.int32),
        "sm_scale": torch.tensor(1.0, dtype=torch.float32),
    }
    bundle = {
        "cases": [case],
        "reference_outputs": [(torch.zeros((1, 1)), torch.zeros((1, 1)))],
    }
    payload = {
        "status": "passed",
        "stage": "S7",
        "comparisons": ["stage7.output allclose ok"],
    }

    class _SolutionModule:
        def run(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
            del q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            return payload

    results, returned_payload = svc._run_stage_validation(
        _SolutionModule(),
        "run",
        bundle,
    )

    assert results[0]["status"] == "PASSED"
    assert results[0]["log"] == ""
    assert returned_payload == payload


def test_stage_validation_service_captures_exceptions_as_payload(
    monkeypatch,
) -> None:
    import torch

    svc = _import_stage_validation_service(monkeypatch)
    monkeypatch.setattr(svc.torch.cuda, "synchronize", lambda: None)

    bundle = {
        "cases": [
            {
                "num_tokens": 1,
                "q_nope": torch.zeros((1, 1), dtype=torch.bfloat16),
                "q_pe": torch.zeros((1, 1), dtype=torch.bfloat16),
                "ckv_cache": torch.zeros((1, 1, 1), dtype=torch.bfloat16),
                "kpe_cache": torch.zeros((1, 1, 1), dtype=torch.bfloat16),
                "sparse_indices": torch.zeros((1, 1), dtype=torch.int32),
                "sm_scale": torch.tensor(1.0, dtype=torch.float32),
            }
        ],
        "reference_outputs": [(torch.zeros((1, 1)), torch.zeros((1, 1)))],
    }

    class _SolutionModule:
        def run(self, *args):
            del args
            print("runtime CuTe object dump")
            raise RuntimeError("stage validation crashed")

    results, returned_payload = svc._run_stage_validation(
        _SolutionModule(),
        "run",
        bundle,
    )

    assert results[0]["status"] == "FAILED"
    assert "runtime CuTe object dump" in results[0]["log"]
    assert "RuntimeError: stage validation crashed" in returned_payload
    assert "Traceback" in returned_payload


def test_stage_validation_remote_result_surfaces_free_form_payload(monkeypatch) -> None:
    svc = _import_stage_validation_service(monkeypatch)

    fake_payload = "all stage values matched"
    stage_bundle_calls: list[dict] = []

    monkeypatch.setattr(
        svc,
        "_ensure_fixture_bundle",
        lambda rebuild_fixture: ({"cases": ["base"]}, "fixture-v1", "fixture-cache"),
    )
    monkeypatch.setattr(
        svc,
        "_build_stage_validation_bundle",
        lambda bundle: stage_bundle_calls.append(bundle) or {"cases": ["stage"]},
    )
    monkeypatch.setattr(
        svc,
        "_ensure_solution_module",
        lambda raw_files, entry_point, entry_file: (object(), "payload-hash", "solution-cache"),
    )
    monkeypatch.setattr(
        svc,
        "_run_stage_validation",
        lambda solution_mod, entry_func, bundle: (
            [
                {
                    "workload_uuid": "synthetic-1",
                    "num_tokens": 1,
                    "num_pages": svc.SYNTHETIC_NUM_PAGES,
                    "status": "PASSED",
                    "log": "",
                }
            ],
            fake_payload,
        ),
    )

    result = svc.run_dsa_synthetic_fast.local(
        raw_files={"kernel_0.py": "AA=="},
        entry_point="kernel_0.py::run",
        stage_validation=True,
    )

    assert result["success"] is True
    assert result["results"][0]["status"] == "PASSED"
    assert result["stage_validation_result"] == fake_payload
    assert stage_bundle_calls == [{"cases": ["base"]}]


def test_stage_validation_launcher_writes_and_replaces_last_run_result(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    import bench_synthetic

    monkeypatch.setattr(bench_synthetic, "PROJECT_ROOT", tmp_path)
    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    (solution_dir / "kernel_0.py").write_text(
        "def run(*args, **kwargs):\n    return 'free-form result'\n",
        encoding="utf-8",
    )

    reports = [
        {
            "success": True,
            "results": [
                {
                    "workload_uuid": "synthetic-1",
                    "num_tokens": 1,
                    "num_pages": 8,
                    "status": "PASSED",
                    "log": "runtime CuTe object dump from passing stage validation",
                }
            ],
            "stage_validation_result": "allclose ok",
            "fixture_version": "fixture-v1",
            "fixture_source": "fixture-cache",
            "solution_source": "solution-cache",
            "payload_hash": "payload-hash",
            "elapsed_s": 0.01,
        },
        {
            "success": True,
            "results": [
                {
                    "workload_uuid": "synthetic-1",
                    "num_tokens": 1,
                    "num_pages": 8,
                    "status": "FAILED",
                    "log": "stage validation crashed",
                }
            ],
            "stage_validation_result": "stage validation crashed",
            "fixture_version": "fixture-v1",
            "fixture_source": "fixture-cache",
            "solution_source": "solution-cache",
            "payload_hash": "payload-hash",
            "elapsed_s": 0.01,
        },
    ]

    class _RemoteFunction:
        def remote(self, **kwargs):
            del kwargs
            return reports.pop(0)

    monkeypatch.setattr(
        bench_synthetic.modal.Function,
        "from_name",
        lambda app_name, function_name: _RemoteFunction(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_synthetic.py",
            "--solution-dir",
            str(solution_dir),
            "--entry-point",
            "kernel_0.py::run",
            "--stage-validation",
        ],
    )

    bench_synthetic.main()
    first_out = capsys.readouterr().out
    result_path = tmp_path / RUN_RESULT_FILE_NAME
    first_result = result_path.read_text(encoding="utf-8")
    assert first_result == "allclose ok\n"
    assert not (solution_dir / "stage_validation_report.json").exists()
    assert "STAGE VALIDATION LOGS (1)" in first_out
    assert "--- num_tokens=1 status=PASSED ---" in first_out
    assert "runtime CuTe object dump from passing stage validation" in first_out

    bench_synthetic.main()
    second_out = capsys.readouterr().out
    second_result = result_path.read_text(encoding="utf-8")
    assert second_result == "stage validation crashed\n"
    assert "allclose ok" not in second_result
    assert "STAGE VALIDATION LOGS (1)" in second_out
    assert "--- num_tokens=1 status=FAILED ---" in second_out
    assert "stage validation crashed" in second_out
