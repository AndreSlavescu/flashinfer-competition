from __future__ import annotations

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage_validation_parser import compare_tagged_outputs  # noqa: E402


def _transcript(body: str) -> str:
    return textwrap.dedent(body).strip("\n") + "\n"


def test_matching_csv_block_passes() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] q_tile: BEGIN
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0
        [PyTorch Val] q_tile: END
        [CuTe Val] q_tile: BEGIN
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0
        [CuTe Val] q_tile: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert report.overall_pass
    assert len(report.fields) == 1
    (fr,) = report.fields
    assert fr.status == "PASS"
    assert fr.pytorch_shape is None
    assert fr.cute_shape is None


def test_multiline_csv_with_whitespace_and_trailing_commas_passes() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] a: BEGIN
          1.0, 2.0,
          3.0, 4.0,
        [PyTorch Val] a: END
        [CuTe Val] a: BEGIN
        1.0,
        2.0, 3.0, 4.0,
        [CuTe Val] a: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert report.overall_pass


def test_value_mismatch_reports_top_diffs() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] q: BEGIN
        0.0, 1.0, 2.0, 3.0
        [PyTorch Val] q: END
        [CuTe Val] q: BEGIN
        0.0, 1.0, 2.0, 10.0
        [CuTe Val] q: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert not report.overall_pass
    (fr,) = report.fields
    assert fr.status == "FAIL"
    assert fr.max_abs_diff is not None and fr.max_abs_diff > 6.9
    assert "i=3" in fr.detail


def test_element_count_mismatch_flagged() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] x: BEGIN
        1.0, 2.0, 3.0
        [PyTorch Val] x: END
        [CuTe Val] x: BEGIN
        1.0, 2.0, 3.0, 4.0
        [CuTe Val] x: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert not report.overall_pass
    (fr,) = report.fields
    assert fr.status == "FAIL"
    assert "pytorch=3" in fr.detail and "cute=4" in fr.detail


def test_missing_sides_reported() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] only_pt: BEGIN
        1.0
        [PyTorch Val] only_pt: END
        [CuTe Val] only_cute: BEGIN
        7.0
        [CuTe Val] only_cute: END
        """
    )
    report = compare_tagged_outputs(transcript)
    statuses = {fr.name: fr.status for fr in report.fields}
    assert statuses == {"only_pt": "MISSING_CUTE", "only_cute": "MISSING_PYTORCH"}
    assert not report.overall_pass


def test_duplicate_block_reported_as_parse_error() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] dup: BEGIN
        1.0
        [PyTorch Val] dup: END
        [PyTorch Val] dup: BEGIN
        2.0
        [PyTorch Val] dup: END
        [CuTe Val] dup: BEGIN
        1.0
        [CuTe Val] dup: END
        """
    )
    report = compare_tagged_outputs(transcript)
    statuses = [fr.status for fr in report.fields]
    assert "PARSE_ERROR" in statuses


def test_empty_transcript_returns_empty_report() -> None:
    report = compare_tagged_outputs("no tags here\njust noise\n")
    assert report.fields == []
    assert not report.overall_pass


def test_signed_ints_and_scientific_notation_parse() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] mixed: BEGIN
        -1, 2.5e-3, +4, 0.0
        [PyTorch Val] mixed: END
        [CuTe Val] mixed: BEGIN
        -1.0, 0.0025, 4.0, 0
        [CuTe Val] mixed: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert report.overall_pass


def test_malformed_csv_body_is_parse_error() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] bad: BEGIN
        1.0, nope, 3.0
        [PyTorch Val] bad: END
        [CuTe Val] bad: BEGIN
        1.0, 2.0, 3.0
        [CuTe Val] bad: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert not report.overall_pass
    (fr,) = [field for field in report.fields if field.name == "bad"]
    assert fr.status == "PARSE_ERROR"
    assert "CSV numeric values" in fr.detail


def test_host_scope_block_surfaces_as_info_row() -> None:
    transcript = _transcript(
        """
        [CuTe Host] qk_tiled_mma: BEGIN
        TiledMMA
          ThrLayoutVMNK: (_128,_1,_1,_1):(_1,_0,_0,_0)
        [CuTe Host] qk_tiled_mma: END
        """
    )
    report = compare_tagged_outputs(transcript)
    (fr,) = report.fields
    assert fr.status == "INFO"
    assert fr.name == "qk_tiled_mma"
    assert "CuTe Host" in fr.detail
    assert "TiledMMA" in fr.detail
    assert report.overall_pass
    summary = report.format_summary()
    assert "[INFO] qk_tiled_mma" in summary


def test_pytorch_host_block_surfaces_as_info_row() -> None:
    transcript = _transcript(
        """
        [PyTorch Host] q_debug: BEGIN
        tensor([[1., 2.]], device='cuda:0')
        [PyTorch Host] q_debug: END
        """
    )
    report = compare_tagged_outputs(transcript)
    (fr,) = report.fields
    assert fr.status == "INFO"
    assert fr.name == "q_debug"
    assert "PyTorch Host" in fr.detail
    assert "tensor([[1., 2.]]" in fr.detail
    assert report.overall_pass


def test_host_blocks_do_not_mask_validation_failure() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] out: BEGIN
        1.0, 2.0
        [PyTorch Val] out: END
        [CuTe Val] out: BEGIN
        9.0, 9.0
        [CuTe Val] out: END
        [PyTorch Host] q_obj: BEGIN
        tensor([1., 2.], device='cuda:0')
        [PyTorch Host] q_obj: END
        [CuTe Host] mma_atom: BEGIN
        MMA_Atom SM100_MMA_F16_SS
        [CuTe Host] mma_atom: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert not report.overall_pass
    statuses = {fr.name: fr.status for fr in report.fields}
    assert statuses == {"out": "FAIL", "q_obj": "INFO", "mma_atom": "INFO"}


def test_duplicate_host_block_reported_as_parse_error() -> None:
    transcript = _transcript(
        """
        [PyTorch Host] shared_storage: BEGIN
        object 1
        [PyTorch Host] shared_storage: END
        [PyTorch Host] shared_storage: BEGIN
        object 2
        [PyTorch Host] shared_storage: END
        """
    )
    report = compare_tagged_outputs(transcript)
    statuses = [fr.status for fr in report.fields]
    assert "PARSE_ERROR" in statuses


def test_summary_omits_shape_text_under_csv_protocol() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] frag: BEGIN
        1.0, 2.0
        [PyTorch Val] frag: END
        [CuTe Val] frag: BEGIN
        1.0, 2.0
        [CuTe Val] frag: END
        """
    )
    report = compare_tagged_outputs(transcript)
    summary = report.format_summary()
    assert "[PASS] frag" in summary
    assert "pt_shape=" not in summary
    assert report.overall_pass
