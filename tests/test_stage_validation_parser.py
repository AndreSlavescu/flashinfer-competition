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


def test_matching_2d_block_passes() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] q_tile: BEGIN
        shape=(2,3) dtype=torch.float32 scope=gmem
        data=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        [PyTorch Val] q_tile: END
        [CuTe Val] q_tile: BEGIN
        tensor(raw_ptr(0x00007f5f81200400: f32, gmem, align<4>) o (2,3):(3,1), data=
               [[ 0.000000,  1.000000,  2.000000, ],
                [ 3.000000,  4.000000,  5.000000, ]])
        [CuTe Val] q_tile: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert report.overall_pass
    assert len(report.fields) == 1
    (fr,) = report.fields
    assert fr.status == "PASS"
    assert fr.pytorch_shape == (2, 3)
    assert fr.cute_shape == (2, 3)


def test_address_variance_does_not_affect_parse() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] a: BEGIN
        shape=(4,) dtype=torch.float32 scope=gmem
        data=[1.0, 2.0, 3.0, 4.0]
        [PyTorch Val] a: END
        [CuTe Val] a: BEGIN
        tensor(raw_ptr(0xDEADBEEFCAFE: f32, gmem, align<4>) o (4):(1), data=
               [ 1.000000,  2.000000,  3.000000,  4.000000, ])
        [CuTe Val] a: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert report.overall_pass


def test_value_mismatch_reports_top_diffs() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] q: BEGIN
        shape=(2,2) dtype=torch.float32 scope=rmem
        data=[0.0, 1.0, 2.0, 3.0]
        [PyTorch Val] q: END
        [CuTe Val] q: BEGIN
        tensor(raw_ptr(0x1: f32, rmem, align<32>) o (2,2):(2,1), data=
               [[ 0.000000,  1.000000, ],
                [ 2.000000, 10.000000, ]])
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
        shape=(3,) dtype=torch.float32 scope=gmem
        data=[1.0, 2.0, 3.0]
        [PyTorch Val] x: END
        [CuTe Val] x: BEGIN
        tensor(raw_ptr(0x1: f32, gmem, align<4>) o (4):(1), data=
               [ 1.000000,  2.000000,  3.000000,  4.000000, ])
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
        shape=(1,) dtype=torch.float32 scope=gmem
        data=[1.0]
        [PyTorch Val] only_pt: END
        [CuTe Val] only_cute: BEGIN
        tensor(raw_ptr(0x1: f32, gmem, align<4>) o (1):(1), data=
               [ 7.000000, ])
        [CuTe Val] only_cute: END
        """
    )
    report = compare_tagged_outputs(transcript)
    statuses = {fr.name: fr.status for fr in report.fields}
    assert statuses == {"only_pt": "MISSING_CUTE", "only_cute": "MISSING_PYTORCH"}
    assert not report.overall_pass


def test_3d_verbose_cute_block_parses() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] t3: BEGIN
        shape=(2,2,2) dtype=torch.float32 scope=gmem
        data=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        [PyTorch Val] t3: END
        [CuTe Val] t3: BEGIN
        tensor(raw_ptr(0xA: f32, gmem, align<4>) o (2,2,2):(4,2,1), data= (
            (0,0,0)= 0.000000
            (0,0,1)= 1.000000
            (0,1,0)= 2.000000
            (0,1,1)= 3.000000
            (1,0,0)= 4.000000
            (1,0,1)= 5.000000
            (1,1,0)= 6.000000
            (1,1,1)= 7.000000
        )
        [CuTe Val] t3: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert report.overall_pass, report.format_summary()


def test_duplicate_block_reported_as_parse_error() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] dup: BEGIN
        shape=(1,) dtype=torch.float32 scope=gmem
        data=[1.0]
        [PyTorch Val] dup: END
        [PyTorch Val] dup: BEGIN
        shape=(1,) dtype=torch.float32 scope=gmem
        data=[2.0]
        [PyTorch Val] dup: END
        [CuTe Val] dup: BEGIN
        tensor(raw_ptr(0x1: f32, gmem, align<4>) o (1):(1), data=
               [ 1.000000, ])
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


def test_negative_values_and_scientific_notation_parse() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] mixed: BEGIN
        shape=(3,) dtype=torch.float32 scope=gmem
        data=[-1.5, 2.5e-3, 0.0]
        [PyTorch Val] mixed: END
        [CuTe Val] mixed: BEGIN
        tensor(raw_ptr(0x1: f32, gmem, align<4>) o (3):(1), data=
               [-1.500000,  0.002500,  0.000000, ])
        [CuTe Val] mixed: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert report.overall_pass


def test_host_scope_block_surfaces_as_info_row() -> None:
    transcript = _transcript(
        """
        [CuTe Host] qk_tiled_mma: BEGIN
        TiledMMA
          ThrLayoutVMNK: (_128,_1,_1,_1):(_1,_0,_0,_0)
          PermutationMNK: (_,_,_)
          MMA_Atom
            ThrID: 128:1
        [CuTe Host] qk_tiled_mma: END
        """
    )
    report = compare_tagged_outputs(transcript)
    (fr,) = report.fields
    assert fr.status == "INFO"
    assert fr.name == "qk_tiled_mma"
    assert "TiledMMA" in fr.detail
    assert report.overall_pass
    summary = report.format_summary()
    assert "[INFO] qk_tiled_mma" in summary


def test_host_scope_without_pytorch_counterpart_does_not_flag_missing() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] out: BEGIN
        shape=(2,) dtype=torch.float32 scope=gmem
        data=[1.0, 2.0]
        [PyTorch Val] out: END
        [CuTe Val] out: BEGIN
        tensor(raw_ptr(0x1: f32, gmem, align<4>) o (2):(1), data=
               [ 1.000000,  2.000000, ])
        [CuTe Val] out: END
        [CuTe Host] smem_layout_q: BEGIN
        (((_64,_64)),_1):(((_1,_64)),_0)
        [CuTe Host] smem_layout_q: END
        """
    )
    report = compare_tagged_outputs(transcript)
    statuses = {fr.name: fr.status for fr in report.fields}
    assert statuses == {"out": "PASS", "smem_layout_q": "INFO"}
    assert report.overall_pass


def test_host_scope_does_not_mask_validation_failure() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] out: BEGIN
        shape=(2,) dtype=torch.float32 scope=gmem
        data=[1.0, 2.0]
        [PyTorch Val] out: END
        [CuTe Val] out: BEGIN
        tensor(raw_ptr(0x1: f32, gmem, align<4>) o (2):(1), data=
               [ 9.000000,  9.000000, ])
        [CuTe Val] out: END
        [CuTe Host] mma_atom: BEGIN
        MMA_Atom SM100_MMA_F16_SS
        [CuTe Host] mma_atom: END
        """
    )
    report = compare_tagged_outputs(transcript)
    assert not report.overall_pass
    statuses = {fr.name: fr.status for fr in report.fields}
    assert statuses == {"out": "FAIL", "mma_atom": "INFO"}


def test_duplicate_host_block_reported_as_parse_error() -> None:
    transcript = _transcript(
        """
        [CuTe Host] shared_storage: BEGIN
        struct SharedStorage { ... }
        [CuTe Host] shared_storage: END
        [CuTe Host] shared_storage: BEGIN
        struct SharedStorage { ... }
        [CuTe Host] shared_storage: END
        """
    )
    report = compare_tagged_outputs(transcript)
    statuses = [fr.status for fr in report.fields]
    assert "PARSE_ERROR" in statuses


def test_scope_metadata_preserved() -> None:
    transcript = _transcript(
        """
        [PyTorch Val] frag: BEGIN
        shape=(2,) dtype=torch.float32 scope=rmem
        data=[1.0, 2.0]
        [PyTorch Val] frag: END
        [CuTe Val] frag: BEGIN
        tensor(raw_ptr(0x1: f32, rmem, align<32>) o (2):(1), data=
               [ 1.000000,  2.000000, ])
        [CuTe Val] frag: END
        """
    )
    report = compare_tagged_outputs(transcript)
    summary = report.format_summary()
    assert "[PASS] frag" in summary
    assert report.overall_pass
