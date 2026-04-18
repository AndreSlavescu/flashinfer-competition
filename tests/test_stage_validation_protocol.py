from __future__ import annotations

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
