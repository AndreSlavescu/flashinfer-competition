"""Parser for stage-validation transcripts emitted via runtime-tagged prints.

The stage validation protocol relies on tagged print streams inside the
synthetic bench stdout:

    [PyTorch Val] <field>: BEGIN
    0.0, 1.0, 2.0, 3.0
    [PyTorch Val] <field>: END

    [CuTe Val] <field>: BEGIN
    0.0, 1.0, 2.0, 3.0
    [CuTe Val] <field>: END

    [PyTorch Host] <name>: BEGIN
    <python print(obj) output for an arbitrary PyTorch object>
    [PyTorch Host] <name>: END

    [CuTe Host] <name>: BEGIN
    <python print(obj) output for a host-side CuTe object>
    [CuTe Host] <name>: END

`[PyTorch Val]` and `[CuTe Val]` blocks are machine-compared and must contain
comma-separated numeric values. Signed ints, decimals, and scientific notation
are accepted; whitespace and multiline CSV are allowed; trailing commas are
ignored.

`[PyTorch Host]` and `[CuTe Host]` blocks are inspection-only: they surface raw
debug output without gating `overall_pass`.

Single-part-per-field: duplicate BEGIN for the same (source, field) is a parse
error. Fields present on only one numeric side (PyTorch Val / CuTe Val) are
reported as MISSING.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

MAX_ELEMENTS = 1 << 20

_PYTORCH_BEGIN = re.compile(r"^\s*\[PyTorch Val\]\s+(?P<name>.+?)\s*:\s*BEGIN\s*$")
_PYTORCH_END = re.compile(r"^\s*\[PyTorch Val\]\s+(?P<name>.+?)\s*:\s*END\s*$")
_CUTE_BEGIN = re.compile(r"^\s*\[CuTe Val\]\s+(?P<name>.+?)\s*:\s*BEGIN\s*$")
_CUTE_END = re.compile(r"^\s*\[CuTe Val\]\s+(?P<name>.+?)\s*:\s*END\s*$")
_PYTORCH_HOST_BEGIN = re.compile(
    r"^\s*\[PyTorch Host\]\s+(?P<name>.+?)\s*:\s*BEGIN\s*$"
)
_PYTORCH_HOST_END = re.compile(
    r"^\s*\[PyTorch Host\]\s+(?P<name>.+?)\s*:\s*END\s*$"
)
_CUTE_HOST_BEGIN = re.compile(r"^\s*\[CuTe Host\]\s+(?P<name>.+?)\s*:\s*BEGIN\s*$")
_CUTE_HOST_END = re.compile(r"^\s*\[CuTe Host\]\s+(?P<name>.+?)\s*:\s*END\s*$")


@dataclass
class RawBlock:
    name: str
    lines: list[str] = field(default_factory=list)


@dataclass
class ParsedNumeric:
    name: str
    values: np.ndarray


@dataclass
class FieldReport:
    name: str
    status: str  # "PASS" | "FAIL" | "MISSING_PYTORCH" | "MISSING_CUTE" | "PARSE_ERROR" | "INFO"
    detail: str = ""
    pytorch_shape: tuple[int, ...] | None = None
    cute_shape: tuple[int, ...] | None = None
    max_abs_diff: float | None = None


@dataclass
class ComparisonReport:
    fields: list[FieldReport]
    overall_pass: bool

    def format_summary(self) -> str:
        lines = ["=== Stage Validation Summary ==="]
        if not self.fields:
            lines.append("No tagged validation blocks found in transcript.")
            lines.append(f"Overall: {'PASS' if self.overall_pass else 'FAIL'}")
            return "\n".join(lines)
        for fr in self.fields:
            head = f"[{fr.status}] {fr.name}"
            if fr.pytorch_shape is not None or fr.cute_shape is not None:
                head += (
                    f" (pt_shape={fr.pytorch_shape}, cute_shape={fr.cute_shape})"
                )
            if fr.max_abs_diff is not None:
                head += f" max_abs_diff={fr.max_abs_diff:.3e}"
            lines.append(head)
            if fr.detail:
                for extra in fr.detail.splitlines():
                    lines.append(f"    {extra}")
        lines.append(f"Overall: {'PASS' if self.overall_pass else 'FAIL'}")
        return "\n".join(lines)


class ParseError(ValueError):
    """Raised when a tagged block cannot be parsed."""


def _extract_blocks(
    transcript: str,
) -> tuple[
    dict[str, RawBlock],
    dict[str, RawBlock],
    dict[str, RawBlock],
    dict[str, RawBlock],
    list[str],
]:
    pytorch: dict[str, RawBlock] = {}
    cute: dict[str, RawBlock] = {}
    pytorch_host: dict[str, RawBlock] = {}
    cute_host: dict[str, RawBlock] = {}
    errors: list[str] = []

    current: RawBlock | None = None
    current_source: str | None = None  # "pytorch" | "cute" | "pytorch_host" | "cute_host"

    def _store_block(
        target: dict[str, RawBlock],
        source_label: str,
        name: str,
    ) -> None:
        nonlocal current, current_source
        assert current is not None
        if name in target:
            errors.append(f"Duplicate {source_label} block for {name!r}.")
        else:
            target[name] = current
        current = None
        current_source = None

    for raw_line in transcript.splitlines():
        line = raw_line.rstrip("\n")
        m = _PYTORCH_BEGIN.match(line)
        if m:
            if current is not None:
                errors.append(
                    f"Unclosed {current_source} block for {current.name!r} before new BEGIN."
                )
            current_source = "pytorch"
            current = RawBlock(name=m.group("name").strip())
            continue
        m = _CUTE_BEGIN.match(line)
        if m:
            if current is not None:
                errors.append(
                    f"Unclosed {current_source} block for {current.name!r} before new BEGIN."
                )
            current_source = "cute"
            current = RawBlock(name=m.group("name").strip())
            continue
        m = _PYTORCH_HOST_BEGIN.match(line)
        if m:
            if current is not None:
                errors.append(
                    f"Unclosed {current_source} block for {current.name!r} before new BEGIN."
                )
            current_source = "pytorch_host"
            current = RawBlock(name=m.group("name").strip())
            continue
        m = _CUTE_HOST_BEGIN.match(line)
        if m:
            if current is not None:
                errors.append(
                    f"Unclosed {current_source} block for {current.name!r} before new BEGIN."
                )
            current_source = "cute_host"
            current = RawBlock(name=m.group("name").strip())
            continue
        m = _PYTORCH_END.match(line)
        if m:
            name = m.group("name").strip()
            if current is None or current_source != "pytorch" or current.name != name:
                errors.append(f"Unexpected PyTorch END for {name!r}.")
                current = None
                current_source = None
                continue
            _store_block(pytorch, "PyTorch", name)
            continue
        m = _CUTE_END.match(line)
        if m:
            name = m.group("name").strip()
            if current is None or current_source != "cute" or current.name != name:
                errors.append(f"Unexpected CuTe END for {name!r}.")
                current = None
                current_source = None
                continue
            _store_block(cute, "CuTe", name)
            continue
        m = _PYTORCH_HOST_END.match(line)
        if m:
            name = m.group("name").strip()
            if current is None or current_source != "pytorch_host" or current.name != name:
                errors.append(f"Unexpected PyTorch Host END for {name!r}.")
                current = None
                current_source = None
                continue
            _store_block(pytorch_host, "PyTorch Host", name)
            continue
        m = _CUTE_HOST_END.match(line)
        if m:
            name = m.group("name").strip()
            if current is None or current_source != "cute_host" or current.name != name:
                errors.append(f"Unexpected CuTe Host END for {name!r}.")
                current = None
                current_source = None
                continue
            _store_block(cute_host, "CuTe Host", name)
            continue
        if current is not None:
            current.lines.append(line)

    if current is not None:
        errors.append(
            f"Transcript ended mid-block: {current_source} {current.name!r}."
        )

    return pytorch, cute, pytorch_host, cute_host, errors


def _parse_csv_numeric_values(block: RawBlock, *, source_label: str) -> ParsedNumeric:
    body = "\n".join(block.lines).strip()
    tokens = [token.strip() for token in body.split(",")]
    values_raw = [token for token in tokens if token]
    if not values_raw:
        raise ParseError(f"No comma-separated numeric data found in {source_label} block.")
    try:
        values = np.asarray([float(token) for token in values_raw], dtype=np.float64)
    except ValueError as exc:
        raise ParseError(f"Could not parse {source_label} CSV numeric values: {exc}") from exc
    if values.size > MAX_ELEMENTS:
        raise ParseError(
            f"{source_label} block has {values.size} elements; exceeds cap {MAX_ELEMENTS}."
        )
    return ParsedNumeric(name=block.name, values=values)


def _compare_field(
    name: str,
    pt: ParsedNumeric,
    ct: ParsedNumeric,
    rtol: float,
    atol: float,
) -> FieldReport:
    report = FieldReport(
        name=name,
        status="PASS",
    )
    if pt.values.size != ct.values.size:
        report.status = "FAIL"
        report.detail = (
            f"Element-count mismatch: pytorch={pt.values.size} cute={ct.values.size}."
        )
        return report
    diff = np.abs(pt.values - ct.values)
    report.max_abs_diff = float(diff.max()) if diff.size else 0.0
    ok = np.allclose(pt.values, ct.values, rtol=rtol, atol=atol)
    if ok:
        return report
    report.status = "FAIL"
    mismatch_idx = np.argsort(-diff)[:5]
    preview = ", ".join(
        f"i={int(i)}: pt={pt.values[i]:.6g} cute={ct.values[i]:.6g}"
        for i in mismatch_idx
    )
    report.detail = f"Top diffs: {preview}"
    return report


def compare_tagged_outputs(
    transcript: str,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> ComparisonReport:
    """Parse a transcript and compare tagged PyTorch vs CuTe numeric CSV blocks.

    `[PyTorch Host]` and `[CuTe Host]` blocks are emitted as inspection-only
    `INFO` rows carrying the raw `print(obj)` text; they do not gate
    `overall_pass`.
    """
    (
        pytorch_blocks,
        cute_blocks,
        pytorch_host_blocks,
        cute_host_blocks,
        block_errors,
    ) = _extract_blocks(transcript)
    all_names = sorted(set(pytorch_blocks) | set(cute_blocks))
    field_reports: list[FieldReport] = []

    for err in block_errors:
        field_reports.append(
            FieldReport(name="<transcript>", status="PARSE_ERROR", detail=err)
        )

    for name in sorted(pytorch_host_blocks):
        body = "\n".join(pytorch_host_blocks[name].lines).strip()
        detail = f"Source: PyTorch Host\n{body}" if body else "Source: PyTorch Host"
        field_reports.append(
            FieldReport(name=name, status="INFO", detail=detail)
        )

    for name in sorted(cute_host_blocks):
        body = "\n".join(cute_host_blocks[name].lines).strip()
        detail = f"Source: CuTe Host\n{body}" if body else "Source: CuTe Host"
        field_reports.append(
            FieldReport(name=name, status="INFO", detail=detail)
        )

    for name in all_names:
        if name not in pytorch_blocks:
            field_reports.append(
                FieldReport(
                    name=name,
                    status="MISSING_PYTORCH",
                    detail="No [PyTorch Val] block emitted for this field.",
                )
            )
            continue
        if name not in cute_blocks:
            field_reports.append(
                FieldReport(
                    name=name,
                    status="MISSING_CUTE",
                    detail="No [CuTe Val] block emitted for this field.",
                )
            )
            continue
        try:
            pt = _parse_csv_numeric_values(
                pytorch_blocks[name],
                source_label="PyTorch",
            )
        except ParseError as exc:
            field_reports.append(
                FieldReport(name=name, status="PARSE_ERROR", detail=f"PyTorch: {exc}")
            )
            continue
        try:
            ct = _parse_csv_numeric_values(
                cute_blocks[name],
                source_label="CuTe",
            )
        except ParseError as exc:
            field_reports.append(
                FieldReport(name=name, status="PARSE_ERROR", detail=f"CuTe: {exc}")
            )
            continue
        field_reports.append(_compare_field(name, pt, ct, rtol=rtol, atol=atol))

    gating_reports = [fr for fr in field_reports if fr.status != "INFO"]
    overall = bool(field_reports) and all(
        fr.status == "PASS" for fr in gating_reports
    )
    return ComparisonReport(fields=field_reports, overall_pass=overall)


def main(argv: Iterable[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Compare CSV bodies inside [PyTorch Val] / [CuTe Val] tagged blocks."
    )
    parser.add_argument(
        "transcript",
        type=Path,
        help="Path to the transcript file (e.g., last_shell_dump.txt).",
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    text = args.transcript.read_text()
    report = compare_tagged_outputs(text, rtol=args.rtol, atol=args.atol)
    print(report.format_summary())
    return 0 if report.overall_pass else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(main(sys.argv[1:]))
