"""Parser for stage-validation transcripts emitted via runtime-tagged prints.

The stage validation protocol relies on tagged print streams inside the
synthetic bench stdout:

    [PyTorch Val] <field>: BEGIN
    shape=(d0,d1,...) dtype=<torch-dtype> scope=<gmem|rmem|smem>
    data=[v0, v1, v2, ..., vN-1]
    [PyTorch Val] <field>: END

    [CuTe Val] <field>: BEGIN
    <cute.print_tensor(...) output -- multi-line>
    [CuTe Val] <field>: END

    [CuTe Host] <name>: BEGIN
    <python print(obj) output for a host-side CuTe object>
    [CuTe Host] <name>: END

`[CuTe Host]` blocks are inspection-only: they surface static JIT-trace-time
metadata (TiledMma, SMEM layouts, TMA atoms, SharedStorage structs, etc.).
No PyTorch counterpart is expected; the parser reports them as `INFO` rows
without gating `overall_pass`.

Single-part-per-field: duplicate BEGIN for the same (source, field) is a parse
error. Fields present on only one side (PyTorch Val / CuTe Val) are reported
as MISSING.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

MAX_ELEMENTS = 1 << 20

_PYTORCH_BEGIN = re.compile(r"^\s*\[PyTorch Val\]\s+(?P<name>.+?)\s*:\s*BEGIN\s*$")
_PYTORCH_END = re.compile(r"^\s*\[PyTorch Val\]\s+(?P<name>.+?)\s*:\s*END\s*$")
_CUTE_BEGIN = re.compile(r"^\s*\[CuTe Val\]\s+(?P<name>.+?)\s*:\s*BEGIN\s*$")
_CUTE_END = re.compile(r"^\s*\[CuTe Val\]\s+(?P<name>.+?)\s*:\s*END\s*$")
_CUTE_HOST_BEGIN = re.compile(r"^\s*\[CuTe Host\]\s+(?P<name>.+?)\s*:\s*BEGIN\s*$")
_CUTE_HOST_END = re.compile(r"^\s*\[CuTe Host\]\s+(?P<name>.+?)\s*:\s*END\s*$")

_PYTORCH_HEADER = re.compile(
    r"shape\s*=\s*\((?P<shape>[^)]*)\)"
    r".*?dtype\s*=\s*(?P<dtype>\S+)"
    r".*?scope\s*=\s*(?P<scope>\S+)",
    re.DOTALL,
)
_PYTORCH_DATA = re.compile(r"data\s*=\s*(?P<body>\[.*\])", re.DOTALL)

_CUTE_RAW_PTR = re.compile(
    r"raw_ptr\([^:]*:\s*(?P<dtype>[^,\)]+)(?:,\s*[^)]*)?\)"
)
_CUTE_LAYOUT = re.compile(r"\bo\s*\((?P<shape>[^)]*)\)\s*:\s*\((?P<stride>[^)]*)\)")
_CUTE_DATA_MARKER = re.compile(r"data\s*=")
_FLOAT_LITERAL = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")


@dataclass
class RawBlock:
    name: str
    lines: list[str] = field(default_factory=list)


@dataclass
class ParsedPyTorch:
    name: str
    shape: tuple[int, ...]
    dtype: str
    scope: str
    values: np.ndarray


@dataclass
class ParsedCuTe:
    name: str
    shape: tuple[int, ...] | None
    dtype: str | None
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
) -> tuple[dict[str, RawBlock], dict[str, RawBlock], dict[str, RawBlock], list[str]]:
    pytorch: dict[str, RawBlock] = {}
    cute: dict[str, RawBlock] = {}
    host: dict[str, RawBlock] = {}
    errors: list[str] = []

    current: RawBlock | None = None
    current_source: str | None = None  # "pytorch" | "cute" | "host"

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
        m = _CUTE_HOST_BEGIN.match(line)
        if m:
            if current is not None:
                errors.append(
                    f"Unclosed {current_source} block for {current.name!r} before new BEGIN."
                )
            current_source = "host"
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
            if name in pytorch:
                errors.append(f"Duplicate PyTorch block for {name!r}.")
            else:
                pytorch[name] = current
            current = None
            current_source = None
            continue
        m = _CUTE_END.match(line)
        if m:
            name = m.group("name").strip()
            if current is None or current_source != "cute" or current.name != name:
                errors.append(f"Unexpected CuTe END for {name!r}.")
                current = None
                current_source = None
                continue
            if name in cute:
                errors.append(f"Duplicate CuTe block for {name!r}.")
            else:
                cute[name] = current
            current = None
            current_source = None
            continue
        m = _CUTE_HOST_END.match(line)
        if m:
            name = m.group("name").strip()
            if current is None or current_source != "host" or current.name != name:
                errors.append(f"Unexpected CuTe Host END for {name!r}.")
                current = None
                current_source = None
                continue
            if name in host:
                errors.append(f"Duplicate CuTe Host block for {name!r}.")
            else:
                host[name] = current
            current = None
            current_source = None
            continue
        if current is not None:
            current.lines.append(line)

    if current is not None:
        errors.append(
            f"Transcript ended mid-block: {current_source} {current.name!r}."
        )

    return pytorch, cute, host, errors


def _parse_shape(text: str) -> tuple[int, ...]:
    pieces = [p.strip() for p in text.split(",") if p.strip()]
    return tuple(int(p) for p in pieces)


def _parse_pytorch(block: RawBlock) -> ParsedPyTorch:
    body = "\n".join(block.lines)
    hdr = _PYTORCH_HEADER.search(body)
    if hdr is None:
        raise ParseError(
            "Missing 'shape=..., dtype=..., scope=...' header in PyTorch block."
        )
    data = _PYTORCH_DATA.search(body)
    if data is None:
        raise ParseError("Missing 'data=[...]' in PyTorch block.")
    try:
        values = np.asarray(ast.literal_eval(data.group("body")), dtype=np.float64)
    except (ValueError, SyntaxError) as exc:
        raise ParseError(f"Could not parse PyTorch data list: {exc}") from exc
    if values.size > MAX_ELEMENTS:
        raise ParseError(
            f"PyTorch block has {values.size} elements; exceeds cap {MAX_ELEMENTS}."
        )
    return ParsedPyTorch(
        name=block.name,
        shape=_parse_shape(hdr.group("shape")),
        dtype=hdr.group("dtype"),
        scope=hdr.group("scope"),
        values=values.flatten(),
    )


def _parse_cute(block: RawBlock) -> ParsedCuTe:
    body = "\n".join(block.lines)
    raw_ptr_match = _CUTE_RAW_PTR.search(body)
    dtype = raw_ptr_match.group("dtype").strip() if raw_ptr_match else None
    layout_match = _CUTE_LAYOUT.search(body)
    shape: tuple[int, ...] | None = None
    if layout_match:
        try:
            shape = _parse_shape(layout_match.group("shape"))
        except ValueError:
            shape = None

    data_marker = _CUTE_DATA_MARKER.search(body)
    if data_marker is None:
        raise ParseError("Missing 'data=' marker in CuTe block.")
    tail = body[data_marker.end():]
    floats = _FLOAT_LITERAL.findall(tail)
    if not floats:
        raise ParseError("No numeric data found in CuTe block.")
    try:
        values = np.asarray([float(x) for x in floats], dtype=np.float64)
    except ValueError as exc:
        raise ParseError(f"Could not parse CuTe data: {exc}") from exc
    if values.size > MAX_ELEMENTS:
        raise ParseError(
            f"CuTe block has {values.size} elements; exceeds cap {MAX_ELEMENTS}."
        )
    return ParsedCuTe(name=block.name, shape=shape, dtype=dtype, values=values)


def _compare_field(
    name: str,
    pt: ParsedPyTorch,
    ct: ParsedCuTe,
    rtol: float,
    atol: float,
) -> FieldReport:
    report = FieldReport(
        name=name,
        status="PASS",
        pytorch_shape=pt.shape,
        cute_shape=ct.shape,
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
    """Parse a transcript and compare tagged PyTorch vs CuTe blocks per field.

    `[CuTe Host]` blocks are emitted as inspection-only `INFO` rows carrying the
    raw `print(obj)` text; they do not gate `overall_pass`.
    """
    pytorch_blocks, cute_blocks, host_blocks, block_errors = _extract_blocks(transcript)
    all_names = sorted(set(pytorch_blocks) | set(cute_blocks))
    field_reports: list[FieldReport] = []

    for err in block_errors:
        field_reports.append(
            FieldReport(name="<transcript>", status="PARSE_ERROR", detail=err)
        )

    for name in sorted(host_blocks):
        body = "\n".join(host_blocks[name].lines).strip()
        field_reports.append(
            FieldReport(name=name, status="INFO", detail=body)
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
            pt = _parse_pytorch(pytorch_blocks[name])
        except ParseError as exc:
            field_reports.append(
                FieldReport(name=name, status="PARSE_ERROR", detail=f"PyTorch: {exc}")
            )
            continue
        try:
            ct = _parse_cute(cute_blocks[name])
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
        description="Compare [PyTorch Val] / [CuTe Val] tagged blocks in a transcript."
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
