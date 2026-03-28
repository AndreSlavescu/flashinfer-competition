#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUMP_SASS = ROOT / "tools" / "sass" / "dump_sass_modal.py"


@dataclass(frozen=True)
class EntryRule:
    table: str
    measurement_id: str | None = None
    proxy_measurement_id: str | None = None
    kernel_fragments: tuple[str, ...] = ()
    evidence_prefixes: tuple[str, ...] = ()
    proxy_evidence_prefixes: tuple[str, ...] = ()
    direct_note: str = ""
    proxy_note: str = ""
    force_status: str | None = None


ENTRY_RULES: dict[str, EntryRule] = {
    "FFMA": EntryRule("SM100A_SYNC_LATENCIES", "sync_ffma", kernel_fragments=("sync_ffma_kernel",), evidence_prefixes=("FFMA",)),
    "FADD": EntryRule("SM100A_SYNC_LATENCIES", "sync_fadd", kernel_fragments=("sync_fadd_kernel",), evidence_prefixes=("FADD",)),
    "FMUL": EntryRule("SM100A_SYNC_LATENCIES", "sync_fmul", kernel_fragments=("sync_fmul_kernel",), evidence_prefixes=("FMUL",)),
    "HFMA2": EntryRule("SM100A_SYNC_LATENCIES", "sync_hfma2", kernel_fragments=("sync_hfma2_kernel",), evidence_prefixes=("HFMA2",)),
    "MUFU": EntryRule("SM100A_SYNC_LATENCIES", "sync_mufu", kernel_fragments=("sync_mufu_kernel",), evidence_prefixes=("MUFU",)),
    "IMAD": EntryRule("SM100A_SYNC_LATENCIES", "sync_imad", kernel_fragments=("sync_imad_kernel",), evidence_prefixes=("IMAD",)),
    "IADD3": EntryRule("SM100A_SYNC_LATENCIES", "sync_iadd3", kernel_fragments=("sync_iadd3_kernel",), evidence_prefixes=("IADD3",)),
    "LOP3": EntryRule("SM100A_SYNC_LATENCIES", "sync_lop3", kernel_fragments=("sync_lop3_kernel",), evidence_prefixes=("LOP3",)),
    "LDS": EntryRule("SM100A_SYNC_LATENCIES", "sync_lds", kernel_fragments=("sync_lds_kernel",), evidence_prefixes=("LDS",)),
    "STS": EntryRule(
        "SM100A_SYNC_LATENCIES",
        "sync_sts",
        kernel_fragments=("sync_sts_proxy_kernel",),
        evidence_prefixes=("STS",),
        proxy_note="proxied via dependent STS+LDS chain",
        force_status="proxy",
    ),
    "MOV": EntryRule("SM100A_SYNC_LATENCIES", "sync_mov", kernel_fragments=("sync_mov_kernel",), evidence_prefixes=("MOV",)),
    "SHFL": EntryRule("SM100A_SYNC_LATENCIES", "sync_shfl", kernel_fragments=("sync_shfl_kernel",), evidence_prefixes=("SHFL",)),
    "S2R": EntryRule("SM100A_SYNC_LATENCIES", "sync_s2r", kernel_fragments=("sync_s2r_kernel",), evidence_prefixes=("S2R",)),
    "UTCHMMA": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_utchmma", kernel_fragments=("async_utchmma_kernel",), evidence_prefixes=("UTCHMMA",)),
    "UTCIMMA": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        proxy_measurement_id="async_utchmma",
        kernel_fragments=("async_utchmma_kernel",),
        proxy_evidence_prefixes=("UTCHMMA",),
        proxy_note="proxied with f16 tcgen05.mma issue kernel",
    ),
    "UTCOMMA": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        proxy_measurement_id="async_utchmma",
        kernel_fragments=("async_utchmma_kernel",),
        proxy_evidence_prefixes=("UTCHMMA",),
        proxy_note="proxied with f16 tcgen05.mma issue kernel",
    ),
    "UTCQMMA": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        proxy_measurement_id="async_utchmma",
        kernel_fragments=("async_utchmma_kernel",),
        proxy_evidence_prefixes=("UTCHMMA",),
        proxy_note="proxied with f16 tcgen05.mma issue kernel",
    ),
    "UTMALDG": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_tma_load", kernel_fragments=("async_tma_load_kernel",), evidence_prefixes=("UTMALDG",)),
    "UTMASTG": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_tma_store", kernel_fragments=("async_tma_store_kernel",), evidence_prefixes=("UTMASTG",)),
    "UTMAREDG": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_tma_reduce", kernel_fragments=("async_tma_reduce_kernel",), evidence_prefixes=("UTMAREDG",)),
    "UTMAPF": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_tma_prefetch", kernel_fragments=("async_tma_prefetch_kernel",), evidence_prefixes=("UTMAPF",)),
    "UTMACCTL": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        proxy_measurement_id="async_tma_prefetch",
        kernel_fragments=("async_tma_prefetch_kernel", "async_cctl_prefetch_kernel"),
        evidence_prefixes=("UTMACCTL",),
        proxy_evidence_prefixes=("UTMAPF", "CCTL", "CCTLL"),
        proxy_note="proxied with TMA prefetch/cache-control path",
    ),
    "UTCCP": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_utccp", kernel_fragments=("async_utccp_kernel",), evidence_prefixes=("UTCCP",)),
    "LDT": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_ldt", kernel_fragments=("async_ldt_kernel",), evidence_prefixes=("LDT",)),
    "LDTM": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        measurement_id="async_ldtm",
        kernel_fragments=("async_ldtm_kernel",),
        evidence_prefixes=("LDTM",),
        proxy_evidence_prefixes=("LDT",),
        proxy_note="ptxas lowered TMEM matrix load to scalar TMEM load form",
    ),
    "STT": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_stt", kernel_fragments=("async_stt_kernel",), evidence_prefixes=("STT",)),
    "STTM": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        measurement_id="async_sttm",
        kernel_fragments=("async_sttm_kernel",),
        evidence_prefixes=("STTM",),
        proxy_evidence_prefixes=("STT",),
        proxy_note="ptxas lowered TMEM matrix store to scalar TMEM store form",
    ),
    "UTCSHIFT": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        proxy_measurement_id="async_ldtm",
        kernel_fragments=("async_ldtm_kernel",),
        evidence_prefixes=("UTCSHIFT",),
        proxy_evidence_prefixes=("LDTM", "LDT"),
        proxy_note="proxied with TMEM rearrangement/load path",
    ),
    "LDGDEPBAR": EntryRule("SM100A_ASYNC_ISSUE_COSTS"),
    "LDGSTS": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_ldgsts", kernel_fragments=("async_ldgsts_kernel",), evidence_prefixes=("LDGSTS",)),
    "LDGMC": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        proxy_measurement_id="async_ldg",
        kernel_fragments=("async_ldg_kernel", "async_cctl_prefetch_kernel"),
        evidence_prefixes=("LDGMC",),
        proxy_evidence_prefixes=("LDG", "CCTL", "CCTLL"),
        proxy_note="proxied with global load/cache-control path",
    ),
    "CCTL": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        measurement_id="async_cctl",
        kernel_fragments=("async_cctl_prefetch_kernel",),
        evidence_prefixes=("CCTL",),
        proxy_evidence_prefixes=("CCTLL",),
        proxy_note="ptxas selected long-form cache-control encoding",
    ),
    "CCTLL": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        measurement_id="async_cctl",
        kernel_fragments=("async_cctl_prefetch_kernel",),
        evidence_prefixes=("CCTLL",),
        proxy_evidence_prefixes=("CCTL",),
        proxy_note="ptxas selected short-form cache-control encoding",
    ),
    "UBLKCP": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_bulk_copy", kernel_fragments=("async_bulk_copy_kernel",), evidence_prefixes=("UBLKCP",)),
    "UBLKPF": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_bulk_prefetch", kernel_fragments=("async_bulk_prefetch_kernel",), evidence_prefixes=("UBLKPF",)),
    "UBLKRED": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_bulk_reduce", kernel_fragments=("async_bulk_reduce_kernel",), evidence_prefixes=("UBLKRED",)),
    "LDG": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_ldg", kernel_fragments=("async_ldg_kernel",), evidence_prefixes=("LDG",)),
    "STG": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_stg", kernel_fragments=("async_stg_kernel",), evidence_prefixes=("STG",)),
    "LDL": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_ldl", kernel_fragments=("async_ldl_kernel",), evidence_prefixes=("LDL",)),
    "STL": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_stl", kernel_fragments=("async_stl_kernel",), evidence_prefixes=("STL",)),
    "STAS": EntryRule("SM100A_ASYNC_ISSUE_COSTS"),
    "REDAS": EntryRule("SM100A_ASYNC_ISSUE_COSTS"),
    "MBAR": EntryRule(
        "SM100A_ASYNC_ISSUE_COSTS",
        "async_mbar",
        kernel_fragments=("async_mbar_kernel",),
        evidence_prefixes=("MBAR", "SYNCS.ARRIVE"),
        direct_note="cuobjdump lowers mbarrier arrive to SYNCS.ARRIVE",
    ),
    "BAR": EntryRule("SM100A_ASYNC_ISSUE_COSTS", "async_bar", kernel_fragments=("async_bar_kernel",), evidence_prefixes=("BAR",)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sm100a opcode cost measurements and optionally update dump_sass_modal.py.")
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--disassembly", required=True, type=Path)
    parser.add_argument("--dump-sass", default=DEFAULT_DUMP_SASS, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--tolerance", default=1, type=int)
    parser.add_argument(
        "--update-policy",
        choices=("confirmed", "measured"),
        default="confirmed",
        help="`confirmed` only patches rows within tolerance; `measured` patches any direct/proxy row with a B200 measurement.",
    )
    parser.add_argument("--no-update", action="store_true")
    return parser.parse_args()


def parse_dump_tables(path: Path) -> dict[str, dict[str, tuple[int, str]]]:
    source = path.read_text()
    module = ast.parse(source)
    tables: dict[str, dict[str, tuple[int, str]]] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in {"SM100A_SYNC_LATENCIES", "SM100A_ASYNC_ISSUE_COSTS"}:
            continue
        value = ast.literal_eval(node.value)
        tables[target.id] = value
    return tables


def parse_measurements(path: Path) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            measurement_id = row["measurement_id"]
            role = row["role"]
            batch = int(row["batch"])
            grouped.setdefault(measurement_id, {}).setdefault(role, {})[batch] = {
                "suite": row["suite"],
                "outer_iters": int(row["outer_iters"]),
                "ops_per_iter": int(row["ops_per_iter"]),
                "runs": int(row["runs"]),
                "total_cycles": float(row["total_cycles"]),
                "total_ns": float(row["total_ns"]),
                "cycles_per_op": float(row["cycles_per_op"]),
                "ns_per_op": float(row["ns_per_op"]),
                "spread_pct": float(row["spread_pct"]),
            }
    return grouped


def split_disassembly_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    current_name = "global"
    blocks[current_name] = []
    func_patterns = [
        re.compile(r"^\s*Function\s*:\s*(.+?)\s*$"),
        re.compile(r"^\s*code for .*?\s+(.+?)\s*$"),
        re.compile(r"^\s*\.text\.(\S+)\s*$"),
    ]
    for line in text.splitlines():
        next_name = None
        for pattern in func_patterns:
            match = pattern.match(line)
            if match:
                next_name = match.group(1).strip()
                break
        if next_name is not None:
            current_name = next_name
            blocks.setdefault(current_name, [])
            continue
        blocks.setdefault(current_name, []).append(line)
    return {name: "\n".join(lines) for name, lines in blocks.items()}


def find_observed_opcode(rule: EntryRule, blocks: dict[str, str]) -> tuple[str | None, str | None]:
    if not rule.kernel_fragments:
        return None, None
    candidate_texts: list[str] = []
    for name, block in blocks.items():
        if any(fragment in name or fragment in block for fragment in rule.kernel_fragments):
            candidate_texts.append(block)
    if not candidate_texts:
        candidate_texts = [text for text in blocks.values() if any(fragment in text for fragment in rule.kernel_fragments)]
    merged = "\n".join(candidate_texts)
    for prefix in rule.evidence_prefixes:
        if re.search(rf"\b{re.escape(prefix)}(?:[\w.]*)\b", merged):
            return prefix, "direct"
    for prefix in rule.proxy_evidence_prefixes:
        if re.search(rf"\b{re.escape(prefix)}(?:[\w.]*)\b", merged):
            return prefix, "proxy"
    return None, None


def fit_async_issue_cost(target_rows: dict[int, dict[str, Any]], baseline_rows: dict[int, dict[str, Any]] | None) -> tuple[int | None, str]:
    points: list[tuple[float, float]] = []
    for batch, target in sorted(target_rows.items()):
        baseline = baseline_rows.get(batch) if baseline_rows else None
        delta_cycles = target["total_cycles"] - (baseline["total_cycles"] if baseline else 0.0)
        y = delta_cycles / max(target["outer_iters"], 1)
        points.append((float(batch), float(y)))
    if not points:
        return None, "no points"
    if len(points) == 1:
        x, y = points[0]
        slope = y / x if x else 0.0
        return int(round(slope)), json.dumps({"points": points, "intercept": 0.0})
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    denom = sum((x - x_mean) ** 2 for x, _ in points)
    if denom == 0:
        slope = y_mean / x_mean if x_mean else 0.0
        intercept = 0.0
    else:
        slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denom
        intercept = y_mean - slope * x_mean
    return int(round(slope)), json.dumps({"points": points, "intercept": intercept, "slope": slope})


def compute_measured_cycles(rule: EntryRule, measurements: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> tuple[int | None, str, str | None]:
    measurement_id = rule.measurement_id
    proxy = False
    if measurement_id is None or measurement_id not in measurements:
        measurement_id = rule.proxy_measurement_id
        proxy = measurement_id is not None
    if measurement_id is None or measurement_id not in measurements:
        return None, "", measurement_id
    grouped = measurements[measurement_id]
    target_rows = grouped.get("target", {})
    baseline_rows = grouped.get("baseline", {})
    if not target_rows:
        return None, "", measurement_id
    suite = next(iter(target_rows.values()))["suite"]
    if suite == "sync":
        target = target_rows[min(target_rows)]
        baseline = baseline_rows.get(min(baseline_rows)) if baseline_rows else None
        delta = target["cycles_per_op"] - (baseline["cycles_per_op"] if baseline else 0.0)
        return int(round(delta)), json.dumps({"target": target["cycles_per_op"], "baseline": baseline["cycles_per_op"] if baseline else 0.0, "proxy": proxy}), measurement_id
    measured, fit_note = fit_async_issue_cost(target_rows, baseline_rows if baseline_rows else None)
    return measured, fit_note, measurement_id


def replace_line_updates(source: str, replacements: dict[str, dict[str, tuple[int, str]]]) -> str:
    lines = source.splitlines()
    current_table: str | None = None
    in_dict = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("SM100A_SYNC_LATENCIES = {"):
            current_table = "SM100A_SYNC_LATENCIES"
            in_dict = True
            continue
        if stripped.startswith("SM100A_ASYNC_ISSUE_COSTS = {"):
            current_table = "SM100A_ASYNC_ISSUE_COSTS"
            in_dict = True
            continue
        if in_dict and stripped == "}":
            current_table = None
            in_dict = False
            continue
        if not in_dict or current_table is None:
            continue
        match = re.match(r'^(\s*)"([^"]+)":\s*\(([^,]+),\s*"([^"]*)"\),?$', line)
        if not match:
            continue
        indent, opcode = match.group(1), match.group(2)
        replacement = replacements.get(current_table, {}).get(opcode)
        if replacement is None:
            continue
        cycles, desc = replacement
        escaped_desc = desc.replace("\\", "\\\\").replace('"', '\\"')
        lines[index] = f'{indent}"{opcode}":    ({cycles}, "{escaped_desc}"),'
    return "\n".join(lines) + "\n"


def combine_notes(*parts: str) -> str:
    items = [part for part in parts if part]
    return " | ".join(items)


def strip_b200_annotation(desc: str) -> str:
    return re.sub(r"; measured on B200 issue-isolation bench \([^)]*\)$", "", desc)


def main() -> None:
    args = parse_args()
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    tables = parse_dump_tables(args.dump_sass)
    measurements = parse_measurements(args.measurements)
    disassembly_text = args.disassembly.read_text()
    blocks = split_disassembly_blocks(disassembly_text)

    summary_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"entries": {}, "disassembly_blocks": sorted(blocks)}
    replacements: dict[str, dict[str, tuple[int, str]]] = {
        "SM100A_SYNC_LATENCIES": {},
        "SM100A_ASYNC_ISSUE_COSTS": {},
    }

    for table_name in ("SM100A_SYNC_LATENCIES", "SM100A_ASYNC_ISSUE_COSTS"):
        for opcode, (expected_cycles, expected_desc) in tables[table_name].items():
            rule = ENTRY_RULES.get(opcode, EntryRule(table=table_name))
            measured_cycles, fit_note, measurement_id = compute_measured_cycles(rule, measurements)
            observed_opcode, evidence_kind = find_observed_opcode(rule, blocks)
            status = "unverified"
            if observed_opcode is not None and evidence_kind is not None:
                status = rule.force_status or evidence_kind
            delta = measured_cycles - expected_cycles if measured_cycles is not None else None
            confirmed = measured_cycles is not None and status != "unverified" and abs(delta) <= args.tolerance
            status_note = ""
            if status == "direct":
                status_note = rule.direct_note
            elif status == "proxy":
                status_note = rule.proxy_note
            note = combine_notes(fit_note, status_note)
            summary_row = {
                "table": table_name,
                "opcode": opcode,
                "expected_cycles": expected_cycles,
                "measured_cycles": "" if measured_cycles is None else measured_cycles,
                "delta_cycles": "" if delta is None else delta,
                "confirmed": "yes" if confirmed else "no",
                "status": status,
                "measurement_id": measurement_id or "",
                "observed_opcode": observed_opcode or "",
                "evidence_kind": evidence_kind or "",
                "fit_note": note,
                "previous_desc": expected_desc,
            }
            summary_rows.append(summary_row)
            manifest["entries"][opcode] = summary_row
            updateable = measured_cycles is not None and status != "unverified"
            should_replace = confirmed if args.update_policy == "confirmed" else updateable
            if should_replace and measured_cycles is not None:
                base_desc = strip_b200_annotation(expected_desc)
                desc_suffix = f"measured on B200 issue-isolation bench ({status}"
                if status_note:
                    desc_suffix += f"; {status_note}"
                desc_suffix += ")"
                replacements[table_name][opcode] = (measured_cycles, f"{base_desc}; {desc_suffix}")

    with args.summary_out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "table",
                "opcode",
                "expected_cycles",
                "measured_cycles",
                "delta_cycles",
                "confirmed",
                "status",
                "measurement_id",
                "observed_opcode",
                "evidence_kind",
                "fit_note",
                "previous_desc",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    direct = [row for row in summary_rows if row["status"] == "direct"]
    proxy = [row for row in summary_rows if row["status"] == "proxy"]
    unverified = [row for row in summary_rows if row["status"] == "unverified"]
    mismatches = [row for row in summary_rows if row["confirmed"] == "no" and row["measured_cycles"] != ""]

    report_lines = [
        "# SM100A Opcode Cost Confirmation",
        "",
        f"- Total entries: {len(summary_rows)}",
        f"- Sync entries: {sum(row['table'] == 'SM100A_SYNC_LATENCIES' for row in summary_rows)}",
        f"- Async entries: {sum(row['table'] == 'SM100A_ASYNC_ISSUE_COSTS' for row in summary_rows)}",
        f"- Direct evidence rows: {len(direct)}",
        f"- Proxy rows: {len(proxy)}",
        f"- Unverified rows: {len(unverified)}",
        f"- Confirmed within +/-{args.tolerance} cycle(s): {sum(row['confirmed'] == 'yes' for row in summary_rows)}",
        "",
        "## Mismatches",
        "",
    ]
    if mismatches:
        for row in mismatches:
            report_lines.append(
                f"- `{row['opcode']}`: expected {row['expected_cycles']}, measured {row['measured_cycles']}, "
                f"delta {row['delta_cycles']} ({row['status']})"
            )
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Direct", ""])
    if direct:
        for row in direct:
            report_lines.append(
                f"- `{row['opcode']}`: measured {row['measured_cycles']} vs expected {row['expected_cycles']} "
                f"(observed `{row['observed_opcode']}` via `{row['measurement_id']}`)"
            )
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Proxy", ""])
    if proxy:
        for row in proxy:
            report_lines.append(
                f"- `{row['opcode']}`: measured {row['measured_cycles']} vs expected {row['expected_cycles']} "
                f"(observed `{row['observed_opcode']}` via `{row['measurement_id']}`; {row['fit_note']})"
            )
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Unverified", ""])
    if unverified:
        for row in unverified:
            report_lines.append(f"- `{row['opcode']}`: no qualifying measurement+evidence pair")
    else:
        report_lines.append("- None")
    args.report_out.write_text("\n".join(report_lines) + "\n")

    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not args.no_update:
        source = args.dump_sass.read_text()
        updated = replace_line_updates(source, replacements)
        args.dump_sass.write_text(updated)


if __name__ == "__main__":
    main()
