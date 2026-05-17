"""Post-hoc aggregator for ToolCallAnalyzer JSONL files.

Usage:
    .venv/bin/python scripts/analyze_tool_calls.py notes/tool_calls/<run_id>.jsonl
    .venv/bin/python scripts/analyze_tool_calls.py notes/tool_calls/*.jsonl --top-k 15

Reads one or more JSONL files produced by ``kernel_agents.tool_call_analyzer.ToolCallAnalyzer``
and prints aggregated metrics: per-round x per-tool tables, per-agent histograms, top-K
argument values, latency percentiles, token usage per round, output-size distributions,
and (tool_k, tool_{k+1}) bigrams per agent.

Stdlib only; no extra deps.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Rough USD / 1M tokens, input + output, for pricing estimates. Sourced from
# publicly posted OpenAI Platform pricing and updated as needed. Unknown models
# fall back to zero cost and are flagged in the report footer.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o4-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
}


@dataclass
class Row:
    raw: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def tool_source(self) -> str:
        return self.raw.get("tool_source") or ""

    @property
    def tool_name(self) -> str:
        return self.raw.get("tool_name") or ""

    @property
    def agent_name(self) -> str:
        return self.raw.get("agent_name") or "(unknown)"

    @property
    def agent_role(self) -> str:
        return self.raw.get("agent_role") or "(unknown)"

    @property
    def round(self) -> Any:
        return self.raw.get("round")

    @property
    def stage_label(self) -> str:
        return self.raw.get("stage_label") or "(unknown)"

    @property
    def duration_ms(self) -> float | None:
        v = self.raw.get("duration_ms")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def started_at(self) -> str | None:
        return self.raw.get("started_at")

    @property
    def output_chars(self) -> int:
        v = self.raw.get("output_chars")
        return int(v) if isinstance(v, (int, float)) else 0


def _load_rows(paths: Iterable[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"warn: {path}:{lineno} skipped (invalid JSON: {exc})",
                        file=sys.stderr,
                    )
                    continue
                rows.append(Row(obj))
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sv = sorted(values)
    k = (len(sv) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sv[int(k)])
    return float(sv[lo] + (sv[hi] - sv[lo]) * (k - lo))


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "-"
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print()
    print(f"== {title} ==")
    if not rows:
        print("  (no data)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def _round_key(r: Any) -> str:
    if r is None:
        return "-"
    return str(r)


# --------------------------------------------------------------- tool-call rows
# A "tool-call row" is any row with tool_source in {function, hosted}. We treat
# llm_turn/agent/handoff rows separately so per-tool counts aren't skewed by
# non-tool spans.

_TOOL_SOURCES = {"function", "hosted"}


def _tool_rows(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.tool_source in _TOOL_SOURCES]


# ------------------------------------------------------------- report sections


def report_per_round_per_tool(rows: list[Row]) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for r in _tool_rows(rows):
        key = (_round_key(r.round), r.tool_name)
        counts[key] += 1
        if r.duration_ms is not None:
            totals[key] += r.duration_ms
    rounds = sorted({k[0] for k in counts}, key=lambda v: (v == "-", v))
    tools = sorted({k[1] for k in counts})
    table_rows = []
    for rnd in rounds:
        for tool in tools:
            n = counts.get((rnd, tool), 0)
            if n == 0:
                continue
            total = totals.get((rnd, tool), 0.0)
            table_rows.append(
                [rnd, tool, _fmt_int(n), _fmt_ms(total), _fmt_ms(total / n if n else None)]
            )
    _print_table(
        "Per-round x per-tool counts + duration",
        ["round", "tool", "count", "total", "avg"],
        table_rows,
    )


def report_per_agent_histogram(rows: list[Row]) -> None:
    per_agent: dict[str, Counter[str]] = defaultdict(Counter)
    for r in _tool_rows(rows):
        per_agent[r.agent_name][r.tool_name] += 1
    table_rows = []
    for agent in sorted(per_agent):
        total = sum(per_agent[agent].values())
        top = per_agent[agent].most_common()
        summary = ", ".join(f"{name}={cnt}" for name, cnt in top[:8])
        if len(top) > 8:
            summary += f", ... (+{len(top) - 8} more)"
        table_rows.append([agent, _fmt_int(total), summary])
    _print_table(
        "Per-agent tool histogram",
        ["agent", "total", "tools"],
        table_rows,
    )


def report_top_args(rows: list[Row], top_k: int) -> None:
    per_tool: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for r in _tool_rows(rows):
        digest = r.raw.get("args_digest") or {}
        if not isinstance(digest, dict):
            continue
        for k, v in digest.items():
            if v is None:
                continue
            val_str = str(v)
            if len(val_str) > 120:
                val_str = val_str[:117] + "..."
            per_tool[r.tool_name][(k, val_str)] += 1
    print()
    print(f"== Top-{top_k} argument values per tool ==")
    any_printed = False
    for tool in sorted(per_tool):
        top = per_tool[tool].most_common(top_k)
        if not top:
            continue
        any_printed = True
        print(f"  [{tool}]")
        for (k, v), cnt in top:
            print(f"    {cnt:>4}  {k}={v}")
    if not any_printed:
        print("  (no data)")


def report_latency(rows: list[Row]) -> None:
    by_tool: dict[str, list[float]] = defaultdict(list)
    for r in _tool_rows(rows):
        if r.duration_ms is not None:
            by_tool[r.tool_name].append(r.duration_ms)
    pairs: list[tuple[float, list[str]]] = []
    for tool in sorted(by_tool):
        values = by_tool[tool]
        total = sum(values)
        pairs.append(
            (
                total,
                [
                    tool,
                    _fmt_int(len(values)),
                    _fmt_ms(_percentile(values, 50)),
                    _fmt_ms(_percentile(values, 90)),
                    _fmt_ms(max(values)),
                    _fmt_ms(total),
                ],
            )
        )
    pairs.sort(key=lambda p: -p[0])
    _print_table(
        "Tool latency (p50/p90/max)",
        ["tool", "n", "p50", "p90", "max", "total"],
        [p[1] for p in pairs],
    )


def _estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    if not model:
        return 0.0
    base = model
    for candidate in _PRICE_PER_MTOK:
        if base.startswith(candidate):
            pin, pout = _PRICE_PER_MTOK[candidate]
            return (input_tokens * pin + output_tokens * pout) / 1_000_000.0
    return 0.0


def report_tokens(rows: list[Row]) -> None:
    per_round: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"turns": 0, "input": 0, "output": 0, "cached": 0, "cost": 0.0, "models": Counter()}
    )
    unknown_models: set[str] = set()
    for r in rows:
        if r.tool_source != "llm_turn":
            continue
        usage = r.raw.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        key = _round_key(r.round)
        input_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        cached = 0
        details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens") or 0)
        model = r.raw.get("model")
        per_round[key]["turns"] += 1
        per_round[key]["input"] += input_tok
        per_round[key]["output"] += output_tok
        per_round[key]["cached"] += cached
        per_round[key]["cost"] += _estimate_cost(model, input_tok, output_tok)
        if model:
            per_round[key]["models"][model] += 1
            if _estimate_cost(model, 1_000_000, 0) == 0.0:
                unknown_models.add(model)
    table_rows = []
    for key in sorted(per_round, key=lambda v: (v == "-", v)):
        d = per_round[key]
        top_model = d["models"].most_common(1)[0][0] if d["models"] else "-"
        table_rows.append(
            [
                key,
                _fmt_int(d["turns"]),
                _fmt_int(d["input"]),
                _fmt_int(d["output"]),
                _fmt_int(d["cached"]),
                f"${d['cost']:.4f}",
                top_model,
            ]
        )
    _print_table(
        "Per-round token usage + estimated cost",
        ["round", "turns", "input", "output", "cached", "est_cost", "primary_model"],
        table_rows,
    )
    if unknown_models:
        print(
            "  note: no price entry for models: "
            + ", ".join(sorted(unknown_models))
            + " (cost shown as $0)"
        )


def report_output_sizes(rows: list[Row]) -> None:
    by_tool: dict[str, list[int]] = defaultdict(list)
    for r in _tool_rows(rows):
        if r.tool_source != "function":
            continue
        by_tool[r.tool_name].append(r.output_chars)
    pairs: list[tuple[int, list[str]]] = []
    for tool in sorted(by_tool):
        vals = by_tool[tool]
        if not vals:
            continue
        floats = [float(v) for v in vals]
        total = sum(vals)
        pairs.append(
            (
                total,
                [
                    tool,
                    _fmt_int(len(vals)),
                    _fmt_int(int(_percentile(floats, 50))),
                    _fmt_int(int(_percentile(floats, 90))),
                    _fmt_int(max(vals)),
                    _fmt_int(total),
                ],
            )
        )
    pairs.sort(key=lambda p: -p[0])
    _print_table(
        "Function tool output size (chars)",
        ["tool", "n", "p50", "p90", "max", "total"],
        [p[1] for p in pairs],
    )


def report_bigrams(rows: list[Row], top_k: int) -> None:
    tool_rows = _tool_rows(rows)
    tool_rows.sort(key=lambda r: (r.agent_name, r.started_at or ""))
    per_agent_seq: dict[str, list[str]] = defaultdict(list)
    for r in tool_rows:
        per_agent_seq[r.agent_name].append(r.tool_name)
    print()
    print(f"== Top-{top_k} tool bigrams per agent ==")
    any_printed = False
    for agent in sorted(per_agent_seq):
        seq = per_agent_seq[agent]
        if len(seq) < 2:
            continue
        bigrams = Counter(zip(seq, seq[1:]))
        top = bigrams.most_common(top_k)
        if not top:
            continue
        any_printed = True
        print(f"  [{agent}] ({len(seq)} calls)")
        for (a, b), cnt in top:
            print(f"    {cnt:>4}  {a} -> {b}")
    if not any_printed:
        print("  (no data)")


def report_summary(rows: list[Row]) -> None:
    tool_rows = _tool_rows(rows)
    llm_turns = sum(1 for r in rows if r.tool_source == "llm_turn")
    agents = {r.agent_name for r in tool_rows}
    rounds = {_round_key(r.round) for r in tool_rows}
    errors = sum(1 for r in tool_rows if (r.raw.get("status") or "").lower() == "error")
    total_dur = sum(r.duration_ms or 0.0 for r in tool_rows)
    print("== Summary ==")
    print(f"  rows                : {_fmt_int(len(rows))}")
    print(f"  tool calls          : {_fmt_int(len(tool_rows))} (errors: {errors})")
    print(f"  llm turns           : {_fmt_int(llm_turns)}")
    print(f"  distinct agents     : {len(agents)}")
    print(f"  distinct rounds     : {len(rounds)} ({', '.join(sorted(rounds, key=lambda v: (v == '-', v)))})")
    print(f"  total tool duration : {_fmt_ms(total_dur)}")


def _expand_paths(raw_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_paths:
        matches = sorted(glob.glob(raw))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(raw))
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: {p} does not exist", file=sys.stderr)
        sys.exit(2)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSONL files (glob patterns supported).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many entries to show for top-K lists (default: 10).",
    )
    parser.add_argument(
        "--only",
        choices=["summary", "rounds", "agents", "args", "latency", "tokens", "sizes", "bigrams"],
        action="append",
        help="Restrict output to selected sections (repeatable). Default: all.",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.paths)
    rows = _load_rows(paths)
    if not rows:
        print("No rows loaded.", file=sys.stderr)
        sys.exit(1)

    sections = set(args.only) if args.only else {
        "summary", "rounds", "agents", "args", "latency", "tokens", "sizes", "bigrams"
    }

    print(f"Loaded {_fmt_int(len(rows))} rows from {len(paths)} file(s):")
    for p in paths:
        print(f"  - {p}")
    print()

    if "summary" in sections:
        report_summary(rows)
    if "rounds" in sections:
        report_per_round_per_tool(rows)
    if "agents" in sections:
        report_per_agent_histogram(rows)
    if "args" in sections:
        report_top_args(rows, args.top_k)
    if "latency" in sections:
        report_latency(rows)
    if "tokens" in sections:
        report_tokens(rows)
    if "sizes" in sections:
        report_output_sizes(rows)
    if "bigrams" in sections:
        report_bigrams(rows, args.top_k)


if __name__ == "__main__":
    main()
