"""Tool call analyzer: a TracingProcessor that emits one JSONL row per tool call.

Attribution (round, stage, agent_role, quality_profile) is sourced from trace
metadata set by the loop orchestrator via ``agents.trace(workflow_name=...,
metadata={...})``. The agent name is resolved by walking each span's parent
chain up to the nearest cached ``AgentSpanData`` span.

The processor is additive: registered via ``agents.add_trace_processor(...)``,
it runs alongside the SDK's default OpenAI backend exporter without disturbing
it. JSONL is appended to ``<out_dir>/<run_id>.jsonl``, one row per tool call,
per hosted-tool item in a response, per LLM turn, and per agent/handoff span.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agents.tracing import (
    AgentSpanData,
    FunctionSpanData,
    HandoffSpanData,
    ResponseSpanData,
    Span,
    Trace,
    TracingProcessor,
)

from .stream_logging import _HINT_KEYS, normalize_preview

_HOSTED_TOOL_TYPES = frozenset(
    {
        "apply_patch_call",
        "local_shell_call",
        "shell_call",
        "web_search_call",
        "tool_search_call",
    }
)
_OUTPUT_PREVIEW_CHARS = 500


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def default_run_id() -> str:
    """Timestamp-based run id suitable for per-run JSONL filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _maybe_parse_args(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {"_raw": raw}
    if raw is None:
        return {}
    return {"_raw": str(raw)}


def _extract_args_digest(parsed_args: Any) -> dict[str, Any]:
    if not isinstance(parsed_args, dict):
        return {}
    return {key: parsed_args[key] for key in _HINT_KEYS if key in parsed_args}


def _duration_ms(span: Span[Any]) -> float | None:
    if not span.started_at or not span.ended_at:
        return None
    try:
        start = datetime.fromisoformat(span.started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(span.ended_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((end - start).total_seconds() * 1000.0, 3)


def _iter_hosted_tool_items(response: Any) -> Iterable[Any]:
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, dict):
            item_type = item.get("type")
        if item_type in _HOSTED_TOOL_TYPES:
            yield item


def _to_plain_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_unset=True)
        except TypeError:
            return dump()
    return {"_repr": repr(item)}


def _hosted_args(item_dict: dict[str, Any]) -> dict[str, Any]:
    skip = {"id", "call_id", "type", "status", "output"}
    return {k: v for k, v in item_dict.items() if k not in skip}


def _response_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_unset=True)
        except TypeError:
            return dump()
    if isinstance(usage, dict):
        return dict(usage)
    return None


@dataclass
class _TraceInfo:
    workflow_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolCallAnalyzer(TracingProcessor):
    """JSONL-emitting TracingProcessor for the kernel generation loop."""

    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        run_id: str | None = None,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id or default_run_id()
        self._out_path = self._out_dir / f"{self._run_id}.jsonl"
        self._file = self._out_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._traces: dict[str, _TraceInfo] = {}
        self._agent_by_span_id: dict[str, str] = {}
        self._parent_by_span_id: dict[str, str | None] = {}

    @property
    def output_path(self) -> Path:
        return self._out_path

    @property
    def run_id(self) -> str:
        return self._run_id

    # ------------------------------------------------------------------ I/O

    def _emit(self, row: dict[str, Any]) -> None:
        row.setdefault("ts", _utc_now_iso())
        row.setdefault("run_id", self._run_id)
        line = json.dumps(row, default=str, sort_keys=True)
        with self._lock:
            if self._file.closed:
                return
            self._file.write(line + "\n")
            self._file.flush()

    # -------------------------------------------------------- attribution

    def _attribution_for(self, span: Span[Any]) -> dict[str, Any]:
        info = self._traces.get(span.trace_id)
        meta = info.metadata if info else {}
        return {
            "stage_label": meta.get("stage_label")
            or (info.workflow_name if info else None),
            "round": meta.get("round"),
            "agent_role": meta.get("agent_role"),
            "quality_profile": meta.get("quality_profile"),
            "agent_name": self._ancestor_agent_name(span.parent_id),
        }

    def _ancestor_agent_name(self, span_id: str | None) -> str | None:
        seen: set[str] = set()
        cursor = span_id
        while cursor and cursor not in seen:
            seen.add(cursor)
            name = self._agent_by_span_id.get(cursor)
            if name is not None:
                return name
            cursor = self._parent_by_span_id.get(cursor)
        return None

    # -------------------------------------------------------- TracingProcessor

    def on_trace_start(self, trace: Trace) -> None:
        exported = trace.export() or {}
        meta = exported.get("metadata") or {}
        workflow = exported.get("workflow_name") or getattr(trace, "name", None)
        self._traces[trace.trace_id] = _TraceInfo(
            workflow_name=workflow or "",
            metadata=dict(meta) if isinstance(meta, dict) else {},
        )

    def on_trace_end(self, trace: Trace) -> None:
        self._traces.pop(trace.trace_id, None)

    def on_span_start(self, span: Span[Any]) -> None:
        self._parent_by_span_id[span.span_id] = span.parent_id
        data = span.span_data
        if isinstance(data, AgentSpanData):
            self._agent_by_span_id[span.span_id] = data.name

    def on_span_end(self, span: Span[Any]) -> None:
        try:
            self._emit_row_for(span)
        finally:
            self._agent_by_span_id.pop(span.span_id, None)
            self._parent_by_span_id.pop(span.span_id, None)

    def _emit_row_for(self, span: Span[Any]) -> None:
        data = span.span_data
        attribution = self._attribution_for(span)
        base = {
            **attribution,
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_id,
            "started_at": span.started_at,
            "ended_at": span.ended_at,
            "duration_ms": _duration_ms(span),
            "status": "error" if span.error else "ok",
            "error": span.error,
        }

        if isinstance(data, FunctionSpanData):
            parsed = _maybe_parse_args(data.input)
            output_str = (
                data.output
                if isinstance(data.output, str)
                else ("" if data.output is None else str(data.output))
            )
            self._emit(
                {
                    **base,
                    "tool_source": "function",
                    "tool_name": data.name,
                    "args_json": parsed if isinstance(parsed, dict) else {"_raw": parsed},
                    "args_digest": _extract_args_digest(parsed),
                    "output_chars": len(output_str),
                    "output_preview": normalize_preview(output_str, _OUTPUT_PREVIEW_CHARS),
                    "mcp_data": data.mcp_data,
                }
            )
            return

        if isinstance(data, ResponseSpanData):
            response = data.response
            if response is None:
                return
            for item in _iter_hosted_tool_items(response):
                item_dict = _to_plain_dict(item)
                hosted_args = _hosted_args(item_dict)
                hosted_type = item_dict.get("type", "") or ""
                tool_name = (
                    hosted_type[:-5] if hosted_type.endswith("_call") else hosted_type
                )
                self._emit(
                    {
                        **base,
                        "tool_source": "hosted",
                        "tool_name": tool_name,
                        "call_id": item_dict.get("call_id") or item_dict.get("id"),
                        "args_json": hosted_args,
                        "args_digest": _extract_args_digest(hosted_args),
                        "hosted_status": item_dict.get("status"),
                    }
                )
            self._emit(
                {
                    **base,
                    "tool_source": "llm_turn",
                    "tool_name": "response",
                    "response_id": getattr(response, "id", None),
                    "model": getattr(response, "model", None),
                    "usage": _response_usage(response),
                }
            )
            return

        if isinstance(data, AgentSpanData):
            self._emit(
                {
                    **base,
                    "tool_source": "agent",
                    "tool_name": data.name,
                    "agent_name": data.name,
                    "handoffs": data.handoffs,
                    "tools": data.tools,
                    "output_type": data.output_type,
                }
            )
            return

        if isinstance(data, HandoffSpanData):
            self._emit(
                {
                    **base,
                    "tool_source": "handoff",
                    "tool_name": "handoff",
                    "from_agent": data.from_agent,
                    "to_agent": data.to_agent,
                }
            )
            return

    def shutdown(self) -> None:
        try:
            self.force_flush()
        finally:
            with self._lock:
                if not self._file.closed:
                    self._file.close()

    def force_flush(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
