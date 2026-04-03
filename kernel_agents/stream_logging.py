"""Structured streaming progress logging for Agents SDK runs."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agents.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent, StreamEvent

MAX_PREVIEW_CHARS = 120
_HINT_KEYS = ("file_path", "path", "command", "query", "url", "pattern")
_APPLY_PATCH_VERBS = {
    "create_file": "create",
    "update_file": "update",
    "delete_file": "delete",
}


def _get_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _tool_call_id(raw_item: Any) -> str | None:
    call_id = _get_field(raw_item, "call_id")
    return call_id if isinstance(call_id, str) and call_id else None


def _agent_name(item: Any, fallback: str = "agent") -> str:
    agent = getattr(item, "agent", None)
    name = getattr(agent, "name", None)
    return name if isinstance(name, str) and name else fallback


def normalize_preview(value: Any, limit: int = MAX_PREVIEW_CHARS) -> str:
    if value is None:
        return ""

    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _format_hint_value(value: Any) -> str:
    if isinstance(value, list):
        if value and all(isinstance(entry, str) for entry in value):
            return shlex.join(value)
        return json.dumps(value, sort_keys=True)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True)
    return str(value)


def format_function_tool_call(raw_item: Any) -> str:
    name = _get_field(raw_item, "name", "function")
    args_str = _get_field(raw_item, "arguments", "") or ""

    try:
        parsed = json.loads(args_str)
    except Exception:
        parsed = None

    if isinstance(parsed, Mapping):
        for key in _HINT_KEYS:
            if key in parsed:
                hint = normalize_preview(_format_hint_value(parsed[key]))
                return f"{name} {key}={hint}"
        if parsed:
            return f"{name} {normalize_preview(json.dumps(dict(parsed), sort_keys=True))}"

    args_preview = normalize_preview(args_str)
    return f"{name} {args_preview}".rstrip()


def format_shell_call(raw_item: Any) -> str:
    action = _get_field(raw_item, "action")
    commands = _get_field(action, "commands") or []
    if isinstance(commands, list) and commands:
        first_command = normalize_preview(commands[0])
        if len(commands) > 1:
            return f"shell {first_command} (+{len(commands) - 1} more)"
        return f"shell {first_command}"
    return "shell"


def format_local_shell_call(raw_item: Any) -> str:
    action = _get_field(raw_item, "action")
    command = _get_field(action, "command") or []
    if isinstance(command, list) and command:
        return f"local_shell {normalize_preview(shlex.join(command))}"
    return "local_shell"


def format_apply_patch_call(raw_item: Any) -> str:
    operation = _get_field(raw_item, "operation")
    op_type = _get_field(operation, "type") or "apply"
    path = _get_field(operation, "path")
    verb = _APPLY_PATCH_VERBS.get(op_type, str(op_type))
    if isinstance(path, str) and path:
        return f"apply_patch {verb} {normalize_preview(path)}"
    return f"apply_patch {verb}"


def format_web_search_call(raw_item: Any) -> str:
    action = _get_field(raw_item, "action")
    action_type = _get_field(action, "type")

    if action_type == "search":
        queries = _get_field(action, "queries") or []
        query = queries[0] if isinstance(queries, list) and queries else _get_field(action, "query")
        if isinstance(query, str) and query:
            return f"web_search query={normalize_preview(query)}"
        return "web_search search"

    if action_type == "open_page":
        url = _get_field(action, "url")
        if isinstance(url, str) and url:
            return f"web_search open url={normalize_preview(url)}"
        return "web_search open"

    if action_type == "find_in_page":
        pattern = _get_field(action, "pattern")
        if isinstance(pattern, str) and pattern:
            return f"web_search find pattern={normalize_preview(pattern)}"
        return "web_search find"

    return "web_search"


def format_tool_search_call(raw_item: Any) -> str:
    arguments = _get_field(raw_item, "arguments")
    if arguments:
        preview = normalize_preview(_format_hint_value(arguments))
        return f"tool_search {preview}"
    return "tool_search"


def format_tool_call_label(raw_item: Any) -> str:
    item_type = _get_field(raw_item, "type")

    if item_type == "function_call":
        return format_function_tool_call(raw_item)
    if item_type == "shell_call":
        return format_shell_call(raw_item)
    if item_type == "local_shell_call":
        return format_local_shell_call(raw_item)
    if item_type == "apply_patch_call":
        return format_apply_patch_call(raw_item)
    if item_type == "web_search_call":
        return format_web_search_call(raw_item)
    if item_type == "tool_search_call":
        return format_tool_search_call(raw_item)

    name = _get_field(raw_item, "name")
    if isinstance(name, str) and name:
        return normalize_preview(name)

    if isinstance(item_type, str) and item_type:
        return normalize_preview(item_type)

    return "tool"


def _shell_status(raw_item: Any) -> str:
    entries = _get_field(raw_item, "output") or []
    if not isinstance(entries, list) or not entries:
        status = _get_field(raw_item, "status")
        return str(status) if status else ""

    outcome = _get_field(entries[0], "outcome")
    outcome_type = _get_field(outcome, "type")
    if outcome_type == "exit":
        exit_code = _get_field(outcome, "exit_code")
        if isinstance(exit_code, int):
            return f"exit={exit_code}"
        return "exit"
    if outcome_type == "timeout":
        return "timeout"

    status = _get_field(raw_item, "status")
    return str(status) if status else ""


def _shell_preview(raw_item: Any, output: Any) -> str:
    text = normalize_preview(output)
    if text:
        return text

    entries = _get_field(raw_item, "output") or []
    if not isinstance(entries, list) or not entries:
        return ""

    fragments: list[str] = []
    first_entry = entries[0]
    stdout = normalize_preview(_get_field(first_entry, "stdout"))
    stderr = normalize_preview(_get_field(first_entry, "stderr"))
    if stdout:
        fragments.append(stdout)
    if stderr:
        fragments.append(stderr)
    return normalize_preview(" ".join(fragments))


def format_tool_output(raw_item: Any, output: Any, label: str | None = None) -> str:
    item_type = _get_field(raw_item, "type")

    if item_type == "shell_call_output":
        details = " ".join(
            part for part in (_shell_status(raw_item), _shell_preview(raw_item, output)) if part
        )
    elif item_type == "apply_patch_call_output":
        details = " ".join(
            part
            for part in (
                normalize_preview(_get_field(raw_item, "status")),
                normalize_preview(output or _get_field(raw_item, "output")),
            )
            if part
        )
    else:
        details = normalize_preview(output or _get_field(raw_item, "output"))

    if label and details:
        return f"{label}: {details}"
    if label:
        return label
    if details:
        return details

    fallback = _get_field(raw_item, "type")
    if isinstance(fallback, str) and fallback:
        return normalize_preview(fallback)
    return "tool output"


@dataclass
class StreamProgressRenderer:
    """Render streamed agent events into concise one-line progress logs."""

    current_agent_name: str = "agent"
    tool_labels: dict[str, str] = field(default_factory=dict)

    def render_event(self, event: StreamEvent) -> str | None:
        if isinstance(event, AgentUpdatedStreamEvent):
            self.current_agent_name = event.new_agent.name
            return f"  [{self.current_agent_name}] agent active"

        if not isinstance(event, RunItemStreamEvent):
            return None

        item = event.item
        agent_name = _agent_name(item, self.current_agent_name)

        if event.name == "tool_called":
            label = format_tool_call_label(item.raw_item)
            call_id = _tool_call_id(item.raw_item)
            if call_id:
                self.tool_labels[call_id] = label
            return f"  [{agent_name}] -> {label}"

        if event.name == "tool_output":
            call_id = _tool_call_id(item.raw_item)
            label = self.tool_labels.get(call_id) if call_id else None
            message = format_tool_output(item.raw_item, getattr(item, "output", None), label)
            return f"  [{agent_name}] <- {message}"

        if event.name == "handoff_requested":
            target = _get_field(item.raw_item, "name") or "agent"
            return f"  [{agent_name}] handoff requested -> {target}"

        if event.name == "handoff_occured":
            target_agent = getattr(item, "target_agent", None)
            target_name = getattr(target_agent, "name", None) or "agent"
            return f"  [{agent_name}] handoff complete -> {target_name}"

        if event.name == "tool_search_called":
            label = format_tool_search_call(item.raw_item)
            call_id = _tool_call_id(item.raw_item)
            if call_id:
                self.tool_labels[call_id] = label
            return f"  [{agent_name}] -> {label}"

        if event.name == "tool_search_output_created":
            call_id = _tool_call_id(item.raw_item)
            label = self.tool_labels.get(call_id, "tool_search results")
            return f"  [{agent_name}] <- {label}: ready"

        return None


def _default_emit(line: str) -> None:
    print(line, flush=True)


async def consume_streamed_run(
    result: Any,
    emit: Callable[[str], None] = _default_emit,
) -> Any:
    """Drain a streamed run and emit structured progress lines as events arrive."""

    renderer = StreamProgressRenderer()
    async for event in result.stream_events():
        line = renderer.render_event(event)
        if line:
            emit(line)
    return result
