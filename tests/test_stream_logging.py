from __future__ import annotations

import unittest

from agents.items import ToolCallItem, ToolCallOutputItem, ToolSearchCallItem, ToolSearchOutputItem
from agents.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_web_search import ResponseFunctionWebSearch
from openai.types.responses.response_tool_search_call import ResponseToolSearchCall
from openai.types.responses.response_tool_search_output_item import ResponseToolSearchOutputItem

from kernel_agents.stream_logging import (
    StreamProgressRenderer,
    consume_streamed_run,
    format_tool_call_label,
    format_tool_output,
    normalize_preview,
)


class FakeAgent:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name: str) -> None:
        self.name = name


class StreamLoggingFormatTests(unittest.TestCase):
    def test_format_function_tool_call_prefers_path_hint(self) -> None:
        raw_item = ResponseFunctionToolCall(
            arguments='{"file_path":"solution/dsa_attention/kernel.py","start_line":1}',
            call_id="call-1",
            name="read_file",
            type="function_call",
        )

        self.assertEqual(
            format_tool_call_label(raw_item),
            "read_file file_path=solution/dsa_attention/kernel.py",
        )

    def test_format_shell_output_includes_exit_and_preview(self) -> None:
        raw_item = {
            "type": "shell_call_output",
            "call_id": "shell-1",
            "status": "completed",
            "output": [
                {
                    "stdout": "benchmark passed\nwith notes",
                    "stderr": "",
                    "outcome": {"type": "exit", "exit_code": 0},
                }
            ],
        }

        self.assertEqual(
            format_tool_output(raw_item, "", "shell modal run scripts/bench.py"),
            "shell modal run scripts/bench.py: exit=0 benchmark passed with notes",
        )

    def test_format_shell_output_handles_timeout(self) -> None:
        raw_item = {
            "type": "shell_call_output",
            "call_id": "shell-2",
            "status": "incomplete",
            "output": [
                {
                    "stdout": "",
                    "stderr": "ERROR: timed out",
                    "outcome": {"type": "timeout"},
                }
            ],
        }

        self.assertEqual(
            format_tool_output(raw_item, "", "shell long task"),
            "shell long task: timeout ERROR: timed out",
        )

    def test_format_apply_patch_call_and_output(self) -> None:
        call_item = {
            "type": "apply_patch_call",
            "call_id": "patch-1",
            "operation": {
                "type": "update_file",
                "path": "solution/dsa_attention/kernel_1.py",
                "diff": "@@ ...",
            },
        }
        output_item = {
            "type": "apply_patch_call_output",
            "call_id": "patch-1",
            "status": "completed",
            "output": "Updated solution/dsa_attention/kernel_1.py",
        }

        self.assertEqual(
            format_tool_call_label(call_item),
            "apply_patch update solution/dsa_attention/kernel_1.py",
        )
        self.assertEqual(
            format_tool_output(output_item, "", format_tool_call_label(call_item)),
            "apply_patch update solution/dsa_attention/kernel_1.py: completed Updated solution/dsa_attention/kernel_1.py",
        )

    def test_format_web_search_call_summary(self) -> None:
        raw_item = ResponseFunctionWebSearch(
            id="web-1",
            action={"type": "search", "query": "CUTLASS Blackwell tcgen05 examples"},
            status="completed",
            type="web_search_call",
        )

        self.assertEqual(
            format_tool_call_label(raw_item),
            "web_search query=CUTLASS Blackwell tcgen05 examples",
        )

    def test_normalize_preview_flattens_and_truncates(self) -> None:
        value = "line one\nline two\tline three " + ("x" * 140)
        preview = normalize_preview(value, limit=40)
        self.assertEqual(preview, "line one line two line three xxxxxxxx...")

    def test_unknown_tool_call_fallback(self) -> None:
        self.assertEqual(format_tool_call_label({"type": "mystery_call"}), "mystery_call")


class StreamLoggingRendererTests(unittest.TestCase):
    def test_tool_search_output_uses_cached_label(self) -> None:
        agent = FakeAgent("kernel-planner")
        renderer = StreamProgressRenderer(current_agent_name=agent.name)

        call_item = ToolSearchCallItem(
            agent=agent,
            raw_item=ResponseToolSearchCall(
                id="tool-search-call",
                arguments={"query": "benchmark config"},
                call_id="tool-search-1",
                execution="server",
                status="completed",
                type="tool_search_call",
            ),
        )
        output_item = ToolSearchOutputItem(
            agent=agent,
            raw_item=ResponseToolSearchOutputItem(
                id="tool-search-out",
                call_id="tool-search-1",
                execution="server",
                status="completed",
                tools=[],
                type="tool_search_output",
            ),
        )

        started = renderer.render_event(
            RunItemStreamEvent(name="tool_search_called", item=call_item)
        )
        finished = renderer.render_event(
            RunItemStreamEvent(name="tool_search_output_created", item=output_item)
        )

        self.assertEqual(started, "  [kernel-planner] -> tool_search {\"query\": \"benchmark config\"}")
        self.assertEqual(
            finished,
            "  [kernel-planner] <- tool_search {\"query\": \"benchmark config\"}: ready",
        )


class ConsumeStreamedRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_consume_streamed_run_drains_before_return(self) -> None:
        agent = FakeAgent("kernel-coder")
        lines: list[str] = []

        tool_call = ToolCallItem(
            agent=agent,
            raw_item=ResponseFunctionToolCall(
                arguments='{"file_path":"solution/dsa_attention/kernel.py"}',
                call_id="call-1",
                name="read_file",
                type="function_call",
            ),
        )
        tool_output = ToolCallOutputItem(
            agent=agent,
            raw_item={
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "1: import torch\n2: import cutlass",
            },
            output="1: import torch\n2: import cutlass",
        )

        events = [
            AgentUpdatedStreamEvent(new_agent=agent),
            RunItemStreamEvent(name="tool_called", item=tool_call),
            RunItemStreamEvent(name="tool_output", item=tool_output),
        ]

        class FakeStreamingResult:
            def __init__(self, stream_events):
                self._events = stream_events
                self.generator_completed = False
                self._final_output = "done"
                self.accessed_before_complete = False

            @property
            def final_output(self):
                if not self.generator_completed:
                    self.accessed_before_complete = True
                return self._final_output

            async def stream_events(self):
                for event in self._events:
                    yield event
                self.generator_completed = True

        result = FakeStreamingResult(events)
        returned = await consume_streamed_run(result, emit=lines.append)

        self.assertIs(returned, result)
        self.assertTrue(result.generator_completed)
        self.assertFalse(result.accessed_before_complete)
        self.assertEqual(
            lines,
            [
                "  [kernel-coder] agent active",
                "  [kernel-coder] -> read_file file_path=solution/dsa_attention/kernel.py",
                "  [kernel-coder] <- read_file file_path=solution/dsa_attention/kernel.py: 1: import torch 2: import cutlass",
            ],
        )


if __name__ == "__main__":
    unittest.main()
