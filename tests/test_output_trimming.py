from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.tool_context import ToolContext

import main
from kernel_agents.context import LEGACY_TOOL_LIMITS, SharedContext
from kernel_agents.tools import (
    DEFAULT_READ_FILE_LIMIT,
    DEFAULT_SEARCH_OUTPUT_LIMIT,
    DEFAULT_WEB_FETCH_LIMIT,
    SHELL_OVERFLOW_BANNER,
    SHELL_OVERFLOW_FILE_NAME,
    diff_files,
    grep_search,
    list_directory,
    read_file,
    run_correctness_check,
    run_full_benchmark,
    run_stage_validation,
    run_synthetic_check,
    web_fetch,
)


def _shared_context(project_root: Path) -> SharedContext:
    return SharedContext(
        project_root=str(project_root),
        solution_dir="solution/dsa_attention",
        notes_dir="notes/dsa_attention",
        quality_profile="legacy",
        tool_limits=LEGACY_TOOL_LIMITS,
    )


async def _invoke_tool(tool: Any, project_root: Path, **kwargs: Any) -> str:
    arguments = json.dumps(kwargs)
    context = ToolContext(
        context=_shared_context(project_root),
        tool_name=tool.name,
        tool_call_id="tool-call-1",
        tool_arguments=arguments,
    )
    result = await tool.on_invoke_tool(context, arguments)
    assert isinstance(result, str)
    return result


def _overflow_path(project_root: Path) -> Path:
    return project_root / SHELL_OVERFLOW_FILE_NAME


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: str,
        stderr: str,
        returncode: int | None,
        timeout_on_first_communicate: bool = False,
    ) -> None:
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()
        self.returncode = returncode
        self.timeout_on_first_communicate = timeout_on_first_communicate
        self.communicate_calls = 0
        self.pid = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self.timeout_on_first_communicate and self.communicate_calls == 1:
            raise asyncio.TimeoutError
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True


@pytest.mark.asyncio
async def test_read_file_keeps_full_project_files(tmp_path: Path) -> None:
    target = tmp_path / "kernel_agents" / "small.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('a')\nprint('b')\n", encoding="utf-8")

    result = await _invoke_tool(read_file, tmp_path, file_path="kernel_agents/small.py")

    assert "retrieved trimmed" not in result
    assert "1: print('a')" in result
    assert "2: print('b')" in result


@pytest.mark.asyncio
async def test_read_file_auto_windows_large_reference_file(tmp_path: Path) -> None:
    target = tmp_path / "references" / "big_ref.py"
    target.parent.mkdir(parents=True)
    target.write_text("".join(f"line_{i}\n" for i in range(1, 501)), encoding="utf-8")

    result = await _invoke_tool(read_file, tmp_path, file_path="references/big_ref.py")

    assert result.startswith(
        "retrieved trimmed references/big_ref.py:[1]-[400] of 500 lines;"
    )
    assert "1: line_1" in result
    assert "400: line_400" in result
    assert "401: line_401" not in result


@pytest.mark.asyncio
async def test_read_file_respects_explicit_ranges_on_large_reference_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "references" / "big_ref.py"
    target.parent.mkdir(parents=True)
    target.write_text("".join(f"line_{i}\n" for i in range(1, 501)), encoding="utf-8")

    result = await _invoke_tool(
        read_file,
        tmp_path,
        file_path="references/big_ref.py",
        start_line=450,
        end_line=460,
    )

    assert "retrieved trimmed" not in result
    assert "450: line_450" in result
    assert "460: line_460" in result
    assert "449: line_449" not in result


@pytest.mark.asyncio
async def test_read_file_marks_hard_cap_truncation(tmp_path: Path) -> None:
    target = tmp_path / "kernel_agents" / "huge.py"
    target.parent.mkdir(parents=True)
    long_line = "x" * 1000
    target.write_text("".join(f"{long_line}\n" for _ in range(100)), encoding="utf-8")

    result = await _invoke_tool(read_file, tmp_path, file_path="kernel_agents/huge.py")

    assert (
        f"retrieved trimmed kernel_agents/huge.py:[1]-[100] at {DEFAULT_READ_FILE_LIMIT} chars"
        in result
    )
    assert f"... (truncated at {DEFAULT_READ_FILE_LIMIT} chars)" in result


@pytest.mark.asyncio
async def test_grep_search_marks_truncated_results(tmp_path: Path) -> None:
    target = tmp_path / "source.txt"
    target.write_text("".join(f"match_{i}\n" for i in range(1, 151)), encoding="utf-8")

    result = await _invoke_tool(
        grep_search,
        tmp_path,
        pattern="match_",
        path="source.txt",
    )

    assert result.startswith(
        "retrieved trimmed search results for source.txt; showing up to the first 100 matches"
    )
    assert "100:match_100" in result or "100: match_100" in result
    assert "101:match_101" not in result and "101: match_101" not in result


@pytest.mark.asyncio
async def test_grep_search_marks_char_truncation(tmp_path: Path) -> None:
    target = tmp_path / "long_matches.txt"
    long_line = "x" * 195
    target.write_text(
        "".join(f"{i}:{long_line}\n" for i in range(1, 150)),
        encoding="utf-8",
    )

    result = await _invoke_tool(
        grep_search,
        tmp_path,
        pattern="x+",
        path="long_matches.txt",
    )

    assert result.startswith(
        "retrieved trimmed search results for long_matches.txt; showing up to the first 100 matches"
    )
    assert (
        f"retrieved trimmed search results for long_matches.txt at {DEFAULT_SEARCH_OUTPUT_LIMIT} chars"
        in result
    )
    assert result.endswith("... (search output truncated)")


@pytest.mark.asyncio
async def test_web_fetch_marks_trimmed_responses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    httpx = pytest.importorskip("httpx")

    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            assert url == "https://example.com/ptx"
            return _FakeResponse("z" * (DEFAULT_WEB_FETCH_LIMIT + 25))

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = await _invoke_tool(web_fetch, tmp_path, url="https://example.com/ptx")

    assert result.startswith(
        f"retrieved trimmed https://example.com/ptx:[1]-[{DEFAULT_WEB_FETCH_LIMIT}] "
        f"of {DEFAULT_WEB_FETCH_LIMIT + 25} chars"
    )


@pytest.mark.asyncio
async def test_persistent_data_tools_do_not_create_shell_overflow_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "kernel_agents" / "huge.py"
    target.parent.mkdir(parents=True)
    target.write_text("".join(("x" * 1000) + "\n" for _ in range(100)), encoding="utf-8")

    read_result = await _invoke_tool(read_file, tmp_path, file_path="kernel_agents/huge.py")
    assert f"... (truncated at {DEFAULT_READ_FILE_LIMIT} chars)" in read_result
    assert _overflow_path(tmp_path).exists() is False

    httpx = pytest.importorskip("httpx")

    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            assert url == "https://example.com/ptx"
            return _FakeResponse("z" * (DEFAULT_WEB_FETCH_LIMIT + 25))

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    fetch_result = await _invoke_tool(web_fetch, tmp_path, url="https://example.com/ptx")
    assert fetch_result.startswith("retrieved trimmed https://example.com/ptx")
    assert _overflow_path(tmp_path).exists() is False


@pytest.mark.asyncio
async def test_list_directory_returns_entries(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file_a.py").write_text("hello", encoding="utf-8")
    (tmp_path / "file_b.txt").write_text("world!", encoding="utf-8")

    result = await _invoke_tool(list_directory, tmp_path, path="")

    assert "d  subdir/" in result
    assert "f  file_a.py" in result
    assert "f  file_b.txt" in result


@pytest.mark.asyncio
async def test_list_directory_rejects_path_escape(tmp_path: Path) -> None:
    result = await _invoke_tool(list_directory, tmp_path, path="../../etc")

    assert "ERROR" in result


@pytest.mark.asyncio
async def test_diff_files_identical(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("line1\nline2\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("line1\nline2\n", encoding="utf-8")

    result = await _invoke_tool(diff_files, tmp_path, file_a="a.py", file_b="b.py")

    assert result == "Files are identical."


@pytest.mark.asyncio
async def test_diff_files_shows_changes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("line1\nline2\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("line1\nchanged\n", encoding="utf-8")

    result = await _invoke_tool(diff_files, tmp_path, file_a="a.py", file_b="b.py")

    assert "--- a.py" in result
    assert "+++ b.py" in result
    assert "-line2" in result
    assert "+changed" in result


@pytest.mark.asyncio
async def test_diff_files_rejects_escape(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x", encoding="utf-8")

    result = await _invoke_tool(
        diff_files, tmp_path, file_a="a.py", file_b="../../etc/passwd"
    )

    assert "ERROR" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "stdout", "expected_source", "expected_command", "expected_summary"),
    [
        (
            run_stage_validation,
            ("log\n" * 6_000) + "SYNTHETIC RESULTS: 2/2 cases passed\n",
            "run_stage_validation",
            "scripts/bench_synthetic.py",
            "Summary: 2/2 synthetic cases passed",
        ),
        (
            run_synthetic_check,
            ("log\n" * 6_000) + "SYNTHETIC RESULTS: 2/2 cases passed\n",
            "run_synthetic_check",
            "scripts/bench_synthetic.py",
            "Summary: 2/2 synthetic cases passed",
        ),
        (
            run_correctness_check,
            ("log\n" * 6_000) + "RESULTS: 3/3 workloads passed\nAverage speedup: 1.5x\n",
            "run_correctness_check",
            "scripts/bench.py",
            "Summary: 3/3 workloads passed",
        ),
        (
            run_full_benchmark,
            ("log\n" * 6_000) + "RESULTS: 3/3 workloads passed\nAverage speedup: 1.5x\n",
            "run_full_benchmark",
            "scripts/bench.py",
            "Summary: 3/3 workloads passed",
        ),
    ],
)
async def test_workflow_tools_spill_large_transcripts_and_keep_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: Any,
    stdout: str,
    expected_source: str,
    expected_command: str,
    expected_summary: str,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    process = _FakeProcess(stdout=stdout, stderr="", returncode=0)

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr("kernel_agents.tools.asyncio.create_subprocess_exec", _create_subprocess_exec)

    kwargs: dict[str, Any] = {}
    if tool is run_stage_validation:
        kwargs = {
            "stage_id": "warp1::qk_mma",
        }

    result = await _invoke_tool(tool, tmp_path, **kwargs)
    overflow_text = _overflow_path(tmp_path).read_text(encoding="utf-8")

    assert result.startswith(SHELL_OVERFLOW_BANNER)
    assert expected_summary in result
    assert "Command:" in result
    assert f"Source tool: {expected_source}" in overflow_text
    assert expected_command in overflow_text
    assert "This file is overwritten by the next overflowing shell-like tool call." in overflow_text


def test_public_codex_is_the_default_run_config_profile() -> None:
    assert main.make_run_config().call_model_input_filter is main.public_codex_input_filter


def test_legacy_run_config_has_no_prompt_compaction_filter() -> None:
    assert main.make_run_config("legacy").call_model_input_filter is None
