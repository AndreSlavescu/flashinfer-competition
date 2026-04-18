from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kernel_agents.context import ToolLimitSettings
from kernel_agents.tools import (
    SHELL_DUMP_FILE_NAME,
    diff_files,
    glob_files,
    grep_search,
    list_directory,
    read_file,
    run_synthetic_check,
    web_fetch,
)
from tests.support import FakeProcess, invoke_tool, shared_context


@pytest.mark.asyncio
async def test_glob_files_blocks_traversal_and_normalizes_allowed_matches(
    tmp_path: Path,
) -> None:
    allowed = (
        tmp_path
        / "references"
        / "cutlass"
        / "python"
        / "CuTeDSL"
        / "cutlass"
        / "cute"
        / "example.py"
    )
    allowed.parent.mkdir(parents=True)
    allowed.write_text("pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def outside():\n    pass\n", encoding="utf-8")

    ctx = shared_context(tmp_path, current_agent_role="designer")

    traversal_result = await invoke_tool(
        glob_files,
        tmp_path,
        context=ctx,
        pattern="../*.py",
        directory="references",
    )
    normalized_result = await invoke_tool(
        glob_files,
        tmp_path,
        context=ctx,
        pattern="**/*.py",
        directory="references/cutlass/python/CuTeDSL/cutlass/cute",
    )

    assert traversal_result == "No files found."
    assert "references/cutlass/python/CuTeDSL/cutlass/cute/example.py" in normalized_result
    assert ".." not in normalized_result


@pytest.mark.asyncio
async def test_glob_files_requires_valid_current_agent_role(tmp_path: Path) -> None:
    (tmp_path / "references").mkdir()

    result = await invoke_tool(
        glob_files,
        tmp_path,
        context=shared_context(tmp_path, current_agent_role=""),
        pattern="*.py",
        directory="references",
    )

    assert result == "ERROR: Scoped tool call is missing a valid current_agent_role."


@pytest.mark.asyncio
async def test_context_aware_limits_propagate_to_repo_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_limits = ToolLimitSettings(
        shell_output_limit_chars=60,
        search_output_limit_chars=60,
        web_fetch_limit_chars=10,
        read_file_limit_chars=50,
        diff_limit_chars=30,
        search_match_limit=2,
        glob_match_limit=3,
        list_directory_cap=3,
        reference_read_window_lines=5,
        grep_max_columns=80,
    )
    ctx = shared_context(
        tmp_path,
        tool_limits=custom_limits,
        current_agent_role="designer",
    )

    grep_target = tmp_path / "matches.txt"
    grep_target.write_text("match_1\nmatch_2\nmatch_3\nmatch_4\n", encoding="utf-8")
    grep_result = await invoke_tool(
        grep_search,
        tmp_path,
        context=ctx,
        pattern="match_",
        path="matches.txt",
    )
    assert grep_result.startswith(
        "retrieved trimmed search results for matches.txt; showing up to the first 2 matches"
    )
    assert "match_3" not in grep_result

    glob_dir = (
        tmp_path
        / "references"
        / "cutlass"
        / "python"
        / "CuTeDSL"
        / "cutlass"
        / "cute"
        / "globbed"
    )
    glob_dir.mkdir(parents=True)
    for index in range(1, 6):
        (glob_dir / f"file_{index}.py").write_text("pass\n", encoding="utf-8")
    glob_result = await invoke_tool(
        glob_files,
        tmp_path,
        context=ctx,
        pattern="*.py",
        directory="references/cutlass/python/CuTeDSL/cutlass/cute/globbed",
    )
    assert glob_result.startswith(
        "retrieved trimmed glob results for references/cutlass/python/CuTeDSL/cutlass/cute/globbed; showing up to the first 3 matches"
    )
    assert "file_3.py" in glob_result
    assert "file_4.py" not in glob_result

    list_dir = tmp_path / "listed"
    list_dir.mkdir()
    for index in range(1, 6):
        (list_dir / f"entry_{index}.txt").write_text("x\n", encoding="utf-8")
    list_result = await invoke_tool(list_directory, tmp_path, context=ctx, path="listed")
    assert list_result.startswith("listed/  (5 entries) — showing first 3")
    assert "entry_3.txt" in list_result
    assert "entry_4.txt" not in list_result

    ref_target = tmp_path / "references" / "big_ref.py"
    ref_target.parent.mkdir(exist_ok=True)
    ref_target.write_text("".join(f"line_{i}\n" for i in range(1, 20)), encoding="utf-8")
    read_result = await invoke_tool(
        read_file,
        tmp_path,
        context=ctx,
        file_path="references/big_ref.py",
    )
    assert read_result.startswith("retrieved trimmed references/big_ref.py:[1]-[5] of 19 lines;")
    assert "6: line_6" not in read_result

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
            assert url == "https://example.com/small-cap"
            return _FakeResponse("abcdefghijklmnop")

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    fetch_result = await invoke_tool(
        web_fetch,
        tmp_path,
        context=ctx,
        url="https://example.com/small-cap",
    )
    assert fetch_result.startswith(
        "retrieved trimmed https://example.com/small-cap:[1]-[10] of 16 chars"
    )

    (tmp_path / "a.py").write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("line1\nchanged2\nchanged3\nchanged4\n", encoding="utf-8")
    diff_result = await invoke_tool(
        diff_files,
        tmp_path,
        context=ctx,
        file_a="a.py",
        file_b="b.py",
    )
    assert diff_result.endswith("... (truncated)")

    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    fake_process = FakeProcess(
        stdout=("log\n" * 100) + "SYNTHETIC RESULTS: 1/1 cases passed\n"
    )

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        return fake_process

    monkeypatch.setattr(
        "kernel_agents.tools.asyncio.create_subprocess_exec",
        _create_subprocess_exec,
    )
    shell_result = await invoke_tool(run_synthetic_check, tmp_path, context=ctx)

    assert shell_result.startswith(
        "retrieved trimmed shell output to the last 200 lines and 60 chars; full transcript saved to last_shell_dump.txt"
    )
    dump_text = (tmp_path / SHELL_DUMP_FILE_NAME).read_text(encoding="utf-8")
    assert "Source tool: run_synthetic_check" in dump_text
    assert "SYNTHETIC RESULTS: 1/1 cases passed" in dump_text
