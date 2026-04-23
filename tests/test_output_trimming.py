from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kernel_agents.tools import (
    DEFAULT_READ_FILE_LIMIT,
    DEFAULT_SEARCH_OUTPUT_LIMIT,
    DEFAULT_WEB_FETCH_LIMIT,
    SHELL_DUMP_FILE_NAME,
    SHELL_DUMP_TAIL_BANNER,
    diff_files,
    grep_search,
    read_file,
    run_correctness_check,
    run_full_benchmark,
    run_ncu_profile,
    run_sass_analysis,
    run_stage_validation,
    run_synthetic_check,
    web_fetch,
)
from tests.support import FakeProcess, invoke_tool


def _dump_path(project_root: Path) -> Path:
    return project_root / SHELL_DUMP_FILE_NAME


@pytest.mark.asyncio
async def test_read_file_windows_large_references_without_creating_shell_dump(
    tmp_path: Path,
) -> None:
    target = tmp_path / "references" / "big_ref.py"
    target.parent.mkdir(parents=True)
    target.write_text("".join(f"line_{i}\n" for i in range(1, 501)), encoding="utf-8")

    result = await invoke_tool(read_file, tmp_path, file_path="references/big_ref.py")

    assert result.startswith("retrieved trimmed references/big_ref.py:[1]-[400] of 500 lines;")
    assert "400: line_400" in result
    assert "401: line_401" not in result
    assert not _dump_path(tmp_path).exists()


@pytest.mark.asyncio
async def test_read_file_marks_hard_cap_truncation(tmp_path: Path) -> None:
    target = tmp_path / "kernel_agents" / "huge.py"
    target.parent.mkdir(parents=True)
    long_line = "x" * 1000
    target.write_text("".join(f"{long_line}\n" for _ in range(100)), encoding="utf-8")

    result = await invoke_tool(read_file, tmp_path, file_path="kernel_agents/huge.py")

    assert (
        f"retrieved trimmed kernel_agents/huge.py:[1]-[100] at {DEFAULT_READ_FILE_LIMIT} chars"
        in result
    )
    assert f"... (truncated at {DEFAULT_READ_FILE_LIMIT} chars)" in result


@pytest.mark.asyncio
async def test_grep_search_reports_match_and_char_truncation(tmp_path: Path) -> None:
    target = tmp_path / "long_matches.txt"
    long_line = "x" * 195
    target.write_text(
        "".join(f"{i}:{long_line}\n" for i in range(1, 150)),
        encoding="utf-8",
    )

    result = await invoke_tool(
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
    assert not _dump_path(tmp_path).exists()


@pytest.mark.asyncio
async def test_web_fetch_marks_trimmed_responses_without_creating_shell_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = await invoke_tool(web_fetch, tmp_path, url="https://example.com/ptx")

    assert result.startswith(
        f"retrieved trimmed https://example.com/ptx:[1]-[{DEFAULT_WEB_FETCH_LIMIT}] "
        f"of {DEFAULT_WEB_FETCH_LIMIT + 25} chars"
    )
    assert not _dump_path(tmp_path).exists()


@pytest.mark.asyncio
async def test_diff_files_shows_unified_changes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("line1\nline2\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("line1\nchanged\n", encoding="utf-8")

    result = await invoke_tool(diff_files, tmp_path, file_a="a.py", file_b="b.py")

    assert "--- a.py" in result
    assert "+++ b.py" in result
    assert "-line2" in result
    assert "+changed" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "source_tool", "command_fragment", "kwargs", "tail_marker"),
    [
        (
            run_stage_validation,
            "run_stage_validation",
            "scripts/bench_synthetic.py",
            {"stage_id": "warp1::qk_mma"},
            "SYNTHETIC RESULTS: 1/1 cases passed",
        ),
        (
            run_synthetic_check,
            "run_synthetic_check",
            "scripts/bench_synthetic.py",
            {},
            "SYNTHETIC RESULTS: 2/2 cases passed",
        ),
        (
            run_correctness_check,
            "run_correctness_check",
            "scripts/bench.py",
            {},
            "RESULTS: 3/3 workloads passed",
        ),
        (
            run_full_benchmark,
            "run_full_benchmark",
            "scripts/bench.py",
            {},
            "Average speedup: 1.5x",
        ),
        (
            run_ncu_profile,
            "run_ncu_profile",
            "tools/ncu/ncu_modal.py",
            {},
            "metric_259",
        ),
        (
            run_sass_analysis,
            "run_sass_analysis",
            "tools/sass/dump_sass_modal.py",
            {"kernel_file": "solution/dsa_attention/kernel.py"},
            "metric_259",
        ),
    ],
)
async def test_modal_tools_write_shell_dump_and_return_recent_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: Any,
    source_tool: str,
    command_fragment: str,
    kwargs: dict[str, Any],
    tail_marker: str,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    (solution_dir / "kernel.py").write_text("def kernel():\n    pass\n", encoding="utf-8")

    stdout = "".join(f"metric_{i}\n" for i in range(1, 260))
    if tool is run_stage_validation:
        stdout += "SYNTHETIC RESULTS: 1/1 cases passed\n"
    elif tool is run_synthetic_check:
        stdout += "SYNTHETIC RESULTS: 2/2 cases passed\n"
    elif tool in (run_correctness_check, run_full_benchmark):
        stdout += "RESULTS: 3/3 workloads passed\nAverage speedup: 1.5x\n"
    process = FakeProcess(stdout=stdout, stderr="", returncode=0)

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(
        "kernel_agents.tools.asyncio.create_subprocess_exec",
        _create_subprocess_exec,
    )

    result = await invoke_tool(tool, tmp_path, **kwargs)
    dump_text = _dump_path(tmp_path).read_text(encoding="utf-8")

    assert result.startswith(SHELL_DUMP_TAIL_BANNER)
    assert "metric_1\n" not in result
    assert tail_marker in result
    assert f"Source tool: {source_tool}" in dump_text
    assert command_fragment in dump_text
    assert "This file is overwritten by the next shell-like tool call." in dump_text
    assert "metric_1" in dump_text


@pytest.mark.asyncio
async def test_run_stage_validation_invokes_single_case_stage_validation_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    invoked: list[tuple[Any, ...]] = []

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        del kwargs
        invoked.append(args)
        return FakeProcess(stdout="SYNTHETIC RESULTS: 1/1 cases passed\n", stderr="", returncode=0)

    monkeypatch.setattr(
        "kernel_agents.tools.asyncio.create_subprocess_exec",
        _create_subprocess_exec,
    )

    await invoke_tool(run_stage_validation, tmp_path, stage_id="warp1::qk_mma")

    assert invoked
    command = list(invoked[0])
    assert command[:2] == [".venv/bin/python", "scripts/bench_synthetic.py"]
    assert "--stage-validation" in command
    assert "kernel_0.py::run" in command


@pytest.mark.asyncio
async def test_modal_tools_overwrite_shell_dump_on_subsequent_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_dir = tmp_path / "solution" / "dsa_attention"
    solution_dir.mkdir(parents=True)
    processes = [
        FakeProcess(stdout="first run\n", stderr="", returncode=0),
        FakeProcess(stdout="second run\n", stderr="", returncode=0),
    ]

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        return processes.pop(0)

    monkeypatch.setattr(
        "kernel_agents.tools.asyncio.create_subprocess_exec",
        _create_subprocess_exec,
    )

    first_result = await invoke_tool(run_correctness_check, tmp_path)
    first_dump = _dump_path(tmp_path).read_text(encoding="utf-8")
    second_result = await invoke_tool(run_correctness_check, tmp_path)
    second_dump = _dump_path(tmp_path).read_text(encoding="utf-8")

    assert "first run" in first_result
    assert "second run" in second_result
    assert "first run" in first_dump
    assert "first run" not in second_dump
    assert "second run" in second_dump
