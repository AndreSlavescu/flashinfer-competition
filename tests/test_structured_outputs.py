from __future__ import annotations

from typing import Any

import pytest

from agents import AgentOutputSchema

import main
from kernel_agents.context import (
    CoderResult,
    DesignerResult,
    OptimizerResult,
    PlannerResult,
)


@pytest.mark.parametrize(
    ("result_type", "expected_fields"),
    [
        (DesignerResult, {"plan_file", "status", "message"}),
        (
            CoderResult,
            {"generated", "correctness_verified", "status", "message", "reflection"},
        ),
        (
            PlannerResult,
            {"latency_ms", "bottleneck", "strategy_summary", "strategy_file", "is_new_best", "ncu_metrics"},
        ),
        (OptimizerResult, {"kernel_file", "correctness_verified", "status", "message", "reflection"}),
    ],
)
def test_result_models_use_top_level_structured_output(
    result_type: type[Any],
    expected_fields: set[str],
) -> None:
    schema = AgentOutputSchema(result_type).json_schema()

    assert set(schema["properties"]) == expected_fields
    assert set(schema["required"]) == expected_fields
    assert "response" not in schema["properties"]
    assert "response" not in schema["required"]
    assert schema["additionalProperties"] is False


class _FakeRunResult:
    def __init__(self, typed_output: Any) -> None:
        self.final_output = '{"generated":["solution/dsa_attention/kernel_0.py"]}'
        self._typed_output = typed_output
        self.calls: list[tuple[type[Any], bool]] = []

    def final_output_as(self, cls: type[Any], raise_if_incorrect_type: bool = False) -> Any:
        self.calls.append((cls, raise_if_incorrect_type))
        if raise_if_incorrect_type and not isinstance(self._typed_output, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(self._typed_output).__name__}")
        return self._typed_output


def test_require_structured_output_uses_sdk_typed_accessor() -> None:
    typed_output = CoderResult(
        generated=["solution/dsa_attention/kernel_0.py"],
        correctness_verified=True,
        status="success",
        message="validated",
        reflection="all good",
    )
    result = _FakeRunResult(typed_output)

    extracted = main.require_structured_output(result, CoderResult)

    assert extracted == typed_output
    assert result.calls == [(CoderResult, True)]


def test_require_structured_output_rejects_wrong_typed_result() -> None:
    result = _FakeRunResult("not a typed result")

    with pytest.raises(TypeError, match="Expected DesignerResult, got str"):
        main.require_structured_output(result, DesignerResult)

    assert result.calls == [(DesignerResult, True)]


def test_require_structured_output_requires_final_output_as() -> None:
    with pytest.raises(TypeError, match="final_output_as"):
        main.require_structured_output(object(), PlannerResult)
