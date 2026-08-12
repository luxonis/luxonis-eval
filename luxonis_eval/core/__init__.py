from typing import TYPE_CHECKING

from .results import (
    EvaluationResult,
    MetricsResult,
    MetricValues,
    ThroughputResult,
)

if TYPE_CHECKING:
    from .core import LuxonisEval


def __getattr__(name: str) -> object:
    if name == "LuxonisEval":
        from .core import LuxonisEval

        return LuxonisEval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EvaluationResult",
    "LuxonisEval",
    "MetricsResult",
    "MetricValues",
    "ThroughputResult",
]
