"""Public surface for baseline feature workflows."""

from numereng.features.baseline.contracts import BaselineBuildRequest, BaselineBuildResult
from numereng.features.baseline.service import (
    BaselineError,
    BaselineExecutionError,
    BaselineValidationError,
    build_baseline,
)

__all__ = [
    "BaselineBuildRequest",
    "BaselineBuildResult",
    "BaselineError",
    "BaselineExecutionError",
    "BaselineValidationError",
    "build_baseline",
]
