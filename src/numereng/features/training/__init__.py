"""Public surface for training feature services."""

from numereng.features.training.errors import (
    TrainingConfigError,
    TrainingDataError,
    TrainingError,
    TrainingMetricsError,
    TrainingModelError,
)
from numereng.features.training.models import TrainingRunResult
from numereng.features.training.service import run_training
from numereng.features.training.strategies import TrainingEngineMode, TrainingProfile

__all__ = [
    "TrainingConfigError",
    "TrainingDataError",
    "TrainingError",
    "TrainingMetricsError",
    "TrainingModelError",
    "TrainingEngineMode",
    "TrainingProfile",
    "TrainingRunResult",
    "run_training",
]
