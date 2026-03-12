"""Public surface for training feature services."""

from numereng.features.scoring.run_service import score_run
from numereng.features.training.errors import (
    TrainingConfigError,
    TrainingDataError,
    TrainingError,
    TrainingMetricsError,
    TrainingModelError,
)
from numereng.features.training.models import ScoreRunResult, TrainingRunResult
from numereng.features.training.service import run_training
from numereng.features.training.strategies import TrainingEngineMode, TrainingProfile

__all__ = [
    "ScoreRunResult",
    "TrainingConfigError",
    "TrainingDataError",
    "TrainingError",
    "TrainingMetricsError",
    "TrainingModelError",
    "TrainingEngineMode",
    "TrainingProfile",
    "TrainingRunResult",
    "score_run",
    "run_training",
]
