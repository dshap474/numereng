"""Internal training engine infrastructure."""

from numereng.features.training.strategies.core.dispatch import resolve_training_engine
from numereng.features.training.strategies.core.protocol import (
    TrainingEngineMode,
    TrainingEnginePlan,
    TrainingProfile,
)

__all__ = [
    "TrainingEngineMode",
    "TrainingProfile",
    "TrainingEnginePlan",
    "resolve_training_engine",
]
