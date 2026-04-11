"""Public surface for training feature services."""

from __future__ import annotations

from importlib import import_module

from numereng.config.training.contracts import PostTrainingScoringPolicy

__all__ = [
    "ScoreRunResult",
    "PostTrainingScoringPolicy",
    "TrainingCanceledError",
    "TrainingConfigError",
    "TrainingDataError",
    "TrainingError",
    "TrainingMetricsError",
    "TrainingModelError",
    "TrainingEngineMode",
    "TrainingProfile",
    "TrainingRunResult",
    "TrainingRunPreview",
    "score_run",
    "preview_training_run",
    "run_training",
]

_LAZY_EXPORT_MODULES: tuple[str, ...] = (
    "numereng.features.training.errors",
    "numereng.features.training.models",
    "numereng.features.training.strategies.core.protocol",
    "numereng.features.training.service",
    "numereng.features.scoring.run_service",
)


def __getattr__(name: str) -> object:
    if name == "PostTrainingScoringPolicy":
        return PostTrainingScoringPolicy
    for module_name in _LAZY_EXPORT_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
