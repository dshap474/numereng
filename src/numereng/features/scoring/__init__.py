"""Public scoring surface for canonical run metrics."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CanonicalScoringStage",
    "PostTrainingScoringRequest",
    "PostTrainingScoringResult",
    "ResolvedScoringPolicy",
    "RunScoringRequest",
    "RunScoringResult",
    "run_post_training_scoring",
    "run_scoring",
]

_LAZY_EXPORT_MODULES: tuple[str, ...] = (
    "numereng.features.scoring.models",
    "numereng.features.scoring.service",
)


def __getattr__(name: str) -> object:
    for module_name in _LAZY_EXPORT_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
