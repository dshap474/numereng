"""Public scoring surface for canonical run metrics."""

from __future__ import annotations

from importlib import import_module

# Canonical scoring-contract revision. Bumped only when the run scoring pipeline
# changes in a way that makes cross-run metrics incomparable; portfolio surface
# IDs embed this so runs scored under different contracts never compare equal.
SCORING_CONTRACT_VERSION = 1

__all__ = [
    "SCORING_CONTRACT_VERSION",
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
