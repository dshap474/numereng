"""Typed errors for modular post-training scoring workflows."""

from __future__ import annotations

from numereng.features.training.errors import TrainingError


class ScoringError(TrainingError):
    """Base error for scoring feature failures."""


class ScoringConfigError(ScoringError):
    """Raised when scoring inputs are invalid."""


class ScoringDataError(ScoringError):
    """Raised when scoring data retrieval or alignment fails."""


class ScoringExecutionError(ScoringError):
    """Raised when scoring execution fails unexpectedly."""
