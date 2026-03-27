"""Typed errors for training feature workflows."""

from __future__ import annotations


class TrainingError(Exception):
    """Base error for training feature failures."""


class TrainingConfigError(TrainingError):
    """Raised when training config parsing or validation fails."""


class TrainingDataError(TrainingError):
    """Raised when training data retrieval or preparation fails."""


class TrainingModelError(TrainingError):
    """Raised when model construction or training fails."""


class TrainingMetricsError(TrainingError):
    """Raised when metric computation fails."""


class TrainingCanceledError(TrainingError):
    """Raised when cooperative cancellation interrupts one local run."""
