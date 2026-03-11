"""Internal API stage adapters for public workflow orchestration."""

from __future__ import annotations

from pathlib import Path

from numereng.api.contracts import TrainRunRequest
from numereng.features.training._pipeline import (
    TrainingPipelineState,
    cleanup_training_run as cleanup_training_stage,
    fail_training_run as fail_training_stage,
    finalize_training_run as finalize_training_stage,
    load_training_data as load_training_data_stage,
    prepare_training_run as prepare_training_stage,
    score_predictions as score_predictions_stage,
    train_model as train_model_stage,
)


def prepare_training_run(request: TrainRunRequest) -> TrainingPipelineState:
    """Initialize one local training run and persist bootstrap artifacts."""
    return prepare_training_stage(
        config_path=request.config_path,
        output_dir=request.output_dir,
        profile=request.profile,
        engine_mode=request.engine_mode,
        window_size_eras=request.window_size_eras,
        embargo_eras=request.embargo_eras,
        experiment_id=request.experiment_id,
    )


def load_training_data(state: TrainingPipelineState) -> TrainingPipelineState:
    """Load training data and prepare the model data loader."""
    return load_training_data_stage(state)


def train_model(state: TrainingPipelineState) -> TrainingPipelineState:
    """Fit the configured model and persist the prediction artifact."""
    return train_model_stage(state)


def score_predictions(state: TrainingPipelineState) -> TrainingPipelineState:
    """Compute post-run scoring from the saved predictions artifact."""
    return score_predictions_stage(state)


def finalize_training_run(state: TrainingPipelineState) -> tuple[str, Path, Path]:
    """Finalize the run manifest and return canonical output paths."""
    result = finalize_training_stage(state)
    return result.run_id, result.predictions_path, result.results_path


def fail_training_run(state: TrainingPipelineState, exc: Exception) -> None:
    """Persist failure state for a staged training pipeline."""
    fail_training_stage(state, exc)


def cleanup_training_run(state: TrainingPipelineState) -> None:
    """Release runtime resources for a staged training pipeline."""
    cleanup_training_stage(state)


__all__ = [
    "TrainingPipelineState",
    "cleanup_training_run",
    "fail_training_run",
    "finalize_training_run",
    "load_training_data",
    "prepare_training_run",
    "score_predictions",
    "train_model",
]
