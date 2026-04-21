"""Training and scoring API handlers."""

from __future__ import annotations

from contextlib import nullcontext

from numereng.api.contracts import ScoreRunRequest, ScoreRunResponse, TrainRunRequest, TrainRunResponse
from numereng.features.telemetry import bind_launch_metadata, get_launch_metadata
from numereng.features.training import (
    TrainingCanceledError,
    TrainingConfigError,
    TrainingDataError,
    TrainingError,
    TrainingMetricsError,
    TrainingModelError,
)
from numereng.platform.errors import NumeraiClientError, PackageError


def _map_training_data_error(exc: TrainingDataError) -> str:
    message = str(exc)
    if message.startswith("training_target_rows_all_unlabeled:"):
        return message
    return f"training_data_load_failed:{message}"


def _map_training_error_detail(exc: TrainingError) -> str:
    message = str(exc)
    if message == "training_launch_metadata_missing":
        return message
    if message.startswith("training_lifecycle_bootstrap_failed:"):
        return message
    if message.startswith("training_run_failed:"):
        return message
    return f"training_run_failed:{message}"


def run_training(request: TrainRunRequest) -> TrainRunResponse:
    """Run full training pipeline from config and return artifact paths."""
    from numereng import api as api_module

    launch_scope = (
        nullcontext()
        if get_launch_metadata() is not None
        else bind_launch_metadata(source="api.run.train", operation_type="run", job_type="run")
    )
    try:
        with launch_scope:
            kwargs = {
                "config_path": request.config_path,
                "output_dir": request.output_dir,
                "post_training_scoring": request.post_training_scoring,
                "engine_mode": request.engine_mode,
                "window_size_eras": request.window_size_eras,
                "embargo_eras": request.embargo_eras,
                "experiment_id": request.experiment_id,
            }
            if request.profile is not None:
                kwargs["profile"] = request.profile
            result = api_module.run_training_pipeline(**kwargs)
    except TrainingConfigError as exc:
        raise PackageError(str(exc)) from exc
    except TrainingDataError as exc:
        raise PackageError(_map_training_data_error(exc)) from exc
    except TrainingModelError as exc:
        message = str(exc)
        if "training_model_backend_missing_lightgbm" in message:
            raise PackageError("training_model_backend_missing") from exc
        raise PackageError(f"training_model_failed:{message}") from exc
    except TrainingMetricsError as exc:
        raise PackageError(f"training_metrics_failed:{exc}") from exc
    except TrainingCanceledError as exc:
        raise PackageError("training_run_canceled") from exc
    except TrainingError as exc:
        message = _map_training_error_detail(exc)
        raise PackageError(message) from exc
    except ValueError as exc:
        raise PackageError(f"training_config_invalid:{exc}") from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"training_unexpected_error:{exc.__class__.__name__}") from exc

    if isinstance(result, TrainRunResponse):
        return result

    return TrainRunResponse(
        run_id=result.run_id,
        predictions_path=str(result.predictions_path),
        results_path=str(result.results_path),
    )


def score_run(request: ScoreRunRequest) -> ScoreRunResponse:
    """Recompute scoring artifacts for one persisted run-id."""
    from numereng import api as api_module

    launch_scope = (
        nullcontext()
        if get_launch_metadata() is not None
        else bind_launch_metadata(source="api.run.score", operation_type="run", job_type="run")
    )
    try:
        with launch_scope:
            result = api_module.score_run_pipeline(
                run_id=request.run_id,
                store_root=request.store_root,
                stage=request.stage,
            )
    except TrainingConfigError as exc:
        raise PackageError(f"training_score_config_invalid:{exc}") from exc
    except TrainingDataError as exc:
        raise PackageError(f"training_score_data_load_failed:{exc}") from exc
    except TrainingMetricsError as exc:
        raise PackageError(f"training_score_metrics_failed:{exc}") from exc
    except TrainingError as exc:
        message = str(exc)
        if message.startswith("training_score_run_not_found:"):
            raise PackageError("training_score_run_not_found") from exc
        if message.startswith("training_score_run_id_invalid:"):
            raise PackageError("training_score_run_id_invalid") from exc
        if message.startswith("training_score_predictions_not_found:"):
            raise PackageError("training_score_predictions_not_found") from exc
        if message.startswith("training_score_store_index_failed:"):
            raise PackageError("training_score_store_index_failed") from exc
        raise PackageError(message) from exc
    except ValueError as exc:
        raise PackageError(f"training_score_config_invalid:{exc}") from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"training_score_unexpected_error:{exc.__class__.__name__}") from exc

    return ScoreRunResponse(
        run_id=result.run_id,
        predictions_path=str(result.predictions_path),
        results_path=str(result.results_path),
        metrics_path=str(result.metrics_path),
        score_provenance_path=str(result.score_provenance_path),
        requested_stage=result.requested_stage,
        refreshed_stages=list(result.refreshed_stages),
    )
