"""Experiment-linked training API handlers."""

from __future__ import annotations

from contextlib import nullcontext

from numereng.api.contracts import ExperimentTrainRequest, ExperimentTrainResponse
from numereng.features.experiments import ExperimentError, ExperimentNotFoundError, ExperimentValidationError
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


def experiment_train(request: ExperimentTrainRequest) -> ExperimentTrainResponse:
    """Run one training job linked to an experiment."""
    from numereng import api as api_module

    launch_scope = (
        nullcontext()
        if get_launch_metadata() is not None
        else bind_launch_metadata(source="api.experiment.train", operation_type="run", job_type="run")
    )
    try:
        with launch_scope:
            kwargs = {
                "store_root": request.store_root,
                "experiment_id": request.experiment_id,
                "config_path": request.config_path,
                "output_dir": request.output_dir,
                "post_training_scoring": request.post_training_scoring,
                "engine_mode": request.engine_mode,
                "window_size_eras": request.window_size_eras,
                "embargo_eras": request.embargo_eras,
            }
            if request.profile is not None:
                kwargs["profile"] = request.profile
            result = api_module.train_experiment_record(**kwargs)
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc
    except TrainingConfigError as exc:
        raise PackageError(str(exc)) from exc
    except TrainingDataError as exc:
        raise PackageError(str(exc)) from exc
    except TrainingModelError as exc:
        message = str(exc)
        if "training_model_backend_missing_lightgbm" in message:
            raise PackageError("training_model_backend_missing") from exc
        raise PackageError(message) from exc
    except TrainingMetricsError as exc:
        raise PackageError(str(exc)) from exc
    except TrainingCanceledError as exc:
        raise PackageError("training_run_canceled") from exc
    except TrainingError as exc:
        message = str(exc)
        if message.startswith("training_lifecycle_bootstrap_failed:"):
            raise PackageError("training_lifecycle_bootstrap_failed") from exc
        if message.startswith("training_run_failed:"):
            raise PackageError(message) from exc
        raise PackageError(f"training_run_failed:{message}") from exc
    except ValueError as exc:
        raise PackageError(f"training_config_invalid:{exc}") from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentTrainResponse(
        experiment_id=result.experiment_id,
        run_id=result.run_id,
        predictions_path=str(result.predictions_path),
        results_path=str(result.results_path),
    )
