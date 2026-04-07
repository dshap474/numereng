"""Public pipeline entrypoints composed from internal API stage adapters."""

from __future__ import annotations

from contextlib import nullcontext

from numereng.api._pipeline import (
    cleanup_training_run,
    fail_training_run,
    finalize_training_run,
    load_training_data,
    prepare_training_run,
    train_model,
)
from numereng.api.contracts import TrainRunRequest, TrainRunResponse
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
    return "training_data_load_failed"


def run_training_pipeline(request: TrainRunRequest) -> TrainRunResponse:
    """Run the full local train workflow through explicit API stages."""
    launch_scope = (
        nullcontext()
        if get_launch_metadata() is not None
        else bind_launch_metadata(source="api.pipeline.train", operation_type="run", job_type="run")
    )
    state = None
    try:
        with launch_scope:
            state = prepare_training_run(request)
            state = load_training_data(state)
            state = train_model(state)
            run_id, predictions_path, results_path = finalize_training_run(state)
    except TrainingConfigError as exc:
        if state is not None:
            fail_training_run(state, exc)
        raise PackageError("training_config_invalid") from exc
    except TrainingDataError as exc:
        if state is not None:
            fail_training_run(state, exc)
        raise PackageError(_map_training_data_error(exc)) from exc
    except TrainingModelError as exc:
        if state is not None:
            fail_training_run(state, exc)
        message = str(exc)
        if "training_model_backend_missing_lightgbm" in message:
            raise PackageError("training_model_backend_missing") from exc
        raise PackageError("training_model_failed") from exc
    except TrainingMetricsError as exc:
        if state is not None:
            fail_training_run(state, exc)
        raise PackageError("training_metrics_failed") from exc
    except TrainingCanceledError as exc:
        if state is not None:
            fail_training_run(state, exc)
        raise PackageError("training_run_canceled") from exc
    except TrainingError as exc:
        if state is not None:
            fail_training_run(state, exc)
        if str(exc) == "training_launch_metadata_missing":
            raise PackageError("training_launch_metadata_missing") from exc
        if str(exc).startswith("training_lifecycle_bootstrap_failed:"):
            raise PackageError("training_lifecycle_bootstrap_failed") from exc
        raise PackageError("training_run_failed") from exc
    except ValueError as exc:
        if state is not None:
            fail_training_run(state, exc)
        raise PackageError("training_config_invalid") from exc
    except NumeraiClientError as exc:
        if state is not None:
            fail_training_run(state, exc)
        raise PackageError(str(exc)) from exc
    except PackageError:
        raise
    except Exception as exc:
        if state is not None:
            fail_training_run(state, exc)
        raise PackageError(f"training_unexpected_error:{exc.__class__.__name__}") from exc
    finally:
        if state is not None:
            cleanup_training_run(state)

    return TrainRunResponse(
        run_id=run_id,
        predictions_path=str(predictions_path),
        results_path=str(results_path),
    )


__all__ = ["run_training_pipeline"]
