"""Run and submission API handlers."""

from __future__ import annotations

from contextlib import nullcontext

from numereng.api.contracts import (
    ScoreRunRequest,
    ScoreRunResponse,
    SubmissionRequest,
    SubmissionResponse,
    TrainRunRequest,
    TrainRunResponse,
)
from numereng.features.feature_neutralization import (
    NeutralizationDataError,
    NeutralizationExecutionError,
    NeutralizationValidationError,
)
from numereng.features.submission import (
    SubmissionLiveUniverseUnavailableError,
    SubmissionModelNotFoundError,
    SubmissionPredictionsFileNotFoundError,
    SubmissionPredictionsFormatUnsupportedError,
    SubmissionPredictionsReadError,
    SubmissionRunIdInvalidError,
    SubmissionRunNotFoundError,
    SubmissionRunPredictionsNotFoundError,
    SubmissionRunPredictionsNotLiveEligibleError,
    SubmissionRunPredictionsPathUnsafeError,
)
from numereng.features.telemetry import bind_launch_metadata, get_launch_metadata
from numereng.features.training import (
    TrainingConfigError,
    TrainingDataError,
    TrainingError,
    TrainingMetricsError,
    TrainingModelError,
)
from numereng.platform.errors import NumeraiClientError, PackageError


def submit_predictions(request: SubmissionRequest) -> SubmissionResponse:
    """Submit predictions by file path or run-id artifact lookup."""
    from numereng import api as api_module

    try:
        if request.run_id is not None:
            result = api_module.submit_run_predictions(
                run_id=request.run_id,
                model_name=request.model_name,
                tournament=request.tournament,
                store_root=request.store_root,
                allow_non_live_artifact=request.allow_non_live_artifact,
                neutralize=request.neutralize,
                neutralizer_path=request.neutralizer_path,
                neutralization_proportion=request.neutralization_proportion,
                neutralization_mode=request.neutralization_mode,
                neutralizer_cols=None if request.neutralizer_cols is None else tuple(request.neutralizer_cols),
                neutralization_rank_output=request.neutralization_rank_output,
            )
        else:
            if request.predictions_path is None:  # pragma: no cover - guarded by model validation
                raise PackageError("submission_request_invalid")
            result = api_module.submit_predictions_file(
                predictions_path=request.predictions_path,
                model_name=request.model_name,
                tournament=request.tournament,
                allow_non_live_artifact=request.allow_non_live_artifact,
                neutralize=request.neutralize,
                neutralizer_path=request.neutralizer_path,
                neutralization_proportion=request.neutralization_proportion,
                neutralization_mode=request.neutralization_mode,
                neutralizer_cols=None if request.neutralizer_cols is None else tuple(request.neutralizer_cols),
                neutralization_rank_output=request.neutralization_rank_output,
            )
    except SubmissionModelNotFoundError as exc:
        raise PackageError("submission_model_not_found") from exc
    except SubmissionPredictionsFileNotFoundError as exc:
        raise PackageError("submission_predictions_file_not_found") from exc
    except SubmissionPredictionsFormatUnsupportedError as exc:
        raise PackageError("submission_predictions_format_unsupported") from exc
    except SubmissionPredictionsReadError as exc:
        raise PackageError("submission_predictions_read_failed") from exc
    except SubmissionRunIdInvalidError as exc:
        raise PackageError("submission_run_id_invalid") from exc
    except SubmissionRunNotFoundError as exc:
        raise PackageError("submission_run_not_found") from exc
    except SubmissionRunPredictionsPathUnsafeError as exc:
        raise PackageError("submission_run_predictions_path_unsafe") from exc
    except SubmissionRunPredictionsNotFoundError as exc:
        raise PackageError("submission_run_predictions_not_found") from exc
    except SubmissionRunPredictionsNotLiveEligibleError as exc:
        raise PackageError("submission_run_predictions_not_live_eligible") from exc
    except SubmissionLiveUniverseUnavailableError as exc:
        raise PackageError("submission_live_universe_unavailable") from exc
    except (NeutralizationValidationError, NeutralizationDataError, NeutralizationExecutionError) as exc:
        raise PackageError(str(exc)) from exc
    except ValueError as exc:
        if "submission_neutralizer_path_required" in str(exc):
            raise PackageError("submission_neutralizer_path_required") from exc
        raise PackageError("submission_request_invalid") from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    except Exception as exc:
        raise PackageError(f"submission_unexpected_error:{exc.__class__.__name__}") from exc

    return SubmissionResponse(
        submission_id=result.submission_id,
        model_name=result.model_name,
        model_id=result.model_id,
        predictions_path=str(result.predictions_path),
        run_id=result.run_id,
    )


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
            if request.profile is None:
                result = api_module.run_training_pipeline(
                    config_path=request.config_path,
                    output_dir=request.output_dir,
                    engine_mode=request.engine_mode,
                    window_size_eras=request.window_size_eras,
                    embargo_eras=request.embargo_eras,
                    experiment_id=request.experiment_id,
                )
            else:
                result = api_module.run_training_pipeline(
                    config_path=request.config_path,
                    output_dir=request.output_dir,
                    profile=request.profile,
                    engine_mode=request.engine_mode,
                    window_size_eras=request.window_size_eras,
                    embargo_eras=request.embargo_eras,
                    experiment_id=request.experiment_id,
                )
    except TrainingConfigError as exc:
        raise PackageError("training_config_invalid") from exc
    except TrainingDataError as exc:
        raise PackageError("training_data_load_failed") from exc
    except TrainingModelError as exc:
        message = str(exc)
        if "training_model_backend_missing_lightgbm" in message:
            raise PackageError("training_model_backend_missing") from exc
        raise PackageError("training_model_failed") from exc
    except TrainingMetricsError as exc:
        raise PackageError("training_metrics_failed") from exc
    except TrainingError as exc:
        raise PackageError("training_run_failed") from exc
    except ValueError as exc:
        raise PackageError("training_config_invalid") from exc
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
        raise PackageError("training_score_config_invalid") from exc
    except TrainingDataError as exc:
        raise PackageError("training_score_data_load_failed") from exc
    except TrainingMetricsError as exc:
        raise PackageError("training_score_metrics_failed") from exc
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
        raise PackageError("training_score_failed") from exc
    except ValueError as exc:
        raise PackageError("training_score_config_invalid") from exc
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
        effective_scoring_backend=result.effective_scoring_backend,
        requested_stage=result.requested_stage,
        refreshed_stages=list(result.refreshed_stages),
    )


__all__ = ["run_training", "score_run", "submit_predictions"]
