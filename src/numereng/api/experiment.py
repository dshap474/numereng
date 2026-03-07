"""Experiment API handlers."""

from __future__ import annotations

from contextlib import nullcontext

from numereng.api.contracts import (
    ExperimentCreateRequest,
    ExperimentGetRequest,
    ExperimentListRequest,
    ExperimentListResponse,
    ExperimentPromoteRequest,
    ExperimentPromoteResponse,
    ExperimentReportRequest,
    ExperimentReportResponse,
    ExperimentReportRowResponse,
    ExperimentResponse,
    ExperimentTrainRequest,
    ExperimentTrainResponse,
)
from numereng.features.experiments import (
    ExperimentAlreadyExistsError,
    ExperimentError,
    ExperimentNotFoundError,
    ExperimentRunNotFoundError,
    ExperimentValidationError,
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


def experiment_create(request: ExperimentCreateRequest) -> ExperimentResponse:
    """Create one experiment and index metadata."""
    from numereng import api as api_module

    try:
        record = api_module.create_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            name=request.name,
            hypothesis=request.hypothesis,
            tags=request.tags,
        )
    except (ExperimentAlreadyExistsError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentResponse(
        experiment_id=record.experiment_id,
        name=record.name,
        status=record.status,
        hypothesis=record.hypothesis,
        tags=list(record.tags),
        created_at=record.created_at,
        updated_at=record.updated_at,
        champion_run_id=record.champion_run_id,
        runs=list(record.runs),
        metadata=record.metadata,
        manifest_path=str(record.manifest_path),
    )


def experiment_list(request: ExperimentListRequest | None = None) -> ExperimentListResponse:
    """List experiments from manifest storage."""
    from numereng import api as api_module

    resolved_request = ExperimentListRequest() if request is None else request
    try:
        records = api_module.list_experiment_records(
            store_root=resolved_request.store_root,
            status=resolved_request.status,
        )
    except ExperimentError as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentListResponse(
        experiments=[
            ExperimentResponse(
                experiment_id=record.experiment_id,
                name=record.name,
                status=record.status,
                hypothesis=record.hypothesis,
                tags=list(record.tags),
                created_at=record.created_at,
                updated_at=record.updated_at,
                champion_run_id=record.champion_run_id,
                runs=list(record.runs),
                metadata=record.metadata,
                manifest_path=str(record.manifest_path),
            )
            for record in records
        ]
    )


def experiment_get(request: ExperimentGetRequest) -> ExperimentResponse:
    """Load one experiment by ID."""
    from numereng import api as api_module

    try:
        record = api_module.get_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentResponse(
        experiment_id=record.experiment_id,
        name=record.name,
        status=record.status,
        hypothesis=record.hypothesis,
        tags=list(record.tags),
        created_at=record.created_at,
        updated_at=record.updated_at,
        champion_run_id=record.champion_run_id,
        runs=list(record.runs),
        metadata=record.metadata,
        manifest_path=str(record.manifest_path),
    )


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
            if request.profile is None:
                result = api_module.train_experiment_record(
                    store_root=request.store_root,
                    experiment_id=request.experiment_id,
                    config_path=request.config_path,
                    output_dir=request.output_dir,
                    engine_mode=request.engine_mode,
                    window_size_eras=request.window_size_eras,
                    embargo_eras=request.embargo_eras,
                )
            else:
                result = api_module.train_experiment_record(
                    store_root=request.store_root,
                    experiment_id=request.experiment_id,
                    config_path=request.config_path,
                    output_dir=request.output_dir,
                    profile=request.profile,
                    engine_mode=request.engine_mode,
                    window_size_eras=request.window_size_eras,
                    embargo_eras=request.embargo_eras,
                )
    except (
        ExperimentNotFoundError,
        ExperimentValidationError,
        ExperimentError,
        TrainingConfigError,
        TrainingDataError,
        TrainingModelError,
        TrainingMetricsError,
        TrainingError,
    ) as exc:
        message = str(exc)
        if "training_model_backend_missing_lightgbm" in message:
            raise PackageError("training_model_backend_missing") from exc
        raise PackageError(message) from exc
    except ValueError as exc:
        raise PackageError("training_config_invalid") from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentTrainResponse(
        experiment_id=result.experiment_id,
        run_id=result.run_id,
        predictions_path=str(result.predictions_path),
        results_path=str(result.results_path),
    )


def experiment_promote(request: ExperimentPromoteRequest) -> ExperimentPromoteResponse:
    """Promote one experiment run to champion."""
    from numereng import api as api_module

    try:
        result = api_module.promote_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            run_id=request.run_id,
            metric=request.metric,
        )
    except (ExperimentNotFoundError, ExperimentRunNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentPromoteResponse(
        experiment_id=result.experiment_id,
        champion_run_id=result.champion_run_id,
        metric=result.metric,
        metric_value=result.metric_value,
        auto_selected=result.auto_selected,
    )


def experiment_report(request: ExperimentReportRequest) -> ExperimentReportResponse:
    """Build one ranked experiment report payload."""
    from numereng import api as api_module

    try:
        report = api_module.report_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            metric=request.metric,
            limit=request.limit,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentReportResponse(
        experiment_id=report.experiment_id,
        metric=report.metric,
        total_runs=report.total_runs,
        champion_run_id=report.champion_run_id,
        rows=[
            ExperimentReportRowResponse(
                run_id=row.run_id,
                status=row.status,
                created_at=row.created_at,
                metric_value=row.metric_value,
                corr_mean=row.corr_mean,
                mmc_mean=row.mmc_mean,
                cwmm_mean=row.cwmm_mean,
                bmc_mean=row.bmc_mean,
                bmc_last_200_eras_mean=row.bmc_last_200_eras_mean,
                is_champion=row.is_champion,
            )
            for row in report.rows
        ],
    )


__all__ = [
    "experiment_create",
    "experiment_get",
    "experiment_list",
    "experiment_promote",
    "experiment_report",
    "experiment_train",
]
