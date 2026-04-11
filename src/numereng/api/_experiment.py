"""Experiment API handlers."""

from __future__ import annotations

from contextlib import nullcontext

from numereng.api.contracts import (
    ExperimentArchiveRequest,
    ExperimentArchiveResponse,
    ExperimentCreateRequest,
    ExperimentGetRequest,
    ExperimentListRequest,
    ExperimentListResponse,
    ExperimentPackRequest,
    ExperimentPackResponse,
    ExperimentPromoteRequest,
    ExperimentPromoteResponse,
    ExperimentReportRequest,
    ExperimentReportResponse,
    ExperimentReportRowResponse,
    ExperimentResponse,
    ExperimentRunPlanRequest,
    ExperimentRunPlanResponse,
    ExperimentScoreRoundRequest,
    ExperimentScoreRoundResponse,
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
    TrainingCanceledError,
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


def experiment_archive(request: ExperimentArchiveRequest) -> ExperimentArchiveResponse:
    """Archive one experiment and move it under the archive root."""
    from numereng import api as api_module

    try:
        result = api_module.archive_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentArchiveResponse(
        experiment_id=result.experiment_id,
        status=result.status,
        manifest_path=str(result.manifest_path),
        archived=result.archived,
    )


def experiment_unarchive(request: ExperimentArchiveRequest) -> ExperimentArchiveResponse:
    """Restore one archived experiment to the live experiment root."""
    from numereng import api as api_module

    try:
        result = api_module.unarchive_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentArchiveResponse(
        experiment_id=result.experiment_id,
        status=result.status,
        manifest_path=str(result.manifest_path),
        archived=result.archived,
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
                    post_training_scoring=request.post_training_scoring,
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
                    post_training_scoring=request.post_training_scoring,
                    engine_mode=request.engine_mode,
                    window_size_eras=request.window_size_eras,
                    embargo_eras=request.embargo_eras,
                )
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


def experiment_run_plan(request: ExperimentRunPlanRequest) -> ExperimentRunPlanResponse:
    """Execute one source-owned experiment run_plan window."""
    from numereng import api as api_module

    try:
        result = api_module.run_experiment_plan_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            start_index=request.start_index,
            end_index=request.end_index,
            score_stage=request.score_stage,
            resume=request.resume,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentRunPlanResponse(
        experiment_id=result.experiment_id,
        state_path=str(result.state_path),
        window={
            "start_index": result.window.start_index,
            "end_index": result.window.end_index,
            "total_rows": result.window.total_rows,
        },
        phase=result.phase,
        requested_score_stage=result.requested_score_stage,
        completed_score_stages=list(result.completed_score_stages),
        current_index=result.current_index,
        current_round=result.current_round,
        current_config_path=str(result.current_config_path) if result.current_config_path is not None else None,
        current_run_id=result.current_run_id,
        last_completed_row_index=result.last_completed_row_index,
        supervisor_pid=result.supervisor_pid,
        active_worker_pid=result.active_worker_pid,
        last_successful_heartbeat_at=result.last_successful_heartbeat_at,
        failure_classifier=result.failure_classifier,
        retry_count=result.retry_count,
        terminal_error=result.terminal_error,
        updated_at=result.updated_at,
    )


def experiment_score_round(request: ExperimentScoreRoundRequest) -> ExperimentScoreRoundResponse:
    """Deferred-score one experiment round."""
    from numereng import api as api_module

    try:
        result = api_module.score_experiment_round_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            round=request.round,
            stage=request.stage,
        )
    except (ExperimentNotFoundError, ExperimentRunNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc
    except (
        TrainingConfigError,
        TrainingDataError,
        TrainingMetricsError,
        TrainingError,
    ) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentScoreRoundResponse(
        experiment_id=result.experiment_id,
        round=result.round,
        stage=result.stage,
        run_ids=list(result.run_ids),
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


def experiment_pack(request: ExperimentPackRequest) -> ExperimentPackResponse:
    """Write one experiment markdown pack artifact."""
    from numereng import api as api_module

    try:
        result = api_module.pack_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc

    return ExperimentPackResponse(
        experiment_id=result.experiment_id,
        output_path=str(result.output_path),
        experiment_path=str(result.experiment_path),
        source_markdown_path=str(result.source_markdown_path),
        run_count=result.run_count,
        packed_at=result.packed_at,
    )


__all__ = [
    "experiment_archive",
    "experiment_create",
    "experiment_get",
    "experiment_list",
    "experiment_pack",
    "experiment_promote",
    "experiment_report",
    "experiment_run_plan",
    "experiment_score_round",
    "experiment_train",
    "experiment_unarchive",
]
