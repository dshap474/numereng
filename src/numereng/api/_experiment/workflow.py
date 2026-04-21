"""Experiment workflow API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    ExperimentPromoteRequest,
    ExperimentPromoteResponse,
    ExperimentRunPlanRequest,
    ExperimentRunPlanResponse,
    ExperimentScoreRoundRequest,
    ExperimentScoreRoundResponse,
)
from numereng.features.experiments import (
    ExperimentError,
    ExperimentNotFoundError,
    ExperimentRunNotFoundError,
    ExperimentValidationError,
)
from numereng.features.training import TrainingConfigError, TrainingDataError, TrainingError, TrainingMetricsError
from numereng.platform.errors import PackageError


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
    except (TrainingConfigError, TrainingDataError, TrainingMetricsError, TrainingError) as exc:
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
