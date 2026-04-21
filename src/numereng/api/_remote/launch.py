"""Remote launch, status, and maintenance API handlers."""

from __future__ import annotations

import numereng.api._remote as remote_api_module

from numereng.api.contracts import (
    RemoteExperimentLaunchRequest,
    RemoteExperimentLaunchResponse,
    RemoteExperimentMaintainRequest,
    RemoteExperimentMaintainResponse,
    RemoteExperimentStatusRequest,
    RemoteExperimentStatusResponse,
    RemoteExperimentStopRequest,
    RemoteExperimentStopResponse,
    RemoteTrainLaunchRequest,
    RemoteTrainLaunchResponse,
)
from numereng.platform.errors import PackageError


def remote_experiment_launch(request: RemoteExperimentLaunchRequest) -> RemoteExperimentLaunchResponse:
    try:
        result = remote_api_module.remote_launch_experiment_record(
            target_id=request.target_id,
            experiment_id=request.experiment_id,
            start_index=request.start_index,
            end_index=request.end_index,
            score_stage=request.score_stage,
            sync_repo=request.sync_repo,
            store_root=request.store_root,
        )
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteExperimentLaunchResponse(
        target_id=result.target_id,
        experiment_id=result.experiment_id,
        state_path=result.state_path,
        launch_id=result.launch_id,
        remote_log_path=result.remote_log_path,
        remote_metadata_path=result.remote_metadata_path,
        remote_pid=result.remote_pid,
        launched_at=result.launched_at,
        repo_synced=result.repo_synced,
        experiment_synced=result.experiment_synced,
    )


def remote_experiment_status(request: RemoteExperimentStatusRequest) -> RemoteExperimentStatusResponse:
    try:
        result = remote_api_module.remote_experiment_status_record(
            target_id=request.target_id,
            experiment_id=request.experiment_id,
            start_index=request.start_index,
            end_index=request.end_index,
            store_root=request.store_root,
        )
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteExperimentStatusResponse(
        target_id=result.target_id,
        experiment_id=result.experiment_id,
        state_path=result.state_path,
        exists=result.exists,
        phase=result.phase,
        current_index=result.current_index,
        current_run_id=result.current_run_id,
        current_config_path=result.current_config_path,
        last_completed_row_index=result.last_completed_row_index,
        supervisor_pid=result.supervisor_pid,
        supervisor_alive=result.supervisor_alive,
        active_worker_pid=result.active_worker_pid,
        last_successful_heartbeat_at=result.last_successful_heartbeat_at,
        retry_count=result.retry_count,
        failure_classifier=result.failure_classifier,
        terminal_error=result.terminal_error,
        raw_state=result.raw_state,
    )


def remote_experiment_stop(request: RemoteExperimentStopRequest) -> RemoteExperimentStopResponse:
    try:
        result = remote_api_module.remote_stop_experiment_record(
            target_id=request.target_id,
            experiment_id=request.experiment_id,
            start_index=request.start_index,
            end_index=request.end_index,
            store_root=request.store_root,
        )
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteExperimentStopResponse(
        target_id=result.target_id,
        experiment_id=result.experiment_id,
        state_path=result.state_path,
        stopped=result.stopped,
        supervisor_pid=result.supervisor_pid,
        note=result.note,
    )


def remote_experiment_maintain(request: RemoteExperimentMaintainRequest) -> RemoteExperimentMaintainResponse:
    try:
        result = remote_api_module.remote_maintain_experiment_record(
            target_id=request.target_id,
            experiment_id=request.experiment_id,
            start_index=request.start_index,
            end_index=request.end_index,
            store_root=request.store_root,
        )
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteExperimentMaintainResponse(
        target_id=result.target_id,
        experiment_id=result.experiment_id,
        state_path=result.state_path,
        action=result.action,
        phase=result.phase,
        supervisor_pid=result.supervisor_pid,
        note=result.note,
    )


def remote_train_launch(request: RemoteTrainLaunchRequest) -> RemoteTrainLaunchResponse:
    try:
        result = remote_api_module.remote_run_train_record(
            target_id=request.target_id,
            config_path=request.config_path,
            experiment_id=request.experiment_id,
            sync_repo=request.sync_repo,
            profile=request.profile,
            post_training_scoring=request.post_training_scoring,
            store_root=request.store_root,
        )
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteTrainLaunchResponse(
        target_id=result.target_id,
        launch_id=result.launch_id,
        remote_config_path=result.remote_config_path,
        remote_log_path=result.remote_log_path,
        remote_metadata_path=result.remote_metadata_path,
        remote_pid=result.remote_pid,
        launched_at=result.launched_at,
        sync_repo_policy=result.sync_repo_policy,
        repo_synced=result.repo_synced,
        experiment_synced=result.experiment_synced,
    )
