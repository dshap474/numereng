"""Remote SSH ops API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    RemoteConfigPushRequest,
    RemoteConfigPushResponse,
    RemoteDoctorRequest,
    RemoteDoctorResponse,
    RemoteExperimentLaunchRequest,
    RemoteExperimentLaunchResponse,
    RemoteExperimentMaintainRequest,
    RemoteExperimentMaintainResponse,
    RemoteExperimentPullFailureResponse,
    RemoteExperimentPullRequest,
    RemoteExperimentPullResponse,
    RemoteExperimentStatusRequest,
    RemoteExperimentStatusResponse,
    RemoteExperimentStopRequest,
    RemoteExperimentStopResponse,
    RemoteExperimentSyncRequest,
    RemoteExperimentSyncResponse,
    RemoteRepoSyncRequest,
    RemoteRepoSyncResponse,
    RemoteTargetListRequest,
    RemoteTargetListResponse,
    RemoteTargetResponse,
    RemoteTrainLaunchRequest,
    RemoteTrainLaunchResponse,
    RemoteVizBootstrapRequest,
    RemoteVizBootstrapResponse,
    RemoteVizBootstrapTargetResponse,
)
from numereng.features.remote_ops import (
    bootstrap_viz_remotes as bootstrap_viz_remotes_record,
)
from numereng.features.remote_ops import (
    doctor_remote_target as doctor_remote_target_record,
)
from numereng.features.remote_ops import (
    list_remote_targets as list_remote_targets_record,
)
from numereng.features.remote_ops import (
    pull_remote_experiment as pull_remote_experiment_record,
)
from numereng.features.remote_ops import (
    push_remote_config as push_remote_config_record,
)
from numereng.features.remote_ops import (
    remote_experiment_status as remote_experiment_status_record,
)
from numereng.features.remote_ops import (
    remote_launch_experiment as remote_launch_experiment_record,
)
from numereng.features.remote_ops import (
    remote_maintain_experiment as remote_maintain_experiment_record,
)
from numereng.features.remote_ops import (
    remote_run_train as remote_run_train_record,
)
from numereng.features.remote_ops import (
    remote_stop_experiment as remote_stop_experiment_record,
)
from numereng.features.remote_ops import (
    sync_remote_experiment as sync_remote_experiment_record,
)
from numereng.features.remote_ops import (
    sync_remote_repo as sync_remote_repo_record,
)
from numereng.platform.errors import PackageError


def remote_list_targets(request: RemoteTargetListRequest | None = None) -> RemoteTargetListResponse:
    _ = request
    try:
        targets = list_remote_targets_record()
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteTargetListResponse(
        targets=[
            RemoteTargetResponse(
                id=target.id,
                label=target.label,
                kind=target.kind,
                shell=target.shell,
                repo_root=target.repo_root,
                store_root=target.store_root,
                runner_cmd=target.runner_cmd,
                python_cmd=target.python_cmd,
                tags=list(target.tags),
            )
            for target in targets
        ]
    )


def remote_doctor(request: RemoteDoctorRequest) -> RemoteDoctorResponse:
    try:
        result = doctor_remote_target_record(target_id=request.target_id)
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteDoctorResponse(
        target=RemoteTargetResponse(
            id=result.target.id,
            label=result.target.label,
            kind=result.target.kind,
            shell=result.target.shell,
            repo_root=result.target.repo_root,
            store_root=result.target.store_root,
            runner_cmd=result.target.runner_cmd,
            python_cmd=result.target.python_cmd,
            tags=list(result.target.tags),
        ),
        ok=result.ok,
        checked_at=result.checked_at,
        remote_python_executable=result.remote_python_executable,
        remote_cwd=result.remote_cwd,
        snapshot_ok=result.snapshot_ok,
        snapshot_source_kind=result.snapshot_source_kind,
        snapshot_source_id=result.snapshot_source_id,
        issues=list(result.issues),
    )


def remote_bootstrap_viz(request: RemoteVizBootstrapRequest) -> RemoteVizBootstrapResponse:
    try:
        result = bootstrap_viz_remotes_record(store_root=request.store_root)
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteVizBootstrapResponse(
        store_root=str(result.store_root),
        state_path=str(result.state_path),
        bootstrapped_at=result.bootstrapped_at,
        ready_count=result.ready_count,
        degraded_count=result.degraded_count,
        targets=[
            RemoteVizBootstrapTargetResponse(
                target=RemoteTargetResponse(
                    id=item.target.id,
                    label=item.target.label,
                    kind=item.target.kind,
                    shell=item.target.shell,
                    repo_root=item.target.repo_root,
                    store_root=item.target.store_root,
                    runner_cmd=item.target.runner_cmd,
                    python_cmd=item.target.python_cmd,
                    tags=list(item.target.tags),
                ),
                bootstrap_status=item.bootstrap_status,
                last_bootstrap_at=item.last_bootstrap_at,
                last_bootstrap_error=item.last_bootstrap_error,
                repo_synced=item.repo_synced,
                repo_sync_skipped=item.repo_sync_skipped,
                doctor_ok=item.doctor_ok,
                issues=list(item.issues),
            )
            for item in result.targets
        ],
    )


def remote_repo_sync(request: RemoteRepoSyncRequest) -> RemoteRepoSyncResponse:
    try:
        result = sync_remote_repo_record(target_id=request.target_id, store_root=request.store_root)
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteRepoSyncResponse(
        target_id=result.target_id,
        repo_root=result.repo_root,
        manifest_hash=result.manifest_hash,
        local_commit_sha=result.local_commit_sha,
        dirty=result.dirty,
        synced_files=result.synced_files,
        deleted_files=result.deleted_files,
        synced_at=result.synced_at,
        local_marker_path=str(result.local_marker_path),
        remote_marker_path=result.remote_marker_path,
    )


def remote_experiment_sync(request: RemoteExperimentSyncRequest) -> RemoteExperimentSyncResponse:
    try:
        result = sync_remote_experiment_record(
            target_id=request.target_id,
            experiment_id=request.experiment_id,
            store_root=request.store_root,
        )
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteExperimentSyncResponse(
        target_id=result.target_id,
        experiment_id=result.experiment_id,
        remote_experiment_dir=result.remote_experiment_dir,
        manifest_hash=result.manifest_hash,
        synced_files=result.synced_files,
        deleted_files=result.deleted_files,
        synced_at=result.synced_at,
        local_marker_path=str(result.local_marker_path),
        remote_marker_path=result.remote_marker_path,
    )


def remote_experiment_pull(request: RemoteExperimentPullRequest) -> RemoteExperimentPullResponse:
    try:
        result = pull_remote_experiment_record(
            target_id=request.target_id,
            experiment_id=request.experiment_id,
            mode=request.mode,
            store_root=request.store_root,
        )
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteExperimentPullResponse(
        target_id=result.target_id,
        experiment_id=result.experiment_id,
        pull_mode=result.pull_mode,  # type: ignore[arg-type]
        local_experiment_manifest_path=str(result.local_experiment_manifest_path),
        local_runs_root=str(result.local_runs_root),
        pulled_at=result.pulled_at,
        already_materialized_run_ids=list(result.already_materialized_run_ids),
        materialized_run_ids=list(result.materialized_run_ids),
        partially_materialized_run_ids=list(result.partially_materialized_run_ids),
        materialized_run_count=result.materialized_run_count,
        skipped_non_finished_run_ids=list(result.skipped_non_finished_run_ids),
        failures=[
            RemoteExperimentPullFailureResponse(
                run_id=item.run_id,
                missing_files=list(item.missing_files),
                reason=item.reason,
            )
            for item in result.failures
        ],
    )


def remote_experiment_launch(request: RemoteExperimentLaunchRequest) -> RemoteExperimentLaunchResponse:
    try:
        result = remote_launch_experiment_record(
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
        result = remote_experiment_status_record(
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
        result = remote_stop_experiment_record(
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
        result = remote_maintain_experiment_record(
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


def remote_config_push(request: RemoteConfigPushRequest) -> RemoteConfigPushResponse:
    try:
        result = push_remote_config_record(
            target_id=request.target_id,
            config_path=request.config_path,
            store_root=request.store_root,
        )
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteConfigPushResponse(
        target_id=result.target_id,
        local_config_path=str(result.local_config_path),
        remote_config_path=result.remote_config_path,
        synced_at=result.synced_at,
    )


def remote_train_launch(request: RemoteTrainLaunchRequest) -> RemoteTrainLaunchResponse:
    try:
        result = remote_run_train_record(
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


__all__ = [
    "remote_experiment_launch",
    "remote_experiment_maintain",
    "remote_config_push",
    "remote_bootstrap_viz",
    "remote_doctor",
    "remote_experiment_status",
    "remote_experiment_stop",
    "remote_experiment_pull",
    "remote_experiment_sync",
    "remote_list_targets",
    "remote_repo_sync",
    "remote_train_launch",
]
