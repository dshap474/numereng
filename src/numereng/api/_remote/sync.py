"""Remote sync and pullback API handlers."""

from __future__ import annotations

import numereng.api._remote as remote_api_module

from numereng.api._remote.mappers import pull_failure_response
from numereng.api.contracts import (
    RemoteConfigPushRequest,
    RemoteConfigPushResponse,
    RemoteExperimentPullRequest,
    RemoteExperimentPullResponse,
    RemoteExperimentSyncRequest,
    RemoteExperimentSyncResponse,
    RemoteRepoSyncRequest,
    RemoteRepoSyncResponse,
)
from numereng.platform.errors import PackageError


def remote_repo_sync(request: RemoteRepoSyncRequest) -> RemoteRepoSyncResponse:
    try:
        result = remote_api_module.sync_remote_repo_record(target_id=request.target_id, store_root=request.store_root)
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
        result = remote_api_module.sync_remote_experiment_record(
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
        result = remote_api_module.pull_remote_experiment_record(
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
        failures=[pull_failure_response(item) for item in result.failures],
    )


def remote_config_push(request: RemoteConfigPushRequest) -> RemoteConfigPushResponse:
    try:
        result = remote_api_module.push_remote_config_record(
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
