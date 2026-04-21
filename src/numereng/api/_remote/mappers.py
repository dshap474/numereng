"""Shared response mappers for remote SSH ops API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    RemoteExperimentPullFailureResponse,
    RemoteTargetResponse,
    RemoteVizBootstrapTargetResponse,
)


def target_response(target: object) -> RemoteTargetResponse:
    return RemoteTargetResponse(
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


def bootstrap_target_response(item: object) -> RemoteVizBootstrapTargetResponse:
    return RemoteVizBootstrapTargetResponse(
        target=target_response(item.target),
        bootstrap_status=item.bootstrap_status,
        last_bootstrap_at=item.last_bootstrap_at,
        last_bootstrap_error=item.last_bootstrap_error,
        repo_synced=item.repo_synced,
        repo_sync_skipped=item.repo_sync_skipped,
        doctor_ok=item.doctor_ok,
        issues=list(item.issues),
    )


def pull_failure_response(item: object) -> RemoteExperimentPullFailureResponse:
    return RemoteExperimentPullFailureResponse(
        run_id=item.run_id,
        missing_files=list(item.missing_files),
        reason=item.reason,
    )
