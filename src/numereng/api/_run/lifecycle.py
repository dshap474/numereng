"""Run lifecycle API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    RunCancelRequest,
    RunCancelResponse,
    RunLifecycleRequest,
    RunLifecycleResponse,
)
from numereng.features.store import StoreError
from numereng.platform.errors import PackageError


def get_run_lifecycle(request: RunLifecycleRequest) -> RunLifecycleResponse:
    """Return current canonical lifecycle summary for one run."""
    from numereng import api as api_module

    try:
        result = api_module.get_run_lifecycle_record(store_root=request.store_root, run_id=request.run_id)
    except StoreError as exc:
        raise PackageError(str(exc)) from exc
    if result is None:
        raise PackageError("run_lifecycle_not_found")

    return RunLifecycleResponse(
        run_id=result.run_id,
        run_hash=result.run_hash,
        config_hash=result.config_hash,
        job_id=result.job_id,
        logical_run_id=result.logical_run_id,
        attempt_id=result.attempt_id,
        attempt_no=result.attempt_no,
        source=result.source,
        operation_type=result.operation_type,
        job_type=result.job_type,
        status=result.status,
        experiment_id=result.experiment_id,
        config_id=result.config_id,
        config_source=result.config_source,
        config_path=result.config_path,
        config_sha256=result.config_sha256,
        run_dir=result.run_dir,
        runtime_path=result.runtime_path,
        backend=result.backend,
        worker_id=result.worker_id,
        pid=result.pid,
        host=result.host,
        current_stage=result.current_stage,
        completed_stages=list(result.completed_stages),
        progress_percent=result.progress_percent,
        progress_label=result.progress_label,
        progress_current=result.progress_current,
        progress_total=result.progress_total,
        cancel_requested=result.cancel_requested,
        cancel_requested_at=result.cancel_requested_at,
        created_at=result.created_at,
        queued_at=result.queued_at,
        started_at=result.started_at,
        last_heartbeat_at=result.last_heartbeat_at,
        updated_at=result.updated_at,
        finished_at=result.finished_at,
        terminal_reason=result.terminal_reason,
        terminal_detail=result.terminal_detail,
        latest_metrics=result.latest_metrics,
        latest_sample=result.latest_sample,
        reconciled=result.reconciled,
    )


def cancel_run(request: RunCancelRequest) -> RunCancelResponse:
    """Request cooperative cancel for one active run."""
    from numereng import api as api_module

    try:
        result = api_module.request_run_cancel_record(store_root=request.store_root, run_id=request.run_id)
    except StoreError as exc:
        raise PackageError(str(exc)) from exc

    return RunCancelResponse(
        run_id=result.run_id,
        job_id=result.job_id,
        status=result.status,
        cancel_requested=result.cancel_requested,
        cancel_requested_at=result.cancel_requested_at,
        accepted=result.accepted,
    )
