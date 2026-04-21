"""Target discovery and bootstrap API handlers."""

from __future__ import annotations

import numereng.api._remote as remote_api_module

from numereng.api._remote.mappers import bootstrap_target_response, target_response
from numereng.api.contracts import (
    RemoteDoctorRequest,
    RemoteDoctorResponse,
    RemoteTargetListRequest,
    RemoteTargetListResponse,
    RemoteVizBootstrapRequest,
    RemoteVizBootstrapResponse,
)
from numereng.platform.errors import PackageError


def remote_list_targets(request: RemoteTargetListRequest | None = None) -> RemoteTargetListResponse:
    _ = request
    try:
        targets = remote_api_module.list_remote_targets_record()
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteTargetListResponse(targets=[target_response(target) for target in targets])


def remote_doctor(request: RemoteDoctorRequest) -> RemoteDoctorResponse:
    try:
        result = remote_api_module.doctor_remote_target_record(target_id=request.target_id)
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteDoctorResponse(
        target=target_response(result.target),
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
        result = remote_api_module.bootstrap_viz_remotes_record(store_root=request.store_root)
    except Exception as exc:
        raise PackageError(str(exc)) from exc
    return RemoteVizBootstrapResponse(
        store_root=str(result.store_root),
        state_path=str(result.state_path),
        bootstrapped_at=result.bootstrapped_at,
        ready_count=result.ready_count,
        degraded_count=result.degraded_count,
        targets=[bootstrap_target_response(item) for item in result.targets],
    )
