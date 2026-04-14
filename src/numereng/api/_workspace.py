"""Workspace bootstrap and runtime-sync API handlers."""

from __future__ import annotations

from numereng.api._contracts_ops import (
    WorkspaceInitRequest,
    WorkspaceInitResponse,
    WorkspaceSyncRequest,
    WorkspaceSyncResponse,
)
from numereng.features.workspace import (
    init_workspace as init_workspace_record,
)
from numereng.features.workspace import (
    sync_workspace_environment as sync_workspace_environment_record,
)
from numereng.platform.errors import PackageError


def workspace_init(request: WorkspaceInitRequest | None = None) -> WorkspaceInitResponse:
    """Initialize one canonical numereng workspace scaffold and local runtime."""

    resolved_request = WorkspaceInitRequest() if request is None else request
    try:
        result = init_workspace_record(
            workspace_root=resolved_request.workspace_root,
            runtime_source=resolved_request.runtime_source,
            runtime_path=resolved_request.runtime_path,
            with_training=resolved_request.with_training,
            with_mlops=resolved_request.with_mlops,
        )
    except OSError as exc:
        raise PackageError(f"workspace_init_failed:{exc}") from exc

    return WorkspaceInitResponse(
        workspace_root=str(result.workspace_root),
        store_root=str(result.store_root),
        workspace_project_path=str(result.sync_result.workspace_project_path),
        python_version_path=str(result.sync_result.python_version_path),
        venv_path=str(result.sync_result.venv_path),
        created_paths=[str(path) for path in result.created_paths],
        updated_paths=[str(path) for path in result.sync_result.updated_paths],
        runtime_source=result.sync_result.runtime_source,
        runtime_path=(
            None if result.sync_result.runtime_path is None else str(result.sync_result.runtime_path)
        ),
        extras=list(result.sync_result.extras),
        dependency_spec=result.sync_result.dependency_spec,
        installed_numereng_version=result.sync_result.installed_numereng_version,
        verified_dependencies=list(result.sync_result.verified_dependencies),
        skipped_existing_paths=[str(path) for path in result.skipped_existing_paths],
        installed_skill_ids=list(result.installed_skill_ids),
    )


def workspace_sync(request: WorkspaceSyncRequest | None = None) -> WorkspaceSyncResponse:
    """Ensure one canonical numereng workspace has a healthy local uv runtime."""

    resolved_request = WorkspaceSyncRequest() if request is None else request
    try:
        result = sync_workspace_environment_record(
            workspace_root=resolved_request.workspace_root,
            runtime_source=resolved_request.runtime_source,
            runtime_path=resolved_request.runtime_path,
            with_training=resolved_request.with_training,
            with_mlops=resolved_request.with_mlops,
        )
    except OSError as exc:
        raise PackageError(f"workspace_sync_failed:{exc}") from exc

    return WorkspaceSyncResponse(
        workspace_root=str(result.workspace_root),
        store_root=str(result.store_root),
        workspace_project_path=str(result.workspace_project_path),
        python_version_path=str(result.python_version_path),
        venv_path=str(result.venv_path),
        created_paths=[str(path) for path in result.created_paths],
        updated_paths=[str(path) for path in result.updated_paths],
        runtime_source=result.runtime_source,
        runtime_path=None if result.runtime_path is None else str(result.runtime_path),
        extras=list(result.extras),
        dependency_spec=result.dependency_spec,
        installed_numereng_version=result.installed_numereng_version,
        verified_dependencies=list(result.verified_dependencies),
    )


__all__ = ["workspace_init", "workspace_sync"]
