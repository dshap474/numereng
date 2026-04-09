"""Workspace bootstrap API handlers."""

from __future__ import annotations

from numereng.api.contracts import WorkspaceInitRequest, WorkspaceInitResponse
from numereng.features.workspace import init_workspace as init_workspace_record
from numereng.platform.errors import PackageError


def workspace_init(request: WorkspaceInitRequest | None = None) -> WorkspaceInitResponse:
    """Initialize one canonical numereng workspace scaffold."""

    resolved_request = WorkspaceInitRequest() if request is None else request
    try:
        result = init_workspace_record(workspace_root=resolved_request.workspace_root)
    except OSError as exc:
        raise PackageError(f"workspace_init_failed:{exc}") from exc

    return WorkspaceInitResponse(
        workspace_root=str(result.workspace_root),
        store_root=str(result.store_root),
        created_paths=[str(path) for path in result.created_paths],
        skipped_existing_paths=[str(path) for path in result.skipped_existing_paths],
        installed_skill_ids=list(result.installed_skill_ids),
    )


__all__ = ["workspace_init"]
