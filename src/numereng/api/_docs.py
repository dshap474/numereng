"""Docs sync API handlers."""

from __future__ import annotations

from numereng.api._contracts_ops import DocsSyncRequest, DocsSyncResponse
from numereng.features.docs_sync import sync_numerai_docs
from numereng.platform.errors import PackageError


def sync_docs(request: DocsSyncRequest | None = None) -> DocsSyncResponse:
    """Download one supported upstream docs mirror into the target workspace."""

    resolved_request = DocsSyncRequest() if request is None else request
    try:
        result = sync_numerai_docs(workspace_root=resolved_request.workspace_root)
    except (OSError, RuntimeError) as exc:
        raise PackageError(f"docs_sync_failed:{exc}") from exc

    return DocsSyncResponse(
        workspace_root=str(result.workspace_root),
        destination_root=str(result.destination_root),
        sync_meta_path=str(result.sync_meta_path),
        upstream_commit=result.upstream_commit,
        synced_at=result.synced_at,
        synced_files=result.synced_files,
    )


__all__ = ["sync_docs"]
