"""Packaged viz app API handlers."""

from __future__ import annotations

from fastapi import FastAPI

from numereng.api._contracts_viz import VizAppRequest
from numereng.platform.errors import PackageError


def create_viz_app(request: VizAppRequest | None = None) -> FastAPI:
    """Create one packaged viz app bound to the requested workspace."""

    from numereng_viz import create_app as create_viz_app_record

    resolved_request = VizAppRequest() if request is None else request
    try:
        return create_viz_app_record(workspace_root=resolved_request.workspace_root)
    except Exception as exc:
        raise PackageError(f"viz_create_app_failed:{exc}") from exc


__all__ = ["create_viz_app"]
