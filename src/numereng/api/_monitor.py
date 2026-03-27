"""Monitor snapshot API handlers."""

from __future__ import annotations

from numereng.api.contracts import MonitorSnapshotRequest, MonitorSnapshotResponse
from numereng.platform.errors import PackageError


def build_monitor_snapshot(request: MonitorSnapshotRequest | None = None) -> MonitorSnapshotResponse:
    """Build one normalized read-only monitor snapshot for a single store."""

    from numereng_viz import build_monitor_snapshot as build_monitor_snapshot_record

    resolved_request = MonitorSnapshotRequest() if request is None else request
    try:
        payload = build_monitor_snapshot_record(
            store_root=resolved_request.store_root,
            refresh_cloud=resolved_request.refresh_cloud,
        )
    except Exception as exc:
        raise PackageError(f"monitor_snapshot_failed:{exc}") from exc
    return MonitorSnapshotResponse.model_validate(payload)


__all__ = ["build_monitor_snapshot"]
