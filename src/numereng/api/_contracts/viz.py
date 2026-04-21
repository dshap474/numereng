"""Public contracts for the packaged viz runtime."""

from __future__ import annotations

from numereng.api._contracts.shared import WorkspaceBoundRequest


class VizAppRequest(WorkspaceBoundRequest):
    pass


__all__ = ["VizAppRequest"]
