"""Public contracts for the packaged viz runtime."""

from __future__ import annotations

from numereng.api._contracts_base import WorkspaceBoundRequest


class VizAppRequest(WorkspaceBoundRequest):
    pass


__all__ = ["VizAppRequest"]
