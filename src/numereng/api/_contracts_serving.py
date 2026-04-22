"""Compatibility facade for serving score and diagnostics contracts."""

from __future__ import annotations

from numereng.api._contracts.serving import *  # noqa: F403
from numereng.api._contracts.serving import __all__  # noqa: F401  (re-export for contracts aggregation)
