"""Agentic config-research API facade."""

from __future__ import annotations

from numereng.api._agentic_research.runtime import research_run, research_status

__all__ = [
    "research_run",
    "research_status",
]
