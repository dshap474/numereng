"""Agentic config-research API facade."""

from __future__ import annotations

from numereng.api._agentic_research.closeout import research_closeout, research_closeout_status
from numereng.api._agentic_research.program import research_program_check, research_program_resplice
from numereng.api._agentic_research.runtime import research_run, research_status

__all__ = [
    "research_closeout",
    "research_closeout_status",
    "research_program_check",
    "research_program_resplice",
    "research_run",
    "research_status",
]
