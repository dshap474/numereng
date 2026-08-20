"""Closeout chain subpackage: deterministic finalize of a completed agentic experiment.

The full chain is implemented: deterministic evidence, FINALIZE, CLASSIFY, and selective
EXTRACT/SYNTHESIZE routing. The chain creates and launches nothing; next-experiment design
lives in the pre-run INIT-PROGRAM playbook.

USAGE:
    from numereng.agentic_research.engine.closeout import run_closeout, get_closeout_status
    result = run_closeout(store_root=".numereng", experiment_id="x", until="finalize")
"""

from __future__ import annotations

from numereng.agentic_research.engine.closeout.runner import get_closeout_status, run_closeout
from numereng.agentic_research.engine.closeout.types import (
    CloseoutError,
    CloseoutPhaseReport,
    CloseoutResult,
)

__all__ = [
    "CloseoutError",
    "CloseoutPhaseReport",
    "CloseoutResult",
    "get_closeout_status",
    "run_closeout",
]
