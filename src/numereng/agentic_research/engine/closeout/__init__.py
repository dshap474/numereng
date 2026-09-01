"""Closeout subpackage: deterministic evidence plus one decision memo for a finished experiment.

Closeout creates and launches nothing. Research-memory writes belong to the
``research-memory-update`` skill; next-experiment design lives in the pre-run INIT-PROGRAM playbook.

USAGE:
    from numereng.agentic_research.engine.closeout import run_closeout
    result = run_closeout(store_root=".numereng", experiment_id="x")
"""

from __future__ import annotations

from numereng.agentic_research.engine.closeout.runner import run_closeout
from numereng.agentic_research.engine.closeout.types import CloseoutError, CloseoutResult, memo_path

__all__ = [
    "CloseoutError",
    "CloseoutResult",
    "memo_path",
    "run_closeout",
]
