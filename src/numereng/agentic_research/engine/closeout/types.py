"""Constants, error tokens, and the result model for closeout.

Closeout is one job: build the deterministic evidence bundle, then have the LLM write one decision
memo. This module holds the small shared vocabulary the rest of the subpackage builds on — artifact
filenames, the context and memo size limits, and the stable error tokens (all carrying the
``agentic_research_closeout_`` prefix). Research-memory writes are not Python's job; the
``research-memory-update`` skill owns them and states its own contract.

USAGE:
    from numereng.agentic_research.engine.closeout import types as ct
    raise ct.CloseoutError(ct.ERR_RUN_ACTIVE)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from numereng.agentic_research.engine import types as ar_types

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CLOSEOUT_DIRNAME = "closeout"
CLOSEOUT_EVIDENCE_FILENAME = "evidence_summary.json"
CLOSEOUT_MEMO_FILENAME = "EXPERIMENT.closeout.md"
CLOSEOUT_RESPONSE_FILENAME = "finalize_response.md"
# Filename for the one-time sealed holdout scoring record written at closeout.
CLOSEOUT_HOLDOUT_FILENAME = "holdout_result.json"

CLOSEOUT_TIMEOUT_SECONDS = 1800.0
MAX_CLOSEOUT_CONTEXT_CHARS = 240_000
MEMO_REQUIRED_HEADING = "## Verdict"
MEMO_MIN_CHARS = 1_500

# --------------------------------------------------------------------------- #
# Error tokens
# --------------------------------------------------------------------------- #
ERROR_PREFIX = "agentic_research_closeout_"

ERR_NOT_AGENTIC = f"{ERROR_PREFIX}not_agentic"
ERR_NO_ROUNDS = f"{ERROR_PREFIX}no_rounds"
ERR_RUN_ACTIVE = f"{ERROR_PREFIX}run_active"
ERR_EXPERIMENT_ARCHIVED = f"{ERROR_PREFIX}experiment_archived"
ERR_BELIEVED_BEST_UNRESOLVED = f"{ERROR_PREFIX}believed_best_unresolved"
ERR_LEADERBOARD_EMPTY = f"{ERROR_PREFIX}leaderboard_empty"
ERR_HOLDOUT_TAMPERED = f"{ERROR_PREFIX}holdout_frozen_input_tampered"
ERR_HOLDOUT_REUSE = f"{ERROR_PREFIX}holdout_reuse_blocked"


# --------------------------------------------------------------------------- #
# Shared path helpers
# --------------------------------------------------------------------------- #
def closeout_dir(agentic_research_dir: Path) -> Path:
    """The closeout artifact directory of one experiment's agentic-research dir."""
    return agentic_research_dir / CLOSEOUT_DIRNAME


def debug_dir(directory: Path) -> Path:
    """LLM debug artifact directory inside a closeout dir."""
    return directory / "debug"


def memo_path(agentic_research_dir: Path) -> Path:
    """The decision memo closeout writes for one experiment."""
    return closeout_dir(agentic_research_dir) / CLOSEOUT_MEMO_FILENAME


# --------------------------------------------------------------------------- #
# Exceptions and result model
# --------------------------------------------------------------------------- #
class CloseoutError(ar_types.AgenticResearchError):
    """Any closeout failure: gate refusal, corrupt evidence, or an unusable memo."""


@dataclass(frozen=True)
class CloseoutResult:
    experiment_id: str
    evidence_path: Path
    memo_path: Path
    holdout_summary: dict[str, object] | None
