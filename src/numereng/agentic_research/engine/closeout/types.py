"""Constants, error tokens, phase state models, and atomic write helpers for the closeout chain.

The closeout chain runs deterministic phases over a completed agentic experiment. This module
holds the small shared vocabulary the rest of the subpackage builds on: phase names, error tokens
(all carrying the ``agentic_research_closeout_`` prefix), the persisted phase-state models, and an
atomic text writer (``types.write_text`` in the parent package is NOT atomic; the commit protocol
needs one that is).

USAGE:
    from numereng.agentic_research.engine.closeout import types as ct
    state = ct.CloseoutState.new(experiment_id="x", memory_root_identity="/abs")
    ct.write_text_atomic(path, "content")
    digest = ct.sha256_text("content")
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from numereng.agentic_research.engine import types as ar_types

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CLOSEOUT_DIRNAME = "closeout"
CLOSEOUT_STATE_FILENAME = "state.json"
CLOSEOUT_COMMIT_FILENAME = "commit.json"
CLOSEOUT_EVIDENCE_FILENAME = "evidence_summary.json"
CLOSEOUT_MEMO_FILENAME = "EXPERIMENT.closeout.md"
CLOSEOUT_CLASSIFICATION_FILENAME = "classification.json"
CLOSEOUT_STATE_SCHEMA_VERSION = 2

CLOSEOUT_TIMEOUT_SECONDS = 1800.0
MAX_CLOSEOUT_CONTEXT_CHARS = 240_000

CLOSEOUT_LOCK_FILENAME = "lock"
MEMORY_ROOT_LOCK_FILENAME = ".closeout.lock"
LOCK_STALE_SECONDS = 6 * 3600

# The research-memory branch/ledger contract: six topic ledgers plus README.
MEMORY_TOPIC_FILES = (
    "ensembling",
    "features",
    "hyperparameters",
    "models",
    "neutralization-exposure",
    "targets",
)
# A branch directory holds exactly these seven files (README + the six topics), nothing else.
MEMORY_BRANCH_README = "README.md"
MEMORY_BRANCH_FILE_COUNT = 1 + len(MEMORY_TOPIC_FILES)

# Ledger structure anchors (verified against the real topics/*.md files). Every ledger has THREE
# sections in order: Current Overview, Current Best Understanding (both mutable — the manual skill
# calls the pair the "mutable top overview"), then the append-only learnings.
LEDGER_OVERVIEW_ANCHOR = "## Current Overview"
LEDGER_BEST_UNDERSTANDING_ANCHOR = "## Current Best Understanding"
LEDGER_LEARNINGS_ANCHOR = "## Append-Only Experiment Learnings"
LEDGER_ENTRY_PREFIX = "### "
# Per-ledger view cap when building synthesize context (overview + newest entries).
LEDGER_CONTEXT_CAP = 24_000
LEDGER_NEWEST_ENTRIES = 5

# EXTRACT content contract (§3.2). Each topic file must carry these level-2 headings; README is
# validated separately (it cites the experiment id and links the six topic files).
EXTRACT_TOPIC_HEADINGS = (
    "Experiment-Specific Takeaway",
    "Evidence Snapshot",
    "Evidence Level",
    "Design-Space Role",
    "Confounds",
    "What Not To Infer",
    "Not Established",
    "Scope And Caveats",
    "Future Implication",
    "Master Ledger Update",
)
EXTRACT_EVIDENCE_LEVELS = (
    "verified artifact",
    "computed metric",
    "supported inference",
    "hypothesis / next-step",
)
EXTRACT_DESIGN_SPACE_ROLES = (
    "varied",
    "controlled",
    "inherited",
    "observed",
    "not_tested",
    "confounded",
)

# SYNTHESIZE / CURRENT.md contract (§3.3). Required top-level sections pinned from the live
# CURRENT.md structure (frontier / anchors / constraints); the id and a memory-branch "Full record:"
# pointer must appear, and the file must be a non-trivial compressed rewrite.
CURRENT_MD_REQUIRED_SECTIONS = (
    "Compressed Frontier",
    "Comparison Anchors",
    "Current Constraints",
)
CURRENT_MD_MIN_CHARS = 2_000
CURRENT_MD_FILENAME = "CURRENT.md"

# Phase registry.
PHASE_FINALIZE = "finalize"
PHASE_CLASSIFY = "classify"
PHASE_EXTRACT = "extract"
PHASE_SYNTHESIZE = "synthesize"
PHASE_ORDER = (PHASE_FINALIZE, PHASE_CLASSIFY, PHASE_EXTRACT, PHASE_SYNTHESIZE)
PHASE_TERMINAL_STATUSES = ("done", "skipped")

CLASSIFICATION_DISPOSITIONS = ("master", "branch_only", "exclude")
LEGACY_MASTER_NOTE = "legacy_master"

# --------------------------------------------------------------------------- #
# Error tokens
# --------------------------------------------------------------------------- #
ERROR_PREFIX = "agentic_research_closeout_"

ERR_NOT_AGENTIC = f"{ERROR_PREFIX}not_agentic"
ERR_NO_ROUNDS = f"{ERROR_PREFIX}no_rounds"
ERR_RUN_ACTIVE = f"{ERROR_PREFIX}run_active"
ERR_EXPERIMENT_ARCHIVED = f"{ERROR_PREFIX}experiment_archived"
ERR_MEMORY_ROOT_INVALID = f"{ERROR_PREFIX}memory_root_invalid"
ERR_MEMORY_ROOT_CHANGED = f"{ERROR_PREFIX}memory_root_changed"
ERR_BELIEVED_BEST_UNRESOLVED = f"{ERROR_PREFIX}believed_best_unresolved"
ERR_LEADERBOARD_EMPTY = f"{ERROR_PREFIX}leaderboard_empty"
ERR_RESTART_BLOCKED_AFTER_SYNTHESIZE = f"{ERROR_PREFIX}restart_blocked_after_synthesize"
ERR_RESTART_BLOCKED_AFTER_EXTRACT = f"{ERROR_PREFIX}restart_blocked_after_extract"
ERR_SYNTHESIZE_BACKUP_FAILED = f"{ERROR_PREFIX}synthesize_backup_failed"
ERR_HOLDOUT_TAMPERED = f"{ERROR_PREFIX}holdout_frozen_input_tampered"
ERR_HOLDOUT_REUSE = f"{ERROR_PREFIX}holdout_reuse_blocked"

# Filename for the one-time sealed holdout scoring record written at closeout.
CLOSEOUT_HOLDOUT_FILENAME = "holdout_result.json"


def err_budget_not_reached(done: int, budget: int) -> str:
    return f"{ERROR_PREFIX}budget_not_reached:{done}/{budget}"


def err_journal_malformed(lineno: int) -> str:
    return f"{ERROR_PREFIX}journal_malformed:{lineno}"


def err_journal_entry_invalid(lineno: int, what: str) -> str:
    return f"{ERROR_PREFIX}journal_entry_invalid:{lineno}:{what}"


def err_commit_conflict(path: str) -> str:
    return f"{ERROR_PREFIX}commit_conflict:{path}"


def err_stale_upstream(phase: str, path: str) -> str:
    return f"{ERROR_PREFIX}stale_upstream:{phase}:{path}"


def err_output_path_not_allowed(path: str) -> str:
    return f"{ERROR_PREFIX}output_path_not_allowed:{path}"


def err_output_path_duplicate(path: str) -> str:
    return f"{ERROR_PREFIX}output_path_duplicate:{path}"


def err_output_slot_missing(path: str) -> str:
    return f"{ERROR_PREFIX}output_slot_missing:{path}"


def err_output_content_empty(path: str) -> str:
    return f"{ERROR_PREFIX}output_content_empty:{path}"


def err_lock_held(path: str) -> str:
    return f"{ERROR_PREFIX}lock_held:{path}"


def err_lock_stale(path: str) -> str:
    return f"{ERROR_PREFIX}lock_stale:{path}"


def err_restart_from_invalid(phase: str) -> str:
    return f"{ERROR_PREFIX}restart_from_invalid:{phase}"


def err_until_invalid(phase: str) -> str:
    return f"{ERROR_PREFIX}until_invalid:{phase}"


def err_memo_section_missing(heading: str) -> str:
    return f"{ERROR_PREFIX}memo_section_missing:{heading}"


def err_memo_too_short(length: int, minimum: int) -> str:
    return f"{ERROR_PREFIX}memo_too_short:{length}/{minimum}"


def err_memo_reference_missing(what: str) -> str:
    return f"{ERROR_PREFIX}memo_reference_missing:{what}"


def err_topic_section_missing(topic: str, heading: str) -> str:
    return f"{ERROR_PREFIX}topic_section_missing:{topic}:{heading}"


def err_evidence_level_invalid(topic: str) -> str:
    return f"{ERROR_PREFIX}evidence_level_invalid:{topic}"


def err_design_role_invalid(topic: str) -> str:
    return f"{ERROR_PREFIX}design_role_invalid:{topic}"


def err_readme_link_missing(what: str) -> str:
    return f"{ERROR_PREFIX}readme_link_missing:{what}"


def err_branch_exists(path: str) -> str:
    return f"{ERROR_PREFIX}branch_exists:{path}"


def err_branch_file_count(count: int) -> str:
    return f"{ERROR_PREFIX}branch_file_count:{count}/{MEMORY_BRANCH_FILE_COUNT}"


def err_evidence_missing(path: str) -> str:
    return f"{ERROR_PREFIX}evidence_missing:{path}"


def err_memo_missing(path: str) -> str:
    return f"{ERROR_PREFIX}memo_missing:{path}"


def err_ledger_structure(path: str) -> str:
    return f"{ERROR_PREFIX}ledger_structure:{path}"


def err_entry_heading_invalid(topic: str) -> str:
    return f"{ERROR_PREFIX}entry_heading_invalid:{topic}"


def err_entry_link_missing(topic: str) -> str:
    return f"{ERROR_PREFIX}entry_link_missing:{topic}"


def err_duplicate_ledger_entry(topic: str) -> str:
    return f"{ERROR_PREFIX}duplicate_ledger_entry:{topic}"


def err_section_replacement_invalid(topic: str, section: str) -> str:
    return f"{ERROR_PREFIX}section_replacement_invalid:{topic}:{section}"


def err_classification_field_invalid(field: str) -> str:
    return f"{ERROR_PREFIX}classification_field_invalid:{field}"


def err_entry_count_invalid(topic: str, count: int) -> str:
    return f"{ERROR_PREFIX}entry_count_invalid:{topic}:{count}"


def err_current_md_section_missing(heading: str) -> str:
    return f"{ERROR_PREFIX}current_md_section_missing:{heading}"


def err_current_md_reference_missing(what: str) -> str:
    return f"{ERROR_PREFIX}current_md_reference_missing:{what}"


def err_current_md_too_short(length: int, minimum: int) -> str:
    return f"{ERROR_PREFIX}current_md_too_short:{length}/{minimum}"


def err_branch_file_missing(path: str) -> str:
    return f"{ERROR_PREFIX}branch_file_missing:{path}"


# --------------------------------------------------------------------------- #
# Shared path helpers
# --------------------------------------------------------------------------- #
def debug_dir(closeout_dir: Path) -> Path:
    """Per-phase LLM debug artifact directory inside a closeout dir."""
    return closeout_dir / "debug"


def branch_dir(memory_root: Path, experiment_id: str) -> Path:
    """This experiment's research-memory branch directory."""
    return memory_root / "experiments" / experiment_id


def topics_dir(memory_root: Path) -> Path:
    """The master topic-ledger directory of a research-memory root."""
    return memory_root / "topics"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class CloseoutError(ar_types.AgenticResearchError):
    """Any closeout failure. Gate/setup failures raise; phase failures are captured by the runner."""


# --------------------------------------------------------------------------- #
# Phase-state models
# --------------------------------------------------------------------------- #
@dataclass
class CloseoutPhaseState:
    status: str = "pending"
    completed_at: str | None = None
    duration_seconds: float | None = None
    notes: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"status": self.status}
        if self.completed_at is not None:
            payload["completed_at"] = self.completed_at
        if self.duration_seconds is not None:
            payload["duration_seconds"] = self.duration_seconds
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.outputs:
            payload["outputs"] = dict(self.outputs)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CloseoutPhaseState:
        outputs_raw = data.get("outputs")
        outputs = {str(k): str(v) for k, v in outputs_raw.items()} if isinstance(outputs_raw, dict) else {}
        duration = data.get("duration_seconds")
        return cls(
            status=str(data.get("status", "pending")),
            completed_at=data.get("completed_at") if isinstance(data.get("completed_at"), str) else None,
            duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
            notes=data.get("notes") if isinstance(data.get("notes"), str) else None,
            outputs=outputs,
        )


@dataclass
class CloseoutState:
    experiment_id: str
    memory_root_identity: str
    phases: dict[str, CloseoutPhaseState]
    schema_version: int = CLOSEOUT_STATE_SCHEMA_VERSION

    @classmethod
    def new(cls, *, experiment_id: str, memory_root_identity: str) -> CloseoutState:
        return cls(
            experiment_id=experiment_id,
            memory_root_identity=memory_root_identity,
            phases={name: CloseoutPhaseState() for name in PHASE_ORDER},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "memory_root_identity": self.memory_root_identity,
            "phases": {name: self.phases[name].to_dict() for name in PHASE_ORDER},
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CloseoutState:
        phases_raw = data.get("phases")
        phases: dict[str, CloseoutPhaseState] = {}
        for name in PHASE_ORDER:
            entry = phases_raw.get(name) if isinstance(phases_raw, dict) else None
            phases[name] = CloseoutPhaseState.from_dict(entry) if isinstance(entry, dict) else CloseoutPhaseState()
        if isinstance(phases_raw, dict) and PHASE_CLASSIFY not in phases_raw:
            later_done = any(phases[phase].status == "done" for phase in (PHASE_EXTRACT, PHASE_SYNTHESIZE))
            if later_done:
                phases[PHASE_CLASSIFY] = CloseoutPhaseState(status="done", notes=LEGACY_MASTER_NOTE)
        return cls(
            experiment_id=str(data.get("experiment_id", "")),
            memory_root_identity=str(data.get("memory_root_identity", "")),
            phases=phases,
            schema_version=CLOSEOUT_STATE_SCHEMA_VERSION,
        )


# --------------------------------------------------------------------------- #
# Runner result models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CloseoutPhaseReport:
    name: str
    status: str
    notes: str | None
    duration_seconds: float | None
    outputs: dict[str, str]


@dataclass(frozen=True)
class CloseoutResult:
    experiment_id: str
    phases: tuple[CloseoutPhaseReport, ...]
    stopped_at_phase: str | None
    error: str | None


# --------------------------------------------------------------------------- #
# Atomic write + hashing helpers
# --------------------------------------------------------------------------- #
def write_text_atomic(path: Path, text: str) -> None:
    """Atomic text write (tmp + os.replace); the parent package's write_text is not atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
