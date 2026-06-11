"""Shared types, exceptions, constants, and small utilities for the rebuilt harness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

ResearchStatus = Literal["initialized", "running", "interrupted", "stopped", "failed"]
ResearchAction = Literal["baseline", "run"]

PROGRAM_PATH = Path(__file__).with_name("PROGRAM.md")
CUSTOM_PROGRAM_DIR = Path(__file__).with_name("custom_programs")
PROGRAM_METADATA_KEY = "agentic_research_program"
ALLOWED_PATHS_METADATA_KEY = "agentic_research_allowed_change_paths"
VALUE_CAPS_METADATA_KEY = "agentic_research_value_caps"
ARTIFACT_ROTATION_METADATA_KEY = "agentic_research_artifact_rotation"
BUDGET_ROUNDS_METADATA_KEY = "agentic_research_budget_rounds"

AGENTIC_DIRNAME = "agentic_research"
STATE_FILENAME = "state.json"
JOURNAL_FILENAME = "journal.jsonl"
STATE_SCHEMA_VERSION = 2

PRIMARY_METRIC = "bmc_last_200_eras.mean"
PRIMARY_METRIC_FIELD = "bmc_last_200_eras_mean"
PAYOUT_TARGET_COL = "target_ender_20"
SCORING_STAGE = "post_training_core"
RUN_PLAN_FIELDS = ("plan_index", "round", "seed", "target", "horizon", "config_path", "score_stage_default")

REPORT_LIMIT = 25
RECENT_JOURNAL_LIMIT = 12
CONFIG_CONTEXT_RECENT = 40
MAX_CONTEXT_CHARS = 12_000
CONSECUTIVE_FAILURE_BAIL_THRESHOLD = 5
CODEX_TIMEOUT_SECONDS = 600.0
ARTIFACT_ROTATION_RECENT_ROUND_GRACE = 10


class AgenticResearchError(Exception):
    """Base error for agentic research workflows."""


class AgenticResearchValidationError(AgenticResearchError):
    """Raised when an LLM decision or local research state is invalid."""


class AgenticResearchDuplicateCandidate(AgenticResearchValidationError):
    """Raised when a proposed config hashes to one already on disk with a recorded run."""


@dataclass(frozen=True)
class ResearchBestRun:
    """Best known run for the primary research metric."""

    experiment_id: str | None = None
    run_id: str | None = None
    bmc_last_200_eras_mean: float | None = None
    bmc_mean: float | None = None
    corr_mean: float | None = None
    mmc_mean: float | None = None
    cwmm_mean: float | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class ResearchRoundResult:
    """One completed or terminal research-loop round."""

    round_number: int
    round_label: str
    action: ResearchAction
    status: str
    config_path: Path | None
    run_id: str | None
    metric_value: float | None
    learning: str
    artifact_dir: Path


@dataclass(frozen=True)
class ResearchStatusResult:
    """Current lightweight state for one experiment's research loop."""

    experiment_id: str
    status: ResearchStatus
    next_round_number: int
    total_rounds_completed: int
    last_checkpoint: str
    last_round_label: str | None
    last_run_id: str | None
    stop_reason: str | None
    best_overall: ResearchBestRun
    agentic_research_dir: Path
    state_path: Path
    trace_path: Path
    decision_path: Path
    program_path: Path


@dataclass(frozen=True)
class ResearchRunResult:
    """Result for a foreground research-loop invocation."""

    experiment_id: str
    status: ResearchStatus
    next_round_number: int
    total_rounds_completed: int
    last_checkpoint: str
    stop_reason: str | None
    best_overall: ResearchBestRun
    rounds: tuple[ResearchRoundResult, ...]
    interrupted: bool = False


@dataclass(frozen=True)
class ResearchChange:
    """One validated config change requested by the LLM."""

    path: str
    value: object
    reason: str


@dataclass(frozen=True)
class ResearchDecision:
    """Parsed LLM decision for the next research step."""

    action: Literal["run"]
    learning: str
    belief_update: str
    next_hypothesis: str | None
    parent_config: str | None
    changes: tuple[ResearchChange, ...]
    stop_reason: str | None


@dataclass(frozen=True)
class ResearchLLMResponse:
    """Validated LLM response: research form plus cumulative round memo."""

    decision: ResearchDecision
    round_markdown: str
    experiment_markdown: str | None


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(UTC).isoformat()


def as_int(value: object, *, default: int) -> int:
    """Coerce to int, rejecting booleans (a subclass of int)."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def as_list(value: object) -> list[object]:
    """Return the value as a list, or an empty list if it is not one."""
    return cast(list[object], value) if isinstance(value, list) else []


def optional_str(value: object) -> str | None:
    """Return a stripped non-empty string, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def required_str(payload: dict[str, object], key: str) -> str:
    """Return a required non-empty string field or raise a stable token."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgenticResearchValidationError(f"agentic_research_field_missing:{key}")
    return value.strip()


def optional_float(value: object) -> float | None:
    """Coerce to float (also used for metrics), rejecting booleans, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def status_value(value: object) -> ResearchStatus:
    """Normalize an arbitrary value to a known research status."""
    if value in {"initialized", "running", "interrupted", "stopped", "failed"}:
        return cast(ResearchStatus, value)
    return "initialized"
