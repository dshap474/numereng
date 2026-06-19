"""Shared types, exceptions, constants, and small utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

ResearchStatus = Literal["initialized", "running", "interrupted", "stopped", "failed"]
ResearchAction = Literal["baseline", "run"]

PROGRAM_PATH = Path(__file__).with_name("PROGRAM.md")
CUSTOM_PROGRAM_DIR = Path(__file__).with_name("custom_programs")
PROGRAM_METADATA_KEY, ALLOWED_PATHS_METADATA_KEY = "agentic_research_program", "agentic_research_allowed_change_paths"
VALUE_CAPS_METADATA_KEY, BUDGET_ROUNDS_METADATA_KEY = "agentic_research_value_caps", "agentic_research_budget_rounds"

AGENTIC_DIRNAME, STATE_FILENAME, JOURNAL_FILENAME = "agentic_research", "state.json", "journal.jsonl"
STATE_SCHEMA_VERSION = 2

PRIMARY_METRIC, PRIMARY_METRIC_FIELD = "bmc_last_200_eras.mean", "bmc_last_200_eras_mean"
PAYOUT_TARGET_COL, SCORING_STAGE = "target_ender_20", "post_training_full"
RUN_PLAN_FIELDS = ("plan_index", "round", "seed", "target", "horizon", "config_path", "score_stage_default")

REPORT_LIMIT, RECENT_JOURNAL_LIMIT, CONFIG_CONTEXT_RECENT = 25, 12, 40
MAX_CONTEXT_CHARS, CONSECUTIVE_FAILURE_BAIL_THRESHOLD = 12_000, 5
CODEX_TIMEOUT_SECONDS = 600.0


class AgenticResearchError(Exception):
    pass


class AgenticResearchValidationError(AgenticResearchError):
    pass


class AgenticResearchDuplicateCandidate(AgenticResearchValidationError):
    pass


@dataclass(frozen=True)
class ResearchBestRun:
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
    path: str
    value: object
    reason: str


@dataclass(frozen=True)
class ResearchDecision:
    action: Literal["run"]
    learning: str
    belief_update: str
    next_hypothesis: str | None
    parent_config: str | None
    changes: tuple[ResearchChange, ...]
    stop_reason: str | None


@dataclass(frozen=True)
class ResearchLLMResponse:
    decision: ResearchDecision
    round_markdown: str
    experiment_markdown: str | None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def as_list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgenticResearchValidationError(f"agentic_research_field_missing:{key}")
    return value.strip()


def optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def status_value(value: object) -> ResearchStatus:
    if value in {"initialized", "running", "interrupted", "stopped", "failed"}:
        return cast(ResearchStatus, value)
    return "initialized"


def read_text(path: Path, *, limit: int) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
