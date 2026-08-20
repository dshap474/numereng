"""Shared types, exceptions, constants, and small utilities."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

ResearchStatus = Literal["initialized", "running", "interrupted", "stopped", "failed"]
ResearchAction = Literal["baseline", "run"]

CUSTOM_PROGRAM_DIR = Path(__file__).parents[1] / "programs"
PROGRAM_PATH = CUSTOM_PROGRAM_DIR / "PROGRAM.md"
PROGRAM_METADATA_KEY, ALLOWED_PATHS_METADATA_KEY = "agentic_research_program", "agentic_research_allowed_change_paths"

# CORE program sections are pure harness mechanism, identical across every experiment. The strategy
# sections ("0." frame, "4." substrate/budget, "6." search discipline) are experiment-specific and
# exempt. Every custom program must copy these CORE sections from PROGRAM.md byte-verbatim because
# the runner loads exactly one self-contained file; this pin is shared by the pre-flight check and
# the drift-lint test.
CORE_PROGRAM_SECTION_KEYS = ("1.", "2.", "2.1", "3.", "5.", "7.", "8.", "9.", "10.", "Context")
# A section boundary is a level-2 heading or a numbered level-3 heading (e.g. "### 2.1 ..."); an
# un-numbered level-3 heading (e.g. "### `round_markdown`") stays inside its parent section body.
_PROGRAM_SECTION_BOUNDARY = re.compile(r"^(#{2} |#{3} \d)")
VALUE_CAPS_METADATA_KEY, BUDGET_ROUNDS_METADATA_KEY = "agentic_research_value_caps", "agentic_research_budget_rounds"

AGENTIC_DIRNAME, STATE_FILENAME, JOURNAL_FILENAME = "agentic_research", "state.json", "journal.jsonl"
STATE_SCHEMA_VERSION = 2

PRIMARY_METRIC, PRIMARY_METRIC_FIELD = "bmc_last_200_eras.mean", "bmc_last_200_eras_mean"
PAYOUT_TARGET_COL, SCORING_STAGE = "target_ender_20", "post_training_full"
RUN_PLAN_FIELDS = ("plan_index", "round", "seed", "target", "horizon", "config_path", "score_stage_default")

REPORT_LIMIT, RECENT_JOURNAL_LIMIT, CONFIG_CONTEXT_RECENT = 25, 12, 40
RECIPE_LEADERBOARD_LIMIT, COVERAGE_VALUE_LIMIT, PLATEAU_DIVERSIFY_ROUNDS = 15, 12, 5
MAX_CONTEXT_CHARS, CONSECUTIVE_FAILURE_BAIL_THRESHOLD = 12_000, 5
MAX_TOTAL_CONTEXT_CHARS = 96_000
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
    believed_best: str | None = None
    seeds: tuple[int, ...] = ()


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


def extract_core_sections(text: str) -> dict[str, str]:
    """Parse a program's CORE sections (see ``CORE_PROGRAM_SECTION_KEYS``) keyed by section number."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        if _PROGRAM_SECTION_BOUNDARY.match(line):
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = line.lstrip("#").strip().split()[0]
            buffer = [line]
        elif current_key is not None:
            buffer.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(buffer).strip()
    return {key: sections[key] for key in CORE_PROGRAM_SECTION_KEYS if key in sections}


def first_diverging_core_section(program_text: str, base_text: str) -> str | None:
    """Return the first CORE section key that is missing or differs from the base, else ``None``."""
    program, base = extract_core_sections(program_text), extract_core_sections(base_text)
    for key in CORE_PROGRAM_SECTION_KEYS:
        if program.get(key) != base.get(key):
            return key
    return None
