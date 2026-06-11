"""Durable memory: state.json, journal.jsonl, rounds/rN.md, EXPERIMENT.md, artifact rotation."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from numereng.features.agentic_research.types import (
    AGENTIC_DIRNAME,
    ARTIFACT_ROTATION_METADATA_KEY,
    ARTIFACT_ROTATION_RECENT_ROUND_GRACE,
    CUSTOM_PROGRAM_DIR,
    JOURNAL_FILENAME,
    MAX_CONTEXT_CHARS,
    PRIMARY_METRIC_FIELD,
    PROGRAM_METADATA_KEY,
    PROGRAM_PATH,
    STATE_FILENAME,
    STATE_SCHEMA_VERSION,
    AgenticResearchValidationError,
    ResearchBestRun,
    as_int,
    utc_now_iso,
)
from numereng.features.experiments import ExperimentRecord

# Canonical state defaults. Every field listed here is guaranteed to exist on a
# fresh state.json and on any state loaded from disk (via apply_state_defaults).
_STATE_DEFAULTS: dict[str, object] = {
    "schema_version": STATE_SCHEMA_VERSION,
    "status": "initialized",
    "next_round_number": 1,
    "total_rounds_completed": 0,
    "failed_rounds_counter": 0,
    "last_checkpoint": "initialized",
    "last_round_label": None,
    "last_run_id": None,
    "stop_reason": None,
    "champion": None,
    "best_overall": None,
    "last_error": None,
    "last_heartbeat": None,
}


def apply_state_defaults(state: dict[str, object]) -> dict[str, object]:
    """Backfill missing canonical fields (and bump schema_version) on a loaded state."""
    for key, value in _STATE_DEFAULTS.items():
        state.setdefault(key, deepcopy(value))
    state["schema_version"] = STATE_SCHEMA_VERSION
    return state


def initial_state(experiment: ExperimentRecord) -> dict[str, object]:
    """Return a fresh v2 state for an experiment with no recorded rounds yet."""
    now = utc_now_iso()
    state: dict[str, object] = {
        "experiment_id": experiment.experiment_id,
        "best_overall": asdict(ResearchBestRun()),
        "created_at": now,
        "updated_at": now,
    }
    return apply_state_defaults(state)


def load_state(path: Path) -> dict[str, object] | None:
    """Load state.json, raising a stable token on corruption (never reinitialize)."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgenticResearchValidationError(f"agentic_research_state_invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError(f"agentic_research_state_invalid:{path}")
    return apply_state_defaults(payload)


def save_state(experiment: ExperimentRecord, state: dict[str, object]) -> None:
    """Persist the state to disk atomically."""
    write_json(state_path(experiment), state)


def heartbeat(state: dict[str, object]) -> None:
    """Stamp the current time on the state so a dead session does not read as live forever."""
    state["last_heartbeat"] = utc_now_iso()


def append_journal(experiment: ExperimentRecord, entry: dict[str, object]) -> None:
    """Append one round-attempt line to journal.jsonl (append-only contract)."""
    path = journal_path(experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def journal_tail(experiment: ExperimentRecord, *, limit: int) -> list[dict[str, object]]:
    """Return the last `limit` parseable journal entries."""
    path = journal_path(experiment)
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows[-limit:]


def journal_has_recorded_run(experiment: ExperimentRecord, config_name: str) -> bool:
    """True if any journal entry recorded a run for the given config filename."""
    path = journal_path(experiment)
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("config") == config_name and payload.get("run_id"):
            return True
    return False


def write_round_markdown(experiment: ExperimentRecord, entry: dict[str, object], *, memo: str | None) -> None:
    """Write rounds/rN.md: the model's memo verbatim plus a fixed Machine Result block."""
    round_label = str(entry.get("round_label"))
    lines: list[str] = []
    if memo and memo.strip():
        lines.append(memo.rstrip())
    else:
        lines.append(f"# {round_label} Research State")
    lines.extend(
        [
            "",
            "---",
            "## Machine Result",
            f"- round: {entry.get('round')}",
            f"- action: {entry.get('action')}",
            f"- status: {entry.get('status')}",
            f"- parent: {_value(entry.get('parent_config'))}",
            f"- config: {_value(entry.get('config'))}",
            f"- run_id: {_value(entry.get('run_id'))}",
            f"- seed: {_value(entry.get('seed'))}",
            f"- {PRIMARY_METRIC_FIELD}: {_value(entry.get('metric'))}",
            f"- champion: {'yes' if entry.get('is_champion') else 'no'}",
            f"- wall: {_value(entry.get('wall_seconds'))}",
        ]
    )
    if entry.get("error"):
        lines.append(f"- error: {entry.get('error')}")
    write_text(rounds_dir(experiment) / f"{round_label}.md", "\n".join(lines).rstrip() + "\n")


def write_experiment_markdown(experiment: ExperimentRecord, content: str | None) -> int:
    """Overwrite EXPERIMENT.md atomically; null/empty content preserves the prior file."""
    if not content:
        return 0
    path = experiment.manifest_path.parent / "EXPERIMENT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return len(content)


def write_failure_debug(
    *, artifact_dir: Path, round_label: str, prompt: str, error: str, raw_response: str | None = None
) -> None:
    """Dump the prompt, error, and (when present) the raw response beside the round artifacts."""
    prefix = artifact_dir / f"{round_label}.debug"
    write_text(Path(f"{prefix}.prompt.md"), prompt)
    write_text(Path(f"{prefix}.error.txt"), error.strip() + "\n")
    if raw_response is not None:
        write_text(Path(f"{prefix}.llm_response.txt"), raw_response)


def rotate_run_artifacts(
    *, root: Path, experiment: ExperimentRecord, state: dict[str, object], last_round_number: int
) -> None:
    """When rotation metadata is enabled, delete heavy parquets from runs not worth keeping.

    Protected: the champion run, the last 10 rounds' run_ids (journal tail), and the
    experiment's own run list. Everything else owned by this experiment is fair game."""
    if experiment.metadata.get(ARTIFACT_ROTATION_METADATA_KEY) != "enabled":
        return
    keep: set[str] = set()
    champion = state.get("champion")
    if isinstance(champion, dict) and isinstance(champion.get("run_id"), str):
        keep.add(str(champion["run_id"]))
    grace_start = max(1, last_round_number - ARTIFACT_ROTATION_RECENT_ROUND_GRACE + 1)
    for entry in journal_tail(experiment, limit=ARTIFACT_ROTATION_RECENT_ROUND_GRACE * 2):
        run_id = entry.get("run_id")
        if as_int(entry.get("round"), default=0) >= grace_start and isinstance(run_id, str):
            keep.add(run_id)
    for run_id in set(experiment.runs) - keep:
        for parquet in (root / "runs" / run_id).rglob("*.parquet"):
            try:
                parquet.unlink()
            except OSError:
                continue


def agentic_dir(experiment: ExperimentRecord) -> Path:
    """Return the experiment's agentic_research directory."""
    return experiment.manifest_path.parent / AGENTIC_DIRNAME


def rounds_dir(experiment: ExperimentRecord) -> Path:
    """Return the experiment's rounds artifact directory."""
    return agentic_dir(experiment) / "rounds"


def state_path(experiment: ExperimentRecord) -> Path:
    """Return the experiment's state.json path."""
    return agentic_dir(experiment) / STATE_FILENAME


def journal_path(experiment: ExperimentRecord) -> Path:
    """Return the experiment's journal.jsonl path."""
    return agentic_dir(experiment) / JOURNAL_FILENAME


def experiment_markdown_path(experiment: ExperimentRecord) -> Path:
    """Return the experiment's EXPERIMENT.md path."""
    return experiment.manifest_path.parent / "EXPERIMENT.md"


def program_path(experiment: ExperimentRecord) -> Path:
    """Resolve the active program file (default PROGRAM.md or a named custom program)."""
    raw = experiment.metadata.get(PROGRAM_METADATA_KEY)
    if raw is None:
        return PROGRAM_PATH
    if not isinstance(raw, str) or not raw.strip():
        raise AgenticResearchValidationError("agentic_research_program_invalid")
    name = raw.strip()
    if Path(name).name != name or not name.endswith(".md"):
        raise AgenticResearchValidationError(f"agentic_research_program_invalid:{name}")
    if name == PROGRAM_PATH.name:
        return PROGRAM_PATH
    path = CUSTOM_PROGRAM_DIR / name
    if not path.is_file():
        raise AgenticResearchValidationError(f"agentic_research_program_missing:{name}")
    return path


def latest_round_markdown(experiment: ExperimentRecord) -> str | None:
    """Return the most recent rounds/rN.md content, capped at the context char limit."""
    directory = rounds_dir(experiment)
    if not directory.is_dir():
        return None
    candidates = [path for path in directory.glob("r*.md") if re.fullmatch(r"r\d{3,}\.md", path.name)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: int(p.stem[1:]))
    return read_text(latest, limit=MAX_CONTEXT_CHARS)


def read_text(path: Path, *, limit: int) -> str | None:
    """Read a text file, truncating to `limit` chars; None if absent or unreadable."""
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def write_json(path: Path, payload: object) -> None:
    """Write JSON atomically (temp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    """Write text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def unique_config_path(config_dir: Path, filename: str) -> Path:
    """Return a non-colliding path for a config filename in config_dir."""
    path = config_dir / filename
    if not path.exists():
        return path
    stem, suffix, index = path.stem, path.suffix, 2
    while True:
        candidate = config_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _value(value: object) -> str:
    return "none" if value is None else str(value)
