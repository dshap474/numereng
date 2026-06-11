"""Durable memory for the rebuilt research loop."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from numereng.features.agentic_research import types as ar_types
from numereng.features.experiments import ExperimentRecord

read_text = ar_types.read_text
write_json = ar_types.write_json
write_text = ar_types.write_text

_STATE_DEFAULTS: dict[str, object] = {
    "schema_version": ar_types.STATE_SCHEMA_VERSION,
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
    for key, value in _STATE_DEFAULTS.items():
        state.setdefault(key, deepcopy(value))
    state["schema_version"] = ar_types.STATE_SCHEMA_VERSION
    return state


def initial_state(experiment: ExperimentRecord) -> dict[str, object]:
    now = ar_types.utc_now_iso()
    return apply_state_defaults(
        {
            "experiment_id": experiment.experiment_id,
            "best_overall": asdict(ar_types.ResearchBestRun()),
            "created_at": now,
            "updated_at": now,
        }
    )


def load_state(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ar_types.AgenticResearchValidationError(f"agentic_research_state_invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise ar_types.AgenticResearchValidationError(f"agentic_research_state_invalid:{path}")
    return apply_state_defaults(payload)


def save_state(experiment: ExperimentRecord, state: dict[str, object]) -> None:
    write_json(state_path(experiment), state)


def heartbeat(state: dict[str, object]) -> None:
    state["last_heartbeat"] = ar_types.utc_now_iso()


def append_journal(experiment: ExperimentRecord, entry: dict[str, object]) -> None:
    path = journal_path(experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def journal_tail(experiment: ExperimentRecord, *, limit: int) -> list[dict[str, object]]:
    return _journal_entries(journal_path(experiment))[-limit:]


def journal_has_recorded_run(experiment: ExperimentRecord, config_name: str) -> bool:
    return any(
        entry.get("config") == config_name and entry.get("run_id") and entry.get("status") != "failed"
        for entry in _journal_entries(journal_path(experiment))
    )


def _journal_entries(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    entries: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def write_round_markdown(experiment: ExperimentRecord, entry: dict[str, object], *, memo: str | None) -> None:
    round_label = str(entry.get("round_label"))
    lines = [memo.rstrip() if memo and memo.strip() else f"# {round_label} Research State"]
    lines.extend(
        (
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
            f"- {ar_types.PRIMARY_METRIC_FIELD}: {_value(entry.get('metric'))}",
            f"- champion: {'yes' if entry.get('is_champion') else 'no'}",
            f"- wall: {_value(entry.get('wall_seconds'))}",
        )
    )
    if entry.get("error"):
        lines.append(f"- error: {entry.get('error')}")
    write_text(rounds_dir(experiment) / f"{round_label}.md", "\n".join(lines).rstrip() + "\n")


def write_experiment_markdown(experiment: ExperimentRecord, content: str | None) -> int:
    if not content:
        return 0
    path = experiment_markdown_path(experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return len(content)


def write_failure_debug(
    *, artifact_dir: Path, round_label: str, prompt: str, error: str, raw_response: str | None = None
) -> None:
    prefix = artifact_dir / f"{round_label}.debug"
    write_text(Path(f"{prefix}.prompt.md"), prompt)
    write_text(Path(f"{prefix}.error.txt"), error.strip() + "\n")
    if raw_response is not None:
        write_text(Path(f"{prefix}.llm_response.txt"), raw_response)


def agentic_dir(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / ar_types.AGENTIC_DIRNAME


def rounds_dir(experiment: ExperimentRecord) -> Path:
    return agentic_dir(experiment) / "rounds"


def state_path(experiment: ExperimentRecord) -> Path:
    return agentic_dir(experiment) / ar_types.STATE_FILENAME


def journal_path(experiment: ExperimentRecord) -> Path:
    return agentic_dir(experiment) / ar_types.JOURNAL_FILENAME


def experiment_markdown_path(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / "EXPERIMENT.md"


def program_path(experiment: ExperimentRecord) -> Path:
    raw = experiment.metadata.get(ar_types.PROGRAM_METADATA_KEY)
    if raw is None:
        return ar_types.PROGRAM_PATH
    if not isinstance(raw, str) or not raw.strip():
        raise ar_types.AgenticResearchValidationError("agentic_research_program_invalid")
    name = raw.strip()
    if Path(name).name != name or not name.endswith(".md"):
        raise ar_types.AgenticResearchValidationError(f"agentic_research_program_invalid:{name}")
    if name == ar_types.PROGRAM_PATH.name:
        return ar_types.PROGRAM_PATH
    path = ar_types.CUSTOM_PROGRAM_DIR / name
    if not path.is_file():
        raise ar_types.AgenticResearchValidationError(f"agentic_research_program_missing:{name}")
    return path


def latest_round_markdown(experiment: ExperimentRecord) -> str | None:
    directory = rounds_dir(experiment)
    if not directory.is_dir():
        return None
    candidates = [path for path in directory.glob("r*.md") if re.fullmatch(r"r\d{3,}\.md", path.name)]
    if not candidates:
        return None
    return read_text(max(candidates, key=lambda path: int(path.stem[1:])), limit=ar_types.MAX_CONTEXT_CHARS)


def _value(value: object) -> str:
    return "none" if value is None else str(value)
