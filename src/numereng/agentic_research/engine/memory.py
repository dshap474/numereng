"""Durable memory for the rebuilt research loop."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

from numereng.agentic_research.engine import types as ar_types
from numereng.features.experiments import ExperimentRecord

# Round memos are named ``r`` + at least three digits (``r001.md``); debug sidecars such as
# ``r001.debug.prompt.md`` are deliberately excluded by the full-stem match.
_ROUND_LABEL_RE = re.compile(r"r\d{3,}")

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
    "believed_best": None,
    "believed_best_changed_round": None,
    "last_error": None,
    "last_heartbeat": None,
}


def apply_state_defaults(state: dict[str, object]) -> dict[str, object]:
    for key, value in _STATE_DEFAULTS.items():
        state.setdefault(key, deepcopy(value))
    # `best_overall` was a report-derived duplicate of `champion`; older state files still carry it.
    state.pop("best_overall", None)
    state["schema_version"] = ar_types.STATE_SCHEMA_VERSION
    return state


def initial_state(experiment: ExperimentRecord) -> dict[str, object]:
    now = ar_types.utc_now_iso()
    return apply_state_defaults(
        {
            "experiment_id": experiment.experiment_id,
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
    ar_types.write_json(state_path(experiment), state)


def heartbeat(state: dict[str, object]) -> None:
    state["last_heartbeat"] = ar_types.utc_now_iso()


def append_journal(experiment: ExperimentRecord, entry: dict[str, object]) -> None:
    path = journal_path(experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def journal_tail(experiment: ExperimentRecord, *, limit: int) -> list[dict[str, object]]:
    return _journal_entries(journal_path(experiment))[-limit:]


def journal_all(experiment: ExperimentRecord) -> list[dict[str, object]]:
    return _journal_entries(journal_path(experiment))


def journal_has_recorded_run(experiment: ExperimentRecord, config_name: str) -> bool:
    return any(
        entry.get("config") == config_name and entry.get("run_id") and entry.get("status") != "failed"
        for entry in _journal_entries(journal_path(experiment))
    )


def iter_journal_lines(path: Path, *, strict: bool) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield ``(lineno, entry)`` for each non-blank journal line.

    ``strict=False`` is the in-run reader: blank, unparseable, and non-dict lines are skipped.
    ``strict=True`` raises ``JournalLineError`` on the first malformed line; closeout translates it
    into its own error token. The split is deliberate — the loop must survive a torn line, the
    distillers must not.
    """
    if not path.is_file():
        return
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise ar_types.JournalLineError(lineno) from exc
            continue
        if not isinstance(payload, dict):
            if strict:
                raise ar_types.JournalLineError(lineno)
            continue
        yield lineno, payload


def _journal_entries(path: Path) -> list[dict[str, object]]:
    return [entry for _, entry in iter_journal_lines(path, strict=False)]


def write_round_markdown(
    experiment: ExperimentRecord,
    entry: dict[str, object],
    *,
    memo: str | None,
    extra_lines: list[str] | None = None,
) -> None:
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
    lines.extend(extra_lines or ())
    ar_types.write_text(rounds_dir(experiment) / f"{round_label}.md", "\n".join(lines).rstrip() + "\n")


def write_experiment_markdown(experiment: ExperimentRecord, content: str | None) -> None:
    if not content:
        return
    ar_types.write_text(experiment_markdown_path(experiment), content)


def write_failure_debug(
    *, artifact_dir: Path, round_label: str, prompt: str, error: str, raw_response: str | None = None
) -> None:
    prefix = artifact_dir / f"{round_label}.debug"
    ar_types.write_text(Path(f"{prefix}.prompt.md"), prompt)
    ar_types.write_text(Path(f"{prefix}.error.txt"), error.strip() + "\n")
    if raw_response is not None:
        ar_types.write_text(Path(f"{prefix}.llm_response.txt"), raw_response)


def agentic_dir(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / ar_types.AGENTIC_DIRNAME


def rounds_dir(experiment: ExperimentRecord) -> Path:
    return agentic_dir(experiment) / "rounds"


def configs_dir(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / "configs"


def round_label(round_number: int) -> str:
    return f"r{round_number:03d}"


def parse_round_label(stem: str) -> int | None:
    """Parse a round-memo stem (``r`` + at least three digits) to its round number, else ``None``."""
    return int(stem[1:]) if _ROUND_LABEL_RE.fullmatch(stem) else None


def state_path(experiment: ExperimentRecord) -> Path:
    return agentic_dir(experiment) / ar_types.STATE_FILENAME


def journal_path(experiment: ExperimentRecord) -> Path:
    return agentic_dir(experiment) / ar_types.JOURNAL_FILENAME


def scout_digest_path(experiment: ExperimentRecord) -> Path:
    """Out-of-band research digest a human-side scout may drop next to the run state."""
    return agentic_dir(experiment) / ar_types.SCOUT_DIGEST_FILENAME


def experiment_markdown_path(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / "EXPERIMENT.md"


def strategy_path(experiment: ExperimentRecord) -> Path:
    """The experiment's own brief at a fixed filename, else the tracked generic brief."""
    experiment_strategy = agentic_dir(experiment) / ar_types.STRATEGY_FILENAME
    return experiment_strategy if experiment_strategy.is_file() else ar_types.DEFAULT_STRATEGY_PATH


def latest_round_markdown(experiment: ExperimentRecord) -> str | None:
    directory = rounds_dir(experiment)
    if not directory.is_dir():
        return None
    candidates = [path for path in directory.glob("r*.md") if parse_round_label(path.stem) is not None]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: parse_round_label(path.stem) or 0)
    return ar_types.read_text(latest, limit=ar_types.MAX_CONTEXT_CHARS)


def _value(value: object) -> str:
    return "none" if value is None else str(value)
