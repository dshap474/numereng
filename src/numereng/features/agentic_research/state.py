"""State persistence helpers for the agentic research supervisor."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from numereng.features.agentic_research.contracts import (
    CodexConfigPayload,
    CodexDecision,
    ResearchBestRun,
    ResearchLineageLink,
    ResearchLineageState,
    ResearchPathState,
    ResearchPhaseState,
    ResearchProgramState,
    ResearchRoundState,
)

_PROGRAM_FILENAME = "program.json"
_LINEAGE_FILENAME = "lineage.json"
_LLM_TRACE_FILENAME = "llm_trace.jsonl"
_LLM_TRACE_MARKDOWN_FILENAME = "llm_trace.md"


def utc_now_iso() -> str:
    """Return a canonical UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def ensure_agentic_research_dirs(agentic_research_dir: Path) -> None:
    """Create the canonical directory layout for persisted supervisor state."""
    (agentic_research_dir / "rounds").mkdir(parents=True, exist_ok=True)


def program_path(agentic_research_dir: Path) -> Path:
    """Return the canonical program state path."""
    return agentic_research_dir / _PROGRAM_FILENAME


def lineage_path(agentic_research_dir: Path) -> Path:
    """Return the canonical lineage state path."""
    return agentic_research_dir / _LINEAGE_FILENAME


def llm_trace_path(agentic_research_dir: Path) -> Path:
    """Return the canonical append-only planner trace log path."""
    return agentic_research_dir / _LLM_TRACE_FILENAME


def llm_trace_markdown_path(agentic_research_dir: Path) -> Path:
    """Return the canonical human-readable planner trace markdown path."""
    return agentic_research_dir / _LLM_TRACE_MARKDOWN_FILENAME


def round_dir(agentic_research_dir: Path, round_label: str) -> Path:
    """Return one round artifact directory path."""
    return agentic_research_dir / "rounds" / round_label


def save_program_state(path: Path, state: ResearchProgramState) -> None:
    """Persist the full program state as canonical JSON."""
    _write_json(path, asdict(state))


def load_program_state(path: Path) -> ResearchProgramState:
    """Load one program state from disk."""
    payload = _read_json_mapping(path, error_code="agentic_research_program_invalid")
    return _program_from_dict(payload)


def save_lineage_state(path: Path, state: ResearchLineageState) -> None:
    """Persist the lineage graph as canonical JSON."""
    _write_json(path, asdict(state))


def load_lineage_state(path: Path) -> ResearchLineageState:
    """Load one lineage graph from disk."""
    payload = _read_json_mapping(path, error_code="agentic_research_lineage_invalid")
    return _lineage_from_dict(payload)


def save_round_artifact(path: Path, payload: dict[str, object]) -> None:
    """Persist one round-scoped artifact payload."""
    _write_json(path, payload)


def save_text_artifact(path: Path, payload: str) -> None:
    """Persist one round-scoped text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def append_jsonl_artifact(path: Path, payload: dict[str, object]) -> None:
    """Append one JSON object to a JSONL artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_text_artifact(path: Path, payload: str) -> None:
    """Append text to one artifact file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)


def decision_to_dict(decision: CodexDecision) -> dict[str, object]:
    """Convert one Codex decision into a JSON-safe payload."""
    return asdict(decision)


def decision_from_dict(payload: dict[str, object]) -> CodexDecision:
    """Build one Codex decision object from decoded JSON."""
    configs_raw = payload.get("configs")
    configs: list[CodexConfigPayload] = []
    if isinstance(configs_raw, list):
        for item in configs_raw:
            if not isinstance(item, dict):
                continue
            configs.append(
                CodexConfigPayload(
                    filename=_as_str(item.get("filename")) or "",
                    rationale=_as_str(item.get("rationale")) or "",
                    overrides=_overrides_from_obj(item.get("overrides")),
                )
            )
    return CodexDecision(
        experiment_question=_as_str(payload.get("experiment_question")) or "",
        winner_criteria=_as_str(payload.get("winner_criteria")) or "",
        decision_rationale=_as_str(payload.get("decision_rationale")) or "",
        next_action=cast(str, _as_str(payload.get("next_action")) or "continue"),
        path_hypothesis=_as_str(payload.get("path_hypothesis")) or "",
        path_slug=_as_str(payload.get("path_slug")) or "",
        phase_action=cast(str | None, _as_str(payload.get("phase_action"))),
        phase_transition_rationale=_as_str(payload.get("phase_transition_rationale")),
        configs=configs,
    )


def _program_from_dict(payload: dict[str, object]) -> ResearchProgramState:
    strategy = _as_str(payload.get("strategy"))
    if strategy is None:
        raise ValueError("agentic_research_program_strategy_missing")
    return ResearchProgramState(
        root_experiment_id=_as_str(payload.get("root_experiment_id")) or "",
        program_experiment_id=_as_str(payload.get("program_experiment_id")) or "",
        strategy=cast(str, strategy),
        strategy_description=_as_str(payload.get("strategy_description")) or strategy,
        status=cast(str, _as_str(payload.get("status")) or "initialized"),
        active_path_id=_as_str(payload.get("active_path_id")) or "",
        active_experiment_id=_as_str(payload.get("active_experiment_id")) or "",
        next_round_number=_as_int(payload.get("next_round_number"), default=1),
        total_rounds_completed=_as_int(payload.get("total_rounds_completed"), default=0),
        total_paths_created=_as_int(payload.get("total_paths_created"), default=1),
        improvement_threshold=_as_float(payload.get("improvement_threshold"), default=0.0002),
        scoring_stage=cast(str, _as_str(payload.get("scoring_stage")) or "post_training_full"),
        codex_command=_as_str_list(payload.get("codex_command")),
        last_checkpoint=_as_str(payload.get("last_checkpoint")) or "initialized",
        stop_reason=_as_str(payload.get("stop_reason")),
        current_round=_round_from_obj(payload.get("current_round")),
        current_phase=_phase_from_obj(payload.get("current_phase")),
        best_overall=_best_from_obj(payload.get("best_overall")),
        paths=_paths_from_obj(payload.get("paths")),
        created_at=_as_str(payload.get("created_at")) or utc_now_iso(),
        updated_at=_as_str(payload.get("updated_at")) or utc_now_iso(),
    )


def _lineage_from_dict(payload: dict[str, object]) -> ResearchLineageState:
    links_raw = payload.get("links")
    links: list[ResearchLineageLink] = []
    if isinstance(links_raw, list):
        for item in links_raw:
            if not isinstance(item, dict):
                continue
            links.append(
                ResearchLineageLink(
                    path_id=_as_str(item.get("path_id")) or "",
                    experiment_id=_as_str(item.get("experiment_id")) or "",
                    parent_experiment_id=_as_str(item.get("parent_experiment_id")),
                    generation=_as_int(item.get("generation"), default=0),
                    source_round=_as_str(item.get("source_round")),
                    pivot_reason=_as_str(item.get("pivot_reason")),
                    created_at=_as_str(item.get("created_at")) or utc_now_iso(),
                )
            )
    return ResearchLineageState(
        root_experiment_id=_as_str(payload.get("root_experiment_id")) or "",
        program_experiment_id=_as_str(payload.get("program_experiment_id")) or "",
        active_path_id=_as_str(payload.get("active_path_id")) or "",
        links=links,
    )


def _round_from_obj(payload: object) -> ResearchRoundState | None:
    if not isinstance(payload, dict):
        return None
    return ResearchRoundState(
        round_number=_as_int(payload.get("round_number"), default=0),
        round_label=_as_str(payload.get("round_label")) or "",
        experiment_id=_as_str(payload.get("experiment_id")) or "",
        path_id=_as_str(payload.get("path_id")) or "",
        status=cast(str, _as_str(payload.get("status")) or "planning"),
        next_config_index=_as_int(payload.get("next_config_index"), default=0),
        config_filenames=_as_str_list(payload.get("config_filenames")),
        run_ids=_as_str_list(payload.get("run_ids")),
        decision_action=cast(str | None, _as_str(payload.get("decision_action"))),
        experiment_question=_as_str(payload.get("experiment_question")),
        winner_criteria=_as_str(payload.get("winner_criteria")),
        decision_rationale=_as_str(payload.get("decision_rationale")),
        decision_path_hypothesis=_as_str(payload.get("decision_path_hypothesis")),
        decision_path_slug=_as_str(payload.get("decision_path_slug")),
        phase_id=_as_str(payload.get("phase_id")),
        phase_action=cast(str | None, _as_str(payload.get("phase_action"))),
        phase_transition_rationale=_as_str(payload.get("phase_transition_rationale")),
        started_at=_as_str(payload.get("started_at")),
        updated_at=_as_str(payload.get("updated_at")),
    )


def _phase_from_obj(payload: object) -> ResearchPhaseState | None:
    if not isinstance(payload, dict):
        return None
    phase_id = _as_str(payload.get("phase_id"))
    if phase_id is None:
        return None
    return ResearchPhaseState(
        phase_id=phase_id,
        phase_title=_as_str(payload.get("phase_title")) or phase_id,
        status=cast(str, _as_str(payload.get("status")) or "active"),
        round_count=_as_int(payload.get("round_count"), default=0),
        transition_rationale=_as_str(payload.get("transition_rationale")),
        started_at=_as_str(payload.get("started_at")) or utc_now_iso(),
        updated_at=_as_str(payload.get("updated_at")) or utc_now_iso(),
    )


def _best_from_obj(payload: object) -> ResearchBestRun:
    if not isinstance(payload, dict):
        return ResearchBestRun()
    return ResearchBestRun(
        experiment_id=_as_str(payload.get("experiment_id")),
        run_id=_as_str(payload.get("run_id")),
        bmc_last_200_eras_mean=_as_float(payload.get("bmc_last_200_eras_mean")),
        bmc_mean=_as_float(payload.get("bmc_mean")),
        corr_mean=_as_float(payload.get("corr_mean")),
        mmc_mean=_as_float(payload.get("mmc_mean")),
        cwmm_mean=_as_float(payload.get("cwmm_mean")),
        updated_at=_as_str(payload.get("updated_at")),
    )


def _paths_from_obj(payload: object) -> list[ResearchPathState]:
    if not isinstance(payload, list):
        return []
    items: list[ResearchPathState] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        items.append(
            ResearchPathState(
                path_id=_as_str(item.get("path_id")) or "",
                experiment_id=_as_str(item.get("experiment_id")) or "",
                parent_experiment_id=_as_str(item.get("parent_experiment_id")),
                generation=_as_int(item.get("generation"), default=0),
                hypothesis=_as_str(item.get("hypothesis")) or "",
                status=cast(str, _as_str(item.get("status")) or "active"),
                pivot_reason=_as_str(item.get("pivot_reason")),
                source_round=_as_str(item.get("source_round")),
                rounds_completed=_as_int(item.get("rounds_completed"), default=0),
                plateau_streak=_as_int(item.get("plateau_streak"), default=0),
                scale_confirmation_used=_as_bool(item.get("scale_confirmation_used")),
                needs_scale_confirmation=_as_bool(item.get("needs_scale_confirmation")),
                best_run_id=_as_str(item.get("best_run_id")),
                created_at=_as_str(item.get("created_at")) or utc_now_iso(),
                updated_at=_as_str(item.get("updated_at")) or utc_now_iso(),
            )
        )
    return items


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json_mapping(path: Path, *, error_code: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{error_code}:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{error_code}:{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{error_code}:{path}")
    return cast(dict[str, object], payload)


def _as_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            items.append(stripped)
    return items


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _as_bool(value: object) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _as_float(value: object, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    items: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(key, str):
            items[key] = item
    return items


def _overrides_from_obj(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return _as_mapping(value)
    if not isinstance(value, list):
        return {}
    payload: dict[str, object] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        path = _as_str(item.get("path"))
        value_json = _as_str(item.get("value_json"))
        if path is None or value_json is None:
            continue
        try:
            parsed = json.loads(value_json)
        except json.JSONDecodeError:
            continue
        _set_override_path(payload, path=path, value=parsed)
    return payload


def _set_override_path(payload: dict[str, object], *, path: str, value: object) -> None:
    parts = [part.strip() for part in path.split(".") if part.strip()]
    if not parts:
        return
    cursor = payload
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value
