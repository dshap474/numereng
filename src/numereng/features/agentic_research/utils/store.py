"""State persistence helpers for the agentic research supervisor."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from numereng.features.agentic_research.utils.programs import (
    legacy_builtin_program,
    load_program_details,
    program_markdown_sha256,
    render_session_program_markdown,
)
from numereng.features.agentic_research.utils.types import (
    CodexConfigPayload,
    CodexDecision,
    ResearchBestRun,
    ResearchLineageLink,
    ResearchLineageState,
    ResearchPathState,
    ResearchPhaseState,
    ResearchProgramConfigPolicy,
    ResearchProgramDefinition,
    ResearchProgramMetricPolicy,
    ResearchProgramPhase,
    ResearchProgramRoundPolicy,
    ResearchProgramState,
    ResearchRoundState,
)
from numereng.features.experiments import ExperimentRecord
from numereng.features.store import StoreError, upsert_experiment
from numereng.platform.clients.openrouter import active_model_source, load_openrouter_config

_AGENTIC_RESEARCH_DIRNAME = "agentic_research"
_PROGRAM_FILENAME = "program.json"
_LINEAGE_FILENAME = "lineage.json"
_SESSION_PROGRAM_FILENAME = "session_program.md"
_LLM_TRACE_FILENAME = "llm_trace.jsonl"
_LLM_TRACE_MARKDOWN_FILENAME = "llm_trace.md"
_ROUND_RECORD_FILENAME = "round.json"
_ROUND_MARKDOWN_FILENAME = "round.md"


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


def session_program_path(agentic_research_dir: Path) -> Path:
    """Return the canonical persisted session program source path."""
    return agentic_research_dir / _SESSION_PROGRAM_FILENAME


def llm_trace_path(agentic_research_dir: Path) -> Path:
    """Return the canonical append-only planner trace log path."""
    return agentic_research_dir / _LLM_TRACE_FILENAME


def llm_trace_markdown_path(agentic_research_dir: Path) -> Path:
    """Return the canonical human-readable planner trace markdown path."""
    return agentic_research_dir / _LLM_TRACE_MARKDOWN_FILENAME


def round_dir(agentic_research_dir: Path, round_label: str) -> Path:
    """Return one round artifact directory path."""
    return agentic_research_dir / "rounds" / round_label


def round_record_path(round_artifact_dir: Path) -> Path:
    """Return the canonical per-round machine-readable artifact path."""
    return round_artifact_dir / _ROUND_RECORD_FILENAME


def round_markdown_path(round_artifact_dir: Path) -> Path:
    """Return the canonical per-round human-readable artifact path."""
    return round_artifact_dir / _ROUND_MARKDOWN_FILENAME


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


def decision_from_dict(payload: dict[str, object]) -> CodexDecision:
    """Build one Codex decision object from decoded JSON."""
    configs = [
        CodexConfigPayload(
            filename=_as_str(item.get("filename")) or "",
            rationale=_as_str(item.get("rationale")) or "",
            overrides=_overrides_from_obj(item.get("overrides")),
        )
        for item in _dict_items(payload.get("configs"))
    ]
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
    program_snapshot = _program_definition_from_obj(payload.get("program_snapshot"))
    program_id, program_title, program_source, program_sha256, program_snapshot = _resolved_program_identity(
        payload,
        program_snapshot=program_snapshot,
    )
    return ResearchProgramState(
        root_experiment_id=_as_str(payload.get("root_experiment_id")) or "",
        program_experiment_id=_as_str(payload.get("program_experiment_id")) or "",
        program_id=program_id,
        program_title=program_title,
        program_source=program_source,
        program_sha256=program_sha256,
        program_snapshot=program_snapshot,
        status=cast(str, _as_str(payload.get("status")) or "initialized"),
        active_path_id=_as_str(payload.get("active_path_id")) or "",
        active_experiment_id=_as_str(payload.get("active_experiment_id")) or "",
        next_round_number=_as_int(payload.get("next_round_number"), default=1),
        total_rounds_completed=_as_int(payload.get("total_rounds_completed"), default=0),
        total_paths_created=_as_int(payload.get("total_paths_created"), default=1),
        improvement_threshold=_as_float(
            payload.get("improvement_threshold"),
            default=program_snapshot.improvement_threshold_default,
        ),
        scoring_stage=cast(str, _as_str(payload.get("scoring_stage")) or program_snapshot.scoring_stage),
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


def _resolved_program_identity(
    payload: dict[str, object],
    *,
    program_snapshot: ResearchProgramDefinition | None,
) -> tuple[str, str, str, str, ResearchProgramDefinition]:
    program_id = _as_str(payload.get("program_id"))
    program_title = _as_str(payload.get("program_title"))
    program_source = cast(str | None, _as_str(payload.get("program_source")))
    program_sha256 = _as_str(payload.get("program_sha256"))
    if program_snapshot is None:
        legacy_program_id = _as_str(payload.get("strategy"))
        selector = program_id or legacy_program_id
        if selector is None:
            raise ValueError("agentic_research_program_selector_missing")
        details = load_program_details(selector)
        if legacy_program_id is not None and program_id is None:
            program_snapshot = legacy_builtin_program(legacy_program_id)
            program_source = "legacy_builtin"
        else:
            program_snapshot = details.definition
            program_source = program_source or program_snapshot.source
        program_id = program_snapshot.program_id
        program_title = program_snapshot.title
        program_sha256 = program_markdown_sha256(details.raw_markdown)
    return (
        program_id or program_snapshot.program_id,
        program_title or program_snapshot.title,
        cast(str, program_source or program_snapshot.source),
        program_sha256 or "",
        program_snapshot,
    )


def _lineage_from_dict(payload: dict[str, object]) -> ResearchLineageState:
    links = [
        ResearchLineageLink(
            path_id=_as_str(item.get("path_id")) or "",
            experiment_id=_as_str(item.get("experiment_id")) or "",
            parent_experiment_id=_as_str(item.get("parent_experiment_id")),
            generation=_as_int(item.get("generation"), default=0),
            source_round=_as_str(item.get("source_round")),
            pivot_reason=_as_str(item.get("pivot_reason")),
            created_at=_as_str(item.get("created_at")) or utc_now_iso(),
        )
        for item in _dict_items(payload.get("links"))
    ]
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
        parent_run_id=_as_str(payload.get("parent_run_id")),
        parent_config_filename=_as_str(payload.get("parent_config_filename")),
        change_set=_as_dict_list(payload.get("change_set")),
        llm_rationale=_as_str(payload.get("llm_rationale")),
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


def _program_definition_from_obj(payload: object) -> ResearchProgramDefinition | None:
    if not isinstance(payload, dict):
        return None
    program_id = _as_str(payload.get("program_id"))
    title = _as_str(payload.get("title"))
    description = _as_str(payload.get("description"))
    planner_contract = _as_str(payload.get("planner_contract"))
    scoring_stage = _as_str(payload.get("scoring_stage"))
    metric_policy = _metric_policy_from_obj(payload.get("metric_policy"))
    round_policy = _round_policy_from_obj(payload.get("round_policy"))
    config_policy = _config_policy_from_obj(payload.get("config_policy"))
    prompt_template = _as_str(payload.get("prompt_template"))
    if (
        program_id is None
        or title is None
        or description is None
        or planner_contract is None
        or scoring_stage is None
        or metric_policy is None
        or round_policy is None
        or config_policy is None
        or prompt_template is None
    ):
        return None
    return ResearchProgramDefinition(
        program_id=program_id,
        title=title,
        description=description,
        source=cast(str, _as_str(payload.get("source")) or "builtin"),
        planner_contract=cast(str, planner_contract),
        scoring_stage=cast(str, scoring_stage),
        metric_policy=metric_policy,
        round_policy=round_policy,
        improvement_threshold_default=_as_float(payload.get("improvement_threshold_default"), default=0.0002) or 0.0002,
        config_policy=config_policy,
        prompt_template=prompt_template,
        phases=_program_phases_from_obj(payload.get("phases")),
        source_path=_as_str(payload.get("source_path")),
    )


def _metric_policy_from_obj(payload: object) -> ResearchProgramMetricPolicy | None:
    if not isinstance(payload, dict):
        return None
    primary = _as_str(payload.get("primary"))
    tie_break = _as_str(payload.get("tie_break"))
    if primary is None or tie_break is None:
        return None
    return ResearchProgramMetricPolicy(
        primary=primary,
        tie_break=tie_break,
        sanity_checks=tuple(_as_str_list(payload.get("sanity_checks"))),
    )


def _round_policy_from_obj(payload: object) -> ResearchProgramRoundPolicy | None:
    if not isinstance(payload, dict):
        return None
    return ResearchProgramRoundPolicy(
        plateau_non_improving_rounds=_as_int(payload.get("plateau_non_improving_rounds"), default=2),
        require_scale_confirmation=_as_bool(payload.get("require_scale_confirmation")),
        scale_confirmation_rounds=_as_int(payload.get("scale_confirmation_rounds"), default=0),
    )


def _config_policy_from_obj(payload: object) -> ResearchProgramConfigPolicy | None:
    if not isinstance(payload, dict):
        return None
    allowed_paths = _as_str_list(payload.get("allowed_paths"))
    max_candidate_configs = _as_int(payload.get("max_candidate_configs"), default=0)
    if not allowed_paths or max_candidate_configs <= 0:
        return None
    return ResearchProgramConfigPolicy(
        allowed_paths=tuple(allowed_paths),
        min_candidate_configs=_as_optional_int(payload.get("min_candidate_configs")),
        max_candidate_configs=max_candidate_configs,
        min_changes=_as_optional_int(payload.get("min_changes")),
        max_changes=_as_optional_int(payload.get("max_changes")),
    )


def _program_phases_from_obj(payload: object) -> tuple[ResearchProgramPhase, ...]:
    phases: list[ResearchProgramPhase] = []
    for item in _dict_items(payload):
        phase_id = _as_str(item.get("phase_id"))
        title = _as_str(item.get("title"))
        if phase_id is None or title is None:
            continue
        phases.append(
            ResearchProgramPhase(
                phase_id=phase_id,
                title=title,
                summary=_as_str(item.get("summary")) or "",
                gate=_as_str(item.get("gate")) or "",
            )
        )
    return tuple(phases)


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
    return [
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
            scale_confirmation_rounds_completed=_as_int(
                item.get("scale_confirmation_rounds_completed"),
                default=0,
            ),
        )
        for item in _dict_items(payload)
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json_mapping(path: Path, *, error_code: str) -> dict[str, object]:
    payload = _read_json_dict_or_none(path)
    if payload is None:
        raise ValueError(f"{error_code}:{path}")
    return payload


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


def _as_dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            items.append(cast(dict[str, object], item))
    return items


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


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
    for item in _dict_items(value):
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


def _dict_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]


def _payload_section(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def load_and_persist_program_state(*, auto_dir: Path) -> ResearchProgramState:
    state_path = program_path(auto_dir)
    state = load_program_state(state_path)
    session_source_path = session_program_path(auto_dir)
    session_markdown: str | None = None
    if not session_source_path.is_file():
        session_markdown = _session_program_markdown(state)
        save_text_artifact(session_source_path, session_markdown)
    if not state.program_sha256:
        if session_markdown is None:
            session_markdown = session_source_path.read_text(encoding="utf-8")
        state = replace(
            state,
            program_sha256=program_markdown_sha256(session_markdown),
        )
    elif state.program_snapshot.source != state.program_source:
        state = replace(
            state,
            program_snapshot=replace(state.program_snapshot, source=state.program_source),
        )
    save_program_state(state_path, state)
    return state


def _session_program_markdown(state: ResearchProgramState) -> str:
    if state.program_source == "legacy_builtin":
        return load_program_details(state.program_id).raw_markdown
    return render_session_program_markdown(state.program_snapshot)


_LEGACY_ROUND_ARTIFACT_FILENAMES = (
    "codex_prompt.txt",
    "codex_usage.json",
    "codex_stdout.jsonl",
    "codex_stderr.txt",
    "codex_last_message.json",
    "codex_last_message.txt",
    "codex_decision.json",
    "planned_configs.json",
    "report.json",
    "round_summary.json",
    "llm_trace.jsonl",
    "llm_trace.md",
    "codex_failure.txt",
    "codex_validation_error.txt",
)


def _planner_attempt_payloads(executions: list[Any]) -> list[dict[str, object]]:
    return [
        {
            "attempt_number": attempt.attempt_number,
            "elapsed_seconds": attempt.elapsed_seconds,
            "returncode": attempt.returncode,
            "thread_id": attempt.thread_id,
            "input_tokens": attempt.input_tokens,
            "cached_input_tokens": attempt.cached_input_tokens,
            "output_tokens": attempt.output_tokens,
            "stdout_line_count": attempt.stdout_line_count,
            "validation_feedback": attempt.validation_feedback,
        }
        for execution in executions
        for attempt in execution.attempts
    ]


def _planner_usage_payload(attempts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "attempts": attempts,
        "final_attempt": attempts[-1] if attempts else None,
        "total_input_tokens": sum(item["input_tokens"] or 0 for item in attempts),
        "total_cached_input_tokens": sum(item["cached_input_tokens"] or 0 for item in attempts),
        "total_output_tokens": sum(item["output_tokens"] or 0 for item in attempts),
        "total_elapsed_seconds": round(sum(float(item["elapsed_seconds"]) for item in attempts), 6),
    }


def planner_trace_payload(
    *,
    round_state: Any,
    program_state: ResearchProgramState,
    round_artifact_dir: Path,
    session_source_path: Path,
    prompt_text: str,
    executions: list[Any],
    status: str,
    error: str | None = None,
) -> dict[str, object]:
    source = active_model_source()
    attempts = _planner_attempt_payloads(executions)
    parsed_response = _trace_parsed_response(executions[-1]) if executions else None
    return {
        "timestamp": utc_now_iso(),
        "event": "planner_trace",
        "status": status,
        "planner_source": source,
        "planner_model": _planner_model_name(source),
        "program_id": program_state.program_id,
        "program_sha256": program_state.program_sha256,
        "session_program_path": str(session_source_path),
        "experiment_id": round_state.experiment_id,
        "path_id": round_state.path_id,
        "round_label": round_state.round_label,
        "round_number": round_state.round_number,
        "round_path": str(round_record_path(round_artifact_dir)),
        "round_markdown_path": str(round_markdown_path(round_artifact_dir)),
        "attempts": attempts,
        "usage": _planner_usage_payload(attempts),
        "prompt_text": prompt_text,
        "raw_response_text": _trace_response_text(executions[-1]) if executions else "",
        "parsed_response": parsed_response,
        "decision": parsed_response,
        "error": error,
    }


def append_planner_trace(*, auto_dir: Path, payload: dict[str, object]) -> None:
    append_jsonl_artifact(llm_trace_path(auto_dir), payload)
    append_text_artifact(llm_trace_markdown_path(auto_dir), _render_planner_trace_markdown(payload))


def _render_planner_trace_markdown(payload: dict[str, object]) -> str:
    prompt_text = str(payload.get("prompt_text") or "")
    raw_response_text = str(payload.get("raw_response_text") or "")
    usage = payload.get("usage")
    parsed_response = payload.get("parsed_response")
    lines = [
        f"## {payload.get('timestamp', '')} {payload.get('round_label', '')}",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Planner source: `{payload.get('planner_source', '')}`",
        f"- Planner model: `{payload.get('planner_model', '')}`",
        (f"- Round record: `{payload.get('round_path', '')}`" if payload.get("round_path") else ""),
    ]
    _append_code_block(lines, "### Sent To LLM", prompt_text)
    _append_code_block(lines, "### Raw LLM Response", raw_response_text)
    _append_json_block(lines, "### Parsed Final Response", parsed_response)
    _append_json_block(lines, "### Usage", usage)
    error = payload.get("error")
    if error:
        lines.extend(["", "### Error", "", str(error)])
    lines.extend(["", "---", ""])
    return "\n".join(line for line in lines if line != "")


def _trace_response_text(execution: Any) -> str:
    raw_response_text = getattr(execution, "raw_response_text", "")
    if isinstance(raw_response_text, str) and raw_response_text.strip():
        return raw_response_text
    stdout_jsonl = getattr(execution, "stdout_jsonl", "")
    if isinstance(stdout_jsonl, str) and stdout_jsonl.strip():
        return stdout_jsonl
    if isinstance(raw_response_text, str):
        return raw_response_text
    return ""


def _trace_parsed_response(execution: Any) -> dict[str, object] | None:
    last_message = getattr(execution, "last_message", None)
    return last_message if isinstance(last_message, dict) else None


def _planner_model_name(source: str) -> str:
    if source != "openrouter":
        return source
    try:
        return load_openrouter_config().active_model
    except Exception:  # noqa: BLE001
        return source


def _append_code_block(lines: list[str], title: str, body: object, *, language: str = "text") -> None:
    if not isinstance(body, str) or not body.strip():
        return
    lines.extend(["", title, "", f"```{language}", body.strip(), "```"])


def _append_json_block(lines: list[str], title: str, payload: object) -> None:
    if not isinstance(payload, dict):
        return
    lines.extend(["", title, "", "```json", json.dumps(payload, indent=2, sort_keys=True), "```"])


def save_round_bundle(
    *,
    round_artifact_dir: Path,
    round_state: Any,
    program_state: ResearchProgramState,
    session_source_path: Path,
    planner_payload: dict[str, object] | None = None,
    results_payload: dict[str, object] | None = None,
) -> None:
    existing = load_round_bundle(round_artifact_dir)
    if planner_payload is None and isinstance(existing.get("planner"), dict):
        planner_payload = existing.get("planner")
    if results_payload is None and isinstance(existing.get("results"), dict):
        results_payload = existing.get("results")
    payload = {
        "round_number": round_state.round_number,
        "round_label": round_state.round_label,
        "experiment_id": round_state.experiment_id,
        "path_id": round_state.path_id,
        "status": round_state.status,
        "started_at": round_state.started_at,
        "updated_at": round_state.updated_at,
        "program": {
            "program_id": program_state.program_id,
            "program_title": program_state.program_title,
            "program_source": program_state.program_source,
            "program_sha256": program_state.program_sha256,
            "session_program_path": str(session_source_path),
        },
        "decision": {
            "action": round_state.decision_action,
            "experiment_question": round_state.experiment_question,
            "winner_criteria": round_state.winner_criteria,
            "decision_rationale": round_state.decision_rationale,
            "path_hypothesis": round_state.decision_path_hypothesis,
            "path_slug": round_state.decision_path_slug,
            "phase_id": round_state.phase_id,
            "phase_action": round_state.phase_action,
            "phase_transition_rationale": round_state.phase_transition_rationale,
        },
        "lineage": {
            "parent_run_id": round_state.parent_run_id,
            "parent_config_filename": round_state.parent_config_filename,
            "child_config_filenames": list(round_state.config_filenames),
            "change_set": list(round_state.change_set),
            "llm_rationale": round_state.llm_rationale,
        },
        "execution": {
            "run_ids": list(round_state.run_ids),
            "next_config_index": round_state.next_config_index,
        },
        "planner": planner_payload,
        "results": results_payload,
    }
    save_round_artifact(round_record_path(round_artifact_dir), payload)
    save_text_artifact(round_markdown_path(round_artifact_dir), render_round_markdown(payload))
    _cleanup_legacy_round_artifacts(round_artifact_dir)


def load_round_bundle(round_artifact_dir: Path) -> dict[str, object]:
    return _read_json_dict_or_none(round_record_path(round_artifact_dir)) or {}


def render_round_markdown(payload: dict[str, object]) -> str:
    decision = _payload_section(payload, "decision")
    lineage = _payload_section(payload, "lineage")
    execution = _payload_section(payload, "execution")
    planner = _payload_section(payload, "planner")
    results = _payload_section(payload, "results")
    lines = [
        f"# {payload.get('round_label', '')}",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Experiment: `{payload.get('experiment_id', '')}`",
        f"- Path: `{payload.get('path_id', '')}`",
        f"- Decision action: `{decision.get('action') or 'n/a'}`",
    ]
    parent_config = lineage.get("parent_config_filename")
    if parent_config:
        lines.append(f"- Parent config: `{parent_config}`")
    parent_run_id = lineage.get("parent_run_id")
    if parent_run_id:
        lines.append(f"- Parent run: `{parent_run_id}`")
    child_configs = lineage.get("child_config_filenames")
    if isinstance(child_configs, list) and child_configs:
        lines.append(f"- Child config(s): `{', '.join(str(item) for item in child_configs)}`")
    run_ids = execution.get("run_ids")
    if isinstance(run_ids, list) and run_ids:
        lines.append(f"- Run id(s): `{', '.join(str(item) for item in run_ids)}`")
    results_best = results.get("best_row") if isinstance(results.get("best_row"), dict) else None
    if results_best is not None:
        lines.extend(
            [
                "",
                "## Outcome",
                "",
                f"- Improved best overall: `{results.get('improved_best_overall')}`",
                f"- Best run: `{results_best.get('run_id')}`",
                f"- `bmc_last_200_eras_mean`: `{results_best.get('bmc_last_200_eras_mean')}`",
                f"- `bmc_mean`: `{results_best.get('bmc_mean')}`",
                f"- `corr_mean`: `{results_best.get('corr_mean')}`",
            ]
        )
    change_set = lineage.get("change_set")
    if isinstance(change_set, list) and change_set:
        lines.extend(["", "## Change Set", ""])
        for item in change_set:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{item.get('path')}` = `{json.dumps(item.get('value'), sort_keys=True)}`")
    llm_rationale = lineage.get("llm_rationale")
    if isinstance(llm_rationale, str) and llm_rationale.strip():
        lines.extend(["", "## LLM Rationale", "", llm_rationale.strip()])
    _append_code_block(lines, "## Sent To LLM", planner.get("prompt_text"))
    _append_code_block(lines, "## Raw LLM Response", planner.get("raw_response_text"))
    _append_json_block(lines, "## Parsed Final Response", planner.get("parsed_response"))
    _append_json_block(lines, "## Usage", planner.get("usage"))
    planner_error = planner.get("error")
    if planner_error:
        lines.extend(["", "## Error", "", str(planner_error)])
    return "\n".join(lines) + "\n"


def _cleanup_legacy_round_artifacts(round_artifact_dir: Path) -> None:
    for filename in _LEGACY_ROUND_ARTIFACT_FILENAMES:
        path = round_artifact_dir / filename
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            continue


def recent_round_summaries(*, auto_dir: Path) -> list[dict[str, object]]:
    rounds_root = auto_dir / "rounds"
    if not rounds_root.is_dir():
        return []
    items: list[dict[str, object]] = []
    for path in sorted(rounds_root.iterdir(), key=lambda item: item.name):
        payload = _read_json_dict_or_none(round_record_path(path))
        if payload is not None:
            items.append(_round_summary_view(payload))
            continue
        legacy_payload = _read_json_dict_or_none(path / "round_summary.json")
        if legacy_payload is not None:
            items.append(legacy_payload)
    return items


def _read_json_dict_or_none(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(dict[str, object], payload) if isinstance(payload, dict) else None


def _round_summary_view(payload: dict[str, object]) -> dict[str, object]:
    decision = _payload_section(payload, "decision")
    lineage = _payload_section(payload, "lineage")
    results = _payload_section(payload, "results")
    return {
        "round_label": payload.get("round_label"),
        "experiment_id": payload.get("experiment_id"),
        "path_id": payload.get("path_id"),
        "decision_action": decision.get("action"),
        "experiment_question": decision.get("experiment_question"),
        "winner_criteria": decision.get("winner_criteria"),
        "decision_rationale": decision.get("decision_rationale"),
        "parent_run_id": lineage.get("parent_run_id"),
        "parent_config_filename": lineage.get("parent_config_filename"),
        "change_set": lineage.get("change_set") if isinstance(lineage.get("change_set"), list) else [],
        "llm_rationale": lineage.get("llm_rationale"),
        "best_row": results.get("best_row") if isinstance(results.get("best_row"), dict) else None,
    }


def agentic_research_dir(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / _AGENTIC_RESEARCH_DIRNAME


def upsert_agentic_research_metadata(
    *,
    root: Path,
    experiment: ExperimentRecord,
    metadata_update: dict[str, object],
) -> None:
    manifest_path = experiment.manifest_path
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"agentic_research_manifest_invalid:{manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"agentic_research_manifest_invalid:{manifest_path}")
    metadata = payload.get("metadata")
    normalized: dict[str, object] = metadata if isinstance(metadata, dict) else {}
    normalized["agentic_research"] = metadata_update
    payload["metadata"] = normalized
    payload["updated_at"] = utc_now_iso()
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        upsert_experiment(
            store_root=root,
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            status=str(payload.get("status", experiment.status)),
            created_at=str(payload.get("created_at", experiment.created_at)),
            updated_at=str(payload["updated_at"]),
            metadata={
                **normalized,
                "hypothesis": payload.get("hypothesis"),
                "tags": list(experiment.tags),
                "champion_run_id": payload.get("champion_run_id"),
                "runs": list(experiment.runs),
            },
        )
    except StoreError as exc:
        raise ValueError("agentic_research_metadata_index_failed") from exc
