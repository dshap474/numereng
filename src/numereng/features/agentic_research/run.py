"""Minimal agentic research loop: prompt LLM for config mutations, then run numereng."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from numereng.config.training import TrainingConfig, load_training_config_json
from numereng.features.experiments import (
    ExperimentError,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    get_experiment,
    report_experiment,
    score_experiment_round,
    train_experiment,
)
from numereng.features.store import resolve_store_root
from numereng.features.telemetry import bind_launch_metadata
from numereng.features.training.run_store import compute_config_hash
from numereng.platform.clients.openrouter import OpenRouterClient, active_model_source
from numereng.platform.errors import OpenRouterClientError

ResearchStatus = Literal["initialized", "running", "interrupted", "stopped", "failed"]
ResearchAction = Literal["baseline", "run", "stop"]

PROGRAM_PATH = Path(__file__).with_name("PROGRAM.md")
AGENTIC_DIRNAME = "agentic_research"
STATE_FILENAME = "state.json"
LEDGER_FILENAME = "ledger.jsonl"
PRIMARY_METRIC = "bmc_last_200_eras.mean"
PRIMARY_METRIC_FIELD = "bmc_last_200_eras_mean"
SCORING_STAGE = "post_training_full"
MAX_CONTEXT_CHARS = 12_000
ALLOWED_CHANGE_PATHS = (
    "data.feature_set",
    "data.target_col",
    "data.scoring_targets",
    "data.target_horizon",
    "preprocessing.nan_missing_all_twos",
    "preprocessing.missing_value",
    "model.type",
    "model.device",
    "model.params.*",
    "model.x_groups",
    "model.data_needed",
    "model.target_transform.*",
    "training.engine.profile",
    "training.engine.window_size_eras",
    "training.engine.embargo_eras",
    "training.resources.parallel_folds",
    "training.resources.max_threads_per_worker",
)


class AgenticResearchError(Exception):
    """Base error for agentic research workflows."""


class AgenticResearchValidationError(AgenticResearchError):
    """Raised when an LLM decision or local research state is invalid."""


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
    stop_reason: str | None
    best_overall: ResearchBestRun
    agentic_research_dir: Path
    state_path: Path
    ledger_path: Path
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

    action: Literal["run", "stop"]
    learning: str
    belief_update: str
    next_hypothesis: str | None
    parent_config: str | None
    changes: tuple[ResearchChange, ...]
    stop_reason: str | None


def get_research_status(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> ResearchStatusResult:
    """Return current research-loop state, synthesizing a blank state if needed."""
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    state = _load_state(_state_path(experiment)) or _initial_state(experiment)
    best = _best_run_from_report(_safe_report(root=root, experiment_id=experiment.experiment_id))
    return _status_result(experiment=experiment, state=state, best=best)


def run_research(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    max_rounds: int = 1,
) -> ResearchRunResult:
    """Run one or more config-mutation research rounds in the foreground."""
    if max_rounds < 1:
        raise AgenticResearchValidationError("agentic_research_max_rounds_invalid")

    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    if experiment.status == "archived":
        raise AgenticResearchValidationError("agentic_research_experiment_archived")

    auto_dir = _agentic_dir(experiment)
    auto_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(_state_path(experiment)) or _initial_state(experiment)
    state["status"] = "running"
    state["stop_reason"] = None
    _save_state(experiment, state)

    rounds: list[ResearchRoundResult] = []
    try:
        for _ in range(max_rounds):
            if _stopped_by_llm(state):
                break
            result = _run_one_round(root=root, experiment_id=experiment.experiment_id, state=state)
            rounds.append(result)
            experiment = get_experiment(store_root=root, experiment_id=experiment_id)
            state = _load_state(_state_path(experiment)) or _initial_state(experiment)
            if result.action == "stop":
                break
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["stop_reason"] = "keyboard_interrupt"
        state["last_checkpoint"] = "interrupted"
        state["updated_at"] = _utc_now_iso()
        _save_state(experiment, state)
        return _run_result(experiment=experiment, state=state, rounds=rounds, interrupted=True)
    except Exception:
        state["status"] = "failed"
        state["last_checkpoint"] = "failed"
        state["updated_at"] = _utc_now_iso()
        _save_state(experiment, state)
        raise

    if state.get("status") == "running":
        state["status"] = "stopped"
        state["stop_reason"] = "max_rounds_reached"
        state["last_checkpoint"] = "stopped"
        state["updated_at"] = _utc_now_iso()
        _save_state(experiment, state)
    return _run_result(experiment=experiment, state=state, rounds=rounds, interrupted=False)


def program_markdown() -> str:
    """Return the active research prompt."""
    return PROGRAM_PATH.read_text(encoding="utf-8")


def _run_one_round(*, root: Path, experiment_id: str, state: dict[str, object]) -> ResearchRoundResult:
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    round_number = _as_int(state.get("next_round_number"), default=1)
    round_label = f"r{round_number:03d}"
    artifact_dir = _agentic_dir(experiment) / "rounds" / round_label
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report = _safe_report(root=root, experiment_id=experiment_id)
    if not _has_scored_primary_row(report):
        return _run_baseline_round(
            root=root,
            experiment=experiment,
            state=state,
            round_number=round_number,
            round_label=round_label,
            artifact_dir=artifact_dir,
        )

    context = _build_context(root=root, experiment=experiment, report=report, state=state)
    prompt = _render_prompt(context)
    _write_text(artifact_dir / "prompt.md", prompt)
    _write_json(artifact_dir / "context.json", context)

    raw_response, model_source = _call_research_llm(prompt=prompt, artifact_dir=artifact_dir)
    _write_text(artifact_dir / "llm_response.txt", raw_response)
    decision = _parse_decision(raw_response)
    _write_json(artifact_dir / "decision.json", _decision_payload(decision, model_source=model_source))

    if decision.action == "stop":
        return _record_stop_round(
            experiment=experiment,
            state=state,
            decision=decision,
            round_number=round_number,
            round_label=round_label,
            artifact_dir=artifact_dir,
        )

    config_path = _materialize_decision_config(
        experiment=experiment,
        round_label=round_label,
        decision=decision,
    )
    return _train_score_record_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action="run",
        config_path=config_path,
        artifact_dir=artifact_dir,
        learning=decision.learning,
        decision_payload=_decision_payload(decision, model_source=model_source),
    )


def _run_baseline_round(
    *,
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    artifact_dir: Path,
) -> ResearchRoundResult:
    parent_path = _first_config_path(experiment)
    config_dir = experiment.manifest_path.parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = _unique_config_path(config_dir, f"{round_label}_baseline_{_slug(parent_path.stem)}.json")
    payload = load_training_config_json(parent_path)
    _write_json(config_path, payload)
    learning = f"Baseline round from `{parent_path.name}` before asking the LLM for mutations."
    decision_payload = {
        "action": "baseline",
        "parent_config": parent_path.name,
        "changes": [],
        "learning": learning,
    }
    _write_json(artifact_dir / "decision.json", decision_payload)
    return _train_score_record_round(
        root=root,
        experiment=experiment,
        state=state,
        round_number=round_number,
        round_label=round_label,
        action="baseline",
        config_path=config_path,
        artifact_dir=artifact_dir,
        learning=learning,
        decision_payload=decision_payload,
    )


def _train_score_record_round(
    *,
    root: Path,
    experiment: ExperimentRecord,
    state: dict[str, object],
    round_number: int,
    round_label: str,
    action: ResearchAction,
    config_path: Path,
    artifact_dir: Path,
    learning: str,
    decision_payload: dict[str, object],
) -> ResearchRoundResult:
    with bind_launch_metadata(source="feature.agentic_research.train", operation_type="run", job_type="run"):
        trained = train_experiment(store_root=root, experiment_id=experiment.experiment_id, config_path=config_path)
    with bind_launch_metadata(source="feature.agentic_research.score_round", operation_type="run", job_type="run"):
        score_experiment_round(
            store_root=root,
            experiment_id=experiment.experiment_id,
            round=round_label,
            stage=SCORING_STAGE,
        )

    report = _safe_report(root=root, experiment_id=experiment.experiment_id)
    row = _row_for_run(report, trained.run_id)
    metric_value = getattr(row, PRIMARY_METRIC_FIELD) if row is not None else None
    best = _best_run_from_report(report)
    round_payload = {
        "round_number": round_number,
        "round_label": round_label,
        "action": action,
        "status": "completed",
        "config_path": str(config_path),
        "run_id": trained.run_id,
        "metric_value": metric_value,
        "learning": learning,
        "decision": decision_payload,
        "completed_at": _utc_now_iso(),
    }
    _write_json(artifact_dir / "round.json", round_payload)
    _write_text(artifact_dir / "learning.md", learning.strip() + "\n")
    _append_ledger(_ledger_path(experiment), round_payload)

    state.update(
        {
            "status": "running",
            "next_round_number": round_number + 1,
            "total_rounds_completed": _as_int(state.get("total_rounds_completed"), default=0) + 1,
            "last_checkpoint": "round_completed",
            "last_round_label": round_label,
            "last_run_id": trained.run_id,
            "best_overall": asdict(best),
            "updated_at": _utc_now_iso(),
        }
    )
    _save_state(experiment, state)
    return ResearchRoundResult(
        round_number=round_number,
        round_label=round_label,
        action=action,
        status="completed",
        config_path=config_path,
        run_id=trained.run_id,
        metric_value=metric_value,
        learning=learning,
        artifact_dir=artifact_dir,
    )


def _record_stop_round(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    decision: ResearchDecision,
    round_number: int,
    round_label: str,
    artifact_dir: Path,
) -> ResearchRoundResult:
    learning = decision.learning or decision.stop_reason or "LLM stopped the research loop."
    payload = {
        "round_number": round_number,
        "round_label": round_label,
        "action": "stop",
        "status": "stopped",
        "run_id": None,
        "metric_value": None,
        "learning": learning,
        "stop_reason": decision.stop_reason,
        "completed_at": _utc_now_iso(),
    }
    _write_json(artifact_dir / "round.json", payload)
    _write_text(artifact_dir / "learning.md", learning.strip() + "\n")
    _append_ledger(_ledger_path(experiment), payload)
    state.update(
        {
            "status": "stopped",
            "stop_reason": f"llm_stop:{decision.stop_reason or 'no_next_run'}",
            "last_checkpoint": "llm_stopped",
            "updated_at": _utc_now_iso(),
        }
    )
    _save_state(experiment, state)
    return ResearchRoundResult(
        round_number=round_number,
        round_label=round_label,
        action="stop",
        status="stopped",
        config_path=None,
        run_id=None,
        metric_value=None,
        learning=learning,
        artifact_dir=artifact_dir,
    )


def _build_context(
    *,
    root: Path,
    experiment: ExperimentRecord,
    report: ExperimentReport | None,
    state: dict[str, object],
) -> dict[str, object]:
    return {
        "objective": {
            "primary_metric": PRIMARY_METRIC_FIELD,
            "tie_break": "bmc_mean",
            "sanity_checks": ["corr_mean", "mmc_mean", "cwmm_mean"],
            "scoring_stage": SCORING_STAGE,
        },
        "allowed_change_paths": list(ALLOWED_CHANGE_PATHS),
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "hypothesis": experiment.hypothesis,
            "tags": list(experiment.tags),
            "champion_run_id": experiment.champion_run_id,
            "run_count": len(experiment.runs),
        },
        "state": state,
        "configs": _config_context(experiment),
        "report": _report_context(report),
        "recent_rounds": _recent_ledger(_ledger_path(experiment), limit=8),
        "experiment_notes": _read_text(experiment.manifest_path.parent / "EXPERIMENT.md", limit=MAX_CONTEXT_CHARS),
        "research_memory": _read_text(root / "notes" / "__RESEARCH_MEMORY__" / "CURRENT.md", limit=MAX_CONTEXT_CHARS),
    }


def _render_prompt(context: dict[str, object]) -> str:
    context_json = json.dumps(context, indent=2, sort_keys=True, default=str)
    return program_markdown().replace("{{CONTEXT_JSON}}", context_json)


def _call_research_llm(*, prompt: str, artifact_dir: Path) -> tuple[str, str]:
    source = active_model_source()
    if source == "openrouter":
        return _call_openrouter(prompt), "openrouter"
    return _call_codex_exec(prompt=prompt, artifact_dir=artifact_dir), "codex-exec"


def _call_openrouter(prompt: str) -> str:
    try:
        response = OpenRouterClient(timeout_seconds=180.0).chat_completions(
            payload={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }
        )
    except OpenRouterClientError as exc:
        raise AgenticResearchError(str(exc)) from exc
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgenticResearchError("agentic_research_openrouter_response_missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise AgenticResearchError("agentic_research_openrouter_content_missing")
    return content


def _call_codex_exec(*, prompt: str, artifact_dir: Path) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=artifact_dir, prefix=".codex_output_", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "-",
        "-o",
        str(output_path),
    ]
    completed = subprocess.run(cmd, input=prompt, text=True, capture_output=True, check=False)
    _write_text(artifact_dir / "codex_stdout.jsonl", completed.stdout)
    _write_text(artifact_dir / "codex_stderr.txt", completed.stderr)
    if completed.returncode != 0:
        raise AgenticResearchError(f"agentic_research_codex_failed:{completed.returncode}:{completed.stderr.strip()}")
    try:
        return output_path.read_text(encoding="utf-8")
    finally:
        try:
            output_path.unlink()
        except OSError:
            pass


def _parse_decision(raw_response: str) -> ResearchDecision:
    payload = _extract_json_object(raw_response)
    action = payload.get("action")
    if action not in {"run", "stop"}:
        raise AgenticResearchValidationError("agentic_research_action_invalid")
    changes = tuple(_parse_change(item) for item in _as_list(payload.get("changes")))
    decision = ResearchDecision(
        action=action,
        learning=_required_str(payload, "learning"),
        belief_update=_required_str(payload, "belief_update"),
        next_hypothesis=_optional_str(payload.get("next_hypothesis")),
        parent_config=_optional_str(payload.get("parent_config")),
        changes=changes,
        stop_reason=_optional_str(payload.get("stop_reason")),
    )
    if decision.action == "run":
        if decision.parent_config is None:
            raise AgenticResearchValidationError("agentic_research_parent_config_missing")
        if not 1 <= len(decision.changes) <= 3:
            raise AgenticResearchValidationError("agentic_research_change_count_invalid")
    if decision.action == "stop" and not decision.stop_reason:
        raise AgenticResearchValidationError("agentic_research_stop_reason_missing")
    return decision


def _parse_change(payload: object) -> ResearchChange:
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError("agentic_research_change_invalid")
    path = _required_str(payload, "path")
    if not _change_path_allowed(path):
        raise AgenticResearchValidationError(f"agentic_research_change_path_not_allowed:{path}")
    return ResearchChange(
        path=path,
        value=deepcopy(payload.get("value")),
        reason=_required_str(payload, "reason"),
    )


def _materialize_decision_config(*, experiment: ExperimentRecord, round_label: str, decision: ResearchDecision) -> Path:
    config_dir = experiment.manifest_path.parent / "configs"
    parent_path = config_dir / str(decision.parent_config)
    if not parent_path.is_file():
        raise AgenticResearchValidationError(f"agentic_research_parent_config_not_found:{decision.parent_config}")
    payload = load_training_config_json(parent_path)
    for change in decision.changes:
        _assign_dotted(payload, change.path.split("."), deepcopy(change.value))
    validated = TrainingConfig.model_validate(payload).model_dump(mode="python", exclude_none=True)
    candidate_hash = compute_config_hash(validated)
    existing_hashes = _existing_config_hashes(config_dir)
    if candidate_hash in existing_hashes:
        raise AgenticResearchValidationError(f"agentic_research_candidate_duplicate:{candidate_hash[:12]}")
    filename = _candidate_filename(round_label=round_label, parent_name=parent_path.stem, changes=decision.changes)
    path = _unique_config_path(config_dir, filename)
    _write_json(path, validated)
    return path


def _candidate_filename(*, round_label: str, parent_name: str, changes: tuple[ResearchChange, ...]) -> str:
    change_slug = "__".join(_slug(change.path.replace(".", "_")) for change in changes)
    stem = f"{round_label}_{_slug(parent_name)}__{change_slug}"[:170].rstrip("_-")
    return f"{stem or round_label}.json"


def _first_config_path(experiment: ExperimentRecord) -> Path:
    config_dir = experiment.manifest_path.parent / "configs"
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise AgenticResearchValidationError(f"agentic_research_config_missing:{experiment.experiment_id}")
    return configs[0]


def _existing_config_hashes(config_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in sorted(config_dir.glob("*.json")):
        try:
            hashes.add(compute_config_hash(load_training_config_json(path)))
        except Exception:
            continue
    return hashes


def _assign_dotted(payload: dict[str, object], parts: list[str], value: object) -> None:
    cursor: dict[str, object] = payload
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise AgenticResearchValidationError(f"agentic_research_change_target_not_mapping:{'.'.join(parts)}")
        cursor = child
    cursor[parts[-1]] = value


def _change_path_allowed(path: str) -> bool:
    for allowed in ALLOWED_CHANGE_PATHS:
        if allowed.endswith(".*") and path.startswith(allowed[:-1]):
            return True
        if path == allowed:
            return True
    return False


def _config_context(experiment: ExperimentRecord) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    config_dir = experiment.manifest_path.parent / "configs"
    for path in sorted(config_dir.glob("*.json")):
        try:
            payload = load_training_config_json(path)
        except Exception as exc:
            items.append({"filename": path.name, "error": str(exc)})
            continue
        items.append({"filename": path.name, "config": _mutable_config_view(payload)})
    return items


def _mutable_config_view(payload: dict[str, object]) -> dict[str, object]:
    view: dict[str, object] = {}
    for path in ALLOWED_CHANGE_PATHS:
        if path.endswith(".*"):
            prefix = path[:-2]
            value = _get_dotted(payload, prefix.split("."))
            if value is not None:
                _assign_view(view, prefix.split("."), value)
            continue
        value = _get_dotted(payload, path.split("."))
        if value is not None:
            _assign_view(view, path.split("."), value)
    return view


def _get_dotted(payload: dict[str, object], parts: list[str]) -> object | None:
    cursor: object = payload
    for part in parts:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _assign_view(payload: dict[str, object], parts: list[str], value: object) -> None:
    cursor = payload
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            return
        cursor = child
    cursor[parts[-1]] = value


def _report_context(report: ExperimentReport | None) -> dict[str, object]:
    if report is None:
        return {"rows": []}
    return {
        "metric": report.metric,
        "total_runs": report.total_runs,
        "champion_run_id": report.champion_run_id,
        "rows": [_row_payload(row) for row in report.rows],
    }


def _safe_report(*, root: Path, experiment_id: str) -> ExperimentReport | None:
    try:
        return report_experiment(store_root=root, experiment_id=experiment_id, metric=PRIMARY_METRIC, limit=25)
    except ExperimentError:
        return None


def _has_scored_primary_row(report: ExperimentReport | None) -> bool:
    return any(getattr(row, PRIMARY_METRIC_FIELD) is not None for row in (report.rows if report else ()))


def _row_for_run(report: ExperimentReport | None, run_id: str) -> ExperimentReportRow | None:
    for row in report.rows if report else ():
        if row.run_id == run_id:
            return row
    return None


def _best_run_from_report(report: ExperimentReport | None) -> ResearchBestRun:
    if report is None or not report.rows:
        return ResearchBestRun()
    for row in report.rows:
        if getattr(row, PRIMARY_METRIC_FIELD) is not None:
            return ResearchBestRun(
                experiment_id=report.experiment_id,
                run_id=row.run_id,
                bmc_last_200_eras_mean=row.bmc_last_200_eras_mean,
                bmc_mean=row.bmc_mean,
                corr_mean=row.corr_mean,
                mmc_mean=row.mmc_mean,
                cwmm_mean=row.cwmm_mean,
                updated_at=_utc_now_iso(),
            )
    return ResearchBestRun()


def _row_payload(row: ExperimentReportRow) -> dict[str, object]:
    return {
        "run_id": row.run_id,
        "status": row.status,
        "created_at": row.created_at,
        "metric_value": row.metric_value,
        "corr_mean": row.corr_mean,
        "mmc_mean": row.mmc_mean,
        "cwmm_mean": row.cwmm_mean,
        "bmc_mean": row.bmc_mean,
        "bmc_last_200_eras_mean": row.bmc_last_200_eras_mean,
        "is_champion": row.is_champion,
    }


def _decision_payload(decision: ResearchDecision, *, model_source: str) -> dict[str, object]:
    return {
        "action": decision.action,
        "learning": decision.learning,
        "belief_update": decision.belief_update,
        "next_hypothesis": decision.next_hypothesis,
        "parent_config": decision.parent_config,
        "changes": [asdict(change) for change in decision.changes],
        "stop_reason": decision.stop_reason,
        "model_source": model_source,
    }


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise AgenticResearchValidationError("agentic_research_json_missing")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AgenticResearchValidationError("agentic_research_json_invalid") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError("agentic_research_json_object_required")
    return payload


def _initial_state(experiment: ExperimentRecord) -> dict[str, object]:
    now = _utc_now_iso()
    return {
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "status": "initialized",
        "next_round_number": 1,
        "total_rounds_completed": 0,
        "last_checkpoint": "initialized",
        "stop_reason": None,
        "best_overall": asdict(ResearchBestRun()),
        "created_at": now,
        "updated_at": now,
    }


def _status_result(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    best: ResearchBestRun,
) -> ResearchStatusResult:
    auto_dir = _agentic_dir(experiment)
    return ResearchStatusResult(
        experiment_id=experiment.experiment_id,
        status=_status_value(state.get("status")),
        next_round_number=_as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=_as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        stop_reason=_optional_str(state.get("stop_reason")),
        best_overall=best,
        agentic_research_dir=auto_dir,
        state_path=auto_dir / STATE_FILENAME,
        ledger_path=auto_dir / LEDGER_FILENAME,
        program_path=PROGRAM_PATH,
    )


def _run_result(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    rounds: list[ResearchRoundResult],
    interrupted: bool,
) -> ResearchRunResult:
    best = _best_from_state(state)
    return ResearchRunResult(
        experiment_id=experiment.experiment_id,
        status=_status_value(state.get("status")),
        next_round_number=_as_int(state.get("next_round_number"), default=1),
        total_rounds_completed=_as_int(state.get("total_rounds_completed"), default=0),
        last_checkpoint=str(state.get("last_checkpoint") or "initialized"),
        stop_reason=_optional_str(state.get("stop_reason")),
        best_overall=best,
        rounds=tuple(rounds),
        interrupted=interrupted,
    )


def _best_from_state(state: dict[str, object]) -> ResearchBestRun:
    payload = state.get("best_overall")
    if not isinstance(payload, dict):
        return ResearchBestRun()
    return ResearchBestRun(
        experiment_id=_optional_str(payload.get("experiment_id")),
        run_id=_optional_str(payload.get("run_id")),
        bmc_last_200_eras_mean=_optional_float(payload.get("bmc_last_200_eras_mean")),
        bmc_mean=_optional_float(payload.get("bmc_mean")),
        corr_mean=_optional_float(payload.get("corr_mean")),
        mmc_mean=_optional_float(payload.get("mmc_mean")),
        cwmm_mean=_optional_float(payload.get("cwmm_mean")),
        updated_at=_optional_str(payload.get("updated_at")),
    )


def _status_value(value: object) -> ResearchStatus:
    return value if value in {"initialized", "running", "interrupted", "stopped", "failed"} else "initialized"


def _stopped_by_llm(state: dict[str, object]) -> bool:
    stop_reason = _optional_str(state.get("stop_reason")) or ""
    return _status_value(state.get("status")) == "stopped" and stop_reason.startswith("llm_stop:")


def _agentic_dir(experiment: ExperimentRecord) -> Path:
    return experiment.manifest_path.parent / AGENTIC_DIRNAME


def _state_path(experiment: ExperimentRecord) -> Path:
    return _agentic_dir(experiment) / STATE_FILENAME


def _ledger_path(experiment: ExperimentRecord) -> Path:
    return _agentic_dir(experiment) / LEDGER_FILENAME


def _load_state(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgenticResearchValidationError(f"agentic_research_state_invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError(f"agentic_research_state_invalid:{path}")
    return payload


def _save_state(experiment: ExperimentRecord, state: dict[str, object]) -> None:
    _write_json(_state_path(experiment), state)


def _recent_ledger(path: Path, *, limit: int) -> list[dict[str, object]]:
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


def _append_ledger(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_text(path: Path, *, limit: int) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _unique_config_path(config_dir: Path, filename: str) -> Path:
    path = config_dir / filename
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = config_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "config"


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgenticResearchValidationError(f"agentic_research_field_missing:{key}")
    return value.strip()


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "AgenticResearchError",
    "AgenticResearchValidationError",
    "ResearchBestRun",
    "ResearchRoundResult",
    "ResearchRunResult",
    "ResearchStatusResult",
    "get_research_status",
    "program_markdown",
    "run_research",
]
