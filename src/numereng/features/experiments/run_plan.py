"""Source-owned experiment run-plan execution with durable state."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from numereng.features.experiments.contracts import ExperimentRunPlanResult, ExperimentRunPlanWindow
from numereng.features.experiments.service import (
    ExperimentError,
    ExperimentNotFoundError,
    ExperimentValidationError,
    _as_str,
    _load_manifest,
    _manifest_to_record,
    _normalize_run_ids,
    _resolved_manifest_path,
    score_experiment_round,
    train_experiment,
)
from numereng.features.store import (
    index_run,
    resolve_store_root,
    resolve_workspace_layout_from_store_root,
    upsert_experiment,
)
from numereng.features.training.run_lock import is_lock_payload_active, read_run_lock, resolve_run_lock_path

_ROUND_RE = re.compile(r"^(r\d+)_")
_RUN_PLAN_SCHEMA_VERSION = "1"
_RUN_PLAN_RUNTIME_DIR = ("remote_ops", "experiment_run_plan")
_RETRYABLE_ERROR_PREFIXES = ("training_run_lock_exists:",)
_TERMINAL_PHASES = {"complete", "failed", "stopped"}


@dataclass(frozen=True)
class _PlanRow:
    index: int
    config_path: Path
    round_label: str


def run_experiment_plan(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    start_index: int = 1,
    end_index: int | None = None,
    score_stage: Literal["post_training_core", "post_training_full"] = "post_training_core",
    resume: bool = False,
) -> ExperimentRunPlanResult:
    """Execute one experiment run_plan window with durable state."""

    root = resolve_store_root(store_root)
    rows = _load_run_plan(root=root, experiment_id=experiment_id)
    total_rows = len(rows)
    resolved_end_index = _resolve_window(
        experiment_id=experiment_id,
        total_rows=total_rows,
        start_index=start_index,
        end_index=end_index,
    )

    state_path = resolve_experiment_run_plan_state_path(
        store_root=root,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=resolved_end_index,
    )
    if resume and state_path.is_file():
        state = _load_state(state_path)
        if state.get("phase") in _TERMINAL_PHASES:
            return _state_to_result(state_path=state_path, state=state)
    else:
        state = _new_state(
            experiment_id=experiment_id,
            start_index=start_index,
            end_index=resolved_end_index,
            total_rows=total_rows,
            score_stage=score_stage,
        )
        _write_state(state_path, state)

    if state["phase"] in _TERMINAL_PHASES:
        return _state_to_result(state_path=state_path, state=state)

    current_index = int(state.get("current_index") or state["window"]["start_index"])
    completed_rounds = set(_as_str_list(state.get("completed_score_stages")))
    retry_count = int(state.get("retry_count") or 0)
    last_index_by_round = _last_index_by_round(rows)

    for row in rows:
        if row.index < current_index or row.index > resolved_end_index:
            continue

        _persist_state(
            state_path,
            state,
            phase="training",
            current_index=row.index,
            current_round=row.round_label,
            current_config_path=str(row.config_path),
            current_run_id=None,
            supervisor_pid=os.getpid(),
        )
        try:
            result = train_experiment(
                store_root=root,
                experiment_id=experiment_id,
                config_path=row.config_path,
                post_training_scoring="none",
            )
        except Exception as exc:
            message = str(exc)
            if (
                _is_retryable_error(message)
                and retry_count < 1
                and _maybe_clear_stale_run_lock(root=root, message=message)
            ):
                retry_count += 1
                _persist_state(state_path, state, retry_count=retry_count)
                return run_experiment_plan(
                    store_root=root,
                    experiment_id=experiment_id,
                    start_index=start_index,
                    end_index=resolved_end_index,
                    score_stage=score_stage,
                    resume=True,
                )
            _persist_state(
                state_path,
                state,
                phase="failed",
                failure_classifier=_classify_failure(message),
                terminal_error=message,
                supervisor_pid=os.getpid(),
            )
            raise

        retry_count = 0
        _persist_state(
            state_path,
            state,
            retry_count=0,
            current_run_id=result.run_id,
            last_completed_row_index=row.index,
        )

        if row.index == last_index_by_round[row.round_label]:
            score_token = f"{row.round_label}:{score_stage}"
            if score_token not in completed_rounds:
                _persist_state(state_path, state, phase="round_scoring", current_round=row.round_label)
                _repair_manifest_links_for_round(root=root, experiment_id=experiment_id, round_label=row.round_label)
                score_experiment_round(
                    store_root=root,
                    experiment_id=experiment_id,
                    round=row.round_label,
                    stage=score_stage,
                )
                completed_rounds.add(score_token)
                _persist_state(state_path, state, completed_score_stages=sorted(completed_rounds))

        next_index = row.index + 1
        _persist_state(
            state_path,
            state,
            current_index=next_index if next_index <= resolved_end_index else None,
            current_round=None,
            current_config_path=None,
            current_run_id=None,
            active_worker_pid=None,
            last_successful_heartbeat_at=_utc_now_iso(),
        )

    _persist_state(
        state_path,
        state,
        phase="complete",
        current_index=None,
        current_round=None,
        current_config_path=None,
        current_run_id=None,
        failure_classifier=None,
        terminal_error=None,
        supervisor_pid=None,
    )
    return _state_to_result(state_path=state_path, state=state)


def get_experiment_run_plan_state(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    start_index: int = 1,
    end_index: int | None = None,
) -> ExperimentRunPlanResult | None:
    """Read one persisted run-plan execution state if it exists."""

    root = resolve_store_root(store_root)
    rows = _load_run_plan(root=root, experiment_id=experiment_id)
    total_rows = len(rows)
    resolved_end_index = total_rows if end_index is None else end_index
    state_path = resolve_experiment_run_plan_state_path(
        store_root=root,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=resolved_end_index,
    )
    if not state_path.is_file():
        return None
    return _state_to_result(state_path=state_path, state=_load_state(state_path))


def stop_experiment_run_plan(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    start_index: int = 1,
    end_index: int | None = None,
) -> ExperimentRunPlanResult:
    """Mark one persisted run-plan execution as stopped."""

    root = resolve_store_root(store_root)
    rows = _load_run_plan(root=root, experiment_id=experiment_id)
    total_rows = len(rows)
    resolved_end_index = total_rows if end_index is None else end_index
    state_path = resolve_experiment_run_plan_state_path(
        store_root=root,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=resolved_end_index,
    )
    if not state_path.is_file():
        raise ExperimentNotFoundError(
            f"experiment_run_plan_state_not_found:{experiment_id}:{start_index}:{resolved_end_index}"
        )
    state = _load_state(state_path)
    _persist_state(state_path, state, phase="stopped", failure_classifier=None, terminal_error=None)
    return _state_to_result(state_path=state_path, state=state)


def resolve_experiment_run_plan_state_path(
    *,
    store_root: str | Path,
    experiment_id: str,
    start_index: int,
    end_index: int,
) -> Path:
    """Return the canonical durable state path for one run-plan window."""

    root = resolve_store_root(store_root)
    runtime_root = root / _RUN_PLAN_RUNTIME_DIR[0] / _RUN_PLAN_RUNTIME_DIR[1]
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root / f"{experiment_id}__{start_index}_{end_index}.json"


def _resolve_window(
    *,
    experiment_id: str,
    total_rows: int,
    start_index: int,
    end_index: int | None,
) -> int:
    if total_rows == 0:
        raise ExperimentValidationError(f"experiment_run_plan_empty:{experiment_id}")
    if start_index < 1 or start_index > total_rows:
        raise ExperimentValidationError(f"experiment_run_plan_start_out_of_range:{start_index}:1..{total_rows}")
    resolved_end_index = total_rows if end_index is None else end_index
    if resolved_end_index < start_index or resolved_end_index > total_rows:
        raise ExperimentValidationError(
            f"experiment_run_plan_end_out_of_range:{resolved_end_index}:{start_index}..{total_rows}"
        )
    return resolved_end_index


def _load_run_plan(*, root: Path, experiment_id: str) -> list[_PlanRow]:
    experiment_dir = resolve_workspace_layout_from_store_root(root).experiments_root / experiment_id
    manifest_path = experiment_dir / "experiment.json"
    if not manifest_path.is_file():
        raise ExperimentNotFoundError(f"experiment_not_found:{experiment_id}")
    run_plan_path = experiment_dir / "run_plan.csv"
    if not run_plan_path.is_file():
        raise ExperimentValidationError(f"experiment_run_plan_missing:{run_plan_path}")
    rows: list[_PlanRow] = []
    with run_plan_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, raw in enumerate(reader, start=1):
            config_value = raw.get("config_path")
            if not isinstance(config_value, str) or not config_value.strip():
                raise ExperimentValidationError(f"experiment_run_plan_config_missing:{experiment_id}:{index}")
            config_path = _resolve_config_path(root=root, raw_config_path=config_value)
            if not config_path.is_file():
                raise ExperimentValidationError(f"experiment_run_plan_config_not_found:{config_path}")
            round_label = _resolve_round_label(raw.get("round"), config_path=config_path)
            rows.append(_PlanRow(index=index, config_path=config_path, round_label=round_label))
    return rows


def _resolve_config_path(*, root: Path, raw_config_path: str) -> Path:
    path = Path(raw_config_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    workspace_root = resolve_workspace_layout_from_store_root(root).workspace_root
    for base in (root, workspace_root):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return (workspace_root / path).resolve()


def _resolve_round_label(raw_round: str | None, *, config_path: Path) -> str:
    if raw_round is not None and raw_round.strip():
        candidate = raw_round.strip()
        if re.fullmatch(r"r\d+", candidate) is None:
            raise ExperimentValidationError(f"experiment_round_invalid:{candidate}:{config_path.name}")
        return candidate
    match = _ROUND_RE.match(config_path.stem)
    if match is None:
        raise ExperimentValidationError(f"experiment_round_missing:{config_path.name}")
    return match.group(1)


def _new_state(
    *,
    experiment_id: str,
    start_index: int,
    end_index: int,
    total_rows: int,
    score_stage: str,
) -> dict[str, object]:
    now = _utc_now_iso()
    return {
        "schema_version": _RUN_PLAN_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "window": {
            "start_index": start_index,
            "end_index": end_index,
            "total_rows": total_rows,
        },
        "phase": "training",
        "requested_score_stage": score_stage,
        "completed_score_stages": [],
        "current_index": start_index,
        "current_round": None,
        "current_config_path": None,
        "current_run_id": None,
        "last_completed_row_index": None,
        "supervisor_pid": os.getpid(),
        "active_worker_pid": None,
        "last_successful_heartbeat_at": None,
        "failure_classifier": None,
        "retry_count": 0,
        "terminal_error": None,
        "updated_at": now,
    }


def _load_state(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"experiment_run_plan_state_invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise ExperimentError(f"experiment_run_plan_state_invalid:{path}")
    return payload


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _persist_state(path: Path, state: dict[str, object], **updates: object) -> None:
    state.update(updates)
    state["updated_at"] = _utc_now_iso()
    _write_state(path, state)


def _state_to_result(*, state_path: Path, state: dict[str, object]) -> ExperimentRunPlanResult:
    window = state.get("window")
    window_payload = window if isinstance(window, dict) else {}
    config_path_raw = state.get("current_config_path")
    return ExperimentRunPlanResult(
        experiment_id=str(state.get("experiment_id") or state_path.stem),
        state_path=state_path,
        window=ExperimentRunPlanWindow(
            start_index=int(window_payload.get("start_index") or 1),
            end_index=int(window_payload.get("end_index") or 1),
            total_rows=int(window_payload.get("total_rows") or 0),
        ),
        phase=_coerce_phase(_as_str(state.get("phase")) or "failed"),
        requested_score_stage=_coerce_score_stage(_as_str(state.get("requested_score_stage")) or "post_training_core"),
        completed_score_stages=tuple(_as_str_list(state.get("completed_score_stages"))),
        current_index=_coerce_optional_int(state.get("current_index")),
        current_round=_as_str(state.get("current_round")),
        current_config_path=Path(config_path_raw).expanduser() if isinstance(config_path_raw, str) else None,
        current_run_id=_as_str(state.get("current_run_id")),
        last_completed_row_index=_coerce_optional_int(state.get("last_completed_row_index")),
        supervisor_pid=_coerce_optional_int(state.get("supervisor_pid")),
        active_worker_pid=_coerce_optional_int(state.get("active_worker_pid")),
        last_successful_heartbeat_at=_as_str(state.get("last_successful_heartbeat_at")),
        failure_classifier=_as_str(state.get("failure_classifier")),
        retry_count=int(state.get("retry_count") or 0),
        terminal_error=_as_str(state.get("terminal_error")),
        updated_at=_as_str(state.get("updated_at")) or _utc_now_iso(),
    )


def _coerce_phase(value: str) -> Literal["training", "round_scoring", "complete", "failed", "stopped"]:
    if value in {"training", "round_scoring", "complete", "failed", "stopped"}:
        return value
    return "failed"


def _coerce_score_stage(value: str) -> Literal["post_training_core", "post_training_full"]:
    if value == "post_training_full":
        return "post_training_full"
    return "post_training_core"


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _last_index_by_round(rows: list[_PlanRow]) -> dict[str, int]:
    payload: dict[str, int] = {}
    for row in rows:
        payload[row.round_label] = row.index
    return payload


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _is_retryable_error(message: str) -> bool:
    return any(message.startswith(prefix) for prefix in _RETRYABLE_ERROR_PREFIXES)


def _classify_failure(message: str) -> str:
    return "restartable" if _is_retryable_error(message) else "terminal"


def _maybe_clear_stale_run_lock(*, root: Path, message: str) -> bool:
    if not message.startswith("training_run_lock_exists:"):
        return False
    _, _, remainder = message.partition(":")
    run_id, _, _owner = remainder.partition(":")
    run_id = run_id.strip()
    if not run_id:
        return False
    run_dir = root / "runs" / run_id
    lock_path = resolve_run_lock_path(run_dir)
    payload = read_run_lock(lock_path)
    if not payload or is_lock_payload_active(payload):
        return False
    lock_path.unlink(missing_ok=True)
    return True


def _repair_manifest_links_for_round(*, root: Path, experiment_id: str, round_label: str) -> None:
    manifest_path = _resolved_manifest_path(root, experiment_id)
    manifest = _load_manifest(manifest_path)
    existing_run_ids = _normalize_run_ids(manifest.get("runs"))
    known_run_ids = set(existing_run_ids)
    experiment_dir = resolve_workspace_layout_from_store_root(root).experiments_root / experiment_id
    expected_stems = {
        row.stem
        for row in experiment_dir.glob("configs/*.json")
        if _ROUND_RE.match(row.stem) and row.stem.startswith(f"{round_label}_")
    }
    if not expected_stems:
        return

    candidates: list[tuple[str, str]] = []
    for run_dir in sorted((root / "runs").iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir():
            continue
        run_manifest = _read_json_dict(run_dir / "run.json")
        run_id = _as_str(run_manifest.get("run_id"))
        if run_id is None or run_id in known_run_ids:
            continue
        if _as_str(run_manifest.get("status")) != "FINISHED":
            continue
        config_payload = run_manifest.get("config")
        config_path = _as_str(config_payload.get("path")) if isinstance(config_payload, dict) else None
        if config_path is None or Path(config_path).stem not in expected_stems:
            continue
        if not _run_predictions_exist(root=root, run_id=run_id, run_manifest=run_manifest):
            continue
        created_at = _as_str(run_manifest.get("created_at")) or ""
        candidates.append((created_at, run_id))

    if not candidates:
        return
    candidates.sort()
    for _created_at, run_id in candidates:
        if run_id not in known_run_ids:
            existing_run_ids.append(run_id)
            known_run_ids.add(run_id)
            try:
                index_run(store_root=root, run_id=run_id)
            except Exception:
                pass

    manifest["runs"] = existing_run_ids
    manifest["updated_at"] = _utc_now_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    record = _manifest_to_record(manifest_path, manifest)
    upsert_experiment(
        store_root=root,
        experiment_id=record.experiment_id,
        name=record.name,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=record.metadata,
    )


def _run_predictions_exist(*, root: Path, run_id: str, run_manifest: dict[str, object]) -> bool:
    run_dir = root / "runs" / run_id
    artifacts = run_manifest.get("artifacts")
    predictions_rel = _as_str(artifacts.get("predictions")) if isinstance(artifacts, dict) else None
    if predictions_rel is not None:
        return (run_dir / predictions_rel).is_file()
    predictions_dir = run_dir / "artifacts" / "predictions"
    return any(predictions_dir.glob("*.parquet"))


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "ExperimentRunPlanResult",
    "get_experiment_run_plan_state",
    "resolve_experiment_run_plan_state_path",
    "run_experiment_plan",
    "stop_experiment_run_plan",
]
