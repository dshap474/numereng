"""Persistence adapters between HPO feature contracts and store rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from numereng.features.hpo.contracts import (
    HpoStatus,
    HpoStopReason,
    HpoStudyRecord,
    HpoStudyResult,
    HpoTrialRecord,
    HpoTrialResult,
    HpoTrialStatus,
)
from numereng.features.store import (
    StoreHpoStudyUpsert,
    StoreHpoTrialUpsert,
    get_hpo_study,
    list_hpo_studies,
    list_hpo_trials,
    upsert_hpo_study,
    upsert_hpo_trial,
)


def save_study(*, store_root: str | Path, payload: HpoStudyResult) -> None:
    """Persist one HPO study snapshot."""

    spec_json = json.dumps(payload.spec, sort_keys=True, ensure_ascii=True)
    upsert_hpo_study(
        store_root=store_root,
        study=StoreHpoStudyUpsert(
            study_id=payload.study_id,
            experiment_id=payload.experiment_id,
            study_name=payload.study_name,
            status=payload.status,
            metric=_study_metric(payload.spec),
            direction=_study_direction(payload.spec),
            n_trials=_study_max_trials(payload.spec),
            sampler=_study_sampler(payload.spec),
            seed=_study_seed(payload.spec),
            best_trial_number=payload.best_trial_number,
            best_value=payload.best_value,
            best_run_id=payload.best_run_id,
            config_json=spec_json,
            attempted_trials=payload.attempted_trials,
            completed_trials=payload.completed_trials,
            failed_trials=payload.failed_trials,
            stop_reason=payload.stop_reason,
            storage_path=str(payload.storage_path),
            error_message=payload.error_message,
        ),
    )


def save_trial(*, store_root: str | Path, payload: HpoTrialResult) -> None:
    """Persist one HPO trial snapshot."""

    upsert_hpo_trial(
        store_root=store_root,
        trial=StoreHpoTrialUpsert(
            study_id=payload.study_id,
            trial_number=payload.trial_number,
            status=payload.status,
            value=payload.value,
            run_id=payload.run_id,
            config_path=str(payload.config_path),
            params_json=json.dumps(payload.params, sort_keys=True, ensure_ascii=True),
            error_message=payload.error_message,
            started_at=payload.started_at,
            finished_at=payload.finished_at,
        ),
    )


def get_study(*, store_root: str | Path, study_id: str) -> HpoStudyRecord | None:
    """Load one HPO study by ID."""

    row = get_hpo_study(store_root=store_root, study_id=study_id)
    if row is None:
        return None
    return _study_record_from_row(row)


def list_studies(
    *,
    store_root: str | Path,
    experiment_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[HpoStudyRecord, ...]:
    """List HPO studies."""

    rows = list_hpo_studies(
        store_root=store_root,
        experiment_id=experiment_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return tuple(_study_record_from_row(row) for row in rows)


def list_trials(*, store_root: str | Path, study_id: str) -> tuple[HpoTrialRecord, ...]:
    """List trials for one study."""

    rows = list_hpo_trials(store_root=store_root, study_id=study_id)
    records: list[HpoTrialRecord] = []
    for row in rows:
        params = _parse_json_object(row.params_json)
        records.append(
            HpoTrialRecord(
                study_id=row.study_id,
                trial_number=row.trial_number,
                status=_coerce_trial_status(row.status),
                value=row.value,
                run_id=row.run_id,
                config_path=Path(row.config_path) if row.config_path else None,
                params=params,
                error_message=row.error_message,
                started_at=row.started_at,
                finished_at=row.finished_at,
                updated_at=row.updated_at,
            )
        )
    return tuple(records)


def _study_record_from_row(row: Any) -> HpoStudyRecord:
    spec = _parse_json_object(row.config_json)
    storage_path = Path(row.storage_path) if row.storage_path else None
    return HpoStudyRecord(
        study_id=row.study_id,
        experiment_id=row.experiment_id,
        study_name=row.study_name,
        status=_coerce_study_status(row.status),
        best_trial_number=row.best_trial_number,
        best_value=row.best_value,
        best_run_id=row.best_run_id,
        spec=spec,
        attempted_trials=row.attempted_trials,
        completed_trials=row.completed_trials,
        failed_trials=row.failed_trials,
        stop_reason=_coerce_stop_reason(row.stop_reason),
        storage_path=storage_path,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): item for key, item in parsed.items()}


def _study_metric(spec: dict[str, Any]) -> str:
    objective = spec.get("objective")
    if isinstance(objective, dict):
        metric = objective.get("metric")
        if isinstance(metric, str) and metric.strip():
            return metric.strip()
    return "post_fold_champion_objective"


def _study_direction(spec: dict[str, Any]) -> str:
    objective = spec.get("objective")
    if isinstance(objective, dict):
        direction = objective.get("direction")
        if direction in {"maximize", "minimize"}:
            return cast(str, direction)
    return "maximize"


def _study_max_trials(spec: dict[str, Any]) -> int:
    stopping = spec.get("stopping")
    if isinstance(stopping, dict):
        value = stopping.get("max_trials")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
    return 100


def _study_sampler(spec: dict[str, Any]) -> str:
    sampler = spec.get("sampler")
    if isinstance(sampler, dict):
        kind = sampler.get("kind")
        if kind in {"tpe", "random"}:
            return cast(str, kind)
    return "tpe"


def _study_seed(spec: dict[str, Any]) -> int | None:
    sampler = spec.get("sampler")
    if isinstance(sampler, dict):
        seed = sampler.get("seed")
        if seed is None:
            return None
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
    return 1337


def _coerce_study_status(value: str) -> HpoStatus:
    if value in {"running", "completed", "failed"}:
        return cast(HpoStatus, value)
    raise ValueError(f"hpo_study_status_invalid:{value}")


def _coerce_stop_reason(value: str | None) -> HpoStopReason | None:
    if value is None:
        return None
    if value in {
        "max_trials_reached",
        "max_completed_trials_reached",
        "timeout_reached",
        "plateau_reached",
        "all_trials_failed",
    }:
        return cast(HpoStopReason, value)
    raise ValueError(f"hpo_stop_reason_invalid:{value}")


def _coerce_trial_status(value: str) -> HpoTrialStatus:
    if value in {"pending", "running", "completed", "failed"}:
        return cast(HpoTrialStatus, value)
    raise ValueError(f"hpo_trial_status_invalid:{value}")
