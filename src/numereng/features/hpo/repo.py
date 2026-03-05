"""Persistence adapters between HPO feature contracts and store rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from numereng.features.hpo.contracts import (
    HpoDirection,
    HpoSampler,
    HpoStatus,
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


def save_study(*, store_root: str | Path, payload: HpoStudyResult, error_message: str | None = None) -> None:
    """Persist one HPO study snapshot."""

    upsert_hpo_study(
        store_root=store_root,
        study=StoreHpoStudyUpsert(
            study_id=payload.study_id,
            experiment_id=payload.experiment_id,
            study_name=payload.study_name,
            status=payload.status,
            metric=payload.metric,
            direction=payload.direction,
            n_trials=payload.n_trials,
            sampler=payload.sampler,
            seed=payload.seed,
            best_trial_number=payload.best_trial_number,
            best_value=payload.best_value,
            best_run_id=payload.best_run_id,
            config_json=json.dumps(payload.config, sort_keys=True, ensure_ascii=True),
            storage_path=str(payload.storage_path),
            error_message=error_message,
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
    config: dict[str, Any] = {}
    if row.config_json:
        try:
            parsed = json.loads(row.config_json)
            if isinstance(parsed, dict):
                config = parsed
        except json.JSONDecodeError:
            config = {}
    storage_path = Path(row.storage_path) if row.storage_path else None
    return HpoStudyRecord(
        study_id=row.study_id,
        experiment_id=row.experiment_id,
        study_name=row.study_name,
        status=_coerce_study_status(row.status),
        metric=row.metric,
        direction=_coerce_direction(row.direction),
        n_trials=row.n_trials,
        sampler=_coerce_sampler(row.sampler),
        seed=row.seed,
        best_trial_number=row.best_trial_number,
        best_value=row.best_value,
        best_run_id=row.best_run_id,
        config=config,
        storage_path=storage_path,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


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
    records: list[HpoStudyRecord] = []
    for row in rows:
        config: dict[str, Any] = {}
        if row.config_json:
            try:
                parsed = json.loads(row.config_json)
                if isinstance(parsed, dict):
                    config = parsed
            except json.JSONDecodeError:
                config = {}
        records.append(
            HpoStudyRecord(
                study_id=row.study_id,
                experiment_id=row.experiment_id,
                study_name=row.study_name,
                status=_coerce_study_status(row.status),
                metric=row.metric,
                direction=_coerce_direction(row.direction),
                n_trials=row.n_trials,
                sampler=_coerce_sampler(row.sampler),
                seed=row.seed,
                best_trial_number=row.best_trial_number,
                best_value=row.best_value,
                best_run_id=row.best_run_id,
                config=config,
                storage_path=Path(row.storage_path) if row.storage_path else None,
                error_message=row.error_message,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        )
    return tuple(records)


def list_trials(*, store_root: str | Path, study_id: str) -> tuple[HpoTrialRecord, ...]:
    """List trials for one study."""

    rows = list_hpo_trials(store_root=store_root, study_id=study_id)
    records: list[HpoTrialRecord] = []
    for row in rows:
        params: dict[str, Any] = {}
        if row.params_json:
            try:
                parsed = json.loads(row.params_json)
                if isinstance(parsed, dict):
                    params = parsed
            except json.JSONDecodeError:
                params = {}
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


def _coerce_study_status(value: str) -> HpoStatus:
    if value in {"running", "completed", "failed"}:
        return cast(HpoStatus, value)
    raise ValueError(f"hpo_study_status_invalid:{value}")


def _coerce_direction(value: str) -> HpoDirection:
    if value in {"maximize", "minimize"}:
        return cast(HpoDirection, value)
    raise ValueError(f"hpo_direction_invalid:{value}")


def _coerce_sampler(value: str) -> HpoSampler:
    if value in {"tpe", "random"}:
        return cast(HpoSampler, value)
    raise ValueError(f"hpo_sampler_invalid:{value}")


def _coerce_trial_status(value: str) -> HpoTrialStatus:
    if value in {"pending", "running", "completed", "failed"}:
        return cast(HpoTrialStatus, value)
    raise ValueError(f"hpo_trial_status_invalid:{value}")
