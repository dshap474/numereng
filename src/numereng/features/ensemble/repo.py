"""Persistence adapters between ensemble contracts and store rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from numereng.features.ensemble.contracts import (
    EnsembleComponent,
    EnsembleMethod,
    EnsembleMetric,
    EnsembleRecord,
    EnsembleResult,
    EnsembleStatus,
)
from numereng.features.store import (
    StoreEnsembleComponentUpsert,
    StoreEnsembleMetricUpsert,
    StoreEnsembleUpsert,
    get_ensemble,
    list_ensemble_components,
    list_ensemble_metrics,
    list_ensembles,
    replace_ensemble_components,
    replace_ensemble_metrics,
    upsert_ensemble,
)


def save_ensemble(*, store_root: str | Path, payload: EnsembleResult) -> None:
    """Persist one ensemble snapshot and its component/metric rows."""

    upsert_ensemble(
        store_root=store_root,
        ensemble=StoreEnsembleUpsert(
            ensemble_id=payload.ensemble_id,
            experiment_id=payload.experiment_id,
            name=payload.name,
            method=payload.method,
            target=payload.target,
            metric=payload.metric,
            status=payload.status,
            config_json=json.dumps(payload.config, sort_keys=True, ensure_ascii=True),
            artifacts_path=str(payload.artifacts_path),
        ),
    )
    replace_ensemble_components(
        store_root=store_root,
        ensemble_id=payload.ensemble_id,
        components=tuple(
            StoreEnsembleComponentUpsert(
                ensemble_id=payload.ensemble_id,
                run_id=component.run_id,
                weight=component.weight,
                rank=component.rank,
            )
            for component in payload.components
        ),
    )
    replace_ensemble_metrics(
        store_root=store_root,
        ensemble_id=payload.ensemble_id,
        metrics=tuple(
            StoreEnsembleMetricUpsert(
                ensemble_id=payload.ensemble_id,
                name=metric.name,
                value=metric.value,
            )
            for metric in payload.metrics
        ),
    )


def get_ensemble_record(*, store_root: str | Path, ensemble_id: str) -> EnsembleRecord | None:
    """Load one persisted ensemble and linked component/metric rows."""

    row = get_ensemble(store_root=store_root, ensemble_id=ensemble_id)
    if row is None:
        return None
    return _assemble_record(store_root=store_root, row_ensemble_id=row.ensemble_id, row_payload=row)


def list_ensemble_records(
    *,
    store_root: str | Path,
    experiment_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[EnsembleRecord, ...]:
    """List persisted ensembles with linked component/metric rows."""

    rows = list_ensembles(
        store_root=store_root,
        experiment_id=experiment_id,
        limit=limit,
        offset=offset,
    )
    return tuple(
        _assemble_record(
            store_root=store_root,
            row_ensemble_id=row.ensemble_id,
            row_payload=row,
        )
        for row in rows
    )


def _assemble_record(*, store_root: str | Path, row_ensemble_id: str, row_payload: Any) -> EnsembleRecord:
    component_rows = list_ensemble_components(store_root=store_root, ensemble_id=row_ensemble_id)
    metric_rows = list_ensemble_metrics(store_root=store_root, ensemble_id=row_ensemble_id)

    config: dict[str, Any] = {}
    if row_payload.config_json:
        try:
            parsed = json.loads(row_payload.config_json)
            if isinstance(parsed, dict):
                config = parsed
        except json.JSONDecodeError:
            config = {}

    return EnsembleRecord(
        ensemble_id=row_payload.ensemble_id,
        experiment_id=row_payload.experiment_id,
        name=row_payload.name,
        method=_coerce_method(str(row_payload.method)),
        target=row_payload.target,
        metric=row_payload.metric,
        status=_coerce_status(str(row_payload.status)),
        components=tuple(
            EnsembleComponent(
                run_id=component.run_id,
                weight=component.weight,
                rank=component.rank,
            )
            for component in component_rows
        ),
        metrics=tuple(
            EnsembleMetric(
                name=metric.name,
                value=metric.value,
            )
            for metric in metric_rows
        ),
        artifacts_path=Path(row_payload.artifacts_path) if row_payload.artifacts_path else None,
        config=config,
        created_at=row_payload.created_at,
        updated_at=row_payload.updated_at,
    )


def _coerce_method(value: str) -> EnsembleMethod:
    if value in {"rank_avg"}:
        return cast(EnsembleMethod, value)
    raise ValueError(f"ensemble_method_invalid:{value}")


def _coerce_status(value: str) -> EnsembleStatus:
    if value in {"running", "completed", "failed"}:
        return cast(EnsembleStatus, value)
    raise ValueError(f"ensemble_status_invalid:{value}")
