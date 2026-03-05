"""Ensemble build orchestration service."""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.ensemble.builder import EnsembleBuildError, build_blended_predictions, load_ranked_components
from numereng.features.ensemble.contracts import (
    EnsembleBuildRequest,
    EnsembleComponent,
    EnsembleRecord,
    EnsembleResult,
)
from numereng.features.ensemble.metrics import (
    bootstrap_metric_summary,
    component_metrics_table,
    correlation_matrix,
    era_metrics_table,
    metric_dict,
    per_era_corr_series,
    regime_metrics_table,
    summarize_metrics,
)
from numereng.features.ensemble.repo import get_ensemble_record, list_ensemble_records, save_ensemble
from numereng.features.ensemble.weights import EnsembleWeightsError, normalize_weights, optimize_weights
from numereng.features.feature_neutralization import (
    NeutralizationDataError,
    NeutralizationExecutionError,
    NeutralizationValidationError,
    load_neutralizer_table,
    neutralize_prediction_frame,
)
from numereng.features.store import resolve_store_root

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_MAX_SELECTION_NOTE_LEN = 2000
_MIN_REGIME_BUCKETS = 2
_MAX_REGIME_BUCKETS = 50


class EnsembleError(Exception):
    """Base error for ensemble workflows."""


class EnsembleValidationError(EnsembleError):
    """Raised when ensemble inputs are invalid."""


class EnsembleNotFoundError(EnsembleError):
    """Raised when ensemble records are missing."""


class EnsembleExecutionError(EnsembleError):
    """Raised when ensemble construction fails."""


def build_ensemble(
    *,
    store_root: str | Path = ".numereng",
    request: EnsembleBuildRequest,
) -> EnsembleResult:
    """Build one rank-averaged ensemble from component run IDs."""

    if request.method != "rank_avg":
        raise EnsembleValidationError("ensemble_method_invalid")

    if request.regime_buckets < _MIN_REGIME_BUCKETS or request.regime_buckets > _MAX_REGIME_BUCKETS:
        raise EnsembleValidationError("ensemble_regime_buckets_invalid")
    if request.neutralization_proportion < 0.0 or request.neutralization_proportion > 1.0:
        raise EnsembleValidationError("ensemble_neutralization_proportion_invalid")
    if request.neutralization_mode not in {"era", "global"}:
        raise EnsembleValidationError("ensemble_neutralization_mode_invalid")
    if (request.neutralize_members or request.neutralize_final) and request.neutralizer_path is None:
        raise EnsembleValidationError("ensemble_neutralizer_path_required")

    selection_note = _normalize_selection_note(request.selection_note)
    if selection_note is not None and len(selection_note) > _MAX_SELECTION_NOTE_LEN:
        raise EnsembleValidationError("ensemble_selection_note_too_long")

    run_ids = tuple(_dedupe_run_ids(request.run_ids))
    if len(run_ids) < 2:
        raise EnsembleValidationError("ensemble_run_ids_insufficient")
    if request.experiment_id is not None and not _SAFE_ID.match(request.experiment_id):
        raise EnsembleValidationError(f"ensemble_experiment_id_invalid:{request.experiment_id}")

    ensemble_id = request.ensemble_id or _build_ensemble_id(experiment_id=request.experiment_id, run_ids=run_ids)
    if not _SAFE_ID.match(ensemble_id):
        raise EnsembleValidationError(f"ensemble_id_invalid:{ensemble_id}")

    ensemble_name = request.name.strip() if isinstance(request.name, str) and request.name.strip() else ensemble_id

    resolved_store = resolve_store_root(store_root)
    artifacts_path = _resolve_artifacts_path(
        store_root=resolved_store,
        experiment_id=request.experiment_id,
        ensemble_id=ensemble_id,
    )
    artifacts_preexisting = artifacts_path.exists()

    try:
        ranked_predictions, era_series, id_series, target_series = load_ranked_components(
            store_root=resolved_store,
            run_ids=run_ids,
            target_col=request.target,
        )
    except EnsembleBuildError as exc:
        raise EnsembleExecutionError(str(exc)) from exc

    neutralizer_table: pd.DataFrame | None = None
    resolved_neutralizer_cols: tuple[str, ...] | None = None
    if request.neutralize_members or request.neutralize_final:
        if request.neutralizer_path is None:  # pragma: no cover - guarded by validation
            raise EnsembleValidationError("ensemble_neutralizer_path_required")
        try:
            neutralizer_table, resolved_neutralizer_cols = load_neutralizer_table(
                neutralizer_path=request.neutralizer_path,
                neutralizer_cols=request.neutralizer_cols,
            )
        except (NeutralizationValidationError, NeutralizationDataError) as exc:
            raise EnsembleValidationError(str(exc)) from exc

    try:
        weights = normalize_weights(raw_weights=request.weights, n_components=len(run_ids))
    except EnsembleWeightsError as exc:
        raise EnsembleValidationError(str(exc)) from exc

    optimization_warning: str | None = None
    if request.optimize_weights:
        if target_series is None:
            optimization_warning = "ensemble_weight_optimization_skipped_missing_target"
        else:
            try:
                weights = optimize_weights(
                    ranked_predictions=ranked_predictions,
                    era_series=era_series,
                    target_series=target_series,
                    metric=request.metric,
                    initial_weights=weights,
                )
            except EnsembleWeightsError as exc:
                optimization_warning = str(exc)

    if request.neutralize_members:
        if neutralizer_table is None or resolved_neutralizer_cols is None:  # pragma: no cover - guarded by validation
            raise EnsembleValidationError("ensemble_neutralizer_path_required")
        for idx, _run_id in enumerate(run_ids):
            component_frame = pd.DataFrame(
                {
                    "era": era_series.values,
                    "id": id_series.values,
                    "prediction": ranked_predictions.iloc[:, idx].to_numpy(dtype=float),
                }
            )
            try:
                neutralized_component = neutralize_prediction_frame(
                    predictions=component_frame,
                    neutralizers=neutralizer_table,
                    neutralizer_cols=resolved_neutralizer_cols,
                    proportion=request.neutralization_proportion,
                    mode=request.neutralization_mode,
                    rank_output=request.neutralization_rank_output,
                )
            except (NeutralizationValidationError, NeutralizationDataError) as exc:
                raise EnsembleValidationError(str(exc)) from exc
            except NeutralizationExecutionError as exc:
                raise EnsembleExecutionError(str(exc)) from exc

            ranked_predictions.iloc[:, idx] = neutralized_component["prediction"].to_numpy(dtype=float)

    blended = build_blended_predictions(ranked_predictions=ranked_predictions, weights=weights)
    pre_final_blended = blended.copy()
    if request.neutralize_final:
        if neutralizer_table is None or resolved_neutralizer_cols is None:  # pragma: no cover - guarded by validation
            raise EnsembleValidationError("ensemble_neutralizer_path_required")
        final_frame = pd.DataFrame(
            {
                "era": era_series.values,
                "id": id_series.values,
                "prediction": blended,
            }
        )
        try:
            neutralized_final = neutralize_prediction_frame(
                predictions=final_frame,
                neutralizers=neutralizer_table,
                neutralizer_cols=resolved_neutralizer_cols,
                proportion=request.neutralization_proportion,
                mode=request.neutralization_mode,
                rank_output=request.neutralization_rank_output,
            )
        except (NeutralizationValidationError, NeutralizationDataError) as exc:
            raise EnsembleValidationError(str(exc)) from exc
        except NeutralizationExecutionError as exc:
            raise EnsembleExecutionError(str(exc)) from exc

        blended = neutralized_final["prediction"].to_numpy(dtype=float)

    per_era_corr = per_era_corr_series(
        blended=blended,
        era_series=era_series,
        target_series=target_series,
    )
    metrics = summarize_metrics(
        blended=blended,
        era_series=era_series,
        target_series=target_series,
    )
    corr_matrix = correlation_matrix(ranked_predictions=ranked_predictions, run_ids=run_ids)
    component_metrics = component_metrics_table(
        ranked_predictions=ranked_predictions,
        run_ids=run_ids,
        era_series=era_series,
        target_series=target_series,
        weights=weights,
    )
    era_metrics = era_metrics_table(per_era_corr=per_era_corr)
    regime_metrics = regime_metrics_table(
        per_era_corr=per_era_corr,
        regime_buckets=request.regime_buckets,
    )

    created_at = _utc_now_iso()
    try:
        artifact_manifest = _write_artifacts(
            artifacts_path=artifacts_path,
            ensemble_id=ensemble_id,
            experiment_id=request.experiment_id,
            ensemble_name=ensemble_name,
            method=request.method,
            target=request.target,
            metric=request.metric,
            run_ids=run_ids,
            weights=weights,
            optimize_weights=request.optimize_weights,
            selection_note=selection_note,
            regime_buckets=request.regime_buckets,
            include_heavy_artifacts=request.include_heavy_artifacts,
            warning=optimization_warning,
            metrics=metric_dict(metrics),
            era_series=era_series,
            id_series=id_series,
            blended=blended,
            pre_neutralized_blended=pre_final_blended if request.neutralize_final else None,
            ranked_predictions=ranked_predictions,
            corr_matrix=corr_matrix,
            component_metrics=component_metrics,
            era_metrics=era_metrics,
            regime_metrics=regime_metrics,
            per_era_corr=per_era_corr,
            created_at=created_at,
        )
    except Exception:
        if not artifacts_preexisting:
            shutil.rmtree(artifacts_path, ignore_errors=True)
        raise

    component_items = tuple(
        EnsembleComponent(
            run_id=run_id,
            weight=float(weights[idx]),
            rank=idx,
        )
        for idx, run_id in enumerate(run_ids)
    )

    result = EnsembleResult(
        ensemble_id=ensemble_id,
        experiment_id=request.experiment_id,
        name=ensemble_name,
        method=request.method,
        target=request.target,
        metric=request.metric,
        status="completed",
        components=component_items,
        metrics=metrics,
        artifacts_path=artifacts_path,
        config={
            "run_ids": list(run_ids),
            "weights": [float(value) for value in weights],
            "optimize_weights": request.optimize_weights,
            "optimization_warning": optimization_warning,
            "include_heavy_artifacts": request.include_heavy_artifacts,
            "selection_note": selection_note,
            "regime_buckets": request.regime_buckets,
            "neutralization": {
                "members_enabled": request.neutralize_members,
                "final_enabled": request.neutralize_final,
                "neutralizer_path": str(request.neutralizer_path) if request.neutralizer_path else None,
                "proportion": request.neutralization_proportion,
                "mode": request.neutralization_mode,
                "neutralizer_cols": list(resolved_neutralizer_cols) if resolved_neutralizer_cols else None,
                "rank_output": request.neutralization_rank_output,
            },
            "artifacts": {
                "always": list(artifact_manifest["always"]),
                "heavy": list(artifact_manifest["heavy"]),
            },
        },
        created_at=created_at,
        updated_at=created_at,
    )
    try:
        save_ensemble(store_root=resolved_store, payload=result)
    except Exception:
        if not artifacts_preexisting:
            shutil.rmtree(artifacts_path, ignore_errors=True)
        raise
    return result


def list_ensembles_view(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[EnsembleRecord, ...]:
    """List persisted ensembles."""

    return list_ensemble_records(
        store_root=store_root,
        experiment_id=experiment_id,
        limit=limit,
        offset=offset,
    )


def get_ensemble_view(*, store_root: str | Path = ".numereng", ensemble_id: str) -> EnsembleRecord:
    """Load one persisted ensemble."""

    payload = get_ensemble_record(store_root=store_root, ensemble_id=ensemble_id)
    if payload is None:
        raise EnsembleNotFoundError(f"ensemble_not_found:{ensemble_id}")
    return payload


def _write_artifacts(
    *,
    artifacts_path: Path,
    ensemble_id: str,
    experiment_id: str | None,
    ensemble_name: str,
    method: str,
    target: str,
    metric: str,
    run_ids: tuple[str, ...],
    weights: tuple[float, ...],
    optimize_weights: bool,
    selection_note: str | None,
    regime_buckets: int,
    include_heavy_artifacts: bool,
    warning: str | None,
    metrics: dict[str, float | None],
    era_series: pd.Series,
    id_series: pd.Series,
    blended: Any,
    pre_neutralized_blended: Any | None,
    ranked_predictions: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    component_metrics: pd.DataFrame,
    era_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    per_era_corr: pd.Series,
    created_at: str,
) -> dict[str, tuple[str, ...]]:
    artifacts_path.mkdir(parents=True, exist_ok=True)
    pre_neutralized_path = artifacts_path / "predictions_pre_neutralization.parquet"
    component_predictions_path = artifacts_path / "component_predictions.parquet"
    bootstrap_metrics_path = artifacts_path / "bootstrap_metrics.json"

    prediction_frame = pd.DataFrame(
        {
            "era": era_series.values,
            "id": id_series.values,
            "prediction": blended,
        }
    )
    prediction_frame.to_parquet(artifacts_path / "predictions.parquet", index=False)
    if pre_neutralized_blended is not None:
        pd.DataFrame(
            {
                "era": era_series.values,
                "id": id_series.values,
                "prediction": pre_neutralized_blended,
            }
        ).to_parquet(pre_neutralized_path, index=False)
    elif pre_neutralized_path.exists():
        pre_neutralized_path.unlink()
    corr_matrix.to_csv(artifacts_path / "correlation_matrix.csv", index=True)

    metrics_payload: dict[str, Any] = {
        "metrics": metrics,
    }
    if warning:
        metrics_payload["warning"] = warning
    (artifacts_path / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "run_id": list(run_ids),
            "weight": [float(value) for value in weights],
            "rank": list(range(len(run_ids))),
        }
    ).to_csv(artifacts_path / "weights.csv", index=False)
    component_metrics.to_csv(artifacts_path / "component_metrics.csv", index=False)
    era_metrics.to_csv(artifacts_path / "era_metrics.csv", index=False)
    regime_metrics.to_csv(artifacts_path / "regime_metrics.csv", index=False)

    heavy_files: list[str] = []
    if include_heavy_artifacts:
        component_predictions = ranked_predictions.copy()
        component_predictions.columns = list(run_ids)
        component_predictions.insert(0, "id", id_series.astype(str).tolist())
        component_predictions.insert(0, "era", era_series.astype(str).tolist())
        component_predictions["prediction"] = blended
        component_predictions.to_parquet(component_predictions_path, index=False)
        heavy_files.append("component_predictions.parquet")

        bootstrap_payload = bootstrap_metric_summary(per_era_corr=per_era_corr)
        bootstrap_metrics_path.write_text(
            json.dumps(bootstrap_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        heavy_files.append("bootstrap_metrics.json")
    else:
        if component_predictions_path.exists():
            component_predictions_path.unlink()
        if bootstrap_metrics_path.exists():
            bootstrap_metrics_path.unlink()

    always_files = [
        "predictions.parquet",
        "correlation_matrix.csv",
        "metrics.json",
        "weights.csv",
        "component_metrics.csv",
        "era_metrics.csv",
        "regime_metrics.csv",
        "lineage.json",
    ]
    if pre_neutralized_blended is not None:
        always_files.append("predictions_pre_neutralization.parquet")
    lineage_payload: dict[str, Any] = {
        "ensemble_id": ensemble_id,
        "experiment_id": experiment_id,
        "name": ensemble_name,
        "method": method,
        "target": target,
        "metric": metric,
        "created_at": created_at,
        "run_ids": list(run_ids),
        "weights": [float(value) for value in weights],
        "optimize_weights": optimize_weights,
        "selection_note": selection_note,
        "regime_buckets": regime_buckets,
        "include_heavy_artifacts": include_heavy_artifacts,
        "optimization_warning": warning,
        "final_neutralization_applied": pre_neutralized_blended is not None,
        "rows": int(len(era_series)),
        "artifacts": {
            "always": always_files,
            "heavy": heavy_files,
        },
    }
    (artifacts_path / "lineage.json").write_text(
        json.dumps(lineage_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "always": tuple(always_files),
        "heavy": tuple(heavy_files),
    }


def _resolve_artifacts_path(*, store_root: Path, experiment_id: str | None, ensemble_id: str) -> Path:
    if experiment_id:
        path = store_root / "experiments" / experiment_id / "ensembles" / ensemble_id
    else:
        path = store_root / "ensembles" / ensemble_id
    return path


def _build_ensemble_id(*, experiment_id: str | None, run_ids: tuple[str, ...]) -> str:
    scope = experiment_id if experiment_id else "global"
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    short = "-".join(run_ids[:3])
    base = f"{scope}-ens-{short}-{stamp}"
    normalized = "".join(ch for ch in base if ch.isalnum() or ch in {"-", "_", "."})
    normalized = normalized.strip("-._")
    if not normalized:
        raise EnsembleValidationError("ensemble_id_invalid")
    return normalized


def _dedupe_run_ids(run_ids: tuple[str, ...]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for run_id in run_ids:
        value = run_id.strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _normalize_selection_note(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return None
    return normalized


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
