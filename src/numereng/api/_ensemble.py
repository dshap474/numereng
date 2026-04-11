"""Ensemble API handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numereng.api.contracts import (
    EnsembleBuildRequest,
    EnsembleComponentResponse,
    EnsembleGetRequest,
    EnsembleListRequest,
    EnsembleListResponse,
    EnsembleMetricResponse,
    EnsembleResponse,
    EnsembleSelectionWinnerResponse,
    EnsembleSelectRequest,
    EnsembleSelectResponse,
)
from numereng.features.ensemble import (
    EnsembleBuildRequest as FeatureEnsembleBuildRequest,
)
from numereng.features.ensemble import (
    EnsembleError,
    EnsembleNotFoundError,
    EnsembleSelectionError,
    EnsembleValidationError,
)
from numereng.features.ensemble import (
    EnsembleSelectionRequest as FeatureEnsembleSelectionRequest,
)
from numereng.features.ensemble import (
    EnsembleSelectionSourceRule as FeatureEnsembleSelectionSourceRule,
)
from numereng.platform.errors import PackageError


def ensemble_build(request: EnsembleBuildRequest) -> EnsembleResponse:
    """Build one ensemble and persist its artifacts/registry rows."""
    from numereng import api as api_module

    try:
        result = api_module.build_ensemble_record(
            store_root=request.store_root,
            request=FeatureEnsembleBuildRequest(
                run_ids=tuple(request.run_ids),
                experiment_id=request.experiment_id,
                method=request.method,
                metric=request.metric,
                target=request.target,
                name=request.name,
                ensemble_id=request.ensemble_id,
                weights=tuple(request.weights) if request.weights is not None else None,
                optimize_weights=request.optimize_weights,
                include_heavy_artifacts=request.include_heavy_artifacts,
                selection_note=request.selection_note,
                regime_buckets=request.regime_buckets,
                neutralize_members=request.neutralize_members,
                neutralize_final=request.neutralize_final,
                neutralizer_path=Path(request.neutralizer_path) if request.neutralizer_path else None,
                neutralization_proportion=request.neutralization_proportion,
                neutralization_mode=request.neutralization_mode,
                neutralizer_cols=None if request.neutralizer_cols is None else tuple(request.neutralizer_cols),
                neutralization_rank_output=request.neutralization_rank_output,
            ),
        )
    except (EnsembleValidationError, EnsembleError, ValueError) as exc:
        raise PackageError(str(exc)) from exc

    return _to_response(result)


def ensemble_select(request: EnsembleSelectRequest) -> EnsembleSelectResponse:
    """Select one experiment-aware ensemble winner from historical source experiments."""
    from numereng import api as api_module

    try:
        result = api_module.select_ensemble_record(
            store_root=request.store_root,
            request=FeatureEnsembleSelectionRequest(
                experiment_id=request.experiment_id,
                source_experiment_ids=tuple(request.source_experiment_ids),
                source_rules=tuple(
                    FeatureEnsembleSelectionSourceRule(
                        experiment_id=rule.experiment_id,
                        selection_mode=rule.selection_mode,
                        explicit_targets=tuple(rule.explicit_targets),
                        top_n=rule.top_n,
                    )
                    for rule in request.source_rules
                ),
                selection_id=request.selection_id,
                target=request.target,
                primary_metric=request.primary_metric,
                tie_break_metric=request.tie_break_metric,
                correlation_threshold=request.correlation_threshold,
                top_weighted_variants=request.top_weighted_variants,
                weight_step=request.weight_step,
                bundle_policy=request.bundle_policy,
                required_seed_count=request.required_seed_count,
                require_full_seed_bundle=request.require_full_seed_bundle,
                blend_variants=tuple(request.blend_variants),
                weighted_promotion_min_gain=request.weighted_promotion_min_gain,
            ),
        )
    except (EnsembleSelectionError, EnsembleValidationError, EnsembleError, ValueError) as exc:
        raise PackageError(str(exc)) from exc

    return EnsembleSelectResponse(
        selection_id=result.selection_id,
        experiment_id=result.experiment_id,
        target=result.target,
        primary_metric=result.primary_metric,
        tie_break_metric=result.tie_break_metric,
        status=result.status,
        artifacts_path=str(result.artifacts_path),
        frozen_candidate_count=result.frozen_candidate_count,
        surviving_candidate_count=result.surviving_candidate_count,
        equal_weight_variant_count=result.equal_weight_variant_count,
        weighted_candidate_count=result.weighted_candidate_count,
        winner=EnsembleSelectionWinnerResponse(
            blend_id=result.winner_blend_id,
            selection_mode=result.winner_selection_mode,
            component_ids=list(result.winner_component_ids),
            weights=list(result.winner_weights),
            metrics={
                str(key): (None if value is None else float(value)) for key, value in result.winner_metrics.items()
            },
        ),
        created_at=result.created_at,
        updated_at=result.updated_at,
    )


def ensemble_list(request: EnsembleListRequest | None = None) -> EnsembleListResponse:
    """List persisted ensembles."""
    from numereng import api as api_module

    resolved_request = EnsembleListRequest() if request is None else request
    try:
        records = api_module.list_ensemble_records_api(
            store_root=resolved_request.store_root,
            experiment_id=resolved_request.experiment_id,
            limit=resolved_request.limit,
            offset=resolved_request.offset,
        )
    except (EnsembleError, ValueError) as exc:
        raise PackageError(str(exc)) from exc

    return EnsembleListResponse(ensembles=[_to_response(record) for record in records])


def ensemble_get(request: EnsembleGetRequest) -> EnsembleResponse:
    """Load one persisted ensemble by ID."""
    from numereng import api as api_module

    try:
        record = api_module.get_ensemble_record_api(
            store_root=request.store_root,
            ensemble_id=request.ensemble_id,
        )
    except (EnsembleNotFoundError, EnsembleValidationError, EnsembleError, ValueError) as exc:
        raise PackageError(str(exc)) from exc

    return _to_response(record)


def _to_response(record: Any) -> EnsembleResponse:
    return EnsembleResponse(
        ensemble_id=record.ensemble_id,
        experiment_id=record.experiment_id,
        name=record.name,
        method=record.method,
        target=record.target,
        metric=record.metric,
        status=record.status,
        components=[
            EnsembleComponentResponse(
                run_id=item.run_id,
                weight=item.weight,
                rank=item.rank,
            )
            for item in record.components
        ],
        metrics=[
            EnsembleMetricResponse(
                name=item.name,
                value=item.value,
            )
            for item in record.metrics
        ],
        artifacts_path=str(record.artifacts_path) if record.artifacts_path else None,
        config=record.config,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


__all__ = [
    "ensemble_build",
    "ensemble_select",
    "ensemble_get",
    "ensemble_list",
]
