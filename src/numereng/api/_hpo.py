"""HPO API handlers."""

from __future__ import annotations

from pathlib import Path

from numereng.api.contracts import (
    HpoStudyCreateRequest,
    HpoStudyGetRequest,
    HpoStudyListRequest,
    HpoStudyListResponse,
    HpoStudyResponse,
    HpoStudyTrialsRequest,
    HpoStudyTrialsResponse,
    HpoTrialResponse,
)
from numereng.features.hpo import (
    HpoDependencyError,
    HpoError,
    HpoNotFoundError,
    HpoValidationError,
)
from numereng.features.hpo import (
    HpoStudyCreateRequest as FeatureHpoStudyCreateRequest,
)
from numereng.platform.errors import PackageError


def hpo_create(request: HpoStudyCreateRequest) -> HpoStudyResponse:
    """Create and run one HPO study."""
    from numereng import api as api_module

    try:
        result = api_module.hpo_create_study(
            store_root=request.store_root,
            request=FeatureHpoStudyCreateRequest(
                study_name=request.study_name,
                config_path=Path(request.config_path),
                experiment_id=request.experiment_id,
                metric=request.metric,
                direction=request.direction,
                n_trials=request.n_trials,
                sampler=request.sampler,
                seed=request.seed,
                search_space=request.search_space,
                neutralize=request.neutralize,
                neutralizer_path=Path(request.neutralizer_path) if request.neutralizer_path else None,
                neutralization_proportion=request.neutralization_proportion,
                neutralization_mode=request.neutralization_mode,
                neutralizer_cols=None if request.neutralizer_cols is None else tuple(request.neutralizer_cols),
                neutralization_rank_output=request.neutralization_rank_output,
            ),
        )
    except (HpoValidationError, HpoDependencyError, HpoError) as exc:
        raise PackageError(str(exc)) from exc

    return HpoStudyResponse(
        study_id=result.study_id,
        experiment_id=result.experiment_id,
        study_name=result.study_name,
        status=result.status,
        metric=result.metric,
        direction=result.direction,
        n_trials=result.n_trials,
        sampler=result.sampler,
        seed=result.seed,
        best_trial_number=result.best_trial_number,
        best_value=result.best_value,
        best_run_id=result.best_run_id,
        config=result.config,
        storage_path=str(result.storage_path),
        error_message=None,
        created_at=result.created_at,
        updated_at=result.updated_at,
    )


def hpo_list(request: HpoStudyListRequest | None = None) -> HpoStudyListResponse:
    """List HPO studies."""
    from numereng import api as api_module

    resolved_request = HpoStudyListRequest() if request is None else request
    try:
        records = api_module.hpo_list_studies(
            store_root=resolved_request.store_root,
            experiment_id=resolved_request.experiment_id,
            status=resolved_request.status,
            limit=resolved_request.limit,
            offset=resolved_request.offset,
        )
    except HpoError as exc:
        raise PackageError(str(exc)) from exc

    return HpoStudyListResponse(
        studies=[
            HpoStudyResponse(
                study_id=record.study_id,
                experiment_id=record.experiment_id,
                study_name=record.study_name,
                status=record.status,
                metric=record.metric,
                direction=record.direction,
                n_trials=record.n_trials,
                sampler=record.sampler,
                seed=record.seed,
                best_trial_number=record.best_trial_number,
                best_value=record.best_value,
                best_run_id=record.best_run_id,
                config=record.config,
                storage_path=str(record.storage_path) if record.storage_path else None,
                error_message=record.error_message,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            for record in records
        ]
    )


def hpo_get(request: HpoStudyGetRequest) -> HpoStudyResponse:
    """Load one HPO study by ID."""
    from numereng import api as api_module

    try:
        record = api_module.hpo_get_study(
            store_root=request.store_root,
            study_id=request.study_id,
        )
    except (HpoNotFoundError, HpoValidationError, HpoError) as exc:
        raise PackageError(str(exc)) from exc

    return HpoStudyResponse(
        study_id=record.study_id,
        experiment_id=record.experiment_id,
        study_name=record.study_name,
        status=record.status,
        metric=record.metric,
        direction=record.direction,
        n_trials=record.n_trials,
        sampler=record.sampler,
        seed=record.seed,
        best_trial_number=record.best_trial_number,
        best_value=record.best_value,
        best_run_id=record.best_run_id,
        config=record.config,
        storage_path=str(record.storage_path) if record.storage_path else None,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def hpo_trials(request: HpoStudyTrialsRequest) -> HpoStudyTrialsResponse:
    """List trials for one HPO study."""
    from numereng import api as api_module

    try:
        rows = api_module.hpo_get_study_trials(
            store_root=request.store_root,
            study_id=request.study_id,
        )
    except (HpoNotFoundError, HpoValidationError, HpoError) as exc:
        raise PackageError(str(exc)) from exc

    return HpoStudyTrialsResponse(
        study_id=request.study_id,
        trials=[
            HpoTrialResponse(
                study_id=row.study_id,
                trial_number=row.trial_number,
                status=row.status,
                value=row.value,
                run_id=row.run_id,
                config_path=str(row.config_path) if row.config_path else None,
                params=row.params,
                error_message=row.error_message,
                started_at=row.started_at,
                finished_at=row.finished_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ],
    )


__all__ = [
    "hpo_create",
    "hpo_get",
    "hpo_list",
    "hpo_trials",
]
