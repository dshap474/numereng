"""HPO API handlers."""

from __future__ import annotations

from pathlib import Path

from numereng.api.contracts import (
    HpoStudyCreateRequest,
    HpoStudyGetRequest,
    HpoStudyListRequest,
    HpoStudyListResponse,
    HpoStudyResponse,
    HpoStudySpecResponse,
    HpoStudyTrialsRequest,
    HpoStudyTrialsResponse,
    HpoTrialResponse,
)
from numereng.config.hpo.contracts import canonicalize_hpo_study_payload
from numereng.features.hpo import (
    HpoDependencyError,
    HpoError,
    HpoNeutralizationSpec,
    HpoNotFoundError,
    HpoObjectiveSpec,
    HpoPlateauSpec,
    HpoSamplerSpec,
    HpoStoppingSpec,
    HpoValidationError,
)
from numereng.features.hpo import HpoStudyCreateRequest as FeatureHpoStudyCreateRequest
from numereng.platform.errors import PackageError


def hpo_create(request: HpoStudyCreateRequest) -> HpoStudyResponse:
    """Create and run one HPO study."""
    from numereng import api as api_module

    try:
        result = api_module.hpo_create_study(
            store_root=request.store_root,
            request=FeatureHpoStudyCreateRequest(
                study_id=request.study_id,
                study_name=request.study_name,
                config_path=Path(request.config_path),
                experiment_id=request.experiment_id,
                objective=HpoObjectiveSpec(
                    metric=request.objective.metric,
                    direction=request.objective.direction,
                    neutralization=HpoNeutralizationSpec(
                        enabled=request.objective.neutralization.enabled,
                        neutralizer_path=(
                            Path(request.objective.neutralization.neutralizer_path)
                            if request.objective.neutralization.neutralizer_path
                            else None
                        ),
                        proportion=request.objective.neutralization.proportion,
                        mode=request.objective.neutralization.mode,
                        neutralizer_cols=(
                            tuple(request.objective.neutralization.neutralizer_cols)
                            if request.objective.neutralization.neutralizer_cols is not None
                            else None
                        ),
                        rank_output=request.objective.neutralization.rank_output,
                    ),
                ),
                search_space={path: spec.model_dump(mode="python") for path, spec in request.search_space.items()},
                sampler=HpoSamplerSpec(
                    kind=request.sampler.kind,
                    seed=request.sampler.seed,
                    n_startup_trials=request.sampler.n_startup_trials,
                    multivariate=request.sampler.multivariate,
                    group=request.sampler.group,
                ),
                stopping=HpoStoppingSpec(
                    max_trials=request.stopping.max_trials,
                    max_completed_trials=request.stopping.max_completed_trials,
                    timeout_seconds=request.stopping.timeout_seconds,
                    plateau=HpoPlateauSpec(
                        enabled=request.stopping.plateau.enabled,
                        min_completed_trials=request.stopping.plateau.min_completed_trials,
                        patience_completed_trials=request.stopping.plateau.patience_completed_trials,
                        min_improvement_abs=request.stopping.plateau.min_improvement_abs,
                    ),
                ),
            ),
        )
    except (HpoValidationError, HpoDependencyError, HpoError) as exc:
        raise PackageError(str(exc)) from exc

    return _study_response_from_result(result)


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

    return HpoStudyListResponse(studies=[_study_response_from_result(record) for record in records])


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

    return _study_response_from_result(record)


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


def _study_response_from_result(result: object) -> HpoStudyResponse:
    spec = HpoStudySpecResponse.model_validate(canonicalize_hpo_study_payload(getattr(result, "spec")))
    return HpoStudyResponse(
        study_id=getattr(result, "study_id"),
        experiment_id=getattr(result, "experiment_id"),
        study_name=getattr(result, "study_name"),
        status=getattr(result, "status"),
        best_trial_number=getattr(result, "best_trial_number"),
        best_value=getattr(result, "best_value"),
        best_run_id=getattr(result, "best_run_id"),
        spec=spec,
        attempted_trials=getattr(result, "attempted_trials"),
        completed_trials=getattr(result, "completed_trials"),
        failed_trials=getattr(result, "failed_trials"),
        stop_reason=getattr(result, "stop_reason"),
        storage_path=str(getattr(result, "storage_path")) if getattr(result, "storage_path") else None,
        error_message=getattr(result, "error_message"),
        created_at=getattr(result, "created_at"),
        updated_at=getattr(result, "updated_at"),
    )


__all__ = [
    "hpo_create",
    "hpo_get",
    "hpo_list",
    "hpo_trials",
]
