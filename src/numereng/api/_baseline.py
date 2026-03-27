"""Baseline API handlers."""

from __future__ import annotations

from numereng.api.contracts import BaselineBuildRequest, BaselineBuildResponse
from numereng.features.baseline import (
    BaselineBuildRequest as FeatureBaselineBuildRequest,
)
from numereng.features.baseline import BaselineError, BaselineValidationError
from numereng.platform.errors import PackageError


def baseline_build(request: BaselineBuildRequest) -> BaselineBuildResponse:
    """Build one named baseline and optionally promote it to active benchmark."""
    from numereng import api as api_module

    try:
        result = api_module.build_baseline_record(
            store_root=request.store_root,
            request=FeatureBaselineBuildRequest(
                run_ids=tuple(request.run_ids),
                name=request.name,
                default_target=request.default_target,
                description=request.description,
                promote_active=request.promote_active,
            ),
        )
    except (BaselineValidationError, BaselineError, ValueError) as exc:
        raise PackageError(str(exc)) from exc

    return BaselineBuildResponse(
        name=result.name,
        baseline_dir=str(result.baseline_dir),
        predictions_path=str(result.predictions_path),
        metadata_path=str(result.metadata_path),
        available_targets=list(result.available_targets),
        default_target=result.default_target,
        source_run_ids=list(result.source_run_ids),
        source_experiment_id=result.source_experiment_id,
        active_predictions_path=(str(result.active_predictions_path) if result.active_predictions_path else None),
        active_metadata_path=(str(result.active_metadata_path) if result.active_metadata_path else None),
        created_at=result.created_at,
    )


__all__ = ["baseline_build"]
