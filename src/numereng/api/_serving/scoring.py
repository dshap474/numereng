"""Serving scoring and diagnostics sync API handlers."""

from __future__ import annotations

import numereng.api._serving as serving_api_module
from numereng.api._serving.responses import package_diagnostics_response, package_response
from numereng.api.contracts import (
    ServePackageScoreRequest,
    ServePackageScoreResponse,
    ServePackageSyncDiagnosticsRequest,
    ServePackageSyncDiagnosticsResponse,
)
from numereng.features.training.errors import TrainingError
from numereng.platform.errors import NumeraiClientError, PackageError


def serve_package_score(request: ServePackageScoreRequest) -> ServePackageScoreResponse:
    """Score one final submission package on local validation data."""
    try:
        result = serving_api_module.score_submission_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            dataset=request.dataset,
            runtime=request.runtime,
            stage=request.stage,
        )
    except (
        serving_api_module.ServingPackageNotFoundError,
        serving_api_module.ServingValidationError,
        serving_api_module.ServingRuntimeError,
        serving_api_module.ServingUnsupportedConfigError,
        TrainingError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServePackageScoreResponse(
        package=package_response(result.package),
        dataset=result.dataset,
        data_version=result.data_version,
        stage=result.stage,
        runtime_requested=result.runtime_requested,
        runtime_used=result.runtime_used,
        predictions_path=str(result.predictions_path),
        score_provenance_path=str(result.score_provenance_path),
        summaries_path=str(result.summaries_path),
        metric_series_path=str(result.metric_series_path),
        manifest_path=str(result.manifest_path),
        row_count=result.row_count,
        era_count=result.era_count,
    )


def serve_package_sync_diagnostics(
    request: ServePackageSyncDiagnosticsRequest,
) -> ServePackageSyncDiagnosticsResponse:
    """Sync the latest Numerai diagnostics snapshot for one uploaded submission package."""
    try:
        result = serving_api_module.sync_submission_package_diagnostics(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            wait=request.wait,
        )
    except (
        serving_api_module.ServingPackageNotFoundError,
        serving_api_module.ServingValidationError,
        serving_api_module.ServingRuntimeError,
        serving_api_module.ServingUnsupportedConfigError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return package_diagnostics_response(result)
