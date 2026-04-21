"""Serving live build and submit API handlers."""

from __future__ import annotations

import numereng.api._serving as serving_api_module

from numereng.api._serving.responses import package_response
from numereng.api.contracts import (
    ServeLiveBuildRequest,
    ServeLiveBuildResponse,
    ServeLiveSubmitRequest,
    ServeLiveSubmitResponse,
)
from numereng.platform.errors import NumeraiClientError, PackageError


def serve_live_build(request: ServeLiveBuildRequest) -> ServeLiveBuildResponse:
    """Build one local live predictions parquet from a submission package."""
    try:
        result = serving_api_module.build_live_submission_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
        )
    except (
        serving_api_module.ServingPackageNotFoundError,
        serving_api_module.ServingValidationError,
        serving_api_module.ServingRuntimeError,
        serving_api_module.ServingUnsupportedConfigError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServeLiveBuildResponse(
        package=package_response(result.package),
        current_round=result.current_round,
        live_dataset_name=result.live_dataset_name,
        live_benchmark_dataset_name=result.live_benchmark_dataset_name,
        live_dataset_path=str(result.live_dataset_path),
        live_benchmark_dataset_path=None
        if result.live_benchmark_dataset_path is None
        else str(result.live_benchmark_dataset_path),
        component_prediction_paths=[str(path) for path in result.component_prediction_paths],
        blended_predictions_path=str(result.blended_predictions_path),
        submission_predictions_path=str(result.submission_predictions_path),
    )


def serve_live_submit(request: ServeLiveSubmitRequest) -> ServeLiveSubmitResponse:
    """Build and submit one live predictions parquet."""
    try:
        result = serving_api_module.submit_live_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            model_name=request.model_name,
        )
    except (
        serving_api_module.ServingPackageNotFoundError,
        serving_api_module.ServingValidationError,
        serving_api_module.ServingRuntimeError,
        serving_api_module.ServingUnsupportedConfigError,
        serving_api_module.SubmissionModelNotFoundError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServeLiveSubmitResponse(
        package=package_response(result.live_build.package),
        current_round=result.live_build.current_round,
        submission_id=result.submission_id,
        model_name=result.model_name,
        model_id=result.model_id,
        submission_predictions_path=str(result.live_build.submission_predictions_path),
    )
