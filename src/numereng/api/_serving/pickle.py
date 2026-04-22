"""Serving pickle build and upload API handlers."""

from __future__ import annotations

import numereng.api._serving as serving_api_module
from numereng.api._serving.responses import package_response
from numereng.api.contracts import (
    ServePickleBuildRequest,
    ServePickleBuildResponse,
    ServePickleUploadRequest,
    ServePickleUploadResponse,
)
from numereng.features.training.errors import TrainingError
from numereng.platform.errors import NumeraiClientError, PackageError


def serve_pickle_build(request: ServePickleBuildRequest) -> ServePickleBuildResponse:
    """Build one Numerai model-upload pickle from a submission package."""
    try:
        result = serving_api_module.build_submission_pickle(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            docker_image=request.docker_image,
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
    return ServePickleBuildResponse(
        package=package_response(result.package),
        pickle_path=str(result.pickle_path),
        docker_image=result.docker_image,
        smoke_verified=result.smoke_verified,
    )


def serve_pickle_upload(request: ServePickleUploadRequest) -> ServePickleUploadResponse:
    """Build and upload one Numerai model pickle."""
    try:
        result = serving_api_module.upload_submission_pickle(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            model_name=request.model_name,
            data_version=request.data_version,
            docker_image=request.docker_image,
        )
        diagnostics = None
        if request.wait_diagnostics:
            diagnostics = serving_api_module.sync_submission_package_diagnostics(
                workspace_root=request.workspace_root,
                experiment_id=request.experiment_id,
                package_id=request.package_id,
                wait=True,
            )
    except (
        serving_api_module.ServingPackageNotFoundError,
        serving_api_module.ServingValidationError,
        serving_api_module.ServingRuntimeError,
        serving_api_module.ServingUnsupportedConfigError,
        serving_api_module.SubmissionModelNotFoundError,
        serving_api_module.SubmissionModelUploadFileNotFoundError,
        serving_api_module.SubmissionModelUploadFormatUnsupportedError,
        TrainingError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServePickleUploadResponse(
        package=package_response(result.package if diagnostics is None else diagnostics.package),
        pickle_path=str(result.pickle_path),
        model_name=result.model_name,
        model_id=result.model_id,
        upload_id=result.upload_id,
        data_version=result.data_version,
        docker_image=result.docker_image,
        diagnostics_synced=diagnostics is not None,
        diagnostics_status=None if diagnostics is None else diagnostics.diagnostics_status,
        diagnostics_terminal=None if diagnostics is None else diagnostics.terminal,
        diagnostics_timed_out=None if diagnostics is None else diagnostics.timed_out,
        diagnostics_synced_at=None if diagnostics is None else diagnostics.synced_at,
        diagnostics_compute_status_path=None if diagnostics is None else str(diagnostics.compute_status_path),
        diagnostics_logs_path=None if diagnostics is None else str(diagnostics.logs_path),
        diagnostics_raw_path=None if diagnostics is None or diagnostics.raw_path is None else str(diagnostics.raw_path),
        diagnostics_summary_path=None
        if diagnostics is None or diagnostics.summary_path is None
        else str(diagnostics.summary_path),
        diagnostics_per_era_path=None
        if diagnostics is None or diagnostics.per_era_path is None
        else str(diagnostics.per_era_path),
    )
