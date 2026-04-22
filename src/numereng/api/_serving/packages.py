"""Serving package CRUD and inspection API handlers."""

from __future__ import annotations

import numereng.api._serving as serving_api_module
from numereng.api._serving.responses import (
    inspection_response,
    package_response,
    to_feature_blend_rule,
    to_feature_component,
    to_feature_neutralization,
)
from numereng.api.contracts import (
    ServePackageCreateRequest,
    ServePackageInspectRequest,
    ServePackageInspectResponse,
    ServePackageListRequest,
    ServePackageListResponse,
    ServePackageResponse,
)
from numereng.platform.errors import PackageError


def serve_package_create(request: ServePackageCreateRequest) -> ServePackageResponse:
    """Create and persist one submission package."""
    try:
        record = serving_api_module.create_submission_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            tournament=request.tournament,
            data_version=request.data_version,
            components=tuple(to_feature_component(item) for item in request.components),
            blend_rule=to_feature_blend_rule(request.blend_rule),
            neutralization=to_feature_neutralization(request.neutralization),
            provenance=dict(request.provenance),
        )
    except (
        serving_api_module.ServingValidationError,
        serving_api_module.ServingRuntimeError,
        serving_api_module.ServingUnsupportedConfigError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    return package_response(record)


def serve_package_list(request: ServePackageListRequest | None = None) -> ServePackageListResponse:
    """List persisted submission packages."""
    resolved_request = ServePackageListRequest() if request is None else request
    try:
        records = serving_api_module.list_submission_packages(
            workspace_root=resolved_request.workspace_root,
            experiment_id=resolved_request.experiment_id,
        )
    except (
        serving_api_module.ServingValidationError,
        serving_api_module.ServingRuntimeError,
        serving_api_module.ServingUnsupportedConfigError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    return ServePackageListResponse(packages=[package_response(item) for item in records])


def serve_package_inspect(request: ServePackageInspectRequest) -> ServePackageInspectResponse:
    """Inspect one submission package for local-live and model-upload compatibility."""
    try:
        result = serving_api_module.inspect_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
        )
    except (
        serving_api_module.ServingPackageNotFoundError,
        serving_api_module.ServingValidationError,
        serving_api_module.ServingRuntimeError,
        serving_api_module.ServingUnsupportedConfigError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    return inspection_response(result)
