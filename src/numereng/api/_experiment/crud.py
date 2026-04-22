"""Experiment CRUD API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    ExperimentArchiveRequest,
    ExperimentArchiveResponse,
    ExperimentCreateRequest,
    ExperimentGetRequest,
    ExperimentListRequest,
    ExperimentListResponse,
    ExperimentResponse,
)
from numereng.features.experiments import (
    ExperimentAlreadyExistsError,
    ExperimentError,
    ExperimentNotFoundError,
    ExperimentValidationError,
)
from numereng.platform.errors import PackageError


def _experiment_response(record: object) -> ExperimentResponse:
    return ExperimentResponse(
        experiment_id=record.experiment_id,
        name=record.name,
        status=record.status,
        hypothesis=record.hypothesis,
        tags=list(record.tags),
        created_at=record.created_at,
        updated_at=record.updated_at,
        champion_run_id=record.champion_run_id,
        runs=list(record.runs),
        metadata=record.metadata,
        manifest_path=str(record.manifest_path),
    )


def experiment_create(request: ExperimentCreateRequest) -> ExperimentResponse:
    """Create one experiment and index metadata."""
    from numereng import api as api_module

    try:
        record = api_module.create_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            name=request.name,
            hypothesis=request.hypothesis,
            tags=request.tags,
        )
    except (ExperimentAlreadyExistsError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc
    return _experiment_response(record)


def experiment_archive(request: ExperimentArchiveRequest) -> ExperimentArchiveResponse:
    """Archive one experiment and move it under the archive root."""
    from numereng import api as api_module

    try:
        result = api_module.archive_experiment_record(
            store_root=request.store_root, experiment_id=request.experiment_id
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc
    return ExperimentArchiveResponse(
        experiment_id=result.experiment_id,
        status=result.status,
        manifest_path=str(result.manifest_path),
        archived=result.archived,
    )


def experiment_unarchive(request: ExperimentArchiveRequest) -> ExperimentArchiveResponse:
    """Restore one archived experiment to the live experiment root."""
    from numereng import api as api_module

    try:
        result = api_module.unarchive_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc
    return ExperimentArchiveResponse(
        experiment_id=result.experiment_id,
        status=result.status,
        manifest_path=str(result.manifest_path),
        archived=result.archived,
    )


def experiment_list(request: ExperimentListRequest | None = None) -> ExperimentListResponse:
    """List experiments from manifest storage."""
    from numereng import api as api_module

    resolved_request = ExperimentListRequest() if request is None else request
    try:
        records = api_module.list_experiment_records(
            store_root=resolved_request.store_root,
            status=resolved_request.status,
        )
    except ExperimentError as exc:
        raise PackageError(str(exc)) from exc
    return ExperimentListResponse(experiments=[_experiment_response(record) for record in records])


def experiment_get(request: ExperimentGetRequest) -> ExperimentResponse:
    """Load one experiment by ID."""
    from numereng import api as api_module

    try:
        record = api_module.get_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc
    return _experiment_response(record)
