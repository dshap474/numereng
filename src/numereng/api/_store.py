"""Store API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    StoreDoctorRequest,
    StoreDoctorResponse,
    StoreIndexRequest,
    StoreIndexResponse,
    StoreInitRequest,
    StoreInitResponse,
    StoreMaterializeVizArtifactsFailureResponse,
    StoreMaterializeVizArtifactsRequest,
    StoreMaterializeVizArtifactsResponse,
    StoreRebuildFailureResponse,
    StoreRebuildRequest,
    StoreRebuildResponse,
    StoreRunLifecycleRepairRequest,
    StoreRunLifecycleRepairResponse,
)
from numereng.features.store import StoreError
from numereng.platform.errors import PackageError


def store_init(request: StoreInitRequest | None = None) -> StoreInitResponse:
    """Bootstrap store DB and required filesystem directories."""
    from numereng import api as api_module

    resolved_request = StoreInitRequest() if request is None else request
    try:
        result = api_module.init_store_db(store_root=resolved_request.store_root)
    except StoreError as exc:
        raise PackageError(str(exc)) from exc

    return StoreInitResponse(
        store_root=str(result.store_root),
        db_path=str(result.db_path),
        created=result.created,
        schema_migration=result.schema_migration,
    )


def store_index_run(request: StoreIndexRequest) -> StoreIndexResponse:
    """Index one run directory into store DB tables."""
    from numereng import api as api_module

    try:
        result = api_module.index_run(store_root=request.store_root, run_id=request.run_id)
    except StoreError as exc:
        raise PackageError(str(exc)) from exc

    return StoreIndexResponse(
        run_id=result.run_id,
        status=result.status,
        metrics_indexed=result.metrics_indexed,
        artifacts_indexed=result.artifacts_indexed,
        run_path=str(result.run_path),
        warnings=list(result.warnings),
    )


def store_rebuild(request: StoreRebuildRequest | None = None) -> StoreRebuildResponse:
    """Re-index all run directories into store DB tables."""
    from numereng import api as api_module

    resolved_request = StoreRebuildRequest() if request is None else request
    try:
        result = api_module.rebuild_run_index(store_root=resolved_request.store_root)
    except StoreError as exc:
        raise PackageError(str(exc)) from exc

    return StoreRebuildResponse(
        store_root=str(result.store_root),
        db_path=str(result.db_path),
        scanned_runs=result.scanned_runs,
        indexed_runs=result.indexed_runs,
        failed_runs=result.failed_runs,
        failures=[StoreRebuildFailureResponse(run_id=item.run_id, error=item.error) for item in result.failures],
    )


def store_doctor(request: StoreDoctorRequest | None = None) -> StoreDoctorResponse:
    """Run store consistency diagnostics against filesystem + DB state."""
    from numereng import api as api_module

    resolved_request = StoreDoctorRequest() if request is None else request
    try:
        result = api_module.doctor_store(
            store_root=resolved_request.store_root,
            fix_strays=resolved_request.fix_strays,
        )
    except StoreError as exc:
        raise PackageError(str(exc)) from exc

    return StoreDoctorResponse(
        store_root=str(result.store_root),
        db_path=str(result.db_path),
        ok=result.ok,
        issues=list(result.issues),
        stats=result.stats,
        stray_cleanup_applied=result.stray_cleanup_applied,
        deleted_paths=list(result.deleted_paths),
        missing_paths=list(result.missing_paths),
    )


def store_materialize_viz_artifacts(
    request: StoreMaterializeVizArtifactsRequest,
) -> StoreMaterializeVizArtifactsResponse:
    """Persist visualization artifacts for existing runs."""
    from numereng import api as api_module

    try:
        result = api_module.materialize_viz_artifacts(
            store_root=request.store_root,
            kind=request.kind,
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            all_runs=request.all,
        )
    except StoreError as exc:
        raise PackageError(str(exc)) from exc

    return StoreMaterializeVizArtifactsResponse(
        store_root=str(result.store_root),
        kind=result.kind,
        scoped_run_count=result.scoped_run_count,
        created_count=result.created_count,
        skipped_count=result.skipped_count,
        failed_count=result.failed_count,
        failures=[
            StoreMaterializeVizArtifactsFailureResponse(run_id=item.run_id, error=item.error)
            for item in result.failures
        ],
    )


def store_repair_run_lifecycles(
    request: StoreRunLifecycleRepairRequest | None = None,
) -> StoreRunLifecycleRepairResponse:
    """Sweep local lifecycle rows and reconcile stale/orphaned active runs."""
    from numereng import api as api_module

    resolved_request = StoreRunLifecycleRepairRequest() if request is None else request
    try:
        result = api_module.reconcile_run_lifecycles_record(
            store_root=resolved_request.store_root,
            run_id=resolved_request.run_id,
            active_only=resolved_request.active_only,
        )
    except StoreError as exc:
        raise PackageError(str(exc)) from exc

    return StoreRunLifecycleRepairResponse(
        store_root=str(result.store_root),
        scanned_count=result.scanned_count,
        unchanged_count=result.unchanged_count,
        reconciled_count=result.reconciled_count,
        reconciled_stale_count=result.reconciled_stale_count,
        reconciled_canceled_count=result.reconciled_canceled_count,
        run_ids=list(result.run_ids),
    )


__all__ = [
    "store_doctor",
    "store_index_run",
    "store_init",
    "store_materialize_viz_artifacts",
    "store_repair_run_lifecycles",
    "store_rebuild",
]
