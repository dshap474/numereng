"""Serving API handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numereng.api.contracts import (
    ServeBlendRuleRequest,
    ServeComponentInspectionResponse,
    ServeComponentRequest,
    ServeLiveBuildRequest,
    ServeLiveBuildResponse,
    ServeLiveSubmitRequest,
    ServeLiveSubmitResponse,
    ServeNeutralizationRequest,
    ServePackageCreateRequest,
    ServePackageInspectRequest,
    ServePackageInspectResponse,
    ServePackageListRequest,
    ServePackageListResponse,
    ServePackageResponse,
    ServePackageScoreRequest,
    ServePackageScoreResponse,
    ServePackageSyncDiagnosticsRequest,
    ServePackageSyncDiagnosticsResponse,
    ServePickleBuildRequest,
    ServePickleBuildResponse,
    ServePickleUploadRequest,
    ServePickleUploadResponse,
)
from numereng.features.serving import (
    PackageDiagnosticsSyncResult,
    ServingBlendRule,
    ServingComponentSpec,
    ServingInspectionResult,
    ServingNeutralizationSpec,
    ServingPackageNotFoundError,
    ServingRuntimeError,
    ServingUnsupportedConfigError,
    ServingValidationError,
    SubmissionPackageRecord,
    build_live_submission_package,
    build_submission_pickle,
    create_submission_package,
    inspect_package,
    list_submission_packages,
    score_submission_package,
    submit_live_package,
    sync_submission_package_diagnostics,
    upload_submission_pickle,
)
from numereng.features.submission import (
    SubmissionModelNotFoundError,
    SubmissionModelUploadFileNotFoundError,
    SubmissionModelUploadFormatUnsupportedError,
)
from numereng.features.training.errors import TrainingError
from numereng.platform.errors import NumeraiClientError, PackageError


def serve_package_create(request: ServePackageCreateRequest) -> ServePackageResponse:
    """Create and persist one submission package."""
    try:
        record = create_submission_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            tournament=request.tournament,
            data_version=request.data_version,
            components=tuple(_to_feature_component(item) for item in request.components),
            blend_rule=_to_feature_blend_rule(request.blend_rule),
            neutralization=_to_feature_neutralization(request.neutralization),
            provenance=dict(request.provenance),
        )
    except (ServingValidationError, ServingRuntimeError, ServingUnsupportedConfigError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return _package_response(record)


def serve_package_list(request: ServePackageListRequest | None = None) -> ServePackageListResponse:
    """List persisted submission packages."""
    resolved_request = ServePackageListRequest() if request is None else request
    try:
        records = list_submission_packages(
            workspace_root=resolved_request.workspace_root,
            experiment_id=resolved_request.experiment_id,
        )
    except (ServingValidationError, ServingRuntimeError, ServingUnsupportedConfigError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return ServePackageListResponse(packages=[_package_response(item) for item in records])


def serve_package_inspect(request: ServePackageInspectRequest) -> ServePackageInspectResponse:
    """Inspect one submission package for local-live and model-upload compatibility."""
    try:
        result = inspect_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
        )
    except (
        ServingPackageNotFoundError,
        ServingValidationError,
        ServingRuntimeError,
        ServingUnsupportedConfigError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    return _inspection_response(result)


def serve_live_build(request: ServeLiveBuildRequest) -> ServeLiveBuildResponse:
    """Build one local live predictions parquet from a submission package."""
    try:
        result = build_live_submission_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
        )
    except (
        ServingPackageNotFoundError,
        ServingValidationError,
        ServingRuntimeError,
        ServingUnsupportedConfigError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServeLiveBuildResponse(
        package=_package_response(result.package),
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
        result = submit_live_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            model_name=request.model_name,
        )
    except (
        ServingPackageNotFoundError,
        ServingValidationError,
        ServingRuntimeError,
        ServingUnsupportedConfigError,
        SubmissionModelNotFoundError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServeLiveSubmitResponse(
        package=_package_response(result.live_build.package),
        current_round=result.live_build.current_round,
        submission_id=result.submission_id,
        model_name=result.model_name,
        model_id=result.model_id,
        submission_predictions_path=str(result.live_build.submission_predictions_path),
    )


def serve_package_score(request: ServePackageScoreRequest) -> ServePackageScoreResponse:
    """Score one final submission package on local validation data."""
    try:
        result = score_submission_package(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            dataset=request.dataset,
            runtime=request.runtime,
            stage=request.stage,
        )
    except (
        ServingPackageNotFoundError,
        ServingValidationError,
        ServingRuntimeError,
        ServingUnsupportedConfigError,
        TrainingError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServePackageScoreResponse(
        package=_package_response(result.package),
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
        result = sync_submission_package_diagnostics(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            wait=request.wait,
        )
    except (
        ServingPackageNotFoundError,
        ServingValidationError,
        ServingRuntimeError,
        ServingUnsupportedConfigError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return _package_diagnostics_response(result)


def serve_pickle_build(request: ServePickleBuildRequest) -> ServePickleBuildResponse:
    """Build one Numerai model-upload pickle from a submission package."""
    try:
        result = build_submission_pickle(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            docker_image=request.docker_image,
        )
    except (
        ServingPackageNotFoundError,
        ServingValidationError,
        ServingRuntimeError,
        ServingUnsupportedConfigError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServePickleBuildResponse(
        package=_package_response(result.package),
        pickle_path=str(result.pickle_path),
        docker_image=result.docker_image,
        smoke_verified=result.smoke_verified,
    )


def serve_pickle_upload(request: ServePickleUploadRequest) -> ServePickleUploadResponse:
    """Build and upload one Numerai model pickle."""
    try:
        result = upload_submission_pickle(
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            package_id=request.package_id,
            model_name=request.model_name,
            data_version=request.data_version,
            docker_image=request.docker_image,
        )
        diagnostics = None
        if request.wait_diagnostics:
            diagnostics = sync_submission_package_diagnostics(
                workspace_root=request.workspace_root,
                experiment_id=request.experiment_id,
                package_id=request.package_id,
                wait=True,
            )
    except (
        ServingPackageNotFoundError,
        ServingValidationError,
        ServingRuntimeError,
        ServingUnsupportedConfigError,
        SubmissionModelNotFoundError,
        SubmissionModelUploadFileNotFoundError,
        SubmissionModelUploadFormatUnsupportedError,
        TrainingError,
    ) as exc:
        raise PackageError(str(exc)) from exc
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return ServePickleUploadResponse(
        package=_package_response(result.package if diagnostics is None else diagnostics.package),
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


def _package_response(record: object) -> ServePackageResponse:
    package = record if isinstance(record, SubmissionPackageRecord) else None
    if package is None:  # pragma: no cover - defensive
        raise TypeError("expected SubmissionPackageRecord")
    return ServePackageResponse(
        package_id=package.package_id,
        experiment_id=package.experiment_id,
        tournament=package.tournament,
        data_version=package.data_version,
        package_path=str(package.package_path),
        status=package.status,
        components=[
            ServeComponentRequest(
                component_id=item.component_id,
                weight=item.weight,
                config_path=None if item.config_path is None else str(item.config_path),
                run_id=item.run_id,
                source_label=item.source_label,
            )
            for item in package.components
        ],
        blend_rule=ServeBlendRuleRequest(
            per_era_rank=package.blend_rule.per_era_rank,
            rank_method=package.blend_rule.rank_method,
            rank_pct=package.blend_rule.rank_pct,
            final_rerank=package.blend_rule.final_rerank,
        ),
        neutralization=None
        if package.neutralization is None
        else ServeNeutralizationRequest(
            enabled=package.neutralization.enabled,
            proportion=package.neutralization.proportion,
            mode=package.neutralization.mode,
            neutralizer_cols=None
            if package.neutralization.neutralizer_cols is None
            else list(package.neutralization.neutralizer_cols),
            rank_output=package.neutralization.rank_output,
        ),
        artifacts=dict(package.artifacts),
        created_at=package.created_at,
        updated_at=package.updated_at,
        provenance=dict(package.provenance),
    )


def _inspection_response(result: ServingInspectionResult) -> ServePackageInspectResponse:
    return ServePackageInspectResponse(
        package=_package_response(result.package),
        checked_at=result.checked_at,
        local_live_compatible=result.local_live_compatible,
        model_upload_compatible=result.model_upload_compatible,
        artifact_backed=result.artifact_backed,
        artifact_ready=result.artifact_ready,
        artifact_live_ready=result.artifact_live_ready,
        pickle_upload_ready=result.pickle_upload_ready,
        deployment_classification=result.deployment_classification,
        local_live_blockers=list(result.local_live_blockers),
        model_upload_blockers=list(result.model_upload_blockers),
        artifact_blockers=list(result.artifact_blockers),
        warnings=list(result.warnings),
        components=[
            ServeComponentInspectionResponse(
                component_id=item.component_id,
                local_live_compatible=item.local_live_compatible,
                model_upload_compatible=item.model_upload_compatible,
                artifact_backed=item.artifact_backed,
                artifact_ready=item.artifact_ready,
                local_live_blockers=list(item.local_live_blockers),
                model_upload_blockers=list(item.model_upload_blockers),
                artifact_blockers=list(item.artifact_blockers),
                warnings=list(item.warnings),
            )
            for item in result.components
        ],
        report_path=None if result.report_path is None else str(result.report_path),
    )


def _package_diagnostics_response(result: PackageDiagnosticsSyncResult) -> ServePackageSyncDiagnosticsResponse:
    diagnostics: Any = result
    return ServePackageSyncDiagnosticsResponse(
        package=_package_response(diagnostics.package),
        model_id=diagnostics.model_id,
        upload_id=diagnostics.upload_id,
        wait_requested=diagnostics.wait_requested,
        diagnostics_status=diagnostics.diagnostics_status,
        terminal=diagnostics.terminal,
        timed_out=diagnostics.timed_out,
        synced_at=diagnostics.synced_at,
        compute_status_path=str(diagnostics.compute_status_path),
        logs_path=str(diagnostics.logs_path),
        raw_path=None if diagnostics.raw_path is None else str(diagnostics.raw_path),
        summary_path=None if diagnostics.summary_path is None else str(diagnostics.summary_path),
        per_era_path=None if diagnostics.per_era_path is None else str(diagnostics.per_era_path),
    )


def _to_feature_component(item: ServeComponentRequest) -> ServingComponentSpec:
    return ServingComponentSpec(
        component_id=item.component_id or "",
        weight=item.weight,
        config_path=None if item.config_path is None else Path(item.config_path),
        run_id=item.run_id,
        source_label=item.source_label,
    )


def _to_feature_blend_rule(item: ServeBlendRuleRequest) -> ServingBlendRule:
    return ServingBlendRule(
        per_era_rank=item.per_era_rank,
        rank_method=item.rank_method,
        rank_pct=item.rank_pct,
        final_rerank=item.final_rerank,
    )


def _to_feature_neutralization(item: ServeNeutralizationRequest | None) -> ServingNeutralizationSpec | None:
    if item is None:
        return None
    return ServingNeutralizationSpec(
        enabled=item.enabled,
        proportion=item.proportion,
        mode=item.mode,
        neutralizer_cols=None if item.neutralizer_cols is None else tuple(item.neutralizer_cols),
        rank_output=item.rank_output,
    )


__all__ = [
    "serve_live_build",
    "serve_live_submit",
    "serve_package_create",
    "serve_package_score",
    "serve_package_sync_diagnostics",
    "serve_package_list",
    "serve_pickle_build",
    "serve_pickle_upload",
]
