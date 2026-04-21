"""Shared response mappers for serving API handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numereng.api.contracts import (
    ServeBlendRuleRequest,
    ServeComponentInspectionResponse,
    ServeComponentRequest,
    ServeNeutralizationRequest,
    ServePackageInspectResponse,
    ServePackageResponse,
    ServePackageSyncDiagnosticsResponse,
)
from numereng.features.serving import (
    PackageDiagnosticsSyncResult,
    ServingBlendRule,
    ServingComponentSpec,
    ServingInspectionResult,
    ServingNeutralizationSpec,
    SubmissionPackageRecord,
)


def package_response(record: object) -> ServePackageResponse:
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


def inspection_response(result: ServingInspectionResult) -> ServePackageInspectResponse:
    return ServePackageInspectResponse(
        package=package_response(result.package),
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


def package_diagnostics_response(result: PackageDiagnosticsSyncResult) -> ServePackageSyncDiagnosticsResponse:
    diagnostics: Any = result
    return ServePackageSyncDiagnosticsResponse(
        package=package_response(diagnostics.package),
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


def to_feature_component(item: ServeComponentRequest) -> ServingComponentSpec:
    return ServingComponentSpec(
        component_id=item.component_id or "",
        weight=item.weight,
        config_path=None if item.config_path is None else Path(item.config_path),
        run_id=item.run_id,
        source_label=item.source_label,
    )


def to_feature_blend_rule(item: ServeBlendRuleRequest) -> ServingBlendRule:
    return ServingBlendRule(
        per_era_rank=item.per_era_rank,
        rank_method=item.rank_method,
        rank_pct=item.rank_pct,
        final_rerank=item.final_rerank,
    )


def to_feature_neutralization(item: ServeNeutralizationRequest | None) -> ServingNeutralizationSpec | None:
    if item is None:
        return None
    return ServingNeutralizationSpec(
        enabled=item.enabled,
        proportion=item.proportion,
        mode=item.mode,
        neutralizer_cols=None if item.neutralizer_cols is None else tuple(item.neutralizer_cols),
        rank_output=item.rank_output,
    )
