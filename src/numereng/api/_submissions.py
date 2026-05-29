"""Submitted-model live snapshot and calibration API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    SubmissionCalibrationMaterializeRequest,
    SubmissionCalibrationMaterializeResponse,
    SubmissionCalibrationReportRequest,
    SubmissionCalibrationReportResponse,
    SubmissionCalibrationUpdateRequest,
    SubmissionCalibrationUpdateResponse,
    SubmissionRefreshItem,
    SubmissionRefreshRequest,
    SubmissionRefreshResponse,
)
from numereng.features.submission.calibration import (
    load_live_calibration_report,
    materialize_live_calibration,
)
from numereng.features.submission.live_refresh import refresh_submission_snapshots


def submissions_refresh(request: SubmissionRefreshRequest | None = None) -> SubmissionRefreshResponse:
    """Refresh local submitted-model live score snapshots from Numerai."""

    resolved_request = SubmissionRefreshRequest() if request is None else request
    results = refresh_submission_snapshots(
        workspace_root=resolved_request.workspace_root,
        model_names=resolved_request.model_names or None,
        dry_run=resolved_request.dry_run,
    )
    items = [
        SubmissionRefreshItem(
            model_name=result.model_name,
            model_id=result.model_id,
            live_rounds_path=str(result.live_rounds_path),
            submission_path=str(result.submission_path),
            round_count=result.round_count,
            scored_round_count=result.scored_round_count,
            resolved_round_count=result.resolved_round_count,
            resolved_scored_round_count=result.resolved_scored_round_count,
            latest_scored_round=result.latest_scored_round,
            latest_resolved_round=result.latest_resolved_round,
            skipped=result.skipped,
            warning=result.warning,
        )
        for result in results
    ]
    return SubmissionRefreshResponse(
        workspace_root=str(resolved_request.workspace_layout.workspace_root),
        dry_run=resolved_request.dry_run,
        refreshed_count=sum(1 for item in items if not item.skipped),
        skipped_count=sum(1 for item in items if item.skipped),
        items=items,
    )


def submissions_calibration_materialize(
    request: SubmissionCalibrationMaterializeRequest | None = None,
) -> SubmissionCalibrationMaterializeResponse:
    """Materialize canonical live calibration row and report artifacts."""

    resolved_request = SubmissionCalibrationMaterializeRequest() if request is None else request
    result = materialize_live_calibration(
        workspace_root=resolved_request.workspace_root,
        dry_run=resolved_request.dry_run,
    )
    return SubmissionCalibrationMaterializeResponse(
        workspace_root=str(result.workspace_root),
        artifact_root=str(result.artifact_root),
        rows_path=str(result.rows_path),
        observations_path=str(result.observations_path),
        report_path=str(result.report_path),
        manifest_path=str(result.manifest_path),
        row_count=result.row_count,
        observation_count=result.observation_count,
        model_count=result.model_count,
        scored_row_count=result.scored_row_count,
        scored_observation_count=result.scored_observation_count,
        dry_run=result.dry_run,
        warnings=list(result.warnings),
    )


def submissions_calibration_report(
    request: SubmissionCalibrationReportRequest | None = None,
) -> SubmissionCalibrationReportResponse:
    """Load the latest materialized live calibration report."""

    resolved_request = SubmissionCalibrationReportRequest() if request is None else request
    result = load_live_calibration_report(
        workspace_root=resolved_request.workspace_root,
        resolved_only=resolved_request.resolved_only,
    )
    return SubmissionCalibrationReportResponse(
        workspace_root=str(resolved_request.workspace_layout.workspace_root),
        artifact_root=str(result.artifact_root),
        rows_path=str(result.rows_path),
        observations_path=str(result.observations_path),
        report_path=str(result.report_path),
        manifest_path=str(result.manifest_path),
        row_count=result.row_count,
        observation_count=result.observation_count,
        scope="resolved_only" if resolved_request.resolved_only else "all",
        report=result.report,
        manifest=result.manifest,
    )


def submissions_calibration_update(
    request: SubmissionCalibrationUpdateRequest | None = None,
) -> SubmissionCalibrationUpdateResponse:
    """Refresh live snapshots, then rebuild and load calibration artifacts."""

    resolved_request = SubmissionCalibrationUpdateRequest() if request is None else request
    refresh = submissions_refresh(
        SubmissionRefreshRequest(
            workspace_root=resolved_request.workspace_root,
            model_names=resolved_request.model_names,
            dry_run=resolved_request.dry_run,
        )
    )
    materialize = submissions_calibration_materialize(
        SubmissionCalibrationMaterializeRequest(
            workspace_root=resolved_request.workspace_root,
            dry_run=resolved_request.dry_run,
        )
    )
    report = submissions_calibration_report(
        SubmissionCalibrationReportRequest(
            workspace_root=resolved_request.workspace_root,
            resolved_only=resolved_request.resolved_only,
        )
    )
    return SubmissionCalibrationUpdateResponse(refresh=refresh, materialize=materialize, report=report)


__all__ = [
    "submissions_calibration_materialize",
    "submissions_calibration_report",
    "submissions_calibration_update",
    "submissions_refresh",
]
