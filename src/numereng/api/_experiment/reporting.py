"""Experiment report and packing API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    ExperimentPackRequest,
    ExperimentPackResponse,
    ExperimentReportRequest,
    ExperimentReportResponse,
    ExperimentReportRowResponse,
)
from numereng.features.experiments import ExperimentError, ExperimentNotFoundError, ExperimentValidationError
from numereng.platform.errors import PackageError


def experiment_report(request: ExperimentReportRequest) -> ExperimentReportResponse:
    """Build one ranked experiment report payload."""
    from numereng import api as api_module

    try:
        report = api_module.report_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            metric=request.metric,
            limit=request.limit,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc
    return ExperimentReportResponse(
        experiment_id=report.experiment_id,
        metric=report.metric,
        total_runs=report.total_runs,
        champion_run_id=report.champion_run_id,
        rows=[
            ExperimentReportRowResponse(
                run_id=row.run_id,
                status=row.status,
                created_at=row.created_at,
                metric_value=row.metric_value,
                corr_mean=row.corr_mean,
                mmc_mean=row.mmc_mean,
                cwmm_mean=row.cwmm_mean,
                bmc_mean=row.bmc_mean,
                bmc_last_200_eras_mean=row.bmc_last_200_eras_mean,
                is_champion=row.is_champion,
            )
            for row in report.rows
        ],
    )


def experiment_pack(request: ExperimentPackRequest) -> ExperimentPackResponse:
    """Write one experiment markdown pack artifact."""
    from numereng import api as api_module

    try:
        result = api_module.pack_experiment_record(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (ExperimentNotFoundError, ExperimentValidationError, ExperimentError) as exc:
        raise PackageError(str(exc)) from exc
    return ExperimentPackResponse(
        experiment_id=result.experiment_id,
        output_path=str(result.output_path),
        experiment_path=str(result.experiment_path),
        source_markdown_path=str(result.source_markdown_path),
        run_count=result.run_count,
        packed_at=result.packed_at,
    )
