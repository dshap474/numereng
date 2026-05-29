from __future__ import annotations

from pathlib import Path

import pytest

from numereng.api import _submissions as submissions_api
from numereng.api.contracts import (
    SubmissionCalibrationMaterializeRequest,
    SubmissionCalibrationMaterializeResponse,
    SubmissionCalibrationReportResponse,
    SubmissionCalibrationUpdateRequest,
    SubmissionRefreshResponse,
)


def test_submissions_calibration_update_passes_dry_run_to_materialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, bool] = {}

    def fake_materialize(request: SubmissionCalibrationMaterializeRequest) -> SubmissionCalibrationMaterializeResponse:
        seen["dry_run"] = request.dry_run
        return SubmissionCalibrationMaterializeResponse(
            workspace_root=str(tmp_path),
            artifact_root=str(tmp_path / ".numereng" / "analysis" / "live_calibration"),
            rows_path=str(tmp_path / "rows.parquet"),
            observations_path=str(tmp_path / "observations.parquet"),
            report_path=str(tmp_path / "report.json"),
            manifest_path=str(tmp_path / "manifest.json"),
            row_count=0,
            observation_count=0,
            model_count=0,
            scored_row_count=0,
            scored_observation_count=0,
            dry_run=request.dry_run,
        )

    monkeypatch.setattr(
        submissions_api,
        "submissions_refresh",
        lambda request: SubmissionRefreshResponse(
            workspace_root=str(tmp_path),
            dry_run=request.dry_run,
            refreshed_count=0,
            skipped_count=0,
            items=[],
        ),
    )
    monkeypatch.setattr(submissions_api, "submissions_calibration_materialize", fake_materialize)
    monkeypatch.setattr(
        submissions_api,
        "submissions_calibration_report",
        lambda request: SubmissionCalibrationReportResponse(
            workspace_root=str(tmp_path),
            artifact_root=str(tmp_path / ".numereng" / "analysis" / "live_calibration"),
            rows_path=str(tmp_path / "rows.parquet"),
            observations_path=str(tmp_path / "observations.parquet"),
            report_path=str(tmp_path / "report.json"),
            manifest_path=str(tmp_path / "manifest.json"),
            row_count=0,
            observation_count=0,
            scope="all",
            report={},
            manifest={},
        ),
    )

    response = submissions_api.submissions_calibration_update(
        SubmissionCalibrationUpdateRequest(workspace_root=str(tmp_path), dry_run=True)
    )

    assert seen["dry_run"] is True
    assert response.materialize.dry_run is True
