from __future__ import annotations

from pathlib import Path

import pytest

from numereng.api import _submissions as submissions_api
from numereng.api.contracts import (
    SubmissionCalibrationMaterializeRequest,
    SubmissionCalibrationMaterializeResponse,
    SubmissionCalibrationReportRequest,
    SubmissionCalibrationReportResponse,
    SubmissionCalibrationUpdateRequest,
    SubmissionRefreshResponse,
)


def _fake_refresh(tmp_path: Path):
    def _refresh(request: object) -> SubmissionRefreshResponse:
        return SubmissionRefreshResponse(
            workspace_root=str(tmp_path),
            dry_run=request.dry_run,
            refreshed_count=0,
            skipped_count=0,
            items=[],
        )

    return _refresh


def _materialize_response(tmp_path: Path, *, dry_run: bool) -> SubmissionCalibrationMaterializeResponse:
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
        dry_run=dry_run,
    )


def _report_response(tmp_path: Path) -> SubmissionCalibrationReportResponse:
    return SubmissionCalibrationReportResponse(
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
    )


def _bind_calibration_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: dict[str, int],
) -> None:
    monkeypatch.setattr(submissions_api, "submissions_refresh", _fake_refresh(tmp_path))

    def fake_materialize(request: SubmissionCalibrationMaterializeRequest) -> SubmissionCalibrationMaterializeResponse:
        calls["materialize"] += 1
        return _materialize_response(tmp_path, dry_run=request.dry_run)

    def fake_report(request: SubmissionCalibrationReportRequest) -> SubmissionCalibrationReportResponse:
        calls["report"] += 1
        return _report_response(tmp_path)

    monkeypatch.setattr(submissions_api, "submissions_calibration_materialize", fake_materialize)
    monkeypatch.setattr(submissions_api, "submissions_calibration_report", fake_report)


def test_submissions_calibration_update_dry_run_skips_materialize_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run refresh does not write live_rounds.parquet, so materialize/report would reflect
    # stale on-disk rounds. The update orchestrator must skip them and preview the pull only.
    calls = {"materialize": 0, "report": 0}
    _bind_calibration_steps(monkeypatch, tmp_path, calls)

    response = submissions_api.submissions_calibration_update(
        SubmissionCalibrationUpdateRequest(workspace_root=str(tmp_path), dry_run=True)
    )

    assert calls == {"materialize": 0, "report": 0}
    assert response.materialize is None
    assert response.report is None
    assert response.refresh.dry_run is True


def test_submissions_calibration_update_real_run_invokes_materialize_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"materialize": 0, "report": 0}
    _bind_calibration_steps(monkeypatch, tmp_path, calls)

    response = submissions_api.submissions_calibration_update(
        SubmissionCalibrationUpdateRequest(workspace_root=str(tmp_path), dry_run=False)
    )

    assert calls == {"materialize": 1, "report": 1}
    assert response.materialize is not None
    assert response.materialize.dry_run is False
    assert response.report is not None
