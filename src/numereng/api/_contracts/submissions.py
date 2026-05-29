"""Submission snapshot and calibration request/response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from numereng.api._contracts.shared import WorkspaceBoundRequest


class SubmissionRefreshRequest(WorkspaceBoundRequest):
    model_names: list[str] = Field(default_factory=list)
    dry_run: bool = False


class SubmissionRefreshItem(BaseModel):
    model_name: str
    model_id: str | None
    live_rounds_path: str
    submission_path: str
    round_count: int
    scored_round_count: int
    resolved_round_count: int
    resolved_scored_round_count: int
    latest_scored_round: int | None = None
    latest_resolved_round: int | None = None
    skipped: bool = False
    warning: str | None = None


class SubmissionRefreshResponse(BaseModel):
    workspace_root: str
    dry_run: bool
    refreshed_count: int
    skipped_count: int
    items: list[SubmissionRefreshItem]


class SubmissionCalibrationMaterializeRequest(WorkspaceBoundRequest):
    dry_run: bool = False


class SubmissionCalibrationMaterializeResponse(BaseModel):
    workspace_root: str
    artifact_root: str
    rows_path: str
    observations_path: str
    report_path: str
    manifest_path: str
    row_count: int
    observation_count: int
    model_count: int
    scored_row_count: int
    scored_observation_count: int
    dry_run: bool = False
    warnings: list[str] = Field(default_factory=list)


class SubmissionCalibrationReportRequest(WorkspaceBoundRequest):
    resolved_only: bool = False


class SubmissionCalibrationReportResponse(BaseModel):
    workspace_root: str
    artifact_root: str
    rows_path: str
    observations_path: str
    report_path: str
    manifest_path: str
    row_count: int
    observation_count: int
    scope: Literal["all", "resolved_only"]
    report: dict[str, Any]
    manifest: dict[str, Any]


class SubmissionCalibrationUpdateRequest(WorkspaceBoundRequest):
    model_names: list[str] = Field(default_factory=list)
    dry_run: bool = False
    resolved_only: bool = False


class SubmissionCalibrationUpdateResponse(BaseModel):
    refresh: SubmissionRefreshResponse
    materialize: SubmissionCalibrationMaterializeResponse
    report: SubmissionCalibrationReportResponse


__all__ = [
    "SubmissionCalibrationMaterializeRequest",
    "SubmissionCalibrationMaterializeResponse",
    "SubmissionCalibrationReportRequest",
    "SubmissionCalibrationReportResponse",
    "SubmissionCalibrationUpdateRequest",
    "SubmissionCalibrationUpdateResponse",
    "SubmissionRefreshItem",
    "SubmissionRefreshRequest",
    "SubmissionRefreshResponse",
]
