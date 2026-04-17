"""Serving evaluation and diagnostics sync contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from numereng.api._contracts_base import WorkspaceBoundRequest
from numereng.api._contracts_ops import ServePackageResponse

ServePackageDataset = Literal["validation"]
ServePackageScoreRuntime = Literal["auto", "pickle", "local"]
ServePackageScoreStage = Literal["post_training_core", "post_training_full"]


class ServePackageScoreRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    dataset: ServePackageDataset = "validation"
    runtime: ServePackageScoreRuntime = "auto"
    stage: ServePackageScoreStage = "post_training_full"


class ServePackageScoreResponse(BaseModel):
    package: ServePackageResponse
    dataset: ServePackageDataset = "validation"
    data_version: str
    stage: ServePackageScoreStage
    runtime_requested: ServePackageScoreRuntime
    runtime_used: Literal["pickle", "local"]
    predictions_path: str
    score_provenance_path: str
    summaries_path: str
    metric_series_path: str
    manifest_path: str
    row_count: int
    era_count: int


class ServePackageSyncDiagnosticsRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    wait: bool = True


class ServePackageSyncDiagnosticsResponse(BaseModel):
    package: ServePackageResponse
    model_id: str
    upload_id: str
    wait_requested: bool
    diagnostics_status: str
    terminal: bool
    timed_out: bool
    synced_at: str
    compute_status_path: str
    logs_path: str
    raw_path: str | None = None
    summary_path: str | None = None
    per_era_path: str | None = None


__all__ = [
    "ServePackageDataset",
    "ServePackageScoreRequest",
    "ServePackageScoreResponse",
    "ServePackageScoreRuntime",
    "ServePackageScoreStage",
    "ServePackageSyncDiagnosticsRequest",
    "ServePackageSyncDiagnosticsResponse",
]
