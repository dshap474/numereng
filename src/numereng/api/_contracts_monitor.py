"""Monitor snapshot public contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MonitorSnapshotRequest(BaseModel):
    store_root: str = ".numereng"
    refresh_cloud: bool = True


class MonitorSourceResponse(BaseModel):
    kind: str
    id: str
    label: str
    host: str | None = None
    store_root: str
    state: str = "live"


class MonitorSummaryResponse(BaseModel):
    total_experiments: int
    active_experiments: int
    completed_experiments: int
    live_experiments: int
    live_runs: int
    queued_runs: int
    attention_count: int


class MonitorExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    run_count: int | None = None
    tags: list[str] = Field(default_factory=list)
    has_live: bool = False
    live_run_count: int = 0
    attention_state: str = "none"
    latest_activity_at: str | None = None
    source_kind: str = "local"
    source_id: str = "local"
    source_label: str = "Local store"
    detail_href: str | None = None


class MonitorLiveRunResponse(BaseModel):
    run_id: str
    experiment_id: str | None = None
    experiment_name: str | None = None
    job_id: str | None = None
    config_id: str | None = None
    config_label: str
    status: str
    current_stage: str | None = None
    progress_percent: float | None = None
    progress_label: str | None = None
    updated_at: str | None = None
    terminal_reason: str | None = None
    source_kind: str = "local"
    source_id: str = "local"
    source_label: str = "Local store"
    backend: str | None = None
    provider_run_id: str | None = None
    detail_href: str | None = None


class MonitorLiveExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    status: str
    tags: list[str] = Field(default_factory=list)
    live_run_count: int
    queued_run_count: int
    attention_state: str = "none"
    latest_activity_at: str | None = None
    aggregate_progress_percent: float | None = None
    runs: list[MonitorLiveRunResponse] = Field(default_factory=list)
    source_kind: str = "local"
    source_id: str = "local"
    source_label: str = "Local store"
    detail_href: str | None = None


class MonitorRecentActivityResponse(BaseModel):
    experiment_id: str
    experiment_name: str
    run_id: str | None = None
    job_id: str | None = None
    config_id: str | None = None
    config_label: str
    status: str
    current_stage: str | None = None
    progress_percent: float | None = None
    progress_label: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    terminal_reason: str | None = None
    source_kind: str = "local"
    source_id: str = "local"
    source_label: str = "Local store"
    backend: str | None = None
    provider_run_id: str | None = None


class MonitorSnapshotResponse(BaseModel):
    generated_at: str
    source: MonitorSourceResponse
    summary: MonitorSummaryResponse
    experiments: list[MonitorExperimentResponse] = Field(default_factory=list)
    live_experiments: list[MonitorLiveExperimentResponse] = Field(default_factory=list)
    live_runs: list[MonitorLiveRunResponse] = Field(default_factory=list)
    recent_activity: list[MonitorRecentActivityResponse] = Field(default_factory=list)


__all__ = [
    "MonitorExperimentResponse",
    "MonitorLiveExperimentResponse",
    "MonitorLiveRunResponse",
    "MonitorRecentActivityResponse",
    "MonitorSnapshotRequest",
    "MonitorSnapshotResponse",
    "MonitorSourceResponse",
    "MonitorSummaryResponse",
]
