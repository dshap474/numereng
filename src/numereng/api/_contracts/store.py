"""Store maintenance request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from numereng.api._contracts.shared import WorkspaceBoundRequest


class StoreInitRequest(WorkspaceBoundRequest):
    pass


class StoreInitResponse(BaseModel):
    store_root: str
    db_path: str
    created: bool
    schema_migration: str


class StoreIndexRequest(WorkspaceBoundRequest):
    run_id: str


class StoreIndexResponse(BaseModel):
    run_id: str
    status: str
    metrics_indexed: int
    artifacts_indexed: int
    run_path: str
    warnings: list[str]


class StoreRebuildRequest(WorkspaceBoundRequest):
    pass


class StoreRebuildFailureResponse(BaseModel):
    run_id: str
    error: str


class StoreRebuildResponse(BaseModel):
    store_root: str
    db_path: str
    scanned_runs: int
    indexed_runs: int
    failed_runs: int
    failures: list[StoreRebuildFailureResponse]


class StoreDoctorRequest(WorkspaceBoundRequest):
    fix_strays: bool = False


class StoreDoctorResponse(BaseModel):
    store_root: str
    db_path: str
    ok: bool
    issues: list[str]
    stats: dict[str, int]
    stray_cleanup_applied: bool = False
    deleted_paths: list[str] = Field(default_factory=list)
    missing_paths: list[str] = Field(default_factory=list)


class StoreRunLifecycleRepairRequest(WorkspaceBoundRequest):
    run_id: str | None = None
    active_only: bool = True


class StoreRunLifecycleRepairResponse(BaseModel):
    store_root: str
    scanned_count: int
    unchanged_count: int
    reconciled_count: int
    reconciled_stale_count: int
    reconciled_canceled_count: int
    run_ids: list[str] = Field(default_factory=list)


class StoreRunExecutionBackfillRequest(WorkspaceBoundRequest):
    run_id: str | None = None
    all: bool = False

    @model_validator(mode="after")
    def _validate_scope(self) -> StoreRunExecutionBackfillRequest:
        scope_flags = int(self.run_id is not None) + int(self.all)
        if scope_flags != 1:
            raise ValueError("exactly one of run_id or all=true is required")
        return self


class StoreRunExecutionBackfillResponse(BaseModel):
    store_root: str
    scanned_count: int
    updated_count: int
    skipped_count: int
    ambiguous_runs: list[str] = Field(default_factory=list)
    updated_run_ids: list[str] = Field(default_factory=list)
    skipped_run_ids: list[str] = Field(default_factory=list)


class StoreMaterializeVizArtifactsRequest(WorkspaceBoundRequest):
    kind: str
    run_id: str | None = None
    experiment_id: str | None = None
    all: bool = False

    @model_validator(mode="after")
    def _validate_scope(self) -> StoreMaterializeVizArtifactsRequest:
        scope_flags = int(self.run_id is not None) + int(self.experiment_id is not None) + int(self.all)
        if scope_flags != 1:
            raise ValueError("exactly one of run_id, experiment_id, or all=true is required")
        return self


class StoreMaterializeVizArtifactsFailureResponse(BaseModel):
    run_id: str
    error: str


class StoreMaterializeVizArtifactsResponse(BaseModel):
    store_root: str
    kind: str
    scoped_run_count: int
    created_count: int
    skipped_count: int
    failed_count: int
    failures: list[StoreMaterializeVizArtifactsFailureResponse]


__all__ = [
    "StoreDoctorRequest",
    "StoreDoctorResponse",
    "StoreIndexRequest",
    "StoreIndexResponse",
    "StoreInitRequest",
    "StoreInitResponse",
    "StoreMaterializeVizArtifactsFailureResponse",
    "StoreMaterializeVizArtifactsRequest",
    "StoreMaterializeVizArtifactsResponse",
    "StoreRebuildFailureResponse",
    "StoreRebuildRequest",
    "StoreRebuildResponse",
    "StoreRunExecutionBackfillRequest",
    "StoreRunExecutionBackfillResponse",
    "StoreRunLifecycleRepairRequest",
    "StoreRunLifecycleRepairResponse",
]
