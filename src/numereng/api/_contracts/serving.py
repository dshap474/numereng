"""Serving package, scoring, and diagnostics contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from numereng.api._contracts.shared import NeutralizationMode, WorkspaceBoundRequest

ServePackageDataset = Literal["validation"]
ServePackageScoreRuntime = Literal["auto", "pickle", "local"]
ServePackageScoreStage = Literal["post_training_core", "post_training_full"]


class ServeComponentRequest(BaseModel):
    component_id: str | None = None
    weight: float = Field(ge=0.0)
    config_path: str | None = None
    run_id: str | None = None
    source_label: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> ServeComponentRequest:
        scope_flags = int(self.config_path is not None) + int(self.run_id is not None)
        if scope_flags != 1:
            raise ValueError("exactly one of config_path or run_id is required")
        return self


class ServeBlendRuleRequest(BaseModel):
    per_era_rank: bool = True
    rank_method: Literal["average", "min", "max", "first", "dense"] = "average"
    rank_pct: bool = True
    final_rerank: bool = False


class ServeNeutralizationRequest(BaseModel):
    enabled: bool = False
    proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    rank_output: bool = True


class ServePackageCreateRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    tournament: Literal["classic"] = "classic"
    data_version: str = "v5.2"
    components: list[ServeComponentRequest]
    blend_rule: ServeBlendRuleRequest = Field(default_factory=ServeBlendRuleRequest)
    neutralization: ServeNeutralizationRequest | None = None
    provenance: dict[str, object] = Field(default_factory=dict)

    @field_validator("experiment_id", "package_id", "data_version")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("components")
    @classmethod
    def _validate_components_nonempty(cls, value: list[ServeComponentRequest]) -> list[ServeComponentRequest]:
        if not value:
            raise ValueError("components must not be empty")
        return value


class ServePackageListRequest(WorkspaceBoundRequest):
    experiment_id: str | None = None


class ServePackageResponse(BaseModel):
    package_id: str
    experiment_id: str
    tournament: str
    data_version: str
    package_path: str
    status: str
    components: list[ServeComponentRequest]
    blend_rule: ServeBlendRuleRequest
    neutralization: ServeNeutralizationRequest | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    provenance: dict[str, object] = Field(default_factory=dict)


class ServePackageListResponse(BaseModel):
    packages: list[ServePackageResponse]


class ServeComponentInspectionResponse(BaseModel):
    component_id: str
    local_live_compatible: bool
    model_upload_compatible: bool
    artifact_backed: bool = False
    artifact_ready: bool = False
    local_live_blockers: list[str] = Field(default_factory=list)
    model_upload_blockers: list[str] = Field(default_factory=list)
    artifact_blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ServePackageInspectRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str


class ServePackageInspectResponse(BaseModel):
    package: ServePackageResponse
    checked_at: str
    local_live_compatible: bool
    model_upload_compatible: bool
    artifact_backed: bool = False
    artifact_ready: bool = False
    artifact_live_ready: bool = False
    pickle_upload_ready: bool = False
    deployment_classification: str = "not_live_ready"
    local_live_blockers: list[str] = Field(default_factory=list)
    model_upload_blockers: list[str] = Field(default_factory=list)
    artifact_blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    components: list[ServeComponentInspectionResponse] = Field(default_factory=list)
    report_path: str | None = None


class ServeLiveBuildRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str


class ServeLiveBuildResponse(BaseModel):
    package: ServePackageResponse
    current_round: int | None = None
    live_dataset_name: str
    live_benchmark_dataset_name: str | None = None
    live_dataset_path: str
    live_benchmark_dataset_path: str | None = None
    component_prediction_paths: list[str]
    blended_predictions_path: str
    submission_predictions_path: str


class ServeLiveSubmitRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    model_name: str


class ServeLiveSubmitResponse(BaseModel):
    package: ServePackageResponse
    current_round: int | None = None
    submission_id: str
    model_name: str
    model_id: str
    submission_predictions_path: str


class ServePickleBuildRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    docker_image: str | None = None


class ServePickleBuildResponse(BaseModel):
    package: ServePackageResponse
    pickle_path: str
    docker_image: str
    smoke_verified: bool = False


class ServePickleUploadRequest(WorkspaceBoundRequest):
    experiment_id: str
    package_id: str
    model_name: str
    data_version: str | None = None
    docker_image: str | None = None
    wait_diagnostics: bool = False


class ServePickleUploadResponse(BaseModel):
    package: ServePackageResponse
    pickle_path: str
    model_name: str
    model_id: str
    upload_id: str
    data_version: str | None = None
    docker_image: str | None = None
    diagnostics_synced: bool = False
    diagnostics_status: str | None = None
    diagnostics_terminal: bool | None = None
    diagnostics_timed_out: bool | None = None
    diagnostics_synced_at: str | None = None
    diagnostics_compute_status_path: str | None = None
    diagnostics_logs_path: str | None = None
    diagnostics_raw_path: str | None = None
    diagnostics_summary_path: str | None = None
    diagnostics_per_era_path: str | None = None


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
    "ServeBlendRuleRequest",
    "ServeComponentInspectionResponse",
    "ServeComponentRequest",
    "ServeLiveBuildRequest",
    "ServeLiveBuildResponse",
    "ServeLiveSubmitRequest",
    "ServeLiveSubmitResponse",
    "ServeNeutralizationRequest",
    "ServePackageCreateRequest",
    "ServePackageDataset",
    "ServePackageInspectRequest",
    "ServePackageInspectResponse",
    "ServePackageListRequest",
    "ServePackageListResponse",
    "ServePackageResponse",
    "ServePackageScoreRequest",
    "ServePackageScoreResponse",
    "ServePackageScoreRuntime",
    "ServePackageScoreStage",
    "ServePackageSyncDiagnosticsRequest",
    "ServePackageSyncDiagnosticsResponse",
    "ServePickleBuildRequest",
    "ServePickleBuildResponse",
    "ServePickleUploadRequest",
    "ServePickleUploadResponse",
]
