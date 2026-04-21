"""Experiment workflow request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from numereng.api._contracts.shared import (
    ExperimentScoreRoundStage,
    ExperimentStatus,
    PostTrainingScoringPolicy,
    TrainingEngineMode,
    TrainingProfile,
    WorkspaceBoundRequest,
)
from numereng.config.training import ensure_json_config_path


class ExperimentCreateRequest(WorkspaceBoundRequest):
    experiment_id: str
    name: str | None = None
    hypothesis: str | None = None
    tags: list[str] = Field(default_factory=list)


class ExperimentListRequest(WorkspaceBoundRequest):
    status: ExperimentStatus | None = None


class ExperimentGetRequest(WorkspaceBoundRequest):
    experiment_id: str


class ExperimentArchiveRequest(WorkspaceBoundRequest):
    experiment_id: str


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    status: ExperimentStatus
    hypothesis: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    champion_run_id: str | None = None
    runs: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    manifest_path: str


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]


class ExperimentArchiveResponse(BaseModel):
    experiment_id: str
    status: ExperimentStatus
    manifest_path: str
    archived: bool


class ExperimentTrainRequest(WorkspaceBoundRequest):
    experiment_id: str
    config_path: str
    output_dir: str | None = None
    profile: TrainingProfile | None = None
    post_training_scoring: PostTrainingScoringPolicy | None = None
    engine_mode: TrainingEngineMode | None = None
    window_size_eras: int | None = Field(default=None, ge=1)
    embargo_eras: int | None = Field(default=None, ge=1)

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")

    @field_validator("profile", mode="before")
    @classmethod
    def _reject_submission_profile(cls, value: object) -> object:
        if value is not None and str(value) == "submission":
            raise ValueError("training profile 'submission' was renamed to 'full_history_refit'")
        return value


class ExperimentTrainResponse(BaseModel):
    experiment_id: str
    run_id: str
    predictions_path: str
    results_path: str


class ExperimentRunPlanRequest(WorkspaceBoundRequest):
    experiment_id: str
    start_index: int = Field(default=1, ge=1)
    end_index: int | None = Field(default=None, ge=1)
    score_stage: ExperimentScoreRoundStage = "post_training_core"
    resume: bool = False


class ExperimentRunPlanWindowResponse(BaseModel):
    start_index: int
    end_index: int
    total_rows: int


class ExperimentRunPlanResponse(BaseModel):
    experiment_id: str
    state_path: str
    window: ExperimentRunPlanWindowResponse
    phase: str
    requested_score_stage: ExperimentScoreRoundStage
    completed_score_stages: list[str] = Field(default_factory=list)
    current_index: int | None = None
    current_round: str | None = None
    current_config_path: str | None = None
    current_run_id: str | None = None
    last_completed_row_index: int | None = None
    supervisor_pid: int | None = None
    active_worker_pid: int | None = None
    last_successful_heartbeat_at: str | None = None
    failure_classifier: str | None = None
    retry_count: int = 0
    terminal_error: str | None = None
    updated_at: str


class ExperimentScoreRoundRequest(WorkspaceBoundRequest):
    experiment_id: str
    round: str
    stage: ExperimentScoreRoundStage


class ExperimentScoreRoundResponse(BaseModel):
    experiment_id: str
    round: str
    stage: ExperimentScoreRoundStage
    run_ids: list[str] = Field(default_factory=list)


class ExperimentPromoteRequest(WorkspaceBoundRequest):
    experiment_id: str
    run_id: str | None = None
    metric: str = "bmc_last_200_eras.mean"


class ExperimentPromoteResponse(BaseModel):
    experiment_id: str
    champion_run_id: str
    metric: str
    metric_value: float | None = None
    auto_selected: bool


class ExperimentReportRequest(WorkspaceBoundRequest):
    experiment_id: str
    metric: str = "bmc_last_200_eras.mean"
    limit: int = Field(default=10, ge=1)


class ExperimentReportRowResponse(BaseModel):
    run_id: str
    status: str | None = None
    created_at: str | None = None
    metric_value: float | None = None
    corr_mean: float | None = None
    mmc_mean: float | None = None
    cwmm_mean: float | None = None
    bmc_mean: float | None = None
    bmc_last_200_eras_mean: float | None = None
    is_champion: bool = False


class ExperimentReportResponse(BaseModel):
    experiment_id: str
    metric: str
    total_runs: int
    champion_run_id: str | None = None
    rows: list[ExperimentReportRowResponse]


class ExperimentPackRequest(WorkspaceBoundRequest):
    experiment_id: str


class ExperimentPackResponse(BaseModel):
    experiment_id: str
    output_path: str
    experiment_path: str
    source_markdown_path: str
    run_count: int
    packed_at: str


__all__ = [
    "ExperimentArchiveRequest",
    "ExperimentArchiveResponse",
    "ExperimentCreateRequest",
    "ExperimentGetRequest",
    "ExperimentListRequest",
    "ExperimentListResponse",
    "ExperimentPackRequest",
    "ExperimentPackResponse",
    "ExperimentPromoteRequest",
    "ExperimentPromoteResponse",
    "ExperimentReportRequest",
    "ExperimentReportResponse",
    "ExperimentReportRowResponse",
    "ExperimentResponse",
    "ExperimentRunPlanRequest",
    "ExperimentRunPlanResponse",
    "ExperimentRunPlanWindowResponse",
    "ExperimentScoreRoundRequest",
    "ExperimentScoreRoundResponse",
    "ExperimentTrainRequest",
    "ExperimentTrainResponse",
]
