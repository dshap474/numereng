"""Run, submission, neutralization, and Numerai API contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from numereng.api._contracts.shared import (
    NeutralizationMode,
    NumeraiTournament,
    PostTrainingScoringPolicy,
    ScoringStage,
    TrainingEngineMode,
    TrainingProfile,
    WorkspaceBoundRequest,
)
from numereng.config.training import ensure_json_config_path


class SubmissionRequest(WorkspaceBoundRequest):
    model_name: str
    tournament: NumeraiTournament = "classic"
    run_id: str | None = None
    predictions_path: str | None = None
    allow_non_live_artifact: bool = False
    neutralize: bool = False
    neutralizer_path: str | None = None
    neutralization_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    neutralization_mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output: bool = True

    @model_validator(mode="after")
    def _validate_source(self) -> SubmissionRequest:
        has_run_id = self.run_id is not None
        has_predictions_path = self.predictions_path is not None
        if has_run_id == has_predictions_path:
            raise ValueError("exactly one of run_id or predictions_path is required")
        if self.neutralize and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralize is true")
        return self


class SubmissionResponse(BaseModel):
    submission_id: str
    model_name: str
    model_id: str
    predictions_path: str
    run_id: str | None = None


class NeutralizeRequest(WorkspaceBoundRequest):
    run_id: str | None = None
    predictions_path: str | None = None
    neutralizer_path: str
    neutralization_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    neutralization_mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output: bool = True
    output_path: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> NeutralizeRequest:
        has_run_id = self.run_id is not None
        has_predictions_path = self.predictions_path is not None
        if has_run_id == has_predictions_path:
            raise ValueError("exactly one of run_id or predictions_path is required")
        return self


class NeutralizeResponse(BaseModel):
    source_path: str
    output_path: str
    run_id: str | None = None
    neutralizer_path: str
    neutralizer_cols: list[str] = Field(default_factory=list)
    neutralization_proportion: float
    neutralization_mode: NeutralizationMode
    neutralization_rank_output: bool
    source_rows: int
    neutralizer_rows: int
    matched_rows: int


class NumeraiDatasetListRequest(BaseModel):
    tournament: NumeraiTournament = "classic"
    round_num: int | None = None


class NumeraiDatasetListResponse(BaseModel):
    datasets: list[str]


class NumeraiDatasetDownloadRequest(BaseModel):
    filename: str
    tournament: NumeraiTournament = "classic"
    dest_path: str | None = None
    round_num: int | None = None


class NumeraiDatasetDownloadResponse(BaseModel):
    path: str


class NumeraiModelsResponse(BaseModel):
    models: dict[str, str]


class NumeraiModelsRequest(BaseModel):
    tournament: NumeraiTournament = "classic"


class NumeraiCurrentRoundResponse(BaseModel):
    round_num: int | None


class NumeraiCurrentRoundRequest(BaseModel):
    tournament: NumeraiTournament = "classic"


class TrainRunRequest(WorkspaceBoundRequest):
    config_path: str
    output_dir: str | None = None
    profile: TrainingProfile | None = None
    post_training_scoring: PostTrainingScoringPolicy | None = None
    engine_mode: TrainingEngineMode | None = None
    window_size_eras: int | None = Field(default=None, ge=1)
    embargo_eras: int | None = Field(default=None, ge=1)
    experiment_id: str | None = None

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


class TrainRunResponse(BaseModel):
    run_id: str
    predictions_path: str
    results_path: str


class ScoreRunRequest(WorkspaceBoundRequest):
    run_id: str = Field(min_length=1)
    stage: ScoringStage = "all"


class ScoreRunResponse(BaseModel):
    run_id: str
    predictions_path: str
    results_path: str
    metrics_path: str
    score_provenance_path: str
    requested_stage: ScoringStage = "all"
    refreshed_stages: list[str] = Field(default_factory=list)


class RunLifecycleRequest(WorkspaceBoundRequest):
    run_id: str = Field(min_length=1)


class RunLifecycleResponse(BaseModel):
    run_id: str
    run_hash: str
    config_hash: str
    job_id: str
    logical_run_id: str
    attempt_id: str
    attempt_no: int
    source: str
    operation_type: str
    job_type: str
    status: str
    experiment_id: str | None = None
    config_id: str
    config_source: str
    config_path: str
    config_sha256: str
    run_dir: str
    runtime_path: str
    backend: str | None = None
    worker_id: str | None = None
    pid: int | None = None
    host: str | None = None
    current_stage: str | None = None
    completed_stages: list[str] = Field(default_factory=list)
    progress_percent: float | None = None
    progress_label: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    cancel_requested: bool = False
    cancel_requested_at: str | None = None
    created_at: str
    queued_at: str | None = None
    started_at: str | None = None
    last_heartbeat_at: str | None = None
    updated_at: str
    finished_at: str | None = None
    terminal_reason: str | None = None
    terminal_detail: dict[str, object] = Field(default_factory=dict)
    latest_metrics: dict[str, object] = Field(default_factory=dict)
    latest_sample: dict[str, object] = Field(default_factory=dict)
    reconciled: bool = False


class RunCancelRequest(WorkspaceBoundRequest):
    run_id: str = Field(min_length=1)


class RunCancelResponse(BaseModel):
    run_id: str
    job_id: str
    status: str
    cancel_requested: bool
    cancel_requested_at: str | None = None
    accepted: bool


__all__ = [
    "NeutralizeRequest",
    "NeutralizeResponse",
    "NumeraiCurrentRoundRequest",
    "NumeraiCurrentRoundResponse",
    "NumeraiDatasetDownloadRequest",
    "NumeraiDatasetDownloadResponse",
    "NumeraiDatasetListRequest",
    "NumeraiDatasetListResponse",
    "NumeraiModelsRequest",
    "NumeraiModelsResponse",
    "RunCancelRequest",
    "RunCancelResponse",
    "RunLifecycleRequest",
    "RunLifecycleResponse",
    "ScoreRunRequest",
    "ScoreRunResponse",
    "SubmissionRequest",
    "SubmissionResponse",
    "TrainRunRequest",
    "TrainRunResponse",
]
