"""Core public API request/response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from numereng.config.training import ensure_json_config_path
from numereng.features.experiments import ExperimentStatus
from numereng.features.training import TrainingEngineMode, TrainingProfile

NumeraiTournament = Literal["classic", "signals", "crypto"]
NeutralizationMode = Literal["era", "global"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    package: Literal["numereng"] = "numereng"
    version: str


class SubmissionRequest(BaseModel):
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
    store_root: str = ".numereng"

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


class NeutralizeRequest(BaseModel):
    run_id: str | None = None
    predictions_path: str | None = None
    neutralizer_path: str
    neutralization_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    neutralization_mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output: bool = True
    output_path: str | None = None
    store_root: str = ".numereng"

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


class TrainRunRequest(BaseModel):
    config_path: str
    output_dir: str | None = None
    profile: TrainingProfile | None = None
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


class ScoreRunRequest(BaseModel):
    run_id: str = Field(min_length=1)
    store_root: str = ".numereng"


class ScoreRunResponse(BaseModel):
    run_id: str
    predictions_path: str
    results_path: str
    metrics_path: str
    score_provenance_path: str
    effective_scoring_backend: str


class ExperimentCreateRequest(BaseModel):
    experiment_id: str
    name: str | None = None
    hypothesis: str | None = None
    tags: list[str] = Field(default_factory=list)
    store_root: str = ".numereng"


class ExperimentListRequest(BaseModel):
    status: ExperimentStatus | None = None
    store_root: str = ".numereng"


class ExperimentGetRequest(BaseModel):
    experiment_id: str
    store_root: str = ".numereng"


class ExperimentArchiveRequest(BaseModel):
    experiment_id: str
    store_root: str = ".numereng"


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


class ExperimentTrainRequest(BaseModel):
    experiment_id: str
    config_path: str
    output_dir: str | None = None
    profile: TrainingProfile | None = None
    engine_mode: TrainingEngineMode | None = None
    window_size_eras: int | None = Field(default=None, ge=1)
    embargo_eras: int | None = Field(default=None, ge=1)
    store_root: str = ".numereng"

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


class ExperimentPromoteRequest(BaseModel):
    experiment_id: str
    run_id: str | None = None
    metric: str = "bmc_last_200_eras.mean"
    store_root: str = ".numereng"


class ExperimentPromoteResponse(BaseModel):
    experiment_id: str
    champion_run_id: str
    metric: str
    metric_value: float | None = None
    auto_selected: bool


class ExperimentReportRequest(BaseModel):
    experiment_id: str
    metric: str = "bmc_last_200_eras.mean"
    limit: int = Field(default=10, ge=1)
    store_root: str = ".numereng"


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


__all__ = [
    "ExperimentCreateRequest",
    "ExperimentArchiveRequest",
    "ExperimentArchiveResponse",
    "ExperimentGetRequest",
    "ExperimentListRequest",
    "ExperimentListResponse",
    "ExperimentPromoteRequest",
    "ExperimentPromoteResponse",
    "ExperimentReportRequest",
    "ExperimentReportResponse",
    "ExperimentReportRowResponse",
    "ExperimentResponse",
    "ExperimentStatus",
    "ExperimentTrainRequest",
    "ExperimentTrainResponse",
    "HealthResponse",
    "NeutralizationMode",
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
    "NumeraiTournament",
    "ScoreRunRequest",
    "ScoreRunResponse",
    "SubmissionRequest",
    "SubmissionResponse",
    "TrainingEngineMode",
    "TrainingProfile",
    "TrainRunRequest",
    "TrainRunResponse",
]
