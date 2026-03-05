"""Public API request/response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from numereng.config.training import ensure_json_config_path
from numereng.features.experiments import ExperimentStatus
from numereng.features.training import TrainingEngineMode, TrainingProfile

NumeraiTournament = Literal["classic", "signals", "crypto"]
NeutralizationMode = Literal["era", "global"]


class HealthResponse(BaseModel):
    """Public bootstrap status payload."""

    status: Literal["ok"] = "ok"
    package: Literal["numereng"] = "numereng"
    version: str


class SubmissionRequest(BaseModel):
    """Public API request for Numerai submission."""

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
    """Public API response for Numerai submission."""

    submission_id: str
    model_name: str
    model_id: str
    predictions_path: str
    run_id: str | None = None


class NeutralizeRequest(BaseModel):
    """Public API request for neutralizing one prediction source."""

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
    """Public API response payload for a neutralized predictions artifact."""

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
    """Public API request for listing Numerai datasets."""

    tournament: NumeraiTournament = "classic"
    round_num: int | None = None


class NumeraiDatasetListResponse(BaseModel):
    """Public API response for Numerai dataset listing."""

    datasets: list[str]


class NumeraiDatasetDownloadRequest(BaseModel):
    """Public API request for downloading a Numerai dataset file."""

    filename: str
    tournament: NumeraiTournament = "classic"
    dest_path: str | None = None
    round_num: int | None = None


class NumeraiDatasetDownloadResponse(BaseModel):
    """Public API response for a downloaded Numerai dataset file."""

    path: str


class NumeraiModelsResponse(BaseModel):
    """Public API response for Numerai model mapping."""

    models: dict[str, str]


class NumeraiModelsRequest(BaseModel):
    """Public API request for Numerai model mapping."""

    tournament: NumeraiTournament = "classic"


class NumeraiCurrentRoundResponse(BaseModel):
    """Public API response for current Numerai round."""

    round_num: int | None


class NumeraiCurrentRoundRequest(BaseModel):
    """Public API request for current Numerai round."""

    tournament: NumeraiTournament = "classic"


class TrainRunRequest(BaseModel):
    """Public API request for running the training pipeline."""

    config_path: str
    output_dir: str | None = None
    profile: TrainingProfile | None = None
    # Legacy compatibility keys retained for API callers migrating to `profile`.
    engine_mode: TrainingEngineMode | None = None
    window_size_eras: int | None = Field(default=None, ge=1)
    embargo_eras: int | None = Field(default=None, ge=1)
    experiment_id: str | None = None

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")


class TrainRunResponse(BaseModel):
    """Public API response for a completed training run."""

    run_id: str
    predictions_path: str
    results_path: str


class ExperimentCreateRequest(BaseModel):
    """Public API request for creating one experiment."""

    experiment_id: str
    name: str | None = None
    hypothesis: str | None = None
    tags: list[str] = Field(default_factory=list)
    store_root: str = ".numereng"


class ExperimentListRequest(BaseModel):
    """Public API request for listing experiments."""

    status: ExperimentStatus | None = None
    store_root: str = ".numereng"


class ExperimentGetRequest(BaseModel):
    """Public API request for loading one experiment."""

    experiment_id: str
    store_root: str = ".numereng"


class ExperimentResponse(BaseModel):
    """Public API response payload for one experiment."""

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
    """Public API response payload for experiment listings."""

    experiments: list[ExperimentResponse]


class ExperimentTrainRequest(BaseModel):
    """Public API request for running one experiment training run."""

    experiment_id: str
    config_path: str
    output_dir: str | None = None
    profile: TrainingProfile | None = None
    # Legacy compatibility keys retained for API callers migrating to `profile`.
    engine_mode: TrainingEngineMode | None = None
    window_size_eras: int | None = Field(default=None, ge=1)
    embargo_eras: int | None = Field(default=None, ge=1)
    store_root: str = ".numereng"

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")


class ExperimentTrainResponse(BaseModel):
    """Public API response payload for experiment-linked training run."""

    experiment_id: str
    run_id: str
    predictions_path: str
    results_path: str


class ExperimentPromoteRequest(BaseModel):
    """Public API request for champion promotion."""

    experiment_id: str
    run_id: str | None = None
    metric: str = "bmc_last_200_eras.mean"
    store_root: str = ".numereng"


class ExperimentPromoteResponse(BaseModel):
    """Public API response payload for champion promotion."""

    experiment_id: str
    champion_run_id: str
    metric: str
    metric_value: float | None = None
    auto_selected: bool


class ExperimentReportRequest(BaseModel):
    """Public API request for experiment leaderboard report."""

    experiment_id: str
    metric: str = "bmc_last_200_eras.mean"
    limit: int = Field(default=10, ge=1)
    store_root: str = ".numereng"


class ExperimentReportRowResponse(BaseModel):
    """One run row in experiment report response."""

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
    """Public API response payload for experiment report."""

    experiment_id: str
    metric: str
    total_runs: int
    champion_run_id: str | None = None
    rows: list[ExperimentReportRowResponse]


class HpoStudyCreateRequest(BaseModel):
    """Public API request for creating/running one HPO study."""

    study_name: str
    config_path: str
    experiment_id: str | None = None
    metric: str = "bmc_last_200_eras.mean"
    direction: Literal["maximize", "minimize"] = "maximize"
    n_trials: int = Field(default=100, ge=1)
    sampler: Literal["tpe", "random"] = "tpe"
    seed: int | None = 1337
    search_space: dict[str, dict[str, object]] | None = None
    neutralize: bool = False
    neutralizer_path: str | None = None
    neutralization_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    neutralization_mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output: bool = True
    store_root: str = ".numereng"

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")

    @model_validator(mode="after")
    def _validate_neutralization(self) -> HpoStudyCreateRequest:
        if self.neutralize and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralize is true")
        return self


class HpoStudyListRequest(BaseModel):
    """Public API request for listing HPO studies."""

    experiment_id: str | None = None
    status: str | None = None
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    store_root: str = ".numereng"


class HpoStudyGetRequest(BaseModel):
    """Public API request for loading one HPO study."""

    study_id: str
    store_root: str = ".numereng"


class HpoStudyTrialsRequest(BaseModel):
    """Public API request for loading HPO trials for one study."""

    study_id: str
    store_root: str = ".numereng"


class HpoTrialResponse(BaseModel):
    """Public API response payload for one HPO trial row."""

    study_id: str
    trial_number: int
    status: str
    value: float | None = None
    run_id: str | None = None
    config_path: str | None = None
    params: dict[str, object] = Field(default_factory=dict)
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str


class HpoStudyResponse(BaseModel):
    """Public API response payload for one HPO study."""

    study_id: str
    experiment_id: str | None = None
    study_name: str
    status: str
    metric: str
    direction: Literal["maximize", "minimize"]
    n_trials: int
    sampler: Literal["tpe", "random"]
    seed: int | None = None
    best_trial_number: int | None = None
    best_value: float | None = None
    best_run_id: str | None = None
    config: dict[str, object] = Field(default_factory=dict)
    storage_path: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str


class HpoStudyListResponse(BaseModel):
    """Public API response payload for HPO study listings."""

    studies: list[HpoStudyResponse]


class HpoStudyTrialsResponse(BaseModel):
    """Public API response payload for HPO study trial listings."""

    study_id: str
    trials: list[HpoTrialResponse]


class EnsembleBuildRequest(BaseModel):
    """Public API request for building one ensemble."""

    run_ids: list[str]
    experiment_id: str | None = None
    method: Literal["rank_avg"] = "rank_avg"
    metric: str = "corr20v2_sharpe"
    target: str = "target_ender_20"
    name: str | None = None
    ensemble_id: str | None = None
    weights: list[float] | None = None
    optimize_weights: bool = False
    include_heavy_artifacts: bool = False
    selection_note: str | None = None
    regime_buckets: int = Field(default=4, ge=2, le=50)
    neutralize_members: bool = False
    neutralize_final: bool = False
    neutralizer_path: str | None = None
    neutralization_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    neutralization_mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output: bool = True
    store_root: str = ".numereng"

    @model_validator(mode="after")
    def _validate_neutralization(self) -> EnsembleBuildRequest:
        if (self.neutralize_members or self.neutralize_final) and self.neutralizer_path is None:
            raise ValueError(
                "neutralizer_path is required when neutralize_members or neutralize_final is true"
            )
        return self


class EnsembleListRequest(BaseModel):
    """Public API request for listing ensembles."""

    experiment_id: str | None = None
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    store_root: str = ".numereng"


class EnsembleGetRequest(BaseModel):
    """Public API request for loading one ensemble."""

    ensemble_id: str
    store_root: str = ".numereng"


class EnsembleComponentResponse(BaseModel):
    """Public API response payload for one ensemble component row."""

    run_id: str
    weight: float
    rank: int


class EnsembleMetricResponse(BaseModel):
    """Public API response payload for one ensemble metric row."""

    name: str
    value: float | None = None


class EnsembleResponse(BaseModel):
    """Public API response payload for one ensemble."""

    ensemble_id: str
    experiment_id: str | None = None
    name: str
    method: Literal["rank_avg"]
    target: str
    metric: str
    status: str
    components: list[EnsembleComponentResponse] = Field(default_factory=list)
    metrics: list[EnsembleMetricResponse] = Field(default_factory=list)
    artifacts_path: str | None = None
    config: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class EnsembleListResponse(BaseModel):
    """Public API response payload for ensemble listings."""

    ensembles: list[EnsembleResponse]


class StoreInitRequest(BaseModel):
    """Public API request for store DB bootstrap."""

    store_root: str = ".numereng"


class StoreInitResponse(BaseModel):
    """Public API response for store DB bootstrap."""

    store_root: str
    db_path: str
    created: bool
    schema_migration: str


class StoreIndexRequest(BaseModel):
    """Public API request for indexing one run into store DB."""

    run_id: str
    store_root: str = ".numereng"


class StoreIndexResponse(BaseModel):
    """Public API response for indexing one run into store DB."""

    run_id: str
    status: str
    metrics_indexed: int
    artifacts_indexed: int
    run_path: str
    warnings: list[str]


class StoreRebuildRequest(BaseModel):
    """Public API request for full run-index rebuild."""

    store_root: str = ".numereng"


class StoreRebuildFailureResponse(BaseModel):
    """Public API response item for one failed run during rebuild."""

    run_id: str
    error: str


class StoreRebuildResponse(BaseModel):
    """Public API response for full run-index rebuild."""

    store_root: str
    db_path: str
    scanned_runs: int
    indexed_runs: int
    failed_runs: int
    failures: list[StoreRebuildFailureResponse]


class StoreDoctorRequest(BaseModel):
    """Public API request for store consistency diagnostics."""

    store_root: str = ".numereng"
    fix_strays: bool = False


class StoreDoctorResponse(BaseModel):
    """Public API response for store consistency diagnostics."""

    store_root: str
    db_path: str
    ok: bool
    issues: list[str]
    stats: dict[str, int]
    stray_cleanup_applied: bool = False
    deleted_paths: list[str] = Field(default_factory=list)
    missing_paths: list[str] = Field(default_factory=list)


class DatasetToolsBuildDownsampleRequest(BaseModel):
    """Public API request for official-style downsampled dataset build."""

    data_version: str = "v5.2"
    data_dir: str = ".numereng/datasets"
    rebuild: bool = False
    downsample_eras_step: int = 4
    downsample_eras_offset: int = 0
    skip_downsample: bool = False


class DatasetToolsBuildDownsampleResponse(BaseModel):
    """Public API response payload for downsampled dataset build."""

    data_version: str
    data_dir: str
    full_path: str
    full_benchmark_path: str
    downsampled_full_path: str | None
    downsampled_full_benchmark_path: str | None
    full_rows: int
    downsampled_rows: int | None
    full_benchmark_rows: int
    downsampled_full_benchmark_rows: int | None
    total_eras: int | None
    kept_eras: int | None
    downsample_step: int
    downsample_offset: int
    downsample_built: bool


__all__ = [
    "DatasetToolsBuildDownsampleRequest",
    "DatasetToolsBuildDownsampleResponse",
    "EnsembleBuildRequest",
    "EnsembleComponentResponse",
    "EnsembleGetRequest",
    "EnsembleListRequest",
    "EnsembleListResponse",
    "EnsembleMetricResponse",
    "EnsembleResponse",
    "ExperimentCreateRequest",
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
    "HpoStudyCreateRequest",
    "HpoStudyGetRequest",
    "HpoStudyListRequest",
    "HpoStudyListResponse",
    "HpoStudyResponse",
    "HpoStudyTrialsRequest",
    "HpoStudyTrialsResponse",
    "HpoTrialResponse",
    "NumeraiCurrentRoundRequest",
    "NumeraiCurrentRoundResponse",
    "NumeraiDatasetDownloadRequest",
    "NumeraiDatasetDownloadResponse",
    "NumeraiDatasetListRequest",
    "NumeraiDatasetListResponse",
    "NumeraiModelsRequest",
    "NumeraiModelsResponse",
    "NumeraiTournament",
    "NeutralizationMode",
    "NeutralizeRequest",
    "NeutralizeResponse",
    "StoreDoctorRequest",
    "StoreDoctorResponse",
    "StoreIndexRequest",
    "StoreIndexResponse",
    "StoreInitRequest",
    "StoreInitResponse",
    "StoreRebuildFailureResponse",
    "StoreRebuildRequest",
    "StoreRebuildResponse",
    "SubmissionRequest",
    "SubmissionResponse",
    "TrainingEngineMode",
    "TrainingProfile",
    "TrainRunRequest",
    "TrainRunResponse",
]
