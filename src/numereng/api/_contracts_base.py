"""Core public API request/response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from numereng.config.training import ensure_json_config_path
from numereng.config.training.contracts import PostTrainingScoringPolicy
from numereng.features.experiments import ExperimentStatus
from numereng.features.training import TrainingEngineMode, TrainingProfile

NumeraiTournament = Literal["classic", "signals", "crypto"]
NeutralizationMode = Literal["era", "global"]
ScoringStage = Literal["all", "run_metric_series", "post_fold", "post_training_core", "post_training_full"]
ExperimentScoreRoundStage = Literal["post_training_core", "post_training_full"]
ResearchSupervisorStatus = Literal["initialized", "running", "interrupted", "stopped", "failed"]
ResearchProgramSource = Literal["builtin", "user", "legacy_builtin"]
ResearchPlannerContract = Literal["config_mutation", "structured_json"]


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


class ScoreRunRequest(BaseModel):
    run_id: str = Field(min_length=1)
    store_root: str = ".numereng"
    stage: ScoringStage = "all"


class ScoreRunResponse(BaseModel):
    run_id: str
    predictions_path: str
    results_path: str
    metrics_path: str
    score_provenance_path: str
    requested_stage: ScoringStage = "all"
    refreshed_stages: list[str] = Field(default_factory=list)


class RunLifecycleRequest(BaseModel):
    run_id: str = Field(min_length=1)
    store_root: str = ".numereng"


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


class RunCancelRequest(BaseModel):
    run_id: str = Field(min_length=1)
    store_root: str = ".numereng"


class RunCancelResponse(BaseModel):
    run_id: str
    job_id: str
    status: str
    cancel_requested: bool
    cancel_requested_at: str | None = None
    accepted: bool


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
    post_training_scoring: PostTrainingScoringPolicy | None = None
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


class ExperimentScoreRoundRequest(BaseModel):
    experiment_id: str
    round: str
    stage: ExperimentScoreRoundStage
    store_root: str = ".numereng"


class ExperimentScoreRoundResponse(BaseModel):
    experiment_id: str
    round: str
    stage: ExperimentScoreRoundStage
    run_ids: list[str] = Field(default_factory=list)


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


class ExperimentPackRequest(BaseModel):
    experiment_id: str
    store_root: str = ".numereng"


class ExperimentPackResponse(BaseModel):
    experiment_id: str
    output_path: str
    experiment_path: str
    source_markdown_path: str
    run_count: int
    packed_at: str


class ResearchProgramMetricPolicyResponse(BaseModel):
    primary: str
    tie_break: str
    sanity_checks: list[str] = Field(default_factory=list)


class ResearchProgramRoundPolicyResponse(BaseModel):
    plateau_non_improving_rounds: int
    require_scale_confirmation: bool
    scale_confirmation_rounds: int


class ResearchProgramConfigPolicyResponse(BaseModel):
    allowed_paths: list[str] = Field(default_factory=list)
    min_candidate_configs: int | None = None
    max_candidate_configs: int
    min_changes: int | None = None
    max_changes: int | None = None


class ResearchProgramPhaseResponse(BaseModel):
    phase_id: str
    title: str
    summary: str
    gate: str


class ResearchProgramCatalogEntryResponse(BaseModel):
    program_id: str
    title: str
    description: str
    source: ResearchProgramSource
    planner_contract: ResearchPlannerContract
    phase_aware: bool
    source_path: str | None = None


class ResearchProgramListRequest(BaseModel):
    pass


class ResearchProgramListResponse(BaseModel):
    programs: list[ResearchProgramCatalogEntryResponse]


class ResearchProgramShowRequest(BaseModel):
    program_id: str = Field(min_length=1)


class ResearchProgramShowResponse(BaseModel):
    program_id: str
    title: str
    description: str
    source: ResearchProgramSource
    planner_contract: ResearchPlannerContract
    scoring_stage: ExperimentScoreRoundStage
    metric_policy: ResearchProgramMetricPolicyResponse
    round_policy: ResearchProgramRoundPolicyResponse
    improvement_threshold_default: float
    config_policy: ResearchProgramConfigPolicyResponse
    phases: list[ResearchProgramPhaseResponse] = Field(default_factory=list)
    source_path: str | None = None
    raw_markdown: str


class ResearchInitRequest(BaseModel):
    experiment_id: str
    program_id: str = Field(min_length=1)
    improvement_threshold: float | None = Field(default=None, gt=0.0)
    store_root: str = ".numereng"


class ResearchBestRunResponse(BaseModel):
    experiment_id: str | None = None
    run_id: str | None = None
    bmc_last_200_eras_mean: float | None = None
    bmc_mean: float | None = None
    corr_mean: float | None = None
    mmc_mean: float | None = None
    cwmm_mean: float | None = None
    updated_at: str | None = None


class ResearchRoundResponse(BaseModel):
    round_number: int
    round_label: str
    experiment_id: str
    path_id: str
    status: str
    next_config_index: int
    config_filenames: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    decision_action: str | None = None
    experiment_question: str | None = None
    winner_criteria: str | None = None
    decision_rationale: str | None = None
    decision_path_hypothesis: str | None = None
    decision_path_slug: str | None = None
    phase_id: str | None = None
    phase_action: str | None = None
    phase_transition_rationale: str | None = None
    started_at: str | None = None
    updated_at: str | None = None


class ResearchPhaseResponse(BaseModel):
    phase_id: str
    phase_title: str
    status: str
    round_count: int
    transition_rationale: str | None = None
    started_at: str
    updated_at: str


class ResearchInitResponse(BaseModel):
    root_experiment_id: str
    program_id: str = ""
    program_title: str = ""
    status: ResearchSupervisorStatus
    active_experiment_id: str
    active_path_id: str
    improvement_threshold: float
    current_phase: ResearchPhaseResponse | None = None
    agentic_research_dir: str
    program_path: str
    lineage_path: str
    session_program_path: str = ""


class ResearchStatusRequest(BaseModel):
    experiment_id: str
    store_root: str = ".numereng"


class ResearchStatusResponse(BaseModel):
    root_experiment_id: str
    program_id: str = ""
    program_title: str = ""
    status: ResearchSupervisorStatus
    active_experiment_id: str
    active_path_id: str
    next_round_number: int
    total_rounds_completed: int
    total_paths_created: int
    improvement_threshold: float
    last_checkpoint: str
    stop_reason: str | None = None
    best_overall: ResearchBestRunResponse
    current_round: ResearchRoundResponse | None = None
    current_phase: ResearchPhaseResponse | None = None
    program_path: str
    lineage_path: str
    session_program_path: str = ""


class ResearchRunRequest(BaseModel):
    experiment_id: str
    max_rounds: int | None = Field(default=None, ge=1)
    max_paths: int | None = Field(default=None, ge=1)
    store_root: str = ".numereng"


class ResearchRunResponse(BaseModel):
    root_experiment_id: str
    program_id: str = ""
    program_title: str = ""
    status: ResearchSupervisorStatus
    active_experiment_id: str
    active_path_id: str
    next_round_number: int
    total_rounds_completed: int
    total_paths_created: int
    last_checkpoint: str
    stop_reason: str | None = None
    current_phase: ResearchPhaseResponse | None = None
    interrupted: bool = False


__all__ = [
    "ExperimentCreateRequest",
    "ExperimentArchiveRequest",
    "ExperimentArchiveResponse",
    "ExperimentGetRequest",
    "ExperimentListRequest",
    "ExperimentListResponse",
    "ExperimentPromoteRequest",
    "ExperimentPromoteResponse",
    "ExperimentScoreRoundRequest",
    "ExperimentScoreRoundResponse",
    "ExperimentPackRequest",
    "ExperimentPackResponse",
    "ExperimentReportRequest",
    "ExperimentReportResponse",
    "ExperimentReportRowResponse",
    "ResearchBestRunResponse",
    "ResearchPlannerContract",
    "ResearchInitRequest",
    "ResearchInitResponse",
    "ResearchPhaseResponse",
    "ResearchProgramCatalogEntryResponse",
    "ResearchProgramConfigPolicyResponse",
    "ResearchProgramListRequest",
    "ResearchProgramListResponse",
    "ResearchProgramMetricPolicyResponse",
    "ResearchProgramPhaseResponse",
    "ResearchProgramRoundPolicyResponse",
    "ResearchProgramShowRequest",
    "ResearchProgramShowResponse",
    "ResearchProgramSource",
    "ResearchRoundResponse",
    "ResearchRunRequest",
    "ResearchRunResponse",
    "ResearchStatusRequest",
    "ResearchStatusResponse",
    "ResearchSupervisorStatus",
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
    "RunCancelRequest",
    "RunCancelResponse",
    "RunLifecycleRequest",
    "RunLifecycleResponse",
    "ScoreRunRequest",
    "ScoreRunResponse",
    "SubmissionRequest",
    "SubmissionResponse",
    "PostTrainingScoringPolicy",
    "TrainingEngineMode",
    "TrainingProfile",
    "TrainRunRequest",
    "TrainRunResponse",
]
