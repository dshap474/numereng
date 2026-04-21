"""Agentic research request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from numereng.api._contracts.shared import (
    ExperimentScoreRoundStage,
    ResearchPlannerContract,
    ResearchProgramSource,
    ResearchSupervisorStatus,
    WorkspaceBoundRequest,
)


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


class ResearchProgramListRequest(WorkspaceBoundRequest):
    pass


class ResearchProgramListResponse(BaseModel):
    programs: list[ResearchProgramCatalogEntryResponse]


class ResearchProgramShowRequest(WorkspaceBoundRequest):
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


class ResearchInitRequest(WorkspaceBoundRequest):
    experiment_id: str
    program_id: str = Field(min_length=1)
    improvement_threshold: float | None = Field(default=None, gt=0.0)


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


class ResearchStatusRequest(WorkspaceBoundRequest):
    experiment_id: str


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


class ResearchRunRequest(WorkspaceBoundRequest):
    experiment_id: str
    max_rounds: int | None = Field(default=None, ge=1)
    max_paths: int | None = Field(default=None, ge=1)


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
    "ResearchBestRunResponse",
    "ResearchInitRequest",
    "ResearchInitResponse",
    "ResearchPhaseResponse",
    "ResearchPlannerContract",
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
]
