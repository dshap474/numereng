"""Agentic config-research request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from numereng.api._contracts.shared import (
    ResearchSupervisorStatus,
    WorkspaceBoundRequest,
)


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
    action: str
    status: str
    config_path: str | None = None
    run_id: str | None = None
    metric_value: float | None = None
    learning: str
    artifact_dir: str


class ResearchStatusRequest(WorkspaceBoundRequest):
    experiment_id: str


class ResearchStatusResponse(BaseModel):
    experiment_id: str
    status: ResearchSupervisorStatus
    next_round_number: int
    total_rounds_completed: int
    last_checkpoint: str
    last_round_label: str | None = None
    last_run_id: str | None = None
    stop_reason: str | None = None
    best_overall: ResearchBestRunResponse
    agentic_research_dir: str
    state_path: str
    trace_path: str
    decision_path: str
    program_path: str


class ResearchRunRequest(WorkspaceBoundRequest):
    experiment_id: str
    max_rounds: int = Field(default=1, ge=1)


class ResearchRunResponse(BaseModel):
    experiment_id: str
    status: ResearchSupervisorStatus
    next_round_number: int
    total_rounds_completed: int
    last_checkpoint: str
    stop_reason: str | None = None
    best_overall: ResearchBestRunResponse
    rounds: list[ResearchRoundResponse] = Field(default_factory=list)
    interrupted: bool = False


class ResearchCloseoutRequest(WorkspaceBoundRequest):
    experiment_id: str
    until: str | None = None
    restart_from: str | None = None
    memory_root: str | None = None
    accept_stale_running: bool = False
    allow_incomplete: bool = False


class ResearchCloseoutStatusRequest(WorkspaceBoundRequest):
    experiment_id: str


class ResearchCloseoutPhaseResponse(BaseModel):
    name: str
    status: str
    notes: str | None = None
    duration_seconds: float | None = None
    outputs: dict[str, str] = Field(default_factory=dict)


class ResearchCloseoutResponse(BaseModel):
    experiment_id: str
    phases: list[ResearchCloseoutPhaseResponse] = Field(default_factory=list)
    stopped_at_phase: str | None = None
    error: str | None = None


class ResearchProgramRequest(WorkspaceBoundRequest):
    experiment_id: str


class ResearchProgramResponse(BaseModel):
    experiment_id: str
    program_path: str
    base_program_path: str
    is_base_program: bool
    in_sync: bool
    diverging_section: str | None = None
    written: bool = False
    backup_path: str | None = None


__all__ = [
    "ResearchBestRunResponse",
    "ResearchCloseoutPhaseResponse",
    "ResearchCloseoutRequest",
    "ResearchCloseoutResponse",
    "ResearchCloseoutStatusRequest",
    "ResearchProgramRequest",
    "ResearchProgramResponse",
    "ResearchRoundResponse",
    "ResearchRunRequest",
    "ResearchRunResponse",
    "ResearchStatusRequest",
    "ResearchStatusResponse",
    "ResearchSupervisorStatus",
]
