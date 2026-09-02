"""Agentic config-research request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from numereng.api._contracts.shared import (
    ResearchSupervisorStatus,
    WorkspaceBoundRequest,
)


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
    champion: dict[str, object] | None = None
    agentic_research_dir: str
    closeout_memo: str


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
    champion: dict[str, object] | None = None
    rounds: list[ResearchRoundResponse] = Field(default_factory=list)
    interrupted: bool = False


class ResearchCloseoutRequest(WorkspaceBoundRequest):
    experiment_id: str
    allow_incomplete: bool = False


class ResearchCloseoutResponse(BaseModel):
    experiment_id: str
    evidence_path: str
    memo_path: str
    holdout_summary: dict[str, object] | None = None


__all__ = [
    "ResearchCloseoutRequest",
    "ResearchCloseoutResponse",
    "ResearchRoundResponse",
    "ResearchRunRequest",
    "ResearchRunResponse",
    "ResearchStatusRequest",
    "ResearchStatusResponse",
    "ResearchSupervisorStatus",
]
