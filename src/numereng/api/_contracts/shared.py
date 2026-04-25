"""Shared API contract primitives, literals, and workspace-bound helpers."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import BaseModel

from numereng.config.training.contracts import PostTrainingScoringPolicy
from numereng.features.store import WorkspaceLayout, resolve_workspace_layout

NumeraiTournament = Literal["classic", "signals", "crypto"]
NeutralizationMode = Literal["era", "global"]
ScoringStage = Literal["all", "run_metric_series", "post_fold", "post_training_core", "post_training_full"]
ExperimentScoreRoundStage = Literal["post_training_core", "post_training_full"]
ResearchSupervisorStatus = Literal["initialized", "running", "interrupted", "stopped", "failed"]
ExperimentStatus = Literal["draft", "active", "complete", "archived"]
TrainingProfile = Literal["simple", "purged_walk_forward", "full_history_refit"]
TrainingEngineMode = Literal["official", "custom", "full_history"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    package: Literal["numereng"] = "numereng"
    version: str


class WorkspaceBoundRequest(BaseModel):
    workspace_root: str = "."

    @property
    def workspace_layout(self) -> WorkspaceLayout:
        return resolve_workspace_layout(self.workspace_root)

    @property
    def store_root(self) -> str:
        if self.workspace_root in ("", "."):
            return ".numereng"
        if "\\" in self.workspace_root or re.match(r"^[A-Za-z]:", self.workspace_root):
            return str(PureWindowsPath(self.workspace_root) / ".numereng")
        return str(Path(self.workspace_root) / ".numereng")


__all__ = [
    "ExperimentScoreRoundStage",
    "ExperimentStatus",
    "HealthResponse",
    "NeutralizationMode",
    "NumeraiTournament",
    "PostTrainingScoringPolicy",
    "ResearchSupervisorStatus",
    "ScoringStage",
    "TrainingEngineMode",
    "TrainingProfile",
    "WorkspaceBoundRequest",
]
