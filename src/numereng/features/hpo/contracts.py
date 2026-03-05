"""Contracts for HPO feature workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

HpoDirection = Literal["maximize", "minimize"]
HpoSampler = Literal["tpe", "random"]
HpoNeutralizationMode = Literal["era", "global"]
HpoParamKind = Literal["float", "int", "categorical"]
HpoStatus = Literal["running", "completed", "failed"]
HpoTrialStatus = Literal["pending", "running", "completed", "failed"]


@dataclass(frozen=True)
class HpoParameterSpec:
    """Resolved spec for one sampled parameter."""

    path: str
    kind: HpoParamKind
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: tuple[str | int | float, ...] = ()


@dataclass(frozen=True)
class HpoStudyCreateRequest:
    """Input payload for creating and running one HPO study."""

    study_name: str
    config_path: Path
    experiment_id: str | None = None
    metric: str = "bmc_last_200_eras.mean"
    direction: HpoDirection = "maximize"
    n_trials: int = 100
    sampler: HpoSampler = "tpe"
    seed: int | None = 1337
    search_space: dict[str, dict[str, Any]] | None = None
    neutralize: bool = False
    neutralizer_path: Path | None = None
    neutralization_proportion: float = 0.5
    neutralization_mode: HpoNeutralizationMode = "era"
    neutralizer_cols: tuple[str, ...] | None = None
    neutralization_rank_output: bool = True


@dataclass(frozen=True)
class HpoTrialResult:
    """Result payload for one executed HPO trial."""

    study_id: str
    trial_number: int
    status: HpoTrialStatus
    params: dict[str, Any]
    value: float | None
    run_id: str | None
    config_path: Path
    started_at: str
    finished_at: str | None
    error_message: str | None = None


@dataclass(frozen=True)
class HpoStudyResult:
    """Result payload for one completed HPO study."""

    study_id: str
    study_name: str
    experiment_id: str | None
    status: HpoStatus
    metric: str
    direction: HpoDirection
    n_trials: int
    sampler: HpoSampler
    seed: int | None
    best_trial_number: int | None
    best_value: float | None
    best_run_id: str | None
    storage_path: Path
    config: dict[str, Any]
    trials: tuple[HpoTrialResult, ...]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class HpoStudyRecord:
    """Read-model for one stored HPO study."""

    study_id: str
    experiment_id: str | None
    study_name: str
    status: HpoStatus
    metric: str
    direction: HpoDirection
    n_trials: int
    sampler: HpoSampler
    seed: int | None
    best_trial_number: int | None
    best_value: float | None
    best_run_id: str | None
    config: dict[str, Any]
    storage_path: Path | None
    error_message: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class HpoTrialRecord:
    """Read-model for one stored HPO trial."""

    study_id: str
    trial_number: int
    status: HpoTrialStatus
    value: float | None
    run_id: str | None
    config_path: Path | None
    params: dict[str, Any]
    error_message: str | None
    started_at: str | None
    finished_at: str | None
    updated_at: str
