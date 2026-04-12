"""Contracts for HPO feature workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

HpoDirection = Literal["maximize", "minimize"]
HpoSamplerKind = Literal["tpe", "random"]
HpoNeutralizationMode = Literal["era", "global"]
HpoParamKind = Literal["float", "int", "categorical"]
HpoStatus = Literal["running", "completed", "failed"]
HpoTrialStatus = Literal["pending", "running", "completed", "failed"]
HpoStopReason = Literal[
    "max_trials_reached",
    "max_completed_trials_reached",
    "timeout_reached",
    "plateau_reached",
    "all_trials_failed",
]


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
class HpoNeutralizationSpec:
    """Prediction-stage neutralization controls for HPO trials."""

    enabled: bool = False
    neutralizer_path: Path | None = None
    proportion: float = 0.5
    mode: HpoNeutralizationMode = "era"
    neutralizer_cols: tuple[str, ...] | None = None
    rank_output: bool = True


@dataclass(frozen=True)
class HpoObjectiveSpec:
    """Objective block for one HPO study."""

    metric: str = "bmc_last_200_eras.mean"
    direction: HpoDirection = "maximize"
    neutralization: HpoNeutralizationSpec = HpoNeutralizationSpec()


@dataclass(frozen=True)
class HpoSamplerSpec:
    """Sampler block for one HPO study."""

    kind: HpoSamplerKind = "tpe"
    seed: int | None = 1337
    n_startup_trials: int = 10
    multivariate: bool = True
    group: bool = False


@dataclass(frozen=True)
class HpoPlateauSpec:
    """Study-level plateau stop configuration."""

    enabled: bool = False
    min_completed_trials: int = 15
    patience_completed_trials: int = 10
    min_improvement_abs: float = 0.00025


@dataclass(frozen=True)
class HpoStoppingSpec:
    """Study-level stopping configuration."""

    max_trials: int = 100
    max_completed_trials: int | None = None
    timeout_seconds: int | None = None
    plateau: HpoPlateauSpec = HpoPlateauSpec()


@dataclass(frozen=True)
class HpoStudyCreateRequest:
    """Input payload for creating and running one HPO study."""

    study_id: str
    study_name: str
    config_path: Path
    experiment_id: str | None = None
    objective: HpoObjectiveSpec = HpoObjectiveSpec()
    search_space: dict[str, dict[str, Any]] | None = None
    sampler: HpoSamplerSpec = HpoSamplerSpec()
    stopping: HpoStoppingSpec = HpoStoppingSpec()


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
    """Result payload for one HPO study snapshot."""

    study_id: str
    study_name: str
    experiment_id: str | None
    status: HpoStatus
    best_trial_number: int | None
    best_value: float | None
    best_run_id: str | None
    spec: dict[str, Any]
    attempted_trials: int
    completed_trials: int
    failed_trials: int
    stop_reason: HpoStopReason | None
    storage_path: Path
    trials: tuple[HpoTrialResult, ...]
    created_at: str
    updated_at: str
    error_message: str | None = None


@dataclass(frozen=True)
class HpoStudyRecord:
    """Read-model for one stored HPO study."""

    study_id: str
    experiment_id: str | None
    study_name: str
    status: HpoStatus
    best_trial_number: int | None
    best_value: float | None
    best_run_id: str | None
    spec: dict[str, Any]
    attempted_trials: int
    completed_trials: int
    failed_trials: int
    stop_reason: HpoStopReason | None
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
