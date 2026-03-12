"""Contracts for ensemble feature workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

EnsembleMethod = Literal["rank_avg"]
EnsembleStatus = Literal["running", "completed", "failed"]
EnsembleNeutralizationMode = Literal["era", "global"]


@dataclass(frozen=True)
class EnsembleBuildRequest:
    """Input payload for building one ensemble from existing run IDs."""

    run_ids: tuple[str, ...]
    experiment_id: str | None = None
    method: EnsembleMethod = "rank_avg"
    metric: str = "corr_sharpe"
    target: str = "target_ender_20"
    name: str | None = None
    ensemble_id: str | None = None
    weights: tuple[float, ...] | None = None
    optimize_weights: bool = False
    include_heavy_artifacts: bool = False
    selection_note: str | None = None
    regime_buckets: int = 4
    neutralize_members: bool = False
    neutralize_final: bool = False
    neutralizer_path: Path | None = None
    neutralization_proportion: float = 0.5
    neutralization_mode: EnsembleNeutralizationMode = "era"
    neutralizer_cols: tuple[str, ...] | None = None
    neutralization_rank_output: bool = True


@dataclass(frozen=True)
class EnsembleComponent:
    """One weighted component in an ensemble."""

    run_id: str
    weight: float
    rank: int


@dataclass(frozen=True)
class EnsembleMetric:
    """One scalar metric for an ensemble."""

    name: str
    value: float | None


@dataclass(frozen=True)
class EnsembleResult:
    """Result payload for one built ensemble."""

    ensemble_id: str
    experiment_id: str | None
    name: str
    method: EnsembleMethod
    target: str
    metric: str
    status: EnsembleStatus
    components: tuple[EnsembleComponent, ...]
    metrics: tuple[EnsembleMetric, ...]
    artifacts_path: Path
    config: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class EnsembleRecord:
    """Read-model for one persisted ensemble."""

    ensemble_id: str
    experiment_id: str | None
    name: str
    method: EnsembleMethod
    target: str
    metric: str
    status: EnsembleStatus
    components: tuple[EnsembleComponent, ...]
    metrics: tuple[EnsembleMetric, ...]
    artifacts_path: Path | None
    config: dict[str, Any]
    created_at: str
    updated_at: str
