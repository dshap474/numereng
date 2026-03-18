"""Contracts for experiment lifecycle feature workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ExperimentStatus = Literal["draft", "active", "complete", "archived"]


@dataclass(frozen=True)
class ExperimentRecord:
    """Canonical experiment record loaded from manifest storage."""

    experiment_id: str
    name: str
    status: ExperimentStatus
    hypothesis: str | None
    tags: tuple[str, ...]
    created_at: str
    updated_at: str
    champion_run_id: str | None
    runs: tuple[str, ...]
    metadata: dict[str, Any]
    manifest_path: Path


@dataclass(frozen=True)
class ExperimentTrainResult:
    """Result payload for training an experiment run."""

    experiment_id: str
    run_id: str
    predictions_path: Path
    results_path: Path


@dataclass(frozen=True)
class ExperimentScoreRoundResult:
    """Result payload for deferred batch scoring of one experiment round."""

    experiment_id: str
    round: str
    stage: Literal["post_training_core", "post_training_full"]
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentArchiveResult:
    """Result payload for archiving or unarchiving an experiment."""

    experiment_id: str
    status: ExperimentStatus
    manifest_path: Path
    archived: bool


@dataclass(frozen=True)
class ExperimentPromotionResult:
    """Result payload for champion promotion."""

    experiment_id: str
    champion_run_id: str
    metric: str
    metric_value: float | None
    auto_selected: bool


@dataclass(frozen=True)
class ExperimentReportRow:
    """One ranked run row in an experiment report."""

    run_id: str
    status: str | None
    created_at: str | None
    metric_value: float | None
    corr_mean: float | None
    mmc_mean: float | None
    cwmm_mean: float | None
    bmc_mean: float | None
    bmc_last_200_eras_mean: float | None
    is_champion: bool


@dataclass(frozen=True)
class ExperimentReport:
    """Ranked report payload for one experiment."""

    experiment_id: str
    metric: str
    total_runs: int
    champion_run_id: str | None
    rows: tuple[ExperimentReportRow, ...]


@dataclass(frozen=True)
class ExperimentPackResult:
    """Generated markdown bundle for one experiment."""

    experiment_id: str
    output_path: Path
    experiment_path: Path
    source_markdown_path: Path
    run_count: int
    packed_at: str
