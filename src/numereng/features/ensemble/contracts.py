"""Contracts for ensemble feature workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

EnsembleMethod = Literal["rank_avg"]
EnsembleStatus = Literal["running", "completed", "failed"]
EnsembleNeutralizationMode = Literal["era", "global"]
EnsembleSelectionBundlePolicy = Literal["seed_avg"]
EnsembleSelectionVariantName = Literal[
    "all_surviving",
    "medium_only",
    "small_only",
    "top2_medium_top2_small",
    "top3_overall",
]
EnsembleSelectionMode = Literal["explicit_targets", "top_n"]
EnsembleSelectionSelectionMode = Literal["equal_weight", "weighted", "equal_weight_retained"]


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


@dataclass(frozen=True)
class EnsembleSelectionSourceRule:
    """Candidate-freeze rule for one historical source experiment."""

    experiment_id: str
    selection_mode: EnsembleSelectionMode
    explicit_targets: tuple[str, ...] = ()
    top_n: int | None = None


@dataclass(frozen=True)
class EnsembleSelectionRequest:
    """Input payload for experiment-aware ensemble selection."""

    experiment_id: str
    source_experiment_ids: tuple[str, ...]
    source_rules: tuple[EnsembleSelectionSourceRule, ...]
    selection_id: str | None = None
    target: str = "target_ender_20"
    primary_metric: str = "bmc_last_200_eras.mean"
    tie_break_metric: str = "bmc.mean"
    correlation_threshold: float = 0.85
    top_weighted_variants: int = 2
    weight_step: float = 0.05
    bundle_policy: EnsembleSelectionBundlePolicy = "seed_avg"
    required_seed_count: int = 1
    require_full_seed_bundle: bool = False
    blend_variants: tuple[EnsembleSelectionVariantName, ...] = (
        "all_surviving",
        "medium_only",
        "small_only",
        "top2_medium_top2_small",
        "top3_overall",
    )
    weighted_promotion_min_gain: float = 0.0005


@dataclass(frozen=True)
class EnsembleSelectionResult:
    """Result payload for one ensemble-selection workflow."""

    selection_id: str
    experiment_id: str
    target: str
    primary_metric: str
    tie_break_metric: str
    status: EnsembleStatus
    artifacts_path: Path
    frozen_candidate_count: int
    surviving_candidate_count: int
    equal_weight_variant_count: int
    weighted_candidate_count: int
    winner_blend_id: str
    winner_selection_mode: EnsembleSelectionSelectionMode
    winner_component_ids: tuple[str, ...]
    winner_weights: tuple[float, ...]
    winner_metrics: dict[str, Any]
    created_at: str
    updated_at: str
