"""Data structures for canonical run-scoring workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

CanonicalScoringStage = Literal[
    "all",
    "run_metric_series",
    "post_fold",
    "post_training_core",
    "post_training_full",
]


@dataclass(frozen=True)
class BenchmarkSource:
    """Resolved benchmark prediction source used during one scoring pass."""

    mode: str
    name: str
    predictions_path: Path
    pred_col: str
    metadata_path: Path | None = None


@dataclass(frozen=True)
class ScoringArtifactBundle:
    """All persisted scoring-side parquet/json artifacts for one run."""

    series_frames: dict[str, pd.DataFrame]
    manifest: dict[str, object]
    stage_frames: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True)
class PostTrainingScoringRequest:
    """Inputs required to compute canonical run-scoring artifacts."""

    run_id: str
    config_hash: str
    seed: int | None
    predictions_path: Path
    pred_cols: tuple[str, ...]
    target_col: str
    scoring_target_cols: tuple[str, ...]
    data_version: str
    dataset_variant: str
    feature_set: str
    feature_source_paths: tuple[Path, ...] | None
    dataset_scope: str
    benchmark_source: BenchmarkSource
    meta_model_col: str
    meta_model_data_path: str | Path | None
    era_col: str
    id_col: str
    data_root: Path
    scoring_targets_explicit: bool = False
    stage: CanonicalScoringStage = "all"


@dataclass(frozen=True)
class ResolvedScoringPolicy:
    """Explicit scoring semantics resolved for a post-training run."""

    fnc_feature_set: str
    fnc_target_policy: str
    benchmark_min_overlap_ratio: float


_DEFAULT_SCORING_POLICY = ResolvedScoringPolicy(
    fnc_feature_set="fncv3_features",
    fnc_target_policy="scoring_target",
    benchmark_min_overlap_ratio=0.0,
)


def default_scoring_policy() -> ResolvedScoringPolicy:
    """Return the canonical scoring policy for training metrics."""
    return _DEFAULT_SCORING_POLICY


@dataclass(frozen=True)
class PostTrainingScoringResult:
    """Computed summaries, provenance, and artifact payloads from one scoring run."""

    summaries: dict[str, pd.DataFrame]
    score_provenance: dict[str, object]
    policy: ResolvedScoringPolicy
    artifacts: ScoringArtifactBundle
    requested_stage: CanonicalScoringStage = "all"
    refreshed_stages: tuple[str, ...] = ()


RunScoringRequest = PostTrainingScoringRequest
RunScoringResult = PostTrainingScoringResult
