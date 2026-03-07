"""Data structures for modular post-training scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PostTrainingScoringRequest:
    """Inputs required to compute post-training scoring artifacts."""

    predictions_path: Path
    pred_cols: tuple[str, ...]
    target_col: str
    data_version: str
    dataset_variant: str
    feature_set: str
    feature_source_paths: tuple[Path, ...] | None
    full_data_path: str | Path | None
    dataset_scope: str
    benchmark_model: str
    benchmark_data_path: str | Path | None
    meta_model_col: str
    meta_model_data_path: str | Path | None
    era_col: str
    id_col: str
    data_root: Path
    scoring_mode: str
    era_chunk_size: int


@dataclass(frozen=True)
class ResolvedScoringPolicy:
    """Explicit scoring semantics resolved for a post-training run."""

    fnc_feature_set: str
    benchmark_overlap_policy: str
    meta_overlap_policy: str


_DEFAULT_SCORING_POLICY = ResolvedScoringPolicy(
    fnc_feature_set="fncv3_features",
    benchmark_overlap_policy="overlap_required",
    meta_overlap_policy="overlap_required",
)


def default_scoring_policy() -> ResolvedScoringPolicy:
    """Return the canonical scoring policy for training metrics."""
    return _DEFAULT_SCORING_POLICY


@dataclass(frozen=True)
class PostTrainingScoringResult:
    """Computed summaries and provenance from one scoring run."""

    summaries: dict[str, pd.DataFrame]
    score_provenance: dict[str, object]
    effective_scoring_backend: str
    policy: ResolvedScoringPolicy
