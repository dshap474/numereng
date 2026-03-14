"""Data structures for modular post-training scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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


@dataclass(frozen=True)
class PostTrainingScoringRequest:
    """Inputs required to compute post-training scoring artifacts."""

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
    scoring_mode: str
    era_chunk_size: int
    include_feature_neutral_metrics: bool = True


@dataclass(frozen=True)
class ResolvedScoringPolicy:
    """Explicit scoring semantics resolved for a post-training run."""

    fnc_feature_set: str
    fnc_target_policy: str
    benchmark_min_overlap_ratio: float
    include_feature_neutral_metrics: bool


_DEFAULT_SCORING_POLICY = ResolvedScoringPolicy(
    fnc_feature_set="fncv3_features",
    fnc_target_policy="scoring_target",
    benchmark_min_overlap_ratio=0.0,
    include_feature_neutral_metrics=True,
)


def default_scoring_policy(include_feature_neutral_metrics: bool | None = None) -> ResolvedScoringPolicy:
    """Return the canonical scoring policy for training metrics."""
    if include_feature_neutral_metrics is None:
        return _DEFAULT_SCORING_POLICY
    return ResolvedScoringPolicy(
        fnc_feature_set=_DEFAULT_SCORING_POLICY.fnc_feature_set,
        fnc_target_policy=_DEFAULT_SCORING_POLICY.fnc_target_policy,
        benchmark_min_overlap_ratio=_DEFAULT_SCORING_POLICY.benchmark_min_overlap_ratio,
        include_feature_neutral_metrics=include_feature_neutral_metrics,
    )


@dataclass(frozen=True)
class PostTrainingScoringResult:
    """Computed summaries, provenance, and artifact payloads from one scoring run."""

    summaries: dict[str, pd.DataFrame]
    score_provenance: dict[str, object]
    effective_scoring_backend: str
    policy: ResolvedScoringPolicy
    artifacts: ScoringArtifactBundle
