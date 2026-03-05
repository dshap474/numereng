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
    feature_cols: tuple[str, ...] | None
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
class PostTrainingScoringResult:
    """Computed summaries and provenance from one scoring run."""

    summaries: dict[str, pd.DataFrame]
    score_provenance: dict[str, object]
    effective_scoring_backend: str
