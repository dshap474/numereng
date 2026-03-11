"""Data structures for training feature workflows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from numereng.features.training.repo import list_lazy_source_eras, load_fold_data_lazy
from numereng.features.scoring.metrics import attach_benchmark_predictions

_KNOWN_X_GROUPS = {
    "features",
    "baseline",
}
_DEFAULT_X_GROUPS = ("features",)


@dataclass(frozen=True)
class ModelDataBatch:
    """Typed training batch for one train/validation fold."""

    X: pd.DataFrame
    y: pd.Series
    era: pd.Series
    id: pd.Series | None


@dataclass(frozen=True)
class ModelDataLoader:
    """Era-based batch loader over a single prepared full dataset."""

    full: pd.DataFrame
    era_col: str
    target_col: str
    id_col: str | None
    x_cols: tuple[str, ...]

    def load(self, eras: Sequence[object]) -> ModelDataBatch:
        """Load one batch for a set of eras."""
        subset = self.full[self.full[self.era_col].isin(eras)]
        X = subset[list(self.x_cols)]
        y = subset[self.target_col]
        era = subset[self.era_col]
        ids = _subset_id_series(subset, self.id_col)
        return ModelDataBatch(X=X, y=y, era=era, id=ids)

    def list_eras(self) -> list[object]:
        """Return sorted unique eras available to this provider."""
        return sorted(set(self.full[self.era_col].tolist()), key=_era_sort_key)


@dataclass(frozen=True)
class LazyParquetFoldDataProvider:
    """Fold-scoped parquet reader with era/column pushdown."""

    source_paths: tuple[Path, ...]
    era_col: str
    target_col: str
    id_col: str | None
    x_cols: tuple[str, ...]
    include_validation_only: bool = True
    join_columns: tuple[FoldJoinColumn, ...] = ()

    def load(self, eras: Sequence[object]) -> ModelDataBatch:
        """Load one fold batch directly from parquet sources."""
        columns = _dedupe_columns([self.era_col, self.target_col, *self.x_cols])
        if self.id_col:
            columns.append(self.id_col)
        frame = load_fold_data_lazy(
            self.source_paths,
            eras=[*eras],
            columns=columns,
            era_col=self.era_col,
            id_col=self.id_col,
            include_validation_only=self.include_validation_only,
        )
        if self.join_columns:
            if not self.id_col:
                raise ValueError("id_col is required for lazy join columns")
            for join_column in self.join_columns:
                frame = attach_benchmark_predictions(
                    frame,
                    join_column.frame,
                    join_column.column,
                    era_col=self.era_col,
                    id_col=self.id_col,
                )
        X = frame[list(self.x_cols)]
        y = frame[self.target_col]
        era = frame[self.era_col]
        ids = frame[self.id_col] if self.id_col else None
        return ModelDataBatch(X=X, y=y, era=era, id=ids)

    def list_eras(self) -> list[object]:
        """Return sorted unique eras from lazy parquet sources."""
        return list_lazy_source_eras(
            self.source_paths,
            era_col=self.era_col,
            include_validation_only=self.include_validation_only,
        )


@runtime_checkable
class ModelDataLoaderProtocol(Protocol):
    """Protocol for loader objects used by CV routines."""

    def load(self, eras: Sequence[object]) -> ModelDataBatch:
        """Load one train/validation batch for a list of eras."""

    def list_eras(self) -> list[object]:
        """Return sorted unique eras available to this loader."""


@dataclass(frozen=True)
class TrainingRunResult:
    """Top-level training run outputs."""

    run_id: str
    predictions_path: Path
    results_path: Path


@dataclass(frozen=True)
class ScoreRunResult:
    """Top-level score-only run outputs."""

    run_id: str
    predictions_path: Path
    results_path: Path
    metrics_path: Path
    score_provenance_path: Path
    effective_scoring_backend: str


@dataclass(frozen=True)
class FoldJoinColumn:
    """Fold-time join source attached by `(id, era)`."""

    frame: pd.DataFrame
    column: str


def build_model_data_loader(
    *,
    full: pd.DataFrame,
    x_cols: Iterable[str],
    era_col: str,
    target_col: str,
    id_col: str | None,
) -> ModelDataLoader:
    """Create validated data loader for CV batches."""
    resolved_x_cols = tuple(x_cols)
    if not resolved_x_cols:
        raise ValueError("x_cols must be a non-empty list")
    return ModelDataLoader(
        full=full,
        era_col=era_col,
        target_col=target_col,
        id_col=id_col,
        x_cols=resolved_x_cols,
    )


def build_lazy_parquet_data_loader(
    *,
    source_paths: Sequence[Path],
    x_cols: Iterable[str],
    era_col: str,
    target_col: str,
    id_col: str | None,
    include_validation_only: bool = True,
    join_columns: Sequence[FoldJoinColumn] | None = None,
) -> LazyParquetFoldDataProvider:
    """Create fold-lazy parquet loader for era-scoped reads."""
    resolved_x_cols = tuple(x_cols)
    if not resolved_x_cols:
        raise ValueError("x_cols must be a non-empty list")
    if not source_paths:
        raise ValueError("source_paths must be non-empty")
    return LazyParquetFoldDataProvider(
        source_paths=tuple(source_paths),
        era_col=era_col,
        target_col=target_col,
        id_col=id_col,
        x_cols=resolved_x_cols,
        include_validation_only=include_validation_only,
        join_columns=tuple(join_columns or []),
    )


def normalize_x_groups(x_groups: Iterable[str] | None) -> list[str]:
    """Normalize requested model x groups with defaults."""
    if not x_groups:
        x_groups = list(_DEFAULT_X_GROUPS)

    normalized: list[str] = []
    for key in x_groups:
        resolved = key
        if resolved in {"target", "y"}:
            continue
        if resolved in {"era", "id"}:
            raise ValueError(f"training_model_x_groups_not_supported:{resolved}")
        # Benchmark model predictions are metrics/ensemble inputs, not model
        # training features.
        if resolved in {"benchmark", "benchmarks", "benchmark_models"}:
            raise ValueError("training_model_x_groups_benchmark_not_supported")

        if resolved not in _KNOWN_X_GROUPS:
            raise ValueError(f"Unknown x_group '{resolved}'. Supported keys: {sorted(_KNOWN_X_GROUPS)}")

        if resolved not in normalized:
            normalized.append(resolved)

    for required in _DEFAULT_X_GROUPS:
        if required not in normalized:
            normalized.append(required)

    return normalized


def build_x_cols(
    *,
    x_groups: Sequence[str],
    features: Sequence[str],
    era_col: str,
    id_col: str | None,
    baseline_col: str | None = None,
) -> list[str]:
    """Expand x groups into explicit ordered feature columns."""
    x_cols: list[str] = []

    for key in x_groups:
        if key == "features":
            x_cols.extend(features)
        elif key == "baseline":
            if not baseline_col:
                raise ValueError("baseline requested but no baseline column provided")
            x_cols.append(baseline_col)
        else:
            raise ValueError(f"Unknown x_group '{key}'")

    seen: set[str] = set()
    ordered: list[str] = []
    for col in x_cols:
        if col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    return ordered


def _era_sort_key(era: object) -> int | str:
    if isinstance(era, str) and era.isdigit():
        return int(era)
    if isinstance(era, int):
        return era
    return str(era)


def _subset_id_series(subset: pd.DataFrame, id_col: str | None) -> pd.Series | None:
    if not id_col:
        return None
    if id_col in subset.columns:
        return subset[id_col]
    if subset.index.name == id_col:
        return subset.index.to_series(index=subset.index, name=id_col)
    return None


def _dedupe_columns(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    return ordered
