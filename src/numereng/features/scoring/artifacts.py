"""Helpers for persisted scoring-side visualization artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_PER_ERA_CORR_PARQUET = "val_per_era_corr20v2.parquet"
_PER_ERA_CORR_CSV = "val_per_era_corr20v2.csv"


@dataclass(frozen=True)
class PerEraCorrArtifactPaths:
    """Absolute artifact paths for the canonical per-era correlation series."""

    parquet_path: Path
    csv_path: Path


def build_primary_per_era_corr_frame(
    *,
    predictions_path: str | Path,
    target_col: str,
    era_col: str = "era",
    prediction_col: str = "prediction",
) -> pd.DataFrame | None:
    """Build the canonical `{era, corr}` frame for the native run target."""

    from numereng.features.scoring.metrics import _prepare_predictions_for_scoring, _read_table, per_era_corr

    predictions = _prepare_predictions_for_scoring(
        _read_table(Path(predictions_path)),
        pred_cols=[prediction_col],
        target_cols=[target_col],
        era_col=era_col,
    )
    per_era = per_era_corr(predictions, [prediction_col], target_col, era_col)
    if prediction_col not in per_era.columns:
        return None
    frame = per_era[[prediction_col]].rename(columns={prediction_col: "corr"}).reset_index()
    if len(frame.columns) >= 1:
        frame = frame.rename(columns={frame.columns[0]: "era"})
    return frame[["era", "corr"]]


def persist_primary_per_era_corr_artifacts(
    frame: pd.DataFrame,
    *,
    predictions_dir: Path,
) -> PerEraCorrArtifactPaths:
    """Persist the canonical per-era correlation frame to parquet and csv."""

    predictions_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = predictions_dir / _PER_ERA_CORR_PARQUET
    csv_path = predictions_dir / _PER_ERA_CORR_CSV
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    return PerEraCorrArtifactPaths(
        parquet_path=parquet_path,
        csv_path=csv_path,
    )


__all__ = [
    "PerEraCorrArtifactPaths",
    "build_primary_per_era_corr_frame",
    "persist_primary_per_era_corr_artifacts",
]
