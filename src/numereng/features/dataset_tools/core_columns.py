"""Shared column helpers for dataset tool scripts."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_ERA_COL = "era"
DEFAULT_ID_COL = "id"
DEFAULT_TARGET_COL = "target"
DEFAULT_FEATURE_PREFIX = "feature_"


def is_feature_column(name: str, *, feature_prefix: str = DEFAULT_FEATURE_PREFIX) -> bool:
    """Return True when *name* matches the configured feature prefix."""
    return name.startswith(feature_prefix)


def is_target_column(name: str, *, target_col: str = DEFAULT_TARGET_COL) -> bool:
    """Return True when *name* is the primary or auxiliary target column."""
    return name == target_col or name.startswith("target_")


def parse_csv_columns(raw: str | None) -> list[str]:
    """Parse comma-separated columns into a cleaned list."""
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def require_columns(frame: pd.DataFrame, *, required: Sequence[str], context: str) -> None:
    """Raise a ValueError when required columns are missing."""
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{context}_missing_columns:{','.join(missing)}")


def resolve_feature_columns(
    frame: pd.DataFrame,
    *,
    features_csv: str | None = None,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
) -> list[str]:
    """Resolve feature column names from explicit CSV or prefix scanning."""
    explicit = parse_csv_columns(features_csv)
    if explicit:
        require_columns(frame, required=explicit, context="feature")
        return explicit
    return [col for col in frame.columns if is_feature_column(col, feature_prefix=feature_prefix)]


def resolve_target_columns(
    frame: pd.DataFrame,
    *,
    targets_csv: str | None = None,
    target_col: str = DEFAULT_TARGET_COL,
) -> list[str]:
    """Resolve target column names from explicit CSV or default pattern scanning."""
    explicit = parse_csv_columns(targets_csv)
    if explicit:
        require_columns(frame, required=explicit, context="target")
        return explicit
    return [col for col in frame.columns if is_target_column(col, target_col=target_col)]
