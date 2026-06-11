"""Shared predicates for persisted run prediction artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast


def run_has_persisted_predictions(
    *,
    root: Path,
    run_id: str,
    run_manifest: Mapping[str, object],
) -> bool:
    """Return whether a run has the prediction artifact shape used by round scoring."""

    run_dir = root / "runs" / run_id
    artifacts = run_manifest.get("artifacts")
    predictions_rel = None
    if isinstance(artifacts, dict):
        predictions_rel = _as_str(cast(dict[str, object], artifacts).get("predictions"))
    if predictions_rel is not None:
        return (run_dir / predictions_rel).is_file()

    predictions_dir = run_dir / "artifacts" / "predictions"
    return len(tuple(predictions_dir.glob("*.parquet"))) == 1


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["run_has_persisted_predictions"]
