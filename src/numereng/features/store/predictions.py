"""Shared predicates for persisted run prediction artifacts.

USAGE:
    from numereng.features.store import classify_run_mode, run_has_persisted_predictions
    mode = classify_run_mode(run_dir=store_root / "runs" / run_id)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

RunMode = Literal["missing", "incomplete", "scoring", "full"]

# Root files a FINISHED run must carry before its scoring/full artifacts are trusted.
_FINISHED_RUN_REQUIRED_FILES: tuple[str, ...] = (
    "run.json",
    "resolved.json",
    "results.json",
    "metrics.json",
)
_PREDICTIONS_DIR_PARTS: tuple[str, str] = ("artifacts", "predictions")


def classify_run_mode(*, run_dir: Path) -> RunMode:
    """Classify the on-disk materialization state of one local run directory.

    - ``missing``: the run directory does not exist.
    - ``incomplete``: the directory exists but a required root artifact is absent.
    - ``scoring``: required artifacts exist but no prediction parquet is present.
    - ``full``: required artifacts plus at least one prediction parquet exist.

    NOTE: a ``full`` classification means predictions are on disk; it does NOT
    imply the run is OOF (full_history_refit deployment runs also persist
    predictions). OOF-vs-FHR must be read from the run's training profile.
    """

    if not run_dir.exists() or not run_dir.is_dir():
        return "missing"
    for relpath in _FINISHED_RUN_REQUIRED_FILES:
        if not (run_dir / relpath).is_file():
            return "incomplete"
    predictions_dir = run_dir / _PREDICTIONS_DIR_PARTS[0] / _PREDICTIONS_DIR_PARTS[1]
    has_predictions = predictions_dir.is_dir() and any(predictions_dir.glob("pred_*.parquet"))
    return "full" if has_predictions else "scoring"


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


__all__ = ["RunMode", "classify_run_mode", "run_has_persisted_predictions"]
