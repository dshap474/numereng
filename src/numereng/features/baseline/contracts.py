"""Contracts for baseline build workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineBuildRequest:
    """Input payload for building one named baseline from existing run IDs."""

    run_ids: tuple[str, ...]
    name: str
    default_target: str = "target_ender_20"
    description: str | None = None
    promote_active: bool = False


@dataclass(frozen=True)
class BaselineBuildResult:
    """Result payload for one persisted baseline build."""

    name: str
    baseline_dir: Path
    predictions_path: Path
    metadata_path: Path
    available_targets: tuple[str, ...]
    default_target: str
    source_run_ids: tuple[str, ...]
    source_experiment_id: str | None
    active_predictions_path: Path | None
    active_metadata_path: Path | None
    created_at: str
