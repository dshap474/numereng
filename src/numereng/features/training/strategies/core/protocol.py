"""Unified training engine protocol types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TrainingProfile = Literal["simple", "purged_walk_forward", "full_history_refit"]
# Backward-compatible legacy engine mode type for API/config migration.
TrainingEngineMode = Literal["official", "custom", "full_history"]


@dataclass(frozen=True)
class TrainingEnginePlan:
    """Resolved training profile plan consumed by the training service."""

    mode: TrainingProfile
    cv_config: dict[str, object]
    resolved_config: dict[str, object] = field(default_factory=dict)
    override_sources: list[str] = field(default_factory=list)
