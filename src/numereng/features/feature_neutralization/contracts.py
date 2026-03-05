"""Contracts for feature-neutralization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

NeutralizationMode = Literal["era", "global"]


@dataclass(frozen=True)
class NeutralizePredictionsRequest:
    """Input payload for neutralizing one predictions file."""

    predictions_path: Path
    neutralizer_path: Path
    output_path: Path | None = None
    proportion: float = 0.5
    mode: NeutralizationMode = "era"
    neutralizer_cols: tuple[str, ...] | None = None
    rank_output: bool = True


@dataclass(frozen=True)
class NeutralizationResult:
    """Result payload for one neutralization operation."""

    source_path: Path
    output_path: Path
    run_id: str | None
    neutralizer_path: Path
    neutralizer_cols: tuple[str, ...]
    proportion: float
    mode: NeutralizationMode
    rank_output: bool
    source_rows: int
    neutralizer_rows: int
    matched_rows: int
