"""Contracts for dataset-tools downsampling workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildDownsampledFullRequest:
    """Input payload for official-style downsampled full dataset build."""

    data_dir: Path
    data_version: str = "v5.2"
    rebuild: bool = False
    downsample_eras_step: int = 4
    downsample_eras_offset: int = 0


@dataclass(frozen=True)
class BuildDownsampledFullResult:
    """Result payload for one downsampled full dataset build."""

    data_dir: Path
    data_version: str
    downsampled_full_path: Path
    downsampled_full_benchmark_path: Path
    downsampled_rows: int
    downsampled_full_benchmark_rows: int
    total_eras: int
    kept_eras: int
    downsample_step: int
    downsample_offset: int
