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
    skip_downsample: bool = False


@dataclass(frozen=True)
class BuildDownsampledFullResult:
    """Result payload for one downsampled full dataset build."""

    data_dir: Path
    data_version: str
    full_path: Path
    full_benchmark_path: Path
    downsampled_full_path: Path | None
    downsampled_full_benchmark_path: Path | None
    full_rows: int
    downsampled_rows: int | None
    full_benchmark_rows: int
    downsampled_full_benchmark_rows: int | None
    total_eras: int | None
    kept_eras: int | None
    downsample_step: int
    downsample_offset: int
    downsample_built: bool
