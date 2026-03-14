"""Helpers for persisted scoring-side parquet/json artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from numereng.features.scoring.models import ScoringArtifactBundle

SCORING_SERIES_FILENAMES: dict[str, str] = {
    "corr_per_era": "corr_per_era.parquet",
    "corr_cumulative": "corr_cumulative.parquet",
    "corr_ender20_per_era": "corr_ender20_per_era.parquet",
    "corr_ender20_cumulative": "corr_ender20_cumulative.parquet",
    "bmc_per_era": "bmc_per_era.parquet",
    "bmc_cumulative": "bmc_cumulative.parquet",
    "bmc_ender20_per_era": "bmc_ender20_per_era.parquet",
    "bmc_ender20_cumulative": "bmc_ender20_cumulative.parquet",
    "corr_with_benchmark_per_era": "corr_with_benchmark_per_era.parquet",
    "corr_with_benchmark_cumulative": "corr_with_benchmark_cumulative.parquet",
    "baseline_corr_per_era": "baseline_corr_per_era.parquet",
    "baseline_corr_cumulative": "baseline_corr_cumulative.parquet",
    "baseline_corr_ender20_per_era": "baseline_corr_ender20_per_era.parquet",
    "baseline_corr_ender20_cumulative": "baseline_corr_ender20_cumulative.parquet",
    "corr_delta_vs_baseline_per_era": "corr_delta_vs_baseline_per_era.parquet",
    "corr_delta_vs_baseline_cumulative": "corr_delta_vs_baseline_cumulative.parquet",
    "corr_delta_vs_baseline_ender20_per_era": "corr_delta_vs_baseline_ender20_per_era.parquet",
    "corr_delta_vs_baseline_ender20_cumulative": "corr_delta_vs_baseline_ender20_cumulative.parquet",
}
SCORING_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class PersistedScoringArtifacts:
    """Absolute and relative locations for one persisted scoring artifact bundle."""

    scoring_dir: Path
    manifest_path: Path
    manifest_relative: Path
    series_paths: dict[str, Path]
    series_relatives: dict[str, Path]


def persist_scoring_artifacts(
    bundle: ScoringArtifactBundle,
    *,
    scoring_dir: Path,
    output_dir: Path,
) -> PersistedScoringArtifacts:
    """Persist canonical scoring artifact files and return their locations."""

    scoring_dir.mkdir(parents=True, exist_ok=True)
    series_paths: dict[str, Path] = {}
    series_relatives: dict[str, Path] = {}
    for key, filename in SCORING_SERIES_FILENAMES.items():
        frame = bundle.series_frames.get(key)
        if frame is None:
            continue
        _validate_scoring_frame(key, frame)
        path = scoring_dir / filename
        frame.to_parquet(path, index=False)
        series_paths[key] = path
        series_relatives[key] = path.relative_to(output_dir)

    manifest_path = scoring_dir / SCORING_MANIFEST_FILENAME
    manifest_payload = dict(bundle.manifest)
    manifest_payload["files"] = {
        key: filename
        for key, filename in SCORING_SERIES_FILENAMES.items()
        if key in series_paths
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    return PersistedScoringArtifacts(
        scoring_dir=scoring_dir,
        manifest_path=manifest_path,
        manifest_relative=manifest_path.relative_to(output_dir),
        series_paths=series_paths,
        series_relatives=series_relatives,
    )


def _validate_scoring_frame(key: str, frame: pd.DataFrame) -> None:
    if list(frame.columns) != ["era", "value"]:
        raise ValueError(f"scoring_artifact_frame_invalid_columns:{key}")


__all__ = [
    "PersistedScoringArtifacts",
    "SCORING_MANIFEST_FILENAME",
    "SCORING_SERIES_FILENAMES",
    "persist_scoring_artifacts",
]
