"""Helpers for persisted scoring-side parquet/json artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from numereng.features.scoring.models import CanonicalScoringStage, ScoringArtifactBundle
from numereng.platform.parquet import write_parquet

SCORING_CHART_FILENAMES: dict[str, str] = {
    "run_metric_series": "run_metric_series.parquet",
}
SCORING_STAGE_FILENAMES: dict[str, str] = {
    "post_fold_per_era": "post_fold_per_era.parquet",
    "post_fold_snapshots": "post_fold_snapshots.parquet",
    "post_training_core_summary": "post_training_core_summary.parquet",
    "post_training_full_summary": "post_training_full_summary.parquet",
}
LEGACY_SCORING_STAGE_FILENAMES: dict[str, str] = {
    "post_training_core_summary": "post_training_summary.parquet",
    "post_training_full_summary": "post_training_features_summary.parquet",
}
SCORING_MANIFEST_FILENAME = "manifest.json"
CANONICAL_SCORING_STAGE_FILES: dict[str, tuple[str, ...]] = {
    "run_metric_series": ("run_metric_series",),
    "post_fold": ("post_fold_per_era", "post_fold_snapshots"),
    "post_training_core": ("post_training_core_summary",),
    "post_training_full": ("post_training_core_summary", "post_training_full_summary"),
}

_STAGE_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "run_metric_series": [
        "run_id",
        "config_hash",
        "seed",
        "target_col",
        "payout_target_col",
        "prediction_col",
        "era",
        "metric_key",
        "series_type",
        "value",
    ],
    "post_fold_per_era": [
        "run_id",
        "config_hash",
        "seed",
        "target_col",
        "payout_target_col",
        "cv_fold",
        "era",
        "corr_native",
    ],
    "post_fold_snapshots": [
        "run_id",
        "config_hash",
        "seed",
        "target_col",
        "payout_target_col",
        "cv_fold",
        "corr_native_fold_mean",
    ],
    "post_training_core_summary": [
        "run_id",
        "config_hash",
        "seed",
        "target_col",
        "payout_target_col",
        "prediction_col",
        "corr_native_mean",
        "corr_native_std",
        "corr_native_sharpe",
        "corr_native_max_drawdown",
    ],
    "post_training_full_summary": [
        "run_id",
        "config_hash",
        "seed",
        "target_col",
        "payout_target_col",
        "prediction_col",
        "feature_exposure_mean",
        "feature_exposure_std",
        "feature_exposure_sharpe",
        "feature_exposure_max_drawdown",
        "max_feature_exposure_mean",
        "max_feature_exposure_std",
        "max_feature_exposure_sharpe",
        "max_feature_exposure_max_drawdown",
    ],
}


@dataclass(frozen=True)
class PersistedScoringArtifacts:
    """Absolute and relative locations for one persisted scoring artifact bundle."""

    scoring_dir: Path
    manifest_path: Path
    manifest_relative: Path
    series_paths: dict[str, Path]
    series_relatives: dict[str, Path]
    chart_paths: dict[str, Path]
    chart_relatives: dict[str, Path]
    stage_paths: dict[str, Path]
    stage_relatives: dict[str, Path]


def persist_scoring_artifacts(
    bundle: ScoringArtifactBundle,
    *,
    scoring_dir: Path,
    output_dir: Path,
    selected_stage: CanonicalScoringStage = "all",
) -> PersistedScoringArtifacts:
    """Persist canonical scoring artifact files and return their locations."""

    scoring_dir.mkdir(parents=True, exist_ok=True)
    series_paths: dict[str, Path] = {}
    series_relatives: dict[str, Path] = {}
    selected_file_keys = resolve_selected_stage_file_keys(selected_stage)

    chart_paths: dict[str, Path] = {}
    chart_relatives: dict[str, Path] = {}
    for key, filename in SCORING_CHART_FILENAMES.items():
        if key not in selected_file_keys:
            continue
        frame = bundle.stage_frames.get(key)
        if frame is None:
            continue
        _validate_stage_frame(key, frame)
        path = scoring_dir / filename
        write_parquet(frame, path, index=False)
        chart_paths[key] = path
        chart_relatives[key] = path.relative_to(output_dir)

    stage_paths: dict[str, Path] = {}
    stage_relatives: dict[str, Path] = {}
    for key, filename in SCORING_STAGE_FILENAMES.items():
        if key not in selected_file_keys:
            continue
        frame = bundle.stage_frames.get(key)
        if frame is None:
            continue
        _validate_stage_frame(key, frame)
        path = scoring_dir / filename
        write_parquet(frame, path, index=False)
        stage_paths[key] = path
        stage_relatives[key] = path.relative_to(output_dir)

    manifest_path = scoring_dir / SCORING_MANIFEST_FILENAME
    manifest_payload = dict(bundle.manifest)
    manifest_payload.pop("files", None)
    current_chart_files = {
        key: filename
        for key, filename in SCORING_CHART_FILENAMES.items()
        if (scoring_dir / filename).is_file()
    }
    current_stage_files = {
        key: filename
        for key, filename in SCORING_STAGE_FILENAMES.items()
        if (scoring_dir / filename).is_file()
    }
    manifest_payload["chart_files"] = current_chart_files
    manifest_payload["stage_files"] = current_stage_files
    manifest_payload["current_canonical_stages"] = current_canonical_stages(scoring_dir)
    manifest_payload["requested_stage"] = selected_stage
    manifest_payload["refreshed_canonical_stages"] = refreshed_canonical_stages(
        selected_stage,
        available_stage_frames=bundle.stage_frames,
    )
    manifest_payload["refreshed_stage_files"] = {
        key: filename
        for key, filename in {**SCORING_CHART_FILENAMES, **SCORING_STAGE_FILENAMES}.items()
        if key in selected_file_keys and key in bundle.stage_frames
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    return PersistedScoringArtifacts(
        scoring_dir=scoring_dir,
        manifest_path=manifest_path,
        manifest_relative=manifest_path.relative_to(output_dir),
        series_paths=series_paths,
        series_relatives=series_relatives,
        chart_paths=chart_paths,
        chart_relatives=chart_relatives,
        stage_paths=stage_paths,
        stage_relatives=stage_relatives,
    )


def resolve_selected_stage_file_keys(selected_stage: CanonicalScoringStage) -> set[str]:
    """Resolve persisted file keys for one canonical stage selection."""

    if selected_stage == "all":
        return {*SCORING_CHART_FILENAMES.keys(), *SCORING_STAGE_FILENAMES.keys()}
    return set(CANONICAL_SCORING_STAGE_FILES[selected_stage])


def refreshed_canonical_stages(
    selected_stage: CanonicalScoringStage,
    *,
    available_stage_frames: dict[str, pd.DataFrame],
) -> list[str]:
    """Return canonical stages refreshed by the latest scoring pass."""

    if selected_stage == "all":
        selected_stages = tuple(CANONICAL_SCORING_STAGE_FILES.keys())
    elif selected_stage == "post_training_full":
        selected_stages = ("post_training_core", "post_training_full")
    else:
        selected_stages = (selected_stage,)
    refreshed: list[str] = []
    for canonical_stage in selected_stages:
        required_keys = CANONICAL_SCORING_STAGE_FILES[canonical_stage]
        if any(key in available_stage_frames for key in required_keys):
            refreshed.append(canonical_stage)
    return refreshed


def current_canonical_stages(scoring_dir: Path) -> list[str]:
    """Return canonical stages currently present on disk."""

    current: list[str] = []
    for canonical_stage, file_keys in CANONICAL_SCORING_STAGE_FILES.items():
        if all(_stage_file_exists(scoring_dir, file_key) for file_key in file_keys):
            current.append(canonical_stage)
    return current


def _resolve_stage_filename(file_key: str) -> str:
    if file_key in SCORING_CHART_FILENAMES:
        return SCORING_CHART_FILENAMES[file_key]
    return SCORING_STAGE_FILENAMES[file_key]


def _stage_file_exists(scoring_dir: Path, file_key: str) -> bool:
    candidates = [_resolve_stage_filename(file_key)]
    legacy = LEGACY_SCORING_STAGE_FILENAMES.get(file_key)
    if legacy is not None:
        candidates.append(legacy)
    return any((scoring_dir / candidate).is_file() for candidate in candidates)


def _validate_stage_frame(key: str, frame: pd.DataFrame) -> None:
    required = _STAGE_REQUIRED_COLUMNS[key]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"scoring_stage_artifact_missing_columns:{key}:{','.join(missing)}")

    if key == "run_metric_series":
        series_values = set(frame["series_type"].dropna().astype(str))
        if not series_values.issubset({"per_era", "cumulative"}):
            raise ValueError("scoring_stage_artifact_invalid_series_type:run_metric_series")
    elif key == "post_fold_per_era":
        _ensure_optional_stage_columns_are_complete(
            frame,
            column_group=["corr_ender20", "bmc_ender20"],
            key=key,
        )
    elif key == "post_fold_snapshots":
        _ensure_optional_stage_columns_are_complete(
            frame,
            column_group=["corr_ender20_fold_mean", "bmc_ender20_fold_mean"],
            key=key,
        )
    elif key == "post_training_core_summary":
        for column_group in (
            [
                "corr_ender20_mean",
                "corr_ender20_std",
                "corr_ender20_sharpe",
                "corr_ender20_max_drawdown",
            ],
            [
                "mmc_ender20_mean",
                "mmc_ender20_std",
                "mmc_ender20_sharpe",
                "mmc_ender20_max_drawdown",
            ],
            [
                "bmc_ender20_mean",
                "bmc_ender20_std",
                "bmc_ender20_sharpe",
                "bmc_ender20_max_drawdown",
            ],
        ):
            _ensure_optional_stage_columns_are_complete(
                frame,
                column_group=column_group,
                key=key,
            )
    elif key == "post_training_full_summary":
        for column_group in (
            [
                "fnc_native_mean",
                "fnc_native_std",
                "fnc_native_sharpe",
                "fnc_native_max_drawdown",
            ],
            [
                "fnc_ender20_mean",
                "fnc_ender20_std",
                "fnc_ender20_sharpe",
                "fnc_ender20_max_drawdown",
            ],
        ):
            _ensure_optional_stage_columns_are_complete(
                frame,
                column_group=column_group,
                key=key,
            )


def _ensure_optional_stage_columns_are_complete(
    frame: pd.DataFrame,
    *,
    column_group: list[str],
    key: str,
) -> None:
    present = [col for col in column_group if col in frame.columns]
    if not present:
        return
    missing = [col for col in column_group if col not in frame.columns]
    if missing:
        raise ValueError(f"scoring_stage_artifact_optional_columns_invalid:{key}:{','.join(missing)}")


__all__ = [
    "PersistedScoringArtifacts",
    "CANONICAL_SCORING_STAGE_FILES",
    "LEGACY_SCORING_STAGE_FILENAMES",
    "SCORING_CHART_FILENAMES",
    "SCORING_MANIFEST_FILENAME",
    "SCORING_STAGE_FILENAMES",
    "current_canonical_stages",
    "persist_scoring_artifacts",
    "refreshed_canonical_stages",
    "resolve_selected_stage_file_keys",
]
