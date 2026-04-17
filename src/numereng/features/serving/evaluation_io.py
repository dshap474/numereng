"""Streaming helpers for package evaluation inputs and pickle-based validation prediction materialization."""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.serving.contracts import SubmissionPackageRecord
from numereng.features.serving.repo import run_model_manifest_path
from numereng.features.serving.runtime import ServingUnsupportedConfigError
from numereng.features.serving.service import build_submission_pickle
from numereng.features.training.errors import TrainingDataError
from numereng.features.training.repo import (
    resolve_data_version_root,
    resolve_variant_dataset_filename,
)


def resolve_validation_dataset_path(*, data_root: Path, data_version: str, dataset: str = "validation") -> Path:
    """Resolve one local validation dataset parquet path."""
    version_root = resolve_data_version_root(data_root=data_root, data_version=data_version)
    filename = resolve_variant_dataset_filename(dataset_variant="non_downsampled", filename=f"{dataset}.parquet")
    return (version_root / filename).resolve()


def ensure_validation_dataset_path(
    *,
    client: Any,
    data_root: Path,
    data_version: str,
    dataset: str = "validation",
) -> Path:
    """Download one validation parquet on demand and return its resolved path."""

    dataset_path = resolve_validation_dataset_path(data_root=data_root, data_version=data_version, dataset=dataset)
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_dataset(f"{data_version}/{dataset}.parquet", dest_path=str(dataset_path))
    return dataset_path


def materialize_pickle_validation_predictions(
    *,
    workspace_root: str | Path,
    experiment_id: str,
    package_id: str,
    package: SubmissionPackageRecord,
    validation_path: Path,
    client: Any,
    scoring_target_cols: tuple[str, ...],
) -> tuple[SubmissionPackageRecord, pd.DataFrame]:
    """Run one built submission pickle across validation row groups without loading the full dataset."""

    built = build_submission_pickle(
        workspace_root=workspace_root,
        experiment_id=experiment_id,
        package_id=package_id,
        client=client,
    )
    predictor = pd.read_pickle(built.pickle_path)
    feature_cols = _pickle_feature_columns(
        predictor=predictor,
        workspace_root=workspace_root,
        package=package,
    )
    requested_columns = ["id", "era", "data_type", *scoring_target_cols, *feature_cols]

    outputs: list[pd.DataFrame] = []
    for chunk in _iter_validation_row_groups(validation_path=validation_path, columns=requested_columns):
        live = _filter_validation_rows(chunk)
        if live.empty:
            continue
        submission = predictor(live[["id", "era", *feature_cols]].copy(), None)
        if "prediction" not in submission.columns:
            raise TrainingDataError("serving_package_pickle_predictions_invalid")
        batch = live[["id", "era"]].copy()
        batch["prediction"] = submission["prediction"].to_numpy(dtype=float)
        for target_col in scoring_target_cols:
            if target_col in live.columns:
                batch[target_col] = live[target_col].to_numpy()
        outputs.append(batch)

    if not outputs:
        raise TrainingDataError("training_validation_rows_missing")
    return built.package, pd.concat(outputs, ignore_index=True)


def _pickle_feature_columns(
    *,
    predictor: Any,
    workspace_root: str | Path,
    package: SubmissionPackageRecord,
) -> list[str]:
    components = getattr(predictor, "_components", None)
    if isinstance(components, list) and components:
        feature_cols = sorted(
            {
                str(col)
                for item in components
                if isinstance(item, dict)
                for col in item.get("feature_cols", ())
            }
        )
        if feature_cols:
            return feature_cols

    feature_cols: set[str] = set()
    for component in package.components:
        if component.run_id is None:
            raise ServingUnsupportedConfigError("serving_package_pickle_feature_columns_unavailable")
        manifest_path = run_model_manifest_path(workspace_root=workspace_root, run_id=component.run_id)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_feature_cols = payload.get("feature_cols")
        if not isinstance(raw_feature_cols, list) or not raw_feature_cols:
            raise ServingUnsupportedConfigError("serving_package_pickle_feature_columns_unavailable")
        feature_cols.update(str(col) for col in raw_feature_cols)

    if not feature_cols:
        raise ServingUnsupportedConfigError("serving_package_pickle_feature_columns_unavailable")
    return sorted(feature_cols)


def _iter_validation_row_groups(*, validation_path: Path, columns: list[str]) -> Iterator[pd.DataFrame]:
    try:
        pq = import_module("pyarrow.parquet")
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise TrainingDataError("training_data_parquet_engine_missing") from exc

    try:
        parquet = pq.ParquetFile(validation_path)
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_read_failed:{validation_path}") from exc

    schema = parquet.schema_arrow
    available_columns = {str(name) for name in schema.names}
    projected = [column for column in columns if column in available_columns]
    for row_group_idx in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group_idx, columns=projected or None)
        frame = table.to_pandas()
        yield _ensure_validation_identity_columns(frame)


def _ensure_validation_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "id" not in normalized.columns:
        if normalized.index.name == "id":
            normalized = normalized.reset_index()
        else:
            raise TrainingDataError("training_data_id_col_missing")
    if "era" not in normalized.columns:
        normalized["era"] = "validation"
    return normalized


def _filter_validation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "data_type" not in frame.columns:
        return frame.reset_index(drop=True)
    filtered = frame[frame["data_type"].astype(str) == "validation"].copy()
    return filtered.reset_index(drop=True)
