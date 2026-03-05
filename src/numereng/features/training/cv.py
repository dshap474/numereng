"""Era-based CV helpers and OOF prediction builder."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from importlib import import_module

import numpy as np
import pandas as pd

from numereng.features.training.errors import TrainingConfigError
from numereng.features.training.model_factory import build_model
from numereng.features.training.models import ModelDataBatch, ModelDataLoaderProtocol


def _era_sort_key(era: object) -> int | str:
    if isinstance(era, (int, np.integer)):
        return int(era)
    if isinstance(era, str):
        if era.isdigit():
            return int(era)
        return era
    return str(era)


def _sorted_unique_eras(eras: Iterable[object]) -> list[object]:
    return sorted(set(eras), key=_era_sort_key)


def era_cv_splits(
    eras: Iterable[object],
    embargo: int = 13,
    mode: str = "official_walkforward",
    min_train_size: int = 1,
    chunk_size: int = 156,
    holdout_train_eras: Iterable[object] | None = None,
    holdout_val_eras: Iterable[object] | None = None,
) -> list[tuple[list[object], list[object]]]:
    """Create profile-specific era splits with optional embargo."""
    if embargo < 0:
        raise TrainingConfigError("training_cv_embargo_invalid")

    eras_sorted = _sorted_unique_eras(eras)
    if mode == "official_walkforward":
        return _official_walkforward_splits(
            eras_sorted,
            embargo=embargo,
            chunk_size=chunk_size,
            min_train_size=min_train_size,
        )

    if mode == "train_validation_holdout":
        train_eras = _sorted_unique_eras(holdout_train_eras or [])
        val_eras = _sorted_unique_eras(holdout_val_eras or [])
        if not train_eras or not val_eras:
            raise TrainingConfigError("training_cv_holdout_eras_missing")
        overlap = set(train_eras).intersection(val_eras)
        if overlap:
            raise TrainingConfigError("training_cv_holdout_eras_overlap")
        if len(train_eras) < min_train_size:
            raise TrainingConfigError("training_cv_train_split_too_small")
        return [(train_eras, val_eras)]

    raise TrainingConfigError("training_cv_mode_invalid")


def _official_walkforward_splits(
    eras_sorted: list[object],
    *,
    embargo: int,
    chunk_size: int,
    min_train_size: int,
) -> list[tuple[list[object], list[object]]]:
    if chunk_size < 1:
        raise TrainingConfigError("training_cv_chunk_size_invalid")
    if min_train_size < 1:
        raise TrainingConfigError("training_cv_min_train_size_invalid")
    if len(eras_sorted) <= chunk_size:
        required = chunk_size + 1
        found = len(eras_sorted)
        raise TrainingConfigError(
            f"training_cv_walkforward_requires_min_eras:required={required}:found={found}"
        )

    splits: list[tuple[list[object], list[object]]] = []
    val_start = chunk_size

    while val_start < len(eras_sorted):
        val_end = min(val_start + chunk_size, len(eras_sorted))
        val_eras = eras_sorted[val_start:val_end]
        if not val_eras:
            break

        train_end = max(0, val_start - embargo)
        train_eras = eras_sorted[:train_end]
        if len(train_eras) >= min_train_size:
            splits.append((train_eras, val_eras))

        val_start += chunk_size

    return splits


def build_oof_predictions(
    eras: Iterable[object],
    data_loader: ModelDataLoaderProtocol | Callable[[Sequence[object]], ModelDataBatch],
    model_type: str,
    model_params: dict[str, object],
    model_config: dict[str, object],
    cv_config: dict[str, object],
    id_col: str | None,
    era_col: str,
    target_col: str,
    feature_cols: list[str] | None = None,
    parallel_folds: int = 1,
    parallel_backend: str = "joblib",
    memmap_enabled: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build out-of-fold predictions and CV metadata."""
    cv_mode = str(cv_config.get("mode", "official_walkforward"))
    cv_embargo = _coerce_int(cv_config.get("embargo"), default=13)
    cv_min_train_size = _coerce_int(cv_config.get("min_train_size"), default=0)
    cv_chunk_size = _coerce_int(cv_config.get("chunk_size"), default=156)
    holdout_train_eras = _coerce_eras_list(cv_config.get("train_eras"), field="training.cv.train_eras")
    holdout_val_eras = _coerce_eras_list(cv_config.get("val_eras"), field="training.cv.val_eras")

    splits = era_cv_splits(
        eras,
        embargo=cv_embargo,
        mode=cv_mode,
        min_train_size=cv_min_train_size,
        chunk_size=cv_chunk_size,
        holdout_train_eras=holdout_train_eras,
        holdout_val_eras=holdout_val_eras,
    )
    splits_with_descriptors: list[tuple[list[object], list[object], dict[str, object]]] = [
        (train_eras, val_eras, {})
        for train_eras, val_eras in splits
    ]

    all_eras_sorted = _sorted_unique_eras(eras)

    predictions: list[pd.DataFrame] = []
    fold_info: list[dict[str, object]] = []
    fold_work: list[tuple[int, list[object], list[object], dict[str, object]]] = []
    for fold_idx, (train_eras, val_eras, extra_descriptor) in enumerate(splits_with_descriptors):
        if not train_eras or not val_eras:
            continue

        descriptor = _build_fold_descriptor(
            all_eras_sorted=all_eras_sorted,
            train_eras=train_eras,
            val_eras=val_eras,
            embargo=cv_embargo,
        )
        descriptor.update(extra_descriptor)
        fold_work.append((fold_idx, train_eras, val_eras, descriptor))

    if parallel_folds < 1:
        raise TrainingConfigError("training_cv_parallel_folds_invalid")

    if parallel_folds == 1:
        for fold_idx, train_eras, val_eras, descriptor in fold_work:
            fold_result = _run_fold_prediction(
                fold_idx=fold_idx,
                train_eras=train_eras,
                val_eras=val_eras,
                descriptor=descriptor,
                data_loader=data_loader,
                model_type=model_type,
                model_params=model_params,
                model_config=model_config,
                id_col=id_col,
                era_col=era_col,
                target_col=target_col,
                feature_cols=feature_cols,
            )
            if fold_result is None:
                continue
            fold_prediction_frame, fold_metadata = fold_result
            predictions.append(fold_prediction_frame)
            fold_info.append(fold_metadata)
    else:
        if parallel_backend != "joblib":
            raise TrainingConfigError("training_cv_parallel_backend_invalid")
        try:
            parallel_mod = import_module("joblib")
            Parallel = parallel_mod.Parallel
            delayed = parallel_mod.delayed
        except ImportError as exc:
            raise TrainingConfigError("training_cv_parallel_backend_missing_joblib") from exc

        mmap_mode: str | None = "r" if memmap_enabled else None
        fold_results = Parallel(
            n_jobs=parallel_folds,
            backend="loky",
            mmap_mode=mmap_mode,
        )(
            delayed(_run_fold_prediction)(
                fold_idx=fold_idx,
                train_eras=train_eras,
                val_eras=val_eras,
                descriptor=descriptor,
                data_loader=data_loader,
                model_type=model_type,
                model_params=model_params,
                model_config=model_config,
                id_col=id_col,
                era_col=era_col,
                target_col=target_col,
                feature_cols=feature_cols,
            )
            for fold_idx, train_eras, val_eras, descriptor in fold_work
        )
        for fold_result in fold_results:
            if fold_result is None:
                continue
            fold_prediction_frame, fold_metadata = fold_result
            predictions.append(fold_prediction_frame)
            fold_info.append(fold_metadata)

    if not predictions:
        raise TrainingConfigError("training_cv_no_predictions")

    oof = pd.concat(predictions, ignore_index=True)
    if id_col and id_col in oof.columns and oof[id_col].duplicated().any():
        raise TrainingConfigError("training_cv_duplicate_ids")

    cv_meta: dict[str, object] = {
        "n_splits": len(splits),
        "embargo": cv_embargo,
        "mode": cv_mode,
        "min_train_size": cv_min_train_size,
        "folds_used": len(fold_info),
        "folds": fold_info,
    }
    if cv_mode == "official_walkforward":
        cv_meta["chunk_size"] = cv_chunk_size
    if cv_mode == "train_validation_holdout":
        cv_meta["holdout_train_eras"] = len(_sorted_unique_eras(holdout_train_eras or []))
        cv_meta["holdout_val_eras"] = len(_sorted_unique_eras(holdout_val_eras or []))
    return oof, cv_meta


def _run_fold_prediction(
    *,
    fold_idx: int,
    train_eras: Sequence[object],
    val_eras: Sequence[object],
    descriptor: dict[str, object],
    data_loader: ModelDataLoaderProtocol | Callable[[Sequence[object]], ModelDataBatch],
    model_type: str,
    model_params: dict[str, object],
    model_config: dict[str, object],
    id_col: str | None,
    era_col: str,
    target_col: str,
    feature_cols: list[str] | None,
) -> tuple[pd.DataFrame, dict[str, object]] | None:
    train_data = _load_data(data_loader, train_eras)
    val_data = _load_data(data_loader, val_eras)

    train_rows = _data_length(train_data)
    val_rows = _data_length(val_data)
    if train_rows == 0 or val_rows == 0:
        return None

    model = build_model(
        model_type,
        model_params,
        model_config,
        feature_cols=feature_cols,
    )
    model.fit(train_data.X, train_data.y)
    preds = model.predict(val_data.X)

    fold_predictions: dict[str, np.ndarray | int] = {}
    if id_col and val_data.id is not None:
        fold_predictions[id_col] = _as_array(val_data.id)

    fold_predictions[era_col] = _as_array(val_data.era)
    fold_predictions[target_col] = _as_array(val_data.y)
    fold_predictions["prediction"] = np.asarray(preds).ravel()
    fold_predictions["cv_fold"] = fold_idx

    fold_metadata: dict[str, object] = {
        "fold": fold_idx,
        "train_eras": len(train_eras),
        "val_eras": len(val_eras),
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "train_intervals": descriptor["train_intervals"],
        "val_interval": descriptor["val_interval"],
        "purge_intervals": descriptor["purge_intervals"],
        "embargo_intervals": descriptor["embargo_intervals"],
    }
    return pd.DataFrame(fold_predictions), fold_metadata


def build_full_history_predictions(
    eras: Iterable[object],
    data_loader: ModelDataLoaderProtocol | Callable[[Sequence[object]], ModelDataBatch],
    model_type: str,
    model_params: dict[str, object],
    model_config: dict[str, object],
    id_col: str | None,
    era_col: str,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train one model on all eras and return in-sample predictions."""
    all_eras = _sorted_unique_eras(eras)
    full_data = _load_data(data_loader, all_eras)
    total_rows = _data_length(full_data)
    if total_rows == 0:
        raise TrainingConfigError("training_full_history_no_rows")

    model = build_model(
        model_type,
        model_params,
        model_config,
        feature_cols=feature_cols,
    )
    model.fit(full_data.X, full_data.y)
    preds = model.predict(full_data.X)

    predictions_payload: dict[str, np.ndarray] = {}
    if id_col and full_data.id is not None:
        predictions_payload[id_col] = _as_array(full_data.id)
    predictions_payload[era_col] = _as_array(full_data.era)
    predictions_payload[target_col] = _as_array(full_data.y)
    predictions_payload["prediction"] = np.asarray(preds).ravel()

    predictions = pd.DataFrame(predictions_payload)
    meta: dict[str, object] = {
        "n_splits": 0,
        "embargo": 0,
        "mode": "full_history",
        "min_train_size": 0,
        "folds_used": 1,
        "folds": [
            {
                "fold": 0,
                "train_eras": len(all_eras),
                "val_eras": len(all_eras),
                "train_rows": int(total_rows),
                "val_rows": int(total_rows),
            }
        ],
    }
    return predictions, meta


def _load_data(
    data_loader: ModelDataLoaderProtocol | Callable[[Sequence[object]], ModelDataBatch],
    eras: Sequence[object],
) -> ModelDataBatch:
    if isinstance(data_loader, ModelDataLoaderProtocol):
        return data_loader.load(eras)

    return data_loader(eras)


def _data_length(data: ModelDataBatch) -> int:
    return len(data.X)


def _as_array(values: pd.Series | np.ndarray) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy())

    return np.asarray(values)


def _build_fold_descriptor(
    *,
    all_eras_sorted: list[object],
    train_eras: Sequence[object],
    val_eras: Sequence[object],
    embargo: int,
) -> dict[str, object]:
    index_by_era = {era: idx for idx, era in enumerate(all_eras_sorted)}
    val_indices = [index_by_era[era] for era in val_eras if era in index_by_era]
    if not val_indices:
        return {
            "train_intervals": _eras_to_intervals(all_eras_sorted, train_eras),
            "val_interval": None,
            "purge_intervals": [],
            "embargo_intervals": [],
        }

    val_start = min(val_indices)
    val_end = max(val_indices)

    embargo_ranges: list[tuple[int, int]] = []
    left_start = max(0, val_start - embargo)
    left_end = val_start - 1
    if left_end >= left_start:
        embargo_ranges.append((left_start, left_end))
    right_start = val_end + 1
    right_end = min(len(all_eras_sorted) - 1, val_end + embargo)
    if right_end >= right_start:
        embargo_ranges.append((right_start, right_end))

    return {
        "train_intervals": _eras_to_intervals(all_eras_sorted, train_eras),
        "val_interval": _eras_to_intervals(all_eras_sorted, val_eras)[0],
        "purge_intervals": _index_ranges_to_intervals(all_eras_sorted, embargo_ranges),
        "embargo_intervals": _index_ranges_to_intervals(all_eras_sorted, embargo_ranges),
    }


def _eras_to_intervals(all_eras_sorted: list[object], selected_eras: Sequence[object]) -> list[dict[str, object]]:
    index_by_era = {era: idx for idx, era in enumerate(all_eras_sorted)}
    indices = sorted({index_by_era[era] for era in selected_eras if era in index_by_era})
    if not indices:
        return []

    ranges: list[tuple[int, int]] = []
    start = indices[0]
    end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
            continue
        ranges.append((start, end))
        start = idx
        end = idx
    ranges.append((start, end))
    return _index_ranges_to_intervals(all_eras_sorted, ranges)


def _index_ranges_to_intervals(
    all_eras_sorted: list[object],
    index_ranges: Sequence[tuple[int, int]],
) -> list[dict[str, object]]:
    intervals: list[dict[str, object]] = []
    for start_idx, end_idx in index_ranges:
        intervals.append(
            {
                "start": all_eras_sorted[start_idx],
                "end": all_eras_sorted[end_idx],
                "start_index": int(start_idx),
                "end_index": int(end_idx),
            }
        )
    return intervals


def _coerce_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TrainingConfigError("training_cv_integer_value_invalid")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise TrainingConfigError("training_cv_integer_value_invalid") from exc
    raise TrainingConfigError("training_cv_integer_value_invalid")


def _coerce_eras_list(value: object, *, field: str) -> list[object] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
        return [item for item in value]
    raise TrainingConfigError(f"training_cv_eras_list_invalid:{field}")
