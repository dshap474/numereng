"""Repository helpers for training configuration, data IO, and artifacts."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from numereng.config.training import TrainingConfigLoaderError, load_training_config_json
from numereng.features.store.layout import ensure_store_root_not_nested
from numereng.features.training.client import TrainingDataClient
from numereng.features.training.errors import TrainingConfigError, TrainingDataError

DEFAULT_DATASETS_DIR = Path(".numereng") / "datasets"
DEFAULT_STORE_DIR = Path(".numereng")
DEFAULT_BASELINES_DIR = Path(".numereng") / "baselines"
DEFAULT_BENCHMARK_MODEL = "v52_lgbm_ender20"
DEFAULT_DATASET_VARIANT = "non_downsampled"
_SUPPORTED_DATASET_VARIANTS = {"non_downsampled", "downsampled"}
_DOWNSAMPLED_VARIANT_FILENAME_MAP: dict[str, str] = {
    "full.parquet": "downsampled_full.parquet",
    "full_benchmark_models.parquet": "downsampled_full_benchmark_models.parquet",
}


def load_config(config_path: Path) -> dict[str, object]:
    """Load canonical training config from one `.json` file."""
    try:
        return load_training_config_json(config_path)
    except TrainingConfigLoaderError as exc:
        raise TrainingConfigError(str(exc)) from exc


def resolve_results_path(config: dict[str, object], config_path: Path, results_dir: Path) -> Path:
    """Resolve canonical run results path."""
    _ = (config, config_path)
    return results_dir / "results.json"


def resolve_predictions_path(config: dict[str, object], config_path: Path, predictions_dir: Path) -> Path:
    """Resolve output path for run prediction parquet."""
    output_config = _as_dict(config.get("output"), field="output")
    predictions_name = output_config.get("predictions_name")
    if not predictions_name:
        predictions_name = output_config.get("results_name") or config_path.stem
    return predictions_dir / f"{predictions_name}.parquet"


def resolve_output_locations(
    config: dict[str, object],
    output_dir_override: Path | None,
    run_id: str,
) -> tuple[Path, Path, Path, Path]:
    """Resolve run-scoped output locations from store-root defaults/overrides."""
    output_config = _as_dict(config.get("output"), field="output")
    data_config = _as_dict(config.get("data"), field="data")

    store_root = _resolve_repo_dir(
        output_dir_override or output_config.get("output_dir"),
        DEFAULT_STORE_DIR,
    )
    try:
        ensure_store_root_not_nested(
            candidate_store_root=store_root,
            canonical_store_root=DEFAULT_STORE_DIR,
            error_code="training_output_store_root_noncanonical",
        )
    except ValueError as exc:
        raise TrainingConfigError(str(exc)) from exc
    baselines_dir = _resolve_repo_dir(
        output_config.get("baselines_dir") or data_config.get("baselines_dir"),
        DEFAULT_BASELINES_DIR,
    )

    run_dir = store_root / "runs" / run_id
    results_dir = run_dir
    predictions_dir = run_dir / "artifacts" / "predictions"
    return run_dir, baselines_dir, results_dir, predictions_dir


def resolve_metrics_path(run_dir: Path) -> Path:
    """Resolve canonical run metrics path."""
    return run_dir / "metrics.json"


def resolve_score_provenance_path(run_dir: Path) -> Path:
    """Resolve canonical run score-provenance path."""
    return run_dir / "score_provenance.json"


def resolve_run_manifest_path(run_dir: Path) -> Path:
    """Resolve canonical run manifest path."""
    return run_dir / "run.json"


def resolve_resolved_config_path(run_dir: Path) -> Path:
    """Resolve canonical resolved-config path for one run."""
    return run_dir / "resolved.json"


def resolve_data_path(path: str | Path | None, *, data_root: Path = DEFAULT_DATASETS_DIR) -> Path:
    """Resolve one data path relative to default dataset root."""
    if path is None:
        raise TrainingDataError("training_data_path_missing")

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    data_root_resolved = data_root.expanduser().resolve()
    data_root_suffix = data_root.parts[-2:] if len(data_root.parts) >= 2 else data_root.parts
    if data_root_suffix and candidate.parts[: len(data_root_suffix)] == data_root_suffix:
        if len(candidate.parts) > len(data_root_suffix):
            remainder = Path(*candidate.parts[len(data_root_suffix) :])
        else:
            remainder = Path()
        return (data_root_resolved / remainder).resolve()

    candidate_resolved = candidate.resolve()
    if candidate_resolved.is_relative_to(data_root_resolved):
        return candidate_resolved

    return (data_root_resolved / candidate).resolve()


def resolve_data_version_root(
    *,
    data_root: Path = DEFAULT_DATASETS_DIR,
    data_version: str,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> Path:
    """Resolve dataset-version directory for one data variant."""
    resolved_root = data_root.expanduser().resolve()
    if dataset_variant in _SUPPORTED_DATASET_VARIANTS:
        return (resolved_root / data_version).resolve()
    raise TrainingConfigError("training_data_dataset_variant_invalid")


def resolve_variant_dataset_filename(*, dataset_variant: str, filename: str) -> str:
    """Resolve canonical dataset filename for one variant."""
    if dataset_variant == "non_downsampled":
        return filename
    if dataset_variant == "downsampled":
        return _DOWNSAMPLED_VARIANT_FILENAME_MAP.get(filename, filename)
    raise TrainingConfigError("training_data_dataset_variant_invalid")


def resolve_derived_dataset_path(
    *,
    data_root: Path = DEFAULT_DATASETS_DIR,
    data_version: str,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    filename: str,
) -> Path:
    """Resolve path for locally-derived dataset artifacts outside canonical dataset cache."""
    data_root_resolved = data_root.expanduser().resolve()
    if data_root_resolved.name == "datasets":
        derived_root = data_root_resolved.parent / "cache" / "derived_datasets"
    else:
        derived_root = data_root_resolved / "_derived"
    if dataset_variant not in _SUPPORTED_DATASET_VARIANTS:
        raise TrainingConfigError("training_data_dataset_variant_invalid")
    version_root = Path(data_version)
    resolved_filename = resolve_variant_dataset_filename(dataset_variant=dataset_variant, filename=filename)
    return (derived_root / version_root / resolved_filename).resolve()


def load_features(
    client: TrainingDataClient,
    data_version: str,
    feature_set: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> list[str]:
    """Load feature list for one data version and feature set."""
    version_root = resolve_data_version_root(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
    )
    features_filename = resolve_variant_dataset_filename(
        dataset_variant=dataset_variant,
        filename="features.json",
    )
    features_path = (version_root / features_filename).resolve()
    features_path.parent.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        _safe_download(
            client,
            filename=f"{data_version}/features.json",
            dest_path=str(features_path),
        )

    try:
        with features_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as exc:
        raise TrainingDataError("training_features_json_invalid") from exc

    feature_sets = metadata.get("feature_sets")
    if not isinstance(feature_sets, dict):
        raise TrainingDataError("training_features_json_missing_feature_sets")

    selected = feature_sets.get(feature_set)
    if not isinstance(selected, list):
        raise TrainingDataError(f"training_feature_set_not_found:{feature_set}")

    return [str(item) for item in selected]


def ensure_full_dataset(
    client: TrainingDataClient,
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> Path:
    """Ensure variant-appropriate full dataset exists locally."""
    version_root = resolve_data_version_root(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
    )
    full_filename = resolve_variant_dataset_filename(dataset_variant=dataset_variant, filename="full.parquet")

    if dataset_variant == "downsampled":
        downsampled_full_path = (version_root / full_filename).resolve()
        if downsampled_full_path.exists():
            return downsampled_full_path
        downsampled_full_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/{full_filename}",
            dest_path=str(downsampled_full_path),
        )
        return downsampled_full_path

    full_path = resolve_derived_dataset_path(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
        filename="full.parquet",
    )
    if full_path.exists():
        return full_path

    train_filename = resolve_variant_dataset_filename(dataset_variant=dataset_variant, filename="train.parquet")
    validation_filename = resolve_variant_dataset_filename(
        dataset_variant=dataset_variant,
        filename="validation.parquet",
    )
    train_path = (version_root / train_filename).resolve()
    validation_path = (version_root / validation_filename).resolve()

    if not train_path.exists():
        train_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/train.parquet",
            dest_path=str(train_path),
        )

    if not validation_path.exists():
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/validation.parquet",
            dest_path=str(validation_path),
        )

    train = _read_parquet(train_path)
    validation = _read_parquet(validation_path)

    if "data_type" in validation.columns:
        validation = validation[validation["data_type"] == "validation"].copy()

    full = pd.concat([train, validation], ignore_index=False)
    full = full.drop(columns=["data_type"], errors="ignore")

    if full.index.name and full.index.name not in full.columns:
        full = full.reset_index()

    full_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(full, full_path, index=False)
    return full_path


def ensure_train_dataset(
    client: TrainingDataClient,
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> Path:
    """Ensure canonical train dataset exists locally and return the parquet path."""
    if dataset_variant == "downsampled":
        return ensure_full_dataset(
            client,
            data_version,
            dataset_variant=dataset_variant,
            data_root=data_root,
        )

    version_root = resolve_data_version_root(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
    )
    train_filename = resolve_variant_dataset_filename(dataset_variant=dataset_variant, filename="train.parquet")
    train_path = (version_root / train_filename).resolve()
    if train_path.exists():
        return train_path

    train_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_download(
        client,
        filename=f"{data_version}/train.parquet",
        dest_path=str(train_path),
    )
    return train_path


def ensure_split_dataset_paths(
    client: TrainingDataClient,
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> tuple[Path, Path]:
    """Ensure canonical train/validation parquet sources are available locally."""
    if dataset_variant == "downsampled":
        raise TrainingConfigError("training_data_dataset_variant_downsampled_disallows_split_sources")

    version_root = resolve_data_version_root(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
    )
    train_filename = resolve_variant_dataset_filename(dataset_variant=dataset_variant, filename="train.parquet")
    validation_filename = resolve_variant_dataset_filename(
        dataset_variant=dataset_variant,
        filename="validation.parquet",
    )
    train_path = (version_root / train_filename).resolve()
    validation_path = (version_root / validation_filename).resolve()

    if not train_path.exists():
        train_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/train.parquet",
            dest_path=str(train_path),
        )

    if not validation_path.exists():
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/validation.parquet",
            dest_path=str(validation_path),
        )

    return train_path, validation_path


def resolve_fold_lazy_source_paths(
    client: TrainingDataClient,
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    full_data_path: str | Path | None = None,
    dataset_scope: str = "train_only",
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> tuple[Path, ...]:
    """Resolve parquet sources used by fold-lazy loading mode."""
    if full_data_path is not None:
        resolved = resolve_data_path(full_data_path, data_root=data_root)
        if not resolved.exists():
            raise TrainingDataError(f"training_full_data_file_not_found:{resolved}")
        return (resolved,)

    if dataset_variant == "downsampled":
        full_path = ensure_full_dataset(
            client,
            data_version,
            dataset_variant=dataset_variant,
            data_root=data_root,
        )
        return (full_path,)

    train_path, validation_path = ensure_split_dataset_paths(
        client,
        data_version,
        dataset_variant=dataset_variant,
        data_root=data_root,
    )
    if dataset_scope == "train_only":
        return (train_path,)
    if dataset_scope == "train_plus_validation":
        return (train_path, validation_path)
    raise TrainingConfigError("training_data_dataset_scope_invalid")


def list_lazy_source_eras(
    source_paths: tuple[Path, ...],
    *,
    era_col: str,
    include_validation_only: bool = True,
) -> list[object]:
    """Return sorted unique eras from lazy parquet sources."""
    eras: list[object] = []
    for source_path in source_paths:
        source_validation_only = _resolve_source_validation_filter(
            source_path,
            include_validation_only=include_validation_only,
        )
        frame = _scan_parquet_source(
            source_path,
            columns=[era_col],
            era_values=None,
            era_col=era_col,
            include_validation_only=source_validation_only,
        )
        if era_col not in frame.columns:
            continue
        eras.extend(frame[era_col].tolist())
    return sorted(set(eras), key=_era_sort_key)


def load_fold_data_lazy(
    source_paths: tuple[Path, ...],
    *,
    eras: list[object],
    columns: list[str],
    era_col: str,
    id_col: str | None,
    include_validation_only: bool = True,
) -> pd.DataFrame:
    """Load one fold-scoped dataframe with parquet predicate/projection pushdown."""
    if not eras:
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    for source_path in source_paths:
        source_validation_only = _resolve_source_validation_filter(
            source_path,
            include_validation_only=include_validation_only,
        )
        scanned = _scan_parquet_source(
            source_path,
            columns=columns,
            era_values=eras,
            era_col=era_col,
            include_validation_only=source_validation_only,
        )
        if not scanned.empty:
            frames.append(scanned)

    if not frames:
        return pd.DataFrame(columns=columns)

    merged = pd.concat(frames, ignore_index=False)

    ordered = [col for col in columns if col in merged.columns]
    merged = merged[ordered]

    if id_col and id_col not in merged.columns:
        merged[id_col] = merged.index

    return merged


def load_full_data(
    client: TrainingDataClient,
    data_version: str,
    dataset_variant: str,
    features: list[str],
    era_col: str,
    target_col: str,
    id_col: str | None,
    full_data_path: str | Path | None = None,
    dataset_scope: str = "train_only",
    extra_cols: list[str] | None = None,
    *,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> pd.DataFrame:
    """Load one full modeling dataframe with selected columns."""
    extra_cols = extra_cols or []
    projected_columns = _dedupe_columns([era_col, target_col, *features, *extra_cols])
    if id_col:
        projected_columns = _dedupe_columns([*projected_columns, id_col])

    if full_data_path:
        full_path = resolve_data_path(full_data_path, data_root=data_root)
        if not full_path.exists():
            raise TrainingDataError(f"training_full_data_file_not_found:{full_path}")
        return _read_full_data_source(
            full_path,
            columns=projected_columns,
            era_col=era_col,
            target_col=target_col,
            feature_cols=features,
            id_col=id_col,
        )

    if dataset_variant == "downsampled":
        full_path = ensure_full_dataset(
            client,
            data_version,
            dataset_variant=dataset_variant,
            data_root=data_root,
        )
        return _read_full_data_source(
            full_path,
            columns=projected_columns,
            era_col=era_col,
            target_col=target_col,
            feature_cols=features,
            id_col=id_col,
        )

    if dataset_scope == "train_only":
        train_path = ensure_train_dataset(
            client,
            data_version,
            dataset_variant=dataset_variant,
            data_root=data_root,
        )
        return _read_full_data_source(
            train_path,
            columns=projected_columns,
            era_col=era_col,
            target_col=target_col,
            feature_cols=features,
            id_col=id_col,
        )

    if dataset_scope == "train_plus_validation":
        train_path, validation_path = ensure_split_dataset_paths(
            client,
            data_version,
            dataset_variant=dataset_variant,
            data_root=data_root,
        )
        train = _read_full_data_source(
            train_path,
            columns=projected_columns,
            era_col=era_col,
            target_col=target_col,
            feature_cols=features,
            id_col=id_col,
        )
        validation = _read_full_data_source(
            validation_path,
            columns=projected_columns,
            era_col=era_col,
            target_col=target_col,
            feature_cols=features,
            id_col=id_col,
            include_data_type=True,
        )

        if "data_type" in validation.columns:
            validation = validation[validation["data_type"] == "validation"].copy()
            validation = validation.drop(columns=["data_type"], errors="ignore")

        return pd.concat([train, validation], ignore_index=False)

    raise TrainingConfigError("training_data_dataset_scope_invalid")


def _read_full_data_source(
    path: Path,
    *,
    columns: list[str],
    era_col: str,
    target_col: str,
    feature_cols: list[str],
    id_col: str | None,
    include_data_type: bool = False,
) -> pd.DataFrame:
    requested_columns = columns
    if include_data_type:
        requested_columns = _dedupe_columns([*columns, "data_type"])

    try:
        frame = _read_parquet(path, columns=requested_columns)
    except TrainingDataError:
        # `validation.parquet` does not always persist `data_type`; retry without it.
        if include_data_type:
            try:
                frame = _read_parquet(path, columns=columns)
            except TrainingDataError:
                frame = _read_parquet(path, columns=[era_col, target_col] + feature_cols)
        else:
            frame = _read_parquet(path, columns=[era_col, target_col] + feature_cols)

        if id_col and id_col not in frame.columns:
            frame[id_col] = frame.index

    return frame


def apply_missing_all_twos_as_nan(
    df: pd.DataFrame,
    feature_cols: list[str],
    era_col: str,
    missing_value: float,
) -> pd.DataFrame:
    """Replace per-era feature values with NaN when entire era has all 2s."""
    if not feature_cols:
        return df.copy()

    updated = df.copy()
    features = updated[feature_cols].astype("float32", copy=False)
    is_all_twos = features.eq(missing_value).groupby(updated[era_col]).transform("all")
    updated.loc[:, feature_cols] = features.mask(is_all_twos, np.nan)
    return updated


def select_prediction_columns(
    predictions: pd.DataFrame,
    id_col: str | None,
    era_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Filter output prediction columns into canonical ordering."""
    prediction_cols = [col for col in [id_col, era_col, target_col, "prediction", "cv_fold"] if col]
    prediction_cols = [col for col in prediction_cols if col in predictions.columns]
    return predictions[prediction_cols].copy()


def save_predictions(
    predictions: pd.DataFrame,
    config: dict[str, object],
    config_path: Path,
    predictions_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist OOF predictions parquet and return absolute+relative paths."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = resolve_predictions_path(config, config_path, predictions_dir)
    _write_parquet(predictions, predictions_path, index=False)
    predictions_relative = predictions_path.relative_to(output_dir)
    return predictions_path, predictions_relative


def save_results(results: dict[str, object], results_path: Path) -> None:
    """Persist training results payload as JSON."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def save_metrics(metrics: dict[str, object], metrics_path: Path) -> None:
    """Persist run metrics payload as JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def save_score_provenance(score_provenance: dict[str, object], score_provenance_path: Path) -> None:
    """Persist scoring provenance payload as JSON."""
    score_provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with score_provenance_path.open("w", encoding="utf-8") as f:
        json.dump(score_provenance, f, indent=2, sort_keys=True)


def save_resolved_config(config: dict[str, object], resolved_config_path: Path) -> None:
    """Persist resolved config snapshot for one run."""
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)


def save_run_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
    """Persist run manifest atomically."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp_path.replace(manifest_path)


def _resolve_repo_dir(path: object, default: Path) -> Path:
    if not path:
        return default.resolve()

    if isinstance(path, Path):
        candidate = path.expanduser()
    elif isinstance(path, str):
        candidate = Path(path).expanduser()
    else:
        raise TrainingConfigError("training_output_path_invalid_type")

    if candidate.is_absolute():
        return candidate.resolve()

    return candidate.resolve()


def _safe_download(client: TrainingDataClient, *, filename: str, dest_path: str) -> None:
    try:
        client.download_dataset(filename=filename, dest_path=dest_path)
    except Exception as exc:
        raise TrainingDataError(f"training_dataset_download_failed:{filename}") from exc


def _resolve_source_validation_filter(path: Path, *, include_validation_only: bool) -> bool:
    """Apply validation-only filtering only for validation parquet sources."""
    if not include_validation_only:
        return False

    source_name = path.name.lower()
    return "validation" in source_name and "train" not in source_name


def _scan_parquet_source(
    path: Path,
    *,
    columns: list[str],
    era_values: list[object] | None,
    era_col: str,
    include_validation_only: bool,
) -> pd.DataFrame:
    try:
        ds = import_module("pyarrow.dataset")
    except ImportError as exc:
        raise TrainingDataError("training_data_lazy_backend_missing_pyarrow") from exc

    try:
        dataset = ds.dataset(str(path), format="parquet")
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_read_failed:{path}") from exc

    available_columns = {str(name) for name in dataset.schema.names}
    projected = _dedupe_columns(columns)
    projected = [col for col in projected if col in available_columns]

    predicate = None
    if era_values is not None:
        if era_col not in available_columns:
            raise TrainingDataError(f"training_data_era_col_not_found:{era_col}")
        predicate = ds.field(era_col).isin(list(era_values))

    include_data_type_filter = include_validation_only and "data_type" in available_columns
    if include_data_type_filter:
        if "data_type" not in projected:
            projected.append("data_type")
        validation_filter = (ds.field("data_type") == "validation") | ds.field("data_type").is_null()
        predicate = validation_filter if predicate is None else predicate & validation_filter

    try:
        table = dataset.to_table(columns=projected or None, filter=predicate)
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_read_failed:{path}") from exc

    frame = table.to_pandas()
    if include_data_type_filter and "data_type" in frame.columns:
        frame = frame.drop(columns=["data_type"])

    ordered_cols = [col for col in columns if col in frame.columns]
    extra_cols = [col for col in frame.columns if col not in ordered_cols]
    return cast(pd.DataFrame, frame[ordered_cols + extra_cols])


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    return ordered


def _era_sort_key(era: object) -> int | str:
    if isinstance(era, (int, np.integer)):
        return int(era)
    if isinstance(era, str):
        if era.isdigit():
            return int(era)
        return era
    return str(era)


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except ImportError as exc:
        raise TrainingDataError("training_data_parquet_engine_missing") from exc
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_read_failed:{path}") from exc


def _write_parquet(df: pd.DataFrame, path: Path, *, index: bool) -> None:
    try:
        df.to_parquet(path, index=index)
    except ImportError as exc:
        raise TrainingDataError("training_data_parquet_engine_missing") from exc
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_write_failed:{path}") from exc


def _as_dict(value: object, *, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TrainingConfigError(f"training_config_field_not_mapping:{field}")
