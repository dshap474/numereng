"""Reusable Numerai scoring helpers for training pipelines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from numereng.features.training.client import TrainingDataClient
from numereng.features.training.errors import TrainingDataError, TrainingMetricsError
from numereng.features.training.repo import (
    DEFAULT_BENCHMARK_MODEL,
    DEFAULT_DATASET_VARIANT,
    DEFAULT_DATASETS_DIR,
    load_features,
    load_fold_data_lazy,
    resolve_data_path,
    resolve_data_version_root,
    resolve_derived_dataset_path,
    resolve_fold_lazy_source_paths,
    resolve_variant_dataset_filename,
)
from numereng.features.scoring.models import ResolvedScoringPolicy, default_scoring_policy

DEFAULT_META_MODEL_COL = "numerai_meta_model"


def _resolve_scoring_policy(scoring_policy: ResolvedScoringPolicy | None) -> ResolvedScoringPolicy:
    return scoring_policy or default_scoring_policy()


def _resolve_scoring_targets(
    *,
    target_col: str,
    scoring_target_cols: Sequence[str] | None,
) -> list[str]:
    if scoring_target_cols is None:
        return [target_col]
    resolved = [str(col).strip() for col in scoring_target_cols]
    if not resolved:
        raise TrainingDataError("training_scoring_targets_empty")
    if any(not col for col in resolved):
        raise TrainingDataError("training_scoring_targets_invalid")
    deduped: list[str] = []
    seen: set[str] = set()
    for col in resolved:
        if col in seen:
            continue
        seen.add(col)
        deduped.append(col)
    return deduped


def _target_metric_alias(target_col: str) -> str:
    alias = target_col
    if alias.startswith("target_"):
        alias = alias[len("target_") :]
    parts = [part for part in alias.split("_") if part]
    if not parts:
        return alias.replace("_", "")
    return "".join(parts)


def _validate_target_aliases(target_cols: Sequence[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    seen: dict[str, str] = {}
    for target_col in target_cols:
        alias = _target_metric_alias(target_col)
        prior = seen.get(alias)
        if prior is not None and prior != target_col:
            raise TrainingDataError(f"training_scoring_target_alias_collision:{prior},{target_col}->{alias}")
        seen[alias] = target_col
        aliases[target_col] = alias
    return aliases


def _target_metric_key(*, metric_name: str, target_col: str, native_target_col: str, aliases: dict[str, str]) -> str:
    if target_col == native_target_col:
        return metric_name
    return f"{metric_name}_{aliases[target_col]}"


def _validate_min_overlap_ratio(
    *,
    source_name: str,
    predictions_count: int,
    source_count: int,
    overlap_count: int,
    min_overlap_ratio: float,
) -> float:
    if predictions_count < 1:
        raise TrainingDataError("training_predictions_missing_rows")
    overlap_ratio = overlap_count / predictions_count
    if overlap_ratio < min_overlap_ratio:
        raise TrainingDataError(
            f"training_{source_name}_partial_id_overlap:"
            f"predictions={predictions_count},"
            f"{source_name}={source_count},"
            f"overlap={overlap_count},"
            f"min_overlap_ratio={min_overlap_ratio:.2f}"
        )
    return overlap_ratio


def _load_scoring_functions() -> tuple[Callable[..., pd.Series], Callable[..., pd.Series]]:
    try:
        from numerai_tools.scoring import (
            correlation_contribution,
            numerai_corr,
        )
    except ImportError as exc:
        raise TrainingMetricsError("training_metrics_dependency_missing_numerai_tools") from exc
    return correlation_contribution, numerai_corr


def _load_feature_neutral_corr() -> Callable[..., pd.Series]:
    try:
        from numerai_tools.scoring import feature_neutral_corr
    except ImportError as exc:
        raise TrainingMetricsError("training_metrics_dependency_missing_numerai_tools") from exc
    return feature_neutral_corr


def _load_prediction_corr_functions() -> tuple[Callable[..., pd.DataFrame], Callable[..., float]]:
    try:
        from numerai_tools.scoring import (
            pearson_correlation,
            tie_kept_rank__gaussianize__pow_1_5,
        )
    except ImportError as exc:
        raise TrainingMetricsError("training_metrics_dependency_missing_numerai_tools") from exc
    return tie_kept_rank__gaussianize__pow_1_5, pearson_correlation


def _as_list(cols: Iterable[object]) -> list[object]:
    if isinstance(cols, (list, tuple, pd.Index, np.ndarray)):
        return list(cols)
    return [cols]


def _sort_era_index(scores: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if scores.index.dtype == object:
        try:
            order = sorted(scores.index, key=lambda value: int(value))
            return scores.loc[order]
        except (TypeError, ValueError):
            return scores.sort_index()
    return scores.sort_index()


def _normalize_per_era(per_era: object, pred_cols: Sequence[object]) -> pd.DataFrame:
    if isinstance(per_era, pd.DataFrame):
        return cast(pd.DataFrame, _sort_era_index(per_era))

    if isinstance(per_era, pd.Series):
        if per_era.index.nlevels > 1:
            unstacked = per_era.unstack()
            return cast(pd.DataFrame, _sort_era_index(unstacked))
        return cast(pd.DataFrame, _sort_era_index(per_era.to_frame(str(pred_cols[0]))))

    raise TrainingMetricsError("training_metrics_unexpected_per_era_type")


def _last_n_eras(per_era_scores: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    if n <= 0:
        raise TrainingMetricsError("training_metrics_last_n_eras_invalid")

    sorted_scores = _sort_era_index(per_era_scores)
    return sorted_scores.tail(n)


def _groupby_apply_per_era(
    frame: pd.DataFrame,
    *,
    era_col: str,
    fn: Callable[[pd.DataFrame], pd.Series | float],
) -> object:
    """Apply per-era callback without including grouping columns in the payload."""

    grouped = frame.groupby(era_col)
    apply_fn = cast(Callable[..., object], grouped.apply)
    try:
        return apply_fn(fn, include_groups=False)
    except TypeError as exc:
        if "include_groups" not in str(exc):
            raise
        return apply_fn(fn)


def _single_prediction_per_era(
    *,
    frame: pd.DataFrame,
    pred_col: str,
    required_cols: Sequence[str],
    era_col: str,
    scorer: Callable[[pd.DataFrame, str], pd.Series],
) -> pd.Series:
    """Compute per-era scores for one prediction column with column-local NaN filtering."""
    filtered = frame.dropna(subset=[pred_col, *required_cols])
    if filtered.empty:
        return pd.Series(dtype="float64", name=pred_col)

    def _score(group: pd.DataFrame) -> float:
        scores = scorer(group, pred_col)
        if pred_col in scores.index:
            return float(scores[pred_col])
        if len(scores.index) == 1:
            return float(scores.iloc[0])
        raise TrainingMetricsError("training_metrics_unexpected_per_era_type")

    per_era = _groupby_apply_per_era(filtered, era_col=era_col, fn=_score)
    if not isinstance(per_era, pd.Series):
        raise TrainingMetricsError("training_metrics_unexpected_per_era_type")
    result = cast(pd.Series, _sort_era_index(per_era.astype("float64", copy=False)))
    result.name = pred_col
    return result


def per_era_corr(
    df: pd.DataFrame,
    pred_cols: Sequence[object],
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era Numerai correlation for one or more prediction columns."""
    _, numerai_corr = _load_scoring_functions()

    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    per_col: list[pd.Series] = []
    for pred_col in resolved_pred_cols:
        per_col.append(
            _single_prediction_per_era(
                frame=df,
                pred_col=pred_col,
                required_cols=(target_col,),
                era_col=era_col,
                scorer=lambda group, col: numerai_corr(group[[col]], group[target_col]),
            )
        )
    if not per_col:
        return pd.DataFrame()
    return cast(pd.DataFrame, _sort_era_index(pd.concat(per_col, axis=1)))


def per_era_cwmm(
    df: pd.DataFrame,
    pred_cols: Sequence[object],
    reference_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era CWMM-style correlation against a raw reference series."""
    transform_predictions, pearson_correlation = _load_prediction_corr_functions()

    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    per_col: list[pd.Series] = []
    for pred_col in resolved_pred_cols:
        per_col.append(
            _single_prediction_per_era(
                frame=df,
                pred_col=pred_col,
                required_cols=(reference_col,),
                era_col=era_col,
                scorer=lambda group, col: _prediction_reference_corr(
                    predictions=group[[col]],
                    reference=group[reference_col],
                    transform_predictions=transform_predictions,
                    pearson_correlation=pearson_correlation,
                ),
            )
        )
    if not per_col:
        return pd.DataFrame()
    return cast(pd.DataFrame, _sort_era_index(pd.concat(per_col, axis=1)))


def per_era_reference_corr(
    df: pd.DataFrame,
    pred_cols: Sequence[object],
    reference_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era Numerai correlation between predictions and a reference series."""
    _, numerai_corr = _load_scoring_functions()

    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    per_col: list[pd.Series] = []
    for pred_col in resolved_pred_cols:
        per_col.append(
            _single_prediction_per_era(
                frame=df,
                pred_col=pred_col,
                required_cols=(reference_col,),
                era_col=era_col,
                scorer=lambda group, col: numerai_corr(group[[col]], group[reference_col]),
            )
        )
    if not per_col:
        return pd.DataFrame()
    return cast(pd.DataFrame, _sort_era_index(pd.concat(per_col, axis=1)))


def _prediction_reference_corr(
    *,
    predictions: pd.DataFrame,
    reference: pd.Series,
    transform_predictions: Callable[..., pd.DataFrame],
    pearson_correlation: Callable[..., float],
) -> pd.Series:
    transformed_predictions = transform_predictions(predictions)
    scores = transformed_predictions.apply(lambda submission: pearson_correlation(reference, submission))
    return cast(pd.Series, scores)


def per_era_bmc(
    df: pd.DataFrame,
    pred_cols: Sequence[object],
    benchmark_col: str,
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era benchmark contribution scores."""
    correlation_contribution, _ = _load_scoring_functions()

    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    per_col: list[pd.Series] = []
    for pred_col in resolved_pred_cols:
        per_col.append(
            _single_prediction_per_era(
                frame=df,
                pred_col=pred_col,
                required_cols=(benchmark_col, target_col),
                era_col=era_col,
                scorer=lambda group, col: correlation_contribution(
                    group[[col]],
                    group[benchmark_col],
                    group[target_col],
                ),
            )
        )
    if not per_col:
        return pd.DataFrame()
    return cast(pd.DataFrame, _sort_era_index(pd.concat(per_col, axis=1)))


def per_era_mmc(
    df: pd.DataFrame,
    pred_cols: Sequence[object],
    meta_model_col: str,
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era meta model contribution scores."""
    correlation_contribution, _ = _load_scoring_functions()

    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    per_col: list[pd.Series] = []
    for pred_col in resolved_pred_cols:
        per_col.append(
            _single_prediction_per_era(
                frame=df,
                pred_col=pred_col,
                required_cols=(meta_model_col, target_col),
                era_col=era_col,
                scorer=lambda group, col: correlation_contribution(
                    group[[col]],
                    group[meta_model_col],
                    group[target_col],
                ),
            )
        )
    if not per_col:
        return pd.DataFrame()
    return cast(pd.DataFrame, _sort_era_index(pd.concat(per_col, axis=1)))


def per_era_fnc(
    df: pd.DataFrame,
    pred_cols: Sequence[object],
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era feature neutral correlation for one or more prediction columns."""
    feature_neutral_corr = _load_feature_neutral_corr()
    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    resolved_feature_cols = [str(col) for col in feature_cols]
    if not resolved_feature_cols:
        raise TrainingDataError("training_fnc_feature_cols_empty")

    per_col: list[pd.Series] = []
    required_cols = [target_col, *resolved_feature_cols]
    for pred_col in resolved_pred_cols:
        per_col.append(
            _single_prediction_per_era(
                frame=df,
                pred_col=pred_col,
                required_cols=required_cols,
                era_col=era_col,
                scorer=lambda group, col: feature_neutral_corr(
                    group[[col]],
                    group[resolved_feature_cols],
                    group[target_col],
                ),
            )
        )
    if not per_col:
        return pd.DataFrame()
    return cast(pd.DataFrame, _sort_era_index(pd.concat(per_col, axis=1)))


def _rank_normalize_series(series: pd.Series) -> pd.Series:
    values = series.astype("float64", copy=False)
    if values.empty:
        return values
    return values.rank(method="average", pct=True)


def _feature_exposure_stats_for_frame(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    feature_cols: Sequence[str],
) -> tuple[float, float]:
    filtered = frame.dropna(subset=[pred_col, *feature_cols])
    if filtered.empty:
        return float(np.nan), float(np.nan)

    ranked_pred = _rank_normalize_series(filtered[pred_col])
    ranked_features = filtered[list(feature_cols)].apply(_rank_normalize_series, axis=0)
    valid_feature_cols = ranked_features.std(axis=0).replace(0.0, np.nan).dropna().index.tolist()
    if not valid_feature_cols:
        return float(np.nan), float(np.nan)
    ranked_features = ranked_features[valid_feature_cols]
    exposures = ranked_features.corrwith(ranked_pred).abs()
    exposures = exposures.dropna()
    if exposures.empty:
        return float(np.nan), float(np.nan)

    rms_exposure = float(np.sqrt(np.square(exposures).mean()))
    max_exposure = float(exposures.max())
    return rms_exposure, max_exposure


def per_era_feature_exposure(
    df: pd.DataFrame,
    pred_cols: Sequence[object],
    feature_cols: Sequence[str],
    era_col: str = "era",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-era RMS and max absolute feature exposure for predictions."""
    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    resolved_feature_cols = [str(col) for col in feature_cols]
    if not resolved_feature_cols:
        raise TrainingDataError("training_feature_exposure_feature_cols_empty")

    rms_series: list[pd.Series] = []
    max_series: list[pd.Series] = []
    for pred_col in resolved_pred_cols:
        rms_per_era = _groupby_apply_per_era(
            df,
            era_col=era_col,
            fn=lambda group: _feature_exposure_stats_for_frame(
                group,
                pred_col=pred_col,
                feature_cols=resolved_feature_cols,
            )[0],
        )
        max_per_era = _groupby_apply_per_era(
            df,
            era_col=era_col,
            fn=lambda group: _feature_exposure_stats_for_frame(
                group,
                pred_col=pred_col,
                feature_cols=resolved_feature_cols,
            )[1],
        )
        if not isinstance(rms_per_era, pd.Series) or not isinstance(max_per_era, pd.Series):
            raise TrainingMetricsError("training_metrics_unexpected_per_era_type")
        rms_result = cast(pd.Series, _sort_era_index(rms_per_era.astype("float64", copy=False)))
        rms_result.name = pred_col
        rms_series.append(rms_result)

        max_result = cast(pd.Series, _sort_era_index(max_per_era.astype("float64", copy=False)))
        max_result.name = pred_col
        max_series.append(max_result)

    if not rms_series:
        return pd.DataFrame(), pd.DataFrame()
    return (
        cast(pd.DataFrame, _sort_era_index(pd.concat(rms_series, axis=1))),
        cast(pd.DataFrame, _sort_era_index(pd.concat(max_series, axis=1))),
    )


def max_drawdown(scores: pd.Series) -> float:
    """Compute max drawdown for one score series."""
    values = scores.dropna()
    if values.empty:
        return float(np.nan)

    cumsum = values.cumsum()
    running_max = cumsum.expanding(min_periods=1).max()
    return float((running_max - cumsum).max())


def score_summary(scores: pd.Series) -> dict[str, float]:
    """Return mean/std/sharpe/max_drawdown summary for one score series."""
    values = scores.dropna()
    if values.empty:
        return {
            "mean": float(np.nan),
            "std": float(np.nan),
            "sharpe": float(np.nan),
            "max_drawdown": float(np.nan),
        }

    mean = float(values.mean())
    std = float(values.std(ddof=0))
    sharpe = float(mean / std) if std != 0.0 else float(np.nan)
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(values),
    }


def summarize_scores(per_era_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-era score dataframe into aggregate metrics."""
    sorted_scores = cast(pd.DataFrame, _sort_era_index(per_era_scores))
    summary = {col: score_summary(sorted_scores[col]) for col in sorted_scores}
    return pd.DataFrame(summary).T


def load_custom_benchmark_predictions(
    predictions_path: str | Path,
    benchmark_name: str,
    pred_col: str = "prediction",
    era_col: str = "era",
    id_col: str = "id",
) -> tuple[pd.DataFrame, str]:
    """Load local predictions file and expose it as one benchmark column."""
    resolved_path = Path(predictions_path).expanduser().resolve()
    benchmark = _read_table(resolved_path)

    required = [col for col in [era_col, pred_col, id_col] if col in benchmark.columns]
    if required:
        benchmark = benchmark[required]

    if pred_col not in benchmark.columns:
        raise TrainingDataError(f"training_benchmark_predictions_missing_col:{pred_col}")

    if pred_col != benchmark_name:
        benchmark = benchmark.rename(columns={pred_col: benchmark_name})

    return benchmark, benchmark_name


def ensure_full_benchmark_models(
    client: TrainingDataClient,
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> Path:
    """Ensure full benchmark model parquet exists in derived cache."""
    version_root = resolve_data_version_root(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
    )
    full_filename = resolve_variant_dataset_filename(
        dataset_variant=dataset_variant,
        filename="full_benchmark_models.parquet",
    )

    if dataset_variant == "downsampled":
        downsampled_full_benchmark_path = (version_root / full_filename).resolve()
        if downsampled_full_benchmark_path.exists():
            return downsampled_full_benchmark_path
        downsampled_full_benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/{full_filename}",
            dest_path=str(downsampled_full_benchmark_path),
        )
        return downsampled_full_benchmark_path

    full_path = resolve_derived_dataset_path(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
        filename="full_benchmark_models.parquet",
    )
    if full_path.exists():
        return full_path

    train_filename = resolve_variant_dataset_filename(
        dataset_variant=dataset_variant,
        filename="train_benchmark_models.parquet",
    )
    validation_filename = resolve_variant_dataset_filename(
        dataset_variant=dataset_variant,
        filename="validation_benchmark_models.parquet",
    )
    validation_data_filename = resolve_variant_dataset_filename(
        dataset_variant=dataset_variant,
        filename="validation.parquet",
    )
    train_path = (version_root / train_filename).resolve()
    validation_path = (version_root / validation_filename).resolve()
    validation_data_path = (version_root / validation_data_filename).resolve()

    if not train_path.exists():
        train_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/train_benchmark_models.parquet",
            dest_path=str(train_path),
        )

    if not validation_path.exists():
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/validation_benchmark_models.parquet",
            dest_path=str(validation_path),
        )

    if not validation_data_path.exists():
        validation_data_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_download(
            client,
            filename=f"{data_version}/validation.parquet",
            dest_path=str(validation_data_path),
        )

    validation_meta = _read_parquet(validation_data_path, columns=["data_type"])
    validation_meta = validation_meta[validation_meta["data_type"] == "validation"]
    validation_ids = validation_meta.index

    train = _read_parquet(train_path)
    validation = _read_parquet(validation_path)
    if "id" in train.columns:
        train = train.set_index("id")
    if "id" in validation.columns:
        validation = validation.set_index("id")
    validation = validation.loc[validation.index.intersection(validation_ids)]

    full = pd.concat([train, validation], axis=0)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(full, full_path)
    return full_path


def resolve_benchmark_predictions_path(
    client: TrainingDataClient,
    data_version: str,
    split: str = "full",
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> Path:
    """Resolve local parquet path for benchmark model predictions."""
    if split == "full":
        return ensure_full_benchmark_models(
            client,
            data_version,
            dataset_variant=dataset_variant,
            data_root=data_root,
        )

    remote_name = f"{data_version}/{split}_benchmark_models.parquet"
    version_root = resolve_data_version_root(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
    )
    local_filename = resolve_variant_dataset_filename(
        dataset_variant=dataset_variant,
        filename=f"{split}_benchmark_models.parquet",
    )
    local_path = (version_root / local_filename).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        _safe_download(client, filename=remote_name, dest_path=str(local_path))
    return local_path


def load_benchmark_predictions(
    client: TrainingDataClient,
    data_version: str,
    split: str = "full",
    benchmark_model: str = DEFAULT_BENCHMARK_MODEL,
    era_col: str = "era",
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> tuple[pd.DataFrame, str]:
    """Load benchmark predictions for one split."""
    dataset_path = resolve_benchmark_predictions_path(
        client,
        data_version,
        split=split,
        dataset_variant=dataset_variant,
        data_root=data_root,
    )

    columns = _table_columns(dataset_path)
    benchmark_col = _resolve_benchmark_column(columns, benchmark_model)
    benchmark = _read_parquet(dataset_path, columns=[era_col, benchmark_col])
    if benchmark.index.name is None:
        benchmark.index.name = "id"
    return benchmark, benchmark_col


def load_benchmark_predictions_from_path(
    benchmark_path: str | Path,
    benchmark_model: str,
    era_col: str = "era",
    id_col: str = "id",
    *,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> tuple[pd.DataFrame, str]:
    """Load benchmark predictions from local parquet/csv path."""
    path = resolve_data_path(benchmark_path, data_root=data_root)
    columns = _table_columns(path)
    benchmark_col = _resolve_benchmark_column(columns, benchmark_model)

    read_cols = [benchmark_col]
    if id_col in columns:
        read_cols.append(id_col)
    if era_col in columns:
        read_cols.append(era_col)

    benchmark = _read_table(path, columns=read_cols)

    if id_col in benchmark.columns:
        benchmark = benchmark.set_index(id_col)

    if benchmark.index.name is None:
        benchmark.index.name = id_col

    return benchmark, benchmark_col


def resolve_meta_model_path(
    client: TrainingDataClient,
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    meta_model_data_path: str | Path | None = None,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> Path:
    """Resolve local parquet path for meta model predictions."""
    if meta_model_data_path is not None:
        return resolve_data_path(meta_model_data_path, data_root=data_root)

    version_root = resolve_data_version_root(
        data_root=data_root,
        data_version=data_version,
        dataset_variant=dataset_variant,
    )
    filename = resolve_variant_dataset_filename(dataset_variant=dataset_variant, filename="meta_model.parquet")
    path = (version_root / filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _safe_download(client, filename=f"{data_version}/meta_model.parquet", dest_path=str(path))
    return path


def load_meta_model_predictions(
    client: TrainingDataClient,
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    meta_model_col: str = DEFAULT_META_MODEL_COL,
    meta_model_data_path: str | Path | None = None,
    era_col: str = "era",
    id_col: str = "id",
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> tuple[pd.DataFrame, str, Path]:
    """Load meta model predictions from local parquet/csv path."""
    path = resolve_meta_model_path(
        client,
        data_version,
        dataset_variant=dataset_variant,
        meta_model_data_path=meta_model_data_path,
        data_root=data_root,
    )
    columns = _table_columns(path)
    resolved_col = _resolve_meta_model_column(columns, meta_model_col, era_col=era_col, id_col=id_col)

    read_cols = [resolved_col]
    if era_col in columns:
        read_cols.append(era_col)
    if "data_type" in columns:
        read_cols.append("data_type")
    if id_col in columns:
        read_cols.append(id_col)

    meta_model = _read_table(path, columns=read_cols)
    if id_col in meta_model.columns:
        meta_model = meta_model.set_index(id_col)

    if meta_model.index.name is None:
        meta_model.index.name = id_col

    return meta_model, resolved_col, path


def attach_benchmark_predictions(
    predictions: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_col: str,
    era_col: str = "era",
    id_col: str = "id",
    min_overlap_ratio: float = 0.0,
) -> pd.DataFrame:
    """Align and attach benchmark predictions to model predictions."""
    if id_col in predictions.columns:
        preds = predictions.set_index(id_col)
        bench = benchmark

        if bench.index.name != id_col:
            if id_col in bench.columns:
                bench = bench.set_index(id_col)
            else:
                raise TrainingDataError(f"training_benchmark_missing_id_col:{id_col}")

        common_ids = preds.index.intersection(bench.index)
        if common_ids.empty:
            raise TrainingDataError("training_benchmark_no_overlapping_ids")
        _validate_min_overlap_ratio(
            source_name="benchmark",
            predictions_count=len(preds.index),
            source_count=len(bench.index),
            overlap_count=len(common_ids),
            min_overlap_ratio=min_overlap_ratio,
        )

        preds = preds.loc[common_ids]
        bench = bench.loc[common_ids]

        if era_col in bench.columns and not np.array_equal(
            bench[era_col].astype(str).to_numpy(),
            preds[era_col].astype(str).to_numpy(),
        ):
            raise TrainingDataError("training_benchmark_eras_misaligned")

        preds[benchmark_col] = bench[benchmark_col].to_numpy()
        return preds.reset_index()

    bench = benchmark[benchmark[era_col].isin(predictions[era_col])]
    if len(bench) != len(predictions):
        raise TrainingDataError("training_benchmark_rows_mismatch")

    if not np.array_equal(bench[era_col].to_numpy(), predictions[era_col].to_numpy()):
        raise TrainingDataError("training_benchmark_eras_misaligned")

    enriched = predictions.copy()
    enriched[benchmark_col] = bench[benchmark_col].to_numpy()
    return enriched


def attach_meta_model_predictions(
    predictions: pd.DataFrame,
    meta_model: pd.DataFrame,
    meta_model_col: str,
    *,
    era_col: str = "era",
    id_col: str = "id",
    min_overlap_ratio: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Align and attach meta model predictions to model predictions."""
    if id_col not in predictions.columns:
        raise TrainingDataError(f"training_predictions_missing_columns:{id_col}")

    preds = predictions.set_index(id_col)
    meta = meta_model
    if meta.index.name != id_col:
        if id_col in meta.columns:
            meta = meta.set_index(id_col)
        else:
            raise TrainingDataError(f"training_meta_model_missing_id_col:{id_col}")

    common_ids = preds.index.intersection(meta.index)
    if common_ids.empty:
        raise TrainingDataError("training_meta_model_no_overlapping_ids")
    overlap_ratio = _validate_min_overlap_ratio(
        source_name="meta_model",
        predictions_count=len(preds.index),
        source_count=len(meta.index),
        overlap_count=len(common_ids),
        min_overlap_ratio=min_overlap_ratio,
    )

    preds = preds.loc[common_ids]
    meta = meta.loc[common_ids]
    if era_col in meta.columns and not np.array_equal(
        meta[era_col].astype(str).to_numpy(),
        preds[era_col].astype(str).to_numpy(),
    ):
        raise TrainingDataError("training_meta_model_eras_misaligned")

    preds[meta_model_col] = meta[meta_model_col].to_numpy()
    attached = preds.reset_index()
    stats = {
        "predictions_rows": int(len(predictions)),
        "meta_source_rows": int(len(meta_model)),
        "meta_overlap_rows": int(len(common_ids)),
        "meta_overlap_eras": int(attached[era_col].nunique()) if era_col in attached.columns else 0,
        "meta_overlap_ratio": float(overlap_ratio),
    }
    return attached, stats


def validate_join_source_coverage(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    *,
    source_name: str,
    era_col: str,
    id_col: str,
    min_overlap_ratio: float = 1.0,
    include_missing_counts: bool = False,
    allow_zero_overlap: bool = False,
) -> dict[str, int | float]:
    """Validate `(id, era)` coverage for one scoring join source."""
    prediction_frame = _normalize_join_keys_frame(predictions, id_col=id_col, era_col=era_col, source_name=None)
    source_frame = _normalize_join_keys_frame(source, id_col=id_col, era_col=era_col, source_name=source_name)

    prediction_keys = prediction_frame[[id_col, era_col]].copy()
    source_keys = source_frame[[id_col, era_col]].copy()
    prediction_key_index = pd.MultiIndex.from_frame(prediction_keys)
    source_key_index = pd.MultiIndex.from_frame(source_keys)
    overlap = prediction_key_index.intersection(source_key_index)
    missing = prediction_key_index.difference(source_key_index)
    if overlap.empty and not allow_zero_overlap:
        raise TrainingDataError(f"training_{source_name}_no_overlapping_ids")
    if overlap.empty:
        overlap_ratio = 0.0
    else:
        overlap_ratio = _validate_min_overlap_ratio(
            source_name=source_name,
            predictions_count=len(prediction_key_index),
            source_count=len(source_key_index),
            overlap_count=len(overlap),
            min_overlap_ratio=min_overlap_ratio,
        )

    stats = {
        f"{source_name}_source_rows": int(len(source_keys)),
        f"{source_name}_overlap_rows": int(len(overlap)),
        f"{source_name}_overlap_eras": int(overlap.to_frame(index=False)[era_col].nunique()),
        f"{source_name}_overlap_ratio": float(overlap_ratio),
    }
    if include_missing_counts:
        stats[f"{source_name}_missing_rows"] = int(len(missing))
        stats[f"{source_name}_missing_eras"] = (
            int(missing.to_frame(index=False)[era_col].nunique()) if len(missing) else 0
        )
    return stats


def _normalize_join_keys_frame(
    frame: pd.DataFrame,
    *,
    id_col: str,
    era_col: str,
    source_name: str | None,
) -> pd.DataFrame:
    normalized = frame
    if id_col not in normalized.columns:
        if normalized.index.name == id_col:
            normalized = normalized.reset_index()
        else:
            if source_name is None:
                raise TrainingDataError(f"training_predictions_missing_columns:{id_col}")
            raise TrainingDataError(f"training_{source_name}_missing_id_col:{id_col}")
    if era_col not in normalized.columns:
        if source_name is None:
            raise TrainingDataError(f"training_predictions_missing_columns:{era_col}")
        raise TrainingDataError(f"training_{source_name}_missing_era_col:{era_col}")
    return normalized


def read_join_source_keys(
    path: Path,
    *,
    id_col: str,
    era_col: str,
    extra_cols: Sequence[str] | None = None,
    source_name: str,
) -> pd.DataFrame:
    """Read the minimal join-key frame from one predictions/benchmark/meta file."""
    available_columns = set(_table_columns(path))
    requested = [col for col in [id_col, era_col, *(extra_cols or ())] if col in available_columns]
    frame = _read_table(path, columns=requested or None)
    normalized = _normalize_join_keys_frame(frame, id_col=id_col, era_col=era_col, source_name=source_name)
    keep_cols = [col for col in [id_col, era_col, *(extra_cols or ())] if col in normalized.columns]
    return normalized[keep_cols]


def resolve_fnc_feature_columns(
    *,
    client: TrainingDataClient,
    data_version: str,
    dataset_variant: str,
    scoring_policy: ResolvedScoringPolicy,
    data_root: Path,
) -> list[str]:
    """Resolve feature columns used as neutralizers for FNC."""
    return load_features(
        client,
        data_version,
        scoring_policy.fnc_feature_set,
        dataset_variant=dataset_variant,
        data_root=data_root,
    )


def resolve_fnc_source_paths(
    *,
    client: TrainingDataClient,
    data_version: str,
    dataset_variant: str,
    feature_source_paths: Sequence[Path] | None,
    full_data_path: str | Path | None,
    dataset_scope: str,
    data_root: Path,
) -> tuple[tuple[Path, ...], bool]:
    """Resolve fold-lazy source paths and validation-row filter policy for FNC feature reads."""
    if feature_source_paths:
        source_paths = tuple(path.expanduser().resolve() for path in feature_source_paths)
    else:
        source_paths = resolve_fold_lazy_source_paths(
            client,
            data_version,
            dataset_variant=dataset_variant,
            full_data_path=full_data_path,
            dataset_scope=dataset_scope,
            data_root=data_root,
        )
    include_validation_only = full_data_path is None and dataset_scope == "train_plus_validation"
    return source_paths, include_validation_only


def attach_fnc_features(
    predictions: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    era_col: str,
    id_col: str,
) -> pd.DataFrame:
    """Attach feature neutralizer columns to prediction rows by id and validate era alignment."""
    if id_col not in predictions.columns:
        raise TrainingDataError(f"training_predictions_missing_columns:{id_col}")

    preds = predictions.set_index(id_col)
    features = feature_frame
    if features.index.name != id_col:
        if id_col in features.columns:
            features = features.set_index(id_col)
        else:
            raise TrainingDataError(f"training_fnc_missing_id_col:{id_col}")

    common_ids = preds.index.intersection(features.index)
    if common_ids.empty:
        raise TrainingDataError("training_fnc_no_overlapping_ids")
    if len(common_ids) != len(preds.index):
        raise TrainingDataError(
            "training_fnc_partial_id_overlap:"
            f"predictions={len(preds.index)},features={len(features.index)},overlap={len(common_ids)}"
        )

    preds = preds.loc[common_ids]
    features = features.loc[common_ids]

    if era_col in features.columns and not np.array_equal(
        features[era_col].astype(str).to_numpy(),
        preds[era_col].astype(str).to_numpy(),
    ):
        raise TrainingDataError("training_fnc_eras_misaligned")

    resolved_feature_cols = [str(col) for col in feature_cols]
    missing_features = [col for col in resolved_feature_cols if col not in features.columns]
    if missing_features:
        raise TrainingDataError(f"training_fnc_missing_feature_cols:{','.join(missing_features)}")

    # Attach the full neutralizer block in one shot to avoid pandas fragmentation warnings
    # from repeated internal column insertions.
    if any(col in preds.columns for col in resolved_feature_cols):
        preds = preds.drop(columns=[col for col in resolved_feature_cols if col in preds.columns])
    feature_block = features[resolved_feature_cols]
    attached = pd.concat([preds, feature_block], axis=1)
    return attached.reset_index()


def attach_scoring_targets(
    predictions: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    target_cols: Sequence[str],
    era_col: str,
    id_col: str,
) -> pd.DataFrame:
    """Attach target columns to prediction rows by id and validate full coverage."""
    if id_col not in predictions.columns:
        raise TrainingDataError(f"training_predictions_missing_columns:{id_col}")

    preds = predictions.set_index(id_col)
    targets = target_frame
    if targets.index.name != id_col:
        if id_col in targets.columns:
            targets = targets.set_index(id_col)
        else:
            raise TrainingDataError(f"training_scoring_target_missing_id_col:{id_col}")

    common_ids = preds.index.intersection(targets.index)
    if common_ids.empty:
        raise TrainingDataError("training_scoring_target_no_overlapping_ids")
    if len(common_ids) != len(preds.index):
        raise TrainingDataError(
            "training_scoring_target_partial_id_overlap:"
            f"predictions={len(preds.index)},targets={len(targets.index)},overlap={len(common_ids)}"
        )

    preds = preds.loc[common_ids]
    targets = targets.loc[common_ids]

    if era_col in targets.columns and not np.array_equal(
        targets[era_col].astype(str).to_numpy(),
        preds[era_col].astype(str).to_numpy(),
    ):
        raise TrainingDataError("training_scoring_target_eras_misaligned")

    resolved_target_cols = [str(col) for col in target_cols]
    missing_targets = [col for col in resolved_target_cols if col not in targets.columns]
    if missing_targets:
        raise TrainingDataError(f"training_scoring_target_missing_cols:{','.join(missing_targets)}")

    if any(col in preds.columns for col in resolved_target_cols):
        preds = preds.drop(columns=[col for col in resolved_target_cols if col in preds.columns])
    target_block = targets[resolved_target_cols]
    attached = pd.concat([preds, target_block], axis=1)
    return attached.reset_index()


def _prepare_predictions_for_scoring(
    predictions: pd.DataFrame,
    *,
    pred_cols: Sequence[str],
    target_cols: Sequence[str],
    era_col: str,
    id_col: str = "id",
) -> pd.DataFrame:
    """Validate and subset prediction rows needed for metric calculations."""
    required_cols: list[str] = []
    for col in [era_col, *target_cols, *pred_cols]:
        if col not in required_cols:
            required_cols.append(col)

    if id_col in predictions.columns and id_col not in required_cols:
        required_cols.append(id_col)

    missing = [col for col in required_cols if col not in predictions.columns]
    if missing:
        raise TrainingDataError(f"training_predictions_missing_columns:{','.join(missing)}")

    return predictions[required_cols]


def summarize_prediction_file(
    predictions_path: str | Path,
    pred_cols: Sequence[object],
    target_col: str,
    era_col: str = "era",
) -> dict[str, pd.DataFrame]:
    """Summarize per-era correlation for one prediction file."""
    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    predictions = _prepare_predictions_for_scoring(
        _read_table(Path(predictions_path)),
        pred_cols=resolved_pred_cols,
        target_cols=[target_col],
        era_col=era_col,
    )
    per_era = per_era_corr(predictions, resolved_pred_cols, target_col, era_col)
    return {"corr": summarize_scores(per_era)}


def _summarize_prediction_file_with_scores_materialized(
    predictions_path: str | Path,
    pred_cols: Sequence[object],
    target_col: str,
    scoring_target_cols: Sequence[str],
    data_version: str,
    *,
    dataset_variant: str,
    client: TrainingDataClient,
    feature_set: str,
    feature_source_paths: Sequence[Path] | None = None,
    full_data_path: str | Path | None = None,
    dataset_scope: str = "train_plus_validation",
    benchmark_model: str = DEFAULT_BENCHMARK_MODEL,
    benchmark_data_path: str | Path | None = None,
    meta_model_data_path: str | Path | None = None,
    meta_model_col: str = DEFAULT_META_MODEL_COL,
    era_col: str = "era",
    id_col: str = "id",
    data_root: Path = DEFAULT_DATASETS_DIR,
    era_chunk_size: int = 64,
    scoring_policy: ResolvedScoringPolicy | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Summarize prediction metrics and build per-run scoring provenance."""
    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    resolved_scoring_targets = _resolve_scoring_targets(
        target_col=target_col,
        scoring_target_cols=scoring_target_cols,
    )
    target_aliases = _validate_target_aliases(resolved_scoring_targets)

    predictions_path_resolved = Path(predictions_path).expanduser().resolve()
    prediction_frame = _read_table(predictions_path_resolved)
    prediction_target_cols = [col for col in resolved_scoring_targets if col in prediction_frame.columns]
    predictions = _prepare_predictions_for_scoring(
        prediction_frame,
        pred_cols=resolved_pred_cols,
        target_cols=([target_col] if target_col in prediction_target_cols else []) or prediction_target_cols,
        era_col=era_col,
        id_col=id_col,
    )

    if id_col not in predictions.columns:
        raise TrainingDataError(f"training_predictions_missing_columns:{id_col}")

    resolved_scoring_policy = _resolve_scoring_policy(scoring_policy)
    feature_neutral_metrics_enabled = resolved_scoring_policy.include_feature_neutral_metrics
    missing_scoring_targets = [col for col in resolved_scoring_targets if col not in predictions.columns]
    fnc_source_paths: tuple[Path, ...] = ()
    fnc_include_validation_only = False
    if missing_scoring_targets or feature_neutral_metrics_enabled:
        fnc_source_paths, fnc_include_validation_only = resolve_fnc_source_paths(
            client=client,
            data_version=data_version,
            dataset_variant=dataset_variant,
            feature_source_paths=feature_source_paths,
            full_data_path=full_data_path,
            dataset_scope=dataset_scope,
            data_root=data_root,
        )
    target_join_stats: dict[str, int | float] = {}
    if missing_scoring_targets:
        target_keys = load_fold_data_lazy(
            fnc_source_paths,
            eras=sorted(set(predictions[era_col].tolist()), key=_era_sort_key),
            columns=[era_col, id_col, *missing_scoring_targets],
            era_col=era_col,
            id_col=id_col,
            include_validation_only=fnc_include_validation_only,
        )
        target_join_stats = validate_join_source_coverage(
            predictions,
            target_keys,
            source_name="scoring_target",
            era_col=era_col,
            id_col=id_col,
            include_missing_counts=True,
        )
        predictions = attach_scoring_targets(
            predictions,
            target_keys,
            target_cols=missing_scoring_targets,
            era_col=era_col,
            id_col=id_col,
        )

    summaries: dict[str, pd.DataFrame] = {}
    for scoring_target_col in resolved_scoring_targets:
        corr_key = _target_metric_key(
            metric_name="corr",
            target_col=scoring_target_col,
            native_target_col=target_col,
            aliases=target_aliases,
        )
        summaries[corr_key] = summarize_scores(
            per_era_corr(predictions, resolved_pred_cols, scoring_target_col, era_col)
        )

    fnc_preflight: dict[str, int | float] = {}
    fnc_overlap_rows = 0
    fnc_overlap_eras: set[object] = set()
    resolved_feature_cols: list[str] = []
    if feature_neutral_metrics_enabled:
        resolved_feature_cols = resolve_fnc_feature_columns(
            client=client,
            data_version=data_version,
            dataset_variant=dataset_variant,
            scoring_policy=resolved_scoring_policy,
            data_root=data_root,
        )
        fnc_preflight = validate_join_source_coverage(
            predictions,
            load_fold_data_lazy(
                fnc_source_paths,
                eras=sorted(set(predictions[era_col].tolist()), key=_era_sort_key),
                columns=[era_col, id_col],
                era_col=era_col,
                id_col=id_col,
                include_validation_only=fnc_include_validation_only,
            ),
            source_name="fnc",
            era_col=era_col,
            id_col=id_col,
        )
        fnc_chunks: dict[str, list[pd.DataFrame]] = {target: [] for target in resolved_scoring_targets}
        feature_exposure_chunks: list[pd.DataFrame] = []
        max_feature_exposure_chunks: list[pd.DataFrame] = []
        fnc_eras = sorted(set(predictions[era_col].tolist()), key=_era_sort_key)
        for era_chunk in _chunked(fnc_eras, era_chunk_size):
            chunk_predictions = predictions[predictions[era_col].isin(era_chunk)]
            if chunk_predictions.empty:
                continue

            feature_chunk = load_fold_data_lazy(
                fnc_source_paths,
                eras=list(era_chunk),
                columns=[era_col, *resolved_feature_cols, id_col],
                era_col=era_col,
                id_col=id_col,
                include_validation_only=fnc_include_validation_only,
            )
            predictions_for_fnc = attach_fnc_features(
                chunk_predictions,
                feature_chunk,
                feature_cols=resolved_feature_cols,
                era_col=era_col,
                id_col=id_col,
            )
            fnc_overlap_rows += int(len(predictions_for_fnc))
            fnc_overlap_eras.update(predictions_for_fnc[era_col].tolist())
            for scoring_target_col in resolved_scoring_targets:
                fnc_chunks[scoring_target_col].append(
                    per_era_fnc(
                        predictions_for_fnc,
                        resolved_pred_cols,
                        resolved_feature_cols,
                        scoring_target_col,
                        era_col=era_col,
                    )
                )
            feature_exposure_chunk, max_feature_exposure_chunk = per_era_feature_exposure(
                predictions_for_fnc,
                resolved_pred_cols,
                resolved_feature_cols,
                era_col=era_col,
            )
            feature_exposure_chunks.append(feature_exposure_chunk)
            max_feature_exposure_chunks.append(max_feature_exposure_chunk)

        if not any(fnc_chunks.values()):
            raise TrainingDataError("training_fnc_no_overlapping_ids")
        for scoring_target_col, chunks in fnc_chunks.items():
            if not chunks:
                continue
            fnc_per_era = cast(pd.DataFrame, _sort_era_index(pd.concat(chunks, axis=0)))
            fnc_key = _target_metric_key(
                metric_name="fnc",
                target_col=scoring_target_col,
                native_target_col=target_col,
                aliases=target_aliases,
            )
            summaries[fnc_key] = summarize_scores(fnc_per_era)
        if not feature_exposure_chunks or not max_feature_exposure_chunks:
            raise TrainingDataError("training_feature_exposure_no_overlapping_ids")
        feature_exposure_per_era = cast(
            pd.DataFrame, _sort_era_index(pd.concat(feature_exposure_chunks, axis=0))
        )
        max_feature_exposure_per_era = cast(
            pd.DataFrame, _sort_era_index(pd.concat(max_feature_exposure_chunks, axis=0))
        )
        summaries["feature_exposure"] = summarize_scores(feature_exposure_per_era)
        summaries["max_feature_exposure"] = summarize_scores(max_feature_exposure_per_era)

    if benchmark_data_path is not None:
        benchmark_path = resolve_data_path(benchmark_data_path, data_root=data_root)
    else:
        benchmark_path = resolve_benchmark_predictions_path(
            client,
            data_version,
            dataset_variant=dataset_variant,
            data_root=data_root,
        )
    benchmark, benchmark_col = load_benchmark_predictions_from_path(
        benchmark_path,
        benchmark_model,
        era_col=era_col,
        id_col=id_col,
        data_root=data_root,
    )
    benchmark_stats = validate_join_source_coverage(
        predictions,
        benchmark,
        source_name="benchmark",
        era_col=era_col,
        id_col=id_col,
        min_overlap_ratio=resolved_scoring_policy.benchmark_min_overlap_ratio,
        include_missing_counts=True,
    )

    predictions_for_benchmark = attach_benchmark_predictions(
        predictions,
        benchmark,
        benchmark_col,
        era_col=era_col,
        id_col=id_col,
        min_overlap_ratio=resolved_scoring_policy.benchmark_min_overlap_ratio,
    )

    per_era = per_era_bmc(predictions_for_benchmark, resolved_pred_cols, benchmark_col, target_col, era_col)
    benchmark_corr = per_era_reference_corr(
        predictions_for_benchmark,
        resolved_pred_cols,
        benchmark_col,
        era_col=era_col,
    )

    bmc_summary = summarize_scores(per_era)
    benchmark_corr_mean = benchmark_corr.mean()
    for col in bmc_summary.index:
        bmc_summary.loc[col, "avg_corr_with_benchmark"] = float(
            benchmark_corr_mean.get(col, np.nan)
        )
    summaries["bmc"] = bmc_summary

    per_era_recent = cast(pd.DataFrame, _last_n_eras(per_era, 200))
    bmc_recent_summary = summarize_scores(per_era_recent)
    benchmark_corr_recent = cast(pd.DataFrame, _last_n_eras(benchmark_corr, 200))
    benchmark_corr_recent_mean = benchmark_corr_recent.mean()
    for col in bmc_recent_summary.index:
        bmc_recent_summary.loc[col, "avg_corr_with_benchmark"] = float(
            benchmark_corr_recent_mean.get(col, np.nan)
        )
    summaries["bmc_last_200_eras"] = bmc_recent_summary

    meta_model, resolved_meta_model_col, meta_model_path = load_meta_model_predictions(
        client,
        data_version,
        dataset_variant=dataset_variant,
        meta_model_col=meta_model_col,
        meta_model_data_path=meta_model_data_path,
        era_col=era_col,
        id_col=id_col,
        data_root=data_root,
    )
    preflight_meta_stats = validate_join_source_coverage(
        predictions,
        meta_model,
        source_name="meta",
        era_col=era_col,
        id_col=id_col,
        min_overlap_ratio=0.0,
        include_missing_counts=True,
        allow_zero_overlap=True,
    )
    meta_metrics_emitted = False
    meta_metrics_reason: str | None = None
    if int(preflight_meta_stats["meta_overlap_rows"]) > 0:
        predictions_for_meta, meta_stats = attach_meta_model_predictions(
            predictions,
            meta_model,
            resolved_meta_model_col,
            era_col=era_col,
            id_col=id_col,
            min_overlap_ratio=0.0,
        )
        for scoring_target_col in resolved_scoring_targets:
            mmc_key = _target_metric_key(
                metric_name="mmc",
                target_col=scoring_target_col,
                native_target_col=target_col,
                aliases=target_aliases,
            )
            summaries[mmc_key] = summarize_scores(
                per_era_mmc(
                    predictions_for_meta,
                    resolved_pred_cols,
                    resolved_meta_model_col,
                    scoring_target_col,
                    era_col,
                )
            )

        cwmm_per_era = per_era_cwmm(
            predictions_for_meta,
            resolved_pred_cols,
            resolved_meta_model_col,
            era_col=era_col,
        )
        summaries["cwmm"] = summarize_scores(cwmm_per_era)
        meta_metrics_emitted = True
    else:
        meta_stats = {
            "predictions_rows": int(len(predictions)),
            "meta_source_rows": int(preflight_meta_stats["meta_source_rows"]),
            "meta_overlap_rows": 0,
            "meta_overlap_eras": 0,
            "meta_overlap_ratio": 0.0,
        }
        meta_metrics_reason = "no_meta_overlap"

    provenance: dict[str, object] = {
        "schema_version": "2",
        "data_version": data_version,
        "dataset_variant": dataset_variant,
        "columns": {
            "prediction_cols": resolved_pred_cols,
            "target_col": target_col,
            "scoring_target_cols": resolved_scoring_targets,
            "target_metric_aliases": target_aliases,
            "id_col": id_col,
            "era_col": era_col,
            "training_feature_set": feature_set,
            "meta_model_col": resolved_meta_model_col,
            "benchmark_col": benchmark_col,
        },
        "policy": {
            "fnc_feature_set": resolved_scoring_policy.fnc_feature_set,
            "fnc_target_policy": resolved_scoring_policy.fnc_target_policy,
            "benchmark_min_overlap_ratio": resolved_scoring_policy.benchmark_min_overlap_ratio,
            "include_feature_neutral_metrics": feature_neutral_metrics_enabled,
        },
        "meta_metrics": {
            "emitted": meta_metrics_emitted,
            "reason": meta_metrics_reason,
        },
        "sources": {
            "predictions": _file_fingerprint(predictions_path_resolved),
            "meta_model": _file_fingerprint(meta_model_path),
            "benchmark": _file_fingerprint(benchmark_path),
        },
        "joins": {
            "predictions_rows": int(len(predictions)),
            "predictions_eras": int(predictions[era_col].nunique()) if era_col in predictions.columns else 0,
            **target_join_stats,
            **benchmark_stats,
            **preflight_meta_stats,
            **meta_stats,
        },
    }
    if feature_neutral_metrics_enabled:
        provenance_columns = cast(dict[str, object], provenance["columns"])
        provenance_columns["fnc_feature_set"] = resolved_scoring_policy.fnc_feature_set
        provenance_columns["fnc_feature_count"] = len(resolved_feature_cols)
        provenance_sources = cast(dict[str, object], provenance["sources"])
        provenance_sources["fnc_feature_sources"] = [str(path) for path in fnc_source_paths]
        provenance_joins = cast(dict[str, object], provenance["joins"])
        provenance_joins["fnc_overlap_rows"] = fnc_overlap_rows
        provenance_joins["fnc_overlap_eras"] = int(len(fnc_overlap_eras))
        provenance_joins.update(fnc_preflight)
    return summaries, provenance


def summarize_prediction_file_with_scores(
    predictions_path: str | Path,
    pred_cols: Sequence[object],
    target_col: str,
    scoring_target_cols: Sequence[str],
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    client: TrainingDataClient,
    feature_set: str = "small",
    feature_source_paths: Sequence[Path] | None = None,
    full_data_path: str | Path | None = None,
    dataset_scope: str = "train_plus_validation",
    benchmark_model: str = DEFAULT_BENCHMARK_MODEL,
    benchmark_data_path: str | Path | None = None,
    meta_model_data_path: str | Path | None = None,
    meta_model_col: str = DEFAULT_META_MODEL_COL,
    era_col: str = "era",
    id_col: str = "id",
    data_root: Path = DEFAULT_DATASETS_DIR,
    scoring_mode: str = "materialized",
    era_chunk_size: int = 64,
    scoring_policy: ResolvedScoringPolicy | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Summarize prediction metrics with explicit scoring execution mode."""
    resolved_scoring_policy = _resolve_scoring_policy(scoring_policy)
    if scoring_mode == "materialized":
        summaries, provenance = _summarize_prediction_file_with_scores_materialized(
            predictions_path,
            pred_cols,
            target_col,
            scoring_target_cols,
            data_version,
            dataset_variant=dataset_variant,
            client=client,
            feature_set=feature_set,
            feature_source_paths=feature_source_paths,
            full_data_path=full_data_path,
            dataset_scope=dataset_scope,
            benchmark_model=benchmark_model,
            benchmark_data_path=benchmark_data_path,
            meta_model_data_path=meta_model_data_path,
            meta_model_col=meta_model_col,
            era_col=era_col,
            id_col=id_col,
            data_root=data_root,
            era_chunk_size=era_chunk_size,
            scoring_policy=resolved_scoring_policy,
        )
        provenance["execution"] = {
            "requested_scoring_mode": scoring_mode,
            "effective_scoring_mode": "materialized",
            "era_chunk_size": era_chunk_size,
        }
        return summaries, provenance

    if scoring_mode == "era_stream":
        return _summarize_prediction_file_with_scores_era_stream(
            predictions_path,
            pred_cols,
            target_col,
            scoring_target_cols,
            data_version,
            dataset_variant=dataset_variant,
            client=client,
            feature_set=feature_set,
            feature_source_paths=feature_source_paths,
            full_data_path=full_data_path,
            dataset_scope=dataset_scope,
            benchmark_model=benchmark_model,
            benchmark_data_path=benchmark_data_path,
            meta_model_data_path=meta_model_data_path,
            meta_model_col=meta_model_col,
            era_col=era_col,
            id_col=id_col,
            data_root=data_root,
            era_chunk_size=era_chunk_size,
            scoring_policy=resolved_scoring_policy,
        )

    raise TrainingMetricsError("training_metrics_scoring_mode_invalid")


def _summarize_prediction_file_with_scores_era_stream(
    predictions_path: str | Path,
    pred_cols: Sequence[object],
    target_col: str,
    scoring_target_cols: Sequence[str],
    data_version: str,
    *,
    dataset_variant: str,
    client: TrainingDataClient,
    feature_set: str,
    feature_source_paths: Sequence[Path] | None = None,
    full_data_path: str | Path | None = None,
    dataset_scope: str = "train_plus_validation",
    benchmark_model: str = DEFAULT_BENCHMARK_MODEL,
    benchmark_data_path: str | Path | None = None,
    meta_model_data_path: str | Path | None = None,
    meta_model_col: str = DEFAULT_META_MODEL_COL,
    era_col: str = "era",
    id_col: str = "id",
    data_root: Path = DEFAULT_DATASETS_DIR,
    era_chunk_size: int,
    scoring_policy: ResolvedScoringPolicy | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if era_chunk_size < 1:
        raise TrainingMetricsError("training_metrics_era_chunk_size_invalid")

    resolved_pred_cols = [str(col) for col in _as_list(pred_cols)]
    resolved_scoring_targets = _resolve_scoring_targets(
        target_col=target_col,
        scoring_target_cols=scoring_target_cols,
    )
    target_aliases = _validate_target_aliases(resolved_scoring_targets)
    predictions_path_resolved = Path(predictions_path).expanduser().resolve()
    if predictions_path_resolved.suffix.lower() != ".parquet":
        raise TrainingDataError(f"training_metrics_era_stream_requires_parquet:{predictions_path_resolved}")
    prediction_columns = _table_columns(predictions_path_resolved)
    native_prediction_target_cols = [col for col in resolved_scoring_targets if col in prediction_columns]
    required_cols = [era_col, *native_prediction_target_cols, *resolved_pred_cols, id_col]
    missing = [col for col in required_cols if col not in prediction_columns]
    if missing:
        raise TrainingDataError(f"training_predictions_missing_columns:{','.join(missing)}")

    if benchmark_data_path is not None:
        benchmark_path = resolve_data_path(benchmark_data_path, data_root=data_root)
    else:
        benchmark_path = resolve_benchmark_predictions_path(
            client,
            data_version,
            dataset_variant=dataset_variant,
            data_root=data_root,
        )
    benchmark_columns = _table_columns(benchmark_path)
    benchmark_col = _resolve_benchmark_column(benchmark_columns, benchmark_model)

    meta_model_path = resolve_meta_model_path(
        client,
        data_version,
        dataset_variant=dataset_variant,
        meta_model_data_path=meta_model_data_path,
        data_root=data_root,
    )
    meta_columns = _table_columns(meta_model_path)
    resolved_meta_model_col = _resolve_meta_model_column(
        meta_columns,
        meta_model_col,
        era_col=era_col,
        id_col=id_col,
    )
    resolved_scoring_policy = _resolve_scoring_policy(scoring_policy)
    feature_neutral_metrics_enabled = resolved_scoring_policy.include_feature_neutral_metrics
    missing_scoring_targets = [col for col in resolved_scoring_targets if col not in prediction_columns]
    fnc_source_paths: tuple[Path, ...] = ()
    fnc_include_validation_only = False
    if missing_scoring_targets or feature_neutral_metrics_enabled:
        fnc_source_paths, fnc_include_validation_only = resolve_fnc_source_paths(
            client=client,
            data_version=data_version,
            dataset_variant=dataset_variant,
            feature_source_paths=feature_source_paths,
            full_data_path=full_data_path,
            dataset_scope=dataset_scope,
            data_root=data_root,
        )

    era_values = _parquet_unique_values(predictions_path_resolved, column=era_col)
    if not era_values:
        raise TrainingDataError("training_predictions_missing_rows")
    prediction_keys = read_join_source_keys(
        predictions_path_resolved,
        id_col=id_col,
        era_col=era_col,
        source_name="predictions",
    )
    resolved_feature_cols: list[str] = []
    fnc_keys = pd.DataFrame()
    if feature_neutral_metrics_enabled:
        resolved_feature_cols = resolve_fnc_feature_columns(
            client=client,
            data_version=data_version,
            dataset_variant=dataset_variant,
            scoring_policy=resolved_scoring_policy,
            data_root=data_root,
        )
        fnc_keys = load_fold_data_lazy(
            fnc_source_paths,
            eras=era_values,
            columns=[era_col, id_col],
            era_col=era_col,
            id_col=id_col,
            include_validation_only=fnc_include_validation_only,
        )
    benchmark_keys = read_join_source_keys(
        benchmark_path,
        id_col=id_col,
        era_col=era_col,
        extra_cols=[benchmark_col],
        source_name="benchmark",
    )
    meta_keys = read_join_source_keys(
        meta_model_path,
        id_col=id_col,
        era_col=era_col,
        extra_cols=[resolved_meta_model_col],
        source_name="meta_model",
    )
    target_join_stats: dict[str, int | float] = {}
    if missing_scoring_targets:
        target_keys = load_fold_data_lazy(
            fnc_source_paths,
            eras=era_values,
            columns=[era_col, id_col, *missing_scoring_targets],
            era_col=era_col,
            id_col=id_col,
            include_validation_only=fnc_include_validation_only,
        )
        target_join_stats = validate_join_source_coverage(
            prediction_keys,
            target_keys,
            source_name="scoring_target",
            era_col=era_col,
            id_col=id_col,
            include_missing_counts=True,
        )
    fnc_preflight: dict[str, int | float] = {}
    if feature_neutral_metrics_enabled:
        fnc_preflight = validate_join_source_coverage(
            prediction_keys,
            fnc_keys,
            source_name="fnc",
            era_col=era_col,
            id_col=id_col,
        )
    benchmark_stats = validate_join_source_coverage(
        prediction_keys,
        benchmark_keys,
        source_name="benchmark",
        era_col=era_col,
        id_col=id_col,
        min_overlap_ratio=resolved_scoring_policy.benchmark_min_overlap_ratio,
        include_missing_counts=True,
    )
    preflight_meta_stats = validate_join_source_coverage(
        prediction_keys,
        meta_keys,
        source_name="meta",
        era_col=era_col,
        id_col=id_col,
        min_overlap_ratio=0.0,
        include_missing_counts=True,
        allow_zero_overlap=True,
    )

    corr_chunks: dict[str, list[pd.DataFrame]] = {target: [] for target in resolved_scoring_targets}
    fnc_chunks: dict[str, list[pd.DataFrame]] = {target: [] for target in resolved_scoring_targets}
    feature_exposure_chunks: list[pd.DataFrame] = []
    max_feature_exposure_chunks: list[pd.DataFrame] = []
    bmc_chunks: list[pd.DataFrame] = []
    bmc_corr_chunks: list[pd.DataFrame] = []
    mmc_chunks: dict[str, list[pd.DataFrame]] = {target: [] for target in resolved_scoring_targets}
    cwmm_chunks: list[pd.DataFrame] = []
    fnc_overlap_rows = 0
    fnc_overlap_eras: set[object] = set()
    benchmark_overlap_rows = 0
    benchmark_overlap_eras: set[object] = set()
    meta_overlap_rows = 0
    meta_overlap_eras: set[object] = set()
    predictions_rows = 0

    for era_chunk in _chunked(era_values, era_chunk_size):
        chunk_predictions = _read_parquet_era_chunk(
            predictions_path_resolved,
            era_values=era_chunk,
            columns=required_cols,
            era_col=era_col,
            id_col=id_col,
        )
        if id_col not in chunk_predictions.columns and chunk_predictions.index.name == id_col:
            chunk_predictions = chunk_predictions.reset_index()
        if id_col not in chunk_predictions.columns:
            raise TrainingDataError(f"training_predictions_missing_columns:{id_col}")
        if chunk_predictions.empty:
            continue

        predictions_rows += int(len(chunk_predictions))
        if missing_scoring_targets:
            target_chunk = load_fold_data_lazy(
                fnc_source_paths,
                eras=list(era_chunk),
                columns=[era_col, id_col, *missing_scoring_targets],
                era_col=era_col,
                id_col=id_col,
                include_validation_only=fnc_include_validation_only,
            )
            chunk_predictions = attach_scoring_targets(
                chunk_predictions,
                target_chunk,
                target_cols=missing_scoring_targets,
                era_col=era_col,
                id_col=id_col,
            )
        for scoring_target_col in resolved_scoring_targets:
            corr_chunks[scoring_target_col].append(
                per_era_corr(chunk_predictions, resolved_pred_cols, scoring_target_col, era_col)
            )

        if feature_neutral_metrics_enabled:
            fnc_feature_chunk = load_fold_data_lazy(
                fnc_source_paths,
                eras=list(era_chunk),
                columns=[era_col, *resolved_feature_cols, id_col],
                era_col=era_col,
                id_col=id_col,
                include_validation_only=fnc_include_validation_only,
            )
            predictions_for_fnc = attach_fnc_features(
                chunk_predictions,
                fnc_feature_chunk,
                feature_cols=resolved_feature_cols,
                era_col=era_col,
                id_col=id_col,
            )
            fnc_overlap_rows += int(len(predictions_for_fnc))
            fnc_overlap_eras.update(predictions_for_fnc[era_col].tolist())
            for scoring_target_col in resolved_scoring_targets:
                fnc_chunks[scoring_target_col].append(
                    per_era_fnc(
                        predictions_for_fnc,
                        resolved_pred_cols,
                        resolved_feature_cols,
                        scoring_target_col,
                        era_col=era_col,
                    )
                )
            feature_exposure_chunk, max_feature_exposure_chunk = per_era_feature_exposure(
                predictions_for_fnc,
                resolved_pred_cols,
                resolved_feature_cols,
                era_col=era_col,
            )
            feature_exposure_chunks.append(feature_exposure_chunk)
            max_feature_exposure_chunks.append(max_feature_exposure_chunk)

        benchmark_chunk = _read_parquet_era_chunk(
            benchmark_path,
            era_values=era_chunk,
            columns=[benchmark_col, id_col, era_col],
            era_col=era_col,
            id_col=id_col,
        )
        try:
            predictions_for_benchmark = attach_benchmark_predictions(
                chunk_predictions,
                benchmark_chunk,
                benchmark_col,
                era_col=era_col,
                id_col=id_col,
                min_overlap_ratio=resolved_scoring_policy.benchmark_min_overlap_ratio,
            )
        except TrainingDataError as exc:
            if str(exc) != "training_benchmark_no_overlapping_ids":
                raise
            continue
        benchmark_overlap_rows += int(len(predictions_for_benchmark))
        benchmark_overlap_eras.update(predictions_for_benchmark[era_col].tolist())
        bmc_chunks.append(
            per_era_bmc(
                predictions_for_benchmark,
                resolved_pred_cols,
                benchmark_col,
                target_col,
                era_col,
            )
        )
        bmc_corr_chunks.append(
            per_era_reference_corr(
                predictions_for_benchmark,
                resolved_pred_cols,
                benchmark_col,
                era_col=era_col,
            )
        )

        meta_chunk = _read_parquet_era_chunk(
            meta_model_path,
            era_values=era_chunk,
            columns=[resolved_meta_model_col, id_col, era_col],
            era_col=era_col,
            id_col=id_col,
        )
        try:
            predictions_for_meta, meta_stats = attach_meta_model_predictions(
                chunk_predictions,
                meta_chunk,
                resolved_meta_model_col,
                era_col=era_col,
                id_col=id_col,
                min_overlap_ratio=0.0,
            )
        except TrainingDataError as exc:
            if str(exc) != "training_meta_model_no_overlapping_ids":
                raise
            continue
        meta_overlap_rows += int(meta_stats["meta_overlap_rows"])
        meta_overlap_eras.update(predictions_for_meta[era_col].tolist())
        for scoring_target_col in resolved_scoring_targets:
            mmc_chunks[scoring_target_col].append(
                per_era_mmc(
                    predictions_for_meta,
                    resolved_pred_cols,
                    resolved_meta_model_col,
                    scoring_target_col,
                    era_col=era_col,
                )
            )
        cwmm_chunks.append(
            per_era_cwmm(
                predictions_for_meta,
                resolved_pred_cols,
                resolved_meta_model_col,
                era_col=era_col,
            )
        )

    if not any(corr_chunks.values()):
        raise TrainingDataError("training_metrics_era_stream_no_chunks")
    if feature_neutral_metrics_enabled and not any(fnc_chunks.values()):
        raise TrainingDataError("training_fnc_no_overlapping_ids")
    if feature_neutral_metrics_enabled and (not feature_exposure_chunks or not max_feature_exposure_chunks):
        raise TrainingDataError("training_feature_exposure_no_overlapping_ids")
    if not bmc_chunks:
        raise TrainingDataError("training_benchmark_no_overlapping_ids")
    meta_metrics_emitted = any(mmc_chunks.values()) and bool(cwmm_chunks)
    meta_metrics_reason: str | None = None
    if not meta_metrics_emitted:
        meta_metrics_reason = "no_meta_overlap"

    bmc_per_era = cast(pd.DataFrame, _sort_era_index(pd.concat(bmc_chunks, axis=0)))
    bmc_corr_per_era = cast(pd.DataFrame, _sort_era_index(pd.concat(bmc_corr_chunks, axis=0)))
    cwmm_per_era: pd.DataFrame | None = None
    if meta_metrics_emitted:
        cwmm_per_era = cast(pd.DataFrame, _sort_era_index(pd.concat(cwmm_chunks, axis=0)))

    summaries: dict[str, pd.DataFrame] = {}
    if feature_neutral_metrics_enabled:
        feature_exposure_per_era = cast(
            pd.DataFrame, _sort_era_index(pd.concat(feature_exposure_chunks, axis=0))
        )
        max_feature_exposure_per_era = cast(
            pd.DataFrame, _sort_era_index(pd.concat(max_feature_exposure_chunks, axis=0))
        )
        for scoring_target_col, chunks in fnc_chunks.items():
            if not chunks:
                continue
            fnc_per_era = cast(pd.DataFrame, _sort_era_index(pd.concat(chunks, axis=0)))
            fnc_key = _target_metric_key(
                metric_name="fnc",
                target_col=scoring_target_col,
                native_target_col=target_col,
                aliases=target_aliases,
            )
            summaries[fnc_key] = summarize_scores(fnc_per_era)
        summaries["feature_exposure"] = summarize_scores(feature_exposure_per_era)
        summaries["max_feature_exposure"] = summarize_scores(max_feature_exposure_per_era)
    for scoring_target_col, chunks in corr_chunks.items():
        corr_per_era = cast(pd.DataFrame, _sort_era_index(pd.concat(chunks, axis=0)))
        corr_key = _target_metric_key(
            metric_name="corr",
            target_col=scoring_target_col,
            native_target_col=target_col,
            aliases=target_aliases,
        )
        summaries[corr_key] = summarize_scores(corr_per_era)

    bmc_summary = summarize_scores(bmc_per_era)
    bmc_corr_mean = bmc_corr_per_era.mean()
    for col in bmc_summary.index:
        bmc_summary.loc[col, "avg_corr_with_benchmark"] = float(bmc_corr_mean.get(col, np.nan))
    summaries["bmc"] = bmc_summary

    bmc_recent = cast(pd.DataFrame, _last_n_eras(bmc_per_era, 200))
    bmc_corr_recent = cast(pd.DataFrame, _last_n_eras(bmc_corr_per_era, 200))
    bmc_recent_summary = summarize_scores(bmc_recent)
    bmc_corr_recent_mean = bmc_corr_recent.mean()
    for col in bmc_recent_summary.index:
        bmc_recent_summary.loc[col, "avg_corr_with_benchmark"] = float(
            bmc_corr_recent_mean.get(col, np.nan)
        )
    summaries["bmc_last_200_eras"] = bmc_recent_summary

    if meta_metrics_emitted:
        for scoring_target_col, chunks in mmc_chunks.items():
            if not chunks:
                continue
            mmc_per_era = cast(pd.DataFrame, _sort_era_index(pd.concat(chunks, axis=0)))
            mmc_key = _target_metric_key(
                metric_name="mmc",
                target_col=scoring_target_col,
                native_target_col=target_col,
                aliases=target_aliases,
            )
            summaries[mmc_key] = summarize_scores(mmc_per_era)
        if cwmm_per_era is None:
            raise TrainingMetricsError("training_metrics_unexpected_per_era_type")
        summaries["cwmm"] = summarize_scores(cwmm_per_era)

    provenance: dict[str, object] = {
        "schema_version": "2",
        "data_version": data_version,
        "dataset_variant": dataset_variant,
        "columns": {
            "prediction_cols": resolved_pred_cols,
            "target_col": target_col,
            "scoring_target_cols": resolved_scoring_targets,
            "target_metric_aliases": target_aliases,
            "id_col": id_col,
            "era_col": era_col,
            "training_feature_set": feature_set,
            "meta_model_col": resolved_meta_model_col,
            "benchmark_col": benchmark_col,
        },
        "meta_metrics": {
            "emitted": meta_metrics_emitted,
            "reason": meta_metrics_reason,
        },
        "policy": {
            "fnc_feature_set": resolved_scoring_policy.fnc_feature_set,
            "fnc_target_policy": resolved_scoring_policy.fnc_target_policy,
            "benchmark_min_overlap_ratio": resolved_scoring_policy.benchmark_min_overlap_ratio,
            "include_feature_neutral_metrics": feature_neutral_metrics_enabled,
        },
        "sources": {
            "predictions": _file_fingerprint(predictions_path_resolved),
            "meta_model": _file_fingerprint(meta_model_path),
            "benchmark": _file_fingerprint(benchmark_path),
        },
        "joins": {
            "predictions_rows": predictions_rows,
            "predictions_eras": int(len(era_values)),
            **target_join_stats,
            **benchmark_stats,
            **preflight_meta_stats,
            "benchmark_stream_overlap_rows": benchmark_overlap_rows,
            "benchmark_stream_overlap_eras": int(len(benchmark_overlap_eras)),
            "meta_overlap_rows": meta_overlap_rows,
            "meta_overlap_eras": int(len(meta_overlap_eras)),
            "meta_source_rows": _table_row_count(meta_model_path),
        },
        "execution": {
            "requested_scoring_mode": "era_stream",
            "effective_scoring_mode": "era_stream",
            "era_chunk_size": era_chunk_size,
        },
    }
    if feature_neutral_metrics_enabled:
        provenance_columns = cast(dict[str, object], provenance["columns"])
        provenance_columns["fnc_feature_set"] = resolved_scoring_policy.fnc_feature_set
        provenance_columns["fnc_feature_count"] = len(resolved_feature_cols)
        provenance_sources = cast(dict[str, object], provenance["sources"])
        provenance_sources["fnc_feature_sources"] = [str(path) for path in fnc_source_paths]
        provenance_joins = cast(dict[str, object], provenance["joins"])
        provenance_joins["fnc_overlap_rows"] = fnc_overlap_rows
        provenance_joins["fnc_overlap_eras"] = int(len(fnc_overlap_eras))
        provenance_joins.update(fnc_preflight)
    return summaries, provenance


def summarize_prediction_file_with_bmc(
    predictions_path: str | Path,
    pred_cols: Sequence[object],
    target_col: str,
    data_version: str,
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    client: TrainingDataClient,
    benchmark_model: str = DEFAULT_BENCHMARK_MODEL,
    benchmark_data_path: str | Path | None = None,
    era_col: str = "era",
    id_col: str = "id",
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> dict[str, pd.DataFrame]:
    """Backward-compatible subset: corr + bmc summaries only."""
    summaries, _ = summarize_prediction_file_with_scores(
        predictions_path,
        pred_cols,
        target_col,
        [target_col],
        data_version,
        dataset_variant=dataset_variant,
        client=client,
        benchmark_model=benchmark_model,
        benchmark_data_path=benchmark_data_path,
        era_col=era_col,
        id_col=id_col,
        data_root=data_root,
    )
    return {
        "corr": summaries["corr"],
        "bmc": summaries["bmc"],
        "bmc_last_200_eras": summaries["bmc_last_200_eras"],
    }


def _chunked(values: Sequence[object], chunk_size: int) -> list[list[object]]:
    chunks: list[list[object]] = []
    for idx in range(0, len(values), chunk_size):
        chunks.append(list(values[idx : idx + chunk_size]))
    return chunks


def _read_parquet_era_chunk(
    path: Path,
    *,
    era_values: Sequence[object],
    columns: list[str],
    era_col: str,
    id_col: str,
) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise TrainingDataError(f"training_metrics_era_stream_requires_parquet:{path}")
    try:
        ds = import_module("pyarrow.dataset")
    except ImportError as exc:
        raise TrainingDataError("training_data_lazy_backend_missing_pyarrow") from exc

    dataset = ds.dataset(str(path), format="parquet")
    available_columns = {str(name) for name in dataset.schema.names}
    if era_col in available_columns and id_col in available_columns:
        projected = [col for col in columns if col in available_columns]
        predicate = ds.field(era_col).isin(list(era_values))
        table = dataset.to_table(columns=projected or None, filter=predicate)
        frame = table.to_pandas()
        if "data_type" in frame.columns:
            frame = frame.drop(columns=["data_type"])
        if id_col in frame.columns:
            frame = frame.set_index(id_col)
        elif frame.index.name != id_col:
            frame.index.name = id_col
        return cast(pd.DataFrame, frame)

    # Some parquet tables persist `id` as the parquet index metadata, not a physical column.
    frame = _read_table(path)
    if era_col not in frame.columns:
        raise TrainingDataError(f"training_data_era_col_not_found:{era_col}")
    if id_col in frame.columns:
        frame = frame.set_index(id_col)
    elif frame.index.name != id_col:
        frame.index.name = id_col

    frame = frame[frame[era_col].isin(era_values)]
    keep_cols = [col for col in columns if col in frame.columns]
    return frame[keep_cols]


def _era_sort_key(era: object) -> int | str:
    if isinstance(era, (int, np.integer)):
        return int(era)
    if isinstance(era, str) and era.isdigit():
        return int(era)
    return str(era)


def _resolve_benchmark_column(columns: Iterable[str], benchmark_model: str) -> str:
    if benchmark_model in columns:
        return benchmark_model

    suffix = f"_{benchmark_model}"
    matches = [col for col in columns if col.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise TrainingDataError(f"training_benchmark_model_col_not_found:{benchmark_model}")
    raise TrainingDataError(f"training_benchmark_model_col_ambiguous:{benchmark_model}")


def _resolve_meta_model_column(
    columns: Iterable[str],
    requested_col: str,
    *,
    era_col: str,
    id_col: str,
) -> str:
    columns_list = [str(col) for col in columns]
    if requested_col in columns_list:
        return requested_col

    excluded = {era_col, id_col, "data_type"}
    candidates = [col for col in columns_list if col not in excluded]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise TrainingDataError(f"training_meta_model_col_not_found:{requested_col}")
    raise TrainingDataError(f"training_meta_model_col_ambiguous:{requested_col}")


def _safe_download(client: TrainingDataClient, *, filename: str, dest_path: str) -> None:
    try:
        client.download_dataset(filename=filename, dest_path=dest_path)
    except Exception as exc:
        raise TrainingDataError(f"training_dataset_download_failed:{filename}") from exc


def _file_fingerprint(path: Path) -> dict[str, object]:
    if not path.exists():
        raise TrainingDataError(f"training_data_file_not_found:{path}")

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
    }


def _table_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        try:
            pq = import_module("pyarrow.parquet")
        except ImportError:
            return [str(col) for col in _read_parquet(path).columns]

        try:
            schema = pq.read_schema(path)
            columns = [str(name) for name in schema.names]
            metadata = schema.metadata or {}
            pandas_metadata_raw = metadata.get(b"pandas")
            if pandas_metadata_raw is None:
                return columns

            pandas_metadata = json.loads(pandas_metadata_raw.decode("utf-8"))
            index_columns_raw = pandas_metadata.get("index_columns", [])
            index_columns: set[str] = set()
            for entry in index_columns_raw:
                if isinstance(entry, str):
                    index_columns.add(entry)
                    continue
                if isinstance(entry, dict):
                    name = entry.get("name")
                    if isinstance(name, str):
                        index_columns.add(name)
            return [col for col in columns if col not in index_columns]
        except Exception:
            return [str(col) for col in _read_parquet(path).columns]

    if path.suffix.lower() == ".csv":
        try:
            return [str(col) for col in pd.read_csv(path, nrows=0).columns]
        except Exception:
            return [str(col) for col in _read_csv(path).columns]

    return [str(col) for col in _read_table(path).columns]


def _table_row_count(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        try:
            ds = import_module("pyarrow.dataset")
        except ImportError:
            return int(len(_read_parquet(path)))

        try:
            dataset = ds.dataset(str(path), format="parquet")
            return int(dataset.count_rows())
        except Exception:
            return int(len(_read_parquet(path)))

    return int(len(_read_table(path)))


def _parquet_unique_values(path: Path, *, column: str) -> list[object]:
    if path.suffix.lower() != ".parquet":
        raise TrainingDataError(f"training_metrics_era_stream_requires_parquet:{path}")
    try:
        ds = import_module("pyarrow.dataset")
    except ImportError as exc:
        raise TrainingDataError("training_data_lazy_backend_missing_pyarrow") from exc

    try:
        dataset = ds.dataset(str(path), format="parquet")
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_read_failed:{path}") from exc

    available_columns = {str(name) for name in dataset.schema.names}
    if column not in available_columns:
        raise TrainingDataError(f"training_data_era_col_not_found:{column}")

    values: set[object] = set()
    try:
        scanner = dataset.scanner(columns=[column])
        for batch in scanner.to_batches():
            values.update(batch.column(0).to_pylist())
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_read_failed:{path}") from exc
    return sorted(values, key=_era_sort_key)


def _read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path, columns=columns)
    if suffix == ".parquet":
        return _read_parquet(path, columns=columns)
    raise TrainingDataError(f"training_predictions_file_extension_not_supported:{path}")


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        if columns is None:
            return pd.read_csv(path)
        return pd.read_csv(path, usecols=columns)
    except Exception as exc:
        raise TrainingDataError(f"training_data_csv_read_failed:{path}") from exc


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except ImportError as exc:
        raise TrainingDataError("training_data_parquet_engine_missing") from exc
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_read_failed:{path}") from exc


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path)
    except ImportError as exc:
        raise TrainingDataError("training_data_parquet_engine_missing") from exc
    except Exception as exc:
        raise TrainingDataError(f"training_data_parquet_write_failed:{path}") from exc
