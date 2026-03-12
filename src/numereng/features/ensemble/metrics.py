"""Metric and diagnostics helpers for ensemble outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from numereng.features.ensemble.contracts import EnsembleMetric

_EMPTY_METRIC_PAYLOAD = {
    "mean": None,
    "std": None,
    "p05": None,
    "p50": None,
    "p95": None,
}


def summarize_metrics(
    *,
    blended: np.ndarray,
    era_series: pd.Series,
    target_series: pd.Series | None,
) -> tuple[EnsembleMetric, ...]:
    """Summarize core ensemble metrics from blended predictions."""

    per_era = per_era_corr_series(
        blended=blended,
        era_series=era_series,
        target_series=target_series,
    )
    return _summarize_from_per_era_corr(per_era)


def per_era_corr_series(
    *,
    blended: np.ndarray,
    era_series: pd.Series,
    target_series: pd.Series | None,
) -> pd.Series:
    """Compute one CORR value per era for the provided prediction vector."""

    if target_series is None:
        return pd.Series(dtype=float, name="corr")

    frame = pd.DataFrame(
        {
            "era": era_series.values,
            "target": target_series.values,
            "prediction": blended,
        }
    )
    grouped = frame.groupby("era", sort=True)
    rows: list[tuple[str, float]] = []
    for era, group in grouped:
        corr = group["prediction"].corr(group["target"])
        if corr is None or not np.isfinite(corr):
            continue
        rows.append((str(era), float(corr)))

    if not rows:
        return pd.Series(dtype=float, name="corr")

    payload = pd.Series(
        [value for _, value in rows],
        index=[era for era, _ in rows],
        dtype=float,
        name="corr",
    )
    payload.index.name = "era"
    return payload


def component_metrics_table(
    *,
    ranked_predictions: pd.DataFrame,
    run_ids: tuple[str, ...],
    era_series: pd.Series,
    target_series: pd.Series | None,
    weights: tuple[float, ...],
) -> pd.DataFrame:
    """Summarize key metrics for each component run."""

    if ranked_predictions.shape[1] != len(run_ids):
        raise ValueError("ensemble_component_shape_mismatch")
    if len(weights) != len(run_ids):
        raise ValueError("ensemble_weights_length_mismatch")

    rows: list[dict[str, float | int | str | None]] = []
    for index, run_id in enumerate(run_ids):
        series = ranked_predictions.iloc[:, index].to_numpy(dtype=float)
        metrics = metric_dict(
            summarize_metrics(
                blended=series,
                era_series=era_series,
                target_series=target_series,
            )
        )
        rows.append(
            {
                "run_id": run_id,
                "weight": float(weights[index]),
                "rank": int(index),
                "corr_mean": metrics.get("corr_mean"),
                "corr_sharpe": metrics.get("corr_sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
            }
        )
    return pd.DataFrame(rows)


def era_metrics_table(*, per_era_corr: pd.Series) -> pd.DataFrame:
    """Build a canonical per-era metrics table."""

    if per_era_corr.empty:
        return pd.DataFrame(columns=["era", "corr"])

    frame = pd.DataFrame(
        {
            "era": per_era_corr.index.astype(str).tolist(),
            "corr": per_era_corr.to_numpy(dtype=float),
        }
    )
    return frame


def regime_metrics_table(*, per_era_corr: pd.Series, regime_buckets: int) -> pd.DataFrame:
    """Bucket eras into ordered regimes and summarize each regime."""

    if regime_buckets < 2:
        raise ValueError("ensemble_regime_buckets_invalid")

    columns = [
        "regime_bucket",
        "start_era",
        "end_era",
        "n_eras",
        "corr_mean",
        "corr_sharpe",
        "max_drawdown",
    ]
    if per_era_corr.empty:
        return pd.DataFrame(columns=columns)

    values = per_era_corr.to_numpy(dtype=float)
    eras = [str(item) for item in per_era_corr.index.tolist()]
    bucket_count = min(regime_buckets, len(values))
    splits = np.array_split(np.arange(len(values)), bucket_count)

    rows: list[dict[str, float | int | str | None]] = []
    for idx, split in enumerate(splits):
        if split.size == 0:
            continue
        split_values = values[split]
        split_eras = [eras[int(position)] for position in split]

        mean = float(split_values.mean())
        std = float(split_values.std(ddof=0))
        sharpe = None if std == 0.0 else float(mean / std)
        cumsum = np.cumsum(split_values)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = float((running_max - cumsum).max())

        rows.append(
            {
                "regime_bucket": int(idx + 1),
                "start_era": split_eras[0],
                "end_era": split_eras[-1],
                "n_eras": int(split.size),
                "corr_mean": mean,
                "corr_sharpe": sharpe,
                "max_drawdown": drawdown,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def bootstrap_metric_summary(
    *,
    per_era_corr: pd.Series,
    n_resamples: int = 400,
    seed: int = 1337,
) -> dict[str, Any]:
    """Estimate metric stability via bootstrap over per-era CORR values."""

    if n_resamples < 1:
        raise ValueError("ensemble_bootstrap_resamples_invalid")

    values = per_era_corr.to_numpy(dtype=float)
    if values.size == 0:
        return {
            "n_eras": 0,
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "metrics": {
                "corr_mean": dict(_EMPTY_METRIC_PAYLOAD),
                "corr_sharpe": dict(_EMPTY_METRIC_PAYLOAD),
                "max_drawdown": dict(_EMPTY_METRIC_PAYLOAD),
            },
        }

    rng = np.random.default_rng(seed=seed)
    means = np.empty(n_resamples, dtype=float)
    sharpes = np.empty(n_resamples, dtype=float)
    drawdowns = np.empty(n_resamples, dtype=float)
    n_values = values.size

    for idx in range(n_resamples):
        sample = values[rng.integers(0, n_values, size=n_values)]
        mean = float(sample.mean())
        std = float(sample.std(ddof=0))
        sharpe = np.nan if std == 0.0 else float(mean / std)
        cumsum = np.cumsum(sample)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = float((running_max - cumsum).max())

        means[idx] = mean
        sharpes[idx] = sharpe
        drawdowns[idx] = drawdown

    return {
        "n_eras": int(n_values),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "metrics": {
            "corr_mean": _summarize_array(means),
            "corr_sharpe": _summarize_array(sharpes),
            "max_drawdown": _summarize_array(drawdowns),
        },
    }


def correlation_matrix(*, ranked_predictions: pd.DataFrame, run_ids: tuple[str, ...]) -> pd.DataFrame:
    """Build correlation matrix between component predictions."""

    if ranked_predictions.shape[1] != len(run_ids):
        raise ValueError("ensemble_component_shape_mismatch")
    renamed = ranked_predictions.copy()
    renamed.columns = list(run_ids)
    corr = renamed.corr()
    corr = corr.replace([np.inf, -np.inf], np.nan)
    return corr


def metric_dict(metrics: tuple[EnsembleMetric, ...]) -> dict[str, float | None]:
    """Map metric tuple to a dict for serialization."""

    payload: dict[str, float | None] = {}
    for metric in metrics:
        payload[metric.name] = metric.value
    return payload


def _summarize_from_per_era_corr(per_era: pd.Series) -> tuple[EnsembleMetric, ...]:
    if per_era.empty:
        return (
            EnsembleMetric(name="corr_mean", value=None),
            EnsembleMetric(name="corr_sharpe", value=None),
            EnsembleMetric(name="max_drawdown", value=None),
        )

    corr_mean = float(per_era.mean())
    corr_std = float(per_era.std(ddof=0))
    corr_sharpe = None if corr_std == 0.0 else float(corr_mean / corr_std)
    cumsum = per_era.cumsum()
    running_max = cumsum.expanding(min_periods=1).max()
    drawdown = float((running_max - cumsum).max())

    return (
        EnsembleMetric(name="corr_mean", value=corr_mean),
        EnsembleMetric(name="corr_sharpe", value=corr_sharpe),
        EnsembleMetric(name="max_drawdown", value=drawdown),
    )


def _summarize_array(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return dict(_EMPTY_METRIC_PAYLOAD)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }
