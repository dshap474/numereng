"""Weight handling and optimizer utilities for ensembles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class EnsembleWeightsError(ValueError):
    """Raised when ensemble weight configuration is invalid."""


def normalize_weights(*, raw_weights: tuple[float, ...] | None, n_components: int) -> tuple[float, ...]:
    """Validate and normalize explicit weights; fallback to equal weights."""

    if n_components < 1:
        raise EnsembleWeightsError("ensemble_components_empty")

    if raw_weights is None:
        weight = 1.0 / float(n_components)
        return tuple(weight for _ in range(n_components))

    if len(raw_weights) != n_components:
        raise EnsembleWeightsError("ensemble_weights_length_mismatch")

    values = np.array(raw_weights, dtype=float)
    if np.isnan(values).any() or np.isinf(values).any():
        raise EnsembleWeightsError("ensemble_weights_non_finite")
    if (values < 0.0).any():
        raise EnsembleWeightsError("ensemble_weights_negative")

    total = float(values.sum())
    if total <= 0.0:
        raise EnsembleWeightsError("ensemble_weights_sum_nonpositive")

    normalized = values / total
    return tuple(float(value) for value in normalized.tolist())


def optimize_weights(
    *,
    ranked_predictions: pd.DataFrame,
    era_series: pd.Series,
    target_series: pd.Series,
    metric: str,
    initial_weights: tuple[float, ...],
) -> tuple[float, ...]:
    """Optimize non-negative weights under sum-to-one constraint."""

    scipy_optimize = _load_scipy_optimize()

    if ranked_predictions.empty:
        raise EnsembleWeightsError("ensemble_predictions_empty")

    n_components = ranked_predictions.shape[1]
    if n_components < 2:
        raise EnsembleWeightsError("ensemble_components_insufficient")

    initial = np.array(initial_weights, dtype=float)
    if initial.shape[0] != n_components:
        raise EnsembleWeightsError("ensemble_weights_length_mismatch")

    bounds = [(0.0, 1.0)] * n_components

    def objective(weights_array: np.ndarray) -> float:
        weights_sum = float(weights_array.sum())
        if weights_sum <= 0.0:
            return 1e9
        normalized = weights_array / weights_sum
        blended = ranked_predictions.to_numpy() @ normalized
        score = _ensemble_score(
            blended=blended,
            era_series=era_series,
            target_series=target_series,
            metric=metric,
        )
        if score is None:
            return 1e9
        return -score

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda w: float(w.sum()) - 1.0,
        }
    ]

    result = scipy_optimize.minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if not result.success:
        raise EnsembleWeightsError(f"ensemble_weight_optimization_failed:{result.message}")

    optimized = np.array(result.x, dtype=float)
    optimized = np.clip(optimized, 0.0, 1.0)
    total = float(optimized.sum())
    if total <= 0.0:
        raise EnsembleWeightsError("ensemble_weight_optimization_sum_nonpositive")
    optimized /= total
    return tuple(float(value) for value in optimized.tolist())


def _ensemble_score(
    *,
    blended: np.ndarray,
    era_series: pd.Series,
    target_series: pd.Series,
    metric: str,
) -> float | None:
    frame = pd.DataFrame(
        {
            "era": era_series.values,
            "target": target_series.values,
            "prediction": blended,
        }
    )

    grouped = frame.groupby("era", sort=True)
    per_era: list[float] = []
    for _, group in grouped:
        try:
            corr = float(group["prediction"].corr(group["target"]))
        except Exception:
            corr = float("nan")
        if np.isfinite(corr):
            per_era.append(corr)

    if not per_era:
        return None

    values = np.array(per_era, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))

    metric_key = metric.lower()
    if metric_key.endswith("_mean"):
        return mean
    if metric_key.endswith("_sharpe"):
        if std == 0.0:
            return None
        return mean / std
    if metric_key.endswith("max_drawdown"):
        cumsum = np.cumsum(values)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        return -float(np.max(drawdown))

    if std == 0.0:
        return mean
    return mean / std


def _load_scipy_optimize() -> Any:
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise EnsembleWeightsError("ensemble_dependency_missing_scipy") from exc
    return optimize
