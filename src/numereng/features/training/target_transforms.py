"""Target transform helpers for training-time label shaping."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from numereng.features.training.errors import TrainingModelError


def apply_target_transform(
    y: pd.Series,
    X: pd.DataFrame,
    transform: dict[str, Any] | str | None,
) -> pd.Series:
    """Apply configured target transform to training labels."""
    if transform is None or transform == {}:
        return y

    if isinstance(transform, str):
        transform = {"type": transform}

    if not isinstance(transform, dict):
        raise TrainingModelError("training_target_transform_invalid_type")

    transform_type = transform.get("type")
    if transform_type is None:
        raise TrainingModelError("training_target_transform_missing_type")

    if transform_type in {"residual_to_benchmark", "residualize_to_benchmark"}:
        benchmark_col = transform.get("benchmark_col")
        if not benchmark_col:
            raise TrainingModelError("training_target_transform_missing_benchmark_col")

        era_col = str(transform.get("era_col", "era"))
        per_era = bool(transform.get("per_era", True))
        fit_intercept = bool(transform.get("fit_intercept", True))
        proportion = float(transform.get("proportion", 1.0))
        return residualize_to_column(
            y,
            X,
            benchmark_col=str(benchmark_col),
            era_col=era_col,
            per_era=per_era,
            fit_intercept=fit_intercept,
            proportion=proportion,
        )

    if transform_type in {"subtract_benchmark", "subtract_benchmark_zscore"}:
        benchmark_col = transform.get("benchmark_col")
        if not benchmark_col:
            raise TrainingModelError("training_target_transform_missing_benchmark_col")

        era_col = str(transform.get("era_col", "era"))
        scale = float(transform.get("scale", 0.07))
        return subtract_scaled_zscore_column(
            y,
            X,
            benchmark_col=str(benchmark_col),
            era_col=era_col,
            scale=scale,
        )

    raise TrainingModelError(f"training_target_transform_unknown:{transform_type}")


def residualize_to_column(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    benchmark_col: str,
    era_col: str = "era",
    per_era: bool = True,
    fit_intercept: bool = True,
    proportion: float = 1.0,
) -> pd.Series:
    """Residualize target values to one benchmark column."""
    if benchmark_col not in X.columns:
        raise TrainingModelError(f"training_target_transform_benchmark_col_missing:{benchmark_col}")

    if per_era and era_col not in X.columns:
        raise TrainingModelError(f"training_target_transform_era_col_missing:{era_col}")

    if not (0.0 <= proportion <= 1.0):
        raise TrainingModelError("training_target_transform_proportion_out_of_range")

    benchmark = X[benchmark_col]
    eras = X[era_col] if per_era else None
    residual = _linear_residual(y, benchmark, groups=eras, fit_intercept=fit_intercept)
    if proportion == 1.0:
        return residual

    return (1.0 - proportion) * y.astype("float64") + proportion * residual


def subtract_scaled_zscore_column(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    benchmark_col: str,
    era_col: str = "era",
    scale: float = 0.07,
) -> pd.Series:
    """Subtract scaled benchmark z-score from target."""
    if benchmark_col not in X.columns:
        raise TrainingModelError(f"training_target_transform_benchmark_col_missing:{benchmark_col}")

    if era_col not in X.columns:
        raise TrainingModelError(f"training_target_transform_era_col_missing:{era_col}")

    if not np.isfinite(scale):
        raise TrainingModelError("training_target_transform_scale_not_finite")

    benchmark = X[benchmark_col]
    eras = X[era_col]
    z_benchmark = _zscore(benchmark, groups=eras)

    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64", copy=False)
    z_values = pd.to_numeric(z_benchmark, errors="coerce").to_numpy(dtype="float64", copy=False)
    transformed = y_values - float(scale) * z_values
    return pd.Series(transformed, index=y.index, name=y.name)


def _zscore(
    x: pd.Series,
    *,
    groups: pd.Series | None,
) -> pd.Series:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64", copy=False)

    if groups is None:
        z = _zscore_global(x_values)
        return pd.Series(z, index=x.index, name=x.name)

    group_codes, _ = pd.factorize(groups, sort=False)
    z = _zscore_groupwise(x_values, group_codes)
    return pd.Series(z, index=x.index, name=x.name)


def _zscore_global(x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    z = np.full_like(x, np.nan, dtype="float64")
    if not mask.any():
        return z

    xx = x[mask]
    mean = float(xx.mean())
    var = float(np.mean((xx - mean) ** 2))
    std = float(np.sqrt(var))
    if std == 0.0:
        z[mask] = 0.0
        return z

    z[mask] = (xx - mean) / std
    return z


def _zscore_groupwise(x: np.ndarray, group_codes: np.ndarray) -> np.ndarray:
    mask = (group_codes >= 0) & np.isfinite(x)
    z = np.full_like(x, np.nan, dtype="float64")
    if not mask.any():
        return z

    g = group_codes[mask]
    xx = x[mask]

    n_groups = int(g.max()) + 1
    counts = np.bincount(g, minlength=n_groups).astype("float64")
    sum_x = np.bincount(g, weights=xx, minlength=n_groups)
    sum_x2 = np.bincount(g, weights=xx * xx, minlength=n_groups)

    mean = np.divide(sum_x, counts, out=np.zeros_like(sum_x, dtype="float64"), where=counts != 0.0)
    mean_x2 = np.divide(sum_x2, counts, out=np.zeros_like(sum_x2, dtype="float64"), where=counts != 0.0)
    var = mean_x2 - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))

    denom = std[g]
    diff = xx - mean[g]
    z_vals = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0.0)
    z[mask] = z_vals
    return z


def _linear_residual(
    y: pd.Series,
    x: pd.Series,
    *,
    groups: pd.Series | None,
    fit_intercept: bool,
) -> pd.Series:
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64", copy=False)
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64", copy=False)

    if groups is None:
        resid = _linear_residual_global(y_values, x_values, fit_intercept=fit_intercept)
        return pd.Series(resid, index=y.index, name=y.name)

    group_codes, _ = pd.factorize(groups, sort=False)
    resid = _linear_residual_groupwise(
        y_values,
        x_values,
        group_codes,
        fit_intercept=fit_intercept,
    )
    return pd.Series(resid, index=y.index, name=y.name)


def _linear_residual_global(
    y: np.ndarray,
    x: np.ndarray,
    *,
    fit_intercept: bool,
) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(x)
    resid = np.full_like(y, np.nan, dtype="float64")
    if not mask.any():
        return resid

    yy = y[mask]
    xx = x[mask]

    if fit_intercept:
        yy = yy - yy.mean()
        xx = xx - xx.mean()

    denom = float(np.dot(xx, xx))
    alpha = float(np.dot(xx, yy) / denom) if denom != 0.0 else 0.0
    resid_vals = yy - alpha * xx
    resid[mask] = resid_vals
    return resid


def _linear_residual_groupwise(
    y: np.ndarray,
    x: np.ndarray,
    group_codes: np.ndarray,
    *,
    fit_intercept: bool,
) -> np.ndarray:
    mask = (group_codes >= 0) & np.isfinite(y) & np.isfinite(x)
    resid = np.full_like(y, np.nan, dtype="float64")
    if not mask.any():
        return resid

    g = group_codes[mask]
    yy = y[mask]
    xx = x[mask]

    n_groups = int(g.max()) + 1
    counts = np.bincount(g, minlength=n_groups).astype("float64")
    sum_y = np.bincount(g, weights=yy, minlength=n_groups)
    sum_x = np.bincount(g, weights=xx, minlength=n_groups)

    mean_y = np.divide(sum_y, counts, out=np.zeros_like(sum_y, dtype="float64"), where=counts != 0.0)
    mean_x = np.divide(sum_x, counts, out=np.zeros_like(sum_x, dtype="float64"), where=counts != 0.0)

    if fit_intercept:
        y_centered = yy - mean_y[g]
        x_centered = xx - mean_x[g]
    else:
        y_centered = yy
        x_centered = xx

    sum_xx = np.bincount(g, weights=x_centered * x_centered, minlength=n_groups)
    sum_xy = np.bincount(g, weights=x_centered * y_centered, minlength=n_groups)

    alpha = np.zeros(n_groups, dtype="float64")
    good = sum_xx != 0.0
    alpha[good] = sum_xy[good] / sum_xx[good]

    resid_vals = y_centered - alpha[g] * x_centered
    resid[mask] = resid_vals
    return resid


class TargetTransformWrapper:
    """Wrap a model with target transformation behavior during fit."""

    def __init__(self, model: Any, target_transform: dict[str, Any] | str) -> None:
        self._model = model
        self._target_transform = target_transform

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> TargetTransformWrapper:
        if not hasattr(y, "index") or not hasattr(X, "columns"):
            raise TrainingModelError("training_target_transform_requires_pandas_inputs")

        y_transformed = apply_target_transform(y, X, self._target_transform)
        self._model.fit(X, y_transformed, **kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        return self._model.predict(X)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
