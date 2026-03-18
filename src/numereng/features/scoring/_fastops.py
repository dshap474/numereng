"""High-throughput numerical kernels for Numerai-style scoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from numba import njit as _numba_njit
from numpy.typing import NDArray

_Fn = TypeVar("_Fn", bound=Callable[..., Any])
njit = cast(Callable[..., Callable[[_Fn], _Fn]], _numba_njit)

FloatArray = NDArray[np.float64]

try:
    from scipy.special import ndtri as _ndtri
except Exception:  # pragma: no cover - scipy.special missing
    from scipy.stats import norm as _norm

    def _ndtri(values: FloatArray) -> FloatArray:
        return cast(FloatArray, _norm.ppf(values))


_FLOAT_NAN = float("nan")


@njit(cache=True)
def _rank_average_fraction_half_1d(values: FloatArray) -> FloatArray:
    """Return `(average_rank - 0.5) / n` for one 1D array."""
    n = values.size
    out = np.empty(n, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranked_value = (avg_rank - 0.5) / n
        for k in range(i, j):
            out[order[k]] = ranked_value
        i = j
    return out


@njit(cache=True)
def _rank_average_fraction_full_1d(values: FloatArray) -> FloatArray:
    """Return `average_rank / n` for one 1D array."""
    n = values.size
    out = np.empty(n, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranked_value = avg_rank / n
        for k in range(i, j):
            out[order[k]] = ranked_value
        i = j
    return out


@njit(cache=True)
def _rank_average_fraction_half_2d(values: FloatArray) -> FloatArray:
    n, p = values.shape
    out = np.empty((n, p), dtype=np.float64)
    for col_idx in range(p):
        out[:, col_idx] = _rank_average_fraction_half_1d(values[:, col_idx])
    return out


@njit(cache=True)
def _rank_average_fraction_full_2d(values: FloatArray) -> FloatArray:
    n, p = values.shape
    out = np.empty((n, p), dtype=np.float64)
    for col_idx in range(p):
        out[:, col_idx] = _rank_average_fraction_full_1d(values[:, col_idx])
    return out


def as_2d(values: FloatArray) -> FloatArray:
    arr = cast(FloatArray, np.asarray(values, dtype=np.float64))
    if arr.ndim == 1:
        return cast(FloatArray, arr.reshape(-1, 1))
    if arr.ndim != 2:
        raise ValueError("expected 1D or 2D array")
    return arr


def signed_power_1p5(values: FloatArray) -> FloatArray:
    arr = cast(FloatArray, np.asarray(values, dtype=np.float64))
    return cast(FloatArray, np.sign(arr) * np.abs(arr) ** 1.5)


def gaussianize_centered_rank_matrix(values: FloatArray) -> FloatArray:
    ranked = _rank_average_fraction_half_2d(as_2d(values))
    return cast(FloatArray, _ndtri(ranked))


def gaussianize_centered_rank_vector(values: FloatArray) -> FloatArray:
    ranked = _rank_average_fraction_half_1d(cast(FloatArray, np.asarray(values, dtype=np.float64)))
    return cast(FloatArray, _ndtri(ranked))


def transform_predictions_for_corr(values: FloatArray) -> FloatArray:
    return signed_power_1p5(gaussianize_centered_rank_matrix(values))


def center_target_for_corr(target: FloatArray, *, target_pow15: bool = True) -> FloatArray:
    centered = np.asarray(target, dtype=np.float64) - float(np.mean(target))
    if target_pow15:
        return signed_power_1p5(centered)
    return centered


def center_target_for_mmc_like(target: FloatArray) -> FloatArray:
    live = cast(FloatArray, np.asarray(target, dtype=np.float64))
    if np.all(live >= 0.0) and np.all(live <= 1.0):
        live = live * 4.0
    return live - float(np.mean(live))


def pearson_corr_matrix_vs_vector(values: FloatArray, target: FloatArray) -> FloatArray:
    matrix = as_2d(values)
    y = cast(FloatArray, np.asarray(target, dtype=np.float64))
    n = y.size
    if n < 2:
        return np.full(matrix.shape[1], _FLOAT_NAN, dtype=np.float64)
    y_centered = y - float(np.mean(y))
    y_std = float(np.std(y_centered, ddof=1))
    if not np.isfinite(y_std) or y_std == 0.0:
        return np.full(matrix.shape[1], _FLOAT_NAN, dtype=np.float64)
    x_centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    x_std = np.std(x_centered, axis=0, ddof=1)
    numer = x_centered.T @ y_centered
    denom = (n - 1) * x_std * y_std
    out = numer / denom
    out[~np.isfinite(out)] = np.nan
    return cast(FloatArray, out.astype(np.float64, copy=False))


def numerai_corr_matrix_vs_target(
    predictions: FloatArray,
    target: FloatArray,
    *,
    target_pow15: bool = True,
) -> FloatArray:
    transformed_predictions = transform_predictions_for_corr(predictions)
    transformed_target = center_target_for_corr(target, target_pow15=target_pow15)
    return pearson_corr_matrix_vs_vector(transformed_predictions, transformed_target)


def cwmm_matrix_vs_reference(predictions: FloatArray, reference: FloatArray) -> FloatArray:
    transformed_predictions = transform_predictions_for_corr(predictions)
    return pearson_corr_matrix_vs_vector(
        transformed_predictions,
        cast(FloatArray, np.asarray(reference, dtype=np.float64)),
    )


def orthogonalize_matrix_against_vector(values: FloatArray, neutralizer: FloatArray) -> FloatArray:
    matrix = as_2d(values)
    neutralizer_vector = cast(FloatArray, np.asarray(neutralizer, dtype=np.float64).reshape(-1))
    denom = float(neutralizer_vector @ neutralizer_vector)
    if not np.isfinite(denom) or denom == 0.0:
        return np.full(matrix.shape, np.nan, dtype=np.float64)
    coeff = (matrix.T @ neutralizer_vector) / denom
    return matrix - np.outer(neutralizer_vector, coeff)


def correlation_contribution_matrix(
    predictions: FloatArray,
    meta_model: FloatArray,
    target: FloatArray,
) -> FloatArray:
    pred_gauss = gaussianize_centered_rank_matrix(predictions)
    meta_gauss = gaussianize_centered_rank_vector(meta_model)
    neutral_predictions = orthogonalize_matrix_against_vector(pred_gauss, meta_gauss)
    centered_target = center_target_for_mmc_like(target)
    if centered_target.size == 0:
        return np.full(neutral_predictions.shape[1], _FLOAT_NAN, dtype=np.float64)
    scores = (centered_target @ neutral_predictions) / centered_target.size
    result = cast(FloatArray, np.asarray(scores, dtype=np.float64))
    result[~np.isfinite(result)] = np.nan
    return result


def neutralize_matrix(values: FloatArray, neutralizers: FloatArray, proportion: float = 1.0) -> FloatArray:
    matrix = as_2d(values)
    neutralizer_matrix = as_2d(neutralizers)
    if neutralizer_matrix.size == 0:
        return matrix.copy()
    intercept = np.ones((neutralizer_matrix.shape[0], 1), dtype=np.float64)
    design = np.concatenate([neutralizer_matrix, intercept], axis=1)
    least_squares = np.linalg.lstsq(design, matrix, rcond=1e-6)[0]
    adjustments = proportion * (design @ least_squares)
    return cast(FloatArray, matrix - adjustments)


def feature_neutral_corr_matrix(
    predictions: FloatArray,
    features: FloatArray,
    target: FloatArray,
) -> FloatArray:
    pred_gauss = gaussianize_centered_rank_matrix(predictions)
    neutralized = neutralize_matrix(pred_gauss, features)
    std = np.std(neutralized, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = neutralized / std
    normalized[:, ~np.isfinite(normalized).all(axis=0)] = np.nan
    return numerai_corr_matrix_vs_target(normalized, target)


def feature_exposure_matrices(
    predictions: FloatArray,
    features: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    pred_matrix = as_2d(predictions)
    feature_matrix = as_2d(features)
    n_rows, n_preds = pred_matrix.shape
    if n_rows < 2 or feature_matrix.shape[1] == 0:
        nan_arr = np.full(n_preds, _FLOAT_NAN, dtype=np.float64)
        return nan_arr, nan_arr

    ranked_features = _rank_average_fraction_full_2d(feature_matrix)
    feature_centered = ranked_features - np.mean(ranked_features, axis=0, keepdims=True)
    feature_std = np.std(feature_centered, axis=0, ddof=1)
    valid_features = np.isfinite(feature_std) & (feature_std != 0.0)
    if not np.any(valid_features):
        nan_arr = np.full(n_preds, _FLOAT_NAN, dtype=np.float64)
        return nan_arr, nan_arr

    feature_centered = feature_centered[:, valid_features]
    feature_std = feature_std[valid_features]
    rms = np.empty(n_preds, dtype=np.float64)
    max_abs = np.empty(n_preds, dtype=np.float64)
    for pred_idx in range(n_preds):
        ranked_pred = _rank_average_fraction_full_1d(pred_matrix[:, pred_idx])
        pred_centered = ranked_pred - float(np.mean(ranked_pred))
        pred_std = float(np.std(pred_centered, ddof=1))
        if not np.isfinite(pred_std) or pred_std == 0.0:
            rms[pred_idx] = np.nan
            max_abs[pred_idx] = np.nan
            continue
        corr = (pred_centered[:, None] * feature_centered).sum(axis=0) / (
            (n_rows - 1) * pred_std * feature_std
        )
        corr = np.abs(corr[np.isfinite(corr)])
        if corr.size == 0:
            rms[pred_idx] = np.nan
            max_abs[pred_idx] = np.nan
            continue
        rms[pred_idx] = float(np.sqrt(np.mean(np.square(corr))))
        max_abs[pred_idx] = float(np.max(corr))
    return rms, max_abs


__all__ = [
    "correlation_contribution_matrix",
    "cwmm_matrix_vs_reference",
    "feature_exposure_matrices",
    "feature_neutral_corr_matrix",
    "numerai_corr_matrix_vs_target",
]
