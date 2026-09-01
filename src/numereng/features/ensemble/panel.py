"""Policy-free panel-scoring primitives extracted from ensemble/selection.py.

These are the pure numeric building blocks the selection sweep and the
diversity metrics share: per-era range indexing, the weight-matrix CORR/BMC
scorer (last-N window as an explicit parameter, never a hardcode), and
`score_on_panel`, which scores each column of a ranked panel as a standalone
prediction and returns per-era CORR/BMC series.

`selection.py` keeps thin compatibility wrappers around these; there is no reverse
import from `selection.py` into this module.

USAGE:
    from numereng.features.ensemble import panel
    ranges = panel.era_ranges(["e1", "e1", "e2"])
    scored = panel.score_on_panel(matrix, target, benchmark, ranges)  # per-era BMC per column
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numereng.features.scoring._fastops import (
    correlation_contribution_matrix,
    numerai_corr_matrix_vs_target,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_WEIGHT_CHUNK_SIZE = 256
DEFAULT_RECENT_WINDOW_ERAS = 200


@dataclass(frozen=True)
class PanelScores:
    """Per-era CORR/BMC score matrices for one panel (n_eras x n_columns)."""

    eras: tuple[str, ...]
    corr: np.ndarray
    bmc: np.ndarray


# --------------------------------------------------------------------------- #
# Era indexing
# --------------------------------------------------------------------------- #


def era_ranges(eras: list[str]) -> tuple[tuple[str, int, int], ...]:
    """Return contiguous (era, start, end) row ranges over a sorted era list."""

    if not eras:
        return ()
    rows: list[tuple[str, int, int]] = []
    start = 0
    current = eras[0]
    for index, era in enumerate(eras[1:], start=1):
        if era == current:
            continue
        rows.append((current, start, index))
        current = era
        start = index
    rows.append((current, start, len(eras)))
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Per-era scoring
# --------------------------------------------------------------------------- #


def per_era_score_matrices(
    *,
    prediction_matrix: np.ndarray,
    target_vector: np.ndarray,
    benchmark_vector: np.ndarray,
    era_ranges: tuple[tuple[str, int, int], ...],
    weight_matrix: np.ndarray,
    chunk_size: int = DEFAULT_WEIGHT_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Compute per-era CORR/BMC score matrices (n_eras x n_variants) for weight rows."""

    n_eras = len(era_ranges)
    n_variants = weight_matrix.shape[0]
    corr_scores = np.full((n_eras, n_variants), np.nan, dtype=np.float64)
    bmc_scores = np.full((n_eras, n_variants), np.nan, dtype=np.float64)

    for era_index, (_era, start, end) in enumerate(era_ranges):
        pred_slice = prediction_matrix[start:end]
        target_slice = target_vector[start:end]
        benchmark_slice = benchmark_vector[start:end]
        for offset in range(0, n_variants, chunk_size):
            chunk = weight_matrix[offset : offset + chunk_size]
            blended = pred_slice @ chunk.T
            corr_scores[era_index, offset : offset + len(chunk)] = numerai_corr_matrix_vs_target(blended, target_slice)
            bmc_scores[era_index, offset : offset + len(chunk)] = correlation_contribution_matrix(
                blended,
                benchmark_slice,
                target_slice,
            )
    return {"corr": corr_scores, "bmc": bmc_scores}


def score_weight_matrix(
    *,
    prediction_matrix: np.ndarray,
    target_vector: np.ndarray,
    benchmark_vector: np.ndarray,
    era_ranges: tuple[tuple[str, int, int], ...],
    weight_matrix: np.ndarray,
    recent_window_eras: int = DEFAULT_RECENT_WINDOW_ERAS,
    chunk_size: int = DEFAULT_WEIGHT_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Score weight rows across eras; summarize with an explicit recent-era window."""

    matrices = per_era_score_matrices(
        prediction_matrix=prediction_matrix,
        target_vector=target_vector,
        benchmark_vector=benchmark_vector,
        era_ranges=era_ranges,
        weight_matrix=weight_matrix,
        chunk_size=chunk_size,
    )
    corr_scores = matrices["corr"]
    bmc_scores = matrices["bmc"]
    n_eras = len(era_ranges)
    window = min(recent_window_eras, n_eras)
    recent_window = bmc_scores[-window:, :] if window > 0 else bmc_scores[:0, :]
    corr_summary = summary_columns(corr_scores)
    bmc_summary = summary_columns(bmc_scores)
    recent_summary = summary_columns(recent_window)
    return {
        "corr_mean": corr_summary["mean"],
        "corr_std": corr_summary["std"],
        "corr_sharpe": corr_summary["sharpe"],
        "corr_max_drawdown": corr_summary["max_drawdown"],
        "bmc_mean": bmc_summary["mean"],
        "bmc_std": bmc_summary["std"],
        "bmc_sharpe": bmc_summary["sharpe"],
        "bmc_max_drawdown": bmc_summary["max_drawdown"],
        "bmc_last_200_eras_mean": recent_summary["mean"],
        "bmc_last_200_eras_std": recent_summary["std"],
        "bmc_last_200_eras_sharpe": recent_summary["sharpe"],
        "bmc_last_200_eras_max_drawdown": recent_summary["max_drawdown"],
    }


def score_on_panel(
    matrix: np.ndarray,
    target: np.ndarray,
    benchmark: np.ndarray,
    era_ranges: tuple[tuple[str, int, int], ...],
) -> PanelScores:
    """Score each column of `matrix` as a standalone prediction; per-era CORR/BMC.

    Each column is treated as its own weight-1 blend, so the returned CORR/BMC
    matrices carry one column of per-era scores per input column. This is the
    primitive the diversity report uses for per-candidate/per-blend BMC series.
    """

    n_columns = matrix.shape[1] if matrix.ndim == 2 else 1
    prediction_matrix = matrix if matrix.ndim == 2 else matrix.reshape(-1, 1)
    identity = np.eye(n_columns, dtype=np.float64)
    matrices = per_era_score_matrices(
        prediction_matrix=prediction_matrix,
        target_vector=target,
        benchmark_vector=benchmark,
        era_ranges=era_ranges,
        weight_matrix=identity,
    )
    return PanelScores(
        eras=tuple(era for era, _start, _end in era_ranges),
        corr=matrices["corr"],
        bmc=matrices["bmc"],
    )


# --------------------------------------------------------------------------- #
# Column summaries
# --------------------------------------------------------------------------- #


def summary_columns(values: np.ndarray) -> dict[str, np.ndarray]:
    """Column-wise mean/std/sharpe/max-drawdown over a (n_eras x n_cols) matrix."""

    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    std = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column_index in range(values.shape[1]):
        finite = values[np.isfinite(values[:, column_index]), column_index]
        if finite.size == 0:
            continue
        mean[column_index] = float(np.mean(finite))
        std[column_index] = float(np.std(finite, ddof=0))
    sharpe = np.divide(mean, std, out=np.full_like(mean, np.nan), where=std != 0.0)
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown_per_column(values),
    }


def max_drawdown_per_column(values: np.ndarray) -> np.ndarray:
    """Column-wise max cumulative drawdown over a (n_eras x n_cols) matrix."""

    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column_index in range(values.shape[1]):
        series = values[:, column_index]
        running = 0.0
        peak = 0.0
        worst = 0.0
        for value in series:
            if not np.isfinite(value):
                continue
            running += float(value)
            if running > peak:
                peak = running
            drawdown = peak - running
            if drawdown > worst:
                worst = drawdown
        out[column_index] = worst
    return out


__all__ = [
    "DEFAULT_RECENT_WINDOW_ERAS",
    "DEFAULT_WEIGHT_CHUNK_SIZE",
    "PanelScores",
    "era_ranges",
    "max_drawdown_per_column",
    "per_era_score_matrices",
    "score_on_panel",
    "score_weight_matrix",
    "summary_columns",
]
