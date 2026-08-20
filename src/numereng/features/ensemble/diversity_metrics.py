"""Diversity/complementarity metrics over a shared per-era prediction panel.

Three policy-free primitives the cross-lane diversity report builds on:
per-era pairwise Spearman between candidate columns (summarized per pair),
joint drawdown (fraction of eras where both members sit in their own bottom BMC
decile), and a paired circular moving-block bootstrap on per-era differences.

Every tuneable is an explicit function argument — there are no hidden constants.
The bootstrap requires at least two full blocks of eras and hard-errors otherwise.

USAGE:
    from numereng.features.ensemble import diversity_metrics as dm
    pairs = dm.per_era_pairwise_spearman(matrix, labels, era_ranges)
    boot = dm.paired_block_bootstrap(diffs, block_length=10, n_resamples=2000, rng_seed=7)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DiversityMetricError(ValueError):
    """Raised when diversity-metric inputs are degenerate or under-sized."""


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairwiseSpearman:
    """Distribution of per-era Spearman for one column pair."""

    left: str
    right: str
    mean: float | None
    p10: float | None
    p90: float | None
    minimum: float | None
    n_eras: int


@dataclass(frozen=True)
class JointDrawdown:
    """Joint bottom-decile co-occurrence for one column pair."""

    left: str
    right: str
    fraction: float
    joint_eras: int
    n_eras: int


@dataclass(frozen=True)
class BootstrapResult:
    """Paired circular moving-block bootstrap summary on per-era differences."""

    mean: float
    ci90_low: float
    ci90_high: float
    prob_positive: float
    block_length_eras: int
    n_resamples: int
    rng_seed: int
    n_eras: int
    n_blocks: int


# --------------------------------------------------------------------------- #
# Per-era pairwise Spearman
# --------------------------------------------------------------------------- #


def per_era_pairwise_spearman(
    matrix: np.ndarray,
    labels: tuple[str, ...],
    era_ranges: tuple[tuple[str, int, int], ...],
) -> tuple[PairwiseSpearman, ...]:
    """Per-pair distribution {mean, p10, p90, min} of per-era Spearman correlations."""

    if matrix.ndim != 2:
        raise DiversityMetricError("diversity_spearman_matrix_not_2d")
    if matrix.shape[1] != len(labels):
        raise DiversityMetricError("diversity_spearman_label_count_mismatch")
    if len(labels) < 2:
        raise DiversityMetricError("diversity_spearman_needs_two_columns")

    n_cols = len(labels)
    per_era_rho: dict[tuple[int, int], list[float]] = {pair: [] for pair in combinations(range(n_cols), 2)}
    for _era, start, end in era_ranges:
        block = matrix[start:end]
        if block.shape[0] < 2:
            continue
        ranks = _column_ranks(block)
        for left, right in per_era_rho:
            rho = _pearson(ranks[:, left], ranks[:, right])
            if rho is not None and np.isfinite(rho):
                per_era_rho[(left, right)].append(float(rho))

    results: list[PairwiseSpearman] = []
    for (left, right), values in per_era_rho.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            results.append(PairwiseSpearman(labels[left], labels[right], None, None, None, None, 0))
            continue
        results.append(
            PairwiseSpearman(
                left=labels[left],
                right=labels[right],
                mean=float(np.mean(array)),
                p10=float(np.quantile(array, 0.10)),
                p90=float(np.quantile(array, 0.90)),
                minimum=float(np.min(array)),
                n_eras=int(array.size),
            )
        )
    return tuple(results)


# --------------------------------------------------------------------------- #
# Joint drawdown
# --------------------------------------------------------------------------- #


def joint_drawdown(
    era_bmc: np.ndarray,
    labels: tuple[str, ...],
    *,
    decile: float = 0.10,
) -> tuple[JointDrawdown, ...]:
    """Fraction of eras where both members sit in their own bottom-decile BMC.

    Each column is thresholded at its own `decile` quantile of finite per-era BMC;
    a pair co-occurs in an era when both columns are at or below their thresholds.
    """

    if era_bmc.ndim != 2:
        raise DiversityMetricError("diversity_joint_drawdown_matrix_not_2d")
    if era_bmc.shape[1] != len(labels):
        raise DiversityMetricError("diversity_joint_drawdown_label_count_mismatch")
    if len(labels) < 2:
        raise DiversityMetricError("diversity_joint_drawdown_needs_two_columns")

    n_cols = len(labels)
    in_bottom = np.zeros(era_bmc.shape, dtype=bool)
    for column in range(n_cols):
        series = era_bmc[:, column]
        finite = series[np.isfinite(series)]
        if finite.size == 0:
            continue
        threshold = float(np.quantile(finite, decile))
        in_bottom[:, column] = np.isfinite(series) & (series <= threshold)

    results: list[JointDrawdown] = []
    for left, right in combinations(range(n_cols), 2):
        both_defined = np.isfinite(era_bmc[:, left]) & np.isfinite(era_bmc[:, right])
        n_eras = int(np.count_nonzero(both_defined))
        joint = int(np.count_nonzero(in_bottom[:, left] & in_bottom[:, right]))
        fraction = (joint / n_eras) if n_eras > 0 else 0.0
        results.append(JointDrawdown(labels[left], labels[right], fraction, joint, n_eras))
    return tuple(results)


# --------------------------------------------------------------------------- #
# Paired circular moving-block bootstrap
# --------------------------------------------------------------------------- #


def paired_block_bootstrap(
    differences: np.ndarray,
    *,
    block_length_eras: int,
    n_resamples: int,
    rng_seed: int,
) -> BootstrapResult:
    """Circular moving-block bootstrap of the mean of per-era differences.

    Blocks of `block_length_eras` consecutive eras are drawn with circular wrap so
    every era is an equally likely block start; each resample tiles blocks to cover
    all eras. Requires at least two full blocks of eras.
    """

    if block_length_eras < 1:
        raise DiversityMetricError("diversity_bootstrap_block_length_invalid")
    if n_resamples < 1:
        raise DiversityMetricError("diversity_bootstrap_resamples_invalid")

    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise DiversityMetricError("diversity_bootstrap_non_finite_differences")
    n_eras = values.size
    if n_eras < 2 * block_length_eras:
        raise DiversityMetricError("diversity_bootstrap_needs_two_full_blocks")

    n_blocks = int(np.ceil(n_eras / block_length_eras))
    rng = np.random.default_rng(rng_seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        starts = rng.integers(0, n_eras, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block_length_eras)[None, :]) % n_eras
        sample = values[offsets.reshape(-1)[:n_eras]]
        means[index] = float(np.mean(sample))

    return BootstrapResult(
        mean=float(np.mean(values)),
        ci90_low=float(np.quantile(means, 0.05)),
        ci90_high=float(np.quantile(means, 0.95)),
        prob_positive=float(np.mean(means > 0.0)),
        block_length_eras=int(block_length_eras),
        n_resamples=int(n_resamples),
        rng_seed=int(rng_seed),
        n_eras=int(n_eras),
        n_blocks=int(n_blocks),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _column_ranks(block: np.ndarray) -> np.ndarray:
    """Average-tie ranks per column within one era block."""

    ranks = np.empty(block.shape, dtype=np.float64)
    for column in range(block.shape[1]):
        ranks[:, column] = _rankdata_average(block[:, column])
    return ranks


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and sorted_values[end] == sorted_values[position]:
            end += 1
        average_rank = (position + end - 1) / 2.0 + 1.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2:
        return None
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


__all__ = [
    "BootstrapResult",
    "DiversityMetricError",
    "JointDrawdown",
    "PairwiseSpearman",
    "joint_drawdown",
    "paired_block_bootstrap",
    "per_era_pairwise_spearman",
]
