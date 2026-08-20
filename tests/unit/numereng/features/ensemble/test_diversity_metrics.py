"""Diversity-metric primitive tests (spec §6 P2).

Covers per-era pairwise Spearman, joint bottom-decile drawdown, and the paired
circular moving-block bootstrap, including the degenerate cases the report relies
on (constant/zero diffs, same-seed determinism, circular wrap, <2-block error).

USAGE:
    uv run pytest tests/unit/numereng/features/ensemble/test_diversity_metrics.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from numereng.features.ensemble import diversity_metrics as dm
from numereng.features.ensemble.panel import era_ranges

# --------------------------------------------------------------------------- #
# Per-era pairwise Spearman
# --------------------------------------------------------------------------- #


def test_pairwise_spearman_known_two_eras() -> None:
    # era1: identical ordering (rho=+1); era2: reversed ordering (rho=-1).
    matrix = np.asarray(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]],
        dtype=np.float64,
    )
    ranges = era_ranges(["e1", "e1", "e1", "e2", "e2", "e2"])
    (pair,) = dm.per_era_pairwise_spearman(matrix, ("a", "b"), ranges)

    assert (pair.left, pair.right) == ("a", "b")
    assert pair.n_eras == 2
    assert pair.mean == pytest.approx(0.0)
    assert pair.minimum == pytest.approx(-1.0)
    assert pair.p10 == pytest.approx(-0.8)
    assert pair.p90 == pytest.approx(0.8)


def test_pairwise_spearman_perfect_correlation_is_one() -> None:
    matrix = np.asarray([[0.1, 0.2], [0.5, 0.9], [0.9, 1.5]], dtype=np.float64)
    ranges = era_ranges(["e1", "e1", "e1"])
    (pair,) = dm.per_era_pairwise_spearman(matrix, ("a", "b"), ranges)
    assert pair.mean == pytest.approx(1.0)


def test_pairwise_spearman_skips_single_row_and_constant_eras() -> None:
    # era1 has one row (skipped); era2 has a constant column (rho undefined, skipped).
    matrix = np.asarray([[0.1, 0.2], [0.5, 0.5], [0.9, 0.5]], dtype=np.float64)
    ranges = era_ranges(["e1", "e2", "e2"])
    (pair,) = dm.per_era_pairwise_spearman(matrix, ("a", "b"), ranges)
    assert pair.n_eras == 0
    assert pair.mean is None


def test_pairwise_spearman_label_mismatch_raises() -> None:
    matrix = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)
    with pytest.raises(dm.DiversityMetricError, match="label_count_mismatch"):
        dm.per_era_pairwise_spearman(matrix, ("a", "b"), era_ranges(["e1"]))


def test_pairwise_spearman_needs_two_columns() -> None:
    matrix = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float64)
    with pytest.raises(dm.DiversityMetricError, match="needs_two_columns"):
        dm.per_era_pairwise_spearman(matrix, ("a",), era_ranges(["e1", "e1", "e1"]))


# --------------------------------------------------------------------------- #
# Joint drawdown
# --------------------------------------------------------------------------- #


def test_joint_drawdown_co_occurrence_fraction() -> None:
    era_bmc = np.column_stack(
        [
            np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64),
            np.asarray([0, 2, 4, 6, 8, 1, 3, 5, 7, 9], dtype=np.float64),
        ]
    )
    (result,) = dm.joint_drawdown(era_bmc, ("a", "b"), decile=0.10)
    # Both columns place only era-0 in their own bottom decile.
    assert result.joint_eras == 1
    assert result.n_eras == 10
    assert result.fraction == pytest.approx(0.1)


def test_joint_drawdown_ignores_eras_with_nan() -> None:
    era_bmc = np.asarray([[np.nan, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    (result,) = dm.joint_drawdown(era_bmc, ("a", "b"))
    assert result.n_eras == 2  # first era dropped: column a is NaN there


def test_joint_drawdown_matrix_not_2d_raises() -> None:
    with pytest.raises(dm.DiversityMetricError, match="not_2d"):
        dm.joint_drawdown(np.asarray([0.1, 0.2]), ("a", "b"))


# --------------------------------------------------------------------------- #
# Paired circular moving-block bootstrap
# --------------------------------------------------------------------------- #


def test_bootstrap_constant_diffs_zero_width_ci() -> None:
    diffs = np.full(20, 0.5, dtype=np.float64)
    result = dm.paired_block_bootstrap(diffs, block_length_eras=10, n_resamples=200, rng_seed=7)
    assert result.mean == pytest.approx(0.5)
    assert result.ci90_low == pytest.approx(0.5)
    assert result.ci90_high == pytest.approx(0.5)
    assert result.prob_positive == pytest.approx(1.0)


def test_bootstrap_zero_diffs_prob_positive_zero() -> None:
    diffs = np.zeros(20, dtype=np.float64)
    result = dm.paired_block_bootstrap(diffs, block_length_eras=10, n_resamples=200, rng_seed=7)
    assert result.ci90_low == pytest.approx(0.0)
    assert result.ci90_high == pytest.approx(0.0)
    assert result.prob_positive == pytest.approx(0.0)


def test_bootstrap_same_seed_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    diffs = rng.standard_normal(40)
    first = dm.paired_block_bootstrap(diffs, block_length_eras=8, n_resamples=500, rng_seed=11)
    second = dm.paired_block_bootstrap(diffs, block_length_eras=8, n_resamples=500, rng_seed=11)
    assert first == second


def test_bootstrap_circular_wrap_covers_all_eras() -> None:
    diffs = np.arange(10, dtype=np.float64)
    result = dm.paired_block_bootstrap(diffs, block_length_eras=4, n_resamples=100, rng_seed=1)
    assert result.n_eras == 10
    assert result.n_blocks == 3  # ceil(10 / 4)
    assert result.mean == pytest.approx(float(np.mean(diffs)))
    assert 0.0 <= result.prob_positive <= 1.0


def test_bootstrap_under_two_blocks_raises() -> None:
    diffs = np.arange(5, dtype=np.float64)
    with pytest.raises(dm.DiversityMetricError, match="two_full_blocks"):
        dm.paired_block_bootstrap(diffs, block_length_eras=3, n_resamples=10, rng_seed=1)


def test_bootstrap_non_finite_diffs_raises() -> None:
    diffs = np.asarray([1.0, np.nan, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(dm.DiversityMetricError, match="non_finite"):
        dm.paired_block_bootstrap(diffs, block_length_eras=2, n_resamples=10, rng_seed=1)


def test_bootstrap_invalid_block_length_and_resamples() -> None:
    diffs = np.arange(20, dtype=np.float64)
    with pytest.raises(dm.DiversityMetricError, match="block_length_invalid"):
        dm.paired_block_bootstrap(diffs, block_length_eras=0, n_resamples=10, rng_seed=1)
    with pytest.raises(dm.DiversityMetricError, match="resamples_invalid"):
        dm.paired_block_bootstrap(diffs, block_length_eras=2, n_resamples=0, rng_seed=1)
