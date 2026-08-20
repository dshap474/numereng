"""Panel-primitive tests (spec §6 P2): numeric equivalence + era/edge-case behavior.

`panel.py` was extracted from `selection.py`; these tests pin the extraction with an
assert_allclose against a literal brute-force reference (mirroring the selection anchor
at test_selection.py:287) and exercise the era-range / recent-window edge cases.

USAGE:
    uv run pytest tests/unit/numereng/features/ensemble/test_panel.py -q
"""

from __future__ import annotations

import numpy as np

from numereng.features.ensemble import panel
from numereng.features.scoring._fastops import (
    correlation_contribution_matrix,
    numerai_corr_matrix_vs_target,
)

# --------------------------------------------------------------------------- #
# Small literal fixture (same shape as the selection brute-force anchor)
# --------------------------------------------------------------------------- #


def _anchor_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, int, int], ...]]:
    prediction_matrix = np.asarray(
        [
            [0.1, 0.8],
            [0.4, 0.3],
            [0.9, 0.2],
            [0.2, 0.7],
            [0.3, 0.5],
            [0.8, 0.1],
        ],
        dtype=np.float64,
    )
    target_vector = np.asarray([0.1, 0.5, 0.9, 0.2, 0.4, 0.8], dtype=np.float64)
    benchmark_vector = np.asarray([0.2, 0.4, 0.7, 0.3, 0.45, 0.75], dtype=np.float64)
    ranges = panel.era_ranges(["era1", "era1", "era1", "era2", "era2", "era2"])
    return prediction_matrix, target_vector, benchmark_vector, ranges


def _bruteforce_per_column(
    prediction_matrix: np.ndarray,
    target_vector: np.ndarray,
    benchmark_vector: np.ndarray,
    ranges: tuple[tuple[str, int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-era CORR/BMC for each column scored as a standalone (weight-1) prediction."""

    n_columns = prediction_matrix.shape[1]
    corr = np.full((len(ranges), n_columns), np.nan, dtype=np.float64)
    bmc = np.full((len(ranges), n_columns), np.nan, dtype=np.float64)
    for era_index, (_era, start, end) in enumerate(ranges):
        target_slice = target_vector[start:end]
        benchmark_slice = benchmark_vector[start:end]
        for column in range(n_columns):
            blended = prediction_matrix[start:end, column].reshape(-1, 1)
            corr[era_index, column] = float(numerai_corr_matrix_vs_target(blended, target_slice)[0])
            bmc[era_index, column] = float(correlation_contribution_matrix(blended, benchmark_slice, target_slice)[0])
    return corr, bmc


# --------------------------------------------------------------------------- #
# Numeric equivalence
# --------------------------------------------------------------------------- #


def test_score_on_panel_matches_bruteforce_reference() -> None:
    prediction_matrix, target_vector, benchmark_vector, ranges = _anchor_inputs()
    scored = panel.score_on_panel(prediction_matrix, target_vector, benchmark_vector, ranges)
    corr, bmc = _bruteforce_per_column(prediction_matrix, target_vector, benchmark_vector, ranges)

    assert scored.eras == ("era1", "era2")
    np.testing.assert_allclose(scored.corr, corr)
    np.testing.assert_allclose(scored.bmc, bmc)


def test_score_weight_matrix_last_window_matches_full_when_within_window() -> None:
    prediction_matrix, target_vector, benchmark_vector, ranges = _anchor_inputs()
    weights = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float64)

    summary = panel.score_weight_matrix(
        prediction_matrix=prediction_matrix,
        target_vector=target_vector,
        benchmark_vector=benchmark_vector,
        era_ranges=ranges,
        weight_matrix=weights,
    )
    # Only two eras: the "last 200" window is the whole panel, so the summaries match.
    np.testing.assert_allclose(summary["bmc_mean"], summary["bmc_last_200_eras_mean"])


def test_score_on_panel_1d_input_treated_as_single_column() -> None:
    _matrix, target_vector, benchmark_vector, ranges = _anchor_inputs()
    column = np.asarray([0.1, 0.4, 0.9, 0.2, 0.3, 0.8], dtype=np.float64)
    scored = panel.score_on_panel(column, target_vector, benchmark_vector, ranges)
    assert scored.bmc.shape == (2, 1)
    assert scored.corr.shape == (2, 1)


# --------------------------------------------------------------------------- #
# Column-order permutation invariance
# --------------------------------------------------------------------------- #


def test_score_on_panel_column_permutation_permutes_output() -> None:
    prediction_matrix, target_vector, benchmark_vector, ranges = _anchor_inputs()
    base = panel.score_on_panel(prediction_matrix, target_vector, benchmark_vector, ranges)
    swapped = panel.score_on_panel(prediction_matrix[:, ::-1], target_vector, benchmark_vector, ranges)
    np.testing.assert_allclose(swapped.bmc, base.bmc[:, ::-1])
    np.testing.assert_allclose(swapped.corr, base.corr[:, ::-1])


# --------------------------------------------------------------------------- #
# era_ranges edge cases
# --------------------------------------------------------------------------- #


def test_era_ranges_contiguous_groups() -> None:
    assert panel.era_ranges(["e1", "e1", "e2", "e2", "e2", "e3"]) == (
        ("e1", 0, 2),
        ("e2", 2, 5),
        ("e3", 5, 6),
    )


def test_era_ranges_single_era() -> None:
    assert panel.era_ranges(["e1", "e1", "e1"]) == (("e1", 0, 3),)


def test_era_ranges_empty() -> None:
    assert panel.era_ranges([]) == ()


def test_era_ranges_only_groups_adjacent_rows() -> None:
    # A non-contiguous (unsorted) era list splits into one range per contiguous block.
    assert panel.era_ranges(["e1", "e2", "e1"]) == (("e1", 0, 1), ("e2", 1, 2), ("e1", 2, 3))


# --------------------------------------------------------------------------- #
# Recent-window > 200 eras
# --------------------------------------------------------------------------- #


def test_score_weight_matrix_recent_window_uses_last_200_of_many_eras() -> None:
    rng = np.random.default_rng(0)
    n_eras = 250
    rows_per_era = 3
    eras = [f"e{index:04d}" for index in range(n_eras) for _ in range(rows_per_era)]
    ranges = panel.era_ranges(eras)
    n_rows = n_eras * rows_per_era
    prediction_matrix = rng.random((n_rows, 1))
    target_vector = rng.random(n_rows)
    benchmark_vector = rng.random(n_rows)
    weights = np.asarray([[1.0]], dtype=np.float64)

    per_era = panel.per_era_score_matrices(
        prediction_matrix=prediction_matrix,
        target_vector=target_vector,
        benchmark_vector=benchmark_vector,
        era_ranges=ranges,
        weight_matrix=weights,
    )
    summary = panel.score_weight_matrix(
        prediction_matrix=prediction_matrix,
        target_vector=target_vector,
        benchmark_vector=benchmark_vector,
        era_ranges=ranges,
        weight_matrix=weights,
    )

    bmc_scores = per_era["bmc"][:, 0]
    np.testing.assert_allclose(summary["bmc_last_200_eras_mean"][0], np.nanmean(bmc_scores[-200:]))
    np.testing.assert_allclose(summary["bmc_mean"][0], np.nanmean(bmc_scores))
    # With 250 distinct eras the recent window is a strict subset, so the means differ.
    assert summary["bmc_mean"][0] != summary["bmc_last_200_eras_mean"][0]


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #


def test_summary_columns_ignores_nan_eras() -> None:
    values = np.asarray([[0.1], [np.nan], [0.3]], dtype=np.float64)
    summary = panel.summary_columns(values)
    np.testing.assert_allclose(summary["mean"][0], 0.2)
    np.testing.assert_allclose(summary["std"][0], 0.1)


def test_summary_columns_all_nan_column_is_nan() -> None:
    values = np.full((3, 1), np.nan, dtype=np.float64)
    summary = panel.summary_columns(values)
    assert np.isnan(summary["mean"][0])
    assert np.isnan(summary["sharpe"][0])
