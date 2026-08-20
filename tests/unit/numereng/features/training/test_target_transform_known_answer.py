"""Known-answer tests for `residualize_to_column` global-beta residualization.

Every fixture is chosen so the OLS slope and the residual vector are exact rationals that
are also exactly representable in binary floating point, so the expected values below are
literal hand-computed arrays rather than a second implementation of the transform.

Covers `per_era=False` with `fit_intercept=True` / `fit_intercept=False`, and the
`proportion=0.5` blend between the raw target and the residual.

USAGE:
    uv run pytest tests/unit/numereng/features/training/test_target_transform_known_answer.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numereng.features.training.target_transforms import residualize_to_column

# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

# y = [1, 2, 3, 4], b = [0, 1, 0, 1] (one era label, unused when per_era=False).
_Y = pd.Series([1.0, 2.0, 3.0, 4.0], name="target")
_X = pd.DataFrame(
    {
        "era": ["0001", "0001", "0001", "0001"],
        "bench": [0.0, 1.0, 0.0, 1.0],
    }
)


# --------------------------------------------------------------------------- #
# Global beta, with intercept
# --------------------------------------------------------------------------- #


def test_global_residual_with_intercept_is_exact() -> None:
    # mean(y) = 2.5, mean(b) = 0.5
    # y_c = [-1.5, -0.5,  0.5,  1.5]
    # b_c = [-0.5,  0.5, -0.5,  0.5]
    # beta = sum(b_c * y_c) / sum(b_c ** 2)
    #      = (0.75 - 0.25 - 0.25 + 0.75) / (0.25 + 0.25 + 0.25 + 0.25)
    #      = 1.00 / 1.00 = 1.0
    # resid = y_c - 1.0 * b_c = [-1.5 + 0.5, -0.5 - 0.5, 0.5 + 0.5, 1.5 - 0.5]
    expected = np.asarray([-1.0, -1.0, 1.0, 1.0], dtype="float64")

    residual = residualize_to_column(_Y, _X, benchmark_col="bench", per_era=False, fit_intercept=True)

    assert list(residual.index) == list(_Y.index)
    assert residual.name == "target"
    np.testing.assert_allclose(residual.to_numpy(dtype="float64"), expected, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Global beta, no intercept
# --------------------------------------------------------------------------- #


def test_global_residual_without_intercept_is_exact() -> None:
    # No centering, so beta is the raw no-intercept OLS slope:
    # beta = sum(b * y) / sum(b ** 2) = (0*1 + 1*2 + 0*3 + 1*4) / (0 + 1 + 0 + 1) = 6 / 2 = 3.0
    # resid = y - 3.0 * b = [1 - 0, 2 - 3, 3 - 0, 4 - 3]
    expected = np.asarray([1.0, -1.0, 3.0, 1.0], dtype="float64")

    residual = residualize_to_column(_Y, _X, benchmark_col="bench", per_era=False, fit_intercept=False)

    np.testing.assert_allclose(residual.to_numpy(dtype="float64"), expected, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Proportion blend
# --------------------------------------------------------------------------- #


def test_global_residual_half_proportion_blends_raw_and_residual() -> None:
    # proportion=0.5 -> 0.5 * y + 0.5 * resid, reusing the with-intercept residual above:
    # 0.5 * [1, 2, 3, 4] + 0.5 * [-1, -1, 1, 1]
    #   = [0.5, 1.0, 1.5, 2.0] + [-0.5, -0.5, 0.5, 0.5]
    expected = np.asarray([0.0, 0.5, 2.0, 2.5], dtype="float64")

    blended = residualize_to_column(
        _Y,
        _X,
        benchmark_col="bench",
        per_era=False,
        fit_intercept=True,
        proportion=0.5,
    )

    np.testing.assert_allclose(blended.to_numpy(dtype="float64"), expected, rtol=0.0, atol=1e-12)
