from __future__ import annotations

import cloudpickle
import pandas as pd
import pytest

from numereng.features.training.errors import TrainingModelError
from numereng.features.training.target_transforms import (
    TargetTransformWrapper,
    apply_target_transform,
    prediction_inversion_kind,
    subtract_scaled_zscore_column,
)


def test_apply_target_transform_none_returns_original() -> None:
    y = pd.Series([1.0, 2.0, 3.0], name="target")
    X = pd.DataFrame({"era": ["era1", "era1", "era2"], "bench": [0.1, 0.2, 0.3]})

    transformed = apply_target_transform(y, X, None)
    assert transformed.equals(y)


def test_apply_target_transform_unknown_raises() -> None:
    y = pd.Series([1.0, 2.0, 3.0], name="target")
    X = pd.DataFrame({"era": ["era1", "era1", "era2"], "bench": [0.1, 0.2, 0.3]})

    with pytest.raises(TrainingModelError, match="training_target_transform_unknown"):
        apply_target_transform(y, X, {"type": "unknown"})


def test_subtract_scaled_zscore_column_changes_values() -> None:
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="target")
    X = pd.DataFrame(
        {
            "era": ["era1", "era1", "era2", "era2"],
            "bench": [0.0, 1.0, 2.0, 3.0],
        }
    )

    transformed = subtract_scaled_zscore_column(y, X, benchmark_col="bench", era_col="era", scale=0.1)
    assert len(transformed) == 4
    assert not transformed.equals(y)


def test_prediction_inversion_kind_is_identity_for_supported_transforms() -> None:
    assert prediction_inversion_kind(None) == "identity"
    assert prediction_inversion_kind({"type": "residual_to_benchmark"}) == "identity"
    assert prediction_inversion_kind("subtract_benchmark") == "identity"


def test_prediction_inversion_kind_rejects_unknown_transforms() -> None:
    with pytest.raises(TrainingModelError, match="training_target_transform_unknown"):
        prediction_inversion_kind({"type": "invented_transform"})


def test_target_transform_wrapper_round_trips_through_pickle() -> None:
    class _Model:
        accepts_era = True

        def predict(self, X, **kwargs):
            return [0.0] * len(X)

    wrapper = TargetTransformWrapper(_Model(), {"type": "residual_to_benchmark", "benchmark_col": "bench"})

    restored = cloudpickle.loads(cloudpickle.dumps(wrapper))

    assert restored.accepts_era is True
    assert restored.predict(pd.DataFrame({"a": [1.0, 2.0]})) == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# Era kwarg bridge: per-era transforms through the wrapper
# --------------------------------------------------------------------------- #


class _CapturingModel:
    accepts_era = True

    def __init__(self) -> None:
        self.fit_X: pd.DataFrame | None = None
        self.fit_y: pd.Series | None = None
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> _CapturingModel:
        self.fit_X = X
        self.fit_y = y
        self.fit_kwargs = dict(kwargs)
        return self


def _per_era_transform(per_era: bool) -> dict[str, object]:
    return {"type": "residual_to_benchmark", "benchmark_col": "bench", "per_era": per_era}


def _two_era_frame() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Within each era y is an exact affine function of bench with a different slope, so a
    # per-era residual is identically zero while a pooled residual is not.
    X = pd.DataFrame({"bench": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0], "f0": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([1.0, 2.0, 3.0, 10.0, 8.0, 6.0], name="target")
    era = pd.Series(["e1", "e1", "e1", "e2", "e2", "e2"], name="era")
    return X, y, era


def test_wrapper_per_era_uses_era_fit_kwarg_when_X_lacks_era_column() -> None:
    X, y, era = _two_era_frame()
    inner = _CapturingModel()
    wrapper = TargetTransformWrapper(inner, _per_era_transform(True))

    wrapper.fit(X, y, era=era)

    assert inner.fit_y is not None
    assert inner.fit_y.abs().max() < 1e-9, "per-era residual of an exact per-era affine fit must be ~0"
    assert inner.fit_X is X, "the wrapped model must still receive the original X"
    assert "era" not in X.columns
    assert inner.fit_kwargs["era"] is era


def test_wrapper_pooled_residual_differs_from_per_era() -> None:
    X, y, era = _two_era_frame()
    inner = _CapturingModel()
    wrapper = TargetTransformWrapper(inner, _per_era_transform(False))

    wrapper.fit(X, y, era=era)

    assert inner.fit_y is not None
    assert inner.fit_y.abs().max() > 1e-3, "pooled residual cannot be zero when era slopes differ"


def test_wrapper_per_era_without_era_kwarg_still_fails_loudly() -> None:
    X, y, _era = _two_era_frame()
    wrapper = TargetTransformWrapper(_CapturingModel(), _per_era_transform(True))
    with pytest.raises(TrainingModelError, match="training_target_transform_era_col_missing:era"):
        wrapper.fit(X, y)


def test_wrapper_era_kwarg_length_mismatch_raises() -> None:
    X, y, era = _two_era_frame()
    wrapper = TargetTransformWrapper(_CapturingModel(), _per_era_transform(True))
    with pytest.raises(TrainingModelError, match="training_target_transform_era_length_mismatch"):
        wrapper.fit(X, y, era=era.iloc[:-1])


def test_wrapper_prefers_existing_era_column_over_kwarg() -> None:
    X, y, era = _two_era_frame()
    X = X.assign(era=era.to_numpy())
    inner = _CapturingModel()
    wrapper = TargetTransformWrapper(inner, _per_era_transform(True))

    wrapper.fit(X, y, era=pd.Series(["zz"] * len(X)))

    assert inner.fit_y is not None
    assert inner.fit_y.abs().max() < 1e-9
