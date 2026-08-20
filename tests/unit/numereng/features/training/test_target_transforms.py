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
