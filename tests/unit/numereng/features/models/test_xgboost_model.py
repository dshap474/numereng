from __future__ import annotations

import sys
import types
from typing import Any, cast

import pandas as pd
import pytest

from numereng.features.models.xgboost import XGBoostRegressor
from numereng.features.training.errors import TrainingModelError


class _RecordingBackendModel:
    def __init__(self, **params: object) -> None:
        self.params = params
        self.fit_columns: list[str] | None = None
        self.predict_columns: list[str] | None = None
        self.set_param_calls: list[dict[str, object]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> _RecordingBackendModel:
        self.fit_columns = list(cast(Any, X).columns)
        self.fit_kwargs = kwargs
        self.fit_rows = len(X)
        self.fit_target_rows = len(y)
        return self

    def predict(self, X: pd.DataFrame, **kwargs: object) -> list[float]:
        self.predict_columns = list(cast(Any, X).columns)
        self.predict_kwargs = kwargs
        return [0.0] * len(X)

    def get_booster(self) -> _RecordingBackendModel:
        return self

    def set_param(self, params: dict[str, object]) -> None:
        self.set_param_calls.append(params)


def _install_xgboost_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("xgboost")
    setattr(module, "XGBRegressor", _RecordingBackendModel)
    monkeypatch.setitem(sys.modules, "xgboost", module)


def test_xgboost_wrapper_filters_feature_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_xgboost_backend(monkeypatch)
    model = XGBoostRegressor(feature_cols=["feature_a", "feature_c"], max_depth=4)
    frame = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0], "feature_c": [5.0, 6.0]})
    target = pd.Series([0.1, 0.2])

    model.fit(frame, target)
    predictions = model.predict(frame)

    assert model._model.fit_columns == ["feature_a", "feature_c"]
    assert model._model.predict_columns == ["feature_a", "feature_c"]
    assert predictions == [0.0, 0.0]


def test_xgboost_wrapper_missing_feature_columns_raise_training_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_xgboost_backend(monkeypatch)
    model = XGBoostRegressor(feature_cols=["feature_a", "feature_b"])
    frame = pd.DataFrame({"feature_a": [1.0]})

    with pytest.raises(TrainingModelError, match="training_model_feature_columns_missing"):
        model.predict(frame)


def test_xgboost_wrapper_moves_gpu_booster_to_cpu_before_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_xgboost_backend(monkeypatch)
    model = XGBoostRegressor(device="cuda", tree_method="hist")
    frame = pd.DataFrame({"feature_a": [1.0, 2.0]})
    target = pd.Series([0.1, 0.2])

    model.fit(frame, target)
    predictions = model.predict(frame)
    repeated_predictions = model.predict(frame)

    assert model._model.set_param_calls == [{"device": "cpu"}]
    assert predictions == [0.0, 0.0]
    assert repeated_predictions == [0.0, 0.0]
