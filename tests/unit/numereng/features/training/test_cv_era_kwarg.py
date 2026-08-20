from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import numereng.features.training.cv as cv_module
from numereng.features.training.models import build_model_data_loader
from numereng.features.training.target_transforms import TargetTransformWrapper


class _EraAwareModel:
    accepts_era = True

    def __init__(self) -> None:
        self.fit_eras: list[pd.Series] = []
        self.predict_eras: list[pd.Series] = []
        self.aligned: list[bool] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> _EraAwareModel:
        era = kwargs["era"]
        self.fit_eras.append(era)
        self.aligned.append(len(era) == len(X) and era.index.equals(X.index))
        return self

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        era = kwargs["era"]
        self.predict_eras.append(era)
        self.aligned.append(len(era) == len(X) and era.index.equals(X.index))
        return np.full(len(X), 0.42, dtype=float)


class _EraBlindModel:
    """Signatures deliberately reject **kwargs so any era leak raises TypeError."""

    def __init__(self) -> None:
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _EraBlindModel:
        self.fit_calls += 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_calls += 1
        return np.full(len(X), 0.42, dtype=float)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(6)],
            "era": ["1", "1", "2", "2", "3", "3"],
            "target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_1": [1, 2, 3, 4, 5, 6],
            "benchmark": [0.5, 0.6, 0.7, 0.8, 0.2, 0.3],
        }
    )


def _loader(full: pd.DataFrame) -> Any:
    return build_model_data_loader(
        full=full,
        x_cols=["feature_1", "benchmark"],
        era_col="era",
        target_col="target",
        id_col="id",
    )


def _run_oof(full: pd.DataFrame) -> None:
    cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=_loader(full),
        model_type="LGBMRegressor",
        model_params={},
        model_config={},
        cv_config={
            "embargo": 0,
            "mode": "train_validation_holdout",
            "min_train_size": 1,
            "train_eras": ["1", "2"],
            "val_eras": ["3"],
        },
        id_col="id",
        era_col="era",
        target_col="target",
        feature_cols=["feature_1"],
    )


def _run_full_history(full: pd.DataFrame) -> None:
    cv_module.build_full_history_predictions(
        eras=full["era"],
        data_loader=_loader(full),
        model_type="LGBMRegressor",
        model_params={},
        model_config={},
        id_col="id",
        era_col="era",
        target_col="target",
        feature_cols=["feature_1"],
    )


def test_accepts_era_model_receives_aligned_era_on_both_paths(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()

    fold_model = _EraAwareModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: fold_model)
    _run_oof(full)

    assert len(fold_model.fit_eras) == 1
    assert len(fold_model.predict_eras) == 1
    assert list(fold_model.fit_eras[0]) == ["1", "1", "2", "2"]
    assert list(fold_model.predict_eras[0]) == ["3", "3"]
    assert all(fold_model.aligned)

    full_model = _EraAwareModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: full_model)
    _run_full_history(full)

    assert len(full_model.fit_eras) == 1
    assert len(full_model.predict_eras) == 1
    assert list(full_model.fit_eras[0]) == ["1", "1", "2", "2", "3", "3"]
    assert list(full_model.predict_eras[0]) == ["1", "1", "2", "2", "3", "3"]
    assert all(full_model.aligned)


def test_model_without_accepts_era_is_called_without_era_kwarg(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()

    fold_model = _EraBlindModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: fold_model)
    _run_oof(full)

    assert fold_model.fit_calls == 1
    assert fold_model.predict_calls == 1

    full_model = _EraBlindModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: full_model)
    _run_full_history(full)

    assert full_model.fit_calls == 1
    assert full_model.predict_calls == 1


def test_accepts_era_reaches_inner_model_through_target_transform_wrapper(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()
    inner = _EraAwareModel()
    wrapper = TargetTransformWrapper(
        inner,
        {"type": "residual_to_benchmark", "benchmark_col": "benchmark", "per_era": False},
    )

    assert getattr(wrapper, "accepts_era", False) is True

    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: wrapper)
    _run_oof(full)

    assert list(inner.fit_eras[0]) == ["1", "1", "2", "2"]
    assert list(inner.predict_eras[0]) == ["3", "3"]
    assert all(inner.aligned)
