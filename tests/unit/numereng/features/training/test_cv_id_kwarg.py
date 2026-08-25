"""`accepts_id` fit-kwarg plumbing in `cv.py`, mirroring `test_cv_era_kwarg.py`.

`accepts_id` is the opt-in hook that hands a model the Numerai row-`id` Series at fit time. It
is needed because `X` reaches `fit` as a column slice on a positional index -- the id lives only
in `ModelDataBatch.id` -- so a model joining external per-row data has no other correct key.

Covered:
1. both fit paths (fold OOF and full history) pass an id aligned with X, with the right values
2. `accepts_id` and `accepts_era` are independent: either alone works
3. a model declaring neither is still called `fit(X, y)` -- signatures reject **kwargs so a leak
   would raise TypeError
4. `predict` is never given an id
5. no id is passed when the loader carries no id column
6. the kwarg reaches the inner model through `TargetTransformWrapper`

USAGE:
    uv run pytest tests/unit/numereng/features/training/test_cv_id_kwarg.py -q
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import numereng.features.training.cv as cv_module
from numereng.features.training.models import build_model_data_loader
from numereng.features.training.target_transforms import TargetTransformWrapper

# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class _IdAwareModel:
    accepts_era = True
    accepts_id = True

    def __init__(self) -> None:
        self.fit_ids: list[pd.Series] = []
        self.fit_eras: list[pd.Series] = []
        self.predict_kwargs: list[dict[str, Any]] = []
        self.aligned: list[bool] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> _IdAwareModel:
        row_id = kwargs["id"]
        self.fit_ids.append(row_id)
        self.fit_eras.append(kwargs["era"])
        self.aligned.append(len(row_id) == len(X) and row_id.index.equals(X.index))
        return self

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        self.predict_kwargs.append(dict(kwargs))
        return np.full(len(X), 0.42, dtype=float)


class _IdOnlyModel:
    """`accepts_id` without `accepts_era`: the two flags must be independent."""

    accepts_id = True

    def __init__(self) -> None:
        self.fit_kwargs: list[dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> _IdOnlyModel:
        self.fit_kwargs.append(dict(kwargs))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.42, dtype=float)


class _PlainModel:
    """Signatures deliberately reject **kwargs so any id leak raises TypeError."""

    def __init__(self) -> None:
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _PlainModel:
        self.fit_calls += 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_calls += 1
        return np.full(len(X), 0.42, dtype=float)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


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


def _loader(full: pd.DataFrame, *, id_col: str | None = "id") -> Any:
    return build_model_data_loader(
        full=full,
        x_cols=["feature_1", "benchmark"],
        era_col="era",
        target_col="target",
        id_col=id_col,
    )


def _run_oof(full: pd.DataFrame, *, id_col: str | None = "id") -> None:
    cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=_loader(full, id_col=id_col),
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
        id_col=id_col,
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


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_accepts_id_model_receives_aligned_id_on_both_paths(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()

    fold_model = _IdAwareModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: fold_model)
    _run_oof(full)

    assert len(fold_model.fit_ids) == 1
    assert list(fold_model.fit_ids[0]) == ["id-0", "id-1", "id-2", "id-3"]
    assert list(fold_model.fit_eras[0]) == ["1", "1", "2", "2"]
    assert all(fold_model.aligned)

    full_model = _IdAwareModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: full_model)
    _run_full_history(full)

    assert len(full_model.fit_ids) == 1
    assert list(full_model.fit_ids[0]) == [f"id-{idx}" for idx in range(6)]
    assert all(full_model.aligned)


def test_predict_is_never_given_an_id(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()
    model = _IdAwareModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: model)
    _run_oof(full)

    assert len(model.predict_kwargs) == 1
    assert "id" not in model.predict_kwargs[0]
    assert "era" in model.predict_kwargs[0]


def test_accepts_id_is_independent_of_accepts_era(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()
    model = _IdOnlyModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: model)
    _run_oof(full)

    assert len(model.fit_kwargs) == 1
    assert list(model.fit_kwargs[0]) == ["id"]
    assert list(model.fit_kwargs[0]["id"]) == ["id-0", "id-1", "id-2", "id-3"]


def test_model_without_accepts_id_is_called_without_id_kwarg(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()

    fold_model = _PlainModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: fold_model)
    _run_oof(full)
    assert fold_model.fit_calls == 1
    assert fold_model.predict_calls == 1

    full_model = _PlainModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: full_model)
    _run_full_history(full)
    assert full_model.fit_calls == 1
    assert full_model.predict_calls == 1


def test_no_id_is_passed_when_the_loader_has_no_id_column(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()
    model = _IdOnlyModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: model)
    _run_oof(full, id_col=None)

    assert len(model.fit_kwargs) == 1
    assert model.fit_kwargs[0] == {}


def test_accepts_id_reaches_inner_model_through_target_transform_wrapper(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    full = _frame()
    inner = _IdAwareModel()
    wrapper = TargetTransformWrapper(
        inner,
        {"type": "residual_to_benchmark", "benchmark_col": "benchmark", "per_era": False},
    )

    assert getattr(wrapper, "accepts_id", False) is True

    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: wrapper)
    _run_oof(full)

    assert list(inner.fit_ids[0]) == ["id-0", "id-1", "id-2", "id-3"]
    assert all(inner.aligned)
