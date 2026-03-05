from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

import numereng.features.training.cv as cv_module
from numereng.features.training.errors import TrainingConfigError
from numereng.features.training.models import build_model_data_loader


class _FakeModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _FakeModel:
        _ = (X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.42, dtype=float)


def test_era_cv_splits_official_walkforward() -> None:
    eras = [str(i) for i in range(1, 625)]
    splits = cv_module.era_cv_splits(
        eras=eras,
        embargo=8,
        mode="official_walkforward",
        chunk_size=156,
        min_train_size=1,
    )

    assert len(splits) == 3
    first_train, first_val = splits[0]
    second_train, second_val = splits[1]
    assert first_train[0] == "1"
    assert first_train[-1] == "148"
    assert first_val[0] == "157"
    assert first_val[-1] == "312"
    assert second_train[-1] == "304"
    assert second_val[0] == "313"


def test_era_cv_splits_train_validation_holdout() -> None:
    splits = cv_module.era_cv_splits(
        eras=["1", "2", "3", "4"],
        embargo=0,
        mode="train_validation_holdout",
        min_train_size=1,
        holdout_train_eras=["1", "2"],
        holdout_val_eras=["3", "4"],
    )
    assert splits == [(["1", "2"], ["3", "4"])]


def test_era_cv_splits_holdout_overlap_raises() -> None:
    with pytest.raises(TrainingConfigError, match="training_cv_holdout_eras_overlap"):
        cv_module.era_cv_splits(
            eras=["1", "2", "3"],
            mode="train_validation_holdout",
            holdout_train_eras=["1", "2"],
            holdout_val_eras=["2", "3"],
        )


def test_build_oof_predictions_train_validation_holdout(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: _FakeModel())

    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["1", "1", "2", "2", "3", "3"],
            "target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_1": [1, 2, 3, 4, 5, 6],
            "benchmark": [0.5, 0.6, 0.7, 0.8, 0.2, 0.3],
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1", "era", "benchmark"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    predictions, meta = cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=loader,
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

    assert not predictions.empty
    assert set(["id", "era", "target", "prediction", "cv_fold"]).issubset(predictions.columns)
    assert cast(int, meta["folds_used"]) == 1
    assert meta["mode"] == "train_validation_holdout"
    assert "max_train_eras" not in meta


def test_build_oof_predictions_fold_descriptors_are_deterministic(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: _FakeModel())

    full = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(8)],
            "era": ["1", "1", "2", "2", "3", "3", "4", "4"],
            "target": [float(idx) / 10.0 for idx in range(8)],
            "feature_1": list(range(8)),
            "benchmark": [0.1] * 8,
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1", "era", "benchmark"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    _, meta_first = cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=loader,
        model_type="LGBMRegressor",
        model_params={},
        model_config={},
        cv_config={
            "embargo": 0,
            "mode": "train_validation_holdout",
            "min_train_size": 1,
            "train_eras": ["1", "2", "3"],
            "val_eras": ["4"],
        },
        id_col="id",
        era_col="era",
        target_col="target",
        feature_cols=["feature_1"],
    )
    _, meta_second = cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=loader,
        model_type="LGBMRegressor",
        model_params={},
        model_config={},
        cv_config={
            "embargo": 0,
            "mode": "train_validation_holdout",
            "min_train_size": 1,
            "train_eras": ["1", "2", "3"],
            "val_eras": ["4"],
        },
        id_col="id",
        era_col="era",
        target_col="target",
        feature_cols=["feature_1"],
    )

    assert meta_first["folds"] == meta_second["folds"]


def test_build_full_history_predictions(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: _FakeModel())

    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["1", "1", "2", "2"],
            "target": [0.1, 0.2, 0.3, 0.4],
            "feature_1": [1, 2, 3, 4],
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1", "era"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    predictions, meta = cv_module.build_full_history_predictions(
        eras=full["era"],
        data_loader=loader,
        model_type="LGBMRegressor",
        model_params={},
        model_config={},
        id_col="id",
        era_col="era",
        target_col="target",
        feature_cols=["feature_1"],
    )

    assert list(predictions.columns) == ["id", "era", "target", "prediction"]
    assert len(predictions) == len(full)
    assert cast(str, meta["mode"]) == "full_history"
    assert cast(int, meta["folds_used"]) == 1
    assert "max_train_eras" not in meta
