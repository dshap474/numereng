from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import numereng.features.training.cv as cv_module
from numereng.features.training import model_factory
from numereng.features.training.errors import TrainingConfigError, TrainingDataError
from numereng.features.training.models import build_model_data_loader


class _FakeModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _FakeModel:
        _ = (X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.42, dtype=float)


class _RecordingModel:
    def __init__(self) -> None:
        self.fit_rows: list[int] = []
        self.predict_rows: list[int] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _RecordingModel:
        self.fit_rows.append(len(X))
        assert len(X) == len(y)
        assert not y.isna().any()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_rows.append(len(X))
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
            embargo=0,
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
        x_cols=["feature_1", "benchmark"],
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
        x_cols=["feature_1", "benchmark"],
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
        x_cols=["feature_1"],
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
    assert cast(str, meta["mode"]) == "full_history_refit"
    assert cast(int, meta["folds_used"]) == 1
    assert "max_train_eras" not in meta


def test_build_oof_predictions_filters_unlabeled_rows_before_fit_and_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    model = _RecordingModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: model)

    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["1", "1", "2", "2", "3", "3"],
            "target": [0.1, np.nan, 0.3, 0.4, 0.5, np.nan],
            "feature_1": [1, 2, 3, 4, 5, 6],
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
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

    assert model.fit_rows == [3]
    assert model.predict_rows == [1]
    assert predictions["id"].tolist() == ["e"]
    assert predictions["target"].tolist() == [0.5]
    assert cast(list[dict[str, object]], meta["folds"])[0]["train_rows"] == 3
    assert cast(list[dict[str, object]], meta["folds"])[0]["train_rows_unlabeled_dropped"] == 1
    assert cast(list[dict[str, object]], meta["folds"])[0]["val_rows"] == 1
    assert cast(list[dict[str, object]], meta["folds"])[0]["val_rows_unlabeled_dropped"] == 1


def test_build_oof_predictions_raises_when_train_fold_has_no_labeled_rows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: _RecordingModel())

    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["1", "1", "2", "2"],
            "target": [np.nan, np.nan, 0.3, 0.4],
            "feature_1": [1, 2, 3, 4],
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    with pytest.raises(TrainingDataError, match="training_target_rows_all_unlabeled:split=train:target=target:fold=0"):
        cv_module.build_oof_predictions(
            eras=full["era"],
            data_loader=loader,
            model_type="LGBMRegressor",
            model_params={},
            model_config={},
            cv_config={
                "embargo": 0,
                "mode": "train_validation_holdout",
                "min_train_size": 1,
                "train_eras": ["1"],
                "val_eras": ["2"],
            },
            id_col="id",
            era_col="era",
            target_col="target",
            feature_cols=["feature_1"],
        )


def test_build_oof_predictions_raises_when_validation_fold_has_no_labeled_rows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: _RecordingModel())

    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["1", "1", "2", "2"],
            "target": [0.1, 0.2, np.nan, np.nan],
            "feature_1": [1, 2, 3, 4],
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    with pytest.raises(
        TrainingDataError, match="training_target_rows_all_unlabeled:split=validation:target=target:fold=0"
    ):
        cv_module.build_oof_predictions(
            eras=full["era"],
            data_loader=loader,
            model_type="LGBMRegressor",
            model_params={},
            model_config={},
            cv_config={
                "embargo": 0,
                "mode": "train_validation_holdout",
                "min_train_size": 1,
                "train_eras": ["1"],
                "val_eras": ["2"],
            },
            id_col="id",
            era_col="era",
            target_col="target",
            feature_cols=["feature_1"],
        )


def test_build_full_history_predictions_filters_unlabeled_rows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    model = _RecordingModel()
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: model)

    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["1", "1", "2", "2"],
            "target": [0.1, np.nan, 0.3, 0.4],
            "feature_1": [1, 2, 3, 4],
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
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

    assert model.fit_rows == [3]
    assert model.predict_rows == [3]
    assert predictions["id"].tolist() == ["a", "c", "d"]
    assert cast(list[dict[str, object]], meta["folds"])[0]["train_rows_unlabeled_dropped"] == 1


def test_build_oof_predictions_xgboost_trains_after_unlabeled_rows_are_filtered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xgboost")

    custom_models_root = tmp_path / "custom_models"
    custom_models_root.mkdir()
    source_module = Path(model_factory.__file__).resolve().parents[1] / "models" / "custom_models" / "xgboost_model.py"
    (custom_models_root / "xgboost_model.py").write_text(source_module.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(model_factory, "_resolve_custom_models_root", lambda: custom_models_root)

    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["1", "1", "2", "2", "3", "3"],
            "target": [0.1, np.nan, 0.3, 0.4, 0.5, np.nan],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    predictions, meta = cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=loader,
        model_type="XGBoostRegressor",
        model_params={
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cpu",
            "n_estimators": 2,
            "learning_rate": 0.1,
            "random_state": 0,
        },
        model_config={"module_path": "xgboost_model.py"},
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

    assert predictions["id"].tolist() == ["e"]
    assert cast(list[dict[str, object]], meta["folds"])[0]["train_rows_unlabeled_dropped"] == 1


def test_build_oof_predictions_invokes_fold_callback_serial(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: _FakeModel())

    full = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(12)],
            "era": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5", "6", "6"],
            "target": [float(idx) / 10.0 for idx in range(12)],
            "feature_1": list(range(12)),
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    completed: list[int] = []

    def _on_fold_complete(predictions: pd.DataFrame, metadata: dict[str, object]) -> None:
        assert "cv_fold" in predictions.columns
        completed.append(int(metadata["fold"]))

    cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=loader,
        model_type="LGBMRegressor",
        model_params={},
        model_config={},
        cv_config={"embargo": 0, "mode": "official_walkforward", "chunk_size": 2, "min_train_size": 1},
        id_col="id",
        era_col="era",
        target_col="target",
        feature_cols=["feature_1"],
        on_fold_complete=_on_fold_complete,
    )

    assert completed == [0, 1]


def test_build_oof_predictions_invokes_fold_start_callback_serial(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: _FakeModel())

    full = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(12)],
            "era": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5", "6", "6"],
            "target": [float(idx) / 10.0 for idx in range(12)],
            "feature_1": list(range(12)),
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    started: list[tuple[int, object]] = []

    def _on_fold_start(metadata: dict[str, object]) -> None:
        started.append((int(metadata["fold"]), metadata["val_interval"]))

    cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=loader,
        model_type="LGBMRegressor",
        model_params={},
        model_config={},
        cv_config={"embargo": 0, "mode": "official_walkforward", "chunk_size": 2, "min_train_size": 1},
        id_col="id",
        era_col="era",
        target_col="target",
        feature_cols=["feature_1"],
        on_fold_start=_on_fold_start,
    )

    assert started == [
        (0, {"start": "3", "end": "4", "start_index": 2, "end_index": 3}),
        (1, {"start": "5", "end": "6", "start_index": 4, "end_index": 5}),
    ]


def test_build_oof_predictions_invokes_fold_callback_parallel_in_order(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv_module, "build_model", lambda *args, **kwargs: _FakeModel())

    full = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(12)],
            "era": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5", "6", "6"],
            "target": [float(idx) / 10.0 for idx in range(12)],
            "feature_1": list(range(12)),
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    completed: list[int] = []

    def _on_fold_complete(predictions: pd.DataFrame, metadata: dict[str, object]) -> None:
        assert "cv_fold" in predictions.columns
        completed.append(int(metadata["fold"]))

    cv_module.build_oof_predictions(
        eras=full["era"],
        data_loader=loader,
        model_type="LGBMRegressor",
        model_params={},
        model_config={},
        cv_config={"embargo": 0, "mode": "official_walkforward", "chunk_size": 2, "min_train_size": 1},
        id_col="id",
        era_col="era",
        target_col="target",
        feature_cols=["feature_1"],
        parallel_folds=2,
        parallel_backend="joblib",
        memmap_enabled=False,
        on_fold_complete=_on_fold_complete,
    )

    assert completed == [0, 1]
