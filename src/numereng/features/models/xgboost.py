"""XGBoost model adapter for training workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from numereng.features.training.errors import TrainingModelError


class XGBoostRegressor:
    """Thin wrapper around ``xgboost.XGBRegressor``."""

    def __init__(self, feature_cols: list[str] | None = None, **params: Any) -> None:
        try:
            import xgboost

            regressor_cls = xgboost.XGBRegressor
        except Exception as exc:  # noqa: BLE001
            raise TrainingModelError("training_model_backend_missing_xgboost") from exc

        self._feature_cols = feature_cols
        self._gpu_device = str(params.get("device", "")).lower()
        self._booster_moved_to_cpu = False
        self._model = regressor_cls(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> XGBoostRegressor:
        filtered_X = self._filter_features(X, self._feature_cols)
        self._model.fit(filtered_X, y, **kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        filtered_X = self._filter_features(X, self._feature_cols)
        if (
            not self._booster_moved_to_cpu
            and (self._gpu_device.startswith("cuda") or self._gpu_device == "gpu")
            and hasattr(self._model, "get_booster")
        ):
            self._model.get_booster().set_param({"device": "cpu"})
            self._booster_moved_to_cpu = True
        return self._model.predict(filtered_X)

    @staticmethod
    def _filter_features(
        X: pd.DataFrame,
        feature_cols: list[str] | None,
    ) -> pd.DataFrame:
        if not feature_cols or not hasattr(X, "columns"):
            return X

        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise TrainingModelError(
                "training_model_feature_columns_missing:" + ",".join(missing[:5]) + (",..." if len(missing) > 5 else "")
            )

        return X[feature_cols]
