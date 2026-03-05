"""LightGBM model adapter for training workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from numereng.features.training.errors import TrainingModelError


class LGBMRegressor:
    """Minimal wrapper exposing fit/predict for training pipeline usage."""

    def __init__(self, feature_cols: list[str] | None = None, **params: Any) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise TrainingModelError("training_model_backend_missing_lightgbm") from exc

        self._lgb = lgb
        resolved_params = dict(params)
        if "verbosity" not in resolved_params and "verbose" not in resolved_params:
            resolved_params["verbosity"] = -1
        self._params: dict[str, Any] = resolved_params
        self._model = lgb.LGBMRegressor(**resolved_params)
        self._feature_cols = feature_cols

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> LGBMRegressor:
        """Fit model, falling back to CPU when GPU backend is unavailable."""
        filtered_X = self._filter_features(X, self._feature_cols)
        try:
            self._model.fit(filtered_X, y, **kwargs)
        except self._lgb.basic.LightGBMError as exc:
            if self._should_fallback_to_cpu(exc):
                self._params["device_type"] = "cpu"
                self._model = self._lgb.LGBMRegressor(**self._params)
                self._model.fit(filtered_X, y, **kwargs)
            else:
                raise TrainingModelError("training_model_fit_failed") from exc
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict model outputs for feature matrix."""
        filtered_X = self._filter_features(X, self._feature_cols)
        return self._model.predict(filtered_X)

    def _should_fallback_to_cpu(self, exc: Exception) -> bool:
        message = str(exc)
        device_type = str(self._params.get("device_type", "")).lower()
        return device_type == "gpu" and "GPU Tree Learner was not enabled" in message

    @staticmethod
    def _filter_features(X: pd.DataFrame, feature_cols: list[str] | None) -> pd.DataFrame:
        if not feature_cols or not hasattr(X, "columns"):
            return X

        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise TrainingModelError(
                "training_model_feature_columns_missing:"
                + ",".join(missing[:5])
                + (",..." if len(missing) > 5 else "")
            )

        return X[feature_cols]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
