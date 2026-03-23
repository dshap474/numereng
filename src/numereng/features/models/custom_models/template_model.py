"""Copy-me template for Numereng custom model modules."""

from __future__ import annotations

from typing import Any

import pandas as pd

from numereng.features.training.errors import TrainingModelError


class TemplateModel:
    """Rename this class and registry key before using it in a real config."""

    def __init__(self, feature_cols: list[str] | None = None, **params: Any) -> None:
        # Numereng passes feature_cols from the training pipeline. Keep this field and
        # use it to filter X so the adapter works with the standard feature routing.
        self._feature_cols = feature_cols
        self._params = params

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> TemplateModel:
        filtered_X = self._filter_features(X, self._feature_cols)
        _ = (filtered_X, y, kwargs)
        # Replace this no-op with your estimator's real training call.
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        filtered_X = self._filter_features(X, self._feature_cols)
        # Replace this placeholder with your estimator's real inference call.
        return [0.0] * len(filtered_X)

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


# Rename both sides of this mapping to your real model type.
MODEL_REGISTRY = {"TemplateModel": TemplateModel}
