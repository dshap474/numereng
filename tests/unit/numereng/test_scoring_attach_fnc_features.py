from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from numereng.features.training.scoring.metrics import attach_fnc_features


def test_attach_fnc_features_does_not_emit_fragmentation_warning() -> None:
    # Use a wide feature block to reproduce the fragmentation warning seen in real runs.
    # The fix should attach the feature block in one shot (concat/join) and avoid warnings.
    n_rows = 50
    n_features = 600
    feature_cols = [f"feature_{idx}" for idx in range(n_features)]
    ids = [f"id_{idx:04d}" for idx in range(n_rows)]
    eras = ["0001"] * n_rows

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        predictions = pd.DataFrame(
            {
                "id": ids,
                "era": eras,
                "target_ender_20": np.linspace(0.0, 1.0, n_rows),
                "prediction": np.linspace(1.0, 0.0, n_rows),
            }
        )

        # Build the wide feature frame in one shot to avoid fragmentation warnings in test setup.
        feature_values = np.linspace(0.0, 1.0, n_rows, dtype="float32")
        feature_matrix = np.tile(feature_values.reshape(-1, 1), (1, n_features))
        feature_frame = pd.concat(
            [
                pd.DataFrame({"id": ids, "era": eras}),
                pd.DataFrame(feature_matrix, columns=feature_cols),
            ],
            axis=1,
        )

        attached = attach_fnc_features(
            predictions,
            feature_frame,
            feature_cols=feature_cols,
            era_col="era",
            id_col="id",
        )

    assert len(attached) == n_rows
    assert "id" in attached.columns
    assert "prediction" in attached.columns
    assert set(feature_cols).issubset(attached.columns)
