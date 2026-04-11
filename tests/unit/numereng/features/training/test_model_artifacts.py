from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from numereng.features.training.model_artifacts import (
    ModelArtifactManifest,
    load_model_artifact,
    save_model_artifact,
)


class _RoundTripRegressor:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["feature_a"] + X["feature_b"]


def test_model_artifact_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    manifest = ModelArtifactManifest(
        run_id="run-1",
        model_type="LGBMRegressor",
        data_version="v5.2",
        dataset_variant="non_downsampled",
        feature_set="small",
        target_col="target",
        era_col="era",
        id_col="id",
        feature_cols=("feature_a", "feature_b"),
        model_upload_compatible=True,
    )

    artifact_path, manifest_path = save_model_artifact(
        run_dir=run_dir,
        model=_RoundTripRegressor(),
        manifest=manifest,
    )
    loaded = load_model_artifact(run_dir=run_dir)

    frame = pd.DataFrame({"feature_a": [0.1, 0.2], "feature_b": [0.3, 0.4]})
    assert artifact_path.is_file()
    assert manifest_path.is_file()
    assert loaded.manifest.run_id == "run-1"
    assert loaded.model.predict(frame).tolist() == pytest.approx([0.4, 0.6])
