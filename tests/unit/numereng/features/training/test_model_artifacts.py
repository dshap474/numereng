"""Unit tests for persisted model artifacts, including custom-plugin module rebinding.

USAGE:
    uv run pytest tests/unit/numereng/features/training/test_model_artifacts.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import numereng.features.training.model_factory as model_factory
from numereng.features.training.model_artifacts import (
    ModelArtifactError,
    ModelArtifactManifest,
    load_model_artifact,
    save_model_artifact,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_CUSTOM_PLUGIN = """
import numpy as np


class SyntheticPluginRegressor:
    def __init__(self, scale=2.0, feature_cols=None):
        self.scale = scale
        self.feature_cols = feature_cols

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        return np.asarray(X["feature_a"], dtype=float) * self.scale


MODEL_REGISTRY = {"SyntheticPluginRegressor": SyntheticPluginRegressor}
"""


class _RoundTripRegressor:
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["feature_a"] + X["feature_b"]


def _manifest(*, uses_custom_module: bool = False) -> ModelArtifactManifest:
    return ModelArtifactManifest(
        run_id="run-1",
        model_type="LGBMRegressor" if not uses_custom_module else "SyntheticPluginRegressor",
        data_version="v5.2",
        dataset_variant="non_downsampled",
        feature_set="small",
        target_col="target",
        era_col="era",
        id_col="id",
        feature_cols=("feature_a", "feature_b"),
        model_upload_compatible=not uses_custom_module,
        uses_custom_module=uses_custom_module,
    )


def _write_plugin(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    module_path = root / "synthetic_plugin.py"
    module_path.write_text(_CUSTOM_PLUGIN, encoding="utf-8")
    return module_path


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_model_artifact_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"

    artifact_path, manifest_path = save_model_artifact(
        run_dir=run_dir,
        model=_RoundTripRegressor(),
        manifest=_manifest(),
    )
    loaded = load_model_artifact(run_dir=run_dir)

    frame = pd.DataFrame({"feature_a": [0.1, 0.2], "feature_b": [0.3, 0.4]})
    assert artifact_path.is_file()
    assert manifest_path.is_file()
    assert loaded.manifest.run_id == "run-1"
    assert loaded.model.predict(frame).tolist() == pytest.approx([0.4, 0.6])


# --------------------------------------------------------------------------- #
# Custom plugin module rebinding
# --------------------------------------------------------------------------- #


def test_load_model_artifact_rebinds_custom_plugin_module_from_another_machine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A plugin-backed artifact must load where the training-time module name cannot exist.

    The synthetic module name hashes the training machine's absolute plugin path, so an
    artifact trained on the GPU box and pulled to another checkout records a name that
    process never registered.
    """

    trained_plugin = _write_plugin(tmp_path / "other-machine" / "custom_models")
    local_root = tmp_path / "this-machine" / "custom_models"
    _write_plugin(local_root)
    trained_name = model_factory.custom_module_name(trained_plugin)
    local_name = model_factory.custom_module_name(local_root / "synthetic_plugin.py")
    assert trained_name != local_name

    trained_module = model_factory.import_custom_model_module(trained_plugin)
    model = trained_module.MODEL_REGISTRY["SyntheticPluginRegressor"](scale=3.0)
    run_dir = tmp_path / "runs" / "run-custom"
    save_model_artifact(run_dir=run_dir, model=model, manifest=_manifest(uses_custom_module=True))

    # Stand in for a fresh process on the pulling machine: nothing imported the plugin,
    # so no `numereng_custom_*` name is registered and only the local root is discoverable.
    monkeypatch.delitem(sys.modules, trained_name)
    monkeypatch.setattr(model_factory, "_resolve_custom_models_root", lambda: local_root)

    loaded = load_model_artifact(run_dir=run_dir)

    assert type(loaded.model).__module__ == local_name
    assert loaded.model.predict(pd.DataFrame({"feature_a": [0.1, 0.2]})).tolist() == pytest.approx([0.3, 0.6])


def test_load_model_artifact_reports_an_unresolvable_custom_plugin_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trained_plugin = _write_plugin(tmp_path / "other-machine" / "custom_models")
    empty_root = tmp_path / "this-machine" / "custom_models"
    empty_root.mkdir(parents=True)
    trained_name = model_factory.custom_module_name(trained_plugin)

    trained_module = model_factory.import_custom_model_module(trained_plugin)
    model = trained_module.MODEL_REGISTRY["SyntheticPluginRegressor"]()
    run_dir = tmp_path / "runs" / "run-custom"
    save_model_artifact(run_dir=run_dir, model=model, manifest=_manifest(uses_custom_module=True))

    monkeypatch.delitem(sys.modules, trained_name)
    monkeypatch.setattr(model_factory, "_resolve_custom_models_root", lambda: empty_root)

    with pytest.raises(ModelArtifactError, match="serving_model_artifact_custom_module_unresolved"):
        load_model_artifact(run_dir=run_dir)


def test_load_model_artifact_reports_the_underlying_load_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    artifact_path, _ = save_model_artifact(run_dir=run_dir, model=_RoundTripRegressor(), manifest=_manifest())
    artifact_path.write_bytes(b"not-a-pickle")

    with pytest.raises(ModelArtifactError, match="serving_model_artifact_load_failed:"):
        load_model_artifact(run_dir=run_dir)
