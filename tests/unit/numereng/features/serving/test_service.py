from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

import numereng.features.serving.service as serving_service_module
from numereng.features.models.lgbm import LGBMRegressor
from numereng.features.serving import (
    ServingBlendRule,
    ServingComponentSpec,
    ServingUnsupportedConfigError,
    ServingValidationError,
    build_live_submission_package,
    build_submission_pickle,
    create_submission_package,
    inspect_package,
    list_submission_packages,
    submit_live_package,
    upload_submission_pickle,
)
from numereng.features.training.model_artifacts import ModelArtifactManifest, save_model_artifact


class _FakeServingClient:
    def __init__(self) -> None:
        self.uploaded_predictions: list[tuple[str, str]] = []
        self.uploaded_pickles: list[tuple[str, str, str | None, str | None]] = []

    def list_datasets(self, round_num: int | None = None) -> list[str]:
        _ = round_num
        return ["v5.2/live.parquet", "v5.2/live_benchmark_models.parquet"]

    def get_current_round(self) -> int | None:
        return 777

    def get_models(self) -> dict[str, str]:
        return {"main": "model-1"}

    def upload_predictions(self, *, file_path: str, model_id: str) -> str:
        self.uploaded_predictions.append((file_path, model_id))
        return "submission-1"

    def model_upload(
        self,
        *,
        file_path: str,
        model_id: str,
        data_version: str | None = None,
        docker_image: str | None = None,
    ) -> str:
        self.uploaded_pickles.append((file_path, model_id, data_version, docker_image))
        return "pickle-1"

    def model_upload_data_versions(self) -> list[str]:
        return ["v5.2"]

    def model_upload_docker_images(self) -> list[str]:
        return ["Python 3.11", "Python 3.12"]

    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        _ = round_num
        if dest_path is None:
            raise AssertionError("dest_path must be provided")
        path = Path(dest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "v5.2/features.json":
            path.write_text(
                json.dumps({"feature_sets": {"small": ["feature_a", "feature_b"]}}),
                encoding="utf-8",
            )
            return str(path)
        if filename == "v5.2/train.parquet":
            pd.DataFrame(
                {
                    "id": ["t1", "t2", "t3", "t4"],
                    "era": ["0001", "0001", "0002", "0002"],
                    "feature_a": [0.1, 0.2, 0.3, 0.4],
                    "feature_b": [0.5, 0.3, 0.2, 0.1],
                    "target": [0.2, 0.4, 0.6, 0.8],
                }
            ).to_parquet(path, index=False)
            return str(path)
        if filename == "v5.2/validation.parquet":
            pd.DataFrame(
                {
                    "id": ["v1", "v2"],
                    "era": ["0003", "0004"],
                    "feature_a": [0.35, 0.45],
                    "feature_b": [0.25, 0.15],
                    "target": [0.55, 0.75],
                    "data_type": ["validation", "validation"],
                }
            ).to_parquet(path, index=False)
            return str(path)
        if filename == "v5.2/live.parquet":
            pd.DataFrame(
                {
                    "id": ["live_1", "live_2"],
                    "era": ["0999", "0999"],
                    "feature_a": [0.1, 0.4],
                    "feature_b": [0.2, 0.1],
                }
            ).to_parquet(path, index=False)
            return str(path)
        if filename == "v5.2/live_benchmark_models.parquet":
            pd.DataFrame({"id": ["live_1", "live_2"], "era": ["0999", "0999"], "benchmark": [0.3, 0.7]}).to_parquet(
                path,
                index=False,
            )
            return str(path)
        raise AssertionError(f"unexpected filename: {filename}")


@pytest.fixture(autouse=True)
def _stub_pickle_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serving_service_module,
        "_verify_isolated_pickle_runtime",
        lambda **_: {
            "checked_at": "2026-04-11T00:00:00Z",
            "command": "uvx --with ...",
            "runtime": "Python 3.12",
        },
    )


def _write_custom_plugin(tmp_path: Path, *, name: str, expression: str) -> Path:
    plugin_path = tmp_path / f"{name}.py"
    plugin_path.write_text(
        f"""
class DummyRegressor:
    def __init__(self, feature_cols=None, **params):
        self.feature_cols = feature_cols or []
    def fit(self, X, y, **kwargs):
        return self
    def predict(self, X):
        return {expression}

MODEL_REGISTRY = {{"DummyRegressor": DummyRegressor}}
""".strip(),
        encoding="utf-8",
    )
    return plugin_path


def _write_config(
    tmp_path: Path,
    *,
    name: str,
    model_type: str,
    params: dict[str, object],
    module_path: Path | None = None,
) -> Path:
    model_block: dict[str, object] = {
        "type": model_type,
        "params": params,
    }
    if module_path is not None:
        model_block["module_path"] = str(module_path)
    config_path = tmp_path / f"{name}.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "feature_set": "small",
                    "target_col": "target",
                    "era_col": "era",
                    "id_col": "id",
                    "dataset_scope": "train_plus_validation",
                },
                "model": model_block,
                "training": {"engine": {"profile": "full_history_refit"}, "post_training_scoring": "none"},
                "preprocessing": {"nan_missing_all_twos": False, "missing_value": 2.0},
                "output": {},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_run_backed_component(tmp_path: Path, *, run_id: str) -> str:
    run_dir = tmp_path / ".numereng" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
            "dataset_scope": "train_plus_validation",
        },
        "model": {"type": "LGBMRegressor", "params": {}},
        "training": {"engine": {"profile": "full_history_refit"}, "post_training_scoring": "none"},
        "preprocessing": {"nan_missing_all_twos": False, "missing_value": 2.0},
        "output": {},
    }
    (run_dir / "resolved.json").write_text(json.dumps(resolved_config), encoding="utf-8")
    model = LGBMRegressor(
        feature_cols=["feature_a", "feature_b"],
        n_estimators=5,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=1,
        verbosity=-1,
    )
    train = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "feature_b": [0.5, 0.3, 0.2, 0.1],
            "target": [0.2, 0.4, 0.6, 0.8],
        }
    )
    model.fit(train[["feature_a", "feature_b"]], train["target"])
    save_model_artifact(
        run_dir=run_dir,
        model=model,
        manifest=ModelArtifactManifest(
            run_id=run_id,
            model_type="LGBMRegressor",
            data_version="v5.2",
            dataset_variant="non_downsampled",
            feature_set="small",
            target_col="target",
            era_col="era",
            id_col="id",
            feature_cols=("feature_a", "feature_b"),
            model_upload_compatible=True,
            uses_custom_module=False,
        ),
    )
    return run_id


def test_create_and_list_submission_package(tmp_path: Path) -> None:
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"] * 2.0 + X["feature_b"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )

    record = create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="dummy", weight=1.0, config_path=config_path),),
        blend_rule=ServingBlendRule(),
    )

    assert record.package_path.is_dir()
    listed = list_submission_packages(workspace_root=tmp_path, experiment_id="exp-1")
    assert [item.package_id for item in listed] == ["pkg-1"]


def test_create_submission_package_rejects_bad_weights(tmp_path: Path) -> None:
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )

    with pytest.raises(ServingValidationError, match="serving_component_weights_must_sum_to_one"):
        create_submission_package(
            workspace_root=tmp_path,
            experiment_id="exp-1",
            package_id="pkg-1",
            components=(ServingComponentSpec(component_id="dummy", weight=0.5, config_path=config_path),),
        )


def test_create_submission_package_prunes_zero_weight_components(tmp_path: Path) -> None:
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )

    record = create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(
            ServingComponentSpec(component_id="kept", weight=1.0, config_path=config_path),
            ServingComponentSpec(component_id="dropped", weight=0.0, config_path=config_path),
        ),
    )

    assert [item.component_id for item in record.components] == ["kept"]


def test_inspect_package_prunes_legacy_zero_weight_components(tmp_path: Path) -> None:
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )
    record = create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="kept", weight=1.0, config_path=config_path),),
    )
    payload = json.loads((record.package_path / "package.json").read_text(encoding="utf-8"))
    payload["components"].append(
        {
            "component_id": "legacy_zero",
            "weight": 0.0,
            "config_path": str(config_path),
            "run_id": None,
            "source_label": None,
        }
    )
    (record.package_path / "package.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert [item.component_id for item in result.package.components] == ["kept"]


def test_inspect_package_marks_custom_module_as_local_only(tmp_path: Path) -> None:
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="dummy", weight=1.0, config_path=config_path),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.local_live_compatible is True
    assert result.model_upload_compatible is False
    assert "serving_model_upload_custom_modules_not_supported" in result.model_upload_blockers
    assert result.report_path is not None and result.report_path.is_file()
    assert result.deployment_classification == "local_live_only"


def test_inspect_package_marks_config_backed_lgbm_as_local_live_only(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        name="lgbm_component",
        model_type="LGBMRegressor",
        params={"n_estimators": 5, "learning_rate": 0.1, "num_leaves": 8, "min_data_in_leaf": 1, "verbosity": -1},
    )
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, config_path=config_path),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.local_live_compatible is True
    assert result.model_upload_compatible is False
    assert result.artifact_ready is False
    assert result.deployment_classification == "local_live_only"
    assert "serving_model_upload_requires_persisted_model_artifact" in result.model_upload_blockers


def test_build_live_submission_package_writes_rank_blend_for_multi_component_package(tmp_path: Path) -> None:
    client = _FakeServingClient()
    plugin_1 = _write_custom_plugin(tmp_path, name="dummy_plugin_a", expression='X["feature_a"] * 2.0 + X["feature_b"]')
    plugin_2 = _write_custom_plugin(tmp_path, name="dummy_plugin_b", expression='X["feature_a"] + X["feature_b"] * 3.0')
    config_a = _write_config(tmp_path, name="component_a", model_type="DummyRegressor", params={}, module_path=plugin_1)
    config_b = _write_config(tmp_path, name="component_b", model_type="DummyRegressor", params={}, module_path=plugin_2)
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(
            ServingComponentSpec(component_id="dummy_a", weight=0.6, config_path=config_a),
            ServingComponentSpec(component_id="dummy_b", weight=0.4, config_path=config_b),
        ),
    )

    result = build_live_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        client=client,
    )

    submission = pd.read_parquet(result.submission_predictions_path)
    assert list(submission.columns) == ["id", "prediction"]
    assert submission["id"].tolist() == ["live_1", "live_2"]
    assert submission["prediction"].tolist() == pytest.approx([0.6, 0.9])
    assert result.current_round == 777
    assert result.package.artifacts["preflight_local_live_compatible"] == "true"


def test_build_submission_pickle_round_trips_artifact_backed_predictor(tmp_path: Path) -> None:
    client = _FakeServingClient()
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    build_result = build_live_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        client=client,
    )
    pickle_result = build_submission_pickle(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        client=client,
    )

    predictor = pickle.loads(pickle_result.pickle_path.read_bytes())
    live = pd.read_parquet(build_result.live_dataset_path)
    built = pd.read_parquet(build_result.submission_predictions_path)
    submission = predictor(live, None)
    assert submission["prediction"].tolist() == pytest.approx(built["prediction"].tolist())
    assert pickle_result.docker_image == "Python 3.12"
    assert pickle_result.smoke_verified is True
    assert pickle_result.package.artifacts["preflight_pickle_upload_ready"] == "true"


def test_inspect_package_marks_artifact_backed_lgbm_as_not_verified_before_pickle_build(tmp_path: Path) -> None:
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.model_upload_compatible is True
    assert result.pickle_upload_ready is False
    assert result.deployment_classification == "artifact_backed_live_ready"


def test_build_submission_pickle_rejects_local_only_package(tmp_path: Path) -> None:
    client = _FakeServingClient()
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="dummy", weight=1.0, config_path=config_path),),
    )

    with pytest.raises(ServingUnsupportedConfigError, match="serving_model_upload_preflight_failed"):
        build_submission_pickle(
            workspace_root=tmp_path,
            experiment_id="exp-1",
            package_id="pkg-1",
            client=client,
        )


def test_submit_live_package_uses_submission_boundary(tmp_path: Path) -> None:
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"] * 2.0 + X["feature_b"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )
    client = _FakeServingClient()
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="dummy", weight=1.0, config_path=config_path),),
    )

    result = submit_live_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        model_name="main",
        client=client,
    )

    assert result.submission_id == "submission-1"
    assert result.model_id == "model-1"
    assert client.uploaded_predictions


def test_upload_submission_pickle_validates_model_upload_options(tmp_path: Path) -> None:
    client = _FakeServingClient()
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    with pytest.raises(ServingValidationError, match="serving_model_upload_data_version_unsupported"):
        upload_submission_pickle(
            workspace_root=tmp_path,
            experiment_id="exp-1",
            package_id="pkg-1",
            model_name="main",
            data_version="v9.9",
            client=client,
        )


def test_build_submission_pickle_rejects_neutralized_package(tmp_path: Path) -> None:
    client = _FakeServingClient()
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )
    package_path = tmp_path / ".numereng" / "experiments" / "exp-1" / "submission_packages" / "pkg-1" / "package.json"
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    payload["neutralization"] = {
        "enabled": True,
        "proportion": 0.5,
        "mode": "era",
        "neutralizer_cols": [],
        "rank_output": True,
    }
    package_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        ServingUnsupportedConfigError,
        match="serving_model_upload_preflight_failed|serving_model_upload_neutralization_not_supported",
    ):
        build_submission_pickle(
            workspace_root=tmp_path,
            experiment_id="exp-1",
            package_id="pkg-1",
            client=client,
        )


def test_upload_submission_pickle_rejects_runtime_mismatch(tmp_path: Path) -> None:
    client = _FakeServingClient()
    config_path = _write_config(
        tmp_path,
        name="lgbm_component",
        model_type="LGBMRegressor",
        params={"n_estimators": 5, "learning_rate": 0.1, "num_leaves": 8, "min_data_in_leaf": 1, "verbosity": -1},
    )
    package = create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, config_path=config_path),),
    )
    package = type(package)(
        **{
            **package.__dict__,
            "artifacts": {
                "pickle_smoke_verified": "true",
                "pickle_runtime_docker_image": "Python 3.12",
                "pickle_path": str(tmp_path / "model.pkl"),
            },
        }
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        serving_service_module,
        "build_submission_pickle",
        lambda **_: serving_service_module.PickleBuildResult(
            package=package,
            pickle_path=tmp_path / "model.pkl",
            docker_image="Python 3.12",
            smoke_verified=True,
        ),
    )

    try:
        with pytest.raises(ServingValidationError, match="serving_model_upload_runtime_mismatch"):
            upload_submission_pickle(
                workspace_root=tmp_path,
                experiment_id="exp-1",
                package_id="pkg-1",
                model_name="main",
                docker_image="Python 3.11",
                client=client,
            )
    finally:
        monkeypatch.undo()
