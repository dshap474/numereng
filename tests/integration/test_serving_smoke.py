from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
from numerai_tools.submissions import validate_submission_numerai

import numereng.api as api_module
import numereng.features.serving.service as serving_service_module
import numereng.features.training.service as training_service_module
from numereng.api.contracts import TrainRunRequest


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
        return ["ghcr.io/numerai/numerai-inference:py3.11"]

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
    model_block: dict[str, object] = {"type": model_type, "params": params}
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


@pytest.mark.integration
def test_serve_live_build_multi_component_submission_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeServingClient()
    monkeypatch.setattr(serving_service_module, "create_serving_client", lambda: client)
    plugin_a = _write_custom_plugin(tmp_path, name="plugin_a", expression='X["feature_a"] * 2.0 + X["feature_b"]')
    plugin_b = _write_custom_plugin(tmp_path, name="plugin_b", expression='X["feature_a"] + X["feature_b"] * 3.0')
    config_a = _write_config(tmp_path, name="component_a", model_type="DummyRegressor", params={}, module_path=plugin_a)
    config_b = _write_config(tmp_path, name="component_b", model_type="DummyRegressor", params={}, module_path=plugin_b)

    api_module.serve_package_create(
        api_module.ServePackageCreateRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            components=[
                api_module.ServeComponentRequest(component_id="a", weight=0.6, config_path=str(config_a)),
                api_module.ServeComponentRequest(component_id="b", weight=0.4, config_path=str(config_b)),
            ],
            workspace_root=str(tmp_path),
        )
    )
    inspect_result = api_module.serve_package_inspect(
        api_module.ServePackageInspectRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            workspace_root=str(tmp_path),
        )
    )
    assert inspect_result.local_live_compatible is True
    assert inspect_result.model_upload_compatible is False

    live_result = api_module.serve_live_build(
        api_module.ServeLiveBuildRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            workspace_root=str(tmp_path),
        )
    )

    live_ids = pd.read_parquet(live_result.live_dataset_path)["id"]
    submission = pd.read_parquet(live_result.submission_predictions_path)
    _, _, validated, _ = validate_submission_numerai(universe=live_ids, submission=submission)
    assert list(validated.columns) == ["id", "prediction"]
    assert validated["prediction"].tolist() == pytest.approx([0.6, 0.9])


@pytest.mark.integration
def test_serve_pickle_build_roundtrip_from_run_backed_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeServingClient()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(serving_service_module, "create_serving_client", lambda: client)
    monkeypatch.setattr(training_service_module, "create_training_data_client", lambda: client)
    config_path = _write_config(
        tmp_path,
        name="lgbm_component",
        model_type="LGBMRegressor",
        params={"n_estimators": 5, "learning_rate": 0.1, "num_leaves": 8, "min_data_in_leaf": 1, "verbosity": -1},
    )

    train_result = api_module.run_training(
        TrainRunRequest(
            config_path=str(config_path),
            output_dir=str(tmp_path / ".numereng"),
            profile="full_history_refit",
            post_training_scoring="none",
        )
    )
    run_dir = tmp_path / ".numereng" / "runs" / train_result.run_id
    assert (run_dir / "artifacts" / "model" / "model.pkl").is_file()
    assert (run_dir / "artifacts" / "model" / "manifest.json").is_file()

    api_module.serve_package_create(
        api_module.ServePackageCreateRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            components=[
                api_module.ServeComponentRequest(
                    component_id="lgbm",
                    weight=1.0,
                    run_id=train_result.run_id,
                )
            ],
            workspace_root=str(tmp_path),
        )
    )
    inspect_result = api_module.serve_package_inspect(
        api_module.ServePackageInspectRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            workspace_root=str(tmp_path),
        )
    )
    assert inspect_result.artifact_ready is True
    assert inspect_result.pickle_upload_ready is True
    monkeypatch.setattr(
        serving_service_module,
        "fit_component",
        lambda **_: (_ for _ in ()).throw(AssertionError("artifact-backed package should not retrain")),
    )
    live_result = api_module.serve_live_build(
        api_module.ServeLiveBuildRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            workspace_root=str(tmp_path),
        )
    )
    pickle_result = api_module.serve_pickle_build(
        api_module.ServePickleBuildRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            workspace_root=str(tmp_path),
        )
    )

    predictor = pickle.loads(Path(pickle_result.pickle_path).read_bytes())
    live = pd.read_parquet(live_result.live_dataset_path)
    built = pd.read_parquet(live_result.submission_predictions_path)
    predicted = predictor(live, None)
    assert predicted["prediction"].tolist() == pytest.approx(built["prediction"].tolist())
