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
    ServingRuntimeError,
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
        return ["v5.3/live.parquet", "v5.3/live_benchmark_models.parquet"]

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
        return ["v5.3"]

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
        if filename == "v5.3/features.json":
            path.write_text(
                json.dumps({"feature_sets": {"small": ["feature_a", "feature_b"]}}),
                encoding="utf-8",
            )
            return str(path)
        if filename == "v5.3/train.parquet":
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
        if filename == "v5.3/validation.parquet":
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
        if filename == "v5.3/live.parquet":
            pd.DataFrame(
                {
                    "id": ["live_1", "live_2"],
                    "era": ["0999", "0999"],
                    "feature_a": [0.1, 0.4],
                    "feature_b": [0.2, 0.1],
                }
            ).to_parquet(path, index=False)
            return str(path)
        if filename == "v5.3/live_benchmark_models.parquet":
            pd.DataFrame(
                {"id": ["live_1", "live_2"], "era": ["0999", "0999"], "v53_lgbm_ender20": [0.3, 0.7]}
            ).to_parquet(
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
                    "data_version": "v5.3",
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


def _write_run_backed_component(
    tmp_path: Path,
    *,
    run_id: str,
    data_version: str = "v5.3",
    baseline_predictions_path: str | None = None,
    model_upload_compatible: bool = True,
    uses_custom_module: bool = False,
) -> str:
    run_dir = tmp_path / ".numereng" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "data": {
            "data_version": data_version,
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
    feature_cols = ["feature_a", "feature_b"] if baseline_predictions_path is None else ["feature_a", "bench_raw"]
    model = LGBMRegressor(
        feature_cols=feature_cols,
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
            "bench_raw": [0.15, 0.35, 0.55, 0.75],
            "target": [0.2, 0.4, 0.6, 0.8],
        }
    )
    model.fit(train[feature_cols], train["target"])
    save_model_artifact(
        run_dir=run_dir,
        model=model,
        manifest=ModelArtifactManifest(
            run_id=run_id,
            model_type="LGBMRegressor",
            data_version=data_version,
            dataset_variant="non_downsampled",
            feature_set="small",
            target_col="target",
            era_col="era",
            id_col="id",
            feature_cols=tuple(feature_cols),
            baseline_col="bench_raw" if baseline_predictions_path is not None else None,
            baseline_name="bench_raw" if baseline_predictions_path is not None else None,
            baseline_predictions_path=baseline_predictions_path,
            model_upload_compatible=model_upload_compatible,
            uses_custom_module=uses_custom_module,
        ),
    )
    return run_id


def _write_official_benchmark(
    tmp_path: Path,
    *,
    name: str = "official_v53_lgbm_ender20",
    kind: str = "official_numerai_benchmark",
) -> str:
    baseline_dir = tmp_path / ".numereng" / "datasets" / "baselines" / "active_benchmark"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "benchmark.json").write_text(json.dumps({"name": name, "kind": kind}), encoding="utf-8")
    return str(baseline_dir / "predictions.parquet")


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
    assert "serving_model_upload_requires_persisted_model_artifact" in result.model_upload_blockers
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


def test_build_live_submission_package_passes_era_to_era_aware_components(tmp_path: Path) -> None:
    client = _FakeServingClient()
    plugin_path = tmp_path / "era_aware_plugin.py"
    plugin_path.write_text(
        """
class DummyRegressor:
    accepts_era = True
    def __init__(self, feature_cols=None, **params):
        self.feature_cols = feature_cols or []
    def fit(self, X, y, era=None, **kwargs):
        if era is None:
            raise AssertionError("fit_missing_era")
        return self
    def predict(self, X, era=None, **kwargs):
        if era is None:
            raise AssertionError("predict_missing_era")
        return X["feature_a"]

MODEL_REGISTRY = {"DummyRegressor": DummyRegressor}
""".strip(),
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        name="era_component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin_path,
    )
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="era_aware", weight=1.0, config_path=config_path),),
    )

    result = build_live_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        client=client,
    )

    submission = pd.read_parquet(result.submission_predictions_path)
    assert submission["id"].tolist() == ["live_1", "live_2"]


def test_build_live_submission_package_attaches_live_benchmark_to_baseline_components(tmp_path: Path) -> None:
    client = _FakeServingClient()
    run_id = _write_run_backed_component(
        tmp_path,
        run_id="run-baseline",
        baseline_predictions_path=_write_official_benchmark(tmp_path),
    )
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="baseline", weight=1.0, run_id=run_id),),
    )

    result = build_live_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        client=client,
    )

    submission = pd.read_parquet(result.submission_predictions_path)
    assert submission["id"].tolist() == ["live_1", "live_2"]
    assert submission["prediction"].notna().all()


def test_build_submission_pickle_serves_baseline_components_from_live_benchmark_models(tmp_path: Path) -> None:
    client = _FakeServingClient()
    run_id = _write_run_backed_component(
        tmp_path,
        run_id="run-baseline",
        baseline_predictions_path=_write_official_benchmark(tmp_path),
    )
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="baseline", weight=1.0, run_id=run_id),),
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

    artifacts = pickle_result.package.artifacts
    assert json.loads(artifacts["pickle_benchmark_model_cols"]) == {"baseline": "v53_lgbm_ender20"}
    assert artifacts["pickle_uses_baseline_inputs"] == "true"
    assert "hosted_blend_rank_reimplemented" in json.loads(artifacts["pickle_drift_risks"])

    predictor = pickle.loads(pickle_result.pickle_path.read_bytes())
    live = pd.read_parquet(build_result.live_dataset_path)
    benchmark = pd.DataFrame({"id": ["live_1", "live_2"], "v53_lgbm_ender20": [0.3, 0.7]})
    built = pd.read_parquet(build_result.submission_predictions_path)

    submission = predictor(live, benchmark)

    assert submission["prediction"].tolist() == pytest.approx(built["prediction"].tolist())


def test_component_subprocess_reports_missing_output_with_status_and_log_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )
    package = create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="dummy", weight=1.0, config_path=config_path),),
    )

    def fake_run(args: list[str], **kwargs: object) -> object:
        payload_path = Path(args[args.index("--payload") + 1])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        Path(str(payload["status_path"])).write_text(
            json.dumps(
                {
                    "component_id": payload["component"]["component_id"],
                    "phase": "fit_component",
                    "state": "running",
                    "updated_at": "2026-05-15T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        stdout = kwargs.get("stdout")
        if stdout is not None:
            stdout.write("worker reached fit_component\n")
            stdout.flush()
        return serving_service_module.subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(serving_service_module.subprocess, "run", fake_run)

    with pytest.raises(ServingRuntimeError) as exc_info:
        serving_service_module._fit_and_predict_package_subprocess(
            workspace_root=tmp_path,
            package=package,
            components=package.components,
            live_features=pd.DataFrame(
                {
                    "id": ["live_1"],
                    "era": ["0999"],
                    "feature_a": [0.1],
                    "feature_b": [0.2],
                }
            ),
        )

    message = str(exc_info.value)
    assert "serving_component_worker_missing_output:dummy" in message
    assert "last_status=running,fit_component,2026-05-15T00:00:00Z" in message
    assert "worker reached fit_component" in message


def test_component_subprocess_forwards_the_requested_baseline_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _write_custom_plugin(tmp_path, name="dummy_plugin", expression='X["feature_a"]')
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=plugin,
    )
    package = create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="dummy", weight=1.0, config_path=config_path),),
    )
    observed: list[str | None] = []

    def fake_run(args: list[str], **kwargs: object) -> object:
        _ = kwargs
        payload_path = Path(args[args.index("--payload") + 1])
        observed.append(json.loads(payload_path.read_text(encoding="utf-8")).get("baseline_source"))
        return serving_service_module.subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(serving_service_module.subprocess, "run", fake_run)

    with pytest.raises(ServingRuntimeError):
        serving_service_module._fit_and_predict_package_subprocess(
            workspace_root=tmp_path,
            package=package,
            components=package.components,
            live_features=pd.DataFrame({"id": ["v1"], "era": ["0003"], "feature_a": [0.1], "feature_b": [0.2]}),
            baseline_source="historical",
        )

    assert observed == ["historical"]


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


def test_inspect_package_uses_artifact_before_stale_resolved_config(tmp_path: Path) -> None:
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    run_dir = tmp_path / ".numereng" / "runs" / run_id
    resolved = json.loads((run_dir / "resolved.json").read_text(encoding="utf-8"))
    resolved["data"]["loading"] = {"legacy": True}
    resolved["data"]["benchmark_model"] = "legacy_benchmark"
    (run_dir / "resolved.json").write_text(json.dumps(resolved), encoding="utf-8")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.local_live_compatible is True
    assert result.model_upload_compatible is True
    assert result.artifact_ready is True
    assert not result.local_live_blockers
    assert not result.artifact_blockers


def test_inspect_package_missing_artifact_falls_back_to_config_validation(tmp_path: Path) -> None:
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    model_path = tmp_path / ".numereng" / "runs" / run_id / "artifacts" / "model" / "model.pkl"
    model_path.unlink()
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.local_live_compatible is True
    assert result.model_upload_compatible is False
    assert result.artifact_ready is False
    assert "serving_model_artifact_missing" in result.artifact_blockers
    assert "serving_model_upload_requires_persisted_model_artifact" in result.model_upload_blockers


def test_inspect_package_missing_artifact_reports_stale_config_blocker(tmp_path: Path) -> None:
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    run_dir = tmp_path / ".numereng" / "runs" / run_id
    (run_dir / "artifacts" / "model" / "model.pkl").unlink()
    resolved = json.loads((run_dir / "resolved.json").read_text(encoding="utf-8"))
    resolved["data"]["loading"] = {"legacy": True}
    (run_dir / "resolved.json").write_text(json.dumps(resolved), encoding="utf-8")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.local_live_compatible is False
    assert result.model_upload_compatible is False
    assert result.artifact_ready is False
    assert "serving_model_artifact_missing" in result.artifact_blockers
    assert any("training_config_schema_invalid:data.loading" in item for item in result.local_live_blockers)


def test_inspect_package_reports_an_unreadable_artifact_as_a_blocker(tmp_path: Path) -> None:
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    model_path = tmp_path / ".numereng" / "runs" / run_id / "artifacts" / "model" / "model.pkl"
    model_path.write_bytes(b"not-a-pickle")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.artifact_ready is False
    assert any(item.startswith("serving_model_artifact_load_failed:") for item in result.artifact_blockers)
    assert "serving_model_upload_requires_persisted_model_artifact" in result.model_upload_blockers


def test_build_live_submission_package_fails_loudly_on_an_unreadable_artifact(tmp_path: Path) -> None:
    """A broken artifact must not silently degrade to a config refit.

    Refitting would score a different model than the one the package pins, which is the
    exact drift a pinned submission package exists to prevent.
    """

    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    model_path = tmp_path / ".numereng" / "runs" / run_id / "artifacts" / "model" / "model.pkl"
    model_path.write_bytes(b"not-a-pickle")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    with pytest.raises(ServingRuntimeError, match="serving_component_artifact_load_failed:lgbm"):
        build_live_submission_package(
            workspace_root=tmp_path,
            experiment_id="exp-1",
            package_id="pkg-1",
            client=_FakeServingClient(),
        )


def test_build_live_submission_package_refits_from_config_when_no_artifact_was_persisted(tmp_path: Path) -> None:
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    (tmp_path / ".numereng" / "runs" / run_id / "artifacts" / "model" / "model.pkl").unlink()
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    result = build_live_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        client=_FakeServingClient(),
    )

    assert result.submission_predictions_path.is_file()


@pytest.mark.parametrize(
    ("benchmark_kwargs", "expected_blocker"),
    [
        (None, "serving_model_upload_benchmark_provenance_missing"),
        (
            {"name": "blend_v1", "kind": "local_blend"},
            "serving_model_upload_benchmark_not_official",
        ),
        (
            {"name": "official_v50_lgbm_ct_blend"},
            "serving_model_upload_benchmark_data_version_mismatch",
        ),
    ],
)
def test_inspect_package_blocks_uploads_with_unresolvable_benchmarks(
    tmp_path: Path,
    benchmark_kwargs: dict[str, str] | None,
    expected_blocker: str,
) -> None:
    baseline_path = (
        str(tmp_path / ".numereng" / "datasets" / "baselines" / "active_benchmark" / "predictions.parquet")
        if benchmark_kwargs is None
        else _write_official_benchmark(tmp_path, **benchmark_kwargs)
    )
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm", baseline_predictions_path=baseline_path)
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.local_live_compatible is True
    assert result.model_upload_compatible is False
    assert result.artifact_ready is True
    assert any(item.startswith(expected_blocker) for item in result.model_upload_blockers)


def test_inspect_package_accepts_custom_module_artifact_with_resolvable_benchmark(tmp_path: Path) -> None:
    run_id = _write_run_backed_component(
        tmp_path,
        run_id="run-nn",
        baseline_predictions_path=_write_official_benchmark(tmp_path),
        model_upload_compatible=False,
        uses_custom_module=True,
    )
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="nn", weight=1.0, run_id=run_id),),
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert result.model_upload_compatible is True
    assert result.model_upload_blockers == ()
    assert "serving_model_upload_artifact_flag_superseded" in result.warnings
    assert result.deployment_classification == "artifact_backed_live_ready"


def test_inspect_package_warns_when_artifact_data_version_mismatches_package(tmp_path: Path) -> None:
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm", data_version="v5.1")
    create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
        data_version="v5.3",
    )

    result = inspect_package(workspace_root=tmp_path, experiment_id="exp-1", package_id="pkg-1")

    assert "serving_package_component_data_version_mismatch" in result.warnings


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


def test_upload_submission_pickle_records_submission_upload_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeServingClient()
    config_path = _write_config(
        tmp_path,
        name="component",
        model_type="DummyRegressor",
        params={},
        module_path=_write_custom_plugin(tmp_path, name="component_plugin", expression="[0.5 for _ in range(len(X))]"),
    )
    package = create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        components=(ServingComponentSpec(component_id="component", weight=1.0, config_path=config_path),),
    )
    pickle_path = tmp_path / "model.pkl"
    pickle_path.write_bytes(b"pickle")
    package = type(package)(
        **{
            **package.__dict__,
            "artifacts": {
                "pickle_smoke_verified": "true",
                "pickle_runtime_docker_image": "Python 3.12",
                "pickle_path": str(pickle_path),
            },
        }
    )
    monkeypatch.setattr(
        serving_service_module,
        "build_submission_pickle",
        lambda **_: serving_service_module.PickleBuildResult(
            package=package,
            pickle_path=pickle_path,
            docker_image="Python 3.12",
            smoke_verified=True,
        ),
    )

    result = upload_submission_pickle(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        model_name="main",
        client=client,
    )

    assert result.upload_id == "pickle-1"
    metadata_path = tmp_path / ".numereng" / "submissions" / "main" / "submission.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_id"] == "model-1"
    assert metadata["hosted_pickle"]["upload_id"] == "pickle-1"
    assert metadata["hosted_pickle"]["docker_image"] == "Python 3.12"
    assert metadata["source"]["experiment_id"] == "exp-1"
    assert metadata["source"]["package_id"] == "pkg-1"
    assert metadata["uploads"][0]["source"]["package_id"] == "pkg-1"


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


def test_pickle_smoke_command_stays_short_regardless_of_payload_size(tmp_path: Path) -> None:
    """Windows `CreateProcess` caps the command line, so nothing unbounded may ride in argv.

    A real v5.3 package derives thousands of feature columns and an open-ended pin list;
    both must reach the probe as files, leaving an argv whose length is independent of them.
    """

    probe_dir = tmp_path / "smoke"
    feature_cols = [f"feature_{index:04d}_with_a_realistically_long_name" for index in range(3000)]
    requirements = [f"package_number_{index:04d}==1.2.{index}" for index in range(500)]

    command = serving_service_module._pickle_smoke_command(
        uvx="/usr/local/bin/uvx",
        python_version="3.12",
        requirements=tuple(requirements),
        probe_dir=probe_dir,
        pickle_path=tmp_path / "model.pkl",
        feature_cols=feature_cols,
        id_cols=["id"],
        era_cols=["era"],
        benchmark_cols=["v53_lgbm_ender20"],
    )

    assert len(" ".join(command)) < 8000
    assert not any(feature_cols[0] in part for part in command)
    assert not any(requirements[0] in part for part in command)
    assert "--with-requirements" in command
    assert command[-2:] == [str((probe_dir / "probe.py").resolve()), str((probe_dir / "payload.json").resolve())]
    assert "predictor(frame, benchmark)" in (probe_dir / "probe.py").read_text(encoding="utf-8")
    assert json.loads((probe_dir / "payload.json").read_text(encoding="utf-8"))["feature_cols"] == feature_cols
    assert (probe_dir / "requirements.txt").read_text(encoding="utf-8").splitlines() == requirements
