from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.models.lgbm import LGBMRegressor
from numereng.features.serving import (
    ServingBlendRule,
    ServingComponentSpec,
    ServingNeutralizationSpec,
    create_submission_package,
    score_submission_package,
    sync_submission_package_diagnostics,
)
from numereng.features.serving.repo import load_package, save_package
from numereng.features.training.model_artifacts import ModelArtifactManifest, save_model_artifact


class _FakePackageEvalClient:
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
                json.dumps(
                    {
                        "feature_sets": {
                            "small": ["feature_a", "feature_b"],
                            "fncv3_features": ["feature_a", "feature_b"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            return str(path)
        if filename == "v5.2/train.parquet":
            _train_frame().to_parquet(path, index=False)
            return str(path)
        if filename == "v5.2/validation.parquet":
            _validation_frame().to_parquet(path, index=False)
            return str(path)
        if filename == "v5.2/train_benchmark_models.parquet":
            _train_benchmark_frame().to_parquet(path, index=False)
            return str(path)
        if filename == "v5.2/validation_benchmark_models.parquet":
            _validation_benchmark_frame().to_parquet(path, index=False)
            return str(path)
        if filename == "v5.2/meta_model.parquet":
            _meta_model_frame().to_parquet(path, index=False)
            return str(path)
        if filename == "v5.2/validation_example_preds.parquet":
            _validation_example_preds_frame().to_parquet(path, index=False)
            return str(path)
        raise AssertionError(f"unexpected filename: {filename}")

    def diagnostics(self, *, model_id: str, diagnostics_id: str | None = None) -> dict[str, Any]:
        _ = diagnostics_id
        return {
            "id": "diag-1",
            "status": "done",
            "updatedAt": "2026-04-16T12:00:00Z",
            "validationCorrMean": 0.06,
            "validationCorrSharpe": 3.8,
            "validationCorrStd": 0.015,
            "validationMaxDrawdown": -0.003,
            "validationFeatureCorrMax": 0.266,
            "validationFeatureNeutralCorrMean": 0.061,
            "validationBmcMean": 0.043,
            "examplePredsCorrMean": 0.38,
            "validationAutocorr": 0.15,
            "message": f"model:{model_id}",
            "perEraDiagnostics": [
                {"era": 1, "validationCorr": 0.05, "validationBmc": 0.04},
                {"era": 2, "validationCorr": 0.07, "validationBmc": 0.05},
            ],
        }

    def compute_pickle_status(
        self,
        *,
        pickle_id: str,
        model_id: str | None = None,
        unassigned: bool = False,
    ) -> dict[str, Any] | None:
        _ = (model_id, unassigned)
        return {
            "id": pickle_id,
            "diagnosticsStatus": "succeeded",
            "validationStatus": "succeeded",
            "triggerStatus": "succeeded",
        }

    def compute_pickle_diagnostics_logs(self, *, pickle_id: str) -> list[dict[str, Any]]:
        return [{"timestamp": "2026-04-16T12:00:00Z", "message": f"log:{pickle_id}"}]


class _PendingDiagnosticsClient(_FakePackageEvalClient):
    def compute_pickle_status(
        self,
        *,
        pickle_id: str,
        model_id: str | None = None,
        unassigned: bool = False,
    ) -> dict[str, Any] | None:
        _ = (model_id, unassigned)
        return {
            "id": pickle_id,
            "diagnosticsStatus": "pending",
            "validationStatus": "succeeded",
            "triggerStatus": "running",
        }


class _MissingStatusDiagnosticsClient(_FakePackageEvalClient):
    def compute_pickle_status(
        self,
        *,
        pickle_id: str,
        model_id: str | None = None,
        unassigned: bool = False,
    ) -> dict[str, Any] | None:
        _ = (pickle_id, model_id, unassigned)
        return None


class _EventuallyReadyDiagnosticsClient(_FakePackageEvalClient):
    def __init__(self) -> None:
        self._status_calls = 0

    def compute_pickle_status(
        self,
        *,
        pickle_id: str,
        model_id: str | None = None,
        unassigned: bool = False,
    ) -> dict[str, Any] | None:
        _ = (pickle_id, model_id, unassigned)
        self._status_calls += 1
        if self._status_calls == 1:
            return None
        return super().compute_pickle_status(pickle_id=pickle_id, model_id=model_id, unassigned=unassigned)


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["t1", "t2", "t3", "t4"],
            "era": ["0001", "0001", "0002", "0002"],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "feature_b": [0.4, 0.3, 0.2, 0.1],
            "target": [0.2, 0.4, 0.6, 0.8],
            "target_ender_20": [0.2, 0.4, 0.6, 0.8],
            "target_cyrusd_20": [0.3, 0.5, 0.4, 0.7],
        }
    )


def _validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["v1", "v2", "v3", "v4"],
            "era": ["0003", "0003", "0004", "0004"],
            "feature_a": [0.15, 0.25, 0.35, 0.45],
            "feature_b": [0.45, 0.35, 0.25, 0.15],
            "target": [0.3, 0.5, 0.4, 0.7],
            "target_ender_20": [0.3, 0.5, 0.4, 0.7],
            "target_cyrusd_20": [0.4, 0.3, 0.6, 0.5],
            "data_type": ["validation", "validation", "validation", "validation"],
        }
    )


def _train_benchmark_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["t1", "t2", "t3", "t4"],
            "era": ["0001", "0001", "0002", "0002"],
            "v52_lgbm_ender20": [0.2, 0.45, 0.55, 0.75],
        }
    )


def _validation_benchmark_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["v1", "v2", "v3", "v4"],
            "era": ["0003", "0003", "0004", "0004"],
            "v52_lgbm_ender20": [0.25, 0.55, 0.35, 0.65],
        }
    )


def _meta_model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["t1", "t2", "t3", "t4", "v1", "v2", "v3", "v4"],
            "era": ["0001", "0001", "0002", "0002", "0003", "0003", "0004", "0004"],
            "numerai_meta_model": [0.18, 0.41, 0.58, 0.79, 0.28, 0.51, 0.38, 0.68],
        }
    )


def _validation_example_preds_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["v1", "v2", "v3", "v4"],
            "era": ["0003", "0003", "0004", "0004"],
            "prediction": [0.22, 0.52, 0.42, 0.62],
        }
    )


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
    train = _train_frame()
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


def _build_package(tmp_path: Path, *, package_id: str = "pkg-1"):
    run_id = _write_run_backed_component(tmp_path, run_id="run-lgbm")
    return create_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id=package_id,
        components=(ServingComponentSpec(component_id="lgbm", weight=1.0, run_id=run_id),),
        blend_rule=ServingBlendRule(),
    )


def test_score_submission_package_local_writes_explicit_metrics(tmp_path: Path) -> None:
    _build_package(tmp_path)

    result = score_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        runtime="local",
        stage="post_training_core",
        client=_FakePackageEvalClient(),
    )

    assert result.runtime_used == "local"
    assert result.package.status == "created"
    summaries = json.loads(result.summaries_path.read_text(encoding="utf-8"))
    assert "corr_target" in summaries
    assert "corr_target_ender_20" in summaries
    assert "corr_target_cyrusd_20" in summaries
    assert "bmc_target_ender_20" in summaries
    assert "corr_with_example_preds" in summaries
    metric_series = pd.read_parquet(result.metric_series_path)
    assert {"per_era", "cumulative"} == set(metric_series["series_type"])


def test_score_submission_package_auto_falls_back_to_local_for_neutralized_package(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    updated = replace(
        package,
        neutralization=ServingNeutralizationSpec(enabled=True),
    )
    save_package(updated)

    result = score_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        runtime="auto",
        stage="post_training_core",
        client=_FakePackageEvalClient(),
    )

    assert result.runtime_requested == "auto"
    assert result.runtime_used == "local"
    assert result.package.status == "created"


def test_score_submission_package_pickle_preserves_package_status(tmp_path: Path) -> None:
    _build_package(tmp_path)

    result = score_submission_package(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id="pkg-1",
        runtime="pickle",
        stage="post_training_core",
        client=_FakePackageEvalClient(),
    )

    assert result.runtime_used == "pickle"
    assert result.package.status == "created"
    assert result.predictions_path.is_file()


def test_sync_submission_package_diagnostics_success(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    package = save_package(
        replace(
            package,
            status="pickle_uploaded",
            artifacts={
                **package.artifacts,
                "last_pickle_model_id": "model-1",
                "last_pickle_upload_id": "upload-1",
            },
        )
    )

    result = sync_submission_package_diagnostics(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id=package.package_id,
        wait=True,
        client=_FakePackageEvalClient(),
    )

    assert result.package.status == "pickle_uploaded"
    assert result.terminal is True
    assert result.raw_path is not None and result.raw_path.is_file()
    assert result.summary_path is not None and result.summary_path.is_file()
    assert result.per_era_path is not None and result.per_era_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["payload_scope"] == "latest_model"
    assert summary["synced_upload_id"] == "upload-1"


def test_sync_submission_package_diagnostics_pending_without_wait(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    package = save_package(
        replace(
            package,
            status="pickle_uploaded",
            artifacts={
                **package.artifacts,
                "last_pickle_model_id": "model-1",
                "last_pickle_upload_id": "upload-1",
            },
        )
    )

    result = sync_submission_package_diagnostics(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id=package.package_id,
        wait=False,
        client=_PendingDiagnosticsClient(),
    )

    assert result.package.status == "pickle_uploaded"
    assert result.terminal is False
    assert result.timed_out is False
    assert result.raw_path is None
    assert result.compute_status_path.is_file()
    latest = load_package(workspace_root=tmp_path, experiment_id="exp-1", package_id=package.package_id)
    assert latest.artifacts["last_diagnostics_upload_id"] == "upload-1"
    assert "last_diagnostics_raw_path" not in latest.artifacts


def test_sync_submission_package_diagnostics_clears_stale_payload_paths(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    package = save_package(
        replace(
            package,
            status="pickle_uploaded",
            artifacts={
                **package.artifacts,
                "last_pickle_model_id": "model-1",
                "last_pickle_upload_id": "upload-1",
            },
        )
    )

    first = sync_submission_package_diagnostics(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id=package.package_id,
        wait=True,
        client=_FakePackageEvalClient(),
    )
    assert first.raw_path is not None

    package = load_package(workspace_root=tmp_path, experiment_id="exp-1", package_id=package.package_id)
    package = save_package(replace(package, artifacts={**package.artifacts, "last_pickle_upload_id": "upload-2"}))

    second = sync_submission_package_diagnostics(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id=package.package_id,
        wait=False,
        client=_PendingDiagnosticsClient(),
    )

    assert second.raw_path is None
    latest = load_package(workspace_root=tmp_path, experiment_id="exp-1", package_id=package.package_id)
    assert latest.artifacts["last_diagnostics_upload_id"] == "upload-2"
    assert "last_diagnostics_raw_path" not in latest.artifacts
    assert "last_diagnostics_summary_path" not in latest.artifacts
    assert "last_diagnostics_per_era_path" not in latest.artifacts


def test_sync_submission_package_diagnostics_missing_status_is_pending_without_wait(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    package = save_package(
        replace(
            package,
            status="pickle_uploaded",
            artifacts={
                **package.artifacts,
                "last_pickle_model_id": "model-1",
                "last_pickle_upload_id": "upload-1",
            },
        )
    )

    result = sync_submission_package_diagnostics(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id=package.package_id,
        wait=False,
        client=_MissingStatusDiagnosticsClient(),
    )

    assert result.terminal is False
    assert result.timed_out is False
    status_payload = json.loads(result.compute_status_path.read_text(encoding="utf-8"))
    assert status_payload["reason"] == "compute_pickle_status_missing"


def test_sync_submission_package_diagnostics_missing_status_retries_until_visible(
    tmp_path: Path, monkeypatch: Any
) -> None:
    package = _build_package(tmp_path)
    package = save_package(
        replace(
            package,
            status="pickle_uploaded",
            artifacts={
                **package.artifacts,
                "last_pickle_model_id": "model-1",
                "last_pickle_upload_id": "upload-1",
            },
        )
    )
    monkeypatch.setattr("numereng.features.serving.evaluation._DIAGNOSTICS_POLL_INTERVAL_SECONDS", 0)

    result = sync_submission_package_diagnostics(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id=package.package_id,
        wait=True,
        client=_EventuallyReadyDiagnosticsClient(),
    )

    assert result.terminal is True
    assert result.raw_path is not None and result.raw_path.is_file()


def test_sync_submission_package_diagnostics_missing_status_times_out(
    tmp_path: Path, monkeypatch: Any
) -> None:
    package = _build_package(tmp_path)
    package = save_package(
        replace(
            package,
            status="pickle_uploaded",
            artifacts={
                **package.artifacts,
                "last_pickle_model_id": "model-1",
                "last_pickle_upload_id": "upload-1",
            },
        )
    )
    monkeypatch.setattr("numereng.features.serving.evaluation._DIAGNOSTICS_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr("numereng.features.serving.evaluation._DIAGNOSTICS_POLL_TIMEOUT_SECONDS", 0)

    result = sync_submission_package_diagnostics(
        workspace_root=tmp_path,
        experiment_id="exp-1",
        package_id=package.package_id,
        wait=True,
        client=_MissingStatusDiagnosticsClient(),
    )

    assert result.terminal is False
    assert result.timed_out is True
    assert result.raw_path is None
