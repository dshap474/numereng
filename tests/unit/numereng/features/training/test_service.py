from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import numereng.features.training.service as service_module
from numereng.features.scoring.models import (
    PostTrainingScoringRequest,
    PostTrainingScoringResult,
    ResolvedScoringPolicy,
    ScoringArtifactBundle,
)
from numereng.features.store import StoreError
from numereng.features.telemetry import bind_launch_metadata
from numereng.features.training import _pipeline as pipeline_module
from numereng.features.training.errors import TrainingConfigError, TrainingError
from numereng.features.training.strategies.core.protocol import TrainingEnginePlan


class _FakeClient:
    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        _ = (filename, round_num)
        return dest_path or filename


@pytest.fixture(autouse=True)
def _bind_training_launch_metadata() -> object:
    with bind_launch_metadata(source="tests.training.service", operation_type="run", job_type="run"):
        yield


def test_resolve_model_config_requires_params() -> None:
    with pytest.raises(TrainingConfigError, match="training_model_params_missing"):
        service_module.resolve_model_config({"type": "LGBMRegressor"})


def test_resolve_model_config_promotes_model_device_to_device_type() -> None:
    model_type, model_params = service_module.resolve_model_config(
        {"type": "LGBMRegressor", "device": "cuda", "params": {"n_estimators": 10}}
    )

    assert model_type == "LGBMRegressor"
    assert model_params["device_type"] == "cuda"


def test_resolve_model_config_rejects_conflicting_device_inputs() -> None:
    with pytest.raises(TrainingConfigError, match="training_model_device_conflict"):
        service_module.resolve_model_config(
            {
                "type": "LGBMRegressor",
                "device": "cuda",
                "params": {"n_estimators": 10, "device_type": "cpu"},
            }
        )


def test_resolve_model_config_rejects_device_for_non_lgbm() -> None:
    with pytest.raises(TrainingConfigError, match="training_model_device_requires_lgbm"):
        service_module.resolve_model_config({"type": "CustomRegressor", "device": "cuda", "params": {"alpha": 1}})


def test_resolve_resource_policy_defaults_max_threads_from_cpu_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "_available_cpu_count", lambda: 10)

    resolved = service_module._resolve_resource_policy(
        {
            "parallel_folds": 2,
            "parallel_backend": "joblib",
        }
    )

    assert resolved["parallel_folds"] == 2
    assert resolved["max_threads_per_worker"] == 5


def test_resolve_resource_policy_default_literal_uses_cpu_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "_available_cpu_count", lambda: 12)

    resolved = service_module._resolve_resource_policy(
        {
            "parallel_folds": 3,
            "parallel_backend": "joblib",
            "max_threads_per_worker": "default",
        }
    )

    assert resolved["parallel_folds"] == 3
    assert resolved["max_threads_per_worker"] == 4


def test_resolve_resource_policy_explicit_max_threads_wins_over_auto_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "_available_cpu_count", lambda: 10)

    resolved = service_module._resolve_resource_policy(
        {
            "parallel_folds": 3,
            "parallel_backend": "joblib",
            "max_threads_per_worker": 8,
        }
    )

    assert resolved["parallel_folds"] == 3
    assert resolved["max_threads_per_worker"] == 8


def test_available_cpu_count_falls_back_to_one_when_detection_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_os_error(pid: int) -> set[int]:
        _ = pid
        raise OSError("affinity unavailable")

    monkeypatch.setattr(os, "sched_getaffinity", _raise_os_error, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert service_module._available_cpu_count() == 1


def test_run_training_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
            "embargo_eras": 13,
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
            "x_groups": ["features"],
        },
        "training": {
            "engine": {"mode": "custom", "window_size_eras": 156, "embargo_eras": 8},
            "post_training_scoring": "core",
        },
    }

    full = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era2"],
            "target": [0.2, 0.4],
            "feature_1": [1.0, 2.0],
        }
    )

    predictions = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era2"],
            "target": [0.2, 0.4],
            "prediction": [0.1, 0.2],
            "cv_fold": [0, 1],
        }
    )

    corr_df = pd.DataFrame(
        [{"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.1}],
        index=["prediction"],
    )
    fnc_df = pd.DataFrame(
        [{"mean": 0.09, "std": 0.21, "sharpe": 0.43, "max_drawdown": 0.11}],
        index=["prediction"],
    )
    bmc_df = pd.DataFrame(
        [
            {
                "mean": 0.01,
                "std": 0.02,
                "sharpe": 0.5,
                "max_drawdown": 0.1,
                "avg_corr_with_benchmark": 0.02,
            }
        ],
        index=["prediction"],
    )
    mmc_df = pd.DataFrame(
        [{"mean": 0.03, "std": 0.04, "sharpe": 0.75, "max_drawdown": 0.09}],
        index=["prediction"],
    )
    cwmm_df = pd.DataFrame(
        [{"mean": 0.2, "std": 0.3, "sharpe": 0.67, "max_drawdown": 0.11}],
        index=["prediction"],
    )
    feature_exposure_df = pd.DataFrame(
        [{"mean": 0.12, "std": 0.05, "sharpe": 2.4, "max_drawdown": 0.08}],
        index=["prediction"],
    )
    max_feature_exposure_df = pd.DataFrame(
        [{"mean": 0.22, "std": 0.06, "sharpe": 3.67, "max_drawdown": 0.09}],
        index=["prediction"],
    )

    store_root = tmp_path / "store"
    output_dir = store_root / "runs" / "run-temp"
    predictions_dir = output_dir / "artifacts" / "predictions"

    predictions_path = predictions_dir / "run.parquet"

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="purged_walk_forward",
            cv_config={"enabled": True, "n_splits": 2, "embargo": 0, "mode": "blocked", "min_train_size": 1},
            resolved_config={"profile": "purged_walk_forward", "window_size_eras": 156, "embargo_eras": 8},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            store_root / "runs" / run_id,
            tmp_path / "baselines",
            store_root / "runs" / run_id,
            store_root / "runs" / run_id / "artifacts" / "predictions",
            store_root / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "load_features",
        lambda *args, **kwargs: ["feature_1"],
    )
    monkeypatch.setattr(
        service_module,
        "load_full_data",
        lambda *args, **kwargs: full,
    )
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())

    def _fake_build_oof_predictions(*args: object, **kwargs: object) -> tuple[pd.DataFrame, dict[str, object]]:
        on_fold_start = cast(Any, kwargs.get("on_fold_start"))
        on_fold_complete = cast(Any, kwargs.get("on_fold_complete"))
        cv_config = cast(dict[str, object], args[5])
        assert cv_config["n_splits"] == 2
        assert cv_config["mode"] == "blocked"
        if on_fold_start is not None:
            on_fold_start(
                {
                    "fold": 0,
                    "train_eras": 1,
                    "val_eras": 1,
                    "val_interval": {"start": "era1", "end": "era1"},
                    "train_intervals": [],
                    "purge_intervals": [],
                    "embargo_intervals": [],
                }
            )
        if on_fold_complete is not None:
            on_fold_complete(
                predictions.iloc[[0]].copy(),
                {
                    "fold": 0,
                    "train_rows": 1,
                    "val_rows": 1,
                    "train_eras": 1,
                    "val_eras": 1,
                    "val_interval": {"start": "era1", "end": "era1"},
                    "train_intervals": [],
                    "purge_intervals": [],
                    "embargo_intervals": [],
                },
            )
        return (
            predictions,
            {
                "n_splits": 2,
                "embargo": 0,
                "mode": "blocked",
                "min_train_size": 1,
                "max_train_eras": None,
                "folds_used": 2,
                "folds": [],
            },
        )

    monkeypatch.setattr(service_module, "build_oof_predictions", _fake_build_oof_predictions)
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            predictions_path,
            Path("predictions/run.parquet"),
        ),
    )
    monkeypatch.setattr(
        service_module,
        "run_scoring",
        lambda **kwargs: PostTrainingScoringResult(
            summaries={
                "corr": corr_df,
                "fnc": fnc_df,
                "mmc": mmc_df,
                "cwmm": cwmm_df,
                "bmc": bmc_df,
                "bmc_last_200_eras": bmc_df,
                "feature_exposure": feature_exposure_df,
                "max_feature_exposure": max_feature_exposure_df,
            },
            score_provenance={
                "schema_version": "1",
                "stages": {
                    "emitted": [
                        "run_metric_series",
                        "post_fold_per_era",
                        "post_fold_snapshots",
                        "post_training_core_summary",
                    ],
                    "omissions": {"post_training_full": "not_requested"},
                },
            },
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
            ),
            artifacts=ScoringArtifactBundle(
                series_frames={},
                stage_frames={
                    "run_metric_series": pd.DataFrame(
                        [
                            {
                                "run_id": "run-123",
                                "config_hash": "config-hash",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "era": "era1",
                                "metric_key": "corr_native",
                                "series_type": "per_era",
                                "value": 0.1,
                            }
                        ]
                    ),
                    "post_training_core_summary": pd.DataFrame(
                        [
                            {
                                "run_id": "run-123",
                                "config_hash": "config-hash",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "corr_native_mean": 0.1,
                                "corr_native_std": 0.2,
                                "corr_native_sharpe": 0.5,
                                "corr_native_max_drawdown": 0.1,
                            }
                        ]
                    ),
                },
                manifest={},
            ),
            requested_stage="post_training_core",
            refreshed_stages=("run_metric_series", "post_training_core"),
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "run.json",
    )
    manifest_statuses: list[str] = []
    save_run_manifest_original = cast(
        Any,
        getattr(service_module, "save_run_manifest"),
    )

    def _record_save_run_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
        status = manifest.get("status")
        if isinstance(status, str):
            manifest_statuses.append(status)
        save_run_manifest_original(manifest, manifest_path)

    monkeypatch.setattr(service_module, "save_run_manifest", _record_save_run_manifest)

    saved_payload: dict[str, object] = {}

    def _fake_save_results(results: dict[str, object], path: Path) -> None:
        saved_payload["results"] = results
        saved_payload["path"] = path

    monkeypatch.setattr(service_module, "save_results", _fake_save_results)
    indexed: dict[str, object] = {}

    def _fake_index_run(*, store_root: str | Path, run_id: str) -> None:
        indexed["store_root"] = Path(store_root)
        indexed["run_id"] = run_id

    monkeypatch.setattr(service_module, "index_run", _fake_index_run)

    result = service_module.run_training(
        config_path=tmp_path / "config.json",
        output_dir=None,
        client=_FakeClient(),
        post_training_scoring="core",
    )

    assert result.predictions_path == predictions_path
    assert result.results_path == store_root / "runs" / result.run_id / "run.json"
    assert len(result.run_id) == 12
    assert saved_payload["path"] == store_root / "runs" / result.run_id / "run.json"
    saved_results = cast(dict[str, object], saved_payload["results"])
    metrics_block = cast(dict[str, object], saved_results["metrics"])
    assert metrics_block["corr"] == {"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.1}
    assert metrics_block["bmc"] == {
        "mean": 0.01,
        "std": 0.02,
        "sharpe": 0.5,
        "max_drawdown": 0.1,
        "avg_corr_with_benchmark": 0.02,
    }
    training_block = cast(dict[str, object], saved_results["training"])
    assert "data_sampling" not in training_block
    engine_block = cast(dict[str, object], training_block["engine"])
    assert engine_block["mode"] == "purged_walk_forward"
    scoring_block = cast(dict[str, object], training_block["scoring"])
    assert scoring_block["policy"] == "core"
    assert scoring_block["status"] == "succeeded"
    assert scoring_block["requested_stage"] == "post_training_core"
    assert scoring_block["refreshed_stages"] == ["run_metric_series", "post_training_core"]
    assert scoring_block["emitted_stage_files"] == ["run_metric_series", "post_training_core_summary"]
    assert scoring_block["omissions"] == {"post_training_full": "not_requested"}
    resources_block = cast(dict[str, object], training_block["resources"])
    assert resources_block["parallel_backend"] == "joblib"
    cache_block = cast(dict[str, object], training_block["cache"])
    assert cache_block["mode"] == "deterministic"
    data_block = cast(dict[str, object], saved_results["data"])
    assert data_block["dataset_scope"] == "train_plus_validation"
    assert data_block["configured_embargo_eras"] == 13
    assert data_block["effective_embargo_eras"] == 0
    assert data_block["embargo_eras"] == 0
    output_block = cast(dict[str, object], saved_results["output"])
    assert output_block["score_provenance_file"] == "score_provenance.json"
    run_dir = store_root / "runs" / result.run_id
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest_artifacts = cast(dict[str, object], run_manifest["artifacts"])
    assert manifest_artifacts["log"] == "run.log"
    assert manifest_artifacts["runtime"] == "runtime.json"
    assert manifest_artifacts["score_provenance"] == "score_provenance.json"
    assert manifest_artifacts["scoring_manifest"] == "artifacts/scoring/manifest.json"
    lifecycle_block = cast(dict[str, object], run_manifest["lifecycle"])
    assert lifecycle_block["terminal_reason"] == "completed"
    assert lifecycle_block["cancel_requested_at"] is None
    assert lifecycle_block["reconciled"] is False
    runtime_snapshot = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    assert runtime_snapshot["run_id"] == result.run_id
    assert runtime_snapshot["status"] == "completed"
    runtime_block = cast(dict[str, object], runtime_snapshot["runtime"])
    assert runtime_block["current_stage"] == "finalize_manifest"
    assert runtime_block["terminal_reason"] == "completed"
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "run_started" in run_log
    assert "stage_update" in run_log
    assert "fold_started" in run_log
    assert "fold_completed" in run_log
    assert "post_run_scoring_succeeded" in run_log
    assert "run_completed" in run_log
    index_stage_pos = run_log.index("stage_update | index_run")
    scoring_stage_pos = run_log.index("stage_update | score_predictions")
    finalize_stage_pos = run_log.index("stage_update | finalize_manifest")
    assert scoring_stage_pos < finalize_stage_pos
    assert scoring_stage_pos < index_stage_pos
    assert manifest_statuses == ["RUNNING", "FINISHED"]
    assert indexed["store_root"] == store_root
    assert indexed["run_id"] == result.run_id


def test_run_training_requires_launch_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
            "x_groups": ["features"],
        },
        "training": {"engine": {"profile": "simple"}},
    }

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="simple",
            cv_config={"enabled": False},
            resolved_config={"profile": "simple"},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "baselines",
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "store" / "runs" / run_id / "artifacts" / "predictions",
            tmp_path / "store" / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(pipeline_module, "get_launch_metadata", lambda: None)

    with pytest.raises(TrainingError, match="training_launch_metadata_missing"):
        service_module.run_training(
            config_path=tmp_path / "config.json",
            output_dir=None,
            client=_FakeClient(),
        )


def test_run_training_submission_full_history_runs_post_training_scoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
            "embargo_eras": 13,
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
            "x_groups": ["features"],
        },
        "training": {"post_training_scoring": "core"},
    }

    full = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era2"],
            "target": [0.2, 0.4],
            "feature_1": [1.0, 2.0],
        }
    )

    predictions = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era2"],
            "target": [0.2, 0.4],
            "prediction": [0.11, 0.22],
        }
    )

    store_root = tmp_path / "store"
    output_dir = store_root / "runs" / "run-temp"
    predictions_dir = output_dir / "artifacts" / "predictions"

    predictions_path = predictions_dir / "run.parquet"
    corr_df = pd.DataFrame(
        [{"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.1}],
        index=["prediction"],
    )

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            store_root / "runs" / run_id,
            tmp_path / "baselines",
            store_root / "runs" / run_id,
            store_root / "runs" / run_id / "artifacts" / "predictions",
            store_root / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "load_features",
        lambda *args, **kwargs: ["feature_1"],
    )
    monkeypatch.setattr(
        service_module,
        "load_full_data",
        lambda *args, **kwargs: full,
    )
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        service_module,
        "build_oof_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OOF path should not run")),
    )
    monkeypatch.setattr(
        service_module,
        "build_full_history_predictions",
        lambda *args, **kwargs: (
            predictions,
            {
                "n_splits": 0,
                "embargo": 0,
                "mode": "full_history_refit",
                "min_train_size": 0,
                "max_train_eras": None,
                "folds_used": 1,
                "folds": [],
            },
        ),
    )
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            predictions_path,
            Path("predictions/run.parquet"),
        ),
    )
    scoring_calls: list[dict[str, object]] = []

    def _fake_score(**kwargs: object) -> PostTrainingScoringResult:
        request = cast(PostTrainingScoringRequest, kwargs["request"])
        scoring_calls.append(
            {
                "run_id": request.run_id,
                "predictions_path": request.predictions_path,
                "stage": request.stage,
            }
        )
        return PostTrainingScoringResult(
            summaries={"corr": corr_df},
            score_provenance={"schema_version": "1"},
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
            ),
            artifacts=ScoringArtifactBundle(
                series_frames={},
                stage_frames={
                    "run_metric_series": pd.DataFrame(
                        [
                            {
                                "run_id": "run-123",
                                "config_hash": "config-hash",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "era": "era1",
                                "metric_key": "corr_native",
                                "series_type": "per_era",
                                "value": 0.1,
                            }
                        ]
                    )
                },
                manifest={},
            ),
            requested_stage="post_training_core",
            refreshed_stages=("run_metric_series", "post_training_core"),
        )

    monkeypatch.setattr(service_module, "run_scoring", _fake_score)
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "run.json",
    )

    saved_payload: dict[str, object] = {}

    def _fake_save_results(results: dict[str, object], path: Path) -> None:
        saved_payload["results"] = results
        saved_payload["path"] = path

    monkeypatch.setattr(service_module, "save_results", _fake_save_results)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    result = service_module.run_training(
        config_path=tmp_path / "config.json",
        output_dir=None,
        client=_FakeClient(),
        post_training_scoring="core",
    )

    assert result.predictions_path == predictions_path
    assert result.results_path == store_root / "runs" / result.run_id / "run.json"
    saved_results = cast(dict[str, object], saved_payload["results"])
    metrics_block = cast(dict[str, object], saved_results["metrics"])
    assert metrics_block["corr"] == {"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.1}
    training_block = cast(dict[str, object], saved_results["training"])
    assert "data_sampling" not in training_block
    engine_block = cast(dict[str, object], training_block["engine"])
    assert engine_block["mode"] == "full_history_refit"
    data_block = cast(dict[str, object], saved_results["data"])
    assert data_block["dataset_scope"] == "train_plus_validation"
    assert data_block["configured_embargo_eras"] == 13
    assert data_block["effective_embargo_eras"] == 0
    output_block = cast(dict[str, object], saved_results["output"])
    assert output_block["score_provenance_file"] == "score_provenance.json"
    assert scoring_calls == [
        {
            "run_id": result.run_id,
            "predictions_path": predictions_path,
            "stage": "post_training_core",
        }
    ]


def test_run_training_index_failure_raises_training_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {"type": "LGBMRegressor", "params": {"n_estimators": 10}, "x_groups": ["features"]},
    }
    full = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "feature_1": [1.0]})
    predictions = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "prediction": [0.2]})
    output_dir = tmp_path / "training"
    predictions_path = output_dir / "predictions" / "run.parquet"

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            output_dir / "runs" / run_id,
            tmp_path / "baselines",
            output_dir / "runs" / run_id,
            output_dir / "runs" / run_id / "artifacts" / "predictions",
            output_dir / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
    monkeypatch.setattr(service_module, "load_full_data", lambda *args, **kwargs: full)
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        service_module,
        "build_full_history_predictions",
        lambda *args, **kwargs: (
            predictions,
            {
                "n_splits": 0,
                "embargo": 0,
                "mode": "full_history_refit",
                "min_train_size": 0,
                "max_train_eras": None,
                "folds_used": 1,
                "folds": [],
            },
        ),
    )
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            predictions_path,
            Path("artifacts/predictions/run.parquet"),
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "run.json",
    )
    monkeypatch.setattr(
        service_module,
        "run_scoring",
        lambda **kwargs: PostTrainingScoringResult(
            summaries={
                "corr": pd.DataFrame(
                    [{"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.1}],
                    index=["prediction"],
                )
            },
            score_provenance={"schema_version": "1"},
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
            ),
            artifacts=ScoringArtifactBundle(
                series_frames={},
                stage_frames={
                    "run_metric_series": pd.DataFrame(
                        [
                            {
                                "run_id": "run-123",
                                "config_hash": "config-hash",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "era": "era1",
                                "metric_key": "corr_native",
                                "series_type": "per_era",
                                "value": 0.1,
                            }
                        ]
                    )
                },
                manifest={},
            ),
            requested_stage="post_training_core",
            refreshed_stages=("run_metric_series", "post_training_core"),
        ),
    )
    monkeypatch.setattr(service_module, "save_results", lambda results, path: None)
    monkeypatch.setattr(service_module, "save_metrics", lambda metrics, path: None)
    monkeypatch.setattr(service_module, "save_run_manifest", lambda manifest, manifest_path: None)
    monkeypatch.setattr(service_module, "save_resolved_config", lambda config, resolved_config_path: None)
    monkeypatch.setattr(
        service_module,
        "index_run",
        lambda **kwargs: (_ for _ in ()).throw(StoreError("store_failed")),
    )

    with pytest.raises(TrainingError, match="training_store_index_failed"):
        service_module.run_training(
            config_path=tmp_path / "config.json",
            output_dir=None,
            client=_FakeClient(),
        )


def test_run_training_index_failure_writes_failed_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {"type": "LGBMRegressor", "params": {"n_estimators": 10}, "x_groups": ["features"]},
    }
    full = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "feature_1": [1.0]})
    predictions = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "prediction": [0.2]})

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "baselines",
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "store" / "runs" / run_id / "artifacts" / "predictions",
            tmp_path / "store" / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
    monkeypatch.setattr(service_module, "load_full_data", lambda *args, **kwargs: full)
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        service_module,
        "build_full_history_predictions",
        lambda *args, **kwargs: (
            predictions,
            {
                "n_splits": 0,
                "embargo": 0,
                "mode": "full_history_refit",
                "min_train_size": 0,
                "max_train_eras": None,
                "folds_used": 1,
                "folds": [],
            },
        ),
    )
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            tmp_path / "store" / "runs" / "run-temp" / "predictions.parquet",
            Path("artifacts/predictions/predictions.parquet"),
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "run.json",
    )
    monkeypatch.setattr(
        service_module,
        "run_scoring",
        lambda **kwargs: PostTrainingScoringResult(
            summaries={
                "corr": pd.DataFrame(
                    [{"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.1}],
                    index=["prediction"],
                )
            },
            score_provenance={"schema_version": "1"},
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
            ),
            artifacts=ScoringArtifactBundle(
                series_frames={},
                stage_frames={
                    "run_metric_series": pd.DataFrame(
                        [
                            {
                                "run_id": "run-123",
                                "config_hash": "config-hash",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "era": "era1",
                                "metric_key": "corr_native",
                                "series_type": "per_era",
                                "value": 0.1,
                            }
                        ]
                    )
                },
                manifest={},
            ),
            requested_stage="post_training_core",
            refreshed_stages=("run_metric_series", "post_training_core"),
        ),
    )
    monkeypatch.setattr(service_module, "save_results", lambda results, path: None)
    monkeypatch.setattr(service_module, "save_metrics", lambda metrics, path: None)
    monkeypatch.setattr(service_module, "save_resolved_config", lambda config, resolved_config_path: None)
    monkeypatch.setattr(
        service_module,
        "index_run",
        lambda **kwargs: (_ for _ in ()).throw(StoreError("store_failed")),
    )

    written_statuses: list[str] = []

    def _record_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
        _ = manifest_path
        status = manifest.get("status")
        if isinstance(status, str):
            written_statuses.append(status)

    monkeypatch.setattr(service_module, "save_run_manifest", _record_manifest)

    with pytest.raises(TrainingError, match="training_store_index_failed"):
        service_module.run_training(
            config_path=tmp_path / "config.json",
            output_dir=None,
            client=_FakeClient(),
        )

    assert written_statuses == ["RUNNING", "FAILED"]


def test_run_training_second_index_failure_keeps_finished_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {"type": "LGBMRegressor", "params": {"n_estimators": 10}, "x_groups": ["features"]},
    }
    full = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "feature_1": [1.0]})
    predictions = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "prediction": [0.2]})

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "baselines",
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "store" / "runs" / run_id / "artifacts" / "predictions",
            tmp_path / "store" / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
    monkeypatch.setattr(service_module, "load_full_data", lambda *args, **kwargs: full)
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        service_module,
        "build_full_history_predictions",
        lambda *args, **kwargs: (
            predictions,
            {
                "n_splits": 0,
                "embargo": 0,
                "mode": "full_history_refit",
                "min_train_size": 0,
                "max_train_eras": None,
                "folds_used": 1,
                "folds": [],
            },
        ),
    )
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            tmp_path / "store" / "runs" / "run-temp" / "predictions.parquet",
            Path("artifacts/predictions/predictions.parquet"),
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "run.json",
    )
    monkeypatch.setattr(
        service_module,
        "run_scoring",
        lambda **kwargs: PostTrainingScoringResult(
            summaries={
                "corr": pd.DataFrame(
                    [{"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.1}],
                    index=["prediction"],
                )
            },
            score_provenance={"schema_version": "1"},
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
            ),
            artifacts=ScoringArtifactBundle(
                series_frames={},
                stage_frames={
                    "run_metric_series": pd.DataFrame(
                        [
                            {
                                "run_id": "run-123",
                                "config_hash": "config-hash",
                                "seed": None,
                                "target_col": "target",
                                "payout_target_col": "target_ender_20",
                                "prediction_col": "prediction",
                                "era": "era1",
                                "metric_key": "corr_native",
                                "series_type": "per_era",
                                "value": 0.1,
                            }
                        ]
                    )
                },
                manifest={},
            ),
            requested_stage="post_training_core",
            refreshed_stages=("run_metric_series", "post_training_core"),
        ),
    )
    monkeypatch.setattr(service_module, "save_results", lambda results, path: None)
    monkeypatch.setattr(service_module, "save_metrics", lambda metrics, path: None)
    monkeypatch.setattr(service_module, "save_resolved_config", lambda config, resolved_config_path: None)

    index_calls: list[int] = []

    def _index_run(**kwargs: object) -> None:
        _ = kwargs
        index_calls.append(1)
        if len(index_calls) == 2:
            raise StoreError("store_failed")

    monkeypatch.setattr(service_module, "index_run", _index_run)

    written_statuses: list[str] = []

    def _record_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
        _ = manifest_path
        status = manifest.get("status")
        if isinstance(status, str):
            written_statuses.append(status)

    monkeypatch.setattr(service_module, "save_run_manifest", _record_manifest)

    result = service_module.run_training(
        config_path=tmp_path / "config.json",
        output_dir=None,
        client=_FakeClient(),
    )

    assert result.run_id
    assert len(index_calls) == 2
    assert written_statuses == ["RUNNING", "FINISHED"]


def test_run_training_startup_write_failure_writes_failed_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {"type": "LGBMRegressor", "params": {"n_estimators": 10}, "x_groups": ["features"]},
    }

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "baselines",
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "store" / "runs" / run_id / "artifacts" / "predictions",
            tmp_path / "store" / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "save_resolved_config",
        lambda config, resolved_config_path: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    written_statuses: list[str] = []

    def _record_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
        _ = manifest_path
        status = manifest.get("status")
        if isinstance(status, str):
            written_statuses.append(status)

    monkeypatch.setattr(service_module, "save_run_manifest", _record_manifest)

    with pytest.raises(OSError, match="disk full"):
        service_module.run_training(
            config_path=tmp_path / "config.json",
            output_dir=None,
            client=_FakeClient(),
        )

    assert written_statuses == ["RUNNING", "FAILED"]


def test_run_training_unexpected_exception_writes_failed_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {"type": "LGBMRegressor", "params": {"n_estimators": 10}, "x_groups": ["features"]},
    }
    full = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "feature_1": [1.0]})
    predictions = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "prediction": [0.2]})

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "baselines",
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "store" / "runs" / run_id / "artifacts" / "predictions",
            tmp_path / "store" / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
    monkeypatch.setattr(service_module, "load_full_data", lambda *args, **kwargs: full)
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        service_module,
        "build_full_history_predictions",
        lambda *args, **kwargs: (
            predictions,
            {
                "n_splits": 0,
                "embargo": 0,
                "mode": "full_history_refit",
                "min_train_size": 0,
                "max_train_eras": None,
                "folds_used": 1,
                "folds": [],
            },
        ),
    )
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            tmp_path / "store" / "runs" / "run-temp" / "predictions.parquet",
            Path("artifacts/predictions/predictions.parquet"),
        ),
    )
    monkeypatch.setattr(service_module, "save_resolved_config", lambda config, resolved_config_path: None)
    monkeypatch.setattr(service_module, "save_results", lambda results, path: None)
    monkeypatch.setattr(service_module, "save_metrics", lambda metrics, path: None)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    written_statuses: list[str] = []

    def _record_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
        _ = manifest_path
        status = manifest.get("status")
        if isinstance(status, str):
            written_statuses.append(status)

    monkeypatch.setattr(service_module, "save_run_manifest", _record_manifest)

    with pytest.raises(RuntimeError, match="boom"):
        service_module.run_training(
            config_path=tmp_path / "config.json",
            output_dir=None,
            client=_FakeClient(),
        )

    assert written_statuses == ["RUNNING", "FAILED"]


def test_run_training_failed_run_writes_run_local_failure_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {"type": "LGBMRegressor", "params": {"n_estimators": 10}, "x_groups": ["features"]},
    }
    full = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "feature_1": [1.0]})
    predictions = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "prediction": [0.2]})
    captured_run_id: dict[str, str] = {}

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )

    def _resolve_output_locations(cfg: object, override: object, run_id: str) -> tuple[Path, Path, Path, Path, Path]:
        _ = (cfg, override)
        captured_run_id["run_id"] = run_id
        return (
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "baselines",
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "store" / "runs" / run_id / "artifacts" / "predictions",
            tmp_path / "store" / "runs" / run_id / "artifacts" / "scoring",
        )

    monkeypatch.setattr(service_module, "resolve_output_locations", _resolve_output_locations)
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
    monkeypatch.setattr(service_module, "load_full_data", lambda *args, **kwargs: full)
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        service_module,
        "build_full_history_predictions",
        lambda *args, **kwargs: (
            predictions,
            {
                "n_splits": 0,
                "embargo": 0,
                "mode": "full_history_refit",
                "min_train_size": 0,
                "max_train_eras": None,
                "folds_used": 1,
                "folds": [],
            },
        ),
    )
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            tmp_path / "store" / "runs" / "run-temp" / "predictions.parquet",
            Path("artifacts/predictions/predictions.parquet"),
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "results.json",
    )
    monkeypatch.setattr(
        service_module,
        "save_results",
        lambda results, path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(service_module, "save_resolved_config", lambda config, resolved_config_path: None)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="boom"):
        service_module.run_training(
            config_path=tmp_path / "config.json",
            output_dir=None,
            client=_FakeClient(),
        )

    run_dir = tmp_path / "store" / "runs" / captured_run_id["run_id"]
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_manifest["status"] == "FAILED"
    manifest_artifacts = cast(dict[str, object], run_manifest["artifacts"])
    assert manifest_artifacts["log"] == "run.log"
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "run_failed" in run_log
    assert "boom" in run_log


def test_run_training_rejects_non_fresh_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {"type": "LGBMRegressor", "params": {"n_estimators": 10}, "x_groups": ["features"]},
    }
    captured_run_dir: dict[str, Path] = {}

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )

    def _resolve_output_locations(cfg: object, override: object, run_id: str) -> tuple[Path, Path, Path, Path, Path]:
        _ = (cfg, override)
        run_dir = tmp_path / "store" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stale.txt").write_text("stale", encoding="utf-8")
        captured_run_dir["path"] = run_dir
        return (
            run_dir,
            tmp_path / "baselines",
            run_dir,
            run_dir / "artifacts" / "predictions",
            run_dir / "artifacts" / "scoring",
        )

    monkeypatch.setattr(service_module, "resolve_output_locations", _resolve_output_locations)

    with pytest.raises(TrainingError, match="training_run_dir_not_fresh"):
        service_module.run_training(
            config_path=tmp_path / "config.json",
            output_dir=None,
            client=_FakeClient(),
        )

    run_dir = captured_run_dir["path"]
    assert (run_dir / "stale.txt").is_file()
    assert not (run_dir / "run.json").exists()


def test_run_training_scoring_failure_keeps_finished_run_and_records_failed_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
            "embargo_eras": 13,
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
            "x_groups": ["features"],
        },
        "training": {
            "engine": {"mode": "custom", "window_size_eras": 156, "embargo_eras": 8},
            "post_training_scoring": "core",
        },
    }
    full = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era2"],
            "target": [0.2, 0.4],
            "feature_1": [1.0, 2.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era2"],
            "target": [0.2, 0.4],
            "prediction": [0.1, 0.2],
            "cv_fold": [0, 1],
        }
    )

    store_root = tmp_path / "store"
    captured_run_dir: dict[str, Path] = {}

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="purged_walk_forward",
            cv_config={"enabled": True, "n_splits": 2, "embargo": 0, "mode": "blocked", "min_train_size": 1},
            resolved_config={"profile": "purged_walk_forward", "window_size_eras": 156, "embargo_eras": 8},
            override_sources=["default"],
        ),
    )

    def _resolve_output_locations(cfg: object, override: object, run_id: str) -> tuple[Path, Path, Path, Path, Path]:
        _ = (cfg, override)
        run_dir = store_root / "runs" / run_id
        captured_run_dir["path"] = run_dir
        return (
            run_dir,
            tmp_path / "baselines",
            run_dir,
            run_dir / "artifacts" / "predictions",
            run_dir / "artifacts" / "scoring",
        )

    monkeypatch.setattr(service_module, "resolve_output_locations", _resolve_output_locations)
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
    monkeypatch.setattr(service_module, "load_full_data", lambda *args, **kwargs: full)
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        service_module,
        "build_oof_predictions",
        lambda *args, **kwargs: (
            predictions,
            {
                "n_splits": 2,
                "embargo": 0,
                "mode": "blocked",
                "min_train_size": 1,
                "max_train_eras": None,
                "folds_used": 2,
                "folds": [],
            },
        ),
    )
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            output_dir / "artifacts" / "predictions" / "run.parquet",
            Path("predictions/run.parquet"),
        ),
    )
    scoring_called = {"value": False}

    def _raise_scoring_failure(**kwargs: object) -> PostTrainingScoringResult:
        _ = kwargs
        scoring_called["value"] = True
        raise RuntimeError("score boom")

    monkeypatch.setattr(service_module, "run_scoring", _raise_scoring_failure)
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "run.json",
    )
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    written_statuses: list[str] = []
    save_run_manifest_original = cast(Any, getattr(service_module, "save_run_manifest"))

    def _record_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
        status = manifest.get("status")
        if isinstance(status, str):
            written_statuses.append(status)
        save_run_manifest_original(manifest, manifest_path)

    monkeypatch.setattr(service_module, "save_run_manifest", _record_manifest)

    result = service_module.run_training(
        config_path=tmp_path / "config.json",
        output_dir=None,
        client=_FakeClient(),
        post_training_scoring="core",
    )

    run_dir = captured_run_dir["path"]
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert result.run_id
    assert scoring_called["value"] is True
    assert run_manifest["status"] == "FINISHED"
    assert written_statuses == ["RUNNING", "FINISHED"]
    scoring_block = cast(dict[str, object], cast(dict[str, object], run_manifest["training"])["scoring"])
    assert scoring_block["policy"] == "core"
    assert scoring_block["status"] == "failed"
    assert scoring_block["requested_stage"] == "post_training_core"
    assert scoring_block["reason"] == "post_training_scoring_failed"
    assert scoring_block["error"] == "score boom"
    metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["status"] == "failed"
    assert metrics_payload["reason"] == "post_training_scoring_failed"
    assert "post_run_scoring_failed" in run_log
    assert "run_completed" in run_log


def test_run_training_defaults_post_training_scoring_to_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
            "x_groups": ["features"],
        },
        "training": {"engine": {"profile": "full_history_refit"}},
    }
    full = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "feature_1": [1.0]})
    predictions = pd.DataFrame({"id": ["a"], "era": ["era1"], "target": [0.1], "prediction": [0.2]})
    store_root = tmp_path / "store"
    run_scoring_called = {"value": False}

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "resolve_training_engine",
        lambda **kwargs: TrainingEnginePlan(
            mode="full_history_refit",
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
                "max_train_eras": None,
            },
            resolved_config={"mode": "full_history_refit"},
            override_sources=["default"],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "resolve_output_locations",
        lambda cfg, override, run_id: (
            store_root / "runs" / run_id,
            tmp_path / "baselines",
            store_root / "runs" / run_id,
            store_root / "runs" / run_id / "artifacts" / "predictions",
            store_root / "runs" / run_id / "artifacts" / "scoring",
        ),
    )
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
    monkeypatch.setattr(service_module, "load_full_data", lambda *args, **kwargs: full)
    monkeypatch.setattr(service_module, "build_model_data_loader", lambda **kwargs: object())
    monkeypatch.setattr(
        service_module,
        "build_full_history_predictions",
        lambda *args, **kwargs: (
            predictions,
            {
                "n_splits": 0,
                "embargo": 0,
                "mode": "full_history_refit",
                "min_train_size": 0,
                "max_train_eras": None,
                "folds_used": 1,
                "folds": [],
            },
        ),
    )
    monkeypatch.setattr(service_module, "select_prediction_columns", lambda df, id_col, era_col, target_col: df)
    monkeypatch.setattr(
        service_module,
        "save_predictions",
        lambda predictions, config, config_path, predictions_dir, output_dir: (
            output_dir / "artifacts" / "predictions" / "run.parquet",
            Path("predictions/run.parquet"),
        ),
    )
    monkeypatch.setattr(
        service_module,
        "run_scoring",
        lambda **kwargs: run_scoring_called.__setitem__("value", True),  # pragma: no cover - should not run
    )
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "run.json",
    )
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    result = service_module.run_training(
        config_path=tmp_path / "config.json",
        output_dir=None,
        client=_FakeClient(),
    )

    run_dir = store_root / "runs" / result.run_id
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    scoring_block = cast(dict[str, object], cast(dict[str, object], run_manifest["training"])["scoring"])
    assert run_scoring_called["value"] is False
    assert scoring_block["policy"] == "none"
    assert scoring_block["status"] == "deferred"
    assert scoring_block["reason"] == "post_training_scoring_disabled"
