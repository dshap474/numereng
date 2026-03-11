from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import numereng.features.training.service as service_module
from numereng.features.store import StoreError
from numereng.features.training.errors import TrainingConfigError, TrainingError
from numereng.features.scoring.models import (
    PostTrainingScoringRequest,
    PostTrainingScoringResult,
    ResolvedScoringPolicy,
)
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
        service_module.resolve_model_config(
            {"type": "CustomRegressor", "device": "cuda", "params": {"alpha": 1}}
        )


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
        "training": {"engine": {"mode": "custom", "window_size_eras": 156, "embargo_eras": 8}},
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
        _ = kwargs
        cv_config = cast(dict[str, object], args[5])
        assert cv_config["n_splits"] == 2
        assert cv_config["mode"] == "blocked"
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
        "run_post_training_scoring",
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
            score_provenance={"schema_version": "1"},
            effective_scoring_backend="materialized",
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
                include_feature_neutral_metrics=True,
            ),
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
    )

    assert result.predictions_path == predictions_path
    assert result.results_path == store_root / "runs" / result.run_id / "run.json"
    assert len(result.run_id) == 12
    assert saved_payload["path"] == store_root / "runs" / result.run_id / "run.json"
    saved_results = cast(dict[str, object], saved_payload["results"])
    metrics_block = cast(dict[str, object], saved_results["metrics"])
    assert "corr" in metrics_block
    assert "fnc" in metrics_block
    assert "mmc" in metrics_block
    assert "cwmm" in metrics_block
    assert "bmc" in metrics_block
    assert "payout_estimate_mean" not in metrics_block
    training_block = cast(dict[str, object], saved_results["training"])
    assert "data_sampling" not in training_block
    engine_block = cast(dict[str, object], training_block["engine"])
    assert engine_block["mode"] == "purged_walk_forward"
    loading_block = cast(dict[str, object], training_block["loading"])
    assert loading_block["mode"] == "materialized"
    scoring_block = cast(dict[str, object], training_block["scoring"])
    assert scoring_block["mode"] == "materialized"
    assert scoring_block["effective_backend"] == "materialized"
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
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "run_started" in run_log
    assert "stage_update" in run_log
    assert "run_completed" in run_log
    index_stage_pos = run_log.index("stage_update | index_run")
    score_stage_pos = run_log.index("stage_update | score_predictions_post_run")
    finalize_stage_pos = run_log.index("stage_update | finalize_manifest")
    assert index_stage_pos < score_stage_pos < finalize_stage_pos
    assert manifest_statuses == ["RUNNING", "FINISHED"]
    assert indexed["store_root"] == store_root
    assert indexed["run_id"] == result.run_id


def test_run_training_submission_full_history_skips_validation_metrics(
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
    monkeypatch.setattr(
        service_module,
        "run_post_training_scoring",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Metrics summary should not run")),
    )
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
    )

    assert result.predictions_path == predictions_path
    assert result.results_path == store_root / "runs" / result.run_id / "run.json"
    saved_results = cast(dict[str, object], saved_payload["results"])
    assert cast(dict[str, object], saved_results["metrics"])["status"] == "not_applicable"
    training_block = cast(dict[str, object], saved_results["training"])
    assert "data_sampling" not in training_block
    engine_block = cast(dict[str, object], training_block["engine"])
    assert engine_block["mode"] == "full_history_refit"
    loading_block = cast(dict[str, object], training_block["loading"])
    assert loading_block["mode"] == "materialized"
    data_block = cast(dict[str, object], saved_results["data"])
    assert data_block["dataset_scope"] == "train_plus_validation"
    assert data_block["configured_embargo_eras"] == 13
    assert data_block["effective_embargo_eras"] == 0
    output_block = cast(dict[str, object], saved_results["output"])
    assert output_block["score_provenance_file"] is None


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


def test_run_training_fold_lazy_routes_scoring_and_loading_modes(
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
            "dataset_scope": "train_plus_validation",
            "loading": {
                "mode": "fold_lazy",
                "scoring_mode": "era_stream",
                "era_chunk_size": 8,
            },
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
            "x_groups": ["features"],
        },
        "training": {
            "engine": {"mode": "custom", "window_size_eras": 156, "embargo_eras": 8},
            "resources": {
                "parallel_folds": 1,
                "parallel_backend": "joblib",
                "memmap_enabled": True,
                "max_threads_per_worker": 1,
            },
            "cache": {
                "mode": "deterministic",
                "cache_fold_specs": True,
                "cache_features": True,
                "cache_labels": True,
                "cache_fold_matrices": False,
            },
        },
    }

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
    predictions_path = store_root / "runs" / "run-temp" / "artifacts" / "predictions" / "run.parquet"
    source_paths = (
        tmp_path / "datasets" / "v5.2" / "train.parquet",
        tmp_path / "datasets" / "v5.2" / "validation.parquet",
    )

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
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
        ),
    )
    monkeypatch.setattr(service_module, "resolve_fold_lazy_source_paths", lambda *args, **kwargs: source_paths)

    def _fake_list_lazy_source_eras(*args: object, **kwargs: object) -> list[str]:
        assert kwargs["include_validation_only"] is True
        return ["era1", "era2"]

    monkeypatch.setattr(service_module, "list_lazy_source_eras", _fake_list_lazy_source_eras)

    def _fake_load_fold_data_lazy(*args: object, **kwargs: object) -> pd.DataFrame:
        assert kwargs["include_validation_only"] is True
        return pd.DataFrame({"era": ["era1", "era2"]})

    monkeypatch.setattr(service_module, "load_fold_data_lazy", _fake_load_fold_data_lazy)

    def _fake_build_lazy_parquet_data_loader(**kwargs: object) -> object:
        assert kwargs["include_validation_only"] is True
        return object()

    monkeypatch.setattr(service_module, "build_lazy_parquet_data_loader", _fake_build_lazy_parquet_data_loader)
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
            predictions_path,
            Path("predictions/run.parquet"),
        ),
    )

    scoring_calls: list[dict[str, object]] = []

    def _fake_score(**kwargs: object) -> PostTrainingScoringResult:
        request = cast(PostTrainingScoringRequest, kwargs["request"])
        scoring_calls.append(
            {
                "scoring_mode": request.scoring_mode,
                "era_chunk_size": request.era_chunk_size,
            }
        )
        return PostTrainingScoringResult(
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
                "execution": {
                    "requested_scoring_mode": "era_stream",
                    "effective_scoring_mode": "era_stream",
                    "era_chunk_size": 8,
                },
            },
            effective_scoring_backend="era_stream",
            policy=ResolvedScoringPolicy(
                fnc_feature_set="fncv3_features",
                fnc_target_policy="scoring_target",
                benchmark_min_overlap_ratio=0.0,
                include_feature_neutral_metrics=True,
            ),
        )

    monkeypatch.setattr(service_module, "run_post_training_scoring", _fake_score)
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
    )

    assert result.predictions_path == predictions_path
    assert scoring_calls
    score_call = scoring_calls[0]
    assert score_call["scoring_mode"] == "era_stream"
    assert score_call["era_chunk_size"] == 8

    saved_results = cast(dict[str, object], saved_payload["results"])
    training_block = cast(dict[str, object], saved_results["training"])
    assert "data_sampling" not in training_block
    assert cast(dict[str, object], training_block["loading"])["mode"] == "fold_lazy"
    scoring_block = cast(dict[str, object], training_block["scoring"])
    assert scoring_block["mode"] == "era_stream"
    assert scoring_block["effective_backend"] == "era_stream"
    data_block = cast(dict[str, object], saved_results["data"])
    assert data_block["dataset_scope"] == "train_plus_validation"


def test_run_training_fold_lazy_full_data_path_disables_validation_only_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    full_data_path = tmp_path / "full.parquet"
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": "small",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
            "full_data_path": str(full_data_path),
            "loading": {
                "mode": "fold_lazy",
                "scoring_mode": "materialized",
                "era_chunk_size": 8,
            },
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
            "x_groups": ["features"],
        },
        "training": {
            "engine": {"mode": "full_history_refit"},
        },
    }
    store_root = tmp_path / "store"
    predictions_path = store_root / "runs" / "run-temp" / "artifacts" / "predictions" / "run.parquet"
    source_paths = (full_data_path,)
    predictions = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era2"],
            "target": [0.2, 0.4],
            "prediction": [0.1, 0.2],
        }
    )

    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(service_module, "load_features", lambda *args, **kwargs: ["feature_1"])
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
        ),
    )
    monkeypatch.setattr(service_module, "resolve_fold_lazy_source_paths", lambda *args, **kwargs: source_paths)

    def _fake_list_lazy_source_eras(*args: object, **kwargs: object) -> list[str]:
        assert kwargs["include_validation_only"] is False
        return ["era1", "era2"]

    monkeypatch.setattr(service_module, "list_lazy_source_eras", _fake_list_lazy_source_eras)

    def _fake_load_fold_data_lazy(*args: object, **kwargs: object) -> pd.DataFrame:
        assert kwargs["include_validation_only"] is False
        return pd.DataFrame({"era": ["era1", "era2"]})

    monkeypatch.setattr(service_module, "load_fold_data_lazy", _fake_load_fold_data_lazy)

    def _fake_build_lazy_parquet_data_loader(**kwargs: object) -> object:
        assert kwargs["include_validation_only"] is False
        return object()

    monkeypatch.setattr(service_module, "build_lazy_parquet_data_loader", _fake_build_lazy_parquet_data_loader)
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
    monkeypatch.setattr(
        service_module,
        "resolve_results_path",
        lambda cfg, path, results_dir: results_dir / "run.json",
    )
    monkeypatch.setattr(service_module, "save_results", lambda results, path: None)
    monkeypatch.setattr(service_module, "save_metrics", lambda metrics, path: None)
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)

    result = service_module.run_training(
        config_path=tmp_path / "config.json",
        output_dir=None,
        client=_FakeClient(),
    )
    assert result.predictions_path == predictions_path


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

    def _resolve_output_locations(cfg: object, override: object, run_id: str) -> tuple[Path, Path, Path, Path]:
        _ = (cfg, override)
        captured_run_id["run_id"] = run_id
        return (
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "baselines",
            tmp_path / "store" / "runs" / run_id,
            tmp_path / "store" / "runs" / run_id / "artifacts" / "predictions",
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

    def _resolve_output_locations(cfg: object, override: object, run_id: str) -> tuple[Path, Path, Path, Path]:
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


def test_run_training_scoring_failure_marks_run_failed(
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
        "training": {"engine": {"mode": "custom", "window_size_eras": 156, "embargo_eras": 8}},
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

    def _resolve_output_locations(cfg: object, override: object, run_id: str) -> tuple[Path, Path, Path, Path]:
        _ = (cfg, override)
        run_dir = store_root / "runs" / run_id
        captured_run_dir["path"] = run_dir
        return (
            run_dir,
            tmp_path / "baselines",
            run_dir,
            run_dir / "artifacts" / "predictions",
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
    def _raise_scoring_failure(**kwargs: object) -> PostTrainingScoringResult:
        _ = kwargs
        raise RuntimeError("score boom")

    monkeypatch.setattr(service_module, "run_post_training_scoring", _raise_scoring_failure)
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

    with pytest.raises(TrainingError, match="training_post_run_scoring_failed"):
        service_module.run_training(
            config_path=tmp_path / "config.json",
            output_dir=None,
            client=_FakeClient(),
        )

    run_dir = captured_run_dir["path"]
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert run_manifest["status"] == "FAILED"
    assert written_statuses == ["RUNNING", "FAILED"]
    assert "post_run_scoring_failed" in run_log
    assert "run_failed" in run_log
