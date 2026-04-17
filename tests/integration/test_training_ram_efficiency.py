from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import numereng.api as api_module
import numereng.features.training.service as training_service_module
from numereng.api.contracts import TrainRunRequest
from numereng.features.scoring.metrics import summarize_prediction_file_with_scores


@dataclass(frozen=True)
class _SyntheticDataset:
    features: list[str]
    train_path: Path
    validation_path: Path
    benchmark_path: Path
    meta_path: Path


class _NoDownloadClient:
    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        _ = (round_num,)
        if dest_path is not None:
            return dest_path
        return filename


def _write_features_metadata(root: Path, features: list[str]) -> None:
    version_dir = root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    payload = {"feature_sets": {"small": features, "all": features, "fncv3_features": features}}
    (version_dir / "features.json").write_text(json.dumps(payload), encoding="utf-8")


def _build_synthetic_dataset(
    root: Path,
    *,
    n_eras: int,
    rows_per_era: int,
    n_features: int,
    seed: int,
) -> _SyntheticDataset:
    rng = np.random.default_rng(seed)
    feature_cols = [f"feature_{idx}" for idx in range(n_features)]
    version_dir = root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    for era_idx in range(1, n_eras + 1):
        era = f"{era_idx:04d}"
        ids = [f"{era}_{row_idx:05d}" for row_idx in range(rows_per_era)]
        frame = pd.DataFrame(
            {
                "id": ids,
                "era": [era] * rows_per_era,
                "target": rng.random(rows_per_era),
            }
        )
        for feature_col in feature_cols:
            frame[feature_col] = rng.random(rows_per_era).astype("float32")
        rows.append(frame)

    full = pd.concat(rows, ignore_index=True)
    train_eras = [f"{idx:04d}" for idx in range(1, n_eras - 1)]
    validation_eras = [f"{idx:04d}" for idx in range(n_eras - 1, n_eras + 1)]

    train = full[full["era"].isin(train_eras)].copy()
    validation = full[full["era"].isin(validation_eras)].copy()
    validation["data_type"] = "validation"
    # Keep one non-validation row to verify lazy-mode filtering parity.
    if not validation.empty:
        validation.loc[validation.index[0], "data_type"] = "live"

    train_path = version_dir / "train.parquet"
    validation_path = version_dir / "validation.parquet"
    benchmark_path = version_dir / "benchmark.parquet"
    meta_path = version_dir / "meta.parquet"

    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)

    benchmark = full[["id", "era"]].copy()
    benchmark["v52_lgbm_ender20"] = rng.random(len(benchmark))
    benchmark.to_parquet(benchmark_path, index=False)

    meta = full[["id", "era"]].copy()
    meta["numerai_meta_model"] = rng.random(len(meta))
    meta.to_parquet(meta_path, index=False)

    return _SyntheticDataset(
        features=feature_cols,
        train_path=train_path,
        validation_path=validation_path,
        benchmark_path=benchmark_path,
        meta_path=meta_path,
    )


def _stage_default_dataset_root(root: Path, dataset: _SyntheticDataset) -> None:
    version_dir = root / ".numereng" / "datasets" / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "v5.2" / "features.json", version_dir / "features.json")
    shutil.copyfile(dataset.train_path, version_dir / "train.parquet")
    shutil.copyfile(dataset.validation_path, version_dir / "validation.parquet")


def _build_training_config(
    *,
    dataset: _SyntheticDataset,
    store_root: Path,
    parallel_folds: int,
    post_training_scoring: str = "none",
) -> dict[str, object]:
    data_block: dict[str, object] = {
        "data_version": "v5.2",
        "dataset_variant": "non_downsampled",
        "feature_set": "small",
        "target_col": "target",
        "scoring_targets": ["target"],
        "era_col": "era",
        "id_col": "id",
        "benchmark_source": {
            "source": "path",
            "predictions_path": str(dataset.benchmark_path),
            "pred_col": "v52_lgbm_ender20",
        },
        "meta_model_data_path": str(dataset.meta_path),
    }
    return {
        "data": data_block,
        "model": {
            "type": "LGBMRegressor",
            "params": {
                "n_estimators": 12,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "random_state": 13,
                "feature_fraction_seed": 13,
                "bagging_seed": 13,
                "data_random_seed": 13,
                "deterministic": True,
                "force_col_wise": True,
                "num_threads": 1,
                "verbose": -1,
            },
            "x_groups": ["features"],
        },
        "training": {
            "engine": {
                "profile": "simple",
            },
            "post_training_scoring": post_training_scoring,
            "resources": {
                "parallel_folds": parallel_folds,
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
        "output": {"output_dir": str(store_root)},
    }


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.mark.integration
def test_run_training_materialized_smoke_indexes_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _build_synthetic_dataset(tmp_path, n_eras=6, rows_per_era=180, n_features=8, seed=7)
    store_root = tmp_path / ".numereng"
    _write_features_metadata(tmp_path, dataset.features)
    _stage_default_dataset_root(tmp_path, dataset)

    config_path = tmp_path / "train_materialized.json"
    config_payload = _build_training_config(
        dataset=dataset,
        store_root=store_root,
        parallel_folds=1,
        post_training_scoring="core",
    )
    _write_config(config_path, config_payload)

    monkeypatch.setattr(training_service_module, "create_training_data_client", lambda: _NoDownloadClient())
    monkeypatch.setattr(
        training_service_module,
        "load_features",
        lambda client, data_version, feature_set, dataset_variant="non_downsampled": dataset.features,
    )

    response = api_module.run_training(
        TrainRunRequest(
            config_path=str(config_path),
            output_dir=str(store_root),
        )
    )

    run_dir = store_root / "runs" / response.run_id
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "runtime.json").is_file()
    assert (run_dir / "resolved.json").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "score_provenance.json").is_file()
    assert (run_dir / "artifacts" / "predictions").is_dir()

    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_manifest["status"] == "FINISHED"
    lifecycle_block = cast(dict[str, object], run_manifest["lifecycle"])
    assert lifecycle_block["terminal_reason"] == "completed"
    runtime_snapshot = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    assert runtime_snapshot["run_id"] == response.run_id
    assert runtime_snapshot["status"] == "completed"
    training_block = cast(dict[str, object], run_manifest["training"])
    scoring_block = cast(dict[str, object], training_block["scoring"])
    assert scoring_block["policy"] == "core"
    assert scoring_block["status"] == "succeeded"
    assert scoring_block["requested_stage"] == "post_training_core"
    assert "post_training_core" in cast(list[str], scoring_block["refreshed_stages"])
    run_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    run_metrics = cast(dict[str, object], run_results["metrics"])
    assert "fnc" not in run_metrics
    assert "feature_exposure" not in run_metrics

    db_path = store_root / "numereng.db"
    assert db_path.is_file()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT run_id, status FROM runs WHERE run_id = ?",
            (response.run_id,),
        ).fetchone()
        lifecycle_row = conn.execute(
            "SELECT status, current_stage, terminal_reason FROM run_lifecycles WHERE run_id = ?",
            (response.run_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == response.run_id
    assert row[1] == "FINISHED"
    assert lifecycle_row is not None
    assert lifecycle_row[0] == "completed"
    assert lifecycle_row[1] == "finalize_manifest"
    assert lifecycle_row[2] == "completed"


@pytest.mark.integration
def test_run_training_defaults_post_training_scoring_to_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _build_synthetic_dataset(tmp_path, n_eras=6, rows_per_era=120, n_features=6, seed=19)
    store_root = tmp_path / ".numereng"
    _write_features_metadata(tmp_path, dataset.features)
    _stage_default_dataset_root(tmp_path, dataset)

    config_path = tmp_path / "train_default_none.json"
    _write_config(
        config_path,
        _build_training_config(
            dataset=dataset,
            store_root=store_root,
            parallel_folds=1,
        ),
    )

    monkeypatch.setattr(training_service_module, "create_training_data_client", lambda: _NoDownloadClient())
    monkeypatch.setattr(
        training_service_module,
        "load_features",
        lambda client, data_version, feature_set, dataset_variant="non_downsampled": dataset.features,
    )

    response = api_module.run_training(
        TrainRunRequest(
            config_path=str(config_path),
            output_dir=str(store_root),
        )
    )

    run_dir = store_root / "runs" / response.run_id
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    scoring_block = cast(dict[str, object], cast(dict[str, object], run_manifest["training"])["scoring"])
    assert scoring_block["policy"] == "none"
    assert scoring_block["status"] == "deferred"
    assert scoring_block["reason"] == "post_training_scoring_disabled"
    assert scoring_block["requested_stage"] is None
    assert cast(list[str], scoring_block["refreshed_stages"]) == []
    metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["status"] == "deferred"
    assert metrics_payload["reason"] == "post_training_scoring_disabled"
    assert not (run_dir / "score_provenance.json").exists()


@pytest.mark.integration
def test_materialized_scoring_core_vs_full_integration(tmp_path: Path) -> None:
    dataset = _build_synthetic_dataset(tmp_path, n_eras=6, rows_per_era=220, n_features=6, seed=11)
    _write_features_metadata(tmp_path, dataset.features)
    predictions_path = tmp_path / "predictions.parquet"

    train = pd.read_parquet(dataset.train_path)
    validation = pd.read_parquet(dataset.validation_path)
    validation_only = validation[validation["data_type"] == "validation"].drop(columns=["data_type"])
    full = pd.concat([train, validation_only], ignore_index=True)
    prediction_frame = full[["id", "era", "target"]].copy()
    prediction_frame["prediction"] = prediction_frame["target"] * 0.6 + 0.2
    prediction_frame.to_parquet(predictions_path, index=False)

    core_summaries, _ = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_NoDownloadClient(),
        feature_set="small",
        dataset_scope="train_plus_validation",
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=dataset.benchmark_path,
        meta_model_data_path=dataset.meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
        include_feature_neutral_metrics=False,
    )
    full_summaries, _ = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_NoDownloadClient(),
        feature_set="small",
        dataset_scope="train_plus_validation",
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=dataset.benchmark_path,
        meta_model_data_path=dataset.meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
    )

    assert "fnc" not in core_summaries
    assert "feature_exposure" not in core_summaries
    assert "fnc" in full_summaries
    assert "feature_exposure" not in full_summaries
    assert "max_feature_exposure" not in full_summaries


@pytest.mark.integration
def test_parallel_memmap_outputs_match_single_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _build_synthetic_dataset(tmp_path, n_eras=6, rows_per_era=320, n_features=10, seed=23)
    store_root = tmp_path / ".numereng"
    _write_features_metadata(tmp_path, dataset.features)
    _stage_default_dataset_root(tmp_path, dataset)

    monkeypatch.setattr(training_service_module, "create_training_data_client", lambda: _NoDownloadClient())
    monkeypatch.setattr(
        training_service_module,
        "load_features",
        lambda client, data_version, feature_set, dataset_variant="non_downsampled": dataset.features,
    )

    config_single_path = tmp_path / "train_single.json"
    config_parallel_path = tmp_path / "train_parallel.json"
    _write_config(
        config_single_path,
        _build_training_config(
            dataset=dataset,
            store_root=store_root,
            parallel_folds=1,
            post_training_scoring="core",
        ),
    )
    _write_config(
        config_parallel_path,
        _build_training_config(
            dataset=dataset,
            store_root=store_root,
            parallel_folds=2,
            post_training_scoring="core",
        ),
    )

    single = api_module.run_training(
        TrainRunRequest(
            config_path=str(config_single_path),
            output_dir=str(store_root),
        )
    )
    parallel = api_module.run_training(
        TrainRunRequest(
            config_path=str(config_parallel_path),
            output_dir=str(store_root),
        )
    )

    single_predictions = pd.read_parquet(single.predictions_path).sort_values(["id", "era"]).reset_index(drop=True)
    parallel_predictions = pd.read_parquet(parallel.predictions_path).sort_values(["id", "era"]).reset_index(drop=True)
    assert_frame_equal(single_predictions, parallel_predictions, check_exact=False, rtol=1e-12, atol=1e-12)

    single_results = json.loads(Path(single.results_path).read_text(encoding="utf-8"))
    parallel_results = json.loads(Path(parallel.results_path).read_text(encoding="utf-8"))
    single_metrics = cast(dict[str, dict[str, float]], single_results["metrics"])
    parallel_metrics = cast(dict[str, dict[str, float]], parallel_results["metrics"])
    for metric_key in ("corr", "bmc", "mmc", "cwmm"):
        assert single_metrics[metric_key]["mean"] == pytest.approx(parallel_metrics[metric_key]["mean"], abs=1e-12)
        assert single_metrics[metric_key]["std"] == pytest.approx(parallel_metrics[metric_key]["std"], abs=1e-12)
