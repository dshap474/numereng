from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
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
    full_path: Path
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

    full_path = root / "full.parquet"
    train_path = root / "train.parquet"
    validation_path = root / "validation.parquet"
    benchmark_path = root / "benchmark.parquet"
    meta_path = root / "meta.parquet"

    full.to_parquet(full_path, index=False)
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
        full_path=full_path,
        train_path=train_path,
        validation_path=validation_path,
        benchmark_path=benchmark_path,
        meta_path=meta_path,
    )


def _build_training_config(
    *,
    dataset: _SyntheticDataset,
    store_root: Path,
    loading_mode: str,
    scoring_mode: str,
    era_chunk_size: int,
    parallel_folds: int,
    use_full_data_path: bool,
) -> dict[str, object]:
    data_block: dict[str, object] = {
        "data_version": "v5.2",
        "dataset_variant": "non_downsampled",
        "feature_set": "small",
        "target_col": "target",
        "scoring_targets": ["target"],
        "era_col": "era",
        "id_col": "id",
        "benchmark_data_path": str(dataset.benchmark_path),
        "meta_model_data_path": str(dataset.meta_path),
        "loading": {
            "mode": loading_mode,
            "scoring_mode": scoring_mode,
            "era_chunk_size": era_chunk_size,
        },
    }
    if use_full_data_path:
        data_block["full_data_path"] = str(dataset.full_path)

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
def test_run_training_fold_lazy_smoke_indexes_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _build_synthetic_dataset(tmp_path, n_eras=6, rows_per_era=180, n_features=8, seed=7)
    store_root = tmp_path / ".numereng"
    _write_features_metadata(tmp_path, dataset.features)

    config_path = tmp_path / "train_fold_lazy.json"
    config_payload = _build_training_config(
        dataset=dataset,
        store_root=store_root,
        loading_mode="fold_lazy",
        scoring_mode="materialized",
        era_chunk_size=8,
        parallel_folds=1,
        use_full_data_path=False,
    )
    _write_config(config_path, config_payload)

    monkeypatch.setattr(training_service_module, "create_training_data_client", lambda: _NoDownloadClient())
    monkeypatch.setattr(training_service_module, "DEFAULT_DATASETS_DIR", tmp_path)
    monkeypatch.setattr(
        training_service_module,
        "load_features",
        lambda client, data_version, feature_set, dataset_variant="non_downsampled": dataset.features,
    )

    def _resolve_fold_paths(
        client: object,
        data_version: str,
        full_data_path: Path | None = None,
        dataset_scope: str = "train_only",
        data_root: Path | None = None,
        dataset_variant: str = "non_downsampled",
    ) -> tuple[Path, ...]:
        del client, data_version, full_data_path, data_root, dataset_variant
        if dataset_scope == "train_plus_validation":
            return (dataset.train_path, dataset.validation_path)
        return (dataset.train_path,)

    monkeypatch.setattr(
        training_service_module,
        "resolve_fold_lazy_source_paths",
        _resolve_fold_paths,
    )

    response = api_module.run_training(
        TrainRunRequest(
            config_path=str(config_path),
            output_dir=str(store_root),
        )
    )

    run_dir = store_root / "runs" / response.run_id
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "resolved.json").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "score_provenance.json").is_file()
    assert (run_dir / "artifacts" / "predictions").is_dir()

    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_manifest["status"] == "FINISHED"
    training_block = cast(dict[str, object], run_manifest["training"])
    assert cast(dict[str, object], training_block["loading"])["mode"] == "fold_lazy"
    scoring_block = cast(dict[str, object], training_block["scoring"])
    assert scoring_block["mode"] == "materialized"
    run_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    run_metrics = cast(dict[str, object], run_results["metrics"])
    assert "fnc" in run_metrics

    db_path = store_root / "numereng.db"
    assert db_path.is_file()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT run_id, status FROM runs WHERE run_id = ?",
            (response.run_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == response.run_id
    assert row[1] == "FINISHED"


@pytest.mark.integration
def test_scoring_mode_era_stream_parity_integration(tmp_path: Path) -> None:
    dataset = _build_synthetic_dataset(tmp_path, n_eras=6, rows_per_era=220, n_features=6, seed=11)
    _write_features_metadata(tmp_path, dataset.features)
    predictions_path = tmp_path / "predictions.parquet"

    full = pd.read_parquet(dataset.full_path)
    prediction_frame = full[["id", "era", "target"]].copy()
    prediction_frame["prediction"] = prediction_frame["target"] * 0.6 + 0.2
    prediction_frame.to_parquet(predictions_path, index=False)

    materialized_summaries, _ = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_NoDownloadClient(),
        feature_set="small",
        full_data_path=dataset.full_path,
        dataset_scope="train_plus_validation",
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=dataset.benchmark_path,
        meta_model_data_path=dataset.meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
        scoring_mode="materialized",
        era_chunk_size=4,
    )
    stream_summaries, _ = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_NoDownloadClient(),
        feature_set="small",
        full_data_path=dataset.full_path,
        dataset_scope="train_plus_validation",
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=dataset.benchmark_path,
        meta_model_data_path=dataset.meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
        scoring_mode="era_stream",
        era_chunk_size=2,
    )

    assert set(materialized_summaries) == set(stream_summaries)
    for metric_name in materialized_summaries:
        assert_frame_equal(
            materialized_summaries[metric_name].sort_index(axis=0).sort_index(axis=1),
            stream_summaries[metric_name].sort_index(axis=0).sort_index(axis=1),
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )


@pytest.mark.integration
def test_parallel_memmap_outputs_match_single_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _build_synthetic_dataset(tmp_path, n_eras=6, rows_per_era=320, n_features=10, seed=23)
    store_root = tmp_path / ".numereng"
    _write_features_metadata(tmp_path, dataset.features)

    monkeypatch.setattr(training_service_module, "create_training_data_client", lambda: _NoDownloadClient())
    monkeypatch.setattr(training_service_module, "DEFAULT_DATASETS_DIR", tmp_path)
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
            loading_mode="materialized",
            scoring_mode="materialized",
            era_chunk_size=8,
            parallel_folds=1,
            use_full_data_path=True,
        ),
    )
    _write_config(
        config_parallel_path,
        _build_training_config(
            dataset=dataset,
            store_root=store_root,
            loading_mode="materialized",
            scoring_mode="materialized",
            era_chunk_size=8,
            parallel_folds=2,
            use_full_data_path=True,
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
    for metric_key in ("corr", "fnc", "bmc", "mmc", "cwmm"):
        assert single_metrics[metric_key]["mean"] == pytest.approx(parallel_metrics[metric_key]["mean"], abs=1e-12)
        assert single_metrics[metric_key]["std"] == pytest.approx(parallel_metrics[metric_key]["std"], abs=1e-12)


def _run_training_subprocess(
    *,
    workdir: Path,
    config_path: Path,
    output_dir: Path,
    repo_src: Path,
) -> dict[str, object]:
    script = """
import json
import resource
import time
from pathlib import Path

import numereng.api as api_module
from numereng.api.contracts import TrainRunRequest

config_path = Path(__import__("sys").argv[1])
output_dir = Path(__import__("sys").argv[2])
start = time.perf_counter()
response = api_module.run_training(
    TrainRunRequest(config_path=str(config_path), output_dir=str(output_dir))
)
elapsed = time.perf_counter() - start
print(json.dumps({
    "run_id": response.run_id,
    "elapsed_seconds": elapsed,
    "peak_rss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_src)
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(output_dir)],
        cwd=str(workdir),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    return cast(dict[str, object], payload)


@pytest.mark.integration
@pytest.mark.slow
def test_fold_lazy_slow_peak_rss_regression_gate(tmp_path: Path) -> None:
    # Run in an isolated cwd so default `.numereng/datasets` resolution is local.
    workdir = tmp_path / "perf_env"
    workdir.mkdir(parents=True, exist_ok=True)
    dataset = _build_synthetic_dataset(workdir, n_eras=6, rows_per_era=8000, n_features=60, seed=101)
    store_root = workdir / ".numereng_store"
    store_root.mkdir(parents=True, exist_ok=True)

    dataset_root = workdir / ".numereng" / "datasets" / "v5.2"
    dataset_root.mkdir(parents=True, exist_ok=True)
    features_payload = {
        "feature_sets": {
            "small": dataset.features,
            "all": dataset.features,
            "fncv3_features": dataset.features,
        }
    }
    (dataset_root / "features.json").write_text(json.dumps(features_payload), encoding="utf-8")
    pd.read_parquet(dataset.train_path).to_parquet(dataset_root / "train.parquet", index=False)
    pd.read_parquet(dataset.validation_path).to_parquet(dataset_root / "validation.parquet", index=False)

    materialized_path = workdir / "materialized.json"
    fold_lazy_path = workdir / "fold_lazy.json"
    _write_config(
        materialized_path,
        _build_training_config(
            dataset=dataset,
            store_root=store_root,
            loading_mode="materialized",
            scoring_mode="materialized",
            era_chunk_size=64,
            parallel_folds=1,
            use_full_data_path=False,
        ),
    )
    _write_config(
        fold_lazy_path,
        _build_training_config(
            dataset=dataset,
            store_root=store_root,
            loading_mode="fold_lazy",
            scoring_mode="materialized",
            era_chunk_size=64,
            parallel_folds=1,
            use_full_data_path=False,
        ),
    )

    repo_src = Path(__file__).resolve().parents[2] / "src"
    materialized = _run_training_subprocess(
        workdir=workdir,
        config_path=materialized_path,
        output_dir=store_root,
        repo_src=repo_src,
    )
    fold_lazy = _run_training_subprocess(
        workdir=workdir,
        config_path=fold_lazy_path,
        output_dir=store_root,
        repo_src=repo_src,
    )

    materialized_peak = cast(int, materialized["peak_rss"])
    fold_lazy_peak = cast(int, fold_lazy["peak_rss"])
    assert fold_lazy_peak <= int(materialized_peak * 1.1)
