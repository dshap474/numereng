from __future__ import annotations

from typing import cast

from numereng.features.training.run_store import compute_run_hash


def test_compute_run_hash_changes_when_output_baselines_dir_changes() -> None:
    base_config = {
        "data": {
            "data_version": "v5.2",
            "feature_set": "small",
            "target_col": "target",
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
            "x_groups": ["features", "baseline"],
            "baseline": {
                "name": "base_model",
                "predictions_path": "baseline.parquet",
            },
        },
        "output": {
            "baselines_dir": ".numereng/baselines_a",
        },
    }
    alt_config = {
        **base_config,
        "output": {
            "baselines_dir": ".numereng/baselines_b",
        },
    }

    engine_settings: dict[str, object] = {
        "mode": "official",
        "window_size_eras": 156,
        "embargo_eras": 8,
        "target_horizon": "20d",
    }

    base_hash = compute_run_hash(
        config=base_config,
        data_version="v5.2",
        feature_set="small",
        target_col="target",
        model_type="LGBMRegressor",
        engine_mode="official",
        engine_settings=engine_settings,
    )
    alt_hash = compute_run_hash(
        config=alt_config,
        data_version="v5.2",
        feature_set="small",
        target_col="target",
        model_type="LGBMRegressor",
        engine_mode="official",
        engine_settings=engine_settings,
    )
    assert base_hash != alt_hash


def test_compute_run_hash_changes_when_loading_mode_changes() -> None:
    base_config = {
        "data": {
            "data_version": "v5.2",
            "feature_set": "small",
            "target_col": "target",
            "loading": {
                "mode": "materialized",
                "scoring_mode": "materialized",
                "era_chunk_size": 64,
            },
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {"n_estimators": 10},
        },
        "training": {
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
    alt_data = dict(cast(dict[str, object], base_config["data"]))
    alt_data["loading"] = {
        "mode": "fold_lazy",
        "scoring_mode": "materialized",
        "era_chunk_size": 64,
    }
    alt_config: dict[str, object] = {**base_config, "data": alt_data}

    engine_settings: dict[str, object] = {
        "mode": "official",
        "window_size_eras": 156,
        "embargo_eras": 8,
        "target_horizon": "20d",
    }

    base_hash = compute_run_hash(
        config=base_config,
        data_version="v5.2",
        feature_set="small",
        target_col="target",
        model_type="LGBMRegressor",
        engine_mode="official",
        engine_settings=engine_settings,
    )
    alt_hash = compute_run_hash(
        config=alt_config,
        data_version="v5.2",
        feature_set="small",
        target_col="target",
        model_type="LGBMRegressor",
        engine_mode="official",
        engine_settings=engine_settings,
    )
    assert base_hash != alt_hash
