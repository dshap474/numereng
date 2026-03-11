from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from numereng.config.training import (
    TrainingConfigLoaderError,
    ensure_json_config_path,
    ensure_json_config_uri,
    load_training_config_json,
)


def _write_valid_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )


def test_load_training_config_json_accepts_canonical_payload(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    _write_valid_config(config_path)

    payload = load_training_config_json(config_path)
    assert payload["model"] == {"type": "LGBMRegressor", "params": {}}


def test_load_training_config_json_accepts_explicit_model_device(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "device": "cuda", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    payload = load_training_config_json(config_path)
    model_block = cast(dict[str, object], payload["model"])
    assert model_block["device"] == "cuda"


def test_load_training_config_json_rejects_invalid_model_device(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "device": "metal", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_model_device_invalid"):
        load_training_config_json(config_path)


def test_load_training_config_json_accepts_full_history_refit_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {"engine": {"profile": "full_history_refit"}},
            }
        ),
        encoding="utf-8",
    )

    payload = load_training_config_json(config_path)
    training_block = cast(dict[str, object], payload["training"])
    engine_block = cast(dict[str, object], training_block["engine"])
    assert engine_block["profile"] == "full_history_refit"


def test_load_training_config_json_rejects_submission_profile_rename(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {"engine": {"profile": "submission"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        TrainingConfigLoaderError,
        match="training profile 'submission' was renamed to 'full_history_refit'",
    ):
        load_training_config_json(config_path)


def test_load_training_config_json_accepts_downsampled_dataset_variant(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    payload = load_training_config_json(config_path)
    data = cast(dict[str, object], payload["data"])
    assert data["dataset_variant"] == "downsampled"


def test_load_training_config_json_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled", "unknown": "x"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)


@pytest.mark.parametrize("field_name", ["prediction_transform", "era_weighting", "prediction_batch_size"])
def test_load_training_config_json_rejects_removed_model_fields(tmp_path: Path, field_name: str) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}, field_name: "legacy"},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)


def test_load_training_config_json_rejects_max_train_samples(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {"max_train_samples": 100000},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)


def test_load_training_config_json_rejects_sample_seed(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {"sample_seed": 42},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)


def test_ensure_json_config_path_and_uri_validation() -> None:
    assert ensure_json_config_path("configs/train.json") == "configs/train.json"
    assert ensure_json_config_uri("s3://bucket/path/train.json") == "s3://bucket/path/train.json"

    with pytest.raises(ValueError, match="config_path must reference a .json file"):
        ensure_json_config_path("configs/train.yaml")

    with pytest.raises(ValueError, match="config_s3_uri must reference a .json object"):
        ensure_json_config_uri("s3://bucket/path/train.yaml")


def test_load_training_config_json_accepts_loading_resources_cache_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "loading": {
                        "mode": "fold_lazy",
                        "scoring_mode": "era_stream",
                        "era_chunk_size": 32,
                    },
                },
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {
                    "resources": {
                        "parallel_folds": 2,
                        "parallel_backend": "joblib",
                        "memmap_enabled": True,
                        "max_threads_per_worker": "default",
                    },
                    "cache": {
                        "mode": "deterministic",
                        "cache_fold_specs": True,
                        "cache_features": True,
                        "cache_labels": False,
                        "cache_fold_matrices": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = load_training_config_json(config_path)
    data_block = cast(dict[str, object], payload["data"])
    data_loading = cast(dict[str, object], data_block["loading"])
    assert data_loading == {
        "mode": "fold_lazy",
        "scoring_mode": "era_stream",
        "era_chunk_size": 32,
        "include_feature_neutral_metrics": True,
    }
    training_block = cast(dict[str, object], payload["training"])
    training_resources = cast(dict[str, object], training_block["resources"])
    assert training_resources["parallel_folds"] == 2
    assert training_resources["parallel_backend"] == "joblib"
    assert training_resources["max_threads_per_worker"] == "default"
    training_cache = cast(dict[str, object], training_block["cache"])
    assert training_cache["mode"] == "deterministic"
    assert training_cache["cache_labels"] is False


def test_load_training_config_json_defaults_dataset_scope_to_train_only(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    _write_valid_config(config_path)

    payload = load_training_config_json(config_path)
    data_block = cast(dict[str, object], payload["data"])
    assert data_block["dataset_scope"] == "train_only"


def test_load_training_config_json_requires_dataset_variant(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)


def test_load_training_config_json_accepts_train_plus_validation_dataset_scope(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "dataset_scope": "train_plus_validation",
                },
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    payload = load_training_config_json(config_path)
    data_block = cast(dict[str, object], payload["data"])
    assert data_block["dataset_scope"] == "train_plus_validation"


def test_load_training_config_json_rejects_quantized_dataset_variant(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "quantized"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)


def test_load_training_config_json_accepts_target_horizon_enum(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "target_horizon": "60d",
                },
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    payload = load_training_config_json(config_path)
    data_block = cast(dict[str, object], payload["data"])
    assert data_block["target_horizon"] == "60d"


def test_load_training_config_json_rejects_invalid_target_horizon(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "target_horizon": "30d",
                },
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)


def test_load_training_config_json_rejects_invalid_dataset_scope(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "dataset_scope": "bad_scope",
                },
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)


def test_load_training_config_json_rejects_invalid_loading_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "loading": {"mode": "bad_mode"},
                },
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigLoaderError, match="training_config_schema_invalid"):
        load_training_config_json(config_path)
