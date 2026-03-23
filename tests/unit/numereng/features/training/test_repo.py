from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from numereng.features.training.errors import TrainingConfigError
from numereng.features.training.repo import (
    ensure_full_dataset,
    ensure_split_dataset_paths,
    ensure_train_dataset,
    list_lazy_source_eras,
    load_config,
    load_fold_data_lazy,
    load_full_data,
    resolve_data_path,
    resolve_derived_dataset_path,
    resolve_fold_lazy_source_paths,
    resolve_output_locations,
    resolve_score_provenance_path,
    save_predictions,
    save_score_provenance,
    select_prediction_columns,
)


class _NoDownloadClient:
    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        _ = (filename, dest_path, round_num)
        raise AssertionError("download_dataset should not be called")


def test_load_config_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config["model"] == {"type": "LGBMRegressor", "params": {}}


def test_load_config_rejects_non_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(TrainingConfigError, match="training_config_json_required"):
        load_config(config_path)


def test_load_config_requires_canonical_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"data": {}, "model": {"type": "LGBMRegressor", "params": {}}}), encoding="utf-8")

    with pytest.raises(TrainingConfigError, match="training_config_schema_invalid"):
        load_config(config_path)


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TrainingConfigError, match="training_config_file_not_found"):
        load_config(tmp_path / "missing.json")


def test_resolve_output_locations_default() -> None:
    output_dir, baselines_dir, results_dir, predictions_dir, scoring_dir = resolve_output_locations({}, None, "run-1")
    assert output_dir.name == "run-1"
    assert output_dir.parent.name == "runs"
    assert baselines_dir.name == "baselines"
    assert results_dir.name == "run-1"
    assert results_dir.parent.name == "runs"
    assert predictions_dir.name == "predictions"
    assert scoring_dir.name == "scoring"


def test_resolve_output_locations_rejects_nested_store_root() -> None:
    with pytest.raises(TrainingConfigError, match="training_output_store_root_noncanonical"):
        resolve_output_locations(
            {"output": {"output_dir": ".numereng/smoke_live_check"}},
            None,
            "run-1",
        )


def test_save_predictions_writes_zstd_parquet(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "id": ["id_1", "id_2"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
            "prediction": [0.3, 0.4],
        }
    )
    config = {"output": {"predictions_name": "predictions"}}
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "run-1"
    predictions_dir = output_dir / "artifacts" / "predictions"

    predictions_path, predictions_relative = save_predictions(
        predictions=predictions,
        config=config,
        config_path=config_path,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
    )

    assert predictions_path.is_file()
    assert predictions_relative == Path("artifacts/predictions/predictions.parquet")
    parquet_file = pq.ParquetFile(predictions_path)
    assert parquet_file.metadata.row_group(0).column(0).compression == "ZSTD"


def test_resolve_data_path_rooted_relative_path_not_double_prefixed(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    rooted_relative = Path(".numereng/datasets/v5.2/full.parquet")

    resolved = resolve_data_path(rooted_relative, data_root=data_root)
    assert resolved == (tmp_path / ".numereng" / "datasets" / "v5.2" / "full.parquet").resolve()


def test_resolve_data_path_plain_relative_path_is_prefixed(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    resolved = resolve_data_path("v5.2/full.parquet", data_root=data_root)
    assert resolved == (data_root / "v5.2" / "full.parquet").resolve()


def test_resolve_derived_dataset_path_defaults_outside_datasets(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    resolved = resolve_derived_dataset_path(data_root=data_root, data_version="v5.2", filename="full.parquet")
    assert resolved == (tmp_path / ".numereng" / "cache" / "derived_datasets" / "v5.2" / "full.parquet").resolve()


def test_ensure_full_dataset_writes_to_derived_cache(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        {
            "id": ["id_train_1", "id_train_2"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
        }
    ).set_index("id")
    train.to_parquet(version_dir / "train.parquet")

    validation = pd.DataFrame(
        {
            "id": ["id_val_keep", "id_val_drop"],
            "era": ["003", "004"],
            "target": [0.3, 0.4],
            "data_type": ["validation", "live"],
        }
    ).set_index("id")
    validation.to_parquet(version_dir / "validation.parquet")

    full_path = ensure_full_dataset(_NoDownloadClient(), "v5.2", data_root=data_root)

    assert full_path == (tmp_path / ".numereng" / "cache" / "derived_datasets" / "v5.2" / "full.parquet").resolve()
    assert not (version_dir / "full.parquet").exists()

    full = pd.read_parquet(full_path)
    assert list(full["era"]) == ["001", "002", "003"]


def test_ensure_full_dataset_downsampled_uses_downsampled_full_path(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    downsampled_full_path = version_dir / "downsampled_full.parquet"
    pd.DataFrame(
        {
            "id": ["id_a", "id_b"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
        }
    ).to_parquet(downsampled_full_path, index=False)

    resolved = ensure_full_dataset(
        _NoDownloadClient(),
        "v5.2",
        dataset_variant="downsampled",
        data_root=data_root,
    )
    assert resolved == downsampled_full_path.resolve()
    assert not (tmp_path / ".numereng" / "cache" / "derived_datasets" / "v5.2" / "downsampled_full.parquet").exists()


def test_ensure_train_dataset_returns_train_parquet_path(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    train_path = version_dir / "train.parquet"
    pd.DataFrame(
        {
            "id": ["id_train_1", "id_train_2"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
        }
    ).to_parquet(train_path, index=False)

    resolved = ensure_train_dataset(_NoDownloadClient(), "v5.2", data_root=data_root)
    assert resolved == train_path.resolve()


def test_ensure_split_dataset_paths_rejects_invalid_dataset_variant(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(TrainingConfigError, match="training_data_dataset_variant_invalid"):
        ensure_split_dataset_paths(
            _NoDownloadClient(),
            "v5.2",
            dataset_variant="quantized",
            data_root=data_root,
        )


def test_load_full_data_train_only_scope_excludes_validation_rows(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "id": ["id_train_1", "id_train_2"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
            "feature_1": [1.0, 2.0],
        }
    ).to_parquet(version_dir / "train.parquet", index=False)

    pd.DataFrame(
        {
            "id": ["id_val_1"],
            "era": ["003"],
            "target": [0.3],
            "feature_1": [3.0],
            "data_type": ["validation"],
        }
    ).to_parquet(version_dir / "validation.parquet", index=False)

    loaded = load_full_data(
        _NoDownloadClient(),
        "v5.2",
        "non_downsampled",
        ["feature_1"],
        "era",
        "target",
        "id",
        dataset_scope="train_only",
        data_root=data_root,
    )

    assert list(loaded["era"]) == ["001", "002"]


def test_load_full_data_downsampled_variant_defaults_to_downsampled_full(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["id_a", "id_b"],
            "era": ["101", "102"],
            "target": [0.7, 0.8],
            "feature_1": [1.0, 2.0],
        }
    ).to_parquet(version_dir / "downsampled_full.parquet", index=False)

    loaded = load_full_data(
        _NoDownloadClient(),
        "v5.2",
        "downsampled",
        ["feature_1"],
        "era",
        "target",
        "id",
        data_root=data_root,
    )

    assert list(loaded["era"]) == ["101", "102"]
    assert list(loaded.columns) == ["era", "target", "feature_1", "id"]


def test_load_full_data_train_plus_validation_scope_uses_split_sources_without_full_cache(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "id": ["id_train_1", "id_train_2"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
            "feature_1": [1.0, 2.0],
            "unused_col": [10, 20],
        }
    ).to_parquet(version_dir / "train.parquet", index=False)

    pd.DataFrame(
        {
            "id": ["id_val_keep", "id_val_drop"],
            "era": ["003", "004"],
            "target": [0.3, 0.4],
            "feature_1": [3.0, 4.0],
            "unused_col": [30, 40],
            "data_type": ["validation", "live"],
        }
    ).to_parquet(version_dir / "validation.parquet", index=False)

    loaded = load_full_data(
        _NoDownloadClient(),
        "v5.2",
        "non_downsampled",
        ["feature_1"],
        "era",
        "target",
        "id",
        dataset_scope="train_plus_validation",
        data_root=data_root,
    )

    assert list(loaded["era"]) == ["001", "002", "003"]
    assert list(loaded.columns) == ["era", "target", "feature_1", "id"]
    assert "data_type" not in loaded.columns
    assert "unused_col" not in loaded.columns

    derived_full_path = resolve_derived_dataset_path(
        data_root=data_root,
        data_version="v5.2",
        filename="full.parquet",
    )
    assert not derived_full_path.exists()


def test_load_full_data_train_plus_validation_scope_handles_validation_without_data_type(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "id": ["id_train_1"],
            "era": ["001"],
            "target": [0.1],
            "feature_1": [1.0],
        }
    ).to_parquet(version_dir / "train.parquet", index=False)

    pd.DataFrame(
        {
            "id": ["id_val_1"],
            "era": ["002"],
            "target": [0.2],
            "feature_1": [2.0],
        }
    ).to_parquet(version_dir / "validation.parquet", index=False)

    loaded = load_full_data(
        _NoDownloadClient(),
        "v5.2",
        "non_downsampled",
        ["feature_1"],
        "era",
        "target",
        "id",
        dataset_scope="train_plus_validation",
        data_root=data_root,
    )

    assert list(loaded["era"]) == ["001", "002"]
    assert list(loaded.columns) == ["era", "target", "feature_1", "id"]


def test_select_prediction_columns_filters_known_fields() -> None:
    predictions = pd.DataFrame(
        {
            "id": ["a"],
            "era": ["era1"],
            "target": [0.2],
            "prediction": [0.5],
            "cv_fold": [0],
            "extra": [1],
        }
    )

    selected = select_prediction_columns(predictions, "id", "era", "target")
    assert list(selected.columns) == ["id", "era", "target", "prediction", "cv_fold"]


def test_resolve_score_provenance_path_points_to_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / ".numereng" / "runs" / "abc123"
    path = resolve_score_provenance_path(run_dir)
    assert path == run_dir / "score_provenance.json"


def test_save_score_provenance_writes_json(tmp_path: Path) -> None:
    path = tmp_path / "run" / "score_provenance.json"
    payload: dict[str, object] = {"schema_version": "1", "joins": {"predictions_rows": 10}}
    save_score_provenance(payload, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == payload


def test_fold_lazy_scan_filters_validation_rows_and_projects_columns(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        {
            "id": ["id_train_1", "id_train_2"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
            "feature_1": [1.0, 2.0],
            "data_type": ["train", "train"],
            "extra_col": [10, 20],
        }
    )
    train.to_parquet(version_dir / "train.parquet", index=False)

    validation = pd.DataFrame(
        {
            "id": ["id_val_keep", "id_val_drop"],
            "era": ["003", "004"],
            "target": [0.3, 0.4],
            "feature_1": [3.0, 4.0],
            "data_type": ["validation", "live"],
            "extra_col": [30, 40],
        }
    )
    validation.to_parquet(version_dir / "validation.parquet", index=False)

    source_paths = resolve_fold_lazy_source_paths(
        _NoDownloadClient(),
        "v5.2",
        dataset_scope="train_plus_validation",
        data_root=data_root,
    )
    scanned = load_fold_data_lazy(
        source_paths,
        eras=["002", "003", "004"],
        columns=["era", "target", "feature_1", "id"],
        era_col="era",
        id_col="id",
    )

    assert list(scanned.columns) == ["era", "target", "feature_1", "id"]
    # era 004 is dropped because validation rows must satisfy data_type=validation.
    assert list(scanned["era"]) == ["002", "003"]
    assert list(scanned["id"]) == ["id_train_2", "id_val_keep"]


def test_fold_lazy_list_eras_uses_train_and_filtered_validation(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
            "data_type": ["train", "train"],
        }
    ).to_parquet(version_dir / "train.parquet", index=False)

    pd.DataFrame(
        {
            "id": ["c", "d"],
            "era": ["003", "004"],
            "target": [0.3, 0.4],
            "data_type": ["validation", "live"],
        }
    ).to_parquet(version_dir / "validation.parquet", index=False)

    source_paths = resolve_fold_lazy_source_paths(
        _NoDownloadClient(),
        "v5.2",
        dataset_scope="train_plus_validation",
        data_root=data_root,
    )
    eras = list_lazy_source_eras(source_paths, era_col="era")
    assert eras == ["001", "002", "003"]


def test_resolve_fold_lazy_source_paths_defaults_to_train_only(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "train.parquet").write_text("placeholder", encoding="utf-8")
    (version_dir / "validation.parquet").write_text("placeholder", encoding="utf-8")

    source_paths = resolve_fold_lazy_source_paths(_NoDownloadClient(), "v5.2", data_root=data_root)
    assert source_paths == ((version_dir / "train.parquet").resolve(),)


def test_resolve_fold_lazy_source_paths_downsampled_uses_single_full_source(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    downsampled_full_path = version_dir / "downsampled_full.parquet"
    downsampled_full_path.write_text("placeholder", encoding="utf-8")

    source_paths = resolve_fold_lazy_source_paths(
        _NoDownloadClient(),
        "v5.2",
        dataset_variant="downsampled",
        dataset_scope="train_plus_validation",
        data_root=data_root,
    )
    assert source_paths == (downsampled_full_path.resolve(),)
