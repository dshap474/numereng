from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from numereng.features.training.models import (
    FoldJoinColumn,
    build_lazy_parquet_data_loader,
    build_model_data_loader,
    build_x_cols,
    normalize_x_groups,
)


def test_normalize_x_groups_defaults_added() -> None:
    assert normalize_x_groups(["features"]) == ["features"]


def test_normalize_x_groups_rejects_identifier_groups() -> None:
    with pytest.raises(ValueError, match="training_model_x_groups_not_supported:era"):
        normalize_x_groups(["features", "era"])
    with pytest.raises(ValueError, match="training_model_x_groups_not_supported:id"):
        normalize_x_groups(["features", "id"])


def test_normalize_x_groups_rejects_benchmark_models_aliases() -> None:
    with pytest.raises(ValueError, match="training_model_x_groups_benchmark_not_supported"):
        normalize_x_groups(["features", "benchmark_models"])
    with pytest.raises(ValueError, match="training_model_x_groups_benchmark_not_supported"):
        normalize_x_groups(["features", "benchmark"])
    with pytest.raises(ValueError, match="training_model_x_groups_benchmark_not_supported"):
        normalize_x_groups(["features", "benchmarks"])


def test_normalize_x_groups_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown x_group"):
        normalize_x_groups(["unknown"])


def test_build_x_cols_with_features_only() -> None:
    cols = build_x_cols(
        x_groups=["features"],
        features=["feature_1", "feature_2"],
        era_col="era",
        id_col="id",
    )
    assert cols == ["feature_1", "feature_2"]


def test_build_model_data_loader_loads_subset() -> None:
    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.1, 0.2, 0.3, 0.4],
            "feature_1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    batch = loader.load(["era2"])
    assert batch.X.shape == (2, 1)
    assert batch.y.tolist() == [0.3, 0.4]
    assert batch.id is not None
    assert batch.id.tolist() == ["c", "d"]


def test_build_model_data_loader_loads_subset_when_id_is_index() -> None:
    full = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.1, 0.2, 0.3, 0.4],
            "feature_1": [1.0, 2.0, 3.0, 4.0],
        }
    ).set_index("id")
    full.index.name = "id"

    loader = build_model_data_loader(
        full=full,
        x_cols=["feature_1"],
        era_col="era",
        target_col="target",
        id_col="id",
    )

    batch = loader.load(["era2"])
    assert batch.X.shape == (2, 1)
    assert batch.y.tolist() == [0.3, 0.4]
    assert batch.id is not None
    assert batch.id.tolist() == ["c", "d"]


def test_build_lazy_parquet_data_loader_loads_and_joins_sources(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"

    pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["001", "002"],
            "target": [0.1, 0.2],
            "feature_1": [1.0, 2.0],
            "data_type": ["train", "train"],
        }
    ).to_parquet(train_path, index=False)
    pd.DataFrame(
        {
            "id": ["c", "d"],
            "era": ["003", "004"],
            "target": [0.3, 0.4],
            "feature_1": [3.0, 4.0],
            "data_type": ["validation", "live"],
        }
    ).to_parquet(validation_path, index=False)

    benchmark = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["001", "002", "003", "004"],
            "benchmark_pred": [0.5, 0.6, 0.7, 0.8],
        }
    ).set_index("id")

    loader = build_lazy_parquet_data_loader(
        source_paths=[train_path, validation_path],
        x_cols=["feature_1", "benchmark_pred"],
        era_col="era",
        target_col="target",
        id_col="id",
        join_columns=[FoldJoinColumn(frame=benchmark, column="benchmark_pred")],
    )

    batch = loader.load(["002", "003", "004"])
    assert batch.id is not None
    assert batch.id.tolist() == ["b", "c"]
    assert batch.X["benchmark_pred"].tolist() == [0.6, 0.7]
    assert loader.list_eras() == ["001", "002", "003"]
