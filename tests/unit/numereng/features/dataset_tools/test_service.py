from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from numereng.features.dataset_tools import (
    BuildDownsampledFullRequest,
    DatasetToolsValidationError,
    build_downsampled_full,
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


class _SeedDownloadClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        _ = round_num
        assert dest_path is not None
        path = Path(dest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.calls.append(filename)
        if filename.endswith("/train.parquet"):
            pd.DataFrame({"id": ["id_1"], "era": ["0001"], "target": [0.1]}).to_parquet(path, index=False)
        elif filename.endswith("/validation.parquet"):
            pd.DataFrame(
                {"id": ["id_2"], "era": ["0002"], "target": [0.2], "data_type": ["validation"]}
            ).set_index("id").to_parquet(path)
        elif filename.endswith("/train_benchmark_models.parquet"):
            pd.DataFrame({"id": ["id_1"], "v52_lgbm_ender20": [0.01]}).to_parquet(path, index=False)
        elif filename.endswith("/validation_benchmark_models.parquet"):
            pd.DataFrame({"id": ["id_2"], "v52_lgbm_ender20": [0.02]}).to_parquet(path, index=False)
        else:
            raise AssertionError(f"unexpected download filename: {filename}")
        return str(path)


def _seed_local_dataset(version_dir: Path) -> None:
    pd.DataFrame(
        {
            "id": ["train_1", "train_2"],
            "era": ["0001", "0002"],
            "target": [0.1, 0.2],
        }
    ).to_parquet(version_dir / "train.parquet", index=False)
    pd.DataFrame(
        {
            "id": ["val_keep", "val_drop"],
            "era": ["0003", "0004"],
            "target": [0.3, 0.4],
            "data_type": ["validation", "live"],
        }
    ).set_index("id").to_parquet(version_dir / "validation.parquet")
    pd.DataFrame({"id": ["train_1", "train_2"], "v52_lgbm_ender20": [0.01, 0.02]}).to_parquet(
        version_dir / "train_benchmark_models.parquet",
        index=False,
    )
    pd.DataFrame({"id": ["val_keep", "val_drop"], "v52_lgbm_ender20": [0.03, 0.04]}).to_parquet(
        version_dir / "validation_benchmark_models.parquet",
        index=False,
    )


def test_build_downsampled_full_builds_expected_artifacts(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    version_dir = datasets_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    _seed_local_dataset(version_dir)

    result = build_downsampled_full(
        BuildDownsampledFullRequest(data_dir=datasets_root, data_version="v5.2"),
        client=_NoDownloadClient(),
    )

    assert result.full_rows == 3
    assert result.total_eras == 3
    assert result.kept_eras == 1
    assert result.downsampled_rows == 1
    assert result.full_benchmark_rows == 3
    assert result.downsampled_full_benchmark_rows == 1
    assert result.downsampled_full_path is not None
    assert result.downsampled_full_benchmark_path is not None
    assert result.full_path.exists()
    assert result.full_benchmark_path.exists()
    assert result.downsampled_full_path.exists()
    assert result.downsampled_full_benchmark_path.exists()

    downsampled = pd.read_parquet(result.downsampled_full_path)
    assert list(downsampled["era"]) == ["0001"]


def test_build_downsampled_full_downloads_missing_sources(tmp_path: Path) -> None:
    client = _SeedDownloadClient()
    datasets_root = tmp_path / "datasets"

    result = build_downsampled_full(
        BuildDownsampledFullRequest(data_dir=datasets_root, data_version="v5.2"),
        client=client,
    )

    assert result.full_rows == 2
    assert sorted(client.calls) == [
        "v5.2/train.parquet",
        "v5.2/train_benchmark_models.parquet",
        "v5.2/validation.parquet",
        "v5.2/validation_benchmark_models.parquet",
    ]


def test_build_downsampled_full_rejects_missing_era_column(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    version_dir = datasets_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": ["train_1"], "target": [0.1]}).to_parquet(version_dir / "train.parquet", index=False)
    pd.DataFrame(
        {"id": ["val_1"], "target": [0.2], "data_type": ["validation"]}
    ).set_index("id").to_parquet(version_dir / "validation.parquet")
    pd.DataFrame({"id": ["train_1"], "v52_lgbm_ender20": [0.01]}).to_parquet(
        version_dir / "train_benchmark_models.parquet",
        index=False,
    )
    pd.DataFrame({"id": ["val_1"], "v52_lgbm_ender20": [0.02]}).to_parquet(
        version_dir / "validation_benchmark_models.parquet",
        index=False,
    )

    with pytest.raises(DatasetToolsValidationError, match="downsample_missing_era_column"):
        build_downsampled_full(
            BuildDownsampledFullRequest(data_dir=datasets_root, data_version="v5.2"),
            client=_NoDownloadClient(),
        )


def test_build_downsampled_full_rejects_invalid_step(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    version_dir = datasets_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    _seed_local_dataset(version_dir)

    with pytest.raises(DatasetToolsValidationError, match="downsample_step_invalid"):
        build_downsampled_full(
            BuildDownsampledFullRequest(
                data_dir=datasets_root,
                data_version="v5.2",
                downsample_eras_step=1,
            ),
            client=_NoDownloadClient(),
        )


def test_build_downsampled_full_supports_skip_downsample(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    version_dir = datasets_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    _seed_local_dataset(version_dir)

    result = build_downsampled_full(
        BuildDownsampledFullRequest(
            data_dir=datasets_root,
            data_version="v5.2",
            skip_downsample=True,
        ),
        client=_NoDownloadClient(),
    )

    assert result.downsample_built is False
    assert result.downsampled_full_path is None
    assert result.downsampled_full_benchmark_path is None
    assert result.downsampled_rows is None
    assert result.downsampled_full_benchmark_rows is None
    assert result.total_eras is None
    assert result.kept_eras is None
    assert result.full_path.exists()
    assert result.full_benchmark_path.exists()
    assert not (version_dir / "downsampled_full.parquet").exists()
    assert not (version_dir / "downsampled_full_benchmark_models.parquet").exists()


def test_build_downsampled_full_supports_custom_step_and_offset(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    version_dir = datasets_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    _seed_local_dataset(version_dir)

    result = build_downsampled_full(
        BuildDownsampledFullRequest(
            data_dir=datasets_root,
            data_version="v5.2",
            downsample_eras_step=2,
            downsample_eras_offset=1,
        ),
        client=_NoDownloadClient(),
    )

    assert result.downsample_built is True
    assert result.downsampled_full_path is not None
    downsampled = pd.read_parquet(result.downsampled_full_path)
    assert list(downsampled["era"]) == ["0002"]
    assert result.downsample_step == 2
    assert result.downsample_offset == 1
    assert result.kept_eras == 1
    assert result.total_eras == 3


def test_build_downsampled_full_rejects_invalid_offset(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    version_dir = datasets_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    _seed_local_dataset(version_dir)

    with pytest.raises(DatasetToolsValidationError, match="downsample_offset_invalid"):
        build_downsampled_full(
            BuildDownsampledFullRequest(
                data_dir=datasets_root,
                data_version="v5.2",
                downsample_eras_step=4,
                downsample_eras_offset=4,
            ),
            client=_NoDownloadClient(),
        )
