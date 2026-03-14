from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.features.cloud.modal.data_sync import resolve_required_data_files


def _write_file(path: Path, content: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_resolve_required_data_files_defaults_to_train_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / ".numereng" / "datasets" / "v5.2"
    _write_file(data_dir / "train.parquet")
    _write_file(data_dir / "validation.parquet")
    (data_dir / "features.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    _, _, files = resolve_required_data_files(config_path=config_path)

    assert [item.remote_path for item in files] == [
        "v5.2/features.json",
        "v5.2/train.parquet",
        "v5.2/validation.parquet",
    ]


def test_resolve_required_data_files_rejects_quantized_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
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

    with pytest.raises(ValueError, match="training_config_schema_invalid"):
        resolve_required_data_files(config_path=config_path)

def test_resolve_required_data_files_requires_files_to_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                },
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="modal_data_sync_file_not_found"):
        resolve_required_data_files(config_path=config_path)
