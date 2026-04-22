from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.features.cloud.aws import sagemaker_entrypoint


class _FakeS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Path]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("payload", encoding="utf-8")
        self.calls.append((bucket, key, destination))


class _FakeBoto3:
    def __init__(self, client: _FakeS3Client) -> None:
        self._client = client

    def client(self, service_name: str) -> _FakeS3Client:
        assert service_name == "s3"
        return self._client


def test_parse_s3_uri_roundtrip() -> None:
    bucket, key = sagemaker_entrypoint.parse_s3_uri("s3://example-bucket/runs/run-1/config/config.json")
    assert bucket == "example-bucket"
    assert key == "runs/run-1/config/config.json"


def test_parse_s3_uri_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="invalid_s3_uri"):
        sagemaker_entrypoint.parse_s3_uri("https://example.com/config.json")


def test_resolve_config_path_from_channel(tmp_path: Path) -> None:
    channel_dir = tmp_path / "training"
    nested = channel_dir / "nested"
    nested.mkdir(parents=True)
    (nested / "ignore.txt").write_text("x", encoding="utf-8")
    expected = nested / "config.json"
    expected.write_text("{}", encoding="utf-8")

    resolved = sagemaker_entrypoint.resolve_config_path(env={}, channel_dir=channel_dir)
    assert resolved == expected


def test_required_dataset_keys_for_official_config() -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "benchmark_source": {"source": "active"},
            "meta_model_data_path": "v5.2/meta_model.parquet",
        }
    }

    keys = sagemaker_entrypoint.required_dataset_keys(config)
    assert keys == [
        "data/v5.2/features.json",
        "data/v5.2/train.parquet",
        "data/v5.2/validation.parquet",
        "baselines/active_benchmark/benchmark.json",
        "baselines/active_benchmark/predictions.parquet",
        "data/v5.2/meta_model.parquet",
    ]


def test_required_dataset_keys_rejects_quantized_variant() -> None:
    config = {"data": {"data_version": "v5.2", "dataset_variant": "quantized"}}

    with pytest.raises(ValueError, match="dataset_variant_invalid"):
        sagemaker_entrypoint.required_dataset_keys(config)


def test_required_dataset_keys_accepts_downsampled_paths() -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "downsampled",
            "benchmark_source": {
                "source": "path",
                "predictions_path": "v5.2/downsampled_full_benchmark_models.parquet",
            },
        }
    }

    keys = sagemaker_entrypoint.required_dataset_keys(config)
    assert keys == [
        "data/v5.2/features.json",
        "data/v5.2/downsampled_full.parquet",
        "data/v5.2/downsampled_full_benchmark_models.parquet",
    ]


def test_required_dataset_keys_defaults_to_downsampled_full_for_downsampled_variant() -> None:
    config = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "downsampled",
        }
    }

    keys = sagemaker_entrypoint.required_dataset_keys(config)
    assert keys == [
        "data/v5.2/features.json",
        "data/v5.2/downsampled_full.parquet",
        "baselines/active_benchmark/benchmark.json",
        "baselines/active_benchmark/predictions.parquet",
    ]


def test_stage_required_data_downloads_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "benchmark_source": {"source": "active"},
                }
            }
        ),
        encoding="utf-8",
    )
    fake_s3 = _FakeS3Client()
    fake_boto3 = _FakeBoto3(fake_s3)
    monkeypatch.setattr(sagemaker_entrypoint, "boto3", fake_boto3)

    data_root = tmp_path / "datasets"
    sagemaker_entrypoint.stage_required_data(
        config_path,
        env={"NUMERENG_CONFIG_S3_URI": "s3://example-bucket/runs/run-1/config/config.json"},
        data_root=data_root,
    )

    downloaded_keys = [key for _bucket, key, _destination in fake_s3.calls]
    assert downloaded_keys == [
        "data/v5.2/features.json",
        "data/v5.2/train.parquet",
        "data/v5.2/validation.parquet",
        "baselines/active_benchmark/benchmark.json",
        "baselines/active_benchmark/predictions.parquet",
    ]


def test_stage_required_data_rejects_quantized_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "quantized",
                }
            }
        ),
        encoding="utf-8",
    )
    fake_s3 = _FakeS3Client()
    fake_boto3 = _FakeBoto3(fake_s3)
    monkeypatch.setattr(sagemaker_entrypoint, "boto3", fake_boto3)

    with pytest.raises(ValueError, match="dataset_variant_invalid"):
        sagemaker_entrypoint.stage_required_data(
            config_path,
            env={"NUMERENG_CONFIG_S3_URI": "s3://example-bucket/runs/run-1/config/config.json"},
            data_root=tmp_path / "datasets",
        )

    assert fake_s3.calls == []


def test_sanitize_output_dir_removes_store_db_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    db_file = output_dir / "numereng.db"
    db_wal = output_dir / "numereng.db-wal"
    metrics_file = output_dir / "metrics.json"
    db_file.write_text("db", encoding="utf-8")
    db_wal.write_text("wal", encoding="utf-8")
    metrics_file.write_text("{}", encoding="utf-8")

    removed = sagemaker_entrypoint.sanitize_output_dir(output_dir)

    assert sorted(path.name for path in removed) == ["numereng.db", "numereng.db-wal"]
    assert not db_file.exists()
    assert not db_wal.exists()
    assert metrics_file.exists()


def test_main_sanitizes_output_dir_after_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "numereng.db").write_text("db", encoding="utf-8")
    (output_dir / "results.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("NUMERENG_CONFIG_S3_URI", "s3://example-bucket/runs/run-1/config/config.json")
    monkeypatch.setenv("NUMERENG_OUTPUT_S3_URI", "s3://example-bucket/runs/run-1/managed-output/")
    monkeypatch.setenv("NUMERENG_TRAIN_OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(sagemaker_entrypoint, "resolve_config_path", lambda env=None, channel_dir=None: config_path)
    monkeypatch.setattr(sagemaker_entrypoint, "stage_required_data", lambda config_path, env=None, data_root=None: None)

    class _FakeCompletedProcess:
        returncode = 0

    monkeypatch.setattr(
        "numereng.features.cloud.aws.sagemaker_entrypoint.subprocess.run",
        lambda command, check=False: _FakeCompletedProcess(),
    )

    exit_code = sagemaker_entrypoint.main()

    assert exit_code == 0
    assert not (output_dir / "numereng.db").exists()
    assert (output_dir / "results.json").exists()
