"""Container entrypoint for SageMaker-managed numereng training jobs."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import boto3

_TRAINING_CHANNEL_DIR = Path("/opt/ml/input/data/training")
_DEFAULT_OUTPUT_DIR = Path("/opt/ml/output/data")
_DEFAULT_CONFIG_DOWNLOAD_DIR = Path("/tmp/numereng/config")
_DEFAULT_DATA_ROOT = Path(".numereng") / "datasets"
_CONFIG_EXTENSIONS = {".json"}
_SUPPORTED_DATASET_VARIANTS = {"non_downsampled", "downsampled"}
_DOWNSAMPLED_VARIANT_FILENAME_MAP: dict[str, str] = {
    "full.parquet": "downsampled_full.parquet",
    "full_benchmark_models.parquet": "downsampled_full_benchmark_models.parquet",
}
_OFFICIAL_DATASET_FILENAMES = {
    "downsampled_full.parquet",
    "downsampled_full_benchmark_models.parquet",
    "features.json",
    "full.parquet",
    "full_benchmark_models.parquet",
    "live.parquet",
    "live_benchmark_models.parquet",
    "live_example_preds.csv",
    "live_example_preds.parquet",
    "meta_model.parquet",
    "train.parquet",
    "train_benchmark_models.parquet",
    "validation.parquet",
    "validation_benchmark_models.parquet",
    "validation_example_preds.csv",
    "validation_example_preds.parquet",
}
_STORE_DB_FILENAMES = ("numereng.db", "numereng.db-shm", "numereng.db-wal")
_ACTIVE_BENCHMARK_KEYS = {
    "baselines/active_benchmark/predictions.parquet",
    "baselines/active_benchmark/benchmark.json",
}


def _log(message: str) -> None:
    print(f"[numereng-sagemaker] {message}", flush=True)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse S3 URI into (bucket, key)."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key:
        raise ValueError(f"invalid_s3_uri:{uri}")
    return bucket, key


def infer_bucket(*, config_s3_uri: str | None, output_s3_uri: str | None) -> str | None:
    """Infer data bucket from cloud env vars."""
    for candidate in (config_s3_uri, output_s3_uri):
        if candidate is None or not candidate:
            continue
        try:
            bucket, _ = parse_s3_uri(candidate)
            return bucket
        except ValueError:
            continue
    return None


def _candidate_config_paths(channel_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in channel_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _CONFIG_EXTENSIONS
    ]
    candidates.sort(key=lambda value: str(value))
    return candidates


def _download_config_from_s3(config_s3_uri: str, destination_dir: Path) -> Path:
    bucket, key = parse_s3_uri(config_s3_uri)
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(key).name or "training-config.json"
    destination = destination_dir / filename
    boto3.client("s3").download_file(bucket, key, str(destination))
    return destination


def resolve_config_path(
    *,
    env: Mapping[str, str] | None = None,
    channel_dir: Path = _TRAINING_CHANNEL_DIR,
) -> Path:
    """Resolve one config path from explicit env, SageMaker channel, or S3 URI."""
    resolved_env = os.environ if env is None else env

    explicit = resolved_env.get("NUMERENG_CONFIG_PATH")
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.is_file():
            return explicit_path
        raise FileNotFoundError(f"config_path_not_found:{explicit_path}")

    if channel_dir.is_dir():
        candidates = _candidate_config_paths(channel_dir)
        if candidates:
            return candidates[0]

    config_s3_uri = resolved_env.get("NUMERENG_CONFIG_S3_URI")
    if config_s3_uri:
        return _download_config_from_s3(config_s3_uri, _DEFAULT_CONFIG_DOWNLOAD_DIR)

    raise FileNotFoundError("training_config_not_found")


def _load_config_payload(config_path: Path) -> dict[str, object]:
    if config_path.suffix.lower() != ".json":
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _to_data_key(path_value: str) -> str | None:
    if "://" in path_value:
        return None
    normalized = path_value.lstrip("/")
    if normalized.startswith("data/"):
        return normalized
    return f"data/{normalized}"


def _variant_filename(*, dataset_variant: str, filename: str) -> str:
    if dataset_variant == "non_downsampled":
        return filename
    if dataset_variant == "downsampled":
        return _DOWNSAMPLED_VARIANT_FILENAME_MAP.get(filename, filename)
    raise ValueError(f"dataset_variant_invalid:{dataset_variant}")


def _validate_dataset_key(key: str) -> None:
    if key in _ACTIVE_BENCHMARK_KEYS:
        return
    parts = Path(key).parts
    if len(parts) == 3 and parts[0] == "data":
        filename = parts[2]
    else:
        raise ValueError(f"dataset_key_invalid_structure:{key}")

    if filename not in _OFFICIAL_DATASET_FILENAMES:
        raise ValueError(f"dataset_key_not_official:{key}")


def required_dataset_keys(config_payload: Mapping[str, object]) -> list[str]:
    """Compute required S3 dataset keys for one training config."""
    data_payload_raw = config_payload.get("data")
    if not isinstance(data_payload_raw, dict):
        return []
    data_payload = data_payload_raw
    data_version = str(data_payload.get("data_version", "v5.2"))
    dataset_variant = str(data_payload.get("dataset_variant", ""))
    if dataset_variant not in _SUPPORTED_DATASET_VARIANTS:
        raise ValueError(f"dataset_variant_invalid:{dataset_variant}")
    dataset_prefix = f"data/{data_version}"

    features_filename = _variant_filename(dataset_variant=dataset_variant, filename="features.json")
    candidates: list[str] = [f"{dataset_prefix}/{features_filename}"]

    if dataset_variant == "downsampled":
        full_filename = _variant_filename(dataset_variant=dataset_variant, filename="full.parquet")
        candidates.append(f"{dataset_prefix}/{full_filename}")
    else:
        train_filename = _variant_filename(dataset_variant=dataset_variant, filename="train.parquet")
        validation_filename = _variant_filename(dataset_variant=dataset_variant, filename="validation.parquet")
        candidates.extend(
            [
                f"{dataset_prefix}/{train_filename}",
                f"{dataset_prefix}/{validation_filename}",
            ]
        )

    benchmark_source_raw = data_payload.get("benchmark_source")
    benchmark_source = benchmark_source_raw if isinstance(benchmark_source_raw, dict) else {}
    benchmark_source_mode = str(benchmark_source.get("source", "active"))
    benchmark_predictions_path = benchmark_source.get("predictions_path")
    if benchmark_source_mode == "active":
        candidates.extend(sorted(_ACTIVE_BENCHMARK_KEYS))
    elif isinstance(benchmark_predictions_path, str) and benchmark_predictions_path:
        benchmark_key = _to_data_key(benchmark_predictions_path)
        if benchmark_key is not None:
            candidates.append(benchmark_key)

    meta_model_data_path = data_payload.get("meta_model_data_path")
    if isinstance(meta_model_data_path, str) and meta_model_data_path:
        meta_model_key = _to_data_key(meta_model_data_path)
        if meta_model_key is not None:
            candidates.append(meta_model_key)

    deduped: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        _validate_dataset_key(key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def stage_required_data(
    config_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    data_root: Path = _DEFAULT_DATA_ROOT,
) -> None:
    """Download required dataset files from S3 into local training data root."""
    resolved_env = os.environ if env is None else env
    bucket = infer_bucket(
        config_s3_uri=resolved_env.get("NUMERENG_CONFIG_S3_URI"),
        output_s3_uri=resolved_env.get("NUMERENG_OUTPUT_S3_URI"),
    )
    if bucket is None:
        _log("bucket_inference_skipped:no NUMERENG_*_S3_URI env var")
        return

    payload = _load_config_payload(config_path)
    required_keys = required_dataset_keys(payload)
    if not required_keys:
        _log("dataset_staging_skipped:no data payload in config")
        return

    s3_client = boto3.client("s3")
    for key in required_keys:
        relative_path = key.removeprefix("data/")
        destination = data_root / relative_path
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        _log(f"download_s3://{bucket}/{key} -> {destination}")
        try:
            s3_client.download_file(bucket, key, str(destination))
        except Exception as exc:
            raise RuntimeError(f"required_data_missing:s3://{bucket}/{key}") from exc


def build_train_command(*, config_path: Path, output_dir: Path) -> list[str]:
    return [
        "numereng",
        "run",
        "train",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]


def sanitize_output_dir(output_dir: Path) -> list[Path]:
    """Remove DB sidecars from managed output so extract only needs run artifacts."""
    removed: list[Path] = []
    for filename in _STORE_DB_FILENAMES:
        candidate = output_dir / filename
        if not candidate.exists() or not candidate.is_file():
            continue
        candidate.unlink()
        removed.append(candidate)
    return removed


def main() -> int:
    try:
        config_path = resolve_config_path()
        output_dir = Path(os.getenv("NUMERENG_TRAIN_OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR))).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        stage_required_data(config_path)
        command = build_train_command(config_path=config_path, output_dir=output_dir)
        _log(f"exec:{' '.join(command)}")
        result = subprocess.run(command, check=False)
        removed = sanitize_output_dir(output_dir)
        if removed:
            _log(f"removed_store_db_files:{','.join(str(path) for path in removed)}")
        return int(result.returncode)
    except Exception as exc:
        _log(f"fatal_error:{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
