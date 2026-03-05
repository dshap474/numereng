"""Config-aware local dataset manifest helpers for Modal volume sync."""

from __future__ import annotations

from pathlib import Path

from numereng.features.cloud.modal.contracts import ModalDataSyncFile
from numereng.features.training.errors import TrainingConfigError
from numereng.features.training.repo import DEFAULT_DATASETS_DIR, load_config, resolve_data_path

MODAL_DATASETS_MOUNT_PATH = "/app/.numereng/datasets"
_SUPPORTED_DATASET_VARIANTS = {"non_downsampled", "downsampled"}
_DOWNSAMPLED_VARIANT_FILENAME_MAP: dict[str, str] = {
    "full.parquet": "downsampled_full.parquet",
    "full_benchmark_models.parquet": "downsampled_full_benchmark_models.parquet",
}


def _variant_filename(*, dataset_variant: str, filename: str) -> str:
    if dataset_variant == "non_downsampled":
        return filename
    if dataset_variant == "downsampled":
        return _DOWNSAMPLED_VARIANT_FILENAME_MAP.get(filename, filename)
    raise ValueError("modal_data_sync_dataset_variant_invalid")


def _as_dict(value: object | None, *, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): value[key] for key in value}
    raise ValueError(f"modal_data_sync_config_{field}_not_object")


def _optional_path(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        candidate = str(value)
    else:
        candidate = str(value)
    stripped = candidate.strip()
    if not stripped:
        return None
    return stripped


def _resolve_remote_path(*, source_path: Path, data_root: Path, label: str) -> str:
    root = data_root.expanduser().resolve()
    resolved_source = source_path.expanduser().resolve()
    try:
        relative = resolved_source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"modal_data_sync_path_outside_data_root:{label}:{resolved_source}") from exc
    remote_path = relative.as_posix()
    if not remote_path or remote_path == ".":
        raise ValueError(f"modal_data_sync_remote_path_invalid:{label}:{resolved_source}")
    return remote_path


def resolve_required_data_files(
    *,
    config_path: str | Path,
    data_root: Path = DEFAULT_DATASETS_DIR,
) -> tuple[str, str, list[ModalDataSyncFile]]:
    """Return data version, dataset variant, and resolved files required for one training config."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    try:
        config = load_config(resolved_config_path)
    except TrainingConfigError as exc:
        raise ValueError(str(exc)) from exc

    data_config = _as_dict(config.get("data"), field="data")
    data_version = str(data_config.get("data_version", "v5.2"))
    dataset_variant = str(data_config.get("dataset_variant", ""))
    if dataset_variant not in _SUPPORTED_DATASET_VARIANTS:
        raise ValueError("modal_data_sync_dataset_variant_invalid")

    requested_paths: list[tuple[str, str]] = []
    full_data_path = _optional_path(data_config.get("full_data_path"))
    if full_data_path is None:
        if dataset_variant == "downsampled":
            full_filename = _variant_filename(dataset_variant=dataset_variant, filename="full.parquet")
            requested_paths.append(("full.parquet", f"{data_version}/{full_filename}"))
        else:
            train_filename = _variant_filename(dataset_variant=dataset_variant, filename="train.parquet")
            validation_filename = _variant_filename(dataset_variant=dataset_variant, filename="validation.parquet")
            requested_paths.extend(
                [
                    ("train.parquet", f"{data_version}/{train_filename}"),
                    ("validation.parquet", f"{data_version}/{validation_filename}"),
                ]
            )
    else:
        requested_paths.append(("full_data_path", full_data_path))

    benchmark_data_path = _optional_path(data_config.get("benchmark_data_path"))
    if benchmark_data_path is not None:
        requested_paths.append(("benchmark_data_path", benchmark_data_path))

    meta_model_data_path = _optional_path(data_config.get("meta_model_data_path"))
    if meta_model_data_path is not None:
        requested_paths.append(("meta_model_data_path", meta_model_data_path))

    features_filename = _variant_filename(dataset_variant=dataset_variant, filename="features.json")
    requested_paths.append(("features.json", f"{data_version}/{features_filename}"))

    manifest_by_remote: dict[str, ModalDataSyncFile] = {}
    for label, raw_path in requested_paths:
        resolved_data_path = resolve_data_path(raw_path, data_root=data_root)
        if not resolved_data_path.exists():
            raise FileNotFoundError(f"modal_data_sync_file_not_found:{label}:{resolved_data_path}")
        if not resolved_data_path.is_file():
            raise ValueError(f"modal_data_sync_path_not_file:{label}:{resolved_data_path}")

        remote_path = _resolve_remote_path(
            source_path=resolved_data_path,
            data_root=data_root,
            label=label,
        )
        manifest_by_remote[remote_path] = ModalDataSyncFile(
            source_path=str(resolved_data_path),
            remote_path=remote_path,
            size_bytes=resolved_data_path.stat().st_size,
        )

    manifest = sorted(manifest_by_remote.values(), key=lambda item: item.remote_path)
    if not manifest:
        raise ValueError("modal_data_sync_manifest_empty")
    return data_version, dataset_variant, manifest


__all__ = ["MODAL_DATASETS_MOUNT_PATH", "resolve_required_data_files"]
