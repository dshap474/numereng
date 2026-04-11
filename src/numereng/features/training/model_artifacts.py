"""Persistence helpers for fitted full-history model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle

from numereng.features.training.repo import (
    resolve_model_artifact_path,
    resolve_model_manifest_path,
)

_MODEL_ARTIFACT_SCHEMA_VERSION = "1"


class ModelArtifactError(Exception):
    """Raised when a persisted model artifact cannot be saved or loaded."""


@dataclass(frozen=True)
class ModelArtifactManifest:
    """Small manifest needed to reuse one fitted model for live inference."""

    run_id: str
    model_type: str
    data_version: str
    dataset_variant: str
    feature_set: str
    target_col: str
    era_col: str
    id_col: str
    feature_cols: tuple[str, ...]
    baseline_col: str | None = None
    baseline_name: str | None = None
    baseline_predictions_path: str | None = None
    baseline_pred_col: str = "prediction"
    model_upload_compatible: bool = False
    uses_custom_module: bool = False


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Loaded fitted model plus its runtime manifest."""

    model: Any
    manifest: ModelArtifactManifest
    artifact_path: Path
    manifest_path: Path


def save_model_artifact(
    *,
    run_dir: Path,
    model: Any,
    manifest: ModelArtifactManifest,
) -> tuple[Path, Path]:
    """Persist one fitted model and inference manifest under a run root."""

    artifact_path = resolve_model_artifact_path(run_dir)
    manifest_path = resolve_model_manifest_path(run_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with artifact_path.open("wb") as handle:
            cloudpickle.dump(model, handle)
    except Exception as exc:  # pragma: no cover - dependency/backend-specific
        raise ModelArtifactError("training_model_artifact_save_failed") from exc
    payload = {
        "schema_version": _MODEL_ARTIFACT_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "model_type": manifest.model_type,
        "data_version": manifest.data_version,
        "dataset_variant": manifest.dataset_variant,
        "feature_set": manifest.feature_set,
        "target_col": manifest.target_col,
        "era_col": manifest.era_col,
        "id_col": manifest.id_col,
        "feature_cols": list(manifest.feature_cols),
        "baseline_col": manifest.baseline_col,
        "baseline_name": manifest.baseline_name,
        "baseline_predictions_path": manifest.baseline_predictions_path,
        "baseline_pred_col": manifest.baseline_pred_col,
        "model_upload_compatible": manifest.model_upload_compatible,
        "uses_custom_module": manifest.uses_custom_module,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path, manifest_path


def load_model_artifact(*, run_dir: Path) -> LoadedModelArtifact:
    """Load one persisted fitted model artifact from a run root."""

    artifact_path = resolve_model_artifact_path(run_dir)
    manifest_path = resolve_model_manifest_path(run_dir)
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise ModelArtifactError("serving_model_artifact_missing")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelArtifactError("serving_model_artifact_manifest_invalid") from exc
    manifest = _manifest_from_payload(payload)
    try:
        with artifact_path.open("rb") as handle:
            model = cloudpickle.load(handle)
    except Exception as exc:  # pragma: no cover - dependency/backend-specific
        raise ModelArtifactError("serving_model_artifact_load_failed") from exc
    return LoadedModelArtifact(
        model=model,
        manifest=manifest,
        artifact_path=artifact_path.resolve(),
        manifest_path=manifest_path.resolve(),
    )


def _manifest_from_payload(payload: dict[str, object]) -> ModelArtifactManifest:
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != _MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError("serving_model_artifact_manifest_invalid")
    feature_cols = payload.get("feature_cols")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ModelArtifactError("serving_model_artifact_manifest_invalid")
    id_col = payload.get("id_col")
    if not isinstance(id_col, str) or not id_col.strip():
        raise ModelArtifactError("serving_model_artifact_manifest_invalid")
    return ModelArtifactManifest(
        run_id=str(payload.get("run_id", "")),
        model_type=str(payload.get("model_type", "")),
        data_version=str(payload.get("data_version", "")),
        dataset_variant=str(payload.get("dataset_variant", "")),
        feature_set=str(payload.get("feature_set", "")),
        target_col=str(payload.get("target_col", "")),
        era_col=str(payload.get("era_col", "")),
        id_col=id_col,
        feature_cols=tuple(str(item) for item in feature_cols),
        baseline_col=None if payload.get("baseline_col") is None else str(payload.get("baseline_col")),
        baseline_name=None if payload.get("baseline_name") is None else str(payload.get("baseline_name")),
        baseline_predictions_path=None
        if payload.get("baseline_predictions_path") is None
        else str(payload.get("baseline_predictions_path")),
        baseline_pred_col=str(payload.get("baseline_pred_col", "prediction")),
        model_upload_compatible=bool(payload.get("model_upload_compatible", False)),
        uses_custom_module=bool(payload.get("uses_custom_module", False)),
    )


__all__ = [
    "LoadedModelArtifact",
    "ModelArtifactError",
    "ModelArtifactManifest",
    "load_model_artifact",
    "save_model_artifact",
]
