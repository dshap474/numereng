"""Run-id and manifest helpers for training run artifact storage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

RunStatus = Literal["RUNNING", "FINISHED", "FAILED", "CANCELED", "STALE"]

RUN_MANIFEST_SCHEMA_VERSION = "1"


def compute_config_hash(config: dict[str, object]) -> str:
    """Compute deterministic hash for full loaded config payload."""
    canonical = _canonical_json(config)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_run_hash(
    *,
    config: dict[str, object],
    data_version: str,
    feature_set: str,
    target_col: str,
    model_type: str,
    engine_mode: str,
    engine_settings: dict[str, object],
) -> str:
    """Compute deterministic training run identity hash."""
    output_config = _as_mapping(config.get("output"))
    material_config = _strip_removed_training_identity_fields(dict(config))
    material_config.pop("output", None)

    identity: dict[str, object] = {
        "config": material_config,
        "data_version": data_version,
        "feature_set": feature_set,
        "target_col": target_col,
        "model_type": model_type,
        "engine_mode": engine_mode,
        "engine_settings": engine_settings,
        "output_baselines_dir": _normalize_path_value(output_config.get("baselines_dir")),
    }
    canonical = _canonical_json(identity)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_id_from_hash(run_hash: str) -> str:
    """Return deterministic short run-id derived from run hash."""
    return run_hash[:12]


def build_run_manifest(
    *,
    run_id: str,
    run_hash: str,
    status: RunStatus,
    config_path: Path,
    config_hash: str,
    data_version: str,
    feature_set: str,
    target_col: str,
    model_type: str,
    engine_mode: str,
    experiment_id: str | None = None,
    created_at: str | None = None,
    artifacts: dict[str, str] | None = None,
    metrics_summary: dict[str, object] | None = None,
    error: dict[str, str] | None = None,
    training_metadata: dict[str, object] | None = None,
    lifecycle_metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build minimal v1 training run manifest payload."""
    created = created_at or _utc_now_iso()
    training_payload: dict[str, object] = {
        "engine": engine_mode,
    }
    if training_metadata:
        training_payload.update(training_metadata)

    manifest: dict[str, object] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_hash": run_hash,
        "run_type": "training",
        "status": status,
        "created_at": created,
        "config": {
            "path": str(config_path),
            "hash": config_hash,
        },
        "data": {
            "version": data_version,
            "feature_set": feature_set,
            "target_col": target_col,
        },
        "model": {
            "type": model_type,
        },
        "training": training_payload,
        "artifacts": artifacts or {},
    }
    if experiment_id:
        manifest["experiment_id"] = experiment_id
    if status in {"FINISHED", "FAILED", "CANCELED", "STALE"}:
        manifest["finished_at"] = _utc_now_iso()
    if metrics_summary:
        manifest["metrics_summary"] = metrics_summary
    if error:
        manifest["error"] = error
    if lifecycle_metadata:
        manifest["lifecycle"] = lifecycle_metadata
    return manifest


def error_payload(exc: Exception) -> dict[str, str]:
    """Build stable error payload for failed run manifest."""
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def _canonical_json(value: object) -> str:
    normalized = _normalize_for_json(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _normalize_for_json(value: object) -> object:
    if isinstance(value, dict):
        normalized_items: dict[str, object] = {}
        for key, child in value.items():
            normalized_items[str(key)] = _normalize_for_json(child)
        return normalized_items
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _normalize_path_value(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _strip_removed_training_identity_fields(config: dict[str, object]) -> dict[str, object]:
    data_config = _as_mapping(config.get("data"))
    if "loading" not in data_config:
        return config

    normalized = dict(config)
    normalized_data = dict(data_config)
    normalized_data.pop("loading", None)
    normalized["data"] = normalized_data
    return normalized


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
