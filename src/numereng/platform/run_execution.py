"""Helpers for durable run execution provenance."""

from __future__ import annotations

import base64
import json
import os
import socket
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

RUN_EXECUTION_ENV_VAR = "NUMERENG_RUN_EXECUTION_JSON"
RUN_EXECUTION_ENV_B64_VAR = "NUMERENG_RUN_EXECUTION_JSON_B64"
_RUN_EXECUTION_BASE_FIELDS = (
    "kind",
    "provider",
    "backend",
    "provider_job_id",
    "target_id",
    "host",
    "instance_id",
    "region",
    "image_uri",
    "output_uri",
    "state_path",
    "submitted_at",
    "pulled_at",
    "extracted_at",
)


def build_run_execution(
    *,
    kind: str,
    provider: str,
    backend: str,
    provider_job_id: str | None = None,
    target_id: str | None = None,
    host: str | None = None,
    instance_id: str | None = None,
    region: str | None = None,
    image_uri: str | None = None,
    output_uri: str | None = None,
    state_path: str | None = None,
    submitted_at: str | None = None,
    pulled_at: str | None = None,
    extracted_at: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one normalized execution provenance payload."""

    payload: dict[str, object] = {
        "kind": kind.strip(),
        "provider": provider.strip(),
        "backend": backend.strip(),
    }
    optional_values = {
        "provider_job_id": provider_job_id,
        "target_id": target_id,
        "host": host,
        "instance_id": instance_id,
        "region": region,
        "image_uri": image_uri,
        "output_uri": output_uri,
        "state_path": state_path,
        "submitted_at": submitted_at,
        "pulled_at": pulled_at,
        "extracted_at": extracted_at,
    }
    for key, value in optional_values.items():
        normalized = _normalize_scalar(value)
        if normalized is not None:
            payload[key] = normalized
    normalized_metadata = _normalize_metadata(metadata)
    if normalized_metadata:
        payload["metadata"] = normalized_metadata
    return payload


def build_local_run_execution(*, source: str | None = None, host: str | None = None) -> dict[str, object]:
    """Build the default execution payload for a local run."""

    metadata: dict[str, object] = {}
    source_value = _normalize_scalar(source)
    if source_value is not None:
        metadata["source"] = source_value
    return build_run_execution(
        kind="local",
        provider="local",
        backend="local",
        host=_normalize_scalar(host) or socket.gethostname(),
        metadata=metadata or None,
    )


def merge_run_execution(
    existing: object,
    incoming: Mapping[str, object] | None,
    *,
    prefer_incoming: bool = False,
) -> dict[str, object]:
    """Merge execution payloads without clobbering existing populated fields."""

    merged = _normalize_existing(existing)
    if incoming is None:
        return merged
    for key in _RUN_EXECUTION_BASE_FIELDS:
        candidate = _normalize_scalar(incoming.get(key))
        if candidate is None:
            continue
        if prefer_incoming or key not in merged or _normalize_scalar(merged.get(key)) is None:
            merged[key] = candidate
    incoming_metadata = _normalize_metadata(incoming.get("metadata"))
    if incoming_metadata:
        existing_metadata = _normalize_metadata(merged.get("metadata"))
        for key, value in incoming_metadata.items():
            if prefer_incoming or key not in existing_metadata:
                existing_metadata[key] = value
        if existing_metadata:
            merged["metadata"] = existing_metadata
    return merged


def serialize_run_execution(execution: Mapping[str, object]) -> str:
    """Serialize one execution payload for env/process handoff."""

    payload = merge_run_execution(None, execution)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_run_execution_from_env(*, env: Mapping[str, str] | None = None) -> dict[str, object] | None:
    """Load one execution payload from supported env vars when present."""

    resolved_env = os.environ if env is None else env
    raw = resolved_env.get(RUN_EXECUTION_ENV_VAR)
    if raw and raw.strip():
        return _parse_execution_json(raw)
    encoded = resolved_env.get(RUN_EXECUTION_ENV_B64_VAR)
    if encoded and encoded.strip():
        try:
            decoded = base64.b64decode(encoded.encode("ascii"), validate=True).decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("run_execution_env_invalid") from exc
        return _parse_execution_json(decoded)
    return None


def stamp_run_execution(
    *,
    manifest_path: str | Path,
    execution: Mapping[str, object],
    prefer_incoming: bool = False,
) -> dict[str, object]:
    """Merge execution provenance into one persisted `run.json` manifest."""

    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run_manifest_not_object:{path}")
    payload["execution"] = merge_run_execution(
        payload.get("execution"),
        execution,
        prefer_incoming=prefer_incoming,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)
    return payload


def _normalize_existing(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, object] = {}
    for key in _RUN_EXECUTION_BASE_FIELDS:
        candidate = _normalize_scalar(value.get(key))
        if candidate is not None:
            normalized[key] = candidate
    metadata = _normalize_metadata(value.get("metadata"))
    if metadata:
        normalized["metadata"] = metadata
    return normalized


def _parse_execution_json(raw: str) -> dict[str, object]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("run_execution_env_invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("run_execution_env_invalid")
    return merge_run_execution(None, payload)


def _normalize_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, object] = {}
    for key, child in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        normalized_value = _normalize_json_scalar(child)
        if normalized_value is not None:
            normalized[key_text] = normalized_value
    return normalized


def _normalize_json_scalar(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    text = str(value).strip()
    return text or None


def _normalize_scalar(value: object) -> str | None:
    normalized = _normalize_json_scalar(value)
    if normalized is None:
        return None
    return str(normalized)


__all__ = [
    "RUN_EXECUTION_ENV_B64_VAR",
    "RUN_EXECUTION_ENV_VAR",
    "build_local_run_execution",
    "build_run_execution",
    "load_run_execution_from_env",
    "merge_run_execution",
    "serialize_run_execution",
    "stamp_run_execution",
]
