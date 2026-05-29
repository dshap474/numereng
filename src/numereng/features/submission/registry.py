"""Local submitted-model registry helpers."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from numereng.features.store.layout import resolve_workspace_layout

_SAFE_MODEL_NAME = re.compile(r"^[\w\-.]+$")


def record_submission_upload(
    *,
    workspace_root: str | Path,
    model_name: str,
    model_id: str,
    upload_id: str,
    pickle_path: str | Path,
    experiment_id: str,
    package_id: str,
    package_path: str | Path,
    recipe: str | None = None,
    data_version: str | None = None,
    docker_image: str | None = None,
    uploaded_at: str | None = None,
) -> Path:
    """Record a successful hosted-model upload for later live calibration."""

    if not _SAFE_MODEL_NAME.match(model_name):
        raise ValueError(f"Invalid model name: {model_name}")

    layout = resolve_workspace_layout(workspace_root)
    submission_dir = layout.submissions_root / model_name
    submission_path = submission_dir / "submission.json"
    started_at = uploaded_at or datetime.now(UTC).isoformat()
    source = {
        "experiment_id": experiment_id,
        "package_id": package_id,
        "package_path": str(package_path),
    }
    if recipe:
        source["recipe"] = recipe
    hosted_pickle = {
        "upload_id": upload_id,
        "pickle_path": str(pickle_path),
        "uploaded_at": started_at,
    }
    if data_version:
        hosted_pickle["data_version"] = data_version
    if docker_image:
        hosted_pickle["docker_image"] = docker_image

    metadata = _read_json_dict(submission_path)
    uploads = _normalized_uploads(metadata)
    existing_index = _find_upload_index(uploads, upload_id=upload_id)
    new_upload = {
        "upload_id": upload_id,
        "model_id": model_id,
        "model_name": model_name,
        "live_started_at": started_at,
        "live_ended_at": uploads[existing_index].get("live_ended_at") if existing_index is not None else None,
        "source": source,
        "hosted_pickle": hosted_pickle,
    }

    if existing_index is not None:
        uploads[existing_index] = new_upload
    else:
        for upload in uploads:
            if upload.get("live_ended_at") is None:
                upload["live_ended_at"] = started_at
        uploads.append(new_upload)

    uploads.sort(key=lambda item: str(item.get("live_started_at") or ""))
    metadata.update(
        {
            "model_name": model_name,
            "model_id": model_id,
            "status": "hosted_pickle_uploaded",
            "source": source,
            "hosted_pickle": hosted_pickle,
            "uploads": uploads,
            "updated_at": started_at,
        }
    )

    submission_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(metadata, submission_path)
    return submission_path


def _normalized_uploads(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    uploads_raw = metadata.get("uploads")
    uploads: list[dict[str, Any]] = []
    if isinstance(uploads_raw, list):
        uploads.extend(dict(item) for item in uploads_raw if isinstance(item, dict))
    if uploads:
        return uploads

    hosted = metadata.get("hosted_pickle") if isinstance(metadata.get("hosted_pickle"), dict) else {}
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    upload_id = hosted.get("upload_id")
    if not isinstance(upload_id, str) or not upload_id.strip():
        return []
    return [
        {
            "upload_id": upload_id,
            "model_id": metadata.get("model_id"),
            "model_name": metadata.get("model_name"),
            "live_started_at": hosted.get("uploaded_at"),
            "live_ended_at": None,
            "source": source,
            "hosted_pickle": hosted,
        }
    ]


def _find_upload_index(uploads: list[dict[str, Any]], *, upload_id: str) -> int | None:
    for index, upload in enumerate(uploads):
        if upload.get("upload_id") == upload_id:
            return index
        hosted = upload.get("hosted_pickle")
        if isinstance(hosted, dict) and hosted.get("upload_id") == upload_id:
            return index
    return None


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


__all__ = ["record_submission_upload"]
