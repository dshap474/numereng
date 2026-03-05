"""State document helpers for managed AWS cloud command chaining."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from numereng.features.cloud.aws.managed_contracts import CloudAwsState


class CloudAwsStateStore:
    """Loads and saves chain-state documents atomically."""

    def load(self, path: Path) -> CloudAwsState | None:
        """Load state from path if it exists."""
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"state_document_read_failed:{path}") from exc

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"state_document_invalid_json:{path}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"state_document_not_object:{path}")

        try:
            return CloudAwsState.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"state_document_invalid_schema:{path}") from exc

    def save(self, path: Path, state: CloudAwsState) -> None:
        """Atomically persist state to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)
