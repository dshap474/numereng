"""State document helpers for cloud EC2 command chaining."""

from __future__ import annotations

import json
from pathlib import Path

from numereng.features.cloud.aws.contracts import CloudEc2State


class CloudEc2StateStore:
    """Loads and saves chain-state documents atomically."""

    def load(self, path: Path) -> CloudEc2State | None:
        """Load state from path if it exists."""
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid state document at {path}")
        return CloudEc2State.model_validate(payload)

    def save(self, path: Path, state: CloudEc2State) -> None:
        """Atomically persist state to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)
