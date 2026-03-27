"""Load user-local remote monitoring target profiles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from numereng.platform.remotes.contracts import RemoteTargetError, SshRemoteTargetProfile

_PROFILE_GLOB = "*.yaml"
_ENV_REMOTE_PROFILES_DIR = "NUMERENG_REMOTE_PROFILES_DIR"


def default_remote_profiles_dir() -> Path:
    """Return the default repo-local profiles directory."""

    return (Path(__file__).resolve().parent / "profiles").resolve()


def resolve_remote_profiles_dir(path: str | Path | None = None) -> Path:
    """Resolve active remote profiles directory from explicit path or env."""

    if path is not None:
        return Path(path).expanduser().resolve()
    env_value = os.getenv(_ENV_REMOTE_PROFILES_DIR)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return default_remote_profiles_dir()


def load_remote_targets(
    path: str | Path | None = None,
    *,
    strict: bool = False,
) -> list[SshRemoteTargetProfile]:
    """Load all enabled remote SSH profiles from YAML files."""

    profiles_dir = resolve_remote_profiles_dir(path)
    if not profiles_dir.exists():
        return []

    targets: list[SshRemoteTargetProfile] = []
    for profile_path in sorted(profiles_dir.glob(_PROFILE_GLOB)):
        if not profile_path.is_file() or profile_path.name.startswith("."):
            continue
        try:
            targets.extend(_load_remote_target_file(profile_path))
        except RemoteTargetError:
            if strict:
                raise

    seen_ids: set[str] = set()
    deduped: list[SshRemoteTargetProfile] = []
    for target in targets:
        if not target.enabled:
            continue
        if target.id in seen_ids:
            if strict:
                raise RemoteTargetError(f"duplicate remote target id: {target.id}")
            continue
        seen_ids.add(target.id)
        deduped.append(target)
    return deduped


def _load_remote_target_file(path: Path) -> list[SshRemoteTargetProfile]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise RemoteTargetError(f"invalid remote target file {path}: {exc}") from exc

    items: list[dict[str, Any]]
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        items = [item for item in payload if isinstance(item, dict)]
    else:
        raise RemoteTargetError(f"remote target file must contain a mapping or list of mappings: {path}")

    targets: list[SshRemoteTargetProfile] = []
    for item in items:
        try:
            targets.append(SshRemoteTargetProfile.model_validate(item))
        except Exception as exc:
            raise RemoteTargetError(f"invalid remote target in {path}: {exc}") from exc
    return targets


__all__ = [
    "default_remote_profiles_dir",
    "load_remote_targets",
    "resolve_remote_profiles_dir",
]
