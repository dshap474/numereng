"""Paths to packaged numereng assets shipped with the Python package."""

from __future__ import annotations

import re
from pathlib import Path

_ASSETS_ROOT = Path(__file__).resolve().parent
_ALLOWLIST_PATTERN = re.compile(r"^!([A-Za-z0-9_-]+)/\*\*$|^!([A-Za-z0-9_-]+)/$")


def assets_root() -> Path:
    return _ASSETS_ROOT


def shipped_skills_root() -> Path:
    return _ASSETS_ROOT / "shipped_skills"


def shipped_skill_ids() -> tuple[str, ...]:
    allowlist_path = shipped_skills_root() / ".gitignore"
    if allowlist_path.is_file():
        skill_ids: list[str] = []
        for raw_line in allowlist_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = _ALLOWLIST_PATTERN.match(line)
            if match is None:
                continue
            skill_id = match.group(1) or match.group(2)
            if skill_id is None or skill_id in skill_ids:
                continue
            skill_ids.append(skill_id)
        return tuple(skill_ids)

    return tuple(sorted(path.name for path in shipped_skills_root().iterdir() if path.is_dir()))


def viz_static_root() -> Path:
    return _ASSETS_ROOT / "viz_static"


def docs_root(domain: str) -> Path:
    if domain not in {"numerai", "numereng"}:
        raise ValueError(f"unsupported docs domain: {domain}")
    return _ASSETS_ROOT / "docs" / domain


def docs_assets_root() -> Path:
    return _ASSETS_ROOT / "docs" / "assets"


__all__ = [
    "assets_root",
    "docs_assets_root",
    "docs_root",
    "shipped_skill_ids",
    "shipped_skills_root",
    "viz_static_root",
]
