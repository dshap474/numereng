"""Paths to packaged numereng assets shipped with the Python package."""

from __future__ import annotations

import re
from pathlib import Path

_ASSETS_ROOT = Path(__file__).resolve().parent
_SKILL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def assets_root() -> Path:
    return _ASSETS_ROOT


def shipped_skills_root() -> Path:
    return _ASSETS_ROOT / "shipped_skills"


def shipped_skill_ids() -> tuple[str, ...]:
    manifest_path = shipped_skills_root() / "SHIPPED.txt"
    if manifest_path.is_file():
        skill_ids: list[str] = []
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if _SKILL_ID_PATTERN.match(line) is None or line in skill_ids:
                continue
            skill_ids.append(line)
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
