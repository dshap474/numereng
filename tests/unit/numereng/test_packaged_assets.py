from __future__ import annotations

import re
from pathlib import Path

from numereng.assets import assets_root, shipped_skill_ids, shipped_skills_root


def _source_allowlist() -> tuple[str, ...]:
    allowlist_path = Path(".codex/skills/.gitignore")
    skill_ids: list[str] = []
    for raw_line in allowlist_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("!"):
            continue
        if line.endswith("/**"):
            skill_id = line[1:-3]
        elif line.endswith("/"):
            skill_id = line[1:-1]
        else:
            continue
        if not skill_id or skill_id in skill_ids:
            continue
        skill_ids.append(skill_id)
    return tuple(skill_ids)


def test_packaged_assets_are_curated_and_local_path_free() -> None:
    packaged_assets_root = assets_root()
    packaged_numerai_docs_root = packaged_assets_root / "docs" / "numerai"

    assert not packaged_numerai_docs_root.exists()
    assert not any((packaged_assets_root / "docs").rglob(".forum_scraper_state.json"))

    forbidden_patterns = (
        re.compile(r"/Users/[A-Za-z0-9._-]+/"),
        re.compile(r"[A-Z]:\\Users\\[A-Za-z0-9._-]+\\"),
        re.compile(r"\b[A-Z][a-z]+'s PC\b"),
    )
    text_extensions = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh", ".ps1"}
    for path in packaged_assets_root.rglob("*"):
        if not path.is_file() or path.suffix not in text_extensions:
            continue
        payload = path.read_text(encoding="utf-8")
        assert all(pattern.search(payload) is None for pattern in forbidden_patterns), path


def test_packaged_shipped_skills_match_source_allowlist() -> None:
    expected = _source_allowlist()
    packaged = tuple(sorted(path.name for path in shipped_skills_root().iterdir() if path.is_dir()))

    assert expected == shipped_skill_ids()
    assert packaged == tuple(sorted(expected))


def test_machine_specific_remote_profiles_are_not_tracked() -> None:
    profiles_dir = Path("src/numereng/platform/remotes/profiles")
    tracked_profiles = sorted(path.name for path in profiles_dir.glob("*.yaml"))
    tracked_profiles.extend(sorted(path.name for path in profiles_dir.glob("*.yml")))

    assert tracked_profiles == []
