from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.features.docs_sync import sync_numerai_docs


def test_sync_numerai_docs_mirrors_upstream_and_preserves_local_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_clone_shallow(*, repo_url: str, destination: Path) -> str:
        assert repo_url == "https://github.com/numerai/docs.git"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "README.md").write_text("# Numerai Docs\n", encoding="utf-8")
        (destination / "SUMMARY.md").write_text("## Root\n* [Overview](README.md)\n", encoding="utf-8")
        (destination / "guide").mkdir(parents=True)
        (destination / "guide" / "intro.md").write_text("# Intro\n", encoding="utf-8")
        return "deadbeef1234"

    monkeypatch.setattr("numereng.features.docs_sync.clone_shallow", fake_clone_shallow)

    numerai_root = tmp_path / "docs" / "numerai"
    numerai_root.mkdir(parents=True)
    (numerai_root / "SYNC_POLICY.md").write_text("local policy\n", encoding="utf-8")
    (numerai_root / "old.md").write_text("stale\n", encoding="utf-8")
    (numerai_root / "forum").mkdir()
    (numerai_root / "forum" / "INDEX.md").write_text("# forum\n", encoding="utf-8")

    result = sync_numerai_docs(workspace_root=tmp_path)

    assert result.destination_root == numerai_root
    assert result.synced_files == 3
    assert (numerai_root / "README.md").read_text(encoding="utf-8") == "# Numerai Docs\n"
    assert (numerai_root / "guide" / "intro.md").read_text(encoding="utf-8") == "# Intro\n"
    assert not (numerai_root / "old.md").exists()
    assert (numerai_root / "forum" / "INDEX.md").read_text(encoding="utf-8") == "# forum\n"
    assert (numerai_root / "SYNC_POLICY.md").read_text(encoding="utf-8") == "local policy\n"
    assert (numerai_root / ".gitignore").read_text(encoding="utf-8") == ".sync-meta.json\n"

    sync_meta = json.loads((numerai_root / ".sync-meta.json").read_text(encoding="utf-8"))
    assert sync_meta["upstream_commit"] == "deadbeef1234"
    assert sync_meta["destination_root"] == str(numerai_root)
    assert sync_meta["synced_at"].endswith("Z")


def test_sync_numerai_docs_creates_local_policy_files_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_clone_shallow(*, repo_url: str, destination: Path) -> str:
        _ = repo_url
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "README.md").write_text("# Numerai Docs\n", encoding="utf-8")
        return "cafebabe5678"

    monkeypatch.setattr("numereng.features.docs_sync.clone_shallow", fake_clone_shallow)

    sync_numerai_docs(workspace_root=tmp_path)
    numerai_root = tmp_path / "docs" / "numerai"

    assert (numerai_root / "SYNC_POLICY.md").exists()
    assert "uv run numereng docs sync numerai" in (numerai_root / "SYNC_POLICY.md").read_text(encoding="utf-8")
    assert (numerai_root / ".gitignore").read_text(encoding="utf-8") == ".sync-meta.json\n"
