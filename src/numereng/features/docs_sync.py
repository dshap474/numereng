"""Workspace-local synced docs mirrors."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from numereng.features.store import resolve_workspace_layout
from numereng.platform.docs_sync import clone_shallow

_NUMERAI_DOCS_REPO_URL = "https://github.com/numerai/docs.git"
_PRESERVED_ROOT_NAMES = frozenset({"SYNC_POLICY.md", ".sync-meta.json", ".gitignore", "forum"})
_SYNC_POLICY_CONTENT = """# Numerai Docs Sync Policy

This directory is a synced copy of upstream Numerai docs.

## Source

- Upstream repository: `https://github.com/numerai/docs.git`
- Local mirror path: `docs/numerai/`
- Sync workflow: `uv run numereng docs sync numerai`

## Policy

1. `docs/numerai/` content is synced from upstream and treated as vendor docs.
2. Local sync metadata is recorded in `.sync-meta.json` and should stay untracked.
3. This `SYNC_POLICY.md` file is maintained locally.
4. Do not edit mirrored upstream pages in place; refresh them by re-running sync.
5. `docs/numerai/forum/` is a local generated export target used by
   `numereng numerai forum scrape`; it is not part of the upstream mirrored docs pages.
"""

_SYNC_GITIGNORE_CONTENT = ".sync-meta.json\n"


@dataclass(frozen=True)
class DocsSyncResult:
    workspace_root: Path
    destination_root: Path
    sync_meta_path: Path
    upstream_commit: str
    synced_at: str
    synced_files: int


def sync_numerai_docs(*, workspace_root: str | Path = ".") -> DocsSyncResult:
    """Clone upstream Numerai docs and mirror them into one workspace."""

    layout = resolve_workspace_layout(workspace_root)
    destination_root = layout.workspace_root / "docs" / "numerai"
    destination_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="numereng-docs-sync-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        clone_root = tmp_dir / "numerai-docs"
        upstream_commit = clone_shallow(repo_url=_NUMERAI_DOCS_REPO_URL, destination=clone_root)
        synced_files = _mirror_docs_tree(source_root=clone_root, destination_root=destination_root)

    _ensure_local_policy_files(destination_root)
    synced_at = _utc_now_iso()
    sync_meta_path = destination_root / ".sync-meta.json"
    sync_meta_path.write_text(
        json.dumps(
            {
                "source": "https://github.com/numerai/docs.git",
                "upstream_commit": upstream_commit,
                "synced_at": synced_at,
                "destination_root": str(destination_root),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return DocsSyncResult(
        workspace_root=layout.workspace_root,
        destination_root=destination_root,
        sync_meta_path=sync_meta_path,
        upstream_commit=upstream_commit,
        synced_at=synced_at,
        synced_files=synced_files,
    )


def _mirror_docs_tree(*, source_root: Path, destination_root: Path) -> int:
    source_entries = {path.relative_to(source_root) for path in source_root.rglob("*") if ".git" not in path.parts}
    synced_files = 0

    for relative in sorted(source_entries, key=lambda path: (len(path.parts), path.as_posix())):
        target_path = destination_root / relative
        if _is_preserved_path(relative):
            continue
        source_path = source_root / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        synced_files += 1

    for target_path in sorted(destination_root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        relative = target_path.relative_to(destination_root)
        if _is_preserved_path(relative):
            continue
        if relative not in source_entries:
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()

    return synced_files


def _is_preserved_path(relative: Path) -> bool:
    if not relative.parts:
        return False
    return relative.parts[0] in _PRESERVED_ROOT_NAMES


def _ensure_local_policy_files(destination_root: Path) -> None:
    sync_policy_path = destination_root / "SYNC_POLICY.md"
    if not sync_policy_path.exists():
        sync_policy_path.write_text(_SYNC_POLICY_CONTENT, encoding="utf-8")

    sync_gitignore_path = destination_root / ".gitignore"
    if not sync_gitignore_path.exists():
        sync_gitignore_path.write_text(_SYNC_GITIGNORE_CONTENT, encoding="utf-8")
        return

    existing_lines = sync_gitignore_path.read_text(encoding="utf-8").splitlines()
    if ".sync-meta.json" not in existing_lines:
        updated_lines = [*existing_lines, ".sync-meta.json"]
        sync_gitignore_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = ["DocsSyncResult", "sync_numerai_docs"]
