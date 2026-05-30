"""Archive-based remote sync helpers for SSH targets."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from numereng.platform.remotes.contracts import SshRemoteTargetProfile
from numereng.platform.remotes.ssh import build_remote_python_command, build_ssh_command, remote_path_join

_SYNC_MANIFEST_NAME = "__numereng_sync_manifest__.json"
_REMOTE_SYNC_DIR = ("remote_ops", "sync")
_REMOTE_CONFIG_DIR = (".numereng", "tmp", "remote-configs")
_NO_MARKER_SENTINEL = "__NUMERENG_NO_MARKER__"


@dataclass(frozen=True)
class SyncEntry:
    """One local file mapped to one remote relative path."""

    local_path: Path
    remote_relpath: str


@dataclass(frozen=True)
class SyncExecutionResult:
    """Outcome of one archive sync operation."""

    target_id: str
    scope: str
    synced_at: str
    manifest_hash: str
    local_commit_sha: str | None
    dirty: bool
    synced_files: int
    deleted_files: int
    remote_root: str
    local_marker_path: Path | None
    remote_marker_path: str | None
    remote_paths: tuple[str, ...]


def sync_entries_to_remote(
    *,
    target: SshRemoteTargetProfile,
    entries: list[SyncEntry],
    remote_root: str,
    scope: str,
    local_marker_path: Path | None,
    remote_marker_path: str | None,
    local_commit_sha: str | None,
    dirty: bool,
    timeout_seconds: int | None = None,
) -> SyncExecutionResult:
    """Sync one archive bundle to the remote host and return the outcome."""

    synced_at = _utc_now_iso()
    archive_bytes, manifest = _build_archive(
        entries=entries,
        scope=scope,
        synced_at=synced_at,
        local_commit_sha=local_commit_sha,
        dirty=dirty,
    )
    remote_command = build_remote_python_command(
        target,
        _remote_unpack_script(),
        args=[remote_root, remote_marker_path or _NO_MARKER_SENTINEL, _SYNC_MANIFEST_NAME],
        cwd=target.repo_root,
    )
    result = subprocess.run(
        build_ssh_command(target, remote_command),
        input=archive_bytes,
        capture_output=True,
        timeout=timeout_seconds or target.command_timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"remote_sync_command_failed:{stderr or result.returncode}")

    payload = _extract_json_object(result.stdout.decode("utf-8", errors="replace"))
    if payload is None:
        raise RuntimeError("remote_sync_no_json_response")

    if local_marker_path is not None:
        local_marker_path.parent.mkdir(parents=True, exist_ok=True)
        local_marker_path.write_text(
            json.dumps(
                {
                    **manifest,
                    "remote_root": remote_root,
                    "remote_marker_path": payload.get("remote_marker_path"),
                    "synced_files": payload.get("synced_files"),
                    "deleted_files": payload.get("deleted_files"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return SyncExecutionResult(
        target_id=target.id,
        scope=scope,
        synced_at=synced_at,
        manifest_hash=str(manifest["manifest_hash"]),
        local_commit_sha=local_commit_sha,
        dirty=dirty,
        synced_files=int(payload.get("synced_files", 0)),
        deleted_files=int(payload.get("deleted_files", 0)),
        remote_root=remote_root,
        local_marker_path=local_marker_path,
        remote_marker_path=_non_empty_str(payload.get("remote_marker_path")),
        remote_paths=tuple(str(item["path"]) for item in manifest["files"] if isinstance(item, dict)),
    )


def build_repo_marker_paths(*, store_root: Path, target_id: str) -> tuple[Path, str]:
    """Return local/remote marker paths for repo sync scope."""

    local_path = store_root / _REMOTE_SYNC_DIR[0] / _REMOTE_SYNC_DIR[1] / target_id / "repo.json"
    remote_path = remote_path_join_from_store("repo.json", target_id=target_id)
    return local_path, remote_path


def build_experiment_marker_paths(*, store_root: Path, target_id: str, experiment_id: str) -> tuple[Path, str]:
    """Return local/remote marker paths for experiment sync scope."""

    file_name = f"experiment__{_safe_name(experiment_id)}.json"
    local_path = store_root / _REMOTE_SYNC_DIR[0] / _REMOTE_SYNC_DIR[1] / target_id / file_name
    remote_path = remote_path_join_from_store(file_name, target_id=target_id)
    return local_path, remote_path


def remote_config_destination(target: SshRemoteTargetProfile, *, file_name: str) -> str:
    """Return remote absolute config path under the repo-local temp config dir."""

    return remote_path_join(target, target.repo_root, *_REMOTE_CONFIG_DIR, file_name)


def remote_repo_metadata_path(target: SshRemoteTargetProfile, *, target_id: str) -> str:
    """Return remote absolute repo-sync marker path."""

    return remote_path_join(target, target.store_root, *_REMOTE_SYNC_DIR, target_id, "repo.json")


def remote_experiment_metadata_path(
    target: SshRemoteTargetProfile,
    *,
    target_id: str,
    experiment_id: str,
) -> str:
    """Return remote absolute experiment-sync marker path."""

    return remote_path_join(
        target,
        target.store_root,
        *_REMOTE_SYNC_DIR,
        target_id,
        f"experiment__{_safe_name(experiment_id)}.json",
    )


def remote_path_join_from_store(file_name: str, *, target_id: str) -> str:
    """Return a repo-agnostic marker suffix for reuse in service code."""

    return "/".join([*_REMOTE_SYNC_DIR, target_id, file_name])


def _build_archive(
    *,
    entries: list[SyncEntry],
    scope: str,
    synced_at: str,
    local_commit_sha: str | None,
    dirty: bool,
) -> tuple[bytes, dict[str, Any]]:
    file_records: list[dict[str, Any]] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in sorted(entries, key=lambda item: item.remote_relpath):
            data = entry.local_path.read_bytes()
            file_record = {
                "path": entry.remote_relpath,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
            file_records.append(file_record)
            archive.writestr(entry.remote_relpath, data)
        manifest = {
            "version": 1,
            "scope": scope,
            "source_host": socket.gethostname(),
            "local_commit_sha": local_commit_sha,
            "dirty": dirty,
            "synced_at": synced_at,
            "files": file_records,
        }
        manifest["manifest_hash"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        archive.writestr(_SYNC_MANIFEST_NAME, json.dumps(manifest, sort_keys=True))
    return buffer.getvalue(), manifest


def _extract_json_object(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _non_empty_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _remote_unpack_script() -> str:
    return """
import io
import json
import sys
import zipfile
from pathlib import Path


def _safe_path(root: Path, relpath: str) -> Path:
    candidate = (root.joinpath(*relpath.split('/'))).resolve()
    if candidate != root and root not in candidate.parents:
        raise RuntimeError(f"remote_sync_path_escape:{relpath}")
    return candidate


def _cleanup_empty_parents(path: Path, root: Path) -> None:
    parent = path.parent
    while parent != root and parent.exists():
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("remote_sync_arguments_invalid")
    destination_root = Path(sys.argv[1]).expanduser().resolve()
    marker_path_arg = sys.argv[2].strip()
    manifest_name = sys.argv[3]
    destination_root.mkdir(parents=True, exist_ok=True)
    marker_path = None
    if marker_path_arg and marker_path_arg != "__NUMERENG_NO_MARKER__":
        marker_path = Path(marker_path_arg).expanduser().resolve()

    payload = sys.stdin.buffer.read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    manifest = json.loads(archive.read(manifest_name).decode("utf-8"))

    previous_files: set[str] = set()
    if marker_path is not None and marker_path.is_file():
        try:
            previous = json.loads(marker_path.read_text(encoding="utf-8"))
            previous_files = {str(item) for item in previous.get("files", []) if isinstance(item, str)}
        except Exception:
            previous_files = set()

    current_files = {str(item.get("path")) for item in manifest.get("files", []) if isinstance(item, dict)}
    deleted: list[str] = []
    for relpath in sorted(previous_files - current_files, reverse=True):
        target = _safe_path(destination_root, relpath)
        if target.is_file():
            target.unlink()
            deleted.append(relpath)
            _cleanup_empty_parents(target, destination_root)

    for file_entry in manifest.get("files", []):
        if not isinstance(file_entry, dict):
            continue
        relpath = str(file_entry.get("path"))
        target = _safe_path(destination_root, relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(relpath))

    if marker_path is not None:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps(
                {
                    "scope": manifest.get("scope"),
                    "synced_at": manifest.get("synced_at"),
                    "source_host": manifest.get("source_host"),
                    "local_commit_sha": manifest.get("local_commit_sha"),
                    "dirty": manifest.get("dirty"),
                    "manifest_hash": manifest.get("manifest_hash"),
                    "files": sorted(current_files),
                },
                indent=2,
                sort_keys=True,
            )
            + "\\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "remote_marker_path": str(marker_path) if marker_path is not None else None,
                "synced_files": len(current_files),
                "deleted_files": len(deleted),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
""".strip()


@dataclass(frozen=True)
class FetchExtractResult:
    """Outcome of extracting a fetched archive into the local store."""

    manifest_hash: str
    fetched_files: int
    deleted_files: int


def build_experiment_fetch_marker_path(*, store_root: Path, target_id: str, experiment_id: str) -> Path:
    """Return the local marker path tracking the last fetched experiment record.

    Kept distinct from the push marker (``experiment__<id>.json``) so pulling the
    record down never confuses the next push into thinking files were deleted.
    """

    file_name = f"experiment_fetch__{_safe_name(experiment_id)}.json"
    return store_root / _REMOTE_SYNC_DIR[0] / _REMOTE_SYNC_DIR[1] / target_id / file_name


def remote_pack_script() -> str:
    """Return the remote Python source that archives one directory to a temp file.

    Mirrors ``_build_archive``: walks ``source_root`` recursively, writes a
    ZIP_DEFLATED archive plus the shared manifest to ``archive_path`` on the
    remote host, and prints a JSON summary the local side reads back.
    """

    return r"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("remote_fetch_arguments_invalid")
    source_root = Path(sys.argv[1]).expanduser().resolve()
    archive_path = Path(sys.argv[2]).expanduser()
    scope = sys.argv[3]
    manifest_name = sys.argv[4]
    if not source_root.is_dir():
        raise SystemExit("remote_fetch_source_missing:" + str(source_root))
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    file_records = []
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
            relpath = path.relative_to(source_root).as_posix()
            data = path.read_bytes()
            file_records.append(
                {"path": relpath, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            )
            archive.writestr(relpath, data)
        manifest = {"version": 1, "scope": scope, "files": file_records}
        manifest["manifest_hash"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        archive.writestr(manifest_name, json.dumps(manifest, sort_keys=True))

    print(
        json.dumps(
            {
                "archive_path": str(archive_path),
                "manifest_hash": manifest["manifest_hash"],
                "file_count": len(file_records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
""".strip()


def _safe_local_path(root: Path, relpath: str) -> Path:
    candidate = (root.joinpath(*relpath.split("/"))).resolve()
    if candidate != root and root not in candidate.parents:
        raise RuntimeError(f"remote_fetch_path_escape:{relpath}")
    return candidate


def _cleanup_empty_parents_local(path: Path, root: Path) -> None:
    parent = path.parent
    while parent != root and parent.exists():
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def extract_archive_to_local(
    *,
    archive_path: Path,
    destination_root: Path,
    marker_path: Path | None,
    manifest_name: str = _SYNC_MANIFEST_NAME,
) -> FetchExtractResult:
    """Extract a fetched archive into ``destination_root`` with incremental delete.

    Files listed in ``marker_path`` but absent from the new manifest are removed,
    mirroring the remote unpack semantics; everything in the manifest is written.
    """

    destination_root.mkdir(parents=True, exist_ok=True)
    destination_root = destination_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        current_files = {
            str(item["path"]) for item in manifest.get("files", []) if isinstance(item, dict) and item.get("path")
        }
        previous_files: set[str] = set()
        if marker_path is not None and marker_path.is_file():
            try:
                previous = json.loads(marker_path.read_text(encoding="utf-8"))
                previous_files = {str(item) for item in previous.get("files", []) if isinstance(item, str)}
            except (OSError, json.JSONDecodeError):
                previous_files = set()

        deleted: list[str] = []
        for relpath in sorted(previous_files - current_files, reverse=True):
            target = _safe_local_path(destination_root, relpath)
            if target.is_file():
                target.unlink()
                deleted.append(relpath)
                _cleanup_empty_parents_local(target, destination_root)

        for file_entry in manifest.get("files", []):
            if not isinstance(file_entry, dict):
                continue
            relpath = str(file_entry.get("path"))
            target = _safe_local_path(destination_root, relpath)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(relpath))

    manifest_hash = str(manifest.get("manifest_hash", ""))
    if marker_path is not None:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps(
                {
                    "scope": manifest.get("scope"),
                    "manifest_hash": manifest_hash,
                    "files": sorted(current_files),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return FetchExtractResult(
        manifest_hash=manifest_hash,
        fetched_files=len(current_files),
        deleted_files=len(deleted),
    )


__all__ = [
    "FetchExtractResult",
    "SyncEntry",
    "SyncExecutionResult",
    "build_experiment_fetch_marker_path",
    "build_experiment_marker_paths",
    "build_repo_marker_paths",
    "extract_archive_to_local",
    "remote_config_destination",
    "remote_experiment_metadata_path",
    "remote_pack_script",
    "remote_repo_metadata_path",
    "sync_entries_to_remote",
]
