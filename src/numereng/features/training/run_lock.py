"""Run-directory lock helpers for deterministic training run IDs."""

from __future__ import annotations

import json
import logging
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from numereng.features.training.errors import TrainingError

RUN_LOCK_FILENAME = ".train.lock"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunLock:
    """In-memory handle for one acquired run lock file."""

    run_id: str
    path: Path
    metadata: dict[str, object]


def resolve_run_lock_path(run_dir: Path) -> Path:
    """Resolve canonical run lock path for one run directory."""
    return run_dir / RUN_LOCK_FILENAME


def build_local_attempt_id(run_id: str) -> str:
    """Build a unique local attempt identifier for run-local logging."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{run_id}-{timestamp}-pid{os.getpid()}"


def acquire_run_lock(
    *,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
) -> RunLock:
    """Acquire exclusive run lock file for one deterministic run ID."""
    lock_path = resolve_run_lock_path(run_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at": datetime.now(UTC).isoformat(),
    }

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        existing = read_run_lock(lock_path)
        owner = _lock_owner_label(existing)
        raise TrainingError(f"training_run_lock_exists:{run_id}:{owner}") from exc

    payload = json.dumps(metadata, indent=2, sort_keys=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{payload}\n")

    return RunLock(run_id=run_id, path=lock_path, metadata=metadata)


def release_run_lock(lock: RunLock | None) -> None:
    """Release one run lock file."""
    if lock is None:
        return
    try:
        lock.path.unlink(missing_ok=True)
    except Exception:
        logger.exception("failed to release run lock: %s", lock.path)


def read_run_lock(lock_path: Path) -> dict[str, object]:
    """Read run lock payload mapping from disk."""
    if not lock_path.is_file():
        return {}
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def is_lock_payload_active(payload: dict[str, object]) -> bool:
    """Return whether lock owner PID appears alive on this host."""
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    return is_pid_alive(pid)


def is_pid_alive(pid: int) -> bool:
    """Best-effort local PID liveness check."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _lock_owner_label(payload: dict[str, object]) -> str:
    attempt_id = payload.get("attempt_id")
    pid = payload.get("pid")
    host = payload.get("host")
    parts: list[str] = []
    if isinstance(attempt_id, str) and attempt_id:
        parts.append(f"attempt_id={attempt_id}")
    if isinstance(pid, int):
        parts.append(f"pid={pid}")
    if isinstance(host, str) and host:
        parts.append(f"host={host}")
    if not parts:
        return "owner=unknown"
    return ",".join(parts)
