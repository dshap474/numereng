from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from numereng.features.training.errors import TrainingError
from numereng.features.training.run_lock import (
    acquire_run_lock,
    build_local_attempt_id,
    is_lock_payload_active,
    is_pid_alive,
    read_run_lock,
    release_run_lock,
    resolve_run_lock_path,
)


def test_acquire_and_release_run_lock(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "abc123"
    lock = acquire_run_lock(
        run_dir=run_dir,
        run_id="abc123",
        attempt_id=build_local_attempt_id("abc123"),
    )

    lock_payload = read_run_lock(resolve_run_lock_path(run_dir))
    assert lock_payload.get("run_id") == "abc123"
    assert isinstance(lock_payload.get("pid"), int)
    assert is_lock_payload_active(lock_payload) is True

    release_run_lock(lock)
    assert not resolve_run_lock_path(run_dir).exists()


def test_acquire_run_lock_rejects_existing_owner(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "abc123"
    first_lock = acquire_run_lock(
        run_dir=run_dir,
        run_id="abc123",
        attempt_id="attempt-1",
    )

    try:
        with pytest.raises(TrainingError, match="training_run_lock_exists:abc123"):
            acquire_run_lock(
                run_dir=run_dir,
                run_id="abc123",
                attempt_id="attempt-2",
            )
    finally:
        release_run_lock(first_lock)


def test_is_lock_payload_active_handles_missing_pid() -> None:
    assert is_lock_payload_active({}) is False
    assert is_lock_payload_active({"pid": "123"}) is False
    assert is_lock_payload_active({"pid": -5}) is False


def test_is_pid_alive_uses_windows_helper_when_running_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    run_lock_module = importlib.import_module("numereng.features.training.run_lock")
    monkeypatch.setattr(run_lock_module.os, "name", "nt", raising=False)
    monkeypatch.setattr(run_lock_module, "_windows_pid_alive", lambda pid: pid == 123)

    assert is_pid_alive(123) is True
    assert is_pid_alive(456) is False
