from __future__ import annotations

from pathlib import Path

import pytest

import numereng.features.training.run_log as run_log_module


def test_initialize_run_log_preserves_existing_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "abc123"
    log_path = run_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stale\n", encoding="utf-8")

    resolved = run_log_module.initialize_run_log(run_dir)

    assert resolved == log_path
    assert resolved.read_text(encoding="utf-8") == "stale\n"


def test_run_log_helpers_append_structured_lines(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "abc123"
    log_path = run_log_module.initialize_run_log(run_dir)

    run_log_module.log_info(
        log_path,
        event="run_started",
        message="run_id=abc123",
        attempt_id="attempt-1",
    )
    run_log_module.log_stage(log_path, stage_name="load_data", message="Loading data.")
    run_log_module.log_error(log_path, event="run_failed", message="boom")

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    assert "attempt_id=attempt-1 | INFO | run_started | run_id=abc123" in lines[0]
    assert "attempt_id=unknown | INFO | stage_update | load_data :: Loading data." in lines[1]
    assert "attempt_id=unknown | ERROR | run_failed | boom" in lines[2]


def test_run_log_helpers_are_fail_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "abc123"
    log_path = run_log_module.initialize_run_log(run_dir)

    def _boom(path: Path, line: str) -> None:
        _ = (path, line)
        raise OSError("disk full")

    monkeypatch.setattr(run_log_module, "_append_line", _boom)

    # Fail-open behavior: logging side effects must never raise into training flow.
    run_log_module.log_info(log_path, event="run_started", message="run_id=abc123")
    run_log_module.log_stage(log_path, stage_name="load_data", message="Loading data.")
    run_log_module.log_error(log_path, event="run_failed", message="boom")
