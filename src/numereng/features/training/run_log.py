"""Run-local plain-text logging helpers for training runs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

RUN_LOG_FILENAME = "run.log"

logger = logging.getLogger(__name__)


def resolve_run_log_path(run_dir: Path) -> Path:
    """Resolve canonical run-local live log path."""
    return run_dir / RUN_LOG_FILENAME


def initialize_run_log(run_dir: Path) -> Path:
    """Create/truncate one run-local live log file."""
    run_log_path = resolve_run_log_path(run_dir)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_log_path.write_text("", encoding="utf-8")
    return run_log_path


def log_stage(run_log_path: Path | None, *, stage_name: str, message: str) -> None:
    """Append one stage update line to the run-local log."""
    _log_line(
        run_log_path,
        level="INFO",
        event="stage_update",
        message=f"{stage_name} :: {message}",
    )


def log_info(run_log_path: Path | None, *, event: str, message: str) -> None:
    """Append one informational line to the run-local log."""
    _log_line(run_log_path, level="INFO", event=event, message=message)


def log_error(run_log_path: Path | None, *, event: str, message: str) -> None:
    """Append one error line to the run-local log."""
    _log_line(run_log_path, level="ERROR", event=event, message=message)


def _log_line(
    run_log_path: Path | None,
    *,
    level: str,
    event: str,
    message: str,
) -> None:
    if run_log_path is None:
        return

    line = f"{_utc_now_iso()} | {level.upper()} | {event} | {message}\n"
    try:
        _append_line(run_log_path, line)
    except Exception:
        logger.exception("failed to append run-local log line: %s", run_log_path)


def _append_line(run_log_path: Path, line: str) -> None:
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with run_log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()

