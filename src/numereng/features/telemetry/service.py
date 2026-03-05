"""Local run telemetry persistence service."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from numereng.features.store import init_store_db
from numereng.features.telemetry.contracts import LocalResourceSampler, LocalRunTelemetrySession, ResourceSample

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_NAME = "local"
_DEFAULT_PRIORITY = 100
_DEFAULT_BACKEND = "local"


def begin_local_training_session(
    *,
    store_root: str | Path,
    config_path: str | Path,
    source: str,
    experiment_id: str | None,
    operation_type: str,
    job_type: str,
    request_payload: dict[str, Any] | None = None,
) -> LocalRunTelemetrySession | None:
    """Create one queued local run-job row and bootstrap telemetry session state."""

    try:
        init_result = init_store_db(store_root=store_root)
        root = init_result.store_root
        db_path = init_result.db_path
        resolved_config = Path(config_path).expanduser().resolve()
        resolved_experiment_id = experiment_id or _infer_experiment_id_from_config_path(
            config_path=resolved_config,
            store_root=root,
        )
        config_id, config_source = _resolve_config_identity(resolved_config, root)
        token = uuid4().hex[:12]
        created_at = _utc_now_iso()
        session = LocalRunTelemetrySession(
            store_root=root,
            db_path=db_path,
            job_id=f"job-{token}",
            batch_id=f"batch-{token}",
            logical_run_id=f"logical-{token}",
            attempt_id=f"attempt-{token}-1",
            attempt_no=1,
            source=source,
            operation_type=operation_type,
            job_type=job_type,
            experiment_id=resolved_experiment_id,
            config_id=config_id,
            config_source=config_source,
            config_path=str(resolved_config),
            config_sha256=_sha256_file(resolved_config),
            queue_name=_DEFAULT_QUEUE_NAME,
            priority=_DEFAULT_PRIORITY,
            backend=_DEFAULT_BACKEND,
            created_at=created_at,
        )
        metadata_json = _safe_json_dumps(
            {
                "source": source,
                "job_type": job_type,
                "config_id": config_id,
            }
        )
        request_json = _safe_json_dumps(
            {
                "source": source,
                "operation_type": operation_type,
                "job_type": job_type,
                "config_id": config_id,
                "config_path": str(resolved_config),
                "experiment_id": resolved_experiment_id,
                **(request_payload or {}),
            }
        )
        with _connect_rw(db_path) as conn:
            conn.execute(
                """
                INSERT INTO logical_runs (
                    logical_run_id,
                    experiment_id,
                    operation_type,
                    status,
                    created_at,
                    updated_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(logical_run_id) DO UPDATE SET
                    experiment_id = excluded.experiment_id,
                    operation_type = excluded.operation_type,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    session.logical_run_id,
                    session.experiment_id,
                    session.operation_type,
                    "queued",
                    created_at,
                    created_at,
                    metadata_json,
                ),
            )
            conn.execute(
                """
                INSERT INTO run_jobs (
                    job_id,
                    batch_id,
                    experiment_id,
                    logical_run_id,
                    operation_type,
                    attempt_no,
                    attempt_id,
                    config_id,
                    config_source,
                    config_path,
                    config_sha256,
                    request_json,
                    job_type,
                    status,
                    queue_name,
                    priority,
                    created_at,
                    queued_at,
                    started_at,
                    finished_at,
                    updated_at,
                    worker_id,
                    pid,
                    exit_code,
                    signal,
                    backend,
                    tier,
                    budget,
                    timeout_seconds,
                    canonical_run_id,
                    external_run_id,
                    run_dir,
                    cancel_requested,
                    error_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    session.job_id,
                    session.batch_id,
                    session.experiment_id,
                    session.logical_run_id,
                    session.operation_type,
                    session.attempt_no,
                    session.attempt_id,
                    session.config_id,
                    session.config_source,
                    session.config_path,
                    session.config_sha256,
                    request_json,
                    session.job_type,
                    "queued",
                    session.queue_name,
                    session.priority,
                    created_at,
                    created_at,
                    None,
                    None,
                    created_at,
                    None,
                    None,
                    None,
                    None,
                    session.backend,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                ),
            )
            conn.execute(
                """
                INSERT INTO run_attempts (
                    attempt_id,
                    logical_run_id,
                    job_id,
                    attempt_no,
                    status,
                    created_at,
                    started_at,
                    finished_at,
                    updated_at,
                    worker_id,
                    pid,
                    exit_code,
                    signal,
                    error_json,
                    canonical_run_id,
                    external_run_id,
                    run_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.attempt_id,
                    session.logical_run_id,
                    session.job_id,
                    session.attempt_no,
                    "queued",
                    created_at,
                    None,
                    None,
                    created_at,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            conn.commit()
        emit_job_event(session, event_type="job_queued", payload={"status": "queued"})
        append_log_line(session, stream="stdout", line=f"[telemetry] queued {session.job_id}")
        append_resource_sample(session, sample=capture_local_resource_sample())
        return session
    except Exception:
        logger.exception("local telemetry bootstrap failed")
        return None


def mark_job_starting(session: LocalRunTelemetrySession, *, pid: int, worker_id: str = "local") -> None:
    """Transition session state to starting."""

    now = _utc_now_iso()
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                """
                UPDATE run_jobs
                SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?, worker_id = ?, pid = ?
                WHERE job_id = ?
                """,
                ("starting", now, now, worker_id, pid, session.job_id),
            )
            conn.execute(
                """
                UPDATE run_attempts
                SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?, worker_id = ?, pid = ?
                WHERE attempt_id = ?
                """,
                ("starting", now, now, worker_id, pid, session.attempt_id),
            )
            conn.execute(
                "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
                ("running", now, session.logical_run_id),
            )
            conn.commit()
    except Exception:
        logger.exception("failed to mark job as starting: %s", session.job_id)
        return
    emit_job_event(
        session,
        event_type="job_starting",
        payload={"status": "starting", "pid": pid, "worker_id": worker_id},
    )


def mark_job_running(session: LocalRunTelemetrySession) -> None:
    """Transition session state to running."""

    now = _utc_now_iso()
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                "UPDATE run_jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                ("running", now, session.job_id),
            )
            conn.execute(
                "UPDATE run_attempts SET status = ?, updated_at = ? WHERE attempt_id = ?",
                ("running", now, session.attempt_id),
            )
            conn.execute(
                "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
                ("running", now, session.logical_run_id),
            )
            conn.commit()
    except Exception:
        logger.exception("failed to mark job as running: %s", session.job_id)
        return
    emit_job_event(session, event_type="job_running", payload={"status": "running"})


def mark_job_completed(
    session: LocalRunTelemetrySession,
    *,
    canonical_run_id: str,
    run_dir: str,
    exit_code: int = 0,
) -> None:
    """Transition session state to completed and persist canonical run linkage."""

    now = _utc_now_iso()
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                """
                UPDATE run_jobs
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    exit_code = ?,
                    canonical_run_id = ?,
                    run_dir = ?,
                    error_json = NULL
                WHERE job_id = ?
                """,
                ("completed", now, now, exit_code, canonical_run_id, run_dir, session.job_id),
            )
            conn.execute(
                """
                UPDATE run_attempts
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    exit_code = ?,
                    canonical_run_id = ?,
                    run_dir = ?,
                    error_json = NULL
                WHERE attempt_id = ?
                """,
                ("completed", now, now, exit_code, canonical_run_id, run_dir, session.attempt_id),
            )
            conn.execute(
                "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
                ("completed", now, session.logical_run_id),
            )
            conn.commit()
    except Exception:
        logger.exception("failed to mark job as completed: %s", session.job_id)
        return
    emit_job_event(
        session,
        event_type="job_completed",
        payload={"status": "completed", "canonical_run_id": canonical_run_id},
    )


def mark_job_failed(
    session: LocalRunTelemetrySession,
    *,
    error: dict[str, Any],
    exit_code: int | None = None,
    signal: int | None = None,
) -> None:
    """Transition session state to failed and persist error payload."""

    now = _utc_now_iso()
    error_json = _safe_json_dumps(error)
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                """
                UPDATE run_jobs
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    exit_code = ?,
                    signal = ?,
                    error_json = ?
                WHERE job_id = ?
                """,
                ("failed", now, now, exit_code, signal, error_json, session.job_id),
            )
            conn.execute(
                """
                UPDATE run_attempts
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    exit_code = ?,
                    signal = ?,
                    error_json = ?
                WHERE attempt_id = ?
                """,
                ("failed", now, now, exit_code, signal, error_json, session.attempt_id),
            )
            conn.execute(
                "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
                ("failed", now, session.logical_run_id),
            )
            conn.commit()
    except Exception:
        logger.exception("failed to mark job as failed: %s", session.job_id)
        return
    emit_job_event(session, event_type="job_failed", payload={"status": "failed", "error": error})


def emit_job_event(
    session: LocalRunTelemetrySession,
    *,
    event_type: str,
    payload: dict[str, Any],
    source: str | None = None,
) -> None:
    """Append one run-job event row."""

    sequence = session.next_event_sequence()
    now = _utc_now_iso()
    resolved_source = source or session.source
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                """
                INSERT INTO run_job_events (
                    job_id,
                    sequence,
                    event_type,
                    source,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session.job_id,
                    sequence,
                    event_type,
                    resolved_source,
                    _safe_json_dumps(payload),
                    now,
                ),
            )
            conn.commit()
    except Exception:
        logger.exception("failed to append run-job event: %s (%s)", session.job_id, event_type)


def emit_stage_event(
    session: LocalRunTelemetrySession,
    *,
    current_stage: str,
    completed_stages: list[str],
    extra_payload: dict[str, Any] | None = None,
) -> None:
    """Append one normalized stage event."""

    payload: dict[str, Any] = {
        "current_stage": current_stage,
        "completed_stages": list(completed_stages),
    }
    if extra_payload:
        payload.update(extra_payload)
    emit_job_event(session, event_type="stage_update", payload=payload)


def emit_metric_event(session: LocalRunTelemetrySession, *, metrics: dict[str, Any]) -> None:
    """Append one metric update event."""

    emit_job_event(session, event_type="metric_update", payload=metrics)


def append_log_line(session: LocalRunTelemetrySession, *, stream: str, line: str) -> None:
    """Append one run-job log line."""

    now = _utc_now_iso()
    line_no = session.next_log_line_no()
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                """
                INSERT INTO run_job_logs (
                    job_id,
                    line_no,
                    stream,
                    line,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session.job_id, line_no, stream, line, now),
            )
            conn.commit()
    except Exception:
        logger.exception("failed to append run-job log line: %s", session.job_id)


def append_resource_sample(session: LocalRunTelemetrySession, *, sample: ResourceSample) -> None:
    """Append one run-job resource sample row."""

    now = _utc_now_iso()
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                """
                INSERT INTO run_job_samples (
                    job_id,
                    cpu_percent,
                    rss_gb,
                    ram_available_gb,
                    gpu_percent,
                    gpu_mem_gb,
                    process_cpu_percent,
                    process_rss_gb,
                    host_cpu_percent,
                    host_ram_available_gb,
                    host_ram_used_gb,
                    host_gpu_percent,
                    host_gpu_mem_used_gb,
                    scope,
                    status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.job_id,
                    sample.process_cpu_percent,
                    sample.process_rss_gb,
                    sample.host_ram_available_gb,
                    sample.host_gpu_percent,
                    sample.host_gpu_mem_used_gb,
                    sample.process_cpu_percent,
                    sample.process_rss_gb,
                    sample.host_cpu_percent,
                    sample.host_ram_available_gb,
                    sample.host_ram_used_gb,
                    sample.host_gpu_percent,
                    sample.host_gpu_mem_used_gb,
                    sample.scope,
                    sample.status,
                    now,
                ),
            )
            conn.commit()
    except Exception:
        logger.exception("failed to append run-job resource sample: %s", session.job_id)


def capture_local_resource_sample() -> ResourceSample:
    """Capture best-effort local process/host resource sample."""

    process_rss_gb = _process_rss_gb()
    host_ram_available_gb, host_ram_used_gb = _host_ram_snapshot_gb()
    if process_rss_gb is not None:
        scope = "launcher_wrapper_only"
    elif host_ram_available_gb is not None or host_ram_used_gb is not None:
        scope = "launcher_host_only"
    else:
        scope = "unavailable"
    status = "partial" if scope != "unavailable" else "unavailable"
    return ResourceSample(
        process_cpu_percent=None,
        process_rss_gb=process_rss_gb,
        host_cpu_percent=None,
        host_ram_available_gb=host_ram_available_gb,
        host_ram_used_gb=host_ram_used_gb,
        host_gpu_percent=None,
        host_gpu_mem_used_gb=None,
        scope=scope,
        status=status,
    )


def start_local_resource_sampler(
    session: LocalRunTelemetrySession,
    *,
    interval_seconds: float = 5.0,
) -> LocalResourceSampler | None:
    """Start best-effort background resource sampling for one telemetry session."""

    wait_seconds = interval_seconds if interval_seconds > 0 else 5.0
    stop_event = threading.Event()

    def _loop() -> None:
        append_resource_sample(session, sample=capture_local_resource_sample())
        while not stop_event.wait(wait_seconds):
            append_resource_sample(session, sample=capture_local_resource_sample())

    try:
        thread = threading.Thread(
            target=_loop,
            name=f"numereng-telemetry-{session.job_id}",
            daemon=True,
        )
        thread.start()
    except Exception:
        logger.exception("failed to start local telemetry sampler: %s", session.job_id)
        return None

    return LocalResourceSampler(stop_event=stop_event, thread=thread)


def stop_local_resource_sampler(sampler: LocalResourceSampler | None) -> None:
    """Stop and join background resource sampler."""

    if sampler is None:
        return
    sampler.stop_event.set()
    sampler.thread.join(timeout=1.0)


def _resolve_config_identity(config_path: Path, store_root: Path) -> tuple[str, str]:
    try:
        relative = config_path.relative_to(store_root).as_posix()
    except ValueError:
        return str(config_path), "external"
    return relative, "store"


def _infer_experiment_id_from_config_path(*, config_path: Path, store_root: Path) -> str | None:
    try:
        relative_parts = config_path.relative_to(store_root).parts
    except ValueError:
        return None
    if len(relative_parts) < 4:
        return None
    if relative_parts[0] != "experiments":
        return None
    if relative_parts[2] != "configs":
        return None
    experiment_id = relative_parts[1].strip()
    return experiment_id or None


def _connect_rw(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=3.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def _process_rss_gb() -> float | None:
    try:
        import resource
    except Exception:
        return None

    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None

    if rss <= 0:
        return None
    if sys.platform == "darwin":
        return rss / (1024.0**3)
    return rss / (1024.0**2)


def _host_ram_snapshot_gb() -> tuple[float | None, float | None]:
    page_size: int | None
    available_pages: int | None
    total_pages: int | None
    page_size = _read_sysconf_int("SC_PAGE_SIZE")
    available_pages = _read_sysconf_int("SC_AVPHYS_PAGES")
    total_pages = _read_sysconf_int("SC_PHYS_PAGES")

    if page_size is None or total_pages is None or total_pages <= 0:
        return None, None

    total_gb = (float(page_size) * float(total_pages)) / (1024.0**3)
    available_gb: float | None
    if available_pages is None:
        available_gb = None
    else:
        available_gb = (float(page_size) * float(available_pages)) / (1024.0**3)

    used_gb: float | None
    if available_gb is None:
        used_gb = None
    else:
        used = total_gb - available_gb
        used_gb = used if used >= 0 else 0.0
    return available_gb, used_gb


def _read_sysconf_int(name: str) -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        value = os.sysconf(name)
    except (ValueError, OSError):
        return None
    if not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


__all__ = [
    "LocalResourceSampler",
    "LocalRunTelemetrySession",
    "ResourceSample",
    "append_log_line",
    "append_resource_sample",
    "begin_local_training_session",
    "capture_local_resource_sample",
    "emit_job_event",
    "emit_metric_event",
    "emit_stage_event",
    "mark_job_completed",
    "mark_job_failed",
    "mark_job_running",
    "mark_job_starting",
    "start_local_resource_sampler",
    "stop_local_resource_sampler",
]
