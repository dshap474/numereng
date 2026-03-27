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
from numereng.features.telemetry.contracts import (
    LocalResourceSampler,
    LocalRunTelemetrySession,
    ResourceSample,
    RunCancelResult,
    RunLifecycleRecord,
    RunLifecycleRepairResult,
)
from numereng.features.telemetry.lifecycle import (
    current_host_name,
    initialize_run_lifecycle,
    update_run_lifecycle_metrics,
    update_run_lifecycle_sample,
    update_run_lifecycle_stage,
    update_run_lifecycle_status,
)
from numereng.features.telemetry.lifecycle import (
    get_run_lifecycle as get_run_lifecycle_record,
)
from numereng.features.telemetry.lifecycle import (
    is_cancel_requested as is_lifecycle_cancel_requested,
)
from numereng.features.telemetry.lifecycle import (
    reconcile_run_lifecycles as reconcile_lifecycle_rows,
)
from numereng.features.telemetry.lifecycle import (
    request_run_cancel as request_run_lifecycle_cancel,
)

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_NAME = "local"
_DEFAULT_PRIORITY = 100
_DEFAULT_BACKEND = "local"


def begin_local_training_session(
    *,
    store_root: str | Path,
    config_path: str | Path,
    run_id: str,
    run_hash: str,
    config_hash: str,
    run_dir: str | Path,
    runtime_path: str | Path,
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
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        resolved_runtime_path = Path(runtime_path).expanduser().resolve()
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
            run_id=run_id,
            run_hash=run_hash,
            config_hash=config_hash,
            run_dir=resolved_run_dir,
            runtime_path=resolved_runtime_path,
            host=current_host_name(),
            created_at=created_at,
        )
        metadata_json = _safe_json_dumps(
            {
                "source": source,
                "job_type": job_type,
                "config_id": config_id,
                "run_id": run_id,
                "run_dir": str(resolved_run_dir),
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
                "run_id": run_id,
                "run_dir": str(resolved_run_dir),
                "runtime_path": str(resolved_runtime_path),
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
                    cancel_requested_at,
                    terminal_reason,
                    terminal_detail_json,
                    error_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                    session.run_id,
                    None,
                    str(session.run_dir),
                    0,
                    None,
                    None,
                    None,
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
                    cancel_requested_at,
                    terminal_reason,
                    terminal_detail_json,
                    error_json,
                    canonical_run_id,
                    external_run_id,
                    run_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    session.run_id,
                    None,
                    str(session.run_dir),
                ),
            )
            initialize_run_lifecycle(conn, session)
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
                ("starting", now, session.logical_run_id),
            )
            update_run_lifecycle_status(
                conn,
                session,
                status="starting",
                worker_id=worker_id,
                pid=pid,
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
            update_run_lifecycle_status(conn, session, status="running")
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
    terminal_reason: str = "completed",
    terminal_detail: dict[str, Any] | None = None,
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
                    terminal_reason = ?,
                    terminal_detail_json = ?,
                    error_json = NULL
                WHERE job_id = ?
                """,
                (
                    "completed",
                    now,
                    now,
                    exit_code,
                    canonical_run_id,
                    run_dir,
                    terminal_reason,
                    _safe_json_dumps(terminal_detail or {}),
                    session.job_id,
                ),
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
                    terminal_reason = ?,
                    terminal_detail_json = ?,
                    error_json = NULL
                WHERE attempt_id = ?
                """,
                (
                    "completed",
                    now,
                    now,
                    exit_code,
                    canonical_run_id,
                    run_dir,
                    terminal_reason,
                    _safe_json_dumps(terminal_detail or {}),
                    session.attempt_id,
                ),
            )
            conn.execute(
                "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
                ("completed", now, session.logical_run_id),
            )
            update_run_lifecycle_status(
                conn,
                session,
                status="completed",
                finished_at=now,
                terminal_reason=terminal_reason,
                terminal_detail=terminal_detail or {},
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
    terminal_reason: str = "failed",
    terminal_detail: dict[str, Any] | None = None,
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
                    terminal_reason = ?,
                    terminal_detail_json = ?,
                    error_json = ?
                WHERE job_id = ?
                """,
                (
                    "failed",
                    now,
                    now,
                    exit_code,
                    signal,
                    terminal_reason,
                    _safe_json_dumps(terminal_detail or {}),
                    error_json,
                    session.job_id,
                ),
            )
            conn.execute(
                """
                UPDATE run_attempts
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    exit_code = ?,
                    signal = ?,
                    terminal_reason = ?,
                    terminal_detail_json = ?,
                    error_json = ?
                WHERE attempt_id = ?
                """,
                (
                    "failed",
                    now,
                    now,
                    exit_code,
                    signal,
                    terminal_reason,
                    _safe_json_dumps(terminal_detail or {}),
                    error_json,
                    session.attempt_id,
                ),
            )
            conn.execute(
                "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
                ("failed", now, session.logical_run_id),
            )
            update_run_lifecycle_status(
                conn,
                session,
                status="failed",
                finished_at=now,
                terminal_reason=terminal_reason,
                terminal_detail=terminal_detail or error,
            )
            conn.commit()
    except Exception:
        logger.exception("failed to mark job as failed: %s", session.job_id)
        return
    emit_job_event(session, event_type="job_failed", payload={"status": "failed", "error": error})


def mark_job_canceled(
    session: LocalRunTelemetrySession,
    *,
    terminal_reason: str = "cancel_requested",
    terminal_detail: dict[str, Any] | None = None,
) -> None:
    """Transition session state to canceled."""

    now = _utc_now_iso()
    payload = terminal_detail or {}
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                """
                UPDATE run_jobs
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    cancel_requested = 1,
                    cancel_requested_at = COALESCE(cancel_requested_at, ?),
                    terminal_reason = ?,
                    terminal_detail_json = ?,
                    error_json = NULL
                WHERE job_id = ?
                """,
                (
                    "canceled",
                    now,
                    now,
                    now,
                    terminal_reason,
                    _safe_json_dumps(payload),
                    session.job_id,
                ),
            )
            conn.execute(
                """
                UPDATE run_attempts
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    cancel_requested_at = COALESCE(cancel_requested_at, ?),
                    terminal_reason = ?,
                    terminal_detail_json = ?,
                    error_json = NULL
                WHERE attempt_id = ?
                """,
                (
                    "canceled",
                    now,
                    now,
                    now,
                    terminal_reason,
                    _safe_json_dumps(payload),
                    session.attempt_id,
                ),
            )
            conn.execute(
                "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
                ("canceled", now, session.logical_run_id),
            )
            update_run_lifecycle_status(
                conn,
                session,
                status="canceled",
                finished_at=now,
                terminal_reason=terminal_reason,
                terminal_detail=payload,
            )
            conn.commit()
    except Exception:
        logger.exception("failed to mark job as canceled: %s", session.job_id)
        return
    emit_job_event(session, event_type="job_canceled", payload={"status": "canceled", **payload})


def mark_job_stale(
    session: LocalRunTelemetrySession,
    *,
    terminal_reason: str = "stale",
    terminal_detail: dict[str, Any] | None = None,
) -> None:
    """Transition session state to stale."""

    now = _utc_now_iso()
    payload = terminal_detail or {}
    try:
        with _connect_rw(session.db_path) as conn:
            conn.execute(
                """
                UPDATE run_jobs
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    terminal_reason = ?,
                    terminal_detail_json = ?,
                    error_json = ?
                WHERE job_id = ?
                """,
                (
                    "stale",
                    now,
                    now,
                    terminal_reason,
                    _safe_json_dumps(payload),
                    _safe_json_dumps(payload),
                    session.job_id,
                ),
            )
            conn.execute(
                """
                UPDATE run_attempts
                SET status = ?,
                    finished_at = ?,
                    updated_at = ?,
                    terminal_reason = ?,
                    terminal_detail_json = ?,
                    error_json = ?
                WHERE attempt_id = ?
                """,
                (
                    "stale",
                    now,
                    now,
                    terminal_reason,
                    _safe_json_dumps(payload),
                    _safe_json_dumps(payload),
                    session.attempt_id,
                ),
            )
            conn.execute(
                "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
                ("stale", now, session.logical_run_id),
            )
            update_run_lifecycle_status(
                conn,
                session,
                status="stale",
                finished_at=now,
                terminal_reason=terminal_reason,
                terminal_detail=payload,
            )
            conn.commit()
    except Exception:
        logger.exception("failed to mark job as stale: %s", session.job_id)
        return
    emit_job_event(session, event_type="job_stale", payload={"status": "stale", **payload})


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
            if event_type == "stage_update":
                current_stage = payload.get("current_stage")
                completed_stages = payload.get("completed_stages")
                if isinstance(current_stage, str) and isinstance(completed_stages, list):
                    update_run_lifecycle_stage(
                        conn,
                        session,
                        current_stage=current_stage,
                        completed_stages=[str(item) for item in completed_stages],
                        progress_percent=_coerce_optional_float(payload.get("progress_percent")),
                        progress_label=_coerce_optional_str(payload.get("progress_label")),
                        progress_current=_coerce_optional_int(payload.get("progress_current")),
                        progress_total=_coerce_optional_int(payload.get("progress_total")),
                    )
            elif event_type == "metric_update":
                update_run_lifecycle_metrics(conn, session, metrics=payload)
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
            update_run_lifecycle_sample(conn, session, sample=sample)
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


def get_run_lifecycle(*, store_root: str | Path = ".numereng", run_id: str) -> RunLifecycleRecord | None:
    """Load one canonical lifecycle summary by run id."""

    return get_run_lifecycle_record(store_root=store_root, run_id=run_id)


def request_run_cancel(*, store_root: str | Path = ".numereng", run_id: str) -> RunCancelResult:
    """Request cooperative cancel for one local run."""

    return request_run_lifecycle_cancel(store_root=store_root, run_id=run_id)


def reconcile_run_lifecycles(
    *,
    store_root: str | Path = ".numereng",
    run_id: str | None = None,
    active_only: bool = True,
) -> RunLifecycleRepairResult:
    """Reconcile active lifecycle rows against local liveness evidence."""

    return reconcile_lifecycle_rows(store_root=store_root, run_id=run_id, active_only=active_only)


def is_cancel_requested(session: LocalRunTelemetrySession) -> bool:
    """Return whether one session has a cooperative cancel request pending."""

    try:
        return is_lifecycle_cancel_requested(session)
    except Exception:
        logger.exception("failed to read cancel-request flag: %s", session.run_id)
        return False


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
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


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
    "get_run_lifecycle",
    "is_cancel_requested",
    "mark_job_canceled",
    "mark_job_completed",
    "mark_job_failed",
    "mark_job_running",
    "mark_job_starting",
    "mark_job_stale",
    "reconcile_run_lifecycles",
    "request_run_cancel",
    "start_local_resource_sampler",
    "stop_local_resource_sampler",
]
