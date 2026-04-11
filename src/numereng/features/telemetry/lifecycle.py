"""Canonical live lifecycle read model and runtime snapshot helpers."""

from __future__ import annotations

import json
import re
import socket
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from numereng.features.store import StoreError, init_store_db, resolve_store_root
from numereng.features.telemetry.contracts import (
    LocalRunTelemetrySession,
    ResourceSample,
    RunCancelResult,
    RunLifecycleRecord,
    RunLifecycleRepairResult,
)
from numereng.features.training.run_lock import (
    is_lock_payload_active,
    is_pid_alive,
    read_run_lock,
    resolve_run_lock_path,
)

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_ACTIVE_STATUSES = {"queued", "starting", "running"}
_TERMINAL_STATUSES = {"completed", "failed", "canceled", "stale"}
_HEARTBEAT_STALE_AFTER = timedelta(seconds=20)
_SNAPSHOT_SCHEMA_VERSION = "1"


def initialize_run_lifecycle(conn: sqlite3.Connection, session: LocalRunTelemetrySession) -> None:
    """Insert or refresh the canonical live lifecycle row for one local run."""

    now = session.created_at
    conn.execute(
        """
        INSERT INTO run_lifecycles (
            run_id,
            run_hash,
            config_hash,
            job_id,
            logical_run_id,
            attempt_id,
            attempt_no,
            source,
            operation_type,
            job_type,
            status,
            experiment_id,
            config_id,
            config_source,
            config_path,
            config_sha256,
            run_dir,
            runtime_path,
            backend,
            worker_id,
            pid,
            host,
            current_stage,
            completed_stages_json,
            progress_percent,
            progress_label,
            progress_current,
            progress_total,
            cancel_requested,
            cancel_requested_at,
            created_at,
            queued_at,
            started_at,
            last_heartbeat_at,
            updated_at,
            finished_at,
            terminal_reason,
            terminal_detail_json,
            latest_metrics_json,
            latest_sample_json,
            reconciled
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(run_id) DO UPDATE SET
            run_hash = excluded.run_hash,
            config_hash = excluded.config_hash,
            job_id = excluded.job_id,
            logical_run_id = excluded.logical_run_id,
            attempt_id = excluded.attempt_id,
            attempt_no = excluded.attempt_no,
            source = excluded.source,
            operation_type = excluded.operation_type,
            job_type = excluded.job_type,
            status = excluded.status,
            experiment_id = excluded.experiment_id,
            config_id = excluded.config_id,
            config_source = excluded.config_source,
            config_path = excluded.config_path,
            config_sha256 = excluded.config_sha256,
            run_dir = excluded.run_dir,
            runtime_path = excluded.runtime_path,
            backend = excluded.backend,
            worker_id = NULL,
            pid = NULL,
            host = excluded.host,
            current_stage = excluded.current_stage,
            completed_stages_json = excluded.completed_stages_json,
            progress_percent = excluded.progress_percent,
            progress_label = excluded.progress_label,
            progress_current = excluded.progress_current,
            progress_total = excluded.progress_total,
            cancel_requested = 0,
            cancel_requested_at = NULL,
            queued_at = excluded.queued_at,
            started_at = NULL,
            last_heartbeat_at = excluded.last_heartbeat_at,
            updated_at = excluded.updated_at,
            finished_at = NULL,
            terminal_reason = NULL,
            terminal_detail_json = '{}',
            latest_metrics_json = '{}',
            latest_sample_json = '{}',
            reconciled = 0
        """,
        (
            session.run_id,
            session.run_hash,
            session.config_hash,
            session.job_id,
            session.logical_run_id,
            session.attempt_id,
            session.attempt_no,
            session.source,
            session.operation_type,
            session.job_type,
            "queued",
            session.experiment_id,
            session.config_id,
            session.config_source,
            session.config_path,
            session.config_sha256,
            str(session.run_dir),
            str(session.runtime_path),
            session.backend,
            None,
            None,
            session.host,
            "initializing",
            "[]",
            0.0,
            "Queued",
            0,
            None,
            0,
            None,
            session.created_at,
            session.created_at,
            None,
            session.created_at,
            now,
            None,
            None,
            "{}",
            "{}",
            "{}",
            0,
        ),
    )
    rewrite_runtime_snapshot(conn, run_id=session.run_id)


def update_run_lifecycle_status(
    conn: sqlite3.Connection,
    session: LocalRunTelemetrySession,
    *,
    status: str,
    worker_id: str | None = None,
    pid: int | None = None,
    finished_at: str | None = None,
    terminal_reason: str | None = None,
    terminal_detail: dict[str, Any] | None = None,
    reconciled: bool | None = None,
    progress_percent: float | None = None,
    progress_label: str | None = None,
    progress_current: int | None = None,
    progress_total: int | None = None,
) -> None:
    """Update one lifecycle row status and rewrite runtime snapshot."""

    now = finished_at or _utc_now_iso()
    assignments = [
        "status = ?",
        "updated_at = ?",
    ]
    params: list[Any] = [status, now]

    if worker_id is not None:
        assignments.append("worker_id = ?")
        params.append(worker_id)
    if pid is not None:
        assignments.append("pid = ?")
        params.append(pid)
    if status in {"starting", "running"}:
        assignments.append("started_at = COALESCE(started_at, ?)")
        params.append(now)
    if finished_at is not None:
        assignments.append("finished_at = ?")
        params.append(finished_at)
    if terminal_reason is not None:
        assignments.append("terminal_reason = ?")
        params.append(terminal_reason)
    if terminal_detail is not None:
        assignments.append("terminal_detail_json = ?")
        params.append(_safe_json_dumps(terminal_detail))
    if reconciled is not None:
        assignments.append("reconciled = ?")
        params.append(1 if reconciled else 0)
    if status == "completed" and progress_percent is None:
        progress_percent = 100.0
    if status == "completed" and progress_label is None:
        progress_label = "Completed"
    if progress_percent is not None:
        assignments.append("progress_percent = ?")
        params.append(progress_percent)
    if progress_label is not None:
        assignments.append("progress_label = ?")
        params.append(progress_label)
    if progress_current is not None:
        assignments.append("progress_current = ?")
        params.append(progress_current)
    if progress_total is not None:
        assignments.append("progress_total = ?")
        params.append(progress_total)
    if status in _TERMINAL_STATUSES:
        assignments.append("last_heartbeat_at = COALESCE(last_heartbeat_at, ?)")
        params.append(now)

    params.append(session.run_id)
    conn.execute(f"UPDATE run_lifecycles SET {', '.join(assignments)} WHERE run_id = ?", tuple(params))
    rewrite_runtime_snapshot(conn, run_id=session.run_id)


def update_run_lifecycle_stage(
    conn: sqlite3.Connection,
    session: LocalRunTelemetrySession,
    *,
    current_stage: str,
    completed_stages: list[str],
    progress_percent: float | None = None,
    progress_label: str | None = None,
    progress_current: int | None = None,
    progress_total: int | None = None,
) -> None:
    """Update one lifecycle row with current stage + completed stages."""

    now = _utc_now_iso()
    assignments = [
        "current_stage = ?",
        "completed_stages_json = ?",
        "updated_at = ?",
    ]
    params: list[Any] = [
        current_stage,
        _safe_json_dumps(list(completed_stages)),
        now,
    ]
    if progress_percent is not None:
        assignments.append("progress_percent = ?")
        params.append(progress_percent)
    if progress_label is not None:
        assignments.append("progress_label = ?")
        params.append(progress_label)
    if progress_current is not None:
        assignments.append("progress_current = ?")
        params.append(progress_current)
    if progress_total is not None:
        assignments.append("progress_total = ?")
        params.append(progress_total)
    params.append(session.run_id)
    conn.execute(
        f"UPDATE run_lifecycles SET {', '.join(assignments)} WHERE run_id = ?",
        tuple(params),
    )
    rewrite_runtime_snapshot(conn, run_id=session.run_id)


def update_run_lifecycle_metrics(
    conn: sqlite3.Connection,
    session: LocalRunTelemetrySession,
    *,
    metrics: dict[str, Any],
) -> None:
    """Update one lifecycle row with latest metric summary."""

    now = _utc_now_iso()
    conn.execute(
        """
        UPDATE run_lifecycles
        SET latest_metrics_json = ?,
            updated_at = ?
        WHERE run_id = ?
        """,
        (
            _safe_json_dumps(metrics),
            now,
            session.run_id,
        ),
    )
    rewrite_runtime_snapshot(conn, run_id=session.run_id)


def update_run_lifecycle_sample(
    conn: sqlite3.Connection,
    session: LocalRunTelemetrySession,
    *,
    sample: ResourceSample,
) -> None:
    """Update one lifecycle row with latest heartbeat/sample payload."""

    now = _utc_now_iso()
    conn.execute(
        """
        UPDATE run_lifecycles
        SET latest_sample_json = ?,
            last_heartbeat_at = ?,
            updated_at = ?
        WHERE run_id = ?
        """,
        (
            _safe_json_dumps(_sample_to_payload(sample)),
            now,
            now,
            session.run_id,
        ),
    )
    rewrite_runtime_snapshot(conn, run_id=session.run_id)


def get_run_lifecycle(*, store_root: str | Path = ".numereng", run_id: str) -> RunLifecycleRecord | None:
    """Load one canonical live lifecycle row by run id."""

    safe_run_id = _ensure_safe_run_id(run_id)
    root = resolve_store_root(store_root)
    db_path = root / "numereng.db"
    if not db_path.exists():
        return _runtime_snapshot_fallback(root=root, run_id=safe_run_id)

    with _connect_rw(db_path) as conn:
        try:
            row = conn.execute("SELECT * FROM run_lifecycles WHERE run_id = ?", (safe_run_id,)).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None:
            return _row_to_record(row)

    return _runtime_snapshot_fallback(root=root, run_id=safe_run_id)


def request_run_cancel(*, store_root: str | Path = ".numereng", run_id: str) -> RunCancelResult:
    """Mark one active local run as cooperatively cancel-requested."""

    safe_run_id = _ensure_safe_run_id(run_id)
    init_result = init_store_db(store_root=store_root)
    now = _utc_now_iso()

    with _connect_rw(init_result.db_path) as conn:
        row = conn.execute(
            "SELECT run_id, job_id, status, cancel_requested, cancel_requested_at FROM run_lifecycles WHERE run_id = ?",
            (safe_run_id,),
        ).fetchone()
        if row is None:
            raise StoreError(f"run_lifecycle_not_found:{safe_run_id}")

        status = str(row["status"])
        accepted = status in _ACTIVE_STATUSES and not bool(row["cancel_requested"])
        cancel_requested_at = row["cancel_requested_at"] if row["cancel_requested_at"] else now

        conn.execute(
            """
            UPDATE run_lifecycles
            SET cancel_requested = 1,
                cancel_requested_at = COALESCE(cancel_requested_at, ?),
                updated_at = ?
            WHERE run_id = ?
            """,
            (now, now, safe_run_id),
        )
        conn.execute(
            """
            UPDATE run_jobs
            SET cancel_requested = 1,
                cancel_requested_at = COALESCE(cancel_requested_at, ?),
                updated_at = ?
            WHERE canonical_run_id = ? OR job_id = ?
            """,
            (now, now, safe_run_id, str(row["job_id"])),
        )
        conn.execute(
            """
            UPDATE run_attempts
            SET cancel_requested_at = COALESCE(cancel_requested_at, ?),
                updated_at = ?
            WHERE canonical_run_id = ? OR job_id = ?
            """,
            (now, now, safe_run_id, str(row["job_id"])),
        )
        if accepted:
            _append_job_event(
                conn,
                job_id=str(row["job_id"]),
                event_type="cancel_requested",
                source="api.run.cancel",
                payload={"status": status, "run_id": safe_run_id, "cancel_requested_at": now},
            )
        rewrite_runtime_snapshot(conn, run_id=safe_run_id)
        conn.commit()

        return RunCancelResult(
            run_id=safe_run_id,
            job_id=str(row["job_id"]),
            status=status,
            cancel_requested=True,
            cancel_requested_at=str(cancel_requested_at),
            accepted=accepted,
        )


def is_cancel_requested(session: LocalRunTelemetrySession) -> bool:
    """Return whether one active run has a cooperative cancel request pending."""

    with _connect_rw(session.db_path) as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM run_lifecycles WHERE run_id = ?",
            (session.run_id,),
        ).fetchone()
        if row is None:
            return False
        return bool(row["cancel_requested"])


def reconcile_run_lifecycles(
    *,
    store_root: str | Path = ".numereng",
    run_id: str | None = None,
    active_only: bool = True,
) -> RunLifecycleRepairResult:
    """Reconcile active lifecycle rows against lock, pid, and heartbeat evidence."""

    resolved_root = resolve_store_root(store_root)
    db_path = resolved_root / "numereng.db"
    if not db_path.exists():
        return RunLifecycleRepairResult(
            store_root=resolved_root,
            scanned_count=0,
            unchanged_count=0,
            reconciled_count=0,
            reconciled_stale_count=0,
            reconciled_canceled_count=0,
            run_ids=(),
        )
    safe_run_id = _ensure_safe_run_id(run_id) if run_id is not None else None
    scanned_count = 0
    unchanged_count = 0
    reconciled_stale_count = 0
    reconciled_canceled_count = 0
    reconciled_run_ids: list[str] = []

    with _connect_rw(db_path) as conn:
        sql = "SELECT * FROM run_lifecycles"
        params: list[Any] = []
        clauses: list[str] = []
        if safe_run_id is not None:
            clauses.append("run_id = ?")
            params.append(safe_run_id)
        if active_only:
            placeholders = ", ".join("?" for _ in _ACTIVE_STATUSES)
            clauses.append(f"status IN ({placeholders})")
            params.extend(sorted(_ACTIVE_STATUSES))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error:
            rows = []
        for row in rows:
            scanned_count += 1
            record = _row_to_record(row)
            if _lifecycle_is_still_active(record):
                unchanged_count += 1
                continue

            if record.cancel_requested:
                _apply_terminal_reconciliation(
                    conn,
                    record,
                    status="canceled",
                    terminal_reason="reconciled_cancel_requested",
                    terminal_detail={
                        "reconciled": True,
                        "reason": "cancel_requested_process_exited_before_terminal_write",
                    },
                )
                reconciled_canceled_count += 1
                reconciled_run_ids.append(record.run_id)
                continue

            _apply_terminal_reconciliation(
                conn,
                record,
                status="stale",
                terminal_reason="reconciled_stale",
                terminal_detail={
                    "reconciled": True,
                    "reason": "missing_process_or_stale_heartbeat",
                },
            )
            reconciled_stale_count += 1
            reconciled_run_ids.append(record.run_id)

    return RunLifecycleRepairResult(
        store_root=resolved_root,
        scanned_count=scanned_count,
        unchanged_count=unchanged_count,
        reconciled_count=reconciled_stale_count + reconciled_canceled_count,
        reconciled_stale_count=reconciled_stale_count,
        reconciled_canceled_count=reconciled_canceled_count,
        run_ids=tuple(reconciled_run_ids),
    )


def rewrite_runtime_snapshot(conn: sqlite3.Connection, *, run_id: str) -> None:
    """Rewrite one runtime snapshot file from the canonical lifecycle row."""

    row = conn.execute("SELECT * FROM run_lifecycles WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return
    record = _row_to_record(row)
    runtime_path = Path(record.runtime_path)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SNAPSHOT_SCHEMA_VERSION,
        "run_id": record.run_id,
        "run_hash": record.run_hash,
        "config_hash": record.config_hash,
        "job_id": record.job_id,
        "logical_run_id": record.logical_run_id,
        "attempt_id": record.attempt_id,
        "attempt_no": record.attempt_no,
        "source": record.source,
        "operation_type": record.operation_type,
        "job_type": record.job_type,
        "status": record.status,
        "experiment_id": record.experiment_id,
        "config": {
            "id": record.config_id,
            "source": record.config_source,
            "path": record.config_path,
            "sha256": record.config_sha256,
        },
        "runtime": {
            "run_dir": record.run_dir,
            "runtime_path": record.runtime_path,
            "backend": record.backend,
            "worker_id": record.worker_id,
            "pid": record.pid,
            "host": record.host,
            "current_stage": record.current_stage,
            "completed_stages": list(record.completed_stages),
            "progress_percent": record.progress_percent,
            "progress_label": record.progress_label,
            "progress_current": record.progress_current,
            "progress_total": record.progress_total,
            "cancel_requested": record.cancel_requested,
            "cancel_requested_at": record.cancel_requested_at,
            "created_at": record.created_at,
            "queued_at": record.queued_at,
            "started_at": record.started_at,
            "last_heartbeat_at": record.last_heartbeat_at,
            "updated_at": record.updated_at,
            "finished_at": record.finished_at,
            "terminal_reason": record.terminal_reason,
            "terminal_detail": record.terminal_detail,
            "latest_metrics": record.latest_metrics,
            "latest_sample": record.latest_sample,
            "reconciled": record.reconciled,
        },
    }
    _atomic_write_json(runtime_path, payload)


def _apply_terminal_reconciliation(
    conn: sqlite3.Connection,
    record: RunLifecycleRecord,
    *,
    status: str,
    terminal_reason: str,
    terminal_detail: dict[str, Any],
) -> None:
    now = _utc_now_iso()
    error_json = None if status == "canceled" else _safe_json_dumps(terminal_detail)
    conn.execute(
        """
        UPDATE run_lifecycles
        SET status = ?,
            finished_at = ?,
            updated_at = ?,
            terminal_reason = ?,
            terminal_detail_json = ?,
            reconciled = 1
        WHERE run_id = ?
        """,
        (status, now, now, terminal_reason, _safe_json_dumps(terminal_detail), record.run_id),
    )
    conn.execute(
        """
        UPDATE run_jobs
        SET status = ?,
            finished_at = ?,
            updated_at = ?,
            terminal_reason = ?,
            terminal_detail_json = ?,
            error_json = ?,
            cancel_requested = CASE WHEN ? THEN 1 ELSE cancel_requested END
        WHERE job_id = ?
        """,
        (
            status,
            now,
            now,
            terminal_reason,
            _safe_json_dumps(terminal_detail),
            error_json,
            1 if status == "canceled" else 0,
            record.job_id,
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
            status,
            now,
            now,
            terminal_reason,
            _safe_json_dumps(terminal_detail),
            error_json,
            record.attempt_id,
        ),
    )
    conn.execute(
        "UPDATE logical_runs SET status = ?, updated_at = ? WHERE logical_run_id = ?",
        (status, now, record.logical_run_id),
    )
    _append_job_event(
        conn,
        job_id=record.job_id,
        event_type=f"job_{status}",
        source="reconciler",
        payload={
            "status": status,
            "run_id": record.run_id,
            "terminal_reason": terminal_reason,
            **terminal_detail,
        },
    )
    _append_job_log(
        conn,
        job_id=record.job_id,
        stream="stderr" if status == "stale" else "stdout",
        line=f"[reconciler] {status} run {record.run_id}: {terminal_reason}",
    )
    rewrite_runtime_snapshot(conn, run_id=record.run_id)
    _rewrite_terminal_run_manifest(
        record=record,
        status=status,
        terminal_reason=terminal_reason,
        terminal_detail=terminal_detail,
    )
    conn.commit()


def _rewrite_terminal_run_manifest(
    *,
    record: RunLifecycleRecord,
    status: str,
    terminal_reason: str,
    terminal_detail: dict[str, Any],
) -> None:
    manifest_path = Path(record.run_dir) / "run.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload["status"] = status.upper()
    payload["finished_at"] = payload.get("finished_at") or _utc_now_iso()
    payload["lifecycle"] = {
        "terminal_reason": terminal_reason,
        "terminal_detail": terminal_detail,
        "cancel_requested_at": record.cancel_requested_at,
        "reconciled": True,
    }
    _atomic_write_json(manifest_path, payload)


def _lifecycle_is_still_active(record: RunLifecycleRecord) -> bool:
    if record.status not in _ACTIVE_STATUSES:
        return False

    run_dir = Path(record.run_dir)
    lock_payload = read_run_lock(resolve_run_lock_path(run_dir))
    lock_active = is_lock_payload_active(lock_payload)
    pid_active = is_pid_alive(record.pid) if record.pid is not None else False
    heartbeat_fresh = _heartbeat_is_fresh(record.last_heartbeat_at)
    return heartbeat_fresh and (pid_active or lock_active)


def _runtime_snapshot_fallback(*, root: Path, run_id: str) -> RunLifecycleRecord | None:
    runtime_path = root / "runs" / run_id / "runtime.json"
    if not runtime_path.is_file():
        return None
    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    runtime = payload.get("runtime")
    config = payload.get("config")
    if not isinstance(runtime, dict) or not isinstance(config, dict):
        return None
    return RunLifecycleRecord(
        run_id=run_id,
        run_hash=str(payload.get("run_hash") or ""),
        config_hash=str(payload.get("config_hash") or ""),
        job_id=str(payload.get("job_id") or ""),
        logical_run_id=str(payload.get("logical_run_id") or ""),
        attempt_id=str(payload.get("attempt_id") or ""),
        attempt_no=int(payload.get("attempt_no") or 1),
        source=str(payload.get("source") or ""),
        operation_type=str(payload.get("operation_type") or ""),
        job_type=str(payload.get("job_type") or ""),
        status=str(payload.get("status") or "unknown"),
        experiment_id=_as_optional_str(payload.get("experiment_id")),
        config_id=str(config.get("id") or ""),
        config_source=str(config.get("source") or ""),
        config_path=str(config.get("path") or ""),
        config_sha256=str(config.get("sha256") or ""),
        run_dir=str(runtime.get("run_dir") or runtime_path.parent),
        runtime_path=str(runtime_path),
        backend=_as_optional_str(runtime.get("backend")),
        worker_id=_as_optional_str(runtime.get("worker_id")),
        pid=_as_optional_int(runtime.get("pid")),
        host=_as_optional_str(runtime.get("host")),
        current_stage=_as_optional_str(runtime.get("current_stage")),
        completed_stages=tuple(_as_str_list(runtime.get("completed_stages"))),
        progress_percent=_as_optional_float(runtime.get("progress_percent")),
        progress_label=_as_optional_str(runtime.get("progress_label")),
        progress_current=_as_optional_int(runtime.get("progress_current")),
        progress_total=_as_optional_int(runtime.get("progress_total")),
        cancel_requested=bool(runtime.get("cancel_requested")),
        cancel_requested_at=_as_optional_str(runtime.get("cancel_requested_at")),
        created_at=str(runtime.get("created_at") or ""),
        queued_at=_as_optional_str(runtime.get("queued_at")),
        started_at=_as_optional_str(runtime.get("started_at")),
        last_heartbeat_at=_as_optional_str(runtime.get("last_heartbeat_at")),
        updated_at=str(runtime.get("updated_at") or ""),
        finished_at=_as_optional_str(runtime.get("finished_at")),
        terminal_reason=_as_optional_str(runtime.get("terminal_reason")),
        terminal_detail=_as_json_object(runtime.get("terminal_detail")),
        latest_metrics=_as_json_object(runtime.get("latest_metrics")),
        latest_sample=_as_json_object(runtime.get("latest_sample")),
        reconciled=bool(runtime.get("reconciled")),
    )


def _row_to_record(row: sqlite3.Row) -> RunLifecycleRecord:
    return RunLifecycleRecord(
        run_id=str(row["run_id"]),
        run_hash=str(row["run_hash"]),
        config_hash=str(row["config_hash"] or ""),
        job_id=str(row["job_id"]),
        logical_run_id=str(row["logical_run_id"]),
        attempt_id=str(row["attempt_id"]),
        attempt_no=int(row["attempt_no"]),
        source=str(row["source"]),
        operation_type=str(row["operation_type"]),
        job_type=str(row["job_type"]),
        status=str(row["status"]),
        experiment_id=_as_optional_str(row["experiment_id"]),
        config_id=str(row["config_id"]),
        config_source=str(row["config_source"]),
        config_path=str(row["config_path"]),
        config_sha256=str(row["config_sha256"]),
        run_dir=str(row["run_dir"]),
        runtime_path=str(row["runtime_path"]),
        backend=_as_optional_str(row["backend"]),
        worker_id=_as_optional_str(row["worker_id"]),
        pid=_as_optional_int(row["pid"]),
        host=_as_optional_str(row["host"]),
        current_stage=_as_optional_str(row["current_stage"]),
        completed_stages=tuple(_as_str_list(_parse_json_field(row["completed_stages_json"]))),
        progress_percent=_as_optional_float(row["progress_percent"]),
        progress_label=_as_optional_str(row["progress_label"]),
        progress_current=_as_optional_int(row["progress_current"]),
        progress_total=_as_optional_int(row["progress_total"]),
        cancel_requested=bool(row["cancel_requested"]),
        cancel_requested_at=_as_optional_str(row["cancel_requested_at"]),
        created_at=str(row["created_at"]),
        queued_at=_as_optional_str(row["queued_at"]),
        started_at=_as_optional_str(row["started_at"]),
        last_heartbeat_at=_as_optional_str(row["last_heartbeat_at"]),
        updated_at=str(row["updated_at"]),
        finished_at=_as_optional_str(row["finished_at"]),
        terminal_reason=_as_optional_str(row["terminal_reason"]),
        terminal_detail=_as_json_object(_parse_json_field(row["terminal_detail_json"])),
        latest_metrics=_as_json_object(_parse_json_field(row["latest_metrics_json"])),
        latest_sample=_as_json_object(_parse_json_field(row["latest_sample_json"])),
        reconciled=bool(row["reconciled"]),
    )


def _append_job_event(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    event_type: str,
    source: str,
    payload: dict[str, Any],
) -> None:
    sequence_row = conn.execute(
        "SELECT COALESCE(MAX(sequence), 0) AS seq FROM run_job_events WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    sequence = int(sequence_row["seq"]) + 1 if sequence_row is not None else 1
    conn.execute(
        """
        INSERT INTO run_job_events (job_id, sequence, event_type, source, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, sequence, event_type, source, _safe_json_dumps(payload), _utc_now_iso()),
    )


def _append_job_log(conn: sqlite3.Connection, *, job_id: str, stream: str, line: str) -> None:
    row = conn.execute(
        "SELECT COALESCE(MAX(line_no), 0) AS line_no FROM run_job_logs WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    line_no = int(row["line_no"]) + 1 if row is not None else 1
    conn.execute(
        """
        INSERT INTO run_job_logs (job_id, line_no, stream, line, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (job_id, line_no, stream, line, _utc_now_iso()),
    )


def _heartbeat_is_fresh(value: str | None) -> bool:
    if value is None:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return datetime.now(UTC) - parsed.astimezone(UTC) <= _HEARTBEAT_STALE_AFTER


def _sample_to_payload(sample: ResourceSample) -> dict[str, Any]:
    return {
        "process_cpu_percent": sample.process_cpu_percent,
        "process_rss_gb": sample.process_rss_gb,
        "host_cpu_percent": sample.host_cpu_percent,
        "host_ram_available_gb": sample.host_ram_available_gb,
        "host_ram_used_gb": sample.host_ram_used_gb,
        "host_gpu_percent": sample.host_gpu_percent,
        "host_gpu_mem_used_gb": sample.host_gpu_mem_used_gb,
        "scope": sample.scope,
        "status": sample.status,
    }


def _parse_json_field(value: object) -> object:
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {}
    return value


def _as_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _as_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _connect_rw(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=3.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_safe_run_id(run_id: str | None) -> str:
    if run_id is None or not _SAFE_ID.match(run_id):
        raise StoreError(f"store_run_id_invalid:{run_id}")
    return run_id


def current_host_name() -> str:
    """Return current host name for local lifecycle rows."""

    return socket.gethostname()


__all__ = [
    "current_host_name",
    "get_run_lifecycle",
    "initialize_run_lifecycle",
    "is_cancel_requested",
    "reconcile_run_lifecycles",
    "request_run_cancel",
    "rewrite_runtime_snapshot",
    "update_run_lifecycle_metrics",
    "update_run_lifecycle_sample",
    "update_run_lifecycle_stage",
    "update_run_lifecycle_status",
]
