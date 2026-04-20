from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import numereng.features.telemetry.service as telemetry_module
from numereng.features.telemetry import (
    ResourceSample,
    append_log_line,
    append_resource_sample,
    begin_local_training_session,
    emit_metric_event,
    emit_stage_event,
    get_run_lifecycle,
    mark_job_completed,
    mark_job_running,
    mark_job_starting,
    reconcile_run_lifecycles,
    request_run_cancel,
)


def test_begin_local_training_session_writes_queued_rows(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "run.json"
    config_path.write_text('{"model":{"type":"LGBMRegressor","params":{"n_estimators":10}}}', encoding="utf-8")

    session = begin_local_training_session(
        store_root=store_root,
        config_path=config_path,
        run_id="run-queued",
        run_hash="hash-run-queued",
        config_hash="cfg-hash",
        run_dir=store_root / "runs" / "run-queued",
        runtime_path=store_root / "runs" / "run-queued" / "runtime.json",
        source="cli.run.train",
        experiment_id="2026-02-22_test-exp",
        operation_type="run",
        job_type="run",
    )

    assert session is not None

    with sqlite3.connect(store_root / "numereng.db") as conn:
        job_row = conn.execute(
            (
                "SELECT status, experiment_id, operation_type, config_source, "
                "canonical_run_id, run_dir FROM run_jobs WHERE job_id = ?"
            ),
            (session.job_id,),
        ).fetchone()
        assert job_row is not None
        assert job_row[0] == "queued"
        assert job_row[1] == "2026-02-22_test-exp"
        assert job_row[2] == "run"
        assert job_row[3] == "external"
        assert job_row[4] == "run-queued"
        assert job_row[5] == str((store_root / "runs" / "run-queued").resolve())

        attempt_row = conn.execute(
            "SELECT status FROM run_attempts WHERE attempt_id = ?",
            (session.attempt_id,),
        ).fetchone()
        assert attempt_row is not None
        assert attempt_row[0] == "queued"

        logical_row = conn.execute(
            "SELECT status FROM logical_runs WHERE logical_run_id = ?",
            (session.logical_run_id,),
        ).fetchone()
        assert logical_row is not None
        assert logical_row[0] == "queued"

        events = conn.execute(
            "SELECT event_type FROM run_job_events WHERE job_id = ? ORDER BY id ASC",
            (session.job_id,),
        ).fetchall()
        assert [str(row[0]) for row in events] == ["job_queued"]

        lifecycle_row = conn.execute(
            """
            SELECT status, current_stage, progress_percent, progress_label, runtime_path
            FROM run_lifecycles
            WHERE run_id = ?
            """,
            ("run-queued",),
        ).fetchone()
        assert lifecycle_row is not None
        assert lifecycle_row[0] == "queued"
        assert lifecycle_row[1] == "initializing"
        assert lifecycle_row[2] == 0.0
        assert lifecycle_row[3] == "Queued"
        assert lifecycle_row[4].endswith("runtime.json")

    assert (store_root / "runs" / "run-queued" / "runtime.json").is_file()


def test_begin_local_training_session_infers_experiment_id_from_config_path(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = store_root / "experiments" / "2026-02-22_test-exp" / "configs" / "base.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}", encoding="utf-8")

    session = begin_local_training_session(
        store_root=store_root,
        config_path=config_path,
        run_id="run-inferred",
        run_hash="hash-run-inferred",
        config_hash="cfg-hash",
        run_dir=store_root / "runs" / "run-inferred",
        runtime_path=store_root / "runs" / "run-inferred" / "runtime.json",
        source="cli.run.train",
        experiment_id=None,
        operation_type="run",
        job_type="run",
    )
    assert session is not None
    assert session.experiment_id == "2026-02-22_test-exp"

    with sqlite3.connect(store_root / "numereng.db") as conn:
        job_row = conn.execute(
            "SELECT experiment_id FROM run_jobs WHERE job_id = ?",
            (session.job_id,),
        ).fetchone()
        assert job_row is not None
        assert job_row[0] == "2026-02-22_test-exp"


def test_local_telemetry_lifecycle_transitions_to_completed(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = store_root / "experiments" / "2026-02-22_test-exp" / "configs" / "base.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}", encoding="utf-8")

    session = begin_local_training_session(
        store_root=store_root,
        config_path=config_path,
        run_id="run-abc123",
        run_hash="hash-run-abc123",
        config_hash="cfg-hash",
        run_dir=store_root / "runs" / "run-abc123",
        runtime_path=store_root / "runs" / "run-abc123" / "runtime.json",
        source="cli.experiment.train",
        experiment_id="2026-02-22_test-exp",
        operation_type="run",
        job_type="run",
    )
    assert session is not None

    mark_job_starting(session, pid=4321, worker_id="local")
    mark_job_running(session)
    emit_stage_event(
        session,
        current_stage="persist_artifacts",
        completed_stages=["initializing", "load_data"],
    )
    emit_metric_event(session, metrics={"corr_sharpe": 1.23, "mmc_mean": 0.01})
    append_log_line(session, stream="stdout", line="model fit complete")
    append_resource_sample(
        session,
        sample=ResourceSample(
            process_cpu_percent=None,
            process_rss_gb=0.2,
            host_cpu_percent=None,
            host_ram_available_gb=4.0,
            host_ram_used_gb=12.0,
            host_gpu_percent=None,
            host_gpu_mem_used_gb=None,
            scope="launcher_wrapper_only",
            status="partial",
        ),
    )
    mark_job_completed(session, canonical_run_id="run-abc123", run_dir=str(store_root / "runs" / "run-abc123"))

    with sqlite3.connect(store_root / "numereng.db") as conn:
        job_row = conn.execute(
            "SELECT status, canonical_run_id, run_dir, exit_code FROM run_jobs WHERE job_id = ?",
            (session.job_id,),
        ).fetchone()
        assert job_row is not None
        assert job_row[0] == "completed"
        assert job_row[1] == "run-abc123"
        assert job_row[3] == 0

        attempt_row = conn.execute(
            "SELECT status, canonical_run_id FROM run_attempts WHERE attempt_id = ?",
            (session.attempt_id,),
        ).fetchone()
        assert attempt_row is not None
        assert attempt_row[0] == "completed"
        assert attempt_row[1] == "run-abc123"

        logical_row = conn.execute(
            "SELECT status FROM logical_runs WHERE logical_run_id = ?",
            (session.logical_run_id,),
        ).fetchone()
        assert logical_row is not None
        assert logical_row[0] == "completed"

        lifecycle_row = conn.execute(
            """
            SELECT
                status,
                current_stage,
                progress_percent,
                progress_label,
                progress_current,
                progress_total,
                finished_at,
                latest_metrics_json
            FROM run_lifecycles
            WHERE run_id = ?
            """,
            ("run-abc123",),
        ).fetchone()
        assert lifecycle_row is not None
        assert lifecycle_row[0] == "completed"
        assert lifecycle_row[1] == "persist_artifacts"
        assert lifecycle_row[2] == 100.0
        assert lifecycle_row[3] == "Completed"
        assert lifecycle_row[4] == 0
        assert lifecycle_row[5] is None
        assert lifecycle_row[6] is not None
        assert "corr_sharpe" in str(lifecycle_row[7])

        event_types = [
            str(row[0])
            for row in conn.execute(
                "SELECT event_type FROM run_job_events WHERE job_id = ? ORDER BY id ASC",
                (session.job_id,),
            ).fetchall()
        ]
        assert event_types == [
            "job_queued",
            "job_starting",
            "job_running",
            "stage_update",
            "metric_update",
            "job_completed",
        ]

        log_count = conn.execute(
            "SELECT COUNT(*) FROM run_job_logs WHERE job_id = ?",
            (session.job_id,),
        ).fetchone()
        assert log_count is not None
        assert int(log_count[0]) >= 2

        sample_count = conn.execute(
            "SELECT COUNT(*) FROM run_job_samples WHERE job_id = ?",
            (session.job_id,),
        ).fetchone()
        assert sample_count is not None
        assert int(sample_count[0]) >= 2


def test_emit_stage_event_updates_lifecycle_progress_fields(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    session = begin_local_training_session(
        store_root=store_root,
        config_path=config_path,
        run_id="run-progress",
        run_hash="hash-run-progress",
        config_hash="cfg-hash",
        run_dir=store_root / "runs" / "run-progress",
        runtime_path=store_root / "runs" / "run-progress" / "runtime.json",
        source="cli.run.train",
        experiment_id=None,
        operation_type="run",
        job_type="run",
    )
    assert session is not None

    emit_stage_event(
        session,
        current_stage="train_model",
        completed_stages=["initializing", "load_data"],
        extra_payload={
            "progress_percent": 42.5,
            "progress_label": "Fold 2 of 4",
            "progress_current": 1,
            "progress_total": 4,
        },
    )

    with sqlite3.connect(store_root / "numereng.db") as conn:
        lifecycle_row = conn.execute(
            """
            SELECT current_stage, progress_percent, progress_label, progress_current, progress_total
            FROM run_lifecycles
            WHERE run_id = ?
            """,
            ("run-progress",),
        ).fetchone()
        assert lifecycle_row is not None
        assert lifecycle_row[0] == "train_model"
        assert lifecycle_row[1] == 42.5
        assert lifecycle_row[2] == "Fold 2 of 4"
        assert lifecycle_row[3] == 1
        assert lifecycle_row[4] == 4


def test_telemetry_helpers_fail_open_on_sql_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    session = begin_local_training_session(
        store_root=store_root,
        config_path=config_path,
        run_id="run-fail-open",
        run_hash="hash-run-fail-open",
        config_hash="cfg-hash",
        run_dir=store_root / "runs" / "run-fail-open",
        runtime_path=store_root / "runs" / "run-fail-open" / "runtime.json",
        source="cli.run.train",
        experiment_id=None,
        operation_type="run",
        job_type="run",
    )
    assert session is not None

    def _raise_connect(db_path: Path) -> sqlite3.Connection:
        _ = db_path
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(telemetry_module, "_connect_rw", _raise_connect)

    # Fail-open: none of these should raise.
    mark_job_starting(session, pid=100)
    mark_job_running(session)
    append_log_line(session, stream="stdout", line="still running")
    emit_metric_event(session, metrics={"corr_sharpe": 1.0})


def test_request_run_cancel_updates_lifecycle_and_runtime_snapshot(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    session = begin_local_training_session(
        store_root=store_root,
        config_path=config_path,
        run_id="run-cancel",
        run_hash="hash-run-cancel",
        config_hash="cfg-hash",
        run_dir=store_root / "runs" / "run-cancel",
        runtime_path=store_root / "runs" / "run-cancel" / "runtime.json",
        source="cli.run.train",
        experiment_id=None,
        operation_type="run",
        job_type="run",
    )
    assert session is not None

    result = request_run_cancel(store_root=store_root, run_id="run-cancel")

    assert result.run_id == "run-cancel"
    assert result.accepted is True
    lifecycle = get_run_lifecycle(store_root=store_root, run_id="run-cancel")
    assert lifecycle is not None
    assert lifecycle.cancel_requested is True
    assert lifecycle.cancel_requested_at is not None

    runtime_payload = (store_root / "runs" / "run-cancel" / "runtime.json").read_text(encoding="utf-8")
    assert '"cancel_requested": true' in runtime_payload


def test_reconcile_run_lifecycles_marks_cancel_requested_run_canceled(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    session = begin_local_training_session(
        store_root=store_root,
        config_path=config_path,
        run_id="run-reconcile-cancel",
        run_hash="hash-run-reconcile-cancel",
        config_hash="cfg-hash",
        run_dir=store_root / "runs" / "run-reconcile-cancel",
        runtime_path=store_root / "runs" / "run-reconcile-cancel" / "runtime.json",
        source="cli.run.train",
        experiment_id=None,
        operation_type="run",
        job_type="run",
    )
    assert session is not None
    mark_job_starting(session, pid=999999, worker_id="local")
    mark_job_running(session)
    request_run_cancel(store_root=store_root, run_id="run-reconcile-cancel")

    result = reconcile_run_lifecycles(store_root=store_root, run_id="run-reconcile-cancel", active_only=True)

    assert result.reconciled_count == 1
    assert result.reconciled_canceled_count == 1
    lifecycle = get_run_lifecycle(store_root=store_root, run_id="run-reconcile-cancel")
    assert lifecycle is not None
    assert lifecycle.status == "canceled"
    assert lifecycle.reconciled is True


def test_reconcile_run_lifecycles_marks_orphaned_run_stale(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    session = begin_local_training_session(
        store_root=store_root,
        config_path=config_path,
        run_id="run-reconcile-stale",
        run_hash="hash-run-reconcile-stale",
        config_hash="cfg-hash",
        run_dir=store_root / "runs" / "run-reconcile-stale",
        runtime_path=store_root / "runs" / "run-reconcile-stale" / "runtime.json",
        source="cli.run.train",
        experiment_id=None,
        operation_type="run",
        job_type="run",
    )
    assert session is not None
    mark_job_starting(session, pid=999999, worker_id="local")
    mark_job_running(session)

    result = reconcile_run_lifecycles(store_root=store_root, run_id="run-reconcile-stale", active_only=True)

    assert result.reconciled_count == 1
    assert result.reconciled_stale_count == 1
    lifecycle = get_run_lifecycle(store_root=store_root, run_id="run-reconcile-stale")
    assert lifecycle is not None
    assert lifecycle.status == "stale"
    assert lifecycle.reconciled is True
