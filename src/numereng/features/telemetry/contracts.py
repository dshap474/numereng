"""Telemetry contracts for local run lifecycle persistence."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LocalRunTelemetrySession:
    """Mutable in-memory session state for one local telemetry job."""

    store_root: Path
    db_path: Path
    job_id: str
    batch_id: str
    logical_run_id: str
    attempt_id: str
    attempt_no: int
    source: str
    operation_type: str
    job_type: str
    experiment_id: str | None
    config_id: str
    config_source: str
    config_path: str
    config_sha256: str
    queue_name: str
    priority: int
    backend: str | None
    run_id: str
    run_hash: str
    config_hash: str
    run_dir: Path
    runtime_path: Path
    host: str
    created_at: str
    _event_sequence: int = 0
    _log_line_no: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def next_event_sequence(self) -> int:
        """Reserve and return next event sequence value."""

        with self._lock:
            self._event_sequence += 1
            return self._event_sequence

    def next_log_line_no(self) -> int:
        """Reserve and return next log line number."""

        with self._lock:
            self._log_line_no += 1
            return self._log_line_no


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One normalized local resource sample payload."""

    process_cpu_percent: float | None
    process_rss_gb: float | None
    host_cpu_percent: float | None
    host_ram_available_gb: float | None
    host_ram_used_gb: float | None
    host_gpu_percent: float | None
    host_gpu_mem_used_gb: float | None
    scope: str
    status: str


@dataclass(frozen=True, slots=True)
class LocalResourceSampler:
    """Background sampler state."""

    stop_event: threading.Event
    thread: threading.Thread


@dataclass(frozen=True, slots=True)
class RunLifecycleRecord:
    """Current lifecycle summary for one canonical local run."""

    run_id: str
    run_hash: str
    config_hash: str
    job_id: str
    logical_run_id: str
    attempt_id: str
    attempt_no: int
    source: str
    operation_type: str
    job_type: str
    status: str
    experiment_id: str | None
    config_id: str
    config_source: str
    config_path: str
    config_sha256: str
    run_dir: str
    runtime_path: str
    backend: str | None
    worker_id: str | None
    pid: int | None
    host: str | None
    current_stage: str | None
    completed_stages: tuple[str, ...]
    progress_percent: float | None
    progress_label: str | None
    progress_current: int | None
    progress_total: int | None
    cancel_requested: bool
    cancel_requested_at: str | None
    created_at: str
    queued_at: str | None
    started_at: str | None
    last_heartbeat_at: str | None
    updated_at: str
    finished_at: str | None
    terminal_reason: str | None
    terminal_detail: dict[str, Any]
    latest_metrics: dict[str, Any]
    latest_sample: dict[str, Any]
    reconciled: bool


@dataclass(frozen=True, slots=True)
class RunCancelResult:
    """Result payload for one cooperative cancel request."""

    run_id: str
    job_id: str
    status: str
    cancel_requested: bool
    cancel_requested_at: str | None
    accepted: bool


@dataclass(frozen=True, slots=True)
class RunLifecycleRepairResult:
    """Result payload for one lifecycle reconciliation pass."""

    store_root: Path
    scanned_count: int
    unchanged_count: int
    reconciled_count: int
    reconciled_stale_count: int
    reconciled_canceled_count: int
    run_ids: tuple[str, ...]


__all__ = [
    "LocalResourceSampler",
    "LocalRunTelemetrySession",
    "ResourceSample",
    "RunCancelResult",
    "RunLifecycleRecord",
    "RunLifecycleRepairResult",
]
