"""Telemetry contracts for local run lifecycle persistence."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path


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


__all__ = [
    "LocalResourceSampler",
    "LocalRunTelemetrySession",
    "ResourceSample",
]
