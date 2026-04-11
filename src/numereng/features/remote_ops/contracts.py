"""Contracts for SSH-backed remote ops workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RemoteSyncPolicy = Literal["auto", "always", "never"]
RemoteBootstrapStatus = Literal["ready", "degraded"]


@dataclass(frozen=True)
class RemoteTargetRecord:
    """One enabled remote target exposed through the public API."""

    id: str
    label: str
    kind: str
    shell: str
    repo_root: str
    store_root: str
    runner_cmd: str
    python_cmd: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class RemoteDoctorResult:
    """Connectivity and readiness report for one target."""

    target: RemoteTargetRecord
    ok: bool
    checked_at: str
    remote_python_executable: str | None
    remote_cwd: str | None
    snapshot_ok: bool
    snapshot_source_kind: str | None
    snapshot_source_id: str | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RemoteRepoSyncResult:
    """Result of syncing the local git-visible working tree to one target repo."""

    target_id: str
    repo_root: str
    manifest_hash: str
    local_commit_sha: str | None
    dirty: bool
    synced_files: int
    deleted_files: int
    synced_at: str
    local_marker_path: Path
    remote_marker_path: str


@dataclass(frozen=True)
class RemoteExperimentSyncResult:
    """Result of syncing one experiment authoring bundle to one target store."""

    target_id: str
    experiment_id: str
    remote_experiment_dir: str
    manifest_hash: str
    synced_files: int
    deleted_files: int
    synced_at: str
    local_marker_path: Path
    remote_marker_path: str


@dataclass(frozen=True)
class RemoteExperimentPullFailure:
    """One run-level failure encountered during remote pullback."""

    run_id: str
    missing_files: tuple[str, ...]
    reason: str | None = None


@dataclass(frozen=True)
class RemoteExperimentPullResult:
    """Result of pulling one remote experiment into canonical local run storage."""

    target_id: str
    experiment_id: str
    local_experiment_manifest_path: Path
    local_runs_root: Path
    pulled_at: str
    already_materialized_run_ids: tuple[str, ...]
    materialized_run_ids: tuple[str, ...]
    materialized_run_count: int
    skipped_non_finished_run_ids: tuple[str, ...]
    failures: tuple[RemoteExperimentPullFailure, ...]


@dataclass(frozen=True)
class RemoteConfigPushResult:
    """Result of pushing one ad hoc config to the remote repo temp area."""

    target_id: str
    local_config_path: Path
    remote_config_path: str
    synced_at: str


@dataclass(frozen=True)
class RemoteTrainLaunchResult:
    """Detached remote training launch metadata."""

    target_id: str
    launch_id: str
    remote_config_path: str
    remote_log_path: str
    remote_metadata_path: str
    remote_pid: int
    launched_at: str
    sync_repo_policy: RemoteSyncPolicy
    repo_synced: bool
    experiment_synced: bool


@dataclass(frozen=True)
class RemoteExperimentLaunchResult:
    """Detached remote experiment-window launch metadata."""

    target_id: str
    experiment_id: str
    state_path: str
    launch_id: str
    remote_log_path: str
    remote_metadata_path: str
    remote_pid: int
    launched_at: str
    repo_synced: bool
    experiment_synced: bool


@dataclass(frozen=True)
class RemoteExperimentStatusResult:
    """Current remote experiment-window supervisor status."""

    target_id: str
    experiment_id: str
    state_path: str
    exists: bool
    phase: str | None
    current_index: int | None
    current_run_id: str | None
    current_config_path: str | None
    last_completed_row_index: int | None
    supervisor_pid: int | None
    supervisor_alive: bool
    active_worker_pid: int | None
    last_successful_heartbeat_at: str | None
    retry_count: int
    failure_classifier: str | None
    terminal_error: str | None
    raw_state: dict[str, object]


@dataclass(frozen=True)
class RemoteExperimentStopResult:
    """Result of stopping one remote experiment-window supervisor."""

    target_id: str
    experiment_id: str
    state_path: str
    stopped: bool
    supervisor_pid: int | None
    note: str | None = None


@dataclass(frozen=True)
class RemoteExperimentMaintainResult:
    """Result of maintaining one remote experiment-window supervisor."""

    target_id: str
    experiment_id: str
    state_path: str
    action: Literal["noop", "restarted", "terminal"]
    phase: str | None
    supervisor_pid: int | None
    note: str | None = None


@dataclass(frozen=True)
class RemoteVizBootstrapTargetResult:
    """Bootstrap state for one enabled remote source."""

    target: RemoteTargetRecord
    bootstrap_status: RemoteBootstrapStatus
    last_bootstrap_at: str
    last_bootstrap_error: str | None
    repo_synced: bool
    repo_sync_skipped: bool
    doctor_ok: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RemoteVizBootstrapResult:
    """Workspace-wide remote bootstrap summary used by viz startup."""

    store_root: Path
    state_path: Path
    bootstrapped_at: str
    ready_count: int
    degraded_count: int
    targets: tuple[RemoteVizBootstrapTargetResult, ...]


__all__ = [
    "RemoteBootstrapStatus",
    "RemoteConfigPushResult",
    "RemoteDoctorResult",
    "RemoteExperimentPullFailure",
    "RemoteExperimentPullResult",
    "RemoteExperimentLaunchResult",
    "RemoteExperimentMaintainResult",
    "RemoteExperimentStatusResult",
    "RemoteExperimentStopResult",
    "RemoteExperimentSyncResult",
    "RemoteRepoSyncResult",
    "RemoteSyncPolicy",
    "RemoteTargetRecord",
    "RemoteTrainLaunchResult",
    "RemoteVizBootstrapResult",
    "RemoteVizBootstrapTargetResult",
]
