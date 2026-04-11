"""Remote SSH ops public contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from numereng.api._contracts_base import PostTrainingScoringPolicy, TrainingProfile, WorkspaceBoundRequest


class RemoteTargetListRequest(BaseModel):
    pass


class RemoteTargetResponse(BaseModel):
    id: str
    label: str
    kind: str
    shell: str
    repo_root: str
    store_root: str
    runner_cmd: str
    python_cmd: str
    tags: list[str] = Field(default_factory=list)


class RemoteTargetListResponse(BaseModel):
    targets: list[RemoteTargetResponse] = Field(default_factory=list)


class RemoteDoctorRequest(BaseModel):
    target_id: str


class RemoteDoctorResponse(BaseModel):
    target: RemoteTargetResponse
    ok: bool
    checked_at: str
    remote_python_executable: str | None = None
    remote_cwd: str | None = None
    snapshot_ok: bool
    snapshot_source_kind: str | None = None
    snapshot_source_id: str | None = None
    issues: list[str] = Field(default_factory=list)


class RemoteVizBootstrapRequest(WorkspaceBoundRequest):
    pass


class RemoteVizBootstrapTargetResponse(BaseModel):
    target: RemoteTargetResponse
    bootstrap_status: Literal["ready", "degraded"]
    last_bootstrap_at: str
    last_bootstrap_error: str | None = None
    repo_synced: bool
    repo_sync_skipped: bool
    doctor_ok: bool
    issues: list[str] = Field(default_factory=list)


class RemoteVizBootstrapResponse(BaseModel):
    store_root: str
    state_path: str
    bootstrapped_at: str
    ready_count: int
    degraded_count: int
    targets: list[RemoteVizBootstrapTargetResponse] = Field(default_factory=list)


class RemoteRepoSyncRequest(WorkspaceBoundRequest):
    target_id: str


class RemoteRepoSyncResponse(BaseModel):
    target_id: str
    repo_root: str
    manifest_hash: str
    local_commit_sha: str | None = None
    dirty: bool
    synced_files: int
    deleted_files: int
    synced_at: str
    local_marker_path: str
    remote_marker_path: str


class RemoteExperimentSyncRequest(WorkspaceBoundRequest):
    target_id: str
    experiment_id: str


class RemoteExperimentSyncResponse(BaseModel):
    target_id: str
    experiment_id: str
    remote_experiment_dir: str
    manifest_hash: str
    synced_files: int
    deleted_files: int
    synced_at: str
    local_marker_path: str
    remote_marker_path: str


class RemoteExperimentPullRequest(WorkspaceBoundRequest):
    target_id: str
    experiment_id: str


class RemoteExperimentPullFailureResponse(BaseModel):
    run_id: str
    missing_files: list[str] = Field(default_factory=list)
    reason: str | None = None


class RemoteExperimentPullResponse(BaseModel):
    target_id: str
    experiment_id: str
    local_experiment_manifest_path: str
    local_runs_root: str
    pulled_at: str
    already_materialized_run_ids: list[str] = Field(default_factory=list)
    materialized_run_ids: list[str] = Field(default_factory=list)
    materialized_run_count: int
    skipped_non_finished_run_ids: list[str] = Field(default_factory=list)
    failures: list[RemoteExperimentPullFailureResponse] = Field(default_factory=list)


class RemoteExperimentLaunchRequest(WorkspaceBoundRequest):
    target_id: str
    experiment_id: str
    start_index: int = Field(default=1, ge=1)
    end_index: int | None = Field(default=None, ge=1)
    score_stage: Literal["post_training_core", "post_training_full"] = "post_training_core"
    sync_repo: Literal["auto", "always", "never"] = "auto"


class RemoteExperimentLaunchResponse(BaseModel):
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


class RemoteExperimentStatusRequest(WorkspaceBoundRequest):
    target_id: str
    experiment_id: str
    start_index: int = Field(default=1, ge=1)
    end_index: int | None = Field(default=None, ge=1)


class RemoteExperimentStatusResponse(BaseModel):
    target_id: str
    experiment_id: str
    state_path: str
    exists: bool
    phase: str | None = None
    current_index: int | None = None
    current_run_id: str | None = None
    current_config_path: str | None = None
    last_completed_row_index: int | None = None
    supervisor_pid: int | None = None
    supervisor_alive: bool
    active_worker_pid: int | None = None
    last_successful_heartbeat_at: str | None = None
    retry_count: int = 0
    failure_classifier: str | None = None
    terminal_error: str | None = None
    raw_state: dict[str, object] = Field(default_factory=dict)


class RemoteExperimentStopRequest(WorkspaceBoundRequest):
    target_id: str
    experiment_id: str
    start_index: int = Field(default=1, ge=1)
    end_index: int | None = Field(default=None, ge=1)


class RemoteExperimentStopResponse(BaseModel):
    target_id: str
    experiment_id: str
    state_path: str
    stopped: bool
    supervisor_pid: int | None = None
    note: str | None = None


class RemoteExperimentMaintainRequest(WorkspaceBoundRequest):
    target_id: str
    experiment_id: str
    start_index: int = Field(default=1, ge=1)
    end_index: int | None = Field(default=None, ge=1)


class RemoteExperimentMaintainResponse(BaseModel):
    target_id: str
    experiment_id: str
    state_path: str
    action: Literal["noop", "restarted", "terminal"]
    phase: str | None = None
    supervisor_pid: int | None = None
    note: str | None = None


class RemoteConfigPushRequest(WorkspaceBoundRequest):
    target_id: str
    config_path: str


class RemoteConfigPushResponse(BaseModel):
    target_id: str
    local_config_path: str
    remote_config_path: str
    synced_at: str


class RemoteTrainLaunchRequest(WorkspaceBoundRequest):
    target_id: str
    config_path: str
    experiment_id: str | None = None
    sync_repo: Literal["auto", "always", "never"] = "auto"
    profile: TrainingProfile | None = None
    post_training_scoring: PostTrainingScoringPolicy | None = None


class RemoteTrainLaunchResponse(BaseModel):
    target_id: str
    launch_id: str
    remote_config_path: str
    remote_log_path: str
    remote_metadata_path: str
    remote_pid: int
    launched_at: str
    sync_repo_policy: Literal["auto", "always", "never"]
    repo_synced: bool
    experiment_synced: bool


__all__ = [
    "RemoteConfigPushRequest",
    "RemoteConfigPushResponse",
    "RemoteDoctorRequest",
    "RemoteDoctorResponse",
    "RemoteExperimentLaunchRequest",
    "RemoteExperimentLaunchResponse",
    "RemoteExperimentMaintainRequest",
    "RemoteExperimentMaintainResponse",
    "RemoteExperimentPullFailureResponse",
    "RemoteExperimentPullRequest",
    "RemoteExperimentPullResponse",
    "RemoteExperimentStatusRequest",
    "RemoteExperimentStatusResponse",
    "RemoteExperimentStopRequest",
    "RemoteExperimentStopResponse",
    "RemoteExperimentSyncRequest",
    "RemoteExperimentSyncResponse",
    "RemoteRepoSyncRequest",
    "RemoteRepoSyncResponse",
    "RemoteTargetListRequest",
    "RemoteTargetListResponse",
    "RemoteTargetResponse",
    "RemoteVizBootstrapRequest",
    "RemoteVizBootstrapResponse",
    "RemoteVizBootstrapTargetResponse",
    "RemoteTrainLaunchRequest",
    "RemoteTrainLaunchResponse",
]
