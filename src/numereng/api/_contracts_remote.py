"""Remote SSH ops public contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from numereng.api._contracts_base import PostTrainingScoringPolicy, TrainingProfile


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


class RemoteVizBootstrapRequest(BaseModel):
    store_root: str = ".numereng"


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


class RemoteRepoSyncRequest(BaseModel):
    target_id: str
    store_root: str = ".numereng"


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


class RemoteExperimentSyncRequest(BaseModel):
    target_id: str
    experiment_id: str
    store_root: str = ".numereng"


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


class RemoteExperimentPullRequest(BaseModel):
    target_id: str
    experiment_id: str
    store_root: str = ".numereng"


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
    materialized_run_ids: list[str] = Field(default_factory=list)
    materialized_run_count: int
    skipped_non_finished_run_ids: list[str] = Field(default_factory=list)
    failures: list[RemoteExperimentPullFailureResponse] = Field(default_factory=list)


class RemoteConfigPushRequest(BaseModel):
    target_id: str
    config_path: str
    store_root: str = ".numereng"


class RemoteConfigPushResponse(BaseModel):
    target_id: str
    local_config_path: str
    remote_config_path: str
    synced_at: str


class RemoteTrainLaunchRequest(BaseModel):
    target_id: str
    config_path: str
    experiment_id: str | None = None
    sync_repo: Literal["auto", "always", "never"] = "auto"
    profile: TrainingProfile | None = None
    post_training_scoring: PostTrainingScoringPolicy | None = None
    store_root: str = ".numereng"


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
    "RemoteExperimentPullFailureResponse",
    "RemoteExperimentPullRequest",
    "RemoteExperimentPullResponse",
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
