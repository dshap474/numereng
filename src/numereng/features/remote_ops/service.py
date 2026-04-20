"""SSH-backed remote sync and detached launch workflows."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from numereng.features.experiments import resolve_experiment_run_plan_state_path
from numereng.features.remote_ops.bootstrap_state import (
    load_viz_bootstrap_state,
    remote_viz_bootstrap_state_path,
    write_viz_bootstrap_state,
)
from numereng.features.remote_ops.contracts import (
    RemoteBootstrapStatus,
    RemoteConfigPushResult,
    RemoteDoctorResult,
    RemoteExperimentLaunchResult,
    RemoteExperimentMaintainResult,
    RemoteExperimentPullFailure,
    RemoteExperimentPullResult,
    RemoteExperimentStatusResult,
    RemoteExperimentStopResult,
    RemoteExperimentSyncResult,
    RemoteRepoSyncResult,
    RemoteSyncPolicy,
    RemoteTargetRecord,
    RemoteTrainLaunchResult,
    RemoteVizBootstrapResult,
    RemoteVizBootstrapTargetResult,
)
from numereng.features.remote_ops.sync import (
    SyncEntry,
    build_experiment_marker_paths,
    build_repo_marker_paths,
    remote_config_destination,
    remote_experiment_metadata_path,
    remote_repo_metadata_path,
    sync_entries_to_remote,
)
from numereng.features.store import (
    index_run,
    init_store_db,
    resolve_store_root,
    resolve_workspace_layout_from_store_root,
    upsert_experiment,
)
from numereng.features.training import PostTrainingScoringPolicy, TrainingProfile
from numereng.platform import load_remote_targets
from numereng.platform.remotes.contracts import SshRemoteTargetProfile
from numereng.platform.remotes.ssh import (
    build_monitor_snapshot_command,
    build_remote_python_command,
    build_remote_shell_command,
    build_ssh_command,
    powershell_single_quote,
    remote_path_join,
    scp_base_command,
    ssh_destination,
)
from numereng.platform.run_execution import build_run_execution, serialize_run_execution

_SYNC_TIMEOUT_SECONDS = 180
_REMOTE_WINDOWS_STARTUP_TIMEOUT_SECONDS = 10.0
_REMOTE_WINDOWS_STARTUP_POLL_SECONDS = 0.25
_REMOTE_PULL_STAGING_DIR = ("tmp", "remote_ops", "pulls")
_REMOTE_PULL_COPY_TIMEOUT_SECONDS = 900
_FINISHED_RUN_REQUIRED_LOCAL_FILES: tuple[str, ...] = (
    "run.json",
    "resolved.json",
    "results.json",
    "metrics.json",
)

# Scoring-mode pull copies only the files the dashboard needs to render the
# Performance tab — scoring parquet artifacts plus the root manifest/log files.
# Predictions parquets (~138 MB per run) are intentionally skipped.
_SCORING_MODE_ROOT_REQUIRED: tuple[str, ...] = (
    "run.json",
    "resolved.json",
    "results.json",
    "metrics.json",
)
_SCORING_MODE_ROOT_OPTIONAL: tuple[str, ...] = (
    "run.log",
    "score_provenance.json",
)
_SCORING_MODE_SUBDIR: tuple[str, ...] = ("artifacts", "scoring")
_PREDICTIONS_SUBDIR: tuple[str, ...] = ("artifacts", "predictions")
PullMode = Literal["scoring", "full"]


class RemoteOpsError(Exception):
    """Base class for remote ops failures."""


class RemoteTargetNotFoundError(RemoteOpsError):
    """Raised when the requested target profile does not exist."""


class RemoteValidationError(RemoteOpsError):
    """Raised when local inputs or target state are invalid."""


class RemoteExecutionError(RemoteOpsError):
    """Raised when a remote SSH command or sync transport fails."""


def list_remote_targets() -> tuple[RemoteTargetRecord, ...]:
    """List all enabled remote target profiles."""

    return tuple(_to_target_record(target) for target in load_remote_targets(strict=True))


def bootstrap_viz_remotes(
    *,
    store_root: str | Path = ".numereng",
) -> RemoteVizBootstrapResult:
    """Sync and doctor all enabled remotes ahead of local viz startup."""

    resolved_store_root = resolve_store_root(store_root)
    state_path = remote_viz_bootstrap_state_path(store_root=resolved_store_root)
    bootstrapped_at = _utc_now_iso()
    results: list[RemoteVizBootstrapTargetResult] = []

    for target in load_remote_targets():
        repo_synced = False
        repo_sync_skipped = False
        doctor_ok = False
        issues: list[str] = []
        last_error: str | None = None

        try:
            if _should_sync_repo(target.id, resolved_store_root):
                sync_remote_repo(target_id=target.id, store_root=resolved_store_root)
                repo_synced = True
            else:
                repo_sync_skipped = True
        except Exception as exc:
            issues.append("repo_sync_failed")
            last_error = str(exc)

        try:
            doctor = doctor_remote_target(target_id=target.id)
            doctor_ok = doctor.ok
            issues.extend(issue for issue in doctor.issues if issue not in issues)
            if not doctor.ok and last_error is None:
                last_error = _doctor_failure_message(doctor.issues)
        except Exception as exc:
            issues.append("doctor_failed")
            if last_error is None:
                last_error = str(exc)

        bootstrap_status: RemoteBootstrapStatus = "ready" if doctor_ok and last_error is None else "degraded"
        results.append(
            RemoteVizBootstrapTargetResult(
                target=_to_target_record(target),
                bootstrap_status=bootstrap_status,
                last_bootstrap_at=bootstrapped_at,
                last_bootstrap_error=None if bootstrap_status == "ready" else last_error,
                repo_synced=repo_synced,
                repo_sync_skipped=repo_sync_skipped,
                doctor_ok=doctor_ok,
                issues=tuple(issues),
            )
        )

    payload = RemoteVizBootstrapResult(
        store_root=resolved_store_root,
        state_path=state_path,
        bootstrapped_at=bootstrapped_at,
        ready_count=sum(1 for item in results if item.bootstrap_status == "ready"),
        degraded_count=sum(1 for item in results if item.bootstrap_status == "degraded"),
        targets=tuple(results),
    )
    write_viz_bootstrap_state(payload)
    return payload


def doctor_remote_target(*, target_id: str) -> RemoteDoctorResult:
    """Verify one remote target is reachable and numereng-ready."""

    target = _get_target(target_id)
    checked_at = _utc_now_iso()
    issues: list[str] = []

    python_payload = _run_remote_python(target, _doctor_python_script(), args=[])
    if python_payload is None:
        issues.append("remote_python_unavailable")

    snapshot_payload = _run_remote_snapshot(target)
    if snapshot_payload is None:
        issues.append("monitor_snapshot_failed")

    return RemoteDoctorResult(
        target=_to_target_record(target),
        ok=not issues,
        checked_at=checked_at,
        remote_python_executable=_optional_str(python_payload, "python_executable"),
        remote_cwd=_optional_str(python_payload, "cwd"),
        snapshot_ok=snapshot_payload is not None,
        snapshot_source_kind=_optional_nested_str(snapshot_payload, "source", "kind"),
        snapshot_source_id=_optional_nested_str(snapshot_payload, "source", "id"),
        issues=tuple(issues),
    )


def sync_remote_repo(
    *,
    target_id: str,
    store_root: str | Path = ".numereng",
) -> RemoteRepoSyncResult:
    """Sync the local git-visible working tree to the remote repo root."""

    target = _get_target(target_id)
    resolved_store_root = resolve_store_root(store_root)
    repo_root = _repository_root()
    entries = _repo_sync_entries(repo_root)
    commit_sha = _git_head_sha(repo_root)
    dirty = _git_is_dirty(repo_root)
    local_marker_path, _ = build_repo_marker_paths(store_root=resolved_store_root, target_id=target.id)
    remote_marker_path = remote_repo_metadata_path(target, target_id=target.id)

    outcome = sync_entries_to_remote(
        target=target,
        entries=entries,
        remote_root=target.repo_root,
        scope="repo",
        local_marker_path=local_marker_path,
        remote_marker_path=remote_marker_path,
        local_commit_sha=commit_sha,
        dirty=dirty,
        timeout_seconds=max(target.command_timeout_seconds, _SYNC_TIMEOUT_SECONDS),
    )
    return RemoteRepoSyncResult(
        target_id=target.id,
        repo_root=target.repo_root,
        manifest_hash=outcome.manifest_hash,
        local_commit_sha=outcome.local_commit_sha,
        dirty=outcome.dirty,
        synced_files=outcome.synced_files,
        deleted_files=outcome.deleted_files,
        synced_at=outcome.synced_at,
        local_marker_path=outcome.local_marker_path or local_marker_path,
        remote_marker_path=outcome.remote_marker_path or remote_marker_path,
    )


def sync_remote_experiment(
    *,
    target_id: str,
    experiment_id: str,
    store_root: str | Path = ".numereng",
) -> RemoteExperimentSyncResult:
    """Sync one experiment authoring bundle to the remote store."""

    target = _get_target(target_id)
    resolved_store_root = resolve_store_root(store_root)
    _ensure_remote_experiment_created(
        target=target,
        experiment_id=experiment_id,
        store_root=resolved_store_root,
    )
    entries = _experiment_sync_entries(resolved_store_root, experiment_id=experiment_id)
    local_marker_path, _ = build_experiment_marker_paths(
        store_root=resolved_store_root,
        target_id=target.id,
        experiment_id=experiment_id,
    )
    remote_marker_path = remote_experiment_metadata_path(
        target,
        target_id=target.id,
        experiment_id=experiment_id,
    )
    outcome = sync_entries_to_remote(
        target=target,
        entries=entries,
        remote_root=target.repo_root,
        scope=f"experiment:{experiment_id}",
        local_marker_path=local_marker_path,
        remote_marker_path=remote_marker_path,
        local_commit_sha=_git_head_sha(_repository_root()),
        dirty=_git_is_dirty(_repository_root()),
        timeout_seconds=max(target.command_timeout_seconds, _SYNC_TIMEOUT_SECONDS),
    )
    return RemoteExperimentSyncResult(
        target_id=target.id,
        experiment_id=experiment_id,
        remote_experiment_dir=remote_path_join(target, target.store_root, "experiments", experiment_id),
        manifest_hash=outcome.manifest_hash,
        synced_files=outcome.synced_files,
        deleted_files=outcome.deleted_files,
        synced_at=outcome.synced_at,
        local_marker_path=outcome.local_marker_path or local_marker_path,
        remote_marker_path=outcome.remote_marker_path or remote_marker_path,
    )


def pull_remote_experiment(
    *,
    target_id: str,
    experiment_id: str,
    mode: PullMode,
    store_root: str | Path = ".numereng",
) -> RemoteExperimentPullResult:
    """Pull one remote experiment's finished runs into canonical local storage.

    ``mode`` selects which files are copied:

    - ``"scoring"``: copies ``artifacts/scoring/`` + the root manifest/log files.
      Enables the dashboard Performance tab without paying the cost of
      prediction parquets (~138 MB/run). Cannot be used for submit/ensemble/
      rescore locally.
    - ``"full"``: copies the entire run directory including
      ``artifacts/predictions/``. Required for submit/ensemble/rescore.
    """

    if mode not in ("scoring", "full"):
        raise RemoteValidationError(f"remote_experiment_pull_invalid_mode:{mode}")

    target = _get_target(target_id)
    init_result = init_store_db(store_root=store_root)
    resolved_store_root = init_result.store_root
    local_layout = resolve_workspace_layout_from_store_root(resolved_store_root)
    local_runs_root = resolved_store_root / "runs"
    local_experiment_manifest_path = local_layout.experiments_root / experiment_id / "experiment.json"

    preflight = _run_remote_python(
        target,
        _pull_preflight_python_script(),
        args=[experiment_id, target.store_root],
        timeout_seconds=max(target.command_timeout_seconds, 300),
    )
    if preflight is None:
        raise RemoteExecutionError("remote_experiment_pull_preflight_failed")
    preflight_error = _optional_str(preflight, "error")
    if preflight_error is not None:
        raise RemoteExecutionError(preflight_error)

    remote_manifest = preflight.get("experiment")
    if not isinstance(remote_manifest, dict):
        raise RemoteExecutionError("remote_experiment_pull_missing_manifest")

    pulled_at = _utc_now_iso()
    failures = _decode_pull_failures(preflight.get("failures"))
    materializable_run_ids = tuple(
        str(item) for item in preflight.get("eligible_finished_run_ids", []) if isinstance(item, str) and item.strip()
    )
    skipped_non_finished_run_ids = tuple(
        str(item)
        for item in preflight.get("skipped_non_finished_run_ids", [])
        if isinstance(item, str) and item.strip()
    )
    already_materialized_run_ids: tuple[str, ...] = ()
    upgrade_run_ids: tuple[str, ...] = ()
    run_ids_to_copy = materializable_run_ids

    if not failures:
        existing_state = _classify_local_finished_runs(
            local_runs_root=local_runs_root,
            run_ids=materializable_run_ids,
            requested_mode=mode,
        )
        already_materialized_run_ids = existing_state["already_materialized_run_ids"]
        run_ids_to_copy = existing_state["run_ids_to_copy"]
        upgrade_run_ids = existing_state["upgrade_run_ids"]
        failures = existing_state["failures"]
    if failures:
        return RemoteExperimentPullResult(
            target_id=target.id,
            experiment_id=experiment_id,
            pull_mode=mode,
            local_experiment_manifest_path=local_experiment_manifest_path,
            local_runs_root=local_runs_root,
            pulled_at=pulled_at,
            already_materialized_run_ids=already_materialized_run_ids,
            materialized_run_ids=(),
            partially_materialized_run_ids=(),
            materialized_run_count=0,
            skipped_non_finished_run_ids=skipped_non_finished_run_ids,
            failures=failures,
        )

    materialized_run_ids: tuple[str, ...] = ()
    if run_ids_to_copy:
        staging_root = _remote_pull_staging_dir(resolved_store_root, experiment_id=experiment_id, mode=mode)
        try:
            extracted_runs_root = _copy_remote_runs_to_staging(
                target=target,
                remote_store_root=target.store_root,
                staging_root=staging_root,
                run_ids=run_ids_to_copy,
                mode=mode,
            )
            _ensure_staged_runs_present(extracted_runs_root=extracted_runs_root, run_ids=run_ids_to_copy)
            materialized_run_ids = _materialize_staged_runs(
                store_root=resolved_store_root,
                staging_runs_root=extracted_runs_root,
                run_ids=run_ids_to_copy,
                upgrade_run_ids=set(upgrade_run_ids),
            )
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    effective_materialized_run_ids = tuple([*already_materialized_run_ids, *materialized_run_ids])
    local_experiment_manifest_path = _reconcile_local_experiment_manifest(
        store_root=resolved_store_root,
        target_id=target.id,
        experiment_id=experiment_id,
        remote_manifest=remote_manifest,
        pulled_at=pulled_at,
        materialized_run_ids=effective_materialized_run_ids,
    )
    _reindex_materialized_runs(store_root=resolved_store_root, run_ids=effective_materialized_run_ids)
    return RemoteExperimentPullResult(
        target_id=target.id,
        experiment_id=experiment_id,
        pull_mode=mode,
        local_experiment_manifest_path=local_experiment_manifest_path,
        local_runs_root=local_runs_root,
        pulled_at=pulled_at,
        already_materialized_run_ids=already_materialized_run_ids,
        materialized_run_ids=materialized_run_ids,
        partially_materialized_run_ids=upgrade_run_ids,
        materialized_run_count=len(materialized_run_ids),
        skipped_non_finished_run_ids=skipped_non_finished_run_ids,
        failures=(),
    )


def push_remote_config(
    *,
    target_id: str,
    config_path: str | Path,
    store_root: str | Path = ".numereng",
) -> RemoteConfigPushResult:
    """Push one ad hoc config file into the remote repo temp area."""

    _ = resolve_store_root(store_root)
    target = _get_target(target_id)
    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.is_file():
        raise RemoteValidationError(f"remote_config_not_found:{resolved_config_path}")
    remote_file_name = f"{_timestamp_token()}_{resolved_config_path.name}"
    remote_relpath = "/".join([".numereng", "tmp", "remote-configs", remote_file_name])
    remote_abs_path = remote_config_destination(target, file_name=remote_file_name)
    outcome = sync_entries_to_remote(
        target=target,
        entries=[SyncEntry(local_path=resolved_config_path, remote_relpath=remote_relpath)],
        remote_root=target.repo_root,
        scope=f"config:{remote_file_name}",
        local_marker_path=None,
        remote_marker_path=None,
        local_commit_sha=_git_head_sha(_repository_root()),
        dirty=_git_is_dirty(_repository_root()),
        timeout_seconds=max(target.command_timeout_seconds, _SYNC_TIMEOUT_SECONDS),
    )
    return RemoteConfigPushResult(
        target_id=target.id,
        local_config_path=resolved_config_path,
        remote_config_path=remote_abs_path,
        synced_at=outcome.synced_at,
    )


def remote_run_train(
    *,
    target_id: str,
    config_path: str | Path,
    experiment_id: str | None = None,
    sync_repo: RemoteSyncPolicy = "auto",
    profile: TrainingProfile | None = None,
    post_training_scoring: PostTrainingScoringPolicy | None = None,
    store_root: str | Path = ".numereng",
) -> RemoteTrainLaunchResult:
    """Launch a detached remote numereng training process."""

    target = _get_target(target_id)
    resolved_store_root = resolve_store_root(store_root)
    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.is_file():
        raise RemoteValidationError(f"remote_config_not_found:{resolved_config_path}")

    repo_synced = False
    if sync_repo == "always" or (sync_repo == "auto" and _should_sync_repo(target.id, resolved_store_root)):
        sync_remote_repo(target_id=target.id, store_root=resolved_store_root)
        repo_synced = True

    inferred_experiment_id = experiment_id or _infer_experiment_id(resolved_config_path, resolved_store_root)
    experiment_synced = False
    if inferred_experiment_id is not None and _is_experiment_config(
        resolved_config_path,
        resolved_store_root,
        inferred_experiment_id,
    ):
        sync_remote_experiment(
            target_id=target.id,
            experiment_id=inferred_experiment_id,
            store_root=resolved_store_root,
        )
        experiment_synced = True
        remote_config_path = str(_remote_store_relative_path(target, resolved_store_root, resolved_config_path))
    else:
        remote_config_path = push_remote_config(
            target_id=target.id,
            config_path=resolved_config_path,
            store_root=resolved_store_root,
        ).remote_config_path

    launch_id = _launch_id_for_config(resolved_config_path)
    remote_log_path = remote_path_join(target, target.store_root, "remote_ops", "launches", f"{launch_id}.log")
    remote_metadata_path = remote_path_join(target, target.store_root, "remote_ops", "launches", f"{launch_id}.json")
    command = [
        *_split_target_command(target.runner_cmd, shell=target.shell),
        "run",
        "train",
        "--config",
        remote_config_path,
    ]
    if inferred_experiment_id is not None:
        command.extend(["--experiment-id", inferred_experiment_id])
    if profile is not None:
        command.extend(["--profile", profile])
    if post_training_scoring is not None:
        command.extend(["--post-training-scoring", post_training_scoring])
    execution_payload = serialize_run_execution(
        build_run_execution(
            kind="remote_host",
            provider="ssh",
            backend="remote_pc",
            target_id=target.id,
            host=target.label,
        )
    )

    launch_payload = _run_remote_python(
        target,
        _launch_python_script(),
        args=[
            base64.b64encode(json.dumps(command).encode("utf-8")).decode("ascii"),
            target.repo_root,
            remote_log_path,
            remote_metadata_path,
            launch_id,
            base64.b64encode(execution_payload.encode("utf-8")).decode("ascii"),
        ],
    )
    if launch_payload is None:
        raise RemoteExecutionError("remote_train_launch_failed")
    if not bool(launch_payload.get("ok", True)):
        raise RemoteExecutionError(_remote_launch_error_message(launch_payload))

    return RemoteTrainLaunchResult(
        target_id=target.id,
        launch_id=launch_id,
        remote_config_path=remote_config_path,
        remote_log_path=remote_log_path,
        remote_metadata_path=remote_metadata_path,
        remote_pid=int(launch_payload["pid"]),
        launched_at=str(launch_payload["launched_at"]),
        sync_repo_policy=sync_repo,
        repo_synced=repo_synced,
        experiment_synced=experiment_synced,
    )


def remote_launch_experiment(
    *,
    target_id: str,
    experiment_id: str,
    start_index: int = 1,
    end_index: int | None = None,
    score_stage: Literal["post_training_core", "post_training_full"] = "post_training_core",
    sync_repo: RemoteSyncPolicy = "auto",
    store_root: str | Path = ".numereng",
) -> RemoteExperimentLaunchResult:
    """Launch one detached remote experiment run-plan window."""

    target = _get_target(target_id)
    resolved_store_root = resolve_store_root(store_root)
    resolved_end_index = _resolve_effective_end_index(
        store_root=resolved_store_root,
        experiment_id=experiment_id,
        end_index=end_index,
    )

    repo_synced = False
    if sync_repo == "always" or (sync_repo == "auto" and _should_sync_repo(target.id, resolved_store_root)):
        sync_remote_repo(target_id=target.id, store_root=resolved_store_root)
        repo_synced = True
    sync_remote_experiment(target_id=target.id, experiment_id=experiment_id, store_root=resolved_store_root)

    state_path = _remote_run_plan_state_path(
        target=target,
        store_root=resolved_store_root,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=resolved_end_index,
    )
    launch = _launch_remote_experiment_window(
        target=target,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=resolved_end_index,
        score_stage=score_stage,
        resume=False,
    )

    return RemoteExperimentLaunchResult(
        target_id=target.id,
        experiment_id=experiment_id,
        state_path=state_path,
        launch_id=launch["launch_id"],
        remote_log_path=launch["remote_log_path"],
        remote_metadata_path=launch["remote_metadata_path"],
        remote_pid=launch["pid"],
        launched_at=launch["launched_at"],
        repo_synced=repo_synced,
        experiment_synced=True,
    )


def remote_experiment_status(
    *,
    target_id: str,
    experiment_id: str,
    start_index: int = 1,
    end_index: int | None = None,
    store_root: str | Path = ".numereng",
) -> RemoteExperimentStatusResult:
    """Read one remote experiment run-plan state."""

    target = _get_target(target_id)
    resolved_store_root = resolve_store_root(store_root)
    resolved_end_index = _resolve_effective_end_index(
        store_root=resolved_store_root,
        experiment_id=experiment_id,
        end_index=end_index,
    )
    state_path = _remote_run_plan_state_path(
        target=target,
        store_root=resolved_store_root,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=resolved_end_index,
    )
    payload = _run_remote_python(target, _remote_experiment_status_python_script(), args=[state_path])
    if payload is None:
        raise RemoteExecutionError("remote_experiment_status_failed")
    raw_state = payload.get("state")
    raw_state_dict = raw_state if isinstance(raw_state, dict) else {}
    return RemoteExperimentStatusResult(
        target_id=target.id,
        experiment_id=experiment_id,
        state_path=state_path,
        exists=bool(payload.get("exists")),
        phase=_to_non_empty_str(payload.get("phase")),
        current_index=payload.get("current_index") if isinstance(payload.get("current_index"), int) else None,
        current_run_id=_to_non_empty_str(payload.get("current_run_id")),
        current_config_path=_to_non_empty_str(payload.get("current_config_path")),
        last_completed_row_index=(
            payload.get("last_completed_row_index")
            if isinstance(payload.get("last_completed_row_index"), int)
            else None
        ),
        supervisor_pid=payload.get("supervisor_pid") if isinstance(payload.get("supervisor_pid"), int) else None,
        supervisor_alive=bool(payload.get("supervisor_alive")),
        active_worker_pid=payload.get("active_worker_pid")
        if isinstance(payload.get("active_worker_pid"), int)
        else None,
        last_successful_heartbeat_at=_to_non_empty_str(payload.get("last_successful_heartbeat_at")),
        retry_count=int(payload.get("retry_count") or 0),
        failure_classifier=_to_non_empty_str(payload.get("failure_classifier")),
        terminal_error=_to_non_empty_str(payload.get("terminal_error")),
        raw_state=raw_state_dict,
    )


def remote_maintain_experiment(
    *,
    target_id: str,
    experiment_id: str,
    start_index: int = 1,
    end_index: int | None = None,
    store_root: str | Path = ".numereng",
) -> RemoteExperimentMaintainResult:
    """Restart one dead nonterminal remote experiment run-plan window."""

    status = remote_experiment_status(
        target_id=target_id,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=end_index,
        store_root=store_root,
    )
    if not status.exists:
        raise RemoteExecutionError(f"remote_experiment_state_not_found:{experiment_id}")
    if status.phase in {"complete", "failed", "stopped"}:
        return RemoteExperimentMaintainResult(
            target_id=status.target_id,
            experiment_id=status.experiment_id,
            state_path=status.state_path,
            action="terminal",
            phase=status.phase,
            supervisor_pid=status.supervisor_pid,
            note="terminal_state",
        )
    if status.supervisor_alive:
        return RemoteExperimentMaintainResult(
            target_id=status.target_id,
            experiment_id=status.experiment_id,
            state_path=status.state_path,
            action="noop",
            phase=status.phase,
            supervisor_pid=status.supervisor_pid,
            note="supervisor_alive",
        )

    target = _get_target(target_id)
    window = status.raw_state.get("window") if isinstance(status.raw_state.get("window"), dict) else {}
    resume_start_index = int(window.get("start_index") or start_index)
    resume_end_index = int(
        window.get("end_index")
        or _resolve_effective_end_index(
            store_root=resolve_store_root(store_root),
            experiment_id=experiment_id,
            end_index=end_index,
        )
    )
    requested_score_stage = _to_non_empty_str(status.raw_state.get("requested_score_stage")) or "post_training_core"
    launch = _launch_remote_experiment_window(
        target=target,
        experiment_id=experiment_id,
        start_index=resume_start_index,
        end_index=resume_end_index,
        score_stage=requested_score_stage,
        resume=True,
    )
    return RemoteExperimentMaintainResult(
        target_id=status.target_id,
        experiment_id=status.experiment_id,
        state_path=status.state_path,
        action="restarted",
        phase=status.phase,
        supervisor_pid=launch["pid"],
        note="supervisor_restarted",
    )


def remote_stop_experiment(
    *,
    target_id: str,
    experiment_id: str,
    start_index: int = 1,
    end_index: int | None = None,
    store_root: str | Path = ".numereng",
) -> RemoteExperimentStopResult:
    """Stop one remote experiment run-plan supervisor."""

    target = _get_target(target_id)
    resolved_store_root = resolve_store_root(store_root)
    resolved_end_index = _resolve_effective_end_index(
        store_root=resolved_store_root,
        experiment_id=experiment_id,
        end_index=end_index,
    )
    state_path = _remote_run_plan_state_path(
        target=target,
        store_root=resolved_store_root,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=resolved_end_index,
    )
    payload = _run_remote_python(target, _remote_experiment_stop_python_script(), args=[state_path])
    if payload is None:
        raise RemoteExecutionError("remote_experiment_stop_failed")
    return RemoteExperimentStopResult(
        target_id=target.id,
        experiment_id=experiment_id,
        state_path=state_path,
        stopped=bool(payload.get("stopped")),
        supervisor_pid=payload.get("supervisor_pid") if isinstance(payload.get("supervisor_pid"), int) else None,
        note=_to_non_empty_str(payload.get("note")),
    )


def _get_target(target_id: str) -> SshRemoteTargetProfile:
    targets = {target.id: target for target in load_remote_targets(strict=True)}
    target = targets.get(target_id)
    if target is None:
        raise RemoteTargetNotFoundError(f"remote_target_not_found:{target_id}")
    return target


def _to_target_record(target: SshRemoteTargetProfile) -> RemoteTargetRecord:
    return RemoteTargetRecord(
        id=target.id,
        label=target.label,
        kind=target.kind,
        shell=target.shell,
        repo_root=target.repo_root,
        store_root=target.store_root,
        runner_cmd=target.runner_cmd,
        python_cmd=target.python_cmd,
        tags=tuple(target.tags),
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _repo_sync_entries(repo_root: Path) -> list[SyncEntry]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        check=True,
    )
    entries: list[SyncEntry] = []
    for item in result.stdout.decode("utf-8", errors="replace").split("\0"):
        relpath = item.strip()
        if not relpath or _skip_repo_relpath(relpath):
            continue
        local_path = repo_root / relpath
        if not local_path.is_file():
            continue
        entries.append(SyncEntry(local_path=local_path, remote_relpath=relpath.replace("\\", "/")))
    return entries


def _skip_repo_relpath(relpath: str) -> bool:
    normalized = relpath.replace("\\", "/")
    if normalized.startswith(".git/") or normalized == ".git":
        return True
    if normalized.startswith(".numereng/") or normalized == ".numereng":
        return True
    if normalized.startswith("src/numereng/platform/remotes/profiles/"):
        return True
    for prefix in (".venv/", "venv/", "node_modules/", ".pytest_cache/", ".mypy_cache/", "dist/", "build/"):
        if normalized.startswith(prefix):
            return True
    return False


def _experiment_sync_entries(store_root: Path, *, experiment_id: str) -> list[SyncEntry]:
    workspace_layout = resolve_workspace_layout_from_store_root(store_root)
    experiment_root = workspace_layout.experiments_root / experiment_id
    if not experiment_root.is_dir():
        raise RemoteValidationError(f"experiment_not_found:{experiment_id}")

    entries: list[SyncEntry] = []
    for file_name in ("EXPERIMENT.md", "run_plan.csv"):
        path = experiment_root / file_name
        if path.is_file():
            entries.append(
                SyncEntry(
                    local_path=path,
                    remote_relpath=str(path.relative_to(workspace_layout.workspace_root)).replace("\\", "/"),
                )
            )

    for subdir in ("configs", "run_scripts"):
        root = experiment_root / subdir
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            entries.append(
                SyncEntry(
                    local_path=path,
                    remote_relpath=str(path.relative_to(workspace_layout.workspace_root)).replace("\\", "/"),
                )
            )
    return entries


def _ensure_remote_experiment_created(
    *,
    target: SshRemoteTargetProfile,
    experiment_id: str,
    store_root: Path,
) -> None:
    if _remote_experiment_exists(target=target, experiment_id=experiment_id):
        return
    experiment_dir = resolve_workspace_layout_from_store_root(store_root).experiments_root / experiment_id
    manifest = _read_json_dict(experiment_dir / "experiment.json")
    command = [
        *_split_target_command(target.runner_cmd, shell=target.shell),
        "experiment",
        "create",
        "--id",
        experiment_id,
        "--workspace",
        target.repo_root,
    ]
    name = _to_non_empty_str(manifest.get("name"))
    hypothesis = _to_non_empty_str(manifest.get("hypothesis"))
    tags = manifest.get("tags")
    if name is not None:
        command.extend(["--name", name])
    if hypothesis is not None:
        command.extend(["--hypothesis", hypothesis])
    if isinstance(tags, list):
        normalized_tags = [str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()]
        if normalized_tags:
            command.extend(["--tags", ",".join(normalized_tags)])
    result = _run_remote_command(target, command=command, cwd=target.repo_root)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "remote_experiment_create_failed"
        raise RemoteExecutionError(f"remote_experiment_create_failed:{detail}")


def _remote_experiment_exists(*, target: SshRemoteTargetProfile, experiment_id: str) -> bool:
    payload = _run_remote_python(
        target,
        _experiment_exists_python_script(),
        args=[experiment_id, target.store_root],
    )
    return bool(payload and payload.get("exists"))


def _remote_pull_staging_dir(store_root: Path, *, experiment_id: str, mode: PullMode) -> Path:
    staging_root = store_root / _REMOTE_PULL_STAGING_DIR[0] / _REMOTE_PULL_STAGING_DIR[1] / _REMOTE_PULL_STAGING_DIR[2]
    staging_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{_safe_name(experiment_id)}-{mode}-", dir=staging_root))


def _reconcile_local_experiment_manifest(
    *,
    store_root: Path,
    target_id: str,
    experiment_id: str,
    remote_manifest: dict[str, Any],
    pulled_at: str,
    materialized_run_ids: tuple[str, ...],
) -> Path:
    experiment_dir = resolve_workspace_layout_from_store_root(store_root).experiments_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_dir / "experiment.json"
    local_manifest = _read_json_dict(manifest_path)

    merged = dict(local_manifest)
    merged["experiment_id"] = experiment_id
    for key in ("name", "hypothesis", "status", "schema_version", "champion_run_id"):
        value = remote_manifest.get(key)
        if value is not None:
            merged[key] = value

    remote_tags = remote_manifest.get("tags")
    if isinstance(remote_tags, list):
        merged["tags"] = [str(tag) for tag in remote_tags]

    merged["created_at"] = (
        _to_non_empty_str(remote_manifest.get("created_at"))
        or _to_non_empty_str(local_manifest.get("created_at"))
        or pulled_at
    )
    merged["updated_at"] = (
        _to_non_empty_str(remote_manifest.get("updated_at"))
        or _to_non_empty_str(local_manifest.get("updated_at"))
        or pulled_at
    )

    local_runs = local_manifest.get("runs")
    remote_runs = remote_manifest.get("runs")
    merged_runs: list[str] = []
    seen_runs: set[str] = set()
    for payload in (remote_runs, local_runs, list(materialized_run_ids)):
        if not isinstance(payload, list):
            continue
        for run_id in payload:
            if not isinstance(run_id, str):
                continue
            stripped = run_id.strip()
            if not stripped or stripped in seen_runs:
                continue
            seen_runs.add(stripped)
            merged_runs.append(stripped)
    merged["runs"] = merged_runs

    metadata: dict[str, Any] = {}
    local_metadata = local_manifest.get("metadata")
    if isinstance(local_metadata, dict):
        metadata.update(local_metadata)
    remote_metadata = remote_manifest.get("metadata")
    if isinstance(remote_metadata, dict):
        metadata.update(remote_metadata)
    if "hypothesis" not in metadata and isinstance(merged.get("hypothesis"), str):
        metadata["hypothesis"] = merged["hypothesis"]
    if "tags" not in metadata and isinstance(merged.get("tags"), list):
        metadata["tags"] = merged["tags"]
    if "champion_run_id" not in metadata and isinstance(merged.get("champion_run_id"), str):
        metadata["champion_run_id"] = merged["champion_run_id"]
    remote_pull_payload = metadata.get("remote_pull") if isinstance(metadata.get("remote_pull"), dict) else {}
    remote_pull_payload.update(
        {
            "last_target_id": target_id,
            "last_pulled_at": pulled_at,
            "materialized_finished_run_ids": list(materialized_run_ids),
        }
    )
    metadata["remote_pull"] = remote_pull_payload
    merged["metadata"] = metadata

    manifest_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upsert_experiment(
        store_root=store_root,
        experiment_id=experiment_id,
        name=_to_non_empty_str(merged.get("name")) or experiment_id,
        status=_to_non_empty_str(remote_manifest.get("status")) or _to_non_empty_str(merged.get("status")) or "active",
        created_at=str(merged["created_at"]),
        updated_at=str(merged["updated_at"]),
        metadata=metadata,
    )
    return manifest_path


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_head_sha(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _git_is_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain=1", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _should_sync_repo(target_id: str, store_root: Path) -> bool:
    marker_path, _ = build_repo_marker_paths(store_root=store_root, target_id=target_id)
    current_commit = _git_head_sha(_repository_root())
    current_dirty = _git_is_dirty(_repository_root())
    if not marker_path.is_file():
        return True
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return payload.get("local_commit_sha") != current_commit or bool(payload.get("dirty")) != current_dirty


def _infer_experiment_id(config_path: Path, store_root: Path) -> str | None:
    experiments_root = resolve_workspace_layout_from_store_root(store_root).experiments_root
    try:
        relative = config_path.relative_to(experiments_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) >= 2 and parts[1] == "configs":
        return parts[0]
    return None


def _is_experiment_config(config_path: Path, store_root: Path, experiment_id: str) -> bool:
    experiments_root = resolve_workspace_layout_from_store_root(store_root).experiments_root
    try:
        relative = config_path.relative_to(experiments_root)
    except ValueError:
        return False
    parts = relative.parts
    return len(parts) >= 3 and parts[0] == experiment_id and parts[1] == "configs"


def _remote_store_relative_path(target: SshRemoteTargetProfile, store_root: Path, config_path: Path) -> str:
    workspace_root = resolve_workspace_layout_from_store_root(store_root).workspace_root
    relative = config_path.relative_to(workspace_root)
    return remote_path_join(target, target.repo_root, *relative.parts)


def _run_remote_snapshot(target: SshRemoteTargetProfile) -> dict[str, Any] | None:
    command = build_ssh_command(target, build_monitor_snapshot_command(target))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=target.command_timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        return None
    return _extract_json_object(result.stdout)


def _run_remote_python(
    target: SshRemoteTargetProfile,
    script_source: str,
    *,
    args: list[str],
    timeout_seconds: int | None = None,
) -> dict[str, Any] | None:
    command = build_ssh_command(
        target,
        build_remote_python_command(target, script_source, args=args, cwd=target.repo_root),
    )
    effective_timeout = timeout_seconds if timeout_seconds is not None else target.command_timeout_seconds
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, 4):
        try:
            last_result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            if attempt == 3:
                return None
            time.sleep(min(2.0 * attempt, 10.0))
            continue
        if last_result.returncode == 0:
            return _extract_json_object(last_result.stdout)
        combined = (last_result.stderr or "") + (last_result.stdout or "")
        if attempt == 3 or not _is_scp_transient_error(combined):
            return None
        time.sleep(min(2.0 * attempt, 10.0))
    return None


def _extract_json_bytes(payload: bytes) -> dict[str, Any]:
    return _extract_json_object(payload.decode("utf-8", errors="replace")) or {}


def _extract_json_object(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _decode_pull_failures(payload: Any) -> tuple[RemoteExperimentPullFailure, ...]:
    failures: list[RemoteExperimentPullFailure] = []
    if not isinstance(payload, list):
        return ()
    for item in payload:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        missing_files_raw = item.get("missing_files")
        missing_files = (
            tuple(str(name) for name in missing_files_raw if isinstance(name, str))
            if isinstance(missing_files_raw, list)
            else ()
        )
        failures.append(
            RemoteExperimentPullFailure(
                run_id=run_id,
                missing_files=missing_files,
                reason=_to_non_empty_str(item.get("reason")),
            )
        )
    return tuple(failures)


def _detect_local_run_mode(run_dir: Path) -> Literal["missing", "incomplete", "scoring", "full"]:
    """Classify the on-disk materialization state of one local run directory."""
    if not run_dir.exists() or not run_dir.is_dir():
        return "missing"
    for relpath in _FINISHED_RUN_REQUIRED_LOCAL_FILES:
        if not (run_dir / relpath).is_file():
            return "incomplete"
    predictions_dir = run_dir / _PREDICTIONS_SUBDIR[0] / _PREDICTIONS_SUBDIR[1]
    has_predictions = predictions_dir.is_dir() and any(predictions_dir.glob("pred_*.parquet"))
    return "full" if has_predictions else "scoring"


def _classify_local_finished_runs(
    *,
    local_runs_root: Path,
    run_ids: tuple[str, ...],
    requested_mode: PullMode,
) -> dict[str, tuple[str, ...] | tuple[RemoteExperimentPullFailure, ...]]:
    already_materialized_run_ids: list[str] = []
    run_ids_to_copy: list[str] = []
    upgrade_run_ids: list[str] = []
    failures: list[RemoteExperimentPullFailure] = []
    for run_id in run_ids:
        run_dir = local_runs_root / run_id
        local_mode = _detect_local_run_mode(run_dir)
        if local_mode == "missing":
            run_ids_to_copy.append(run_id)
            continue
        if run_dir.exists() and not run_dir.is_dir():
            failures.append(
                RemoteExperimentPullFailure(
                    run_id=run_id,
                    missing_files=(),
                    reason="local_run_path_not_dir",
                )
            )
            continue
        if local_mode == "incomplete":
            core_missing = tuple(
                relpath for relpath in _FINISHED_RUN_REQUIRED_LOCAL_FILES if not (run_dir / relpath).is_file()
            )
            failures.append(
                RemoteExperimentPullFailure(
                    run_id=run_id,
                    missing_files=core_missing,
                    reason="local_run_incomplete",
                )
            )
            continue
        manifest = _read_json_dict(run_dir / "run.json")
        manifest_run_id = _to_non_empty_str(manifest.get("run_id"))
        if manifest_run_id != run_id:
            failures.append(
                RemoteExperimentPullFailure(
                    run_id=run_id,
                    missing_files=(),
                    reason="local_run_manifest_mismatch",
                )
            )
            continue
        # Local state is either "scoring" (no predictions) or "full" (predictions present).
        # - full local + any requested mode -> already materialized (never downgrade)
        # - scoring local + scoring requested -> already materialized
        # - scoring local + full requested -> upgrade: re-pull and overlay predictions
        if local_mode == "full" or (local_mode == "scoring" and requested_mode == "scoring"):
            already_materialized_run_ids.append(run_id)
        else:
            upgrade_run_ids.append(run_id)
            run_ids_to_copy.append(run_id)
    return {
        "already_materialized_run_ids": tuple(already_materialized_run_ids),
        "run_ids_to_copy": tuple(run_ids_to_copy),
        "upgrade_run_ids": tuple(upgrade_run_ids),
        "failures": tuple(failures),
    }


def _copy_remote_runs_to_staging(
    *,
    target: SshRemoteTargetProfile,
    remote_store_root: str,
    staging_root: Path,
    run_ids: tuple[str, ...],
    mode: PullMode,
) -> Path:
    staging_runs_root = staging_root / "runs"
    staging_runs_root.mkdir(parents=True, exist_ok=True)
    for run_id in run_ids:
        remote_run_dir = remote_path_join(target, remote_store_root, "runs", run_id)
        if mode == "full":
            _scp_remote_directory(
                target=target,
                remote_path=remote_run_dir,
                destination_root=staging_runs_root,
            )
        else:
            _copy_remote_run_scoring(
                target=target,
                remote_run_dir=remote_run_dir,
                staging_run_dir=staging_runs_root / run_id,
            )
    return staging_runs_root


def _copy_remote_run_scoring(
    *,
    target: SshRemoteTargetProfile,
    remote_run_dir: str,
    staging_run_dir: Path,
) -> None:
    """Copy scoring-mode artifacts for one run: scoring subtree plus root files."""
    staging_run_dir.mkdir(parents=True, exist_ok=True)
    (staging_run_dir / _SCORING_MODE_SUBDIR[0]).mkdir(parents=True, exist_ok=True)
    scoring_remote = remote_path_join(target, remote_run_dir, *_SCORING_MODE_SUBDIR)
    _scp_remote_directory(
        target=target,
        remote_path=scoring_remote,
        destination_root=staging_run_dir / _SCORING_MODE_SUBDIR[0],
    )
    required_paths = tuple(
        remote_path_join(target, remote_run_dir, filename) for filename in _SCORING_MODE_ROOT_REQUIRED
    )
    _scp_remote_files(
        target=target,
        remote_paths=required_paths,
        destination_dir=staging_run_dir,
        allow_missing=False,
    )
    for filename in _SCORING_MODE_ROOT_OPTIONAL:
        _scp_remote_files(
            target=target,
            remote_paths=(remote_path_join(target, remote_run_dir, filename),),
            destination_dir=staging_run_dir,
            allow_missing=True,
        )


def _scp_remote_directory(
    *,
    target: SshRemoteTargetProfile,
    remote_path: str,
    destination_root: Path,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    command = [
        *scp_base_command(target),
        "-r",
        _scp_remote_spec(target=target, remote_path=remote_path),
        str(destination_root),
    ]
    _run_scp_with_retry(target=target, command=command)


def _run_scp_with_retry(
    *,
    target: SshRemoteTargetProfile,
    command: list[str],
    allow_missing: bool = False,
    max_attempts: int = 3,
) -> None:
    """Run one scp invocation with retry-on-transient-failure semantics.

    Remote pulls over laggy SSH links occasionally drop mid-transfer with
    "Connection closed" or "Operation timed out". Those failures are usually
    transient; retrying with exponential backoff tolerates them without
    failing the whole pull. The base scp flags already enable SSH connection
    multiplexing so retries reuse the existing control socket when possible.
    """
    last_stderr = ""
    last_stdout = ""
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(target.command_timeout_seconds, _REMOTE_PULL_COPY_TIMEOUT_SECONDS),
            check=False,
        )
        if result.returncode == 0:
            return
        last_stderr = (result.stderr or "").strip()
        last_stdout = (result.stdout or "").strip()
        combined = last_stderr + last_stdout
        if allow_missing and _is_scp_missing_error(combined):
            return
        if attempt == max_attempts or not _is_scp_transient_error(combined):
            break
        time.sleep(min(2.0 * attempt, 10.0))
    detail = last_stderr or last_stdout or "remote_scp_failed"
    raise RemoteExecutionError(f"remote_experiment_pull_copy_failed:{detail}")


def _is_scp_transient_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "connection closed",
            "connection reset",
            "operation timed out",
            "broken pipe",
            "lost connection",
            "connect to host",
            "timeout",
        )
    )


def _scp_remote_files(
    *,
    target: SshRemoteTargetProfile,
    remote_paths: tuple[str, ...],
    destination_dir: Path,
    allow_missing: bool,
) -> None:
    """SCP one or more remote files into ``destination_dir``.

    When ``allow_missing`` is True, "No such file" errors are swallowed so that
    legitimately optional files (e.g. ``run.log``) don't fail the whole pull.
    """
    if not remote_paths:
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *scp_base_command(target),
        *[_scp_remote_spec(target=target, remote_path=path) for path in remote_paths],
        str(destination_dir),
    ]
    _run_scp_with_retry(target=target, command=command, allow_missing=allow_missing)


def _is_scp_missing_error(text: str) -> bool:
    lowered = text.lower()
    return "no such file" in lowered or "not found" in lowered or "cannot find" in lowered


def _scp_remote_spec(*, target: SshRemoteTargetProfile, remote_path: str) -> str:
    normalized = remote_path.replace("\\", "/")
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        normalized = f"/{normalized}"
    return f"{ssh_destination(target)}:{normalized}"


def _ensure_staged_runs_present(*, extracted_runs_root: Path, run_ids: tuple[str, ...]) -> None:
    missing_run_ids = [run_id for run_id in run_ids if not (extracted_runs_root / run_id / "run.json").is_file()]
    if missing_run_ids:
        raise RemoteExecutionError(f"remote_experiment_pull_stage_incomplete:{','.join(sorted(missing_run_ids))}")


def _materialize_staged_runs(
    *,
    store_root: Path,
    staging_runs_root: Path,
    run_ids: tuple[str, ...],
    upgrade_run_ids: set[str] | None = None,
) -> tuple[str, ...]:
    local_runs_root = store_root / "runs"
    upgrade_ids = upgrade_run_ids or set()
    moved_run_ids: list[str] = []
    try:
        for run_id in run_ids:
            source_dir = staging_runs_root / run_id
            destination_dir = local_runs_root / run_id
            if destination_dir.exists():
                if run_id in upgrade_ids:
                    # Scoring-only local run being upgraded to full: clear the
                    # stale scoring tree so the atomic rename of the fresh
                    # (full) staged directory can proceed.
                    shutil.rmtree(destination_dir)
                else:
                    raise RemoteValidationError(f"remote_experiment_pull_local_conflict:{run_id}")
            source_dir.rename(destination_dir)
            moved_run_ids.append(run_id)
    except Exception:
        _rollback_materialized_runs(store_root=store_root, run_ids=tuple(moved_run_ids))
        _delete_run_index_rows(store_root=store_root, run_ids=tuple(moved_run_ids))
        raise
    return tuple(run_ids)


def _reindex_materialized_runs(*, store_root: Path, run_ids: tuple[str, ...]) -> None:
    for run_id in run_ids:
        index_run(store_root=store_root, run_id=run_id)


def _rollback_materialized_runs(*, store_root: Path, run_ids: tuple[str, ...]) -> None:
    local_runs_root = store_root / "runs"
    for run_id in run_ids:
        shutil.rmtree(local_runs_root / run_id, ignore_errors=True)


def _delete_run_index_rows(*, store_root: Path, run_ids: tuple[str, ...]) -> None:
    if not run_ids:
        return
    init_result = init_store_db(store_root=store_root)
    try:
        with sqlite3.connect(init_result.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for run_id in run_ids:
                conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM run_artifacts WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
    except sqlite3.Error:
        return


def _optional_str(payload: dict[str, Any] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _optional_nested_str(payload: dict[str, Any] | None, root_key: str, leaf_key: str) -> str | None:
    if payload is None:
        return None
    root = payload.get(root_key)
    if not isinstance(root, dict):
        return None
    value = root.get(leaf_key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _timestamp_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _to_non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _split_target_command(command: str, *, shell: str) -> list[str]:
    return shlex.split(command, posix=shell != "powershell")


def _quote_remote_arg(target: SshRemoteTargetProfile, value: str) -> str:
    if target.shell == "powershell":
        return f"'{powershell_single_quote(value)}'"
    return shlex.quote(value)


def _run_remote_command(
    target: SshRemoteTargetProfile,
    *,
    command: list[str],
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(_quote_remote_arg(target, item) for item in command)
    if target.shell == "powershell":
        rendered = f"& {rendered}"
    ssh_command = build_ssh_command(target, build_remote_shell_command(target, rendered, cwd=cwd))
    return subprocess.run(
        ssh_command,
        capture_output=True,
        text=True,
        timeout=target.command_timeout_seconds,
        check=False,
    )


def _resolve_effective_end_index(*, store_root: Path, experiment_id: str, end_index: int | None) -> int:
    if end_index is not None:
        return end_index
    experiment_dir = resolve_workspace_layout_from_store_root(store_root).experiments_root / experiment_id
    run_plan_path = experiment_dir / "run_plan.csv"
    if not run_plan_path.is_file():
        raise RemoteValidationError(f"experiment_run_plan_missing:{experiment_id}")
    with run_plan_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        count = sum(1 for _ in reader)
    if count < 1:
        raise RemoteValidationError(f"experiment_run_plan_empty:{experiment_id}")
    return count


def _launch_remote_experiment_window(
    *,
    target: SshRemoteTargetProfile,
    experiment_id: str,
    start_index: int,
    end_index: int,
    score_stage: str,
    resume: bool,
) -> dict[str, str | int]:
    launch_prefix = (
        f"{_safe_name(experiment_id)}-resume" if resume else f"{_safe_name(experiment_id)}-{start_index}-{end_index}"
    )
    launch_id = f"{launch_prefix}-{_timestamp_token()}"
    remote_log_path = remote_path_join(target, target.store_root, "remote_ops", "launches", f"{launch_id}.log")
    remote_metadata_path = remote_path_join(target, target.store_root, "remote_ops", "launches", f"{launch_id}.json")
    launch_payload = _run_remote_python(
        target,
        _launch_python_script(),
        args=[
            base64.b64encode(
                json.dumps(
                    _remote_experiment_run_plan_command(
                        target=target,
                        experiment_id=experiment_id,
                        start_index=start_index,
                        end_index=end_index,
                        score_stage=score_stage,
                        resume=resume,
                    )
                ).encode("utf-8")
            ).decode("ascii"),
            target.repo_root,
            remote_log_path,
            remote_metadata_path,
            launch_id,
            base64.b64encode(_remote_execution_payload(target).encode("utf-8")).decode("ascii"),
        ],
    )
    error_code = "remote_experiment_maintain_launch_failed" if resume else "remote_experiment_launch_failed"
    if launch_payload is None:
        raise RemoteExecutionError(error_code)
    if not bool(launch_payload.get("ok", True)):
        raise RemoteExecutionError(_remote_launch_error_message(launch_payload))
    return {
        "launch_id": launch_id,
        "remote_log_path": remote_log_path,
        "remote_metadata_path": remote_metadata_path,
        "pid": int(launch_payload["pid"]),
        "launched_at": str(launch_payload["launched_at"]),
    }


def _remote_experiment_run_plan_command(
    *,
    target: SshRemoteTargetProfile,
    experiment_id: str,
    start_index: int,
    end_index: int,
    score_stage: str,
    resume: bool,
) -> list[str]:
    command = [
        *_split_target_command(target.runner_cmd, shell=target.shell),
        "experiment",
        "run-plan",
        "--id",
        experiment_id,
        "--start-index",
        str(start_index),
        "--end-index",
        str(end_index),
        "--score-stage",
        score_stage,
        "--workspace",
        target.repo_root,
    ]
    if resume:
        command.append("--resume")
    return command


def _remote_execution_payload(target: SshRemoteTargetProfile) -> str:
    return serialize_run_execution(
        build_run_execution(
            kind="remote_host",
            provider="ssh",
            backend="remote_pc",
            target_id=target.id,
            host=target.label,
        )
    )


def _remote_run_plan_state_path(
    *,
    target: SshRemoteTargetProfile,
    store_root: Path,
    experiment_id: str,
    start_index: int,
    end_index: int,
) -> str:
    local_state_path = resolve_experiment_run_plan_state_path(
        store_root=store_root,
        experiment_id=experiment_id,
        start_index=start_index,
        end_index=end_index,
    )
    return remote_path_join(target, target.store_root, "remote_ops", "experiment_run_plan", local_state_path.name)


def _launch_id_for_config(config_path: Path) -> str:
    digest = hashlib.sha256(str(config_path).encode("utf-8")).hexdigest()[:8]
    return f"{_timestamp_token()}-{digest}"


def _remote_launch_error_message(payload: dict[str, Any]) -> str:
    error = str(payload.get("error") or "remote_train_launch_failed")
    exit_code = payload.get("exit_code")
    if isinstance(exit_code, int):
        return f"{error}:exit_code={exit_code}"
    return error


def _doctor_failure_message(issues: tuple[str, ...]) -> str:
    if not issues:
        return "remote_doctor_failed"
    return f"remote_doctor_failed:{','.join(issues)}"


def _doctor_python_script() -> str:
    return """
import json
import os
import sys

print(
    json.dumps(
        {
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
        },
        sort_keys=True,
    )
)
""".strip()


def _launch_python_script() -> str:
    return (
        """
import json
import base64
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


STARTUP_TIMEOUT_SECONDS = __STARTUP_TIMEOUT_SECONDS__
STARTUP_POLL_SECONDS = __STARTUP_POLL_SECONDS__



def _terminate_process(process) -> int | None:
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=1.0)
        except Exception:
            return process.poll()
    return process.poll()


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit("remote_launch_arguments_invalid")
    command = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    cwd = Path(sys.argv[2]).expanduser().resolve()
    log_path = Path(sys.argv[3]).expanduser().resolve()
    metadata_path = Path(sys.argv[4]).expanduser().resolve()
    launch_id = sys.argv[5]
    execution_json = base64.b64decode(sys.argv[6]).decode("utf-8")
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    child_env["NUMERENG_RUN_EXECUTION_JSON"] = execution_json

    with log_path.open("ab") as log_handle:
        if sys.platform == "win32":
            creationflags = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
            )
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                creationflags=creationflags,
                close_fds=True,
            )
            confirmed = False
            error = None
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    error = "remote_train_launch_child_exited_early"
                    break
                try:
                    if log_path.exists() and log_path.stat().st_size > 0:
                        confirmed = True
                        break
                except OSError:
                    pass
                time.sleep(STARTUP_POLL_SECONDS)
            if not confirmed:
                exit_code = process.poll()
                if error is not None:
                    payload = {
                        "ok": False,
                        "launch_id": launch_id,
                        "pid": process.pid,
                        "command": command,
                        "cwd": str(cwd),
                        "log_path": str(log_path),
                        "metadata_path": str(metadata_path),
                        "launched_at": datetime.now(UTC).isoformat(),
                        "execution": json.loads(execution_json),
                        "error": error,
                        "exit_code": exit_code,
                        "log_size_bytes": log_path.stat().st_size if log_path.exists() else 0,
                    }
                    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
                    print(json.dumps(payload, sort_keys=True))
                    return
            startup_confirmed = confirmed
            launch_warning = None
            if not startup_confirmed:
                launch_warning = "remote_train_launch_log_unconfirmed"
                exit_code = process.poll()
                payload = {
                    "ok": True,
                    "launch_id": launch_id,
                    "pid": process.pid,
                    "command": command,
                    "cwd": str(cwd),
                    "log_path": str(log_path),
                    "metadata_path": str(metadata_path),
                    "launched_at": datetime.now(UTC).isoformat(),
                    "execution": json.loads(execution_json),
                    "startup_confirmed": startup_confirmed,
                    "launch_warning": launch_warning,
                    "log_size_bytes": log_path.stat().st_size if log_path.exists() else 0,
                }
        else:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
                close_fds=True,
            )

    payload = {
        "ok": True,
        "launch_id": launch_id,
        "pid": process.pid,
        "command": command,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "metadata_path": str(metadata_path),
        "launched_at": datetime.now(UTC).isoformat(),
        "execution": json.loads(execution_json),
        "startup_confirmed": True,
        "launch_warning": None,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
""".replace(
            "__STARTUP_TIMEOUT_SECONDS__",
            repr(_REMOTE_WINDOWS_STARTUP_TIMEOUT_SECONDS),
        )
        .replace(
            "__STARTUP_POLL_SECONDS__",
            repr(_REMOTE_WINDOWS_STARTUP_POLL_SECONDS),
        )
        .strip()
    )


def _pull_preflight_python_script() -> str:
    return """
import json
import sys
from pathlib import Path

from numereng.features.store import resolve_workspace_layout_from_store_root


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("remote_pull_preflight_arguments_invalid")
    experiment_id = sys.argv[1]
    store_root = Path(sys.argv[2]).expanduser()
    experiments_root = resolve_workspace_layout_from_store_root(store_root).experiments_root

    experiment_dir = experiments_root / experiment_id
    archived_dir = experiments_root / "_archive" / experiment_id
    if (experiment_dir / "experiment.json").is_file():
        resolved_experiment_dir = experiment_dir
    elif (archived_dir / "experiment.json").is_file():
        resolved_experiment_dir = archived_dir
    else:
        print(json.dumps({"error": "remote_experiment_not_found", "experiment_id": experiment_id}, sort_keys=True))
        return

    experiment_manifest_path = resolved_experiment_dir / "experiment.json"
    try:
        # utf-8-sig transparently strips a UTF-8 BOM that some Windows tools
        # (PowerShell Set-Content, Out-File) prepend to JSON files.
        experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8-sig"))
    except Exception:
        print(
            json.dumps(
                {"error": "remote_experiment_manifest_invalid", "experiment_id": experiment_id},
                sort_keys=True,
            )
        )
        return
    if not isinstance(experiment_manifest, dict):
        print(
            json.dumps(
                {"error": "remote_experiment_manifest_invalid", "experiment_id": experiment_id},
                sort_keys=True,
            )
        )
        return

    run_ids = experiment_manifest.get("runs")
    if not isinstance(run_ids, list):
        run_ids = []

    failures = []
    eligible_finished_run_ids = []
    skipped_non_finished_run_ids = []
    for raw_run_id in run_ids:
        if not isinstance(raw_run_id, str) or not raw_run_id.strip():
            continue
        run_id = raw_run_id.strip()
        run_dir = store_root / "runs" / run_id
        run_manifest_path = run_dir / "run.json"
        if not run_dir.is_dir():
            failures.append({"run_id": run_id, "missing_files": ["run_dir"], "reason": "run_dir_missing"})
            continue
        if not run_manifest_path.is_file():
            failures.append({"run_id": run_id, "missing_files": ["run.json"], "reason": "run_manifest_missing"})
            continue
        try:
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8-sig"))
        except Exception:
            failures.append({"run_id": run_id, "missing_files": ["run.json"], "reason": "run_manifest_invalid"})
            continue
        if not isinstance(run_manifest, dict):
            failures.append({"run_id": run_id, "missing_files": ["run.json"], "reason": "run_manifest_invalid"})
            continue
        status = str(run_manifest.get("status") or "").strip()
        if status == "FINISHED":
            eligible_finished_run_ids.append(run_id)
            continue
        skipped_non_finished_run_ids.append(run_id)

    metadata = {
        "experiment": experiment_manifest,
        "experiment_id": experiment_id,
        "remote_experiment_dir": str(resolved_experiment_dir),
        "remote_manifest_path": str(experiment_manifest_path),
        "eligible_finished_run_ids": eligible_finished_run_ids,
        "skipped_non_finished_run_ids": skipped_non_finished_run_ids,
        "failures": failures,
    }
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
""".strip()


def _experiment_exists_python_script() -> str:
    return """
import json
import sys
from pathlib import Path

from numereng.features.store import resolve_workspace_layout_from_store_root


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("remote_experiment_exists_arguments_invalid")
    experiment_id = sys.argv[1]
    store_root = Path(sys.argv[2]).expanduser()
    experiments_root = resolve_workspace_layout_from_store_root(store_root).experiments_root
    experiment_dir = experiments_root / experiment_id
    archived_dir = experiments_root / "_archive" / experiment_id
    exists = (experiment_dir / "experiment.json").is_file() or (archived_dir / "experiment.json").is_file()
    print(json.dumps({"exists": exists}, sort_keys=True))


if __name__ == "__main__":
    main()
""".strip()


def _remote_experiment_status_python_script() -> str:
    return """
import json
import os
import sys
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_query_limited_information = 0x1000
    still_active = 259
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if handle == 0:
        return ctypes.get_last_error() == 5
    try:
        exit_code = ctypes.c_ulong()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("remote_experiment_status_arguments_invalid")
    state_path = Path(sys.argv[1]).expanduser()
    if not state_path.is_file():
        print(json.dumps({"exists": False, "state_path": str(state_path)}, sort_keys=True))
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        print(json.dumps({"exists": False, "state_path": str(state_path), "error": "state_invalid"}, sort_keys=True))
        return
    if not isinstance(state, dict):
        print(json.dumps({"exists": False, "state_path": str(state_path), "error": "state_invalid"}, sort_keys=True))
        return
    supervisor_pid = state.get("supervisor_pid")
    supervisor_alive = isinstance(supervisor_pid, int) and _pid_alive(supervisor_pid)
    payload = {
        "exists": True,
        "state_path": str(state_path),
        "phase": state.get("phase"),
        "current_index": state.get("current_index"),
        "current_run_id": state.get("current_run_id"),
        "current_config_path": state.get("current_config_path"),
        "last_completed_row_index": state.get("last_completed_row_index"),
        "supervisor_pid": supervisor_pid if isinstance(supervisor_pid, int) else None,
        "supervisor_alive": supervisor_alive,
        "active_worker_pid": state.get("active_worker_pid"),
        "last_successful_heartbeat_at": state.get("last_successful_heartbeat_at"),
        "retry_count": state.get("retry_count"),
        "failure_classifier": state.get("failure_classifier"),
        "terminal_error": state.get("terminal_error"),
        "state": state,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
""".strip()


def _remote_experiment_stop_python_script() -> str:
    return """
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stop_pid(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("remote_experiment_stop_arguments_invalid")
    state_path = Path(sys.argv[1]).expanduser()
    if not state_path.is_file():
        print(json.dumps({"stopped": False, "note": "state_missing", "state_path": str(state_path)}, sort_keys=True))
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        print(json.dumps({"stopped": False, "note": "state_invalid", "state_path": str(state_path)}, sort_keys=True))
        return
    if not isinstance(state, dict):
        print(json.dumps({"stopped": False, "note": "state_invalid", "state_path": str(state_path)}, sort_keys=True))
        return
    supervisor_pid = state.get("supervisor_pid")
    stopped = _stop_pid(supervisor_pid) if isinstance(supervisor_pid, int) else False
    state["phase"] = "stopped"
    state["updated_at"] = _utc_now_iso()
    state["supervisor_pid"] = None
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stopped": stopped,
                "supervisor_pid": supervisor_pid if isinstance(supervisor_pid, int) else None,
                "state_path": str(state_path),
                "note": "stopped" if stopped else "supervisor_not_running",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
""".strip()


__all__ = [
    "RemoteExecutionError",
    "RemoteOpsError",
    "RemoteTargetNotFoundError",
    "RemoteValidationError",
    "bootstrap_viz_remotes",
    "doctor_remote_target",
    "load_viz_bootstrap_state",
    "list_remote_targets",
    "remote_experiment_status",
    "remote_launch_experiment",
    "remote_maintain_experiment",
    "pull_remote_experiment",
    "push_remote_config",
    "remote_run_train",
    "remote_stop_experiment",
    "remote_viz_bootstrap_state_path",
    "sync_remote_experiment",
    "sync_remote_repo",
]
