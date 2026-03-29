"""SSH-backed remote sync and detached launch workflows."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from numereng.features.remote_ops.contracts import (
    RemoteBootstrapStatus,
    RemoteConfigPushResult,
    RemoteDoctorResult,
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
from numereng.features.store import resolve_store_root
from numereng.features.training import PostTrainingScoringPolicy, TrainingProfile
from numereng.platform import load_remote_targets
from numereng.platform.run_execution import build_run_execution, serialize_run_execution
from numereng.platform.remotes.contracts import SshRemoteTargetProfile
from numereng.platform.remotes.ssh import (
    build_monitor_snapshot_command,
    build_remote_python_command,
    build_ssh_command,
    remote_path_join,
)

_SYNC_TIMEOUT_SECONDS = 180
_REMOTE_BOOTSTRAP_DIR = ("remote_ops", "bootstrap")
_REMOTE_VIZ_BOOTSTRAP_FILE = "viz.json"
_REMOTE_WINDOWS_STARTUP_TIMEOUT_SECONDS = 10.0
_REMOTE_WINDOWS_STARTUP_POLL_SECONDS = 0.25


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
    _write_viz_bootstrap_state(payload)
    return payload


def load_viz_bootstrap_state(
    *,
    store_root: str | Path = ".numereng",
) -> RemoteVizBootstrapResult | None:
    """Load the last persisted viz bootstrap result for enabled remote sources."""

    resolved_store_root = resolve_store_root(store_root)
    state_path = remote_viz_bootstrap_state_path(store_root=resolved_store_root)
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    targets_payload = payload.get("targets")
    if not isinstance(targets_payload, list):
        targets_payload = []
    targets: list[RemoteVizBootstrapTargetResult] = []
    for item in targets_payload:
        if not isinstance(item, dict):
            continue
        target_payload = item.get("target")
        if not isinstance(target_payload, dict):
            continue
        try:
            targets.append(
                RemoteVizBootstrapTargetResult(
                    target=RemoteTargetRecord(
                        id=str(target_payload["id"]),
                        label=str(target_payload["label"]),
                        kind=str(target_payload["kind"]),
                        shell=str(target_payload["shell"]),
                        repo_root=str(target_payload["repo_root"]),
                        store_root=str(target_payload["store_root"]),
                        runner_cmd=str(target_payload["runner_cmd"]),
                        python_cmd=str(target_payload["python_cmd"]),
                        tags=tuple(str(tag) for tag in target_payload.get("tags", [])),
                    ),
                    bootstrap_status=_bootstrap_status_value(item.get("bootstrap_status")),
                    last_bootstrap_at=str(item.get("last_bootstrap_at") or payload.get("bootstrapped_at") or ""),
                    last_bootstrap_error=_json_optional_str(item.get("last_bootstrap_error")),
                    repo_synced=bool(item.get("repo_synced")),
                    repo_sync_skipped=bool(item.get("repo_sync_skipped")),
                    doctor_ok=bool(item.get("doctor_ok")),
                    issues=tuple(str(issue) for issue in item.get("issues", []) if isinstance(issue, str)),
                )
            )
        except KeyError:
            continue
    return RemoteVizBootstrapResult(
        store_root=resolved_store_root,
        state_path=state_path,
        bootstrapped_at=str(payload.get("bootstrapped_at") or ""),
        ready_count=int(payload.get("ready_count") or 0),
        degraded_count=int(payload.get("degraded_count") or 0),
        targets=tuple(targets),
    )


def remote_viz_bootstrap_state_path(*, store_root: str | Path) -> Path:
    """Return the local viz bootstrap state path under the numereng store root."""

    resolved_store_root = resolve_store_root(store_root)
    return resolved_store_root / _REMOTE_BOOTSTRAP_DIR[0] / _REMOTE_BOOTSTRAP_DIR[1] / _REMOTE_VIZ_BOOTSTRAP_FILE


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
        remote_root=target.store_root,
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
    command = [*_split_target_command(target.runner_cmd, shell=target.shell), "run", "train", "--config", remote_config_path]
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
    experiment_root = store_root / "experiments" / experiment_id
    if not experiment_root.is_dir():
        raise RemoteValidationError(f"experiment_not_found:{experiment_id}")

    entries: list[SyncEntry] = []
    for file_name in ("experiment.json", "EXPERIMENT.md", "run_plan.csv"):
        path = experiment_root / file_name
        if path.is_file():
            entries.append(
                SyncEntry(
                    local_path=path,
                    remote_relpath=str(path.relative_to(store_root)).replace("\\", "/"),
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
                    remote_relpath=str(path.relative_to(store_root)).replace("\\", "/"),
                )
            )
    return entries


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
    try:
        relative = config_path.relative_to(store_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) >= 3 and parts[0] == "experiments" and parts[2] == "configs":
        return parts[1]
    return None


def _is_experiment_config(config_path: Path, store_root: Path, experiment_id: str) -> bool:
    try:
        relative = config_path.relative_to(store_root)
    except ValueError:
        return False
    parts = relative.parts
    return len(parts) >= 4 and parts[0] == "experiments" and parts[1] == experiment_id and parts[2] == "configs"


def _remote_store_relative_path(target: SshRemoteTargetProfile, store_root: Path, config_path: Path) -> str:
    relative = config_path.relative_to(store_root)
    return remote_path_join(target, target.store_root, *relative.parts)


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


def _run_remote_python(target: SshRemoteTargetProfile, script_source: str, *, args: list[str]) -> dict[str, Any] | None:
    command = build_ssh_command(
        target,
        build_remote_python_command(target, script_source, args=args, cwd=target.repo_root),
    )
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


def _split_target_command(command: str, *, shell: str) -> list[str]:
    return shlex.split(command, posix=shell != "powershell")


def _launch_id_for_config(config_path: Path) -> str:
    digest = hashlib.sha256(str(config_path).encode("utf-8")).hexdigest()[:8]
    return f"{_timestamp_token()}-{digest}"


def _remote_launch_error_message(payload: dict[str, Any]) -> str:
    error = str(payload.get("error") or "remote_train_launch_failed")
    exit_code = payload.get("exit_code")
    if isinstance(exit_code, int):
        return f"{error}:exit_code={exit_code}"
    return error


def _write_viz_bootstrap_state(result: RemoteVizBootstrapResult) -> None:
    result.state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "bootstrapped_at": result.bootstrapped_at,
        "ready_count": result.ready_count,
        "degraded_count": result.degraded_count,
        "targets": [
            {
                "target": {
                    "id": item.target.id,
                    "label": item.target.label,
                    "kind": item.target.kind,
                    "shell": item.target.shell,
                    "repo_root": item.target.repo_root,
                    "store_root": item.target.store_root,
                    "runner_cmd": item.target.runner_cmd,
                    "python_cmd": item.target.python_cmd,
                    "tags": list(item.target.tags),
                },
                "bootstrap_status": item.bootstrap_status,
                "last_bootstrap_at": item.last_bootstrap_at,
                "last_bootstrap_error": item.last_bootstrap_error,
                "repo_synced": item.repo_synced,
                "repo_sync_skipped": item.repo_sync_skipped,
                "doctor_ok": item.doctor_ok,
                "issues": list(item.issues),
            }
            for item in result.targets
        ],
    }
    result.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bootstrap_status_value(value: Any) -> RemoteBootstrapStatus:
    return "degraded" if str(value or "").strip().lower() == "degraded" else "ready"


def _doctor_failure_message(issues: tuple[str, ...]) -> str:
    if not issues:
        return "remote_doctor_failed"
    return f"remote_doctor_failed:{','.join(issues)}"


def _json_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


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
    return """
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
    ).replace(
        "__STARTUP_POLL_SECONDS__",
        repr(_REMOTE_WINDOWS_STARTUP_POLL_SECONDS),
    ).strip()


__all__ = [
    "RemoteExecutionError",
    "RemoteOpsError",
    "RemoteTargetNotFoundError",
    "RemoteValidationError",
    "bootstrap_viz_remotes",
    "doctor_remote_target",
    "load_viz_bootstrap_state",
    "list_remote_targets",
    "push_remote_config",
    "remote_run_train",
    "remote_viz_bootstrap_state_path",
    "sync_remote_experiment",
    "sync_remote_repo",
]
