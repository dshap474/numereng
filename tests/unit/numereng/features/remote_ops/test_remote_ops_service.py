from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from numereng.features.remote_ops import service as remote_service
from numereng.features.remote_ops.sync import SyncExecutionResult
from numereng.platform.remotes.contracts import SshRemoteTargetProfile


def _target() -> SshRemoteTargetProfile:
    return SshRemoteTargetProfile(
        id="pc",
        label="Daniel's PC",
        ssh_config_host="pc",
        shell="powershell",
        repo_root=r"C:\Users\dansh\remote-access\numereng",
        store_root=r"C:\Users\dansh\remote-access\numereng\.numereng",
    )


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True, capture_output=True)


def _seed_staging_runs(staging_root: Path, run_manifests: dict[str, dict[str, object]]) -> Path:
    runs_root = staging_root / "runs"
    for run_id, manifest in run_manifests.items():
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "resolved.json").write_text(json.dumps({"seed": 7}), encoding="utf-8")
        (run_dir / "results.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps({"bmc": {"mean": 0.08}}), encoding="utf-8")
    return runs_root


def test_sync_remote_repo_uses_git_visible_working_tree_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    (repo_root / ".gitignore").write_text(
        ".numereng/\nsrc/numereng/platform/remotes/profiles/*.yaml\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "numereng" / "platform" / "remotes" / "profiles").mkdir(parents=True)
    (repo_root / "src" / "app.py").parent.mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (repo_root / "notes.txt").write_text("tracked later\n", encoding="utf-8")
    (repo_root / ".numereng" / "runs").mkdir(parents=True)
    (repo_root / ".numereng" / "runs" / "skip.txt").write_text("skip\n", encoding="utf-8")
    (repo_root / "src" / "numereng" / "platform" / "remotes" / "profiles" / "pc.yaml").write_text(
        "id: pc\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".gitignore", "src/app.py"], cwd=repo_root, check=True, capture_output=True)

    captured: dict[str, object] = {}

    def fake_sync_entries_to_remote(**kwargs: object) -> SyncExecutionResult:
        entries = kwargs["entries"]
        assert isinstance(entries, list)
        captured["entries"] = entries
        local_marker_path = kwargs["local_marker_path"]
        return SyncExecutionResult(
            target_id="pc",
            scope="repo",
            synced_at="2026-03-27T00:00:00+00:00",
            manifest_hash="hash",
            local_commit_sha=None,
            dirty=True,
            synced_files=len(entries),
            deleted_files=0,
            remote_root=_target().repo_root,
            local_marker_path=local_marker_path,
            remote_marker_path=r"C:\Users\dansh\remote-access\numereng\.numereng\remote_ops\sync\pc\repo.json",
            remote_paths=tuple(entry.remote_relpath for entry in entries),
        )

    monkeypatch.setattr(remote_service, "_repository_root", lambda: repo_root)
    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(remote_service, "sync_entries_to_remote", fake_sync_entries_to_remote)

    result = remote_service.sync_remote_repo(target_id="pc", store_root=tmp_path / ".numereng")

    entries = captured["entries"]
    assert isinstance(entries, list)
    assert {entry.remote_relpath for entry in entries} == {"notes.txt", "src/app.py", ".gitignore"}
    assert result.synced_files == 3


def test_sync_remote_experiment_only_syncs_authoring_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_root = store_root / "experiments" / "exp-1"
    (experiment_root / "configs").mkdir(parents=True)
    (experiment_root / "run_scripts").mkdir(parents=True)
    (experiment_root / "results").mkdir(parents=True)
    (experiment_root / "experiment.json").write_text("{}\n", encoding="utf-8")
    (experiment_root / "EXPERIMENT.md").write_text("# exp\n", encoding="utf-8")
    (experiment_root / "run_plan.csv").write_text("run_id\n", encoding="utf-8")
    (experiment_root / "configs" / "base.json").write_text("{}\n", encoding="utf-8")
    (experiment_root / "run_scripts" / "launch.ps1").write_text("Write-Host hi\n", encoding="utf-8")
    (experiment_root / "results" / "metrics.json").write_text("{}\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_sync_entries_to_remote(**kwargs: object) -> SyncExecutionResult:
        entries = kwargs["entries"]
        assert isinstance(entries, list)
        captured["paths"] = [entry.remote_relpath for entry in entries]
        local_marker_path = kwargs["local_marker_path"]
        return SyncExecutionResult(
            target_id="pc",
            scope="experiment:exp-1",
            synced_at="2026-03-27T00:00:00+00:00",
            manifest_hash="hash",
            local_commit_sha="abc",
            dirty=False,
            synced_files=len(entries),
            deleted_files=0,
            remote_root=_target().store_root,
            local_marker_path=local_marker_path,
            remote_marker_path=r"C:\Users\dansh\remote-access\numereng\.numereng\remote_ops\sync\pc\experiment__exp-1.json",
            remote_paths=tuple(entry.remote_relpath for entry in entries),
        )

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(remote_service, "sync_entries_to_remote", fake_sync_entries_to_remote)
    monkeypatch.setattr(remote_service, "_repository_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(remote_service, "_git_head_sha", lambda repo_root: "abc")
    monkeypatch.setattr(remote_service, "_git_is_dirty", lambda repo_root: False)

    result = remote_service.sync_remote_experiment(target_id="pc", experiment_id="exp-1", store_root=store_root)

    assert captured["paths"] == [
        "experiments/exp-1/experiment.json",
        "experiments/exp-1/EXPERIMENT.md",
        "experiments/exp-1/run_plan.csv",
        "experiments/exp-1/configs/base.json",
        "experiments/exp-1/run_scripts/launch.ps1",
    ]
    assert result.remote_experiment_dir.endswith(r"experiments\exp-1")


def test_remote_run_train_syncs_repo_and_experiment_for_experiment_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    config_path = store_root / "experiments" / "exp-1" / "configs" / "base.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")
    sync_calls: list[str] = []

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(remote_service, "_should_sync_repo", lambda target_id, store_root: True)
    monkeypatch.setattr(
        remote_service,
        "sync_remote_repo",
        lambda **kwargs: sync_calls.append("repo") or None,
    )
    monkeypatch.setattr(
        remote_service,
        "sync_remote_experiment",
        lambda **kwargs: sync_calls.append("experiment") or None,
    )
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {"pid": 4321, "launched_at": "2026-03-27T00:00:00+00:00"},
    )

    result = remote_service.remote_run_train(target_id="pc", config_path=config_path, store_root=store_root)

    assert sync_calls == ["repo", "experiment"]
    assert result.repo_synced is True
    assert result.experiment_synced is True
    assert result.remote_config_path.endswith(r"experiments\exp-1\configs\base.json")
    assert result.remote_pid == 4321


def test_remote_run_train_pushes_ad_hoc_config_when_outside_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "adhoc.json"
    config_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "push_remote_config",
        lambda **kwargs: remote_service.RemoteConfigPushResult(
            target_id="pc",
            local_config_path=config_path,
            remote_config_path=r"C:\Users\dansh\remote-access\numereng\.numereng\tmp\remote-configs\adhoc.json",
            synced_at="2026-03-27T00:00:00+00:00",
        ),
    )
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {"pid": 1234, "launched_at": "2026-03-27T00:00:00+00:00"},
    )

    result = remote_service.remote_run_train(
        target_id="pc",
        config_path=config_path,
        sync_repo="never",
        store_root=tmp_path / ".numereng",
    )

    assert result.repo_synced is False
    assert result.experiment_synced is False
    assert result.remote_config_path.endswith(r"adhoc.json")


def test_remote_run_train_preserves_windows_runner_executable_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "adhoc.json"
    config_path.write_text("{}\n", encoding="utf-8")
    target = _target().model_copy(
        update={"runner_cmd": r"C:\Users\dansh\remote-access\numereng\.venv\Scripts\numereng.exe"}
    )
    captured: dict[str, object] = {}

    def fake_run_remote_python(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        raw_args = kwargs["args"]
        assert isinstance(raw_args, list)
        captured["command"] = json.loads(base64.b64decode(raw_args[0]).decode("utf-8"))
        return {"ok": True, "pid": 4321, "launched_at": "2026-03-28T00:00:00+00:00"}

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: target)
    monkeypatch.setattr(
        remote_service,
        "push_remote_config",
        lambda **kwargs: remote_service.RemoteConfigPushResult(
            target_id="pc",
            local_config_path=config_path,
            remote_config_path=r"C:\Users\dansh\remote-access\numereng\.numereng\tmp\remote-configs\adhoc.json",
            synced_at="2026-03-27T00:00:00+00:00",
        ),
    )
    monkeypatch.setattr(remote_service, "_run_remote_python", fake_run_remote_python)

    remote_service.remote_run_train(
        target_id="pc",
        config_path=config_path,
        sync_repo="never",
        store_root=tmp_path / ".numereng",
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == r"C:\Users\dansh\remote-access\numereng\.venv\Scripts\numereng.exe"
    assert command[1:4] == ["run", "train", "--config"]


def test_remote_run_train_raises_when_launcher_reports_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "adhoc.json"
    config_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "push_remote_config",
        lambda **kwargs: remote_service.RemoteConfigPushResult(
            target_id="pc",
            local_config_path=config_path,
            remote_config_path=r"C:\Users\dansh\remote-access\numereng\.numereng\tmp\remote-configs\adhoc.json",
            synced_at="2026-03-27T00:00:00+00:00",
        ),
    )
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {
            "ok": False,
            "error": "remote_train_launch_child_exited_early",
            "exit_code": 1,
        },
    )

    with pytest.raises(remote_service.RemoteExecutionError, match="remote_train_launch_child_exited_early:exit_code=1"):
        remote_service.remote_run_train(
            target_id="pc",
            config_path=config_path,
            sync_repo="never",
            store_root=tmp_path / ".numereng",
        )


def test_launch_python_script_breaks_away_from_windows_job_object() -> None:
    script = remote_service._launch_python_script()

    assert "CREATE_BREAKAWAY_FROM_JOB" in script


def test_bootstrap_viz_remotes_persists_ready_and_degraded_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = _target()
    bad = good.model_copy(update={"id": "offline", "label": "Offline PC"})

    def fake_sync_remote_repo(*, target_id: str, store_root: str | Path = ".numereng") -> object:
        _ = store_root
        if target_id == "offline":
            raise remote_service.RemoteExecutionError("remote_sync_command_failed:ssh failed")
        return object()

    def fake_doctor_remote_target(*, target_id: str) -> remote_service.RemoteDoctorResult:
        target = good if target_id == "pc" else bad
        if target_id == "pc":
            return remote_service.RemoteDoctorResult(
                target=remote_service.RemoteTargetRecord(
                    id=target.id,
                    label=target.label,
                    kind=target.kind,
                    shell=target.shell,
                    repo_root=target.repo_root,
                    store_root=target.store_root,
                    runner_cmd=target.runner_cmd,
                    python_cmd=target.python_cmd,
                    tags=tuple(target.tags),
                ),
                ok=True,
                checked_at="2026-03-28T12:00:00+00:00",
                remote_python_executable="python",
                remote_cwd=target.repo_root,
                snapshot_ok=True,
                snapshot_source_kind="local",
                snapshot_source_id="local",
                issues=(),
            )
        return remote_service.RemoteDoctorResult(
            target=remote_service.RemoteTargetRecord(
                id=target.id,
                label=target.label,
                kind=target.kind,
                shell=target.shell,
                repo_root=target.repo_root,
                store_root=target.store_root,
                runner_cmd=target.runner_cmd,
                python_cmd=target.python_cmd,
                tags=tuple(target.tags),
            ),
            ok=False,
            checked_at="2026-03-28T12:00:00+00:00",
            remote_python_executable=None,
            remote_cwd=None,
            snapshot_ok=False,
            snapshot_source_kind=None,
            snapshot_source_id=None,
            issues=("monitor_snapshot_failed",),
        )

    monkeypatch.setattr(remote_service, "load_remote_targets", lambda: [good, bad])
    monkeypatch.setattr(remote_service, "_should_sync_repo", lambda target_id, store_root: True)
    monkeypatch.setattr(remote_service, "sync_remote_repo", fake_sync_remote_repo)
    monkeypatch.setattr(remote_service, "doctor_remote_target", fake_doctor_remote_target)

    result = remote_service.bootstrap_viz_remotes(store_root=tmp_path / ".numereng")
    reloaded = remote_service.load_viz_bootstrap_state(store_root=tmp_path / ".numereng")

    assert result.ready_count == 1
    assert result.degraded_count == 1
    assert result.state_path.is_file()
    assert reloaded is not None
    assert reloaded.ready_count == 1
    assert reloaded.degraded_count == 1
    assert [item.target.id for item in reloaded.targets] == ["pc", "offline"]
    degraded = next(item for item in reloaded.targets if item.target.id == "offline")
    assert degraded.bootstrap_status == "degraded"
    assert degraded.last_bootstrap_error == "remote_sync_command_failed:ssh failed"


def test_pull_remote_experiment_materializes_finished_runs_and_reconciles_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    local_experiment_dir = store_root / "experiments" / "exp-1"
    local_experiment_dir.mkdir(parents=True)
    (local_experiment_dir / "EXPERIMENT.md").write_text("# local doc\n", encoding="utf-8")

    remote_manifest = {
        "experiment_id": "exp-1",
        "name": "Remote Experiment",
        "status": "active",
        "created_at": "2026-03-31T00:00:00+00:00",
        "updated_at": "2026-03-31T01:00:00+00:00",
        "hypothesis": "Remote hypothesis",
        "tags": ["gpu", "xgboost"],
        "runs": ["run-a"],
        "metadata": {"champion_run_id": "run-a"},
    }
    run_manifest = {
        "run_id": "run-a",
        "status": "FINISHED",
        "experiment_id": "exp-1",
        "execution": {"backend": "remote_pc"},
        "data": {"target_col": "target_ender_20", "feature_set": "medium"},
        "model": {"type": "XGBRegressor"},
    }

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {
            "experiment": remote_manifest,
            "eligible_finished_run_ids": ["run-a"],
            "skipped_non_finished_run_ids": [],
            "failures": [],
            "remote_manifest_path": r"C:\remote\.numereng\experiments\exp-1\experiment.json",
            "remote_experiment_dir": r"C:\remote\.numereng\experiments\exp-1",
        },
    )
    monkeypatch.setattr(
        remote_service,
        "_copy_remote_runs_to_staging",
        lambda **kwargs: _seed_staging_runs(kwargs["staging_root"], {"run-a": run_manifest}),
    )

    result = remote_service.pull_remote_experiment(target_id="pc", experiment_id="exp-1", store_root=store_root)

    assert result.target_id == "pc"
    assert result.already_materialized_run_ids == ()
    assert result.materialized_run_count == 1
    assert result.materialized_run_ids == ("run-a",)
    assert result.failures == ()
    assert result.local_experiment_manifest_path.is_file()
    manifest = json.loads(result.local_experiment_manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "Remote Experiment"
    assert manifest["runs"] == ["run-a"]
    assert manifest["metadata"]["remote_pull"]["last_target_id"] == "pc"
    assert manifest["metadata"]["remote_pull"]["materialized_finished_run_ids"] == ["run-a"]
    assert (store_root / "runs" / "run-a" / "run.json").is_file()
    assert result.local_runs_root == store_root / "runs"


def test_pull_remote_experiment_returns_blockers_without_materializing_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    remote_manifest = {
        "experiment_id": "exp-2",
        "name": "Remote Experiment",
        "status": "complete",
        "created_at": "2026-03-31T00:00:00+00:00",
        "updated_at": "2026-03-31T01:00:00+00:00",
        "runs": ["run-a", "run-b"],
        "metadata": {},
    }
    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {
            "experiment": remote_manifest,
            "eligible_finished_run_ids": ["run-a"],
            "skipped_non_finished_run_ids": [],
            "failures": [{"run_id": "run-b", "missing_files": ["run_dir"], "reason": "run_dir_missing"}],
            "remote_manifest_path": r"C:\remote\.numereng\experiments\exp-2\experiment.json",
            "remote_experiment_dir": r"C:\remote\.numereng\experiments\exp-2",
        },
    )

    first = remote_service.pull_remote_experiment(target_id="pc", experiment_id="exp-2", store_root=store_root)

    assert len(first.failures) == 1
    assert first.failures[0].run_id == "run-b"
    assert first.failures[0].reason == "run_dir_missing"
    assert first.materialized_run_count == 0
    assert not (store_root / "runs" / "run-a").exists()


def test_pull_remote_experiment_rerun_is_idempotent_for_existing_finished_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    existing_run_dir = store_root / "runs" / "run-a"
    existing_run_dir.mkdir(parents=True)
    (existing_run_dir / "run.json").write_text(json.dumps({"run_id": "run-a", "status": "FINISHED"}), encoding="utf-8")
    (existing_run_dir / "resolved.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
    (existing_run_dir / "results.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (existing_run_dir / "metrics.json").write_text(json.dumps({"bmc": {"mean": 0.1}}), encoding="utf-8")
    remote_manifest = {
        "experiment_id": "exp-3",
        "name": "Remote Experiment",
        "status": "active",
        "runs": ["run-a", "run-b"],
        "metadata": {},
    }

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {
            "experiment": remote_manifest,
            "eligible_finished_run_ids": ["run-a"],
            "skipped_non_finished_run_ids": ["run-b"],
            "failures": [],
            "remote_manifest_path": r"C:\remote\.numereng\experiments\exp-3\experiment.json",
            "remote_experiment_dir": r"C:\remote\.numereng\experiments\exp-3",
        },
    )
    monkeypatch.setattr(
        remote_service,
        "_copy_remote_runs_to_staging",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("copy should not run for idempotent re-pull")),
    )

    result = remote_service.pull_remote_experiment(target_id="pc", experiment_id="exp-3", store_root=store_root)

    assert result.materialized_run_count == 0
    assert result.already_materialized_run_ids == ("run-a",)
    assert result.skipped_non_finished_run_ids == ("run-b",)
    assert result.failures == ()


def test_pull_remote_experiment_fails_on_incomplete_existing_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    existing_run_dir = store_root / "runs" / "run-a"
    existing_run_dir.mkdir(parents=True)
    (existing_run_dir / "run.json").write_text(json.dumps({"run_id": "run-a", "status": "FINISHED"}), encoding="utf-8")
    remote_manifest = {
        "experiment_id": "exp-4",
        "name": "Remote Experiment",
        "status": "active",
        "runs": ["run-a"],
        "metadata": {},
    }

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {
            "experiment": remote_manifest,
            "eligible_finished_run_ids": ["run-a"],
            "skipped_non_finished_run_ids": [],
            "failures": [],
            "remote_manifest_path": r"C:\remote\.numereng\experiments\exp-4\experiment.json",
            "remote_experiment_dir": r"C:\remote\.numereng\experiments\exp-4",
        },
    )

    result = remote_service.pull_remote_experiment(target_id="pc", experiment_id="exp-4", store_root=store_root)

    assert result.materialized_run_count == 0
    assert len(result.failures) == 1
    assert result.failures[0].run_id == "run-a"
    assert result.failures[0].reason == "local_run_incomplete"
    assert set(result.failures[0].missing_files) == {"resolved.json", "results.json", "metrics.json"}


def test_pull_remote_experiment_only_copies_missing_finished_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    existing_run_dir = store_root / "runs" / "run-a"
    existing_run_dir.mkdir(parents=True)
    (existing_run_dir / "run.json").write_text(json.dumps({"run_id": "run-a", "status": "FINISHED"}), encoding="utf-8")
    (existing_run_dir / "resolved.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
    (existing_run_dir / "results.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (existing_run_dir / "metrics.json").write_text(json.dumps({"bmc": {"mean": 0.1}}), encoding="utf-8")
    remote_manifest = {
        "experiment_id": "exp-5",
        "name": "Remote Experiment",
        "status": "active",
        "runs": ["run-a", "run-b"],
        "metadata": {},
    }
    copied_run_ids: list[str] = []

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {
            "experiment": remote_manifest,
            "eligible_finished_run_ids": ["run-a", "run-b"],
            "skipped_non_finished_run_ids": [],
            "failures": [],
            "remote_manifest_path": r"C:\remote\.numereng\experiments\exp-5\experiment.json",
            "remote_experiment_dir": r"C:\remote\.numereng\experiments\exp-5",
        },
    )

    def _fake_copy(**kwargs: object) -> Path:
        copied_run_ids.extend(kwargs["run_ids"])
        return _seed_staging_runs(
            kwargs["staging_root"],
            {
                "run-b": {
                    "run_id": "run-b",
                    "status": "FINISHED",
                    "experiment_id": "exp-5",
                    "execution": {"backend": "remote_pc"},
                }
            },
        )

    monkeypatch.setattr(remote_service, "_copy_remote_runs_to_staging", _fake_copy)

    result = remote_service.pull_remote_experiment(target_id="pc", experiment_id="exp-5", store_root=store_root)

    assert copied_run_ids == ["run-b"]
    assert result.already_materialized_run_ids == ("run-a",)
    assert result.materialized_run_ids == ("run-b",)
    assert result.materialized_run_count == 1
    manifest = json.loads(result.local_experiment_manifest_path.read_text(encoding="utf-8"))
    assert manifest["runs"] == ["run-a", "run-b"]
    assert manifest["metadata"]["remote_pull"]["materialized_finished_run_ids"] == ["run-a", "run-b"]
