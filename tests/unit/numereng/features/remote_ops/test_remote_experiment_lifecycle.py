from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from numereng.features.remote_ops import service as remote_service


def _target():
    return remote_service.SshRemoteTargetProfile(
        id="pc",
        label="Windows remote example",
        ssh_config_host="pc",
        shell="powershell",
        repo_root=r"C:\Users\<you>\remote-access\numereng",
        store_root=r"C:\Users\<you>\remote-access\numereng\.numereng",
    )


def _write_run_plan(store_root: Path, *, experiment_id: str, row_count: int) -> None:
    experiment_dir = store_root.parent / "experiments" / experiment_id
    (experiment_dir / "configs").mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for index in range(1, row_count + 1):
        config_name = f"r1_target_alpha_seed{40 + index}.json"
        (experiment_dir / "configs" / config_name).write_text("{}", encoding="utf-8")
        rows.append(
            f"{index},r1,42,target_alpha,20,experiments/{experiment_id}/configs/{config_name},post_training_core"
        )
    (experiment_dir / "run_plan.csv").write_text(
        "plan_index,round,seed,target,horizon,config_path,score_stage_default\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_remote_launch_experiment_builds_run_plan_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-04-09_remote-exp"
    _write_run_plan(store_root, experiment_id=experiment_id, row_count=5)
    captured: dict[str, object] = {}

    def fake_run_remote_python(target, script_source, *, args):
        captured["args"] = args
        return {"ok": True, "pid": 4321, "launched_at": "2026-04-09T00:00:00+00:00"}

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(remote_service, "_should_sync_repo", lambda target_id, store_root: False)
    monkeypatch.setattr(remote_service, "sync_remote_experiment", lambda **kwargs: None)
    monkeypatch.setattr(remote_service, "_run_remote_python", fake_run_remote_python)

    result = remote_service.remote_launch_experiment(
        target_id="pc",
        experiment_id=experiment_id,
        start_index=2,
        store_root=store_root,
    )

    encoded_command = captured["args"][0]
    assert isinstance(encoded_command, str)
    command = json.loads(base64.b64decode(encoded_command).decode("utf-8"))
    assert command[:5] == ["uv", "run", "numereng", "experiment", "run-plan"]
    assert "--id" in command and experiment_id in command
    assert "--start-index" in command and "2" in command
    assert "--end-index" in command and "5" in command
    assert "--score-stage" in command and "post_training_core" in command
    assert "--workspace" in command and _target().repo_root in command
    assert result.remote_pid == 4321
    assert result.repo_synced is False
    assert result.experiment_synced is True
    assert result.state_path.endswith(f"{experiment_id}__2_5.json")


def test_remote_experiment_status_maps_remote_state_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-04-09_status-exp"
    _write_run_plan(store_root, experiment_id=experiment_id, row_count=3)

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "_run_remote_python",
        lambda *args, **kwargs: {
            "exists": True,
            "phase": "training",
            "current_index": 2,
            "current_run_id": "run-222",
            "current_config_path": r"C:\Users\<you>\remote-access\numereng\.numereng\experiments\exp\configs\row2.json",
            "last_completed_row_index": 1,
            "supervisor_pid": 9001,
            "supervisor_alive": True,
            "active_worker_pid": 7331,
            "last_successful_heartbeat_at": "2026-04-09T00:00:00+00:00",
            "retry_count": 1,
            "failure_classifier": None,
            "terminal_error": None,
            "state": {"phase": "training", "window": {"start_index": 1, "end_index": 3}},
        },
    )

    result = remote_service.remote_experiment_status(
        target_id="pc",
        experiment_id=experiment_id,
        store_root=store_root,
    )

    assert result.exists is True
    assert result.phase == "training"
    assert result.current_index == 2
    assert result.current_run_id == "run-222"
    assert result.supervisor_alive is True
    assert result.active_worker_pid == 7331
    assert result.retry_count == 1
    assert result.raw_state["phase"] == "training"


def test_remote_maintain_experiment_restarts_dead_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(remote_service, "_get_target", lambda target_id: _target())
    monkeypatch.setattr(
        remote_service,
        "remote_experiment_status",
        lambda **kwargs: remote_service.RemoteExperimentStatusResult(
            target_id="pc",
            experiment_id="exp-1",
            state_path=r"C:\Users\<you>\remote-access\numereng\.numereng\remote_ops\experiment_run_plan\exp-1__3_9.json",
            exists=True,
            phase="training",
            current_index=4,
            current_run_id="run-444",
            current_config_path=r"C:\tmp\row4.json",
            last_completed_row_index=3,
            supervisor_pid=991,
            supervisor_alive=False,
            active_worker_pid=None,
            last_successful_heartbeat_at="2026-04-09T00:00:00+00:00",
            retry_count=1,
            failure_classifier="restartable",
            terminal_error=None,
            raw_state={
                "window": {"start_index": 3, "end_index": 9},
                "requested_score_stage": "post_training_full",
            },
        ),
    )

    def fake_run_remote_python(target, script_source, *, args):
        captured["args"] = args
        return {"ok": True, "pid": 5511, "launched_at": "2026-04-09T01:00:00+00:00"}

    monkeypatch.setattr(remote_service, "_run_remote_python", fake_run_remote_python)

    result = remote_service.remote_maintain_experiment(
        target_id="pc",
        experiment_id="exp-1",
        start_index=3,
        end_index=9,
    )

    encoded_command = captured["args"][0]
    assert isinstance(encoded_command, str)
    command = json.loads(base64.b64decode(encoded_command).decode("utf-8"))
    assert "--resume" in command
    assert "--start-index" in command and "3" in command
    assert "--end-index" in command and "9" in command
    assert "--score-stage" in command and "post_training_full" in command
    assert result.action == "restarted"
    assert result.supervisor_pid == 5511
