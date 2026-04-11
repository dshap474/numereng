from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import numereng.api as api_module
from numereng import cli
from numereng.features.telemetry import get_launch_metadata
from numereng.platform.errors import PackageError


def _parse_stdout_json(stdout: str) -> dict[str, Any]:
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
    raise AssertionError("expected JSON object payload in stdout")


def test_cli_main_success(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main([])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["package"] == "numereng"


def test_cli_main_boundary_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--fail"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "bootstrap_check_failed" in captured.err


def test_cli_main_unknown_argument(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--unknown"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown arguments: --unknown" in captured.err


def test_cli_monitor_snapshot_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_build_monitor_snapshot(request: api_module.MonitorSnapshotRequest) -> api_module.MonitorSnapshotResponse:
        assert request.store_root == ".numereng"
        assert request.refresh_cloud is False
        return api_module.MonitorSnapshotResponse(
            generated_at="2026-03-25T00:00:00+00:00",
            source=api_module.MonitorSourceResponse(
                kind="local",
                id="local",
                label="Local store",
                host="host",
                store_root=".numereng",
                state="live",
            ),
            summary=api_module.MonitorSummaryResponse(
                total_experiments=1,
                active_experiments=1,
                completed_experiments=0,
                live_experiments=1,
                live_runs=1,
                queued_runs=0,
                attention_count=0,
            ),
            experiments=[],
            live_experiments=[],
            live_runs=[],
            recent_activity=[],
        )

    monkeypatch.setattr(api_module, "build_monitor_snapshot", fake_build_monitor_snapshot)

    exit_code = cli.main(["monitor", "snapshot", "--no-refresh-cloud", "--json"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["summary"]["live_runs"] == 1


def test_cli_viz_launches_packaged_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: dict[str, object] = {}
    fake_app = object()

    def fake_create_viz_app(request: api_module.VizAppRequest) -> object:
        assert request.workspace_root == "/tmp/numerai-dev"
        return fake_app

    def fake_uvicorn_run(app: object, *, host: str, port: int) -> None:
        launched["app"] = app
        launched["host"] = host
        launched["port"] = port

    monkeypatch.setattr("numereng.cli.commands.viz.create_viz_app", fake_create_viz_app)
    monkeypatch.setattr("numereng.cli.commands.viz.uvicorn.run", fake_uvicorn_run)

    exit_code = cli.main(["viz", "--workspace", "/tmp/numerai-dev", "--host", "0.0.0.0", "--port", "8600"])

    assert exit_code == 0
    assert launched == {"app": fake_app, "host": "0.0.0.0", "port": 8600}


def test_cli_remote_list_json_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_remote_list_targets(request: api_module.RemoteTargetListRequest) -> api_module.RemoteTargetListResponse:
        _ = request
        return api_module.RemoteTargetListResponse(
            targets=[
                api_module.RemoteTargetResponse(
                    id="pc",
                    label="Daniel's PC",
                    kind="ssh",
                    shell="powershell",
                    repo_root=r"C:\Users\dansh\remote-access\numereng",
                    store_root=r"C:\Users\dansh\remote-access\numereng\.numereng",
                    runner_cmd="uv run numereng",
                    python_cmd="uv run python",
                    tags=["pc"],
                )
            ]
        )

    monkeypatch.setattr(api_module, "remote_list_targets", fake_remote_list_targets)

    exit_code = cli.main(["remote", "list", "--format", "json"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["targets"][0]["id"] == "pc"


def test_cli_remote_bootstrap_viz_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_remote_bootstrap_viz(
        request: api_module.RemoteVizBootstrapRequest,
    ) -> api_module.RemoteVizBootstrapResponse:
        assert request.store_root == ".numereng"
        return api_module.RemoteVizBootstrapResponse(
            store_root=".numereng",
            state_path=".numereng/remote_ops/bootstrap/viz.json",
            bootstrapped_at="2026-03-28T00:00:00+00:00",
            ready_count=1,
            degraded_count=1,
            targets=[
                api_module.RemoteVizBootstrapTargetResponse(
                    target=api_module.RemoteTargetResponse(
                        id="pc",
                        label="Daniel's PC",
                        kind="ssh",
                        shell="powershell",
                        repo_root=r"C:\Users\dansh\remote-access\numereng",
                        store_root=r"C:\Users\dansh\remote-access\numereng\.numereng",
                        runner_cmd="uv run numereng",
                        python_cmd="uv run python",
                        tags=["pc"],
                    ),
                    bootstrap_status="ready",
                    last_bootstrap_at="2026-03-28T00:00:00+00:00",
                    last_bootstrap_error=None,
                    repo_synced=True,
                    repo_sync_skipped=False,
                    doctor_ok=True,
                    issues=[],
                )
            ],
        )

    monkeypatch.setattr(api_module, "remote_bootstrap_viz", fake_remote_bootstrap_viz)

    exit_code = cli.main(["remote", "bootstrap-viz"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ready_count"] == 1


def test_cli_remote_run_train_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_remote_train_launch(request: api_module.RemoteTrainLaunchRequest) -> api_module.RemoteTrainLaunchResponse:
        assert request.target_id == "pc"
        assert request.config_path == "configs/run.json"
        assert request.sync_repo == "always"
        assert request.profile == "simple"
        return api_module.RemoteTrainLaunchResponse(
            target_id="pc",
            launch_id="launch-1",
            remote_config_path=r"C:\Users\dansh\remote-access\numereng\.numereng\tmp\remote-configs\run.json",
            remote_log_path=r"C:\Users\dansh\remote-access\numereng\.numereng\remote_ops\launches\launch-1.log",
            remote_metadata_path=r"C:\Users\dansh\remote-access\numereng\.numereng\remote_ops\launches\launch-1.json",
            remote_pid=4321,
            launched_at="2026-03-27T00:00:00+00:00",
            sync_repo_policy="always",
            repo_synced=True,
            experiment_synced=False,
        )

    monkeypatch.setattr(api_module, "remote_train_launch", fake_remote_train_launch)

    exit_code = cli.main(
        [
            "remote",
            "run",
            "train",
            "--target",
            "pc",
            "--config",
            "configs/run.json",
            "--sync-repo",
            "always",
            "--profile",
            "simple",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["launch_id"] == "launch-1"
    assert payload["repo_synced"] is True


def test_cli_remote_experiment_pull_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_remote_experiment_pull(
        request: api_module.RemoteExperimentPullRequest,
    ) -> api_module.RemoteExperimentPullResponse:
        assert request.target_id == "pc"
        assert request.experiment_id == "exp-1"
        return api_module.RemoteExperimentPullResponse(
            target_id="pc",
            experiment_id="exp-1",
            local_experiment_manifest_path="experiments/exp-1/experiment.json",
            local_runs_root=".numereng/runs",
            pulled_at="2026-03-31T00:00:00+00:00",
            already_materialized_run_ids=["run-0"],
            materialized_run_ids=["run-a", "run-b"],
            materialized_run_count=2,
            skipped_non_finished_run_ids=["run-c"],
            failures=[],
        )

    monkeypatch.setattr(api_module, "remote_experiment_pull", fake_remote_experiment_pull)

    exit_code = cli.main(["remote", "experiment", "pull", "--target", "pc", "--experiment-id", "exp-1"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["experiment_id"] == "exp-1"
    assert payload["materialized_run_count"] == 2


def test_cli_experiment_run_plan_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_run_plan(
        request: api_module.ExperimentRunPlanRequest,
    ) -> api_module.ExperimentRunPlanResponse:
        assert request.experiment_id == "2026-04-09_exp"
        assert request.start_index == 3
        assert request.end_index == 9
        assert request.score_stage == "post_training_full"
        assert request.resume is True
        return api_module.ExperimentRunPlanResponse(
            experiment_id="2026-04-09_exp",
            state_path=".numereng/remote_ops/experiment_run_plan/2026-04-09_exp__3_9.json",
            window=api_module.ExperimentRunPlanWindowResponse(start_index=3, end_index=9, total_rows=120),
            phase="training",
            requested_score_stage="post_training_full",
            completed_score_stages=[],
            current_index=3,
            current_round="r1",
            current_config_path="experiments/2026-04-09_exp/configs/r1_target_alpha_seed42.json",
            current_run_id=None,
            last_completed_row_index=2,
            supervisor_pid=1111,
            active_worker_pid=None,
            last_successful_heartbeat_at=None,
            failure_classifier=None,
            retry_count=0,
            terminal_error=None,
            updated_at="2026-04-09T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "experiment_run_plan", fake_experiment_run_plan)

    exit_code = cli.main(
        [
            "experiment",
            "run-plan",
            "--id",
            "2026-04-09_exp",
            "--start-index",
            "3",
            "--end-index",
            "9",
            "--score-stage",
            "post_training_full",
            "--resume",
            "--format",
            "json",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["window"]["start_index"] == 3
    assert payload["phase"] == "training"


def test_cli_remote_experiment_launch_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_remote_experiment_launch(
        request: api_module.RemoteExperimentLaunchRequest,
    ) -> api_module.RemoteExperimentLaunchResponse:
        assert request.target_id == "pc"
        assert request.experiment_id == "2026-04-09_exp"
        assert request.start_index == 2
        assert request.end_index == 12
        assert request.score_stage == "post_training_core"
        assert request.sync_repo == "always"
        return api_module.RemoteExperimentLaunchResponse(
            target_id="pc",
            experiment_id="2026-04-09_exp",
            state_path=r"C:\Users\dansh\remote-access\numereng\.numereng\remote_ops\experiment_run_plan\2026-04-09_exp__2_12.json",
            launch_id="launch-remote-exp",
            remote_log_path=r"C:\Users\dansh\remote-access\numereng\.numereng\remote_ops\launches\launch-remote-exp.log",
            remote_metadata_path=r"C:\Users\dansh\remote-access\numereng\.numereng\remote_ops\launches\launch-remote-exp.json",
            remote_pid=4567,
            launched_at="2026-04-09T00:00:00+00:00",
            repo_synced=True,
            experiment_synced=True,
        )

    monkeypatch.setattr(api_module, "remote_experiment_launch", fake_remote_experiment_launch)

    exit_code = cli.main(
        [
            "remote",
            "experiment",
            "launch",
            "--target",
            "pc",
            "--experiment-id",
            "2026-04-09_exp",
            "--start-index",
            "2",
            "--end-index",
            "12",
            "--score-stage",
            "post_training_core",
            "--sync-repo",
            "always",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["experiment_id"] == "2026-04-09_exp"
    assert payload["remote_pid"] == 4567


def test_cli_run_submit_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_submit_predictions(request: api_module.SubmissionRequest) -> api_module.SubmissionResponse:
        assert request.model_name == "main"
        assert request.run_id == "run-1"
        assert request.tournament == "signals"
        return api_module.SubmissionResponse(
            submission_id="submission-1",
            model_name="main",
            model_id="model-1",
            predictions_path="/tmp/predictions.csv",
            run_id="run-1",
        )

    monkeypatch.setattr(api_module, "submit_predictions", fake_submit_predictions)

    exit_code = cli.main(["run", "submit", "--run-id", "run-1", "--model-name", "main", "--tournament", "signals"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["submission_id"] == "submission-1"
    assert payload["run_id"] == "run-1"


def test_cli_run_submit_allow_non_live_artifact_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_submit_predictions(request: api_module.SubmissionRequest) -> api_module.SubmissionResponse:
        assert request.allow_non_live_artifact is True
        return api_module.SubmissionResponse(
            submission_id="submission-1",
            model_name="main",
            model_id="model-1",
            predictions_path="/tmp/predictions.csv",
            run_id="run-1",
        )

    monkeypatch.setattr(api_module, "submit_predictions", fake_submit_predictions)

    exit_code = cli.main(
        [
            "run",
            "submit",
            "--run-id",
            "run-1",
            "--model-name",
            "main",
            "--allow-non-live-artifact",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["submission_id"] == "submission-1"


def test_cli_run_submit_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "submit", "--run-id", "run-1"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --model-name" in captured.err


def test_cli_run_submit_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_submit_predictions(request: api_module.SubmissionRequest) -> api_module.SubmissionResponse:
        _ = request
        raise PackageError("submission_model_not_found")

    monkeypatch.setattr(api_module, "submit_predictions", fake_submit_predictions)

    exit_code = cli.main(["run", "submit", "--run-id", "run-1", "--model-name", "main"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "submission_model_not_found" in captured.err


def test_cli_run_train_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_training(request: api_module.TrainRunRequest) -> api_module.TrainRunResponse:
        assert request.config_path == "configs/run.json"
        assert request.output_dir == "out"
        assert request.engine_mode is None
        assert request.window_size_eras is None
        assert request.embargo_eras is None
        assert request.post_training_scoring is None
        return api_module.TrainRunResponse(
            run_id="run-123",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
        )

    monkeypatch.setattr(api_module, "run_training", fake_run_training)

    exit_code = cli.main(["run", "train", "--config", "configs/run.json", "--output-dir", "out"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"
    assert payload["predictions_path"] == "/tmp/preds.parquet"
    assert payload["results_path"] == "/tmp/results.json"


def test_cli_run_train_sets_cli_launch_metadata_and_experiment_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_training(request: api_module.TrainRunRequest) -> api_module.TrainRunResponse:
        launch_metadata = get_launch_metadata()
        assert launch_metadata is not None
        assert launch_metadata.source == "cli.run.train"
        assert launch_metadata.operation_type == "run"
        assert launch_metadata.job_type == "run"
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.TrainRunResponse(
            run_id="run-123",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
        )

    monkeypatch.setattr(api_module, "run_training", fake_run_training)

    exit_code = cli.main(
        [
            "run",
            "train",
            "--config",
            "configs/run.json",
            "--experiment-id",
            "2026-02-22_test-exp",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"


def test_cli_run_train_profile_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_training(request: api_module.TrainRunRequest) -> api_module.TrainRunResponse:
        assert request.profile == "simple"
        assert request.engine_mode is None
        assert request.window_size_eras is None
        assert request.embargo_eras is None
        return api_module.TrainRunResponse(
            run_id="run-456",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
        )

    monkeypatch.setattr(api_module, "run_training", fake_run_training)

    exit_code = cli.main(
        [
            "run",
            "train",
            "--config",
            "configs/run.json",
            "--profile",
            "simple",
        ]
    )
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "run-456"
    assert payload["results_path"] == "/tmp/results.json"


def test_cli_run_train_post_training_scoring_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_training(request: api_module.TrainRunRequest) -> api_module.TrainRunResponse:
        assert request.post_training_scoring == "full"
        return api_module.TrainRunResponse(
            run_id="run-789",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
        )

    monkeypatch.setattr(api_module, "run_training", fake_run_training)

    exit_code = cli.main(
        [
            "run",
            "train",
            "--config",
            "configs/run.json",
            "--post-training-scoring",
            "full",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "run-789"


def test_cli_run_train_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "train"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --config" in captured.err


def test_cli_run_train_invalid_post_training_scoring(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "run",
            "train",
            "--config",
            "configs/run.json",
            "--post-training-scoring",
            "bad_value",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "invalid value for --post-training-scoring" in captured.err


def test_cli_run_train_invalid_engine_mode(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "train", "--config", "configs/run.json", "--engine-mode", "unknown"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "legacy training option is no longer supported: --engine-mode; use --profile" in captured.err


def test_cli_run_train_rejects_non_json_config(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "train", "--config", "configs/run.yaml"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "config_path must reference a .json file" in captured.err


def test_cli_run_train_legacy_engine_mode_flag_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "train", "--config", "configs/run.json", "--engine-mode", "custom"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "legacy training option is no longer supported: --engine-mode; use --profile" in captured.err


def test_cli_run_train_legacy_knobs_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "run",
            "train",
            "--config",
            "configs/run.json",
            "--window-size-eras",
            "144",
            "--embargo-eras",
            "9",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "legacy training option is no longer supported: --window-size-eras; use --profile" in captured.err


def test_cli_run_train_legacy_method_flag_hard_fails(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "train", "--config", "configs/run.json", "--method", "official_walkforward"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "legacy training option is no longer supported: --method" in captured.err


def test_cli_run_score_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_score_run(request: api_module.ScoreRunRequest) -> api_module.ScoreRunResponse:
        assert request.run_id == "run-123"
        assert request.store_root == ".numereng"
        assert request.stage == "all"
        return api_module.ScoreRunResponse(
            run_id="run-123",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
            metrics_path="/tmp/metrics.json",
            score_provenance_path="/tmp/score_provenance.json",
            requested_stage="all",
            refreshed_stages=["run_metric_series", "post_training_core"],
        )

    monkeypatch.setattr(api_module, "score_run", fake_score_run)

    exit_code = cli.main(["run", "score", "--run-id", "run-123"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"
    assert payload["metrics_path"] == "/tmp/metrics.json"
    assert payload["requested_stage"] == "all"


def test_cli_run_score_stage_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_score_run(request: api_module.ScoreRunRequest) -> api_module.ScoreRunResponse:
        assert request.stage == "post_training_core"
        return api_module.ScoreRunResponse(
            run_id="run-123",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
            metrics_path="/tmp/metrics.json",
            score_provenance_path="/tmp/score_provenance.json",
            requested_stage="post_training_core",
            refreshed_stages=["post_training_core"],
        )

    monkeypatch.setattr(api_module, "score_run", fake_score_run)

    exit_code = cli.main(["run", "score", "--run-id", "run-123", "--stage", "post_training_core"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["requested_stage"] == "post_training_core"


def test_cli_run_score_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "score"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --run-id" in captured.err


def test_cli_run_score_invalid_stage(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "score", "--run-id", "run-123", "--stage", "unknown"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "invalid value for --stage" in captured.err


def test_cli_run_score_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_score_run(request: api_module.ScoreRunRequest) -> api_module.ScoreRunResponse:
        _ = request
        raise PackageError("training_score_run_not_found")

    monkeypatch.setattr(api_module, "score_run", fake_score_run)

    exit_code = cli.main(["run", "score", "--run-id", "run-404"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "training_score_run_not_found" in captured.err


def test_cli_run_cancel_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cancel_run(request: api_module.RunCancelRequest) -> api_module.RunCancelResponse:
        assert request.run_id == "run-123"
        assert request.store_root == ".numereng"
        return api_module.RunCancelResponse(
            run_id="run-123",
            job_id="job-123",
            status="running",
            cancel_requested=True,
            cancel_requested_at="2026-03-24T00:00:00+00:00",
            accepted=True,
        )

    monkeypatch.setattr(api_module, "cancel_run", fake_cancel_run)

    exit_code = cli.main(["run", "cancel", "--run-id", "run-123"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"
    assert payload["accepted"] is True


def test_cli_run_unknown_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "unknown"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unknown arguments: run unknown" in captured.err


def test_cli_experiment_create_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_create(request: api_module.ExperimentCreateRequest) -> api_module.ExperimentResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        assert request.tags == ["quick", "baseline"]
        return api_module.ExperimentResponse(
            experiment_id=request.experiment_id,
            name="Test Experiment",
            status="draft",
            hypothesis="Test hypothesis",
            tags=request.tags,
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
            champion_run_id=None,
            runs=[],
            metadata={},
            manifest_path="/tmp/experiments/2026-02-22_test-exp/experiment.json",
        )

    monkeypatch.setattr(api_module, "experiment_create", fake_experiment_create)

    exit_code = cli.main(
        [
            "experiment",
            "create",
            "--id",
            "2026-02-22_test-exp",
            "--name",
            "Test Experiment",
            "--hypothesis",
            "Test hypothesis",
            "--tags",
            "quick,baseline",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["experiment_id"] == "2026-02-22_test-exp"
    assert payload["status"] == "draft"


def test_cli_experiment_train_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_train(request: api_module.ExperimentTrainRequest) -> api_module.ExperimentTrainResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        assert request.config_path == "configs/run.json"
        assert request.profile == "purged_walk_forward"
        assert request.post_training_scoring is None
        assert request.engine_mode is None
        assert request.window_size_eras is None
        assert request.embargo_eras is None
        return api_module.ExperimentTrainResponse(
            experiment_id=request.experiment_id,
            run_id="run-123",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
        )

    monkeypatch.setattr(api_module, "experiment_train", fake_experiment_train)

    exit_code = cli.main(
        [
            "experiment",
            "train",
            "--id",
            "2026-02-22_test-exp",
            "--config",
            "configs/run.json",
            "--profile",
            "purged_walk_forward",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"
    assert payload["experiment_id"] == "2026-02-22_test-exp"


def test_cli_experiment_train_post_training_scoring_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_train(request: api_module.ExperimentTrainRequest) -> api_module.ExperimentTrainResponse:
        assert request.post_training_scoring == "round_full"
        return api_module.ExperimentTrainResponse(
            experiment_id=request.experiment_id,
            run_id="run-124",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
        )

    monkeypatch.setattr(api_module, "experiment_train", fake_experiment_train)

    exit_code = cli.main(
        [
            "experiment",
            "train",
            "--id",
            "2026-02-22_test-exp",
            "--config",
            "configs/run.json",
            "--post-training-scoring",
            "round_full",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "run-124"


def test_cli_experiment_train_sets_cli_launch_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_train(request: api_module.ExperimentTrainRequest) -> api_module.ExperimentTrainResponse:
        launch_metadata = get_launch_metadata()
        assert launch_metadata is not None
        assert launch_metadata.source == "cli.experiment.train"
        assert launch_metadata.operation_type == "run"
        assert launch_metadata.job_type == "run"
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.ExperimentTrainResponse(
            experiment_id=request.experiment_id,
            run_id="run-123",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
        )

    monkeypatch.setattr(api_module, "experiment_train", fake_experiment_train)

    exit_code = cli.main(
        [
            "experiment",
            "train",
            "--id",
            "2026-02-22_test-exp",
            "--config",
            "configs/run.json",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"


def test_cli_experiment_score_round_sets_cli_launch_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_score_round(
        request: api_module.ExperimentScoreRoundRequest,
    ) -> api_module.ExperimentScoreRoundResponse:
        launch_metadata = get_launch_metadata()
        assert launch_metadata is not None
        assert launch_metadata.source == "cli.experiment.score-round"
        assert launch_metadata.operation_type == "run"
        assert launch_metadata.job_type == "run"
        assert request.experiment_id == "2026-02-22_test-exp"
        assert request.round == "r1"
        assert request.stage == "post_training_full"
        return api_module.ExperimentScoreRoundResponse(
            experiment_id=request.experiment_id,
            round=request.round,
            stage=request.stage,
            run_ids=["run-1", "run-2"],
        )

    monkeypatch.setattr(api_module, "experiment_score_round", fake_experiment_score_round)

    exit_code = cli.main(
        [
            "experiment",
            "score-round",
            "--id",
            "2026-02-22_test-exp",
            "--round",
            "r1",
            "--stage",
            "post_training_full",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["experiment_id"] == "2026-02-22_test-exp"
    assert payload["round"] == "r1"
    assert payload["stage"] == "post_training_full"
    assert payload["run_ids"] == ["run-1", "run-2"]


def test_cli_experiment_report_json_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_report(request: api_module.ExperimentReportRequest) -> api_module.ExperimentReportResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        assert request.metric == "bmc_last_200_eras.mean"
        assert request.limit == 5
        return api_module.ExperimentReportResponse(
            experiment_id=request.experiment_id,
            metric=request.metric,
            total_runs=1,
            champion_run_id="run-1",
            rows=[
                api_module.ExperimentReportRowResponse(
                    run_id="run-1",
                    metric_value=0.12,
                    corr_mean=0.11,
                    bmc_last_200_eras_mean=0.12,
                    is_champion=True,
                )
            ],
        )

    monkeypatch.setattr(api_module, "experiment_report", fake_experiment_report)

    exit_code = cli.main(
        [
            "experiment",
            "report",
            "--id",
            "2026-02-22_test-exp",
            "--limit",
            "5",
            "--format",
            "json",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["experiment_id"] == "2026-02-22_test-exp"
    assert payload["rows"][0]["run_id"] == "run-1"


def test_cli_experiment_list_table_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_list(request: api_module.ExperimentListRequest) -> api_module.ExperimentListResponse:
        assert request.status == "active"
        return api_module.ExperimentListResponse(
            experiments=[
                api_module.ExperimentResponse(
                    experiment_id="2026-02-22_test-exp",
                    name="Test Experiment",
                    status="active",
                    created_at="2026-02-22T00:00:00+00:00",
                    updated_at="2026-02-22T00:05:00+00:00",
                    champion_run_id="run-1",
                    runs=["run-1"],
                    manifest_path="/tmp/experiments/2026-02-22_test-exp/experiment.json",
                )
            ]
        )

    monkeypatch.setattr(api_module, "experiment_list", fake_experiment_list)

    exit_code = cli.main(["experiment", "list", "--status", "active"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ID" in captured.out
    assert "2026-02-22_test-exp" in captured.out


def test_cli_experiment_details_table_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_get(request: api_module.ExperimentGetRequest) -> api_module.ExperimentResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.ExperimentResponse(
            experiment_id=request.experiment_id,
            name="Test Experiment",
            status="active",
            hypothesis="Track uplift",
            tags=["baseline"],
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:05:00+00:00",
            champion_run_id="run-1",
            runs=["run-1", "run-2"],
            manifest_path="/tmp/experiments/2026-02-22_test-exp/experiment.json",
        )

    monkeypatch.setattr(api_module, "experiment_get", fake_experiment_get)

    exit_code = cli.main(["experiment", "details", "--id", "2026-02-22_test-exp"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "experiment_id: 2026-02-22_test-exp" in captured.out
    assert "runs: 2" in captured.out


def test_cli_experiment_archive_and_unarchive_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_archive(request: api_module.ExperimentArchiveRequest) -> api_module.ExperimentArchiveResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.ExperimentArchiveResponse(
            experiment_id=request.experiment_id,
            status="archived",
            manifest_path="/tmp/experiments/_archive/2026-02-22_test-exp/experiment.json",
            archived=True,
        )

    def fake_experiment_unarchive(request: api_module.ExperimentArchiveRequest) -> api_module.ExperimentArchiveResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.ExperimentArchiveResponse(
            experiment_id=request.experiment_id,
            status="active",
            manifest_path="/tmp/experiments/2026-02-22_test-exp/experiment.json",
            archived=False,
        )

    monkeypatch.setattr(api_module, "experiment_archive", fake_experiment_archive)
    monkeypatch.setattr(api_module, "experiment_unarchive", fake_experiment_unarchive)

    exit_code = cli.main(["experiment", "archive", "--id", "2026-02-22_test-exp"])
    archived_payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert archived_payload["archived"] is True
    assert archived_payload["status"] == "archived"

    exit_code = cli.main(["experiment", "unarchive", "--id", "2026-02-22_test-exp"])
    restored_payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert restored_payload["archived"] is False
    assert restored_payload["status"] == "active"


def test_cli_experiment_promote_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_promote(request: api_module.ExperimentPromoteRequest) -> api_module.ExperimentPromoteResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.ExperimentPromoteResponse(
            experiment_id=request.experiment_id,
            champion_run_id="run-2",
            metric=request.metric,
            metric_value=0.15,
            auto_selected=True,
        )

    monkeypatch.setattr(api_module, "experiment_promote", fake_experiment_promote)

    exit_code = cli.main(["experiment", "promote", "--id", "2026-02-22_test-exp"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["champion_run_id"] == "run-2"


def test_cli_experiment_report_table_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_report(request: api_module.ExperimentReportRequest) -> api_module.ExperimentReportResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.ExperimentReportResponse(
            experiment_id=request.experiment_id,
            metric=request.metric,
            total_runs=1,
            champion_run_id="run-1",
            rows=[
                api_module.ExperimentReportRowResponse(
                    run_id="run-1",
                    metric_value=0.12,
                    corr_mean=0.1,
                    mmc_mean=0.02,
                    cwmm_mean=0.03,
                    bmc_mean=0.11,
                    bmc_last_200_eras_mean=0.12,
                    is_champion=True,
                )
            ],
        )

    monkeypatch.setattr(api_module, "experiment_report", fake_experiment_report)

    exit_code = cli.main(["experiment", "report", "--id", "2026-02-22_test-exp"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "experiment=2026-02-22_test-exp" in captured.out
    assert "run-1" in captured.out


def test_cli_experiment_pack_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_experiment_pack(request: api_module.ExperimentPackRequest) -> api_module.ExperimentPackResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.ExperimentPackResponse(
            experiment_id=request.experiment_id,
            output_path="/tmp/experiments/2026-02-22_test-exp/EXPERIMENT.pack.md",
            experiment_path="/tmp/experiments/2026-02-22_test-exp",
            source_markdown_path="/tmp/experiments/2026-02-22_test-exp/EXPERIMENT.md",
            run_count=2,
            packed_at="2026-02-22T00:05:00+00:00",
        )

    monkeypatch.setattr(api_module, "experiment_pack", fake_experiment_pack)

    exit_code = cli.main(["experiment", "pack", "--id", "2026-02-22_test-exp"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_count"] == 2
    assert payload["output_path"].endswith("EXPERIMENT.pack.md")


def test_cli_experiment_pack_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["experiment", "pack"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --id" in captured.err


def test_cli_experiment_promote_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["experiment", "promote"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --id" in captured.err


def test_cli_store_init_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_store_init(request: api_module.StoreInitRequest) -> api_module.StoreInitResponse:
        assert request.store_root == ".numereng"
        return api_module.StoreInitResponse(
            store_root="/tmp/.numereng",
            db_path="/tmp/.numereng/numereng.db",
            created=True,
            schema_migration="2026_02_store_index_v3_experiments",
        )

    monkeypatch.setattr(api_module, "store_init", fake_store_init)

    exit_code = cli.main(["store", "init"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["created"] is True
    assert payload["schema_migration"] == "2026_02_store_index_v3_experiments"


def test_cli_store_index_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_store_index_run(request: api_module.StoreIndexRequest) -> api_module.StoreIndexResponse:
        assert request.run_id == "run-1"
        assert request.store_root == ".numereng"
        return api_module.StoreIndexResponse(
            run_id="run-1",
            status="FINISHED",
            metrics_indexed=7,
            artifacts_indexed=4,
            run_path="/tmp/.numereng/runs/run-1",
            warnings=[],
        )

    monkeypatch.setattr(api_module, "store_index_run", fake_store_index_run)

    exit_code = cli.main(["store", "index", "--run-id", "run-1"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "run-1"
    assert payload["status"] == "FINISHED"


def test_cli_store_index_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["store", "index"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --run-id" in captured.err


def test_cli_store_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_store_rebuild(request: api_module.StoreRebuildRequest) -> api_module.StoreRebuildResponse:
        _ = request
        raise PackageError("store_run_manifest_invalid_json:run-bad")

    monkeypatch.setattr(api_module, "store_rebuild", fake_store_rebuild)

    exit_code = cli.main(["store", "rebuild"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "store_run_manifest_invalid_json:run-bad" in captured.err


def test_cli_store_doctor_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_store_doctor(request: api_module.StoreDoctorRequest) -> api_module.StoreDoctorResponse:
        assert request.store_root == ".numereng"
        assert request.fix_strays is False
        return api_module.StoreDoctorResponse(
            store_root="/tmp/.numereng",
            db_path="/tmp/.numereng/numereng.db",
            ok=True,
            issues=[],
            stats={"filesystem_runs": 1, "indexed_runs": 1},
            stray_cleanup_applied=False,
            deleted_paths=[],
            missing_paths=[],
        )

    monkeypatch.setattr(api_module, "store_doctor", fake_store_doctor)

    exit_code = cli.main(["store", "doctor"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["stats"]["indexed_runs"] == 1


def test_cli_store_doctor_fix_strays_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_store_doctor(request: api_module.StoreDoctorRequest) -> api_module.StoreDoctorResponse:
        assert request.store_root == ".numereng"
        assert request.fix_strays is True
        return api_module.StoreDoctorResponse(
            store_root="/tmp/.numereng",
            db_path="/tmp/.numereng/numereng.db",
            ok=True,
            issues=[],
            stats={"filesystem_runs": 1, "indexed_runs": 1},
            stray_cleanup_applied=True,
            deleted_paths=["/tmp/.numereng/modal_smoke_data"],
            missing_paths=["/tmp/.numereng/smoke_live_check"],
        )

    monkeypatch.setattr(api_module, "store_doctor", fake_store_doctor)

    exit_code = cli.main(["store", "doctor", "--fix-strays"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["stray_cleanup_applied"] is True
    assert payload["deleted_paths"] == ["/tmp/.numereng/modal_smoke_data"]


def test_cli_store_backfill_run_execution_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_store_backfill(
        request: api_module.StoreRunExecutionBackfillRequest,
    ) -> api_module.StoreRunExecutionBackfillResponse:
        assert request.store_root == ".numereng"
        assert request.run_id is None
        assert request.all is True
        return api_module.StoreRunExecutionBackfillResponse(
            store_root="/tmp/.numereng",
            scanned_count=3,
            updated_count=2,
            skipped_count=1,
            ambiguous_runs=["run-3"],
            updated_run_ids=["run-1", "run-2"],
            skipped_run_ids=["run-3"],
        )

    monkeypatch.setattr(api_module, "store_backfill_run_execution", fake_store_backfill)

    exit_code = cli.main(["store", "backfill-run-execution", "--all"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["updated_count"] == 2
    assert payload["ambiguous_runs"] == ["run-3"]


def test_cli_store_repair_run_lifecycles_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_store_repair_run_lifecycles(
        request: api_module.StoreRunLifecycleRepairRequest,
    ) -> api_module.StoreRunLifecycleRepairResponse:
        assert request.store_root == ".numereng"
        assert request.run_id is None
        assert request.active_only is True
        return api_module.StoreRunLifecycleRepairResponse(
            store_root="/tmp/.numereng",
            scanned_count=2,
            unchanged_count=1,
            reconciled_count=1,
            reconciled_stale_count=1,
            reconciled_canceled_count=0,
            run_ids=["run-stale"],
        )

    monkeypatch.setattr(api_module, "store_repair_run_lifecycles", fake_store_repair_run_lifecycles)

    exit_code = cli.main(["store", "repair-run-lifecycles"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["reconciled_stale_count"] == 1
    assert payload["run_ids"] == ["run-stale"]


def test_cli_store_materialize_viz_artifacts_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_materialize(
        request: api_module.StoreMaterializeVizArtifactsRequest,
    ) -> api_module.StoreMaterializeVizArtifactsResponse:
        assert request.kind == "per-era-corr"
        assert request.experiment_id == "exp-1"
        assert request.run_id is None
        assert request.all is False
        return api_module.StoreMaterializeVizArtifactsResponse(
            store_root="/tmp/.numereng",
            kind="per-era-corr",
            scoped_run_count=2,
            created_count=1,
            skipped_count=1,
            failed_count=0,
            failures=[],
        )

    monkeypatch.setattr(api_module, "store_materialize_viz_artifacts", fake_materialize)

    exit_code = cli.main(
        [
            "store",
            "materialize-viz-artifacts",
            "--kind",
            "per-era-corr",
            "--experiment-id",
            "exp-1",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["created_count"] == 1
    assert payload["skipped_count"] == 1


def test_cli_numerai_datasets_list_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_list_numerai_datasets(
        request: api_module.NumeraiDatasetListRequest,
    ) -> api_module.NumeraiDatasetListResponse:
        assert request.round_num == 12
        assert request.tournament == "crypto"
        return api_module.NumeraiDatasetListResponse(datasets=["v5.2/train_int8.parquet"])

    monkeypatch.setattr(api_module, "list_numerai_datasets", fake_list_numerai_datasets)

    exit_code = cli.main(["numerai", "datasets", "list", "--round", "12", "--tournament", "crypto"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["datasets"] == ["v5.2/train_int8.parquet"]


def test_cli_numerai_datasets_download_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_download_numerai_dataset(
        request: api_module.NumeraiDatasetDownloadRequest,
    ) -> api_module.NumeraiDatasetDownloadResponse:
        assert request.filename == "v5.2/train_int8.parquet"
        assert request.dest_path == "cache/train.parquet"
        assert request.round_num == 5
        assert request.tournament == "signals"
        return api_module.NumeraiDatasetDownloadResponse(path="cache/train.parquet")

    monkeypatch.setattr(api_module, "download_numerai_dataset", fake_download_numerai_dataset)

    exit_code = cli.main(
        [
            "numerai",
            "datasets",
            "download",
            "--filename",
            "v5.2/train_int8.parquet",
            "--dest-path",
            "cache/train.parquet",
            "--round",
            "5",
            "--tournament",
            "signals",
        ]
    )
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["path"] == "cache/train.parquet"


def test_cli_numerai_models_list_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_list_numerai_models(
        request: api_module.NumeraiModelsRequest,
    ) -> api_module.NumeraiModelsResponse:
        assert request.tournament == "signals"
        return api_module.NumeraiModelsResponse(models={"main": "model-1"})

    monkeypatch.setattr(api_module, "list_numerai_models", fake_list_numerai_models)

    exit_code = cli.main(["numerai", "models", "list", "--tournament", "signals"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["models"] == {"main": "model-1"}


def test_cli_numerai_round_current_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_get_numerai_current_round(
        request: api_module.NumeraiCurrentRoundRequest,
    ) -> api_module.NumeraiCurrentRoundResponse:
        assert request.tournament == "crypto"
        return api_module.NumeraiCurrentRoundResponse(round_num=777)

    monkeypatch.setattr(api_module, "get_numerai_current_round", fake_get_numerai_current_round)

    exit_code = cli.main(["numerai", "round", "current", "--tournament", "crypto"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["round_num"] == 777


def test_cli_numerai_round_without_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["numerai", "round"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "numerai round current" in captured.out


def test_cli_numerai_forum_without_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["numerai", "forum"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "numerai forum scrape" in captured.out


def test_cli_numerai_forum_scrape_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_scrape_numerai_forum(
        *,
        output_dir: str,
        state_path: str | None,
        full_refresh: bool,
    ) -> api_module.NumeraiForumScrapeResponse:
        assert output_dir == "docs/numerai/forum"
        assert state_path == "tmp/state.json"
        assert full_refresh is True
        return api_module.NumeraiForumScrapeResponse(
            output_dir="docs/numerai/forum",
            posts_dir="docs/numerai/forum/posts",
            index_path="docs/numerai/forum/INDEX.md",
            manifest_path="docs/numerai/forum/.forum_scraper_manifest.json",
            state_path="tmp/state.json",
            mode="full",
            pages_fetched=3,
            fetched_posts=150,
            new_posts=150,
            total_posts=150,
            latest_post_id=150,
            oldest_post_id=1,
            started_at="2026-03-02T00:00:00Z",
            completed_at="2026-03-02T00:01:00Z",
        )

    monkeypatch.setattr(api_module, "scrape_numerai_forum", fake_scrape_numerai_forum)

    exit_code = cli.main(
        [
            "numerai",
            "forum",
            "scrape",
            "--output-dir",
            "docs/numerai/forum",
            "--state-path",
            "tmp/state.json",
            "--full",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "full"
    assert payload["total_posts"] == 150


def test_cli_numerai_forum_scrape_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["numerai", "forum", "scrape", "--output-dir"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing value for --output-dir" in captured.err


def test_cli_numerai_forum_scrape_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_scrape_numerai_forum(
        *,
        output_dir: str,
        state_path: str | None,
        full_refresh: bool,
    ) -> api_module.NumeraiForumScrapeResponse:
        _ = (output_dir, state_path, full_refresh)
        raise PackageError("forum_scraper_network_error")

    monkeypatch.setattr(api_module, "scrape_numerai_forum", fake_scrape_numerai_forum)

    exit_code = cli.main(["numerai", "forum", "scrape"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "forum_scraper_network_error" in captured.err


def test_cli_numerai_datasets_download_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["numerai", "datasets", "download"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --filename" in captured.err


def test_cli_numerai_datasets_list_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["numerai", "datasets", "list", "--round", "abc"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "invalid integer for --round: abc" in captured.err


def test_cli_numerai_datasets_build_quantized_is_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["numerai", "datasets", "build-quantized"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unknown arguments: numerai datasets build-quantized" in captured.err


def test_cli_numerai_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_list_numerai_models(
        request: api_module.NumeraiModelsRequest,
    ) -> api_module.NumeraiModelsResponse:
        _ = request
        raise PackageError("numerai_get_models_failed")

    monkeypatch.setattr(api_module, "list_numerai_models", fake_list_numerai_models)

    exit_code = cli.main(["numerai", "models"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "numerai_get_models_failed" in captured.err


def test_cli_numerai_unknown_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["numerai", "unknown"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unknown arguments: numerai unknown" in captured.err


def test_cli_cloud_ec2_provision_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_provision(request: api_module.Ec2ProvisionRequest) -> api_module.CloudEc2Response:
        assert request.run_id == "run-1"
        assert request.tier == "large"
        assert request.use_spot is False
        assert request.state_path == "tmp/state.json"
        return api_module.CloudEc2Response(
            action="cloud.ec2.provision",
            message="Instance provisioned and SSM-ready",
            result={"instance_id": "i-abc123"},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_provision", fake_cloud_ec2_provision)

    exit_code = cli.main(
        [
            "cloud",
            "ec2",
            "provision",
            "--run-id",
            "run-1",
            "--tier",
            "large",
            "--on-demand",
            "--state-path",
            "tmp/state.json",
        ]
    )
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["action"] == "cloud.ec2.provision"
    assert payload["result"]["instance_id"] == "i-abc123"


def test_cli_cloud_ec2_push_allows_state_only_and_surfaces_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_push(request: api_module.Ec2PushRequest) -> api_module.CloudEc2Response:
        assert request.instance_id is None
        raise PackageError("missing required value: instance_id")

    monkeypatch.setattr(api_module, "cloud_ec2_push", fake_cloud_ec2_push)

    exit_code = cli.main(["cloud", "ec2", "push", "--state-path", "tmp/state.json"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "missing required value: instance_id" in captured.err


def test_cli_cloud_ec2_s3_list_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_s3_list(request: api_module.Ec2S3ListRequest) -> api_module.CloudEc2Response:
        assert request.prefix == "runs/run-1/"
        return api_module.CloudEc2Response(
            action="cloud.ec2.s3.ls",
            message="S3 objects listed",
            result={"keys": ["runs/run-1/run.json"]},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_s3_list", fake_cloud_ec2_s3_list)

    exit_code = cli.main(["cloud", "ec2", "s3", "ls", "--prefix", "runs/run-1/"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["action"] == "cloud.ec2.s3.ls"
    assert payload["result"]["keys"] == ["runs/run-1/run.json"]


def test_cli_cloud_ec2_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_status(request: api_module.Ec2StatusRequest) -> api_module.CloudEc2Response:
        _ = request
        raise PackageError("cloud_status_failed")

    monkeypatch.setattr(api_module, "cloud_ec2_status", fake_cloud_ec2_status)

    exit_code = cli.main(["cloud", "ec2", "status", "--run-id", "run-1"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "cloud_status_failed" in captured.err


def test_cli_cloud_ec2_init_iam_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_init_iam(request: api_module.Ec2InitIamRequest) -> api_module.CloudEc2Response:
        assert request.region == "us-east-2"
        return api_module.CloudEc2Response(
            action="cloud.ec2.init-iam",
            message="IAM and network bootstrap complete",
            result={"security_group_id": "sg-1"},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_init_iam", fake_cloud_ec2_init_iam)
    exit_code = cli.main(["cloud", "ec2", "init-iam", "--region", "us-east-2"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.ec2.init-iam"


def test_cli_cloud_ec2_setup_data_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_setup_data(request: api_module.Ec2SetupDataRequest) -> api_module.CloudEc2Response:
        assert request.data_version == "v5.2"
        return api_module.CloudEc2Response(
            action="cloud.ec2.setup-data",
            message="Data synchronized to S3",
            result={"uploaded": {"train.parquet": "s3://bucket/train.parquet"}},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_setup_data", fake_cloud_ec2_setup_data)
    exit_code = cli.main(["cloud", "ec2", "setup-data", "--data-version", "v5.2"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.ec2.setup-data"


def test_cli_cloud_ec2_package_build_upload_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_package_build_upload(
        request: api_module.Ec2PackageBuildUploadRequest,
    ) -> api_module.CloudEc2Response:
        assert request.run_id == "run-8"
        return api_module.CloudEc2Response(
            action="cloud.ec2.package.build-upload",
            message="Package artifacts uploaded",
            result={"uploaded": {"requirements.txt": "s3://bucket/req.txt"}},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_package_build_upload", fake_cloud_ec2_package_build_upload)
    exit_code = cli.main(["cloud", "ec2", "package", "build-upload", "--run-id", "run-8"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.ec2.package.build-upload"


def test_cli_cloud_ec2_config_upload_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["cloud", "ec2", "config", "upload", "--run-id", "run-1"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "missing required argument: --config" in captured.err


def test_cli_cloud_ec2_config_upload_rejects_non_json_config(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["cloud", "ec2", "config", "upload", "--config", "run.yaml", "--run-id", "run-1"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "config_path must reference a .json file" in captured.err


def test_cli_cloud_ec2_config_upload_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "run.json"
    config_path.write_text("run_input:\n  model:\n    type: lgbm\n", encoding="utf-8")

    def fake_cloud_ec2_config_upload(request: api_module.Ec2ConfigUploadRequest) -> api_module.CloudEc2Response:
        assert request.config_path == str(config_path)
        return api_module.CloudEc2Response(
            action="cloud.ec2.config.upload",
            message="Run config uploaded",
            result={"config_uri": "s3://bucket/runs/run-1/config.json"},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_config_upload", fake_cloud_ec2_config_upload)
    exit_code = cli.main(["cloud", "ec2", "config", "upload", "--config", str(config_path), "--run-id", "run-1"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.ec2.config.upload"


def test_cli_cloud_ec2_train_start_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_train_start(request: api_module.Ec2TrainStartRequest) -> api_module.CloudEc2Response:
        assert request.instance_id == "i-77"
        return api_module.CloudEc2Response(
            action="cloud.ec2.train.start",
            message="Remote training started",
            result={"training_pid": 1234},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_train_start", fake_cloud_ec2_train_start)
    exit_code = cli.main(["cloud", "ec2", "train", "start", "--instance-id", "i-77", "--run-id", "run-77"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["result"]["training_pid"] == 1234


def test_cli_cloud_ec2_train_poll_invalid_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["cloud", "ec2", "train", "poll", "--timeout-seconds", "bad"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid integer for --timeout-seconds: bad" in captured.err


def test_cli_cloud_ec2_logs_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_logs(request: api_module.Ec2LogsRequest) -> api_module.CloudEc2Response:
        assert request.lines == 200
        assert request.follow is True
        return api_module.CloudEc2Response(
            action="cloud.ec2.logs",
            message="Fetched remote log tail",
            result={"log": "ok"},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_logs", fake_cloud_ec2_logs)
    exit_code = cli.main(["cloud", "ec2", "logs", "--lines", "200", "--follow", "--instance-id", "i-9"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.ec2.logs"


def test_cli_cloud_ec2_pull_and_terminate_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_pull(request: api_module.Ec2PullRequest) -> api_module.CloudEc2Response:
        assert request.run_id == "run-9"
        return api_module.CloudEc2Response(
            action="cloud.ec2.pull",
            message="Run artifacts synchronized to local output",
            result={"downloaded_count": 3},
        )

    def fake_cloud_ec2_terminate(request: api_module.Ec2TerminateRequest) -> api_module.CloudEc2Response:
        assert request.instance_id == "i-9"
        return api_module.CloudEc2Response(
            action="cloud.ec2.terminate",
            message="Instance termination requested",
            result={"instance_id": "i-9"},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_pull", fake_cloud_ec2_pull)
    monkeypatch.setattr(api_module, "cloud_ec2_terminate", fake_cloud_ec2_terminate)

    pull_exit = cli.main(["cloud", "ec2", "pull", "--instance-id", "i-9", "--run-id", "run-9"])
    pull_payload = _parse_stdout_json(capsys.readouterr().out)
    assert pull_exit == 0
    assert pull_payload["result"]["downloaded_count"] == 3

    term_exit = cli.main(["cloud", "ec2", "terminate", "--instance-id", "i-9"])
    term_payload = _parse_stdout_json(capsys.readouterr().out)
    assert term_exit == 0
    assert term_payload["action"] == "cloud.ec2.terminate"


def test_cli_cloud_ec2_s3_cp_parse_errors(capsys: pytest.CaptureFixture[str]) -> None:
    missing_src_exit = cli.main(["cloud", "ec2", "s3", "cp", "--dst", "s3://bucket/out"])
    missing_src = capsys.readouterr()
    assert missing_src_exit == 2
    assert "missing required argument: --src" in missing_src.err

    missing_dst_exit = cli.main(["cloud", "ec2", "s3", "cp", "--src", "s3://bucket/in"])
    missing_dst = capsys.readouterr()
    assert missing_dst_exit == 2
    assert "missing required argument: --dst" in missing_dst.err


def test_cli_cloud_ec2_s3_rm_success_recursive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_ec2_s3_remove(request: api_module.Ec2S3RemoveRequest) -> api_module.CloudEc2Response:
        assert request.recursive is True
        return api_module.CloudEc2Response(
            action="cloud.ec2.s3.rm",
            message="S3 objects deleted",
            result={"deleted_count": 4},
        )

    monkeypatch.setattr(api_module, "cloud_ec2_s3_remove", fake_cloud_ec2_s3_remove)
    exit_code = cli.main(["cloud", "ec2", "s3", "rm", "--uri", "s3://bucket/run/", "--recursive"])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["result"]["deleted_count"] == 4


def test_cli_cloud_aws_image_build_push_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_aws_image_build_push(
        request: api_module.AwsImageBuildPushRequest,
    ) -> api_module.CloudAwsResponse:
        assert request.run_id == "run-1"
        assert request.repository == "numereng-training"
        assert request.runtime_profile == "lgbm-cuda"
        return api_module.CloudAwsResponse(
            action="cloud.aws.image.build-push",
            message="image pushed",
            result={"image_uri": "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:run-1"},
        )

    monkeypatch.setattr(api_module, "cloud_aws_image_build_push", fake_cloud_aws_image_build_push)

    exit_code = cli.main(
        [
            "cloud",
            "aws",
            "image",
            "build-push",
            "--run-id",
            "run-1",
            "--repository",
            "numereng-training",
            "--runtime-profile",
            "lgbm-cuda",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.aws.image.build-push"


def test_cli_cloud_aws_train_submit_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_aws_train_submit(request: api_module.AwsTrainSubmitRequest) -> api_module.CloudAwsResponse:
        assert request.run_id == "run-2"
        assert request.backend == "sagemaker"
        assert request.runtime_profile == "lgbm-cuda"
        assert request.image_uri is None
        assert request.use_spot is False
        return api_module.CloudAwsResponse(
            action="cloud.aws.train.submit",
            message="submitted",
            result={"training_job_name": "neng-sm-run-2"},
        )

    monkeypatch.setattr(api_module, "cloud_aws_train_submit", fake_cloud_aws_train_submit)

    exit_code = cli.main(
        [
            "cloud",
            "aws",
            "train",
            "submit",
            "--run-id",
            "run-2",
            "--config",
            "configs/train.json",
            "--runtime-profile",
            "lgbm-cuda",
            "--role-arn",
            "arn:aws:iam::123456789012:role/numereng-sagemaker",
            "--on-demand",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.aws.train.submit"


def test_cli_cloud_aws_train_submit_accepts_explicit_image_uri(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_aws_train_submit(request: api_module.AwsTrainSubmitRequest) -> api_module.CloudAwsResponse:
        assert request.image_uri == "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1"
        return api_module.CloudAwsResponse(
            action="cloud.aws.train.submit",
            message="submitted",
            result={"training_job_name": "neng-sm-run-explicit"},
        )

    monkeypatch.setattr(api_module, "cloud_aws_train_submit", fake_cloud_aws_train_submit)

    exit_code = cli.main(
        [
            "cloud",
            "aws",
            "train",
            "submit",
            "--run-id",
            "run-explicit",
            "--config",
            "configs/train.json",
            "--image-uri",
            "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
            "--role-arn",
            "arn:aws:iam::123456789012:role/numereng-sagemaker",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.aws.train.submit"


def test_cli_cloud_aws_train_status_pull_and_extract_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_aws_train_status(request: api_module.AwsTrainStatusRequest) -> api_module.CloudAwsResponse:
        assert request.run_id == "run-3"
        return api_module.CloudAwsResponse(
            action="cloud.aws.train.status",
            message="status",
            result={"status": "InProgress"},
        )

    def fake_cloud_aws_train_pull(request: api_module.AwsTrainPullRequest) -> api_module.CloudAwsResponse:
        assert request.run_id == "run-3"
        return api_module.CloudAwsResponse(
            action="cloud.aws.train.pull",
            message="pulled",
            result={"downloaded_count": 2},
        )

    def fake_cloud_aws_train_extract(request: api_module.AwsTrainExtractRequest) -> api_module.CloudAwsResponse:
        assert request.run_id == "run-3"
        return api_module.CloudAwsResponse(
            action="cloud.aws.train.extract",
            message="extracted",
            result={"indexed_run_ids": ["run-3"]},
        )

    monkeypatch.setattr(api_module, "cloud_aws_train_status", fake_cloud_aws_train_status)
    monkeypatch.setattr(api_module, "cloud_aws_train_pull", fake_cloud_aws_train_pull)
    monkeypatch.setattr(api_module, "cloud_aws_train_extract", fake_cloud_aws_train_extract)

    status_exit = cli.main(["cloud", "aws", "train", "status", "--run-id", "run-3"])
    status_payload = _parse_stdout_json(capsys.readouterr().out)
    assert status_exit == 0
    assert status_payload["result"]["status"] == "InProgress"

    pull_exit = cli.main(["cloud", "aws", "train", "pull", "--run-id", "run-3"])
    pull_payload = _parse_stdout_json(capsys.readouterr().out)
    assert pull_exit == 0
    assert pull_payload["result"]["downloaded_count"] == 2

    extract_exit = cli.main(["cloud", "aws", "train", "extract", "--run-id", "run-3"])
    extract_payload = _parse_stdout_json(capsys.readouterr().out)
    assert extract_exit == 0
    assert extract_payload["result"]["indexed_run_ids"] == ["run-3"]


def test_cli_cloud_aws_train_parse_and_boundary_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_backend_exit = cli.main(["cloud", "aws", "train", "status", "--backend", "wrong"])
    invalid_backend = capsys.readouterr()
    assert invalid_backend_exit == 2
    assert "invalid value for --backend" in invalid_backend.err

    submit_missing_config_exit = cli.main(
        ["cloud", "aws", "train", "submit", "--run-id", "run-9", "--image-uri", "s3://bad"]
    )
    submit_missing_config = capsys.readouterr()
    assert submit_missing_config_exit == 2
    assert "either config_path or config_s3_uri is required" in submit_missing_config.err

    submit_non_json_config_exit = cli.main(
        ["cloud", "aws", "train", "submit", "--run-id", "run-9", "--config", "configs/train.yaml"]
    )
    submit_non_json_config = capsys.readouterr()
    assert submit_non_json_config_exit == 2
    assert "config_path must reference a .json file" in submit_non_json_config.err

    submit_non_json_s3_exit = cli.main(
        [
            "cloud",
            "aws",
            "train",
            "submit",
            "--run-id",
            "run-9",
            "--config-s3-uri",
            "s3://bucket/runs/run-9/config.yaml",
        ]
    )
    submit_non_json_s3 = capsys.readouterr()
    assert submit_non_json_s3_exit == 2
    assert "config_s3_uri must reference a .json object" in submit_non_json_s3.err

    def fake_cloud_aws_train_logs(request: api_module.AwsTrainLogsRequest) -> api_module.CloudAwsResponse:
        _ = request
        raise PackageError("cloudwatch_failed")

    monkeypatch.setattr(api_module, "cloud_aws_train_logs", fake_cloud_aws_train_logs)
    logs_exit = cli.main(["cloud", "aws", "train", "logs", "--run-id", "run-4"])
    logs_err = capsys.readouterr()
    assert logs_exit == 1
    assert "cloudwatch_failed" in logs_err.err


def test_cli_cloud_modal_deploy_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_modal_deploy(request: api_module.ModalDeployRequest) -> api_module.CloudModalResponse:
        assert request.app_name == "numereng-train"
        assert request.function_name == "train_remote"
        assert request.ecr_image_uri == "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest"
        assert request.environment_name == "main"
        assert request.aws_profile == "default"
        assert request.timeout_seconds == 900
        assert request.gpu == "T4"
        assert request.cpu == 2.0
        assert request.memory_mb == 8192
        assert request.data_volume_name == "numereng-v52"
        assert request.metadata == {"owner": "daniel"}
        assert request.state_path == "tmp/modal-deploy.json"
        return api_module.CloudModalResponse(
            action="cloud.modal.deploy",
            message="deployed",
            result={"deployment_id": "ap-1", "deployed": True},
        )

    monkeypatch.setattr(api_module, "cloud_modal_deploy", fake_cloud_modal_deploy)

    exit_code = cli.main(
        [
            "cloud",
            "modal",
            "deploy",
            "--app-name",
            "numereng-train",
            "--function-name",
            "train_remote",
            "--ecr-image-uri",
            "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
            "--environment-name",
            "main",
            "--aws-profile",
            "default",
            "--timeout-seconds",
            "900",
            "--gpu",
            "T4",
            "--cpu",
            "2",
            "--memory-mb",
            "8192",
            "--data-volume-name",
            "numereng-v52",
            "--metadata",
            "owner=daniel",
            "--state-path",
            "tmp/modal-deploy.json",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.modal.deploy"
    assert payload["result"]["deployment_id"] == "ap-1"


def test_cli_cloud_modal_data_sync_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "modal-train.json"
    config_path.write_text("data:\n  data_version: v5.2\n", encoding="utf-8")

    def fake_cloud_modal_data_sync(request: api_module.ModalDataSyncRequest) -> api_module.CloudModalResponse:
        assert request.config_path == str(config_path)
        assert request.volume_name == "numereng-v52"
        assert request.create_if_missing is False
        assert request.force is True
        assert request.metadata == {"owner": "daniel"}
        assert request.state_path == "tmp/modal-data.json"
        return api_module.CloudModalResponse(
            action="cloud.modal.data.sync",
            message="synced",
            result={"volume_name": "numereng-v52", "file_count": 4},
        )

    monkeypatch.setattr(api_module, "cloud_modal_data_sync", fake_cloud_modal_data_sync)

    exit_code = cli.main(
        [
            "cloud",
            "modal",
            "data",
            "sync",
            "--config",
            str(config_path),
            "--volume-name",
            "numereng-v52",
            "--force",
            "--no-create-if-missing",
            "--metadata",
            "owner=daniel",
            "--state-path",
            "tmp/modal-data.json",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.modal.data.sync"
    assert payload["result"]["file_count"] == 4


def test_cli_cloud_modal_train_submit_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "modal-train.json"
    config_path.write_text("run_input:\n  model:\n    type: lgbm\n", encoding="utf-8")

    def fake_cloud_modal_train_submit(request: api_module.ModalTrainSubmitRequest) -> api_module.CloudModalResponse:
        assert request.config_path == str(config_path)
        assert request.profile == "full_history_refit"
        assert request.engine_mode is None
        assert request.window_size_eras is None
        assert request.embargo_eras is None
        assert request.app_name == "numereng-train"
        assert request.function_name == "train_remote"
        assert request.environment_name == "prod"
        assert request.metadata == {"team": "ml", "owner": "daniel"}
        assert request.state_path == "tmp/modal_state.json"
        return api_module.CloudModalResponse(
            action="cloud.modal.train.submit",
            message="submitted",
            result={"call_id": "fc-123"},
        )

    monkeypatch.setattr(api_module, "cloud_modal_train_submit", fake_cloud_modal_train_submit)

    exit_code = cli.main(
        [
            "cloud",
            "modal",
            "train",
            "submit",
            "--config",
            str(config_path),
            "--profile",
            "full_history_refit",
            "--app-name",
            "numereng-train",
            "--function-name",
            "train_remote",
            "--environment-name",
            "prod",
            "--metadata",
            "team=ml,owner=daniel",
            "--state-path",
            "tmp/modal_state.json",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["action"] == "cloud.modal.train.submit"
    assert payload["result"]["call_id"] == "fc-123"


def test_cli_cloud_modal_train_status_logs_cancel_pull_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_cloud_modal_train_status(request: api_module.ModalTrainStatusRequest) -> api_module.CloudModalResponse:
        assert request.call_id == "fc-7"
        assert request.timeout_seconds == 1.5
        return api_module.CloudModalResponse(
            action="cloud.modal.train.status",
            message="status",
            result={"status": "running"},
        )

    def fake_cloud_modal_train_logs(request: api_module.ModalTrainLogsRequest) -> api_module.CloudModalResponse:
        assert request.call_id == "fc-7"
        assert request.lines == 333
        return api_module.CloudModalResponse(
            action="cloud.modal.train.logs",
            message="logs",
            result={"log_source_hint": "dashboard"},
        )

    def fake_cloud_modal_train_cancel(request: api_module.ModalTrainCancelRequest) -> api_module.CloudModalResponse:
        assert request.call_id == "fc-7"
        return api_module.CloudModalResponse(
            action="cloud.modal.train.cancel",
            message="cancelled",
            result={"status": "cancelled"},
        )

    def fake_cloud_modal_train_pull(request: api_module.ModalTrainPullRequest) -> api_module.CloudModalResponse:
        assert request.call_id == "fc-7"
        assert request.output_dir == "artifacts/modal"
        assert request.timeout_seconds == 2.0
        return api_module.CloudModalResponse(
            action="cloud.modal.train.pull",
            message="pull",
            result={"run_id": "run-7"},
        )

    monkeypatch.setattr(api_module, "cloud_modal_train_status", fake_cloud_modal_train_status)
    monkeypatch.setattr(api_module, "cloud_modal_train_logs", fake_cloud_modal_train_logs)
    monkeypatch.setattr(api_module, "cloud_modal_train_cancel", fake_cloud_modal_train_cancel)
    monkeypatch.setattr(api_module, "cloud_modal_train_pull", fake_cloud_modal_train_pull)

    status_exit = cli.main(["cloud", "modal", "train", "status", "--call-id", "fc-7", "--timeout-seconds", "1.5"])
    status_payload = _parse_stdout_json(capsys.readouterr().out)
    assert status_exit == 0
    assert status_payload["result"]["status"] == "running"

    logs_exit = cli.main(["cloud", "modal", "train", "logs", "--call-id", "fc-7", "--lines", "333"])
    logs_payload = _parse_stdout_json(capsys.readouterr().out)
    assert logs_exit == 0
    assert logs_payload["action"] == "cloud.modal.train.logs"

    cancel_exit = cli.main(["cloud", "modal", "train", "cancel", "--call-id", "fc-7"])
    cancel_payload = _parse_stdout_json(capsys.readouterr().out)
    assert cancel_exit == 0
    assert cancel_payload["result"]["status"] == "cancelled"

    pull_exit = cli.main(
        [
            "cloud",
            "modal",
            "train",
            "pull",
            "--call-id",
            "fc-7",
            "--output-dir",
            "artifacts/modal",
            "--timeout-seconds",
            "2.0",
        ]
    )
    pull_payload = _parse_stdout_json(capsys.readouterr().out)
    assert pull_exit == 0
    assert pull_payload["result"]["run_id"] == "run-7"


def test_cli_cloud_modal_train_parse_and_boundary_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_config_exit = cli.main(["cloud", "modal", "train", "submit"])
    missing_config = capsys.readouterr()
    assert missing_config_exit == 2
    assert "missing required argument: --config" in missing_config.err

    invalid_timeout_exit = cli.main(
        ["cloud", "modal", "train", "status", "--call-id", "fc-1", "--timeout-seconds", "bad"]
    )
    invalid_timeout = capsys.readouterr()
    assert invalid_timeout_exit == 2
    assert "invalid number for --timeout-seconds: bad" in invalid_timeout.err

    invalid_metadata_exit = cli.main(
        [
            "cloud",
            "modal",
            "train",
            "submit",
            "--config",
            "configs/train.json",
            "--metadata",
            "missing_equals",
        ]
    )
    invalid_metadata = capsys.readouterr()
    assert invalid_metadata_exit == 2
    assert "invalid value for --metadata: missing_equals (expected key=value)" in invalid_metadata.err

    non_json_config_exit = cli.main(["cloud", "modal", "train", "submit", "--config", "configs/train.yaml"])
    non_json_config = capsys.readouterr()
    assert non_json_config_exit == 2
    assert "config_path must reference a .json file" in non_json_config.err

    def fake_cloud_modal_train_logs(request: api_module.ModalTrainLogsRequest) -> api_module.CloudModalResponse:
        _ = request
        raise PackageError("modal_logs_failed")

    monkeypatch.setattr(api_module, "cloud_modal_train_logs", fake_cloud_modal_train_logs)
    logs_exit = cli.main(["cloud", "modal", "train", "logs", "--call-id", "fc-1"])
    logs_err = capsys.readouterr()
    assert logs_exit == 1
    assert "modal_logs_failed" in logs_err.err


def test_cli_cloud_modal_deploy_parse_and_boundary_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_ecr_exit = cli.main(["cloud", "modal", "deploy"])
    missing_ecr = capsys.readouterr()
    assert missing_ecr_exit == 2
    assert "missing required argument: --ecr-image-uri" in missing_ecr.err

    invalid_cpu_exit = cli.main(
        [
            "cloud",
            "modal",
            "deploy",
            "--ecr-image-uri",
            "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
            "--cpu",
            "bad",
        ]
    )
    invalid_cpu = capsys.readouterr()
    assert invalid_cpu_exit == 2
    assert "invalid number for --cpu: bad" in invalid_cpu.err

    def fake_cloud_modal_deploy(request: api_module.ModalDeployRequest) -> api_module.CloudModalResponse:
        _ = request
        raise PackageError("modal_deploy_failed")

    monkeypatch.setattr(api_module, "cloud_modal_deploy", fake_cloud_modal_deploy)
    deploy_exit = cli.main(
        [
            "cloud",
            "modal",
            "deploy",
            "--ecr-image-uri",
            "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
        ]
    )
    deploy_err = capsys.readouterr()
    assert deploy_exit == 1
    assert "modal_deploy_failed" in deploy_err.err


def test_cli_cloud_modal_data_sync_parse_and_boundary_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_config_exit = cli.main(["cloud", "modal", "data", "sync", "--volume-name", "numereng-v52"])
    missing_config = capsys.readouterr()
    assert missing_config_exit == 2
    assert "missing required argument: --config" in missing_config.err

    missing_volume_exit = cli.main(["cloud", "modal", "data", "sync", "--config", "train.json"])
    missing_volume = capsys.readouterr()
    assert missing_volume_exit == 2
    assert "missing required argument: --volume-name" in missing_volume.err

    invalid_metadata_exit = cli.main(
        [
            "cloud",
            "modal",
            "data",
            "sync",
            "--config",
            "train.json",
            "--volume-name",
            "numereng-v52",
            "--metadata",
            "missing_equals",
        ]
    )
    invalid_metadata = capsys.readouterr()
    assert invalid_metadata_exit == 2
    assert "invalid value for --metadata: missing_equals (expected key=value)" in invalid_metadata.err

    non_json_config_exit = cli.main(
        [
            "cloud",
            "modal",
            "data",
            "sync",
            "--config",
            "train.yaml",
            "--volume-name",
            "numereng-v52",
        ]
    )
    non_json_config = capsys.readouterr()
    assert non_json_config_exit == 2
    assert "config_path must reference a .json file" in non_json_config.err

    def fake_cloud_modal_data_sync(request: api_module.ModalDataSyncRequest) -> api_module.CloudModalResponse:
        _ = request
        raise PackageError("modal_data_sync_failed")

    monkeypatch.setattr(api_module, "cloud_modal_data_sync", fake_cloud_modal_data_sync)
    sync_exit = cli.main(
        [
            "cloud",
            "modal",
            "data",
            "sync",
            "--config",
            "train.json",
            "--volume-name",
            "numereng-v52",
        ]
    )
    sync_err = capsys.readouterr()
    assert sync_exit == 1
    assert "modal_data_sync_failed" in sync_err.err


def test_cli_cloud_ec2_unknown_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["cloud", "unknown"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown arguments: cloud unknown" in captured.err

    ec2_exit = cli.main(["cloud", "ec2", "unknown"])
    ec2_captured = capsys.readouterr()
    assert ec2_exit == 2
    assert "unknown arguments: cloud ec2 unknown" in ec2_captured.err

    aws_exit = cli.main(["cloud", "aws", "unknown"])
    aws_captured = capsys.readouterr()
    assert aws_exit == 2
    assert "unknown arguments: cloud aws unknown" in aws_captured.err

    modal_exit = cli.main(["cloud", "modal", "unknown"])
    modal_captured = capsys.readouterr()
    assert modal_exit == 2
    assert "unknown arguments: cloud modal unknown" in modal_captured.err

    modal_train_exit = cli.main(["cloud", "modal", "train", "unknown"])
    modal_train_captured = capsys.readouterr()
    assert modal_train_exit == 2
    assert "unknown arguments: cloud modal train unknown" in modal_train_captured.err

    modal_data_exit = cli.main(["cloud", "modal", "data", "unknown"])
    modal_data_captured = capsys.readouterr()
    assert modal_data_exit == 2
    assert "unknown arguments: cloud modal data unknown" in modal_data_captured.err


def _hpo_response(*, study_id: str, experiment_id: str | None = "exp-1") -> api_module.HpoStudyResponse:
    return api_module.HpoStudyResponse(
        study_id=study_id,
        experiment_id=experiment_id,
        study_name=study_id,
        status="completed",
        best_trial_number=1,
        best_value=0.12,
        best_run_id="run-1",
        spec=api_module.HpoStudySpecResponse(
            study_id=study_id,
            study_name=study_id,
            config_path="configs/base.json",
            experiment_id=experiment_id,
            objective=api_module.HpoObjectiveRequest(
                metric="post_fold_champion_objective",
                direction="maximize",
                neutralization=api_module.HpoNeutralizationRequest(),
            ),
            search_space={
                "model.params.learning_rate": api_module.HpoSearchSpaceSpecRequest(
                    type="float",
                    low=0.001,
                    high=0.1,
                    log=True,
                )
            },
            sampler=api_module.HpoSamplerRequest(kind="tpe", seed=1337),
            stopping=api_module.HpoStoppingRequest(max_trials=2),
        ),
        attempted_trials=2,
        completed_trials=2,
        failed_trials=0,
        stop_reason="max_trials_reached",
        storage_path=f".numereng/hpo/{study_id}",
        error_message=None,
        created_at="2026-02-22T00:00:00+00:00",
        updated_at="2026-02-22T00:00:00+00:00",
    )


def test_cli_hpo_create_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_hpo_create(request: api_module.HpoStudyCreateRequest) -> api_module.HpoStudyResponse:
        assert request.study_id == "study-a"
        assert request.study_name == "study-a"
        assert request.stopping.max_trials == 2
        assert request.search_space["model.params.learning_rate"].high == 0.1
        return _hpo_response(study_id="study-a")

    monkeypatch.setattr(api_module, "hpo_create", fake_hpo_create)

    exit_code = cli.main(
        [
            "hpo",
            "create",
            "--study-id",
            "study-a",
            "--study-name",
            "study-a",
            "--config",
            "configs/base.json",
            "--experiment-id",
            "exp-1",
            "--n-trials",
            "2",
            "--search-space",
            '{"model.params.learning_rate":{"type":"float","low":0.001,"high":0.1,"log":true}}',
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["study_id"] == "study-a"
    assert payload["best_run_id"] == "run-1"


def test_cli_hpo_create_from_study_config_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    study_config_path = tmp_path / "study.json"
    study_config_path.write_text(
        json.dumps(
            {
                "study_id": "study-from-file",
                "study_name": "study-from-file",
                "config_path": "configs/base.json",
                "experiment_id": "exp-file",
                "objective": {
                    "metric": "bmc_last_200_eras.mean",
                    "direction": "maximize",
                    "neutralization": {
                        "enabled": True,
                        "neutralizer_path": "neutralizers.parquet",
                        "proportion": 0.25,
                        "mode": "global",
                        "neutralizer_cols": ["feature_x"],
                        "rank_output": False,
                    },
                },
                "search_space": {
                    "model.params.learning_rate": {
                        "type": "float",
                        "low": 0.001,
                        "high": 0.1,
                    }
                },
                "sampler": {
                    "kind": "tpe",
                    "seed": 7,
                    "n_startup_trials": 12,
                    "multivariate": True,
                    "group": False,
                },
                "stopping": {
                    "max_trials": 3,
                    "max_completed_trials": 2,
                    "timeout_seconds": 600,
                    "plateau": {
                        "enabled": True,
                        "min_completed_trials": 3,
                        "patience_completed_trials": 2,
                        "min_improvement_abs": 0.0001,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_hpo_create(request: api_module.HpoStudyCreateRequest) -> api_module.HpoStudyResponse:
        assert request.study_id == "study-from-file"
        assert request.config_path == "configs/base.json"
        assert request.experiment_id == "exp-file"
        assert request.stopping.max_trials == 3
        assert request.stopping.max_completed_trials == 2
        assert request.objective.neutralization.enabled is True
        assert request.objective.neutralization.neutralizer_path == "neutralizers.parquet"
        assert request.objective.neutralization.proportion == 0.25
        assert request.objective.neutralization.mode == "global"
        assert request.objective.neutralization.neutralizer_cols == ["feature_x"]
        assert request.objective.neutralization.rank_output is False
        return _hpo_response(study_id="study-from-file", experiment_id="exp-file")

    monkeypatch.setattr(api_module, "hpo_create", fake_hpo_create)
    exit_code = cli.main(["hpo", "create", "--study-config", str(study_config_path)])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["study_id"] == "study-from-file"


def test_cli_hpo_create_from_study_config_allows_flag_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    study_config_path = tmp_path / "study.json"
    study_config_path.write_text(
        json.dumps(
            {
                "study_id": "study-from-file",
                "study_name": "study-from-file",
                "config_path": "configs/base.json",
                "objective": {
                    "metric": "post_fold_champion_objective",
                    "direction": "maximize",
                    "neutralization": {"enabled": False},
                },
                "search_space": {
                    "model.params.learning_rate": {
                        "type": "float",
                        "low": 0.001,
                        "high": 0.1,
                    }
                },
                "sampler": {
                    "kind": "tpe",
                    "seed": 7,
                    "n_startup_trials": 10,
                    "multivariate": True,
                    "group": False,
                },
                "stopping": {
                    "max_trials": 3,
                    "max_completed_trials": None,
                    "timeout_seconds": None,
                    "plateau": {
                        "enabled": False,
                        "min_completed_trials": 15,
                        "patience_completed_trials": 10,
                        "min_improvement_abs": 0.00025,
                    },  # noqa: E501
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_hpo_create(request: api_module.HpoStudyCreateRequest) -> api_module.HpoStudyResponse:
        assert request.stopping.max_trials == 9
        assert request.objective.neutralization.enabled is True
        assert request.objective.neutralization.neutralizer_path == "neutralizers.parquet"
        return _hpo_response(study_id="study-from-file")

    monkeypatch.setattr(api_module, "hpo_create", fake_hpo_create)
    exit_code = cli.main(
        [
            "hpo",
            "create",
            "--study-config",
            str(study_config_path),
            "--n-trials",
            "9",
            "--neutralize",
            "--neutralizer-path",
            "neutralizers.parquet",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["study_id"] == "study-from-file"


def test_cli_hpo_create_inline_random_sampler_omits_tpe_only_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_hpo_create(request: api_module.HpoStudyCreateRequest) -> api_module.HpoStudyResponse:
        assert request.sampler.kind == "random"
        assert request.sampler.model_fields_set == {"kind", "seed"}
        response = _hpo_response(study_id="study-random")
        response.spec.sampler = api_module.HpoSamplerRequest(kind="random", seed=9)
        return response

    monkeypatch.setattr(api_module, "hpo_create", fake_hpo_create)

    exit_code = cli.main(
        [
            "hpo",
            "create",
            "--study-id",
            "study-random",
            "--study-name",
            "study-random",
            "--config",
            "configs/base.json",
            "--n-trials",
            "2",
            "--sampler",
            "random",
            "--seed",
            "9",
            "--search-space",
            '{"model.params.learning_rate":{"type":"float","low":0.001,"high":0.1}}',
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["study_id"] == "study-random"
    assert payload["spec"]["sampler"] == {
        "kind": "random",
        "seed": 9,
    }


def test_cli_ensemble_build_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_ensemble_build(request: api_module.EnsembleBuildRequest) -> api_module.EnsembleResponse:
        assert request.run_ids == ["run-a", "run-b"]
        assert request.include_heavy_artifacts is False
        assert request.selection_note is None
        assert request.regime_buckets == 4
        return api_module.EnsembleResponse(
            ensemble_id="ens-1",
            experiment_id=request.experiment_id,
            name="ens-1",
            method="rank_avg",
            target=request.target,
            metric=request.metric,
            status="completed",
            components=[
                api_module.EnsembleComponentResponse(run_id="run-a", weight=0.6, rank=0),
                api_module.EnsembleComponentResponse(run_id="run-b", weight=0.4, rank=1),
            ],
            metrics=[api_module.EnsembleMetricResponse(name="corr_mean", value=0.11)],
            artifacts_path="experiments/exp-1/ensembles/ens-1",
            config={},
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "ensemble_build", fake_ensemble_build)

    exit_code = cli.main(
        [
            "ensemble",
            "build",
            "--run-ids",
            "run-a,run-b",
            "--experiment-id",
            "exp-1",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ensemble_id"] == "ens-1"
    assert payload["status"] == "completed"


def test_cli_ensemble_build_with_artifact_flags_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_ensemble_build(request: api_module.EnsembleBuildRequest) -> api_module.EnsembleResponse:
        assert request.run_ids == ["run-a", "run-b"]
        assert request.include_heavy_artifacts is True
        assert request.selection_note == "diversity-priority"
        assert request.regime_buckets == 6
        return api_module.EnsembleResponse(
            ensemble_id="ens-2",
            experiment_id=request.experiment_id,
            name="ens-2",
            method="rank_avg",
            target=request.target,
            metric=request.metric,
            status="completed",
            components=[
                api_module.EnsembleComponentResponse(run_id="run-a", weight=0.5, rank=0),
                api_module.EnsembleComponentResponse(run_id="run-b", weight=0.5, rank=1),
            ],
            metrics=[api_module.EnsembleMetricResponse(name="corr_mean", value=0.1)],
            artifacts_path="experiments/exp-1/ensembles/ens-2",
            config={},
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "ensemble_build", fake_ensemble_build)

    exit_code = cli.main(
        [
            "ensemble",
            "build",
            "--run-ids",
            "run-a,run-b",
            "--experiment-id",
            "exp-1",
            "--include-heavy-artifacts",
            "--selection-note",
            "diversity-priority",
            "--regime-buckets",
            "6",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ensemble_id"] == "ens-2"


def test_cli_hpo_create_rejects_invalid_direction(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "hpo",
            "create",
            "--study-id",
            "study-a",
            "--study-name",
            "study-a",
            "--config",
            "configs/base.json",
            "--search-space",
            '{"model.params.learning_rate":{"type":"float","low":0.001,"high":0.1}}',
            "--direction",
            "bad",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid value for --direction: expected maximize|minimize" in captured.err


def test_cli_hpo_create_rejects_invalid_search_space_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "hpo",
            "create",
            "--study-id",
            "study-a",
            "--study-name",
            "study-a",
            "--config",
            "configs/base.json",
            "--search-space",
            "{bad",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid JSON for --search-space" in captured.err


def test_cli_hpo_list_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_hpo_list(request: api_module.HpoStudyListRequest) -> api_module.HpoStudyListResponse:
        _ = request
        return api_module.HpoStudyListResponse(studies=[_hpo_response(study_id="study-1")])

    monkeypatch.setattr(api_module, "hpo_list", fake_hpo_list)

    exit_code = cli.main(["hpo", "list", "--format", "json"])
    payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["studies"][0]["study_id"] == "study-1"


def test_cli_hpo_details_table_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_hpo_get(request: api_module.HpoStudyGetRequest) -> api_module.HpoStudyResponse:
        _ = request
        return _hpo_response(study_id="study-1")

    monkeypatch.setattr(api_module, "hpo_get", fake_hpo_get)

    exit_code = cli.main(["hpo", "details", "--study-id", "study-1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "study_id: study-1" in captured.out
    assert "completed_trials: 2" in captured.out
    assert "best_run_id: run-1" in captured.out


def test_cli_hpo_trials_translates_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_hpo_trials(request: api_module.HpoStudyTrialsRequest) -> api_module.HpoStudyTrialsResponse:
        _ = request
        raise PackageError("hpo_trials_failed")

    monkeypatch.setattr(api_module, "hpo_trials", fake_hpo_trials)

    exit_code = cli.main(["hpo", "trials", "--study-id", "study-1"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "hpo_trials_failed" in captured.err


def test_cli_ensemble_build_rejects_invalid_method(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "ensemble",
            "build",
            "--run-ids",
            "run-a,run-b",
            "--method",
            "forward",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid value for --method: expected rank_avg" in captured.err


def test_cli_ensemble_build_rejects_invalid_weights(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "ensemble",
            "build",
            "--run-ids",
            "run-a,run-b",
            "--weights",
            "1.0,abc",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid value for --weights: expected comma-separated floats" in captured.err


def test_cli_ensemble_build_rejects_invalid_regime_buckets(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "ensemble",
            "build",
            "--run-ids",
            "run-a,run-b",
            "--regime-buckets",
            "abc",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid integer for --regime-buckets: abc" in captured.err


def test_cli_ensemble_list_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_ensemble_list(request: api_module.EnsembleListRequest) -> api_module.EnsembleListResponse:
        _ = request
        return api_module.EnsembleListResponse(
            ensembles=[
                api_module.EnsembleResponse(
                    ensemble_id="ens-1",
                    experiment_id="exp-1",
                    name="ens-1",
                    method="rank_avg",
                    target="target_ender_20",
                    metric="corr_sharpe",
                    status="completed",
                    components=[api_module.EnsembleComponentResponse(run_id="run-a", weight=1.0, rank=0)],
                    metrics=[api_module.EnsembleMetricResponse(name="corr_mean", value=0.12)],
                    artifacts_path="experiments/exp-1/ensembles/ens-1",
                    config={},
                    created_at="2026-02-22T00:00:00+00:00",
                    updated_at="2026-02-22T00:00:00+00:00",
                )
            ]
        )

    monkeypatch.setattr(api_module, "ensemble_list", fake_ensemble_list)

    exit_code = cli.main(["ensemble", "list", "--format", "json"])
    payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ensembles"][0]["ensemble_id"] == "ens-1"


def test_cli_ensemble_details_translates_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_ensemble_get(request: api_module.EnsembleGetRequest) -> api_module.EnsembleResponse:
        _ = request
        raise PackageError("ensemble_not_found:ens-1")

    monkeypatch.setattr(api_module, "ensemble_get", fake_ensemble_get)

    exit_code = cli.main(["ensemble", "details", "--ensemble-id", "ens-1"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ensemble_not_found:ens-1" in captured.err


def test_cli_run_submit_with_neutralization_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_submit_predictions(request: api_module.SubmissionRequest) -> api_module.SubmissionResponse:
        assert request.neutralize is True
        assert request.neutralizer_path == "neutralizers.parquet"
        assert request.neutralization_proportion == 0.75
        assert request.neutralization_mode == "global"
        assert request.neutralizer_cols == ["feature_a", "feature_b"]
        assert request.neutralization_rank_output is False
        return api_module.SubmissionResponse(
            submission_id="submission-3",
            model_name=request.model_name,
            model_id="model-3",
            predictions_path="/tmp/preds.csv",
            run_id=request.run_id,
        )

    monkeypatch.setattr(api_module, "submit_predictions", fake_submit_predictions)

    exit_code = cli.main(
        [
            "run",
            "submit",
            "--run-id",
            "run-1",
            "--model-name",
            "main",
            "--neutralize",
            "--neutralizer-path",
            "neutralizers.parquet",
            "--neutralization-proportion",
            "0.75",
            "--neutralization-mode",
            "global",
            "--neutralizer-cols",
            "feature_a,feature_b",
            "--no-neutralization-rank",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["submission_id"] == "submission-3"


def test_cli_hpo_create_with_neutralization_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_hpo_create(request: api_module.HpoStudyCreateRequest) -> api_module.HpoStudyResponse:
        neutralization = request.objective.neutralization
        assert neutralization.enabled is True
        assert neutralization.neutralizer_path == "neutralizers.parquet"
        assert neutralization.proportion == 0.5
        assert neutralization.mode == "era"
        assert neutralization.neutralizer_cols == ["feature_x"]
        assert neutralization.rank_output is False
        return _hpo_response(study_id="study-neut-1")

    monkeypatch.setattr(api_module, "hpo_create", fake_hpo_create)
    exit_code = cli.main(
        [
            "hpo",
            "create",
            "--study-id",
            "study-a",
            "--study-name",
            "study-a",
            "--config",
            "configs/base.json",
            "--search-space",
            '{"model.params.learning_rate":{"type":"float","low":0.001,"high":0.1}}',
            "--neutralize",
            "--neutralizer-path",
            "neutralizers.parquet",
            "--neutralizer-cols",
            "feature_x",
            "--no-neutralization-rank",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["study_id"] == "study-neut-1"


def test_cli_ensemble_build_with_neutralization_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_ensemble_build(request: api_module.EnsembleBuildRequest) -> api_module.EnsembleResponse:
        assert request.neutralize_members is True
        assert request.neutralize_final is True
        assert request.neutralizer_path == "neutralizers.parquet"
        assert request.neutralization_proportion == 0.25
        assert request.neutralization_mode == "global"
        assert request.neutralizer_cols == ["feature_1", "feature_2"]
        assert request.neutralization_rank_output is False
        return api_module.EnsembleResponse(
            ensemble_id="ens-neut-1",
            experiment_id=request.experiment_id,
            name="ens-neut-1",
            method="rank_avg",
            target=request.target,
            metric=request.metric,
            status="completed",
            components=[
                api_module.EnsembleComponentResponse(run_id="run-a", weight=0.5, rank=0),
                api_module.EnsembleComponentResponse(run_id="run-b", weight=0.5, rank=1),
            ],
            metrics=[api_module.EnsembleMetricResponse(name="corr_mean", value=0.2)],
            artifacts_path=".numereng/ensembles/ens-neut-1",
            config={},
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "ensemble_build", fake_ensemble_build)
    exit_code = cli.main(
        [
            "ensemble",
            "build",
            "--run-ids",
            "run-a,run-b",
            "--neutralize-members",
            "--neutralize-final",
            "--neutralizer-path",
            "neutralizers.parquet",
            "--neutralization-proportion",
            "0.25",
            "--neutralization-mode",
            "global",
            "--neutralizer-cols",
            "feature_1,feature_2",
            "--no-neutralization-rank",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ensemble_id"] == "ens-neut-1"


def test_cli_neutralize_apply_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_neutralize_apply(request: api_module.NeutralizeRequest) -> api_module.NeutralizeResponse:
        assert request.run_id == "run-1"
        assert request.neutralizer_path == "neutralizers.parquet"
        return api_module.NeutralizeResponse(
            source_path=".numereng/runs/run-1/artifacts/predictions/predictions.parquet",
            output_path=".numereng/runs/run-1/artifacts/predictions/predictions.neutralized.parquet",
            run_id="run-1",
            neutralizer_path="neutralizers.parquet",
            neutralizer_cols=["feature_1", "feature_2"],
            neutralization_proportion=0.5,
            neutralization_mode="era",
            neutralization_rank_output=True,
            source_rows=100,
            neutralizer_rows=100,
            matched_rows=100,
        )

    monkeypatch.setattr(api_module, "neutralize_apply", fake_neutralize_apply)
    exit_code = cli.main(
        [
            "neutralize",
            "apply",
            "--run-id",
            "run-1",
            "--neutralizer-path",
            "neutralizers.parquet",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["run_id"] == "run-1"


def test_cli_neutralize_apply_requires_source(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "neutralize",
            "apply",
            "--neutralizer-path",
            "neutralizers.parquet",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "exactly one of --run-id or --predictions is required" in captured.err


def test_cli_research_commands_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        api_module,
        "research_init",
        lambda request: api_module.ResearchInitResponse(
            root_experiment_id=request.experiment_id,
            program_id=request.program_id or "numerai-experiment-loop",
            program_title="Numerai Experiment Loop",
            status="initialized",
            active_experiment_id=request.experiment_id,
            active_path_id="p00",
            improvement_threshold=0.0002,
            current_phase=None,
            agentic_research_dir="/tmp/agentic_research",
            program_path="/tmp/agentic_research/program.json",
            lineage_path="/tmp/agentic_research/lineage.json",
            session_program_path="/tmp/agentic_research/session_program.md",
        ),
    )
    monkeypatch.setattr(
        api_module,
        "research_program_list",
        lambda request: api_module.ResearchProgramListResponse(
            programs=[
                api_module.ResearchProgramCatalogEntryResponse(
                    program_id="numerai-experiment-loop",
                    title="Numerai Experiment Loop",
                    description="Builtin numerai loop.",
                    source="builtin",
                    planner_contract="config_mutation",
                    phase_aware=False,
                    source_path="/tmp/numerai-experiment-loop.md",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        api_module,
        "research_program_show",
        lambda request: api_module.ResearchProgramShowResponse(
            program_id=request.program_id,
            title="Numerai Experiment Loop",
            description="Builtin numerai loop.",
            source="builtin",
            planner_contract="config_mutation",
            scoring_stage="post_training_full",
            metric_policy=api_module.ResearchProgramMetricPolicyResponse(
                primary="bmc_last_200_eras.mean",
                tie_break="bmc.mean",
                sanity_checks=["corr.mean"],
            ),
            round_policy=api_module.ResearchProgramRoundPolicyResponse(
                plateau_non_improving_rounds=2,
                require_scale_confirmation=True,
                scale_confirmation_rounds=1,
            ),
            improvement_threshold_default=0.0002,
            config_policy=api_module.ResearchProgramConfigPolicyResponse(
                allowed_paths=["model.params.*"],
                min_candidate_configs=None,
                max_candidate_configs=1,
                min_changes=1,
                max_changes=2,
            ),
            phases=[],
            source_path="/tmp/numerai-experiment-loop.md",
            raw_markdown=(
                "---\nid: numerai-experiment-loop\n---\nContext:\n$CONTEXT_JSON\n\n$VALIDATION_FEEDBACK_BLOCK\n"
            ),
        ),
    )
    monkeypatch.setattr(
        api_module,
        "research_status",
        lambda request: api_module.ResearchStatusResponse(
            root_experiment_id=request.experiment_id,
            program_id="numerai-experiment-loop",
            program_title="Numerai Experiment Loop",
            status="running",
            active_experiment_id=request.experiment_id,
            active_path_id="p00",
            next_round_number=2,
            total_rounds_completed=1,
            total_paths_created=1,
            improvement_threshold=0.0002,
            last_checkpoint="round_completed",
            stop_reason=None,
            best_overall=api_module.ResearchBestRunResponse(run_id="run-4"),
            current_round=None,
            current_phase=None,
            program_path="/tmp/agentic_research/program.json",
            lineage_path="/tmp/agentic_research/lineage.json",
            session_program_path="/tmp/agentic_research/session_program.md",
        ),
    )
    monkeypatch.setattr(
        api_module,
        "research_run",
        lambda request: api_module.ResearchRunResponse(
            root_experiment_id=request.experiment_id,
            program_id="numerai-experiment-loop",
            program_title="Numerai Experiment Loop",
            status="stopped",
            active_experiment_id=request.experiment_id,
            active_path_id="p00",
            next_round_number=3,
            total_rounds_completed=2,
            total_paths_created=1,
            last_checkpoint="stopped",
            stop_reason="max_rounds_reached",
            current_phase=None,
            interrupted=False,
        ),
    )

    exit_code = cli.main(
        [
            "research",
            "init",
            "--experiment-id",
            "2026-02-22_test-exp",
            "--program",
            "numerai-experiment-loop",
        ]
    )
    assert exit_code == 0
    assert _parse_stdout_json(capsys.readouterr().out)["active_path_id"] == "p00"

    exit_code = cli.main(["research", "program", "list"])
    assert exit_code == 0
    assert "numerai-experiment-loop" in capsys.readouterr().out

    exit_code = cli.main(["research", "program", "show", "--program", "numerai-experiment-loop"])
    assert exit_code == 0
    assert "program_id: numerai-experiment-loop" in capsys.readouterr().out

    exit_code = cli.main(["research", "status", "--experiment-id", "2026-02-22_test-exp"])
    assert exit_code == 0
    assert "status: running" in capsys.readouterr().out

    exit_code = cli.main(["research", "run", "--experiment-id", "2026-02-22_test-exp", "--max-rounds", "1"])
    assert exit_code == 0
    assert _parse_stdout_json(capsys.readouterr().out)["stop_reason"] == "max_rounds_reached"

    exit_code = cli.main(
        [
            "research",
            "init",
            "--experiment-id",
            "2026-02-22_test-exp",
            "--program",
            "numerai-experiment-loop",
        ]
    )
    assert exit_code == 0
    assert _parse_stdout_json(capsys.readouterr().out)["program_id"] == "numerai-experiment-loop"


def test_cli_baseline_build_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_baseline_build(request: api_module.BaselineBuildRequest) -> api_module.BaselineBuildResponse:
        assert request.run_ids == ["run20a", "run20b", "run60a"]
        assert request.name == "medium_ender20_ender60_6run_blend"
        assert request.default_target == "target_ender_20"
        assert request.promote_active is True
        return api_module.BaselineBuildResponse(
            name=request.name,
            baseline_dir="/tmp/.numereng/datasets/baselines/medium_ender20_ender60_6run_blend",
            predictions_path="/tmp/.numereng/datasets/baselines/medium_ender20_ender60_6run_blend/pred_medium_ender20_ender60_6run_blend.parquet",
            metadata_path="/tmp/.numereng/datasets/baselines/medium_ender20_ender60_6run_blend/baseline.json",
            available_targets=["target_ender_20", "target_ender_60"],
            default_target=request.default_target,
            source_run_ids=request.run_ids,
            source_experiment_id="2026-03-15_medium-deep-lgbm-ender20-ender60-3seed",
            active_predictions_path="/tmp/.numereng/datasets/baselines/active_benchmark/predictions.parquet",
            active_metadata_path="/tmp/.numereng/datasets/baselines/active_benchmark/benchmark.json",
            created_at="2026-03-27T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "baseline_build", fake_baseline_build)

    exit_code = cli.main(
        [
            "baseline",
            "build",
            "--run-ids",
            "run20a,run20b,run60a",
            "--name",
            "medium_ender20_ender60_6run_blend",
            "--promote-active",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["name"] == "medium_ender20_ender60_6run_blend"
    assert payload["default_target"] == "target_ender_20"
