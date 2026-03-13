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

    exit_code = cli.main(
        ["run", "submit", "--run-id", "run-1", "--model-name", "main", "--tournament", "signals"]
    )
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


def test_cli_run_train_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "train"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --config" in captured.err


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
        return api_module.ScoreRunResponse(
            run_id="run-123",
            predictions_path="/tmp/preds.parquet",
            results_path="/tmp/results.json",
            metrics_path="/tmp/metrics.json",
            score_provenance_path="/tmp/score_provenance.json",
            effective_scoring_backend="materialized",
        )

    monkeypatch.setattr(api_module, "score_run", fake_score_run)

    exit_code = cli.main(["run", "score", "--run-id", "run-123"])
    captured = capsys.readouterr()
    payload = _parse_stdout_json(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"
    assert payload["metrics_path"] == "/tmp/metrics.json"


def test_cli_run_score_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "score"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing required argument: --run-id" in captured.err


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
            manifest_path="/tmp/.numereng/experiments/2026-02-22_test-exp/experiment.json",
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
                    manifest_path="/tmp/.numereng/experiments/2026-02-22_test-exp/experiment.json",
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
            manifest_path="/tmp/.numereng/experiments/2026-02-22_test-exp/experiment.json",
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
            manifest_path="/tmp/.numereng/experiments/_archive/2026-02-22_test-exp/experiment.json",
            archived=True,
        )

    def fake_experiment_unarchive(request: api_module.ExperimentArchiveRequest) -> api_module.ExperimentArchiveResponse:
        assert request.experiment_id == "2026-02-22_test-exp"
        return api_module.ExperimentArchiveResponse(
            experiment_id=request.experiment_id,
            status="active",
            manifest_path="/tmp/.numereng/experiments/2026-02-22_test-exp/experiment.json",
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
            output_path="/tmp/.numereng/experiments/2026-02-22_test-exp/EXPERIMENT.pack.md",
            experiment_path="/tmp/.numereng/experiments/2026-02-22_test-exp",
            source_markdown_path="/tmp/.numereng/experiments/2026-02-22_test-exp/EXPERIMENT.md",
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
            "--image-uri",
            "123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:v1",
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
        assert (
            request.ecr_image_uri
            == "699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest"
        )
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
            "699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
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

    status_exit = cli.main(
        ["cloud", "modal", "train", "status", "--call-id", "fc-7", "--timeout-seconds", "1.5"]
    )
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
            "699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
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
            "699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest",
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


def test_cli_hpo_create_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_hpo_create(request: api_module.HpoStudyCreateRequest) -> api_module.HpoStudyResponse:
        assert request.study_name == "study-a"
        assert request.n_trials == 2
        return api_module.HpoStudyResponse(
            study_id="study-a-1",
            experiment_id=request.experiment_id,
            study_name=request.study_name,
            status="completed",
            metric=request.metric,
            direction=request.direction,
            n_trials=request.n_trials,
            sampler=request.sampler,
            seed=request.seed,
            best_trial_number=1,
            best_value=0.12,
            best_run_id="run-1",
            config={},
            storage_path=".numereng/experiments/exp-1/hpo/study-a-1",
            error_message=None,
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "hpo_create", fake_hpo_create)

    exit_code = cli.main(
        [
            "hpo",
            "create",
            "--study-name",
            "study-a",
            "--config",
            "configs/base.json",
            "--experiment-id",
            "exp-1",
            "--n-trials",
            "2",
        ]
    )
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["study_id"] == "study-a-1"
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
                "study_name": "study-from-file",
                "config_path": "configs/base.json",
                "experiment_id": "exp-file",
                "metric": "bmc_last_200_eras.mean",
                "direction": "maximize",
                "n_trials": 3,
                "sampler": "tpe",
                "seed": 7,
                "search_space": {
                    "model.params.learning_rate": {
                        "type": "float",
                        "low": 0.001,
                        "high": 0.1,
                    }
                },
                "neutralization": {
                    "enabled": True,
                    "neutralizer_path": "neutralizers.parquet",
                    "proportion": 0.25,
                    "mode": "global",
                    "neutralizer_cols": ["feature_x"],
                    "rank_output": False,
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_hpo_create(request: api_module.HpoStudyCreateRequest) -> api_module.HpoStudyResponse:
        assert request.study_name == "study-from-file"
        assert request.config_path == "configs/base.json"
        assert request.experiment_id == "exp-file"
        assert request.n_trials == 3
        assert request.neutralize is True
        assert request.neutralizer_path == "neutralizers.parquet"
        assert request.neutralization_proportion == 0.25
        assert request.neutralization_mode == "global"
        assert request.neutralizer_cols == ["feature_x"]
        assert request.neutralization_rank_output is False
        return api_module.HpoStudyResponse(
            study_id="study-file-1",
            experiment_id=request.experiment_id,
            study_name=request.study_name,
            status="completed",
            metric=request.metric,
            direction=request.direction,
            n_trials=request.n_trials,
            sampler=request.sampler,
            seed=request.seed,
            best_trial_number=0,
            best_value=0.1,
            best_run_id="run-1",
            config={},
            storage_path=".numereng/hpo/study-file-1",
            error_message=None,
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "hpo_create", fake_hpo_create)
    exit_code = cli.main(["hpo", "create", "--study-config", str(study_config_path)])
    payload = _parse_stdout_json(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["study_id"] == "study-file-1"


def test_cli_hpo_create_from_study_config_allows_flag_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    study_config_path = tmp_path / "study.json"
    study_config_path.write_text(
        json.dumps(
            {
                "study_name": "study-from-file",
                "config_path": "configs/base.json",
                "n_trials": 3,
                "neutralization": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    def fake_hpo_create(request: api_module.HpoStudyCreateRequest) -> api_module.HpoStudyResponse:
        assert request.study_name == "study-from-file"
        assert request.n_trials == 9
        assert request.neutralize is True
        assert request.neutralizer_path == "neutralizers.parquet"
        return api_module.HpoStudyResponse(
            study_id="study-file-2",
            experiment_id=request.experiment_id,
            study_name=request.study_name,
            status="completed",
            metric=request.metric,
            direction=request.direction,
            n_trials=request.n_trials,
            sampler=request.sampler,
            seed=request.seed,
            best_trial_number=0,
            best_value=0.1,
            best_run_id="run-1",
            config={},
            storage_path=".numereng/hpo/study-file-2",
            error_message=None,
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
        )

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
    assert payload["study_id"] == "study-file-2"


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
            artifacts_path=".numereng/experiments/exp-1/ensembles/ens-1",
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
            artifacts_path=".numereng/experiments/exp-1/ensembles/ens-2",
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
            "--study-name",
            "study-a",
            "--config",
            "configs/base.json",
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
        return api_module.HpoStudyListResponse(
            studies=[
                api_module.HpoStudyResponse(
                    study_id="study-1",
                    experiment_id="exp-1",
                    study_name="study-a",
                    status="completed",
                    metric="bmc_last_200_eras.mean",
                    direction="maximize",
                    n_trials=2,
                    sampler="tpe",
                    seed=1337,
                    best_trial_number=1,
                    best_value=0.12,
                    best_run_id="run-1",
                    config={},
                    storage_path=".numereng/experiments/exp-1/hpo/study-1",
                    error_message=None,
                    created_at="2026-02-22T00:00:00+00:00",
                    updated_at="2026-02-22T00:00:00+00:00",
                )
            ]
        )

    monkeypatch.setattr(api_module, "hpo_list", fake_hpo_list)

    exit_code = cli.main(["hpo", "list", "--format", "json"])
    payload = _parse_stdout_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["studies"][0]["study_id"] == "study-1"


def test_cli_hpo_details_table_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_hpo_get(request: api_module.HpoStudyGetRequest) -> api_module.HpoStudyResponse:
        _ = request
        return api_module.HpoStudyResponse(
            study_id="study-1",
            experiment_id="exp-1",
            study_name="study-a",
            status="completed",
            metric="bmc_last_200_eras.mean",
            direction="maximize",
            n_trials=2,
            sampler="tpe",
            seed=1337,
            best_trial_number=1,
            best_value=0.12,
            best_run_id="run-1",
            config={},
            storage_path=".numereng/experiments/exp-1/hpo/study-1",
            error_message=None,
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "hpo_get", fake_hpo_get)

    exit_code = cli.main(["hpo", "details", "--study-id", "study-1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "study_id: study-1" in captured.out
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
                    artifacts_path=".numereng/experiments/exp-1/ensembles/ens-1",
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
        assert request.neutralize is True
        assert request.neutralizer_path == "neutralizers.parquet"
        assert request.neutralization_proportion == 0.5
        assert request.neutralization_mode == "era"
        assert request.neutralizer_cols == ["feature_x"]
        assert request.neutralization_rank_output is False
        return api_module.HpoStudyResponse(
            study_id="study-neut-1",
            experiment_id=request.experiment_id,
            study_name=request.study_name,
            status="completed",
            metric=request.metric,
            direction=request.direction,
            n_trials=request.n_trials,
            sampler=request.sampler,
            seed=request.seed,
            best_trial_number=0,
            best_value=0.12,
            best_run_id="run-1",
            config={},
            storage_path=".numereng/hpo/study-neut-1",
            error_message=None,
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
        )

    monkeypatch.setattr(api_module, "hpo_create", fake_hpo_create)
    exit_code = cli.main(
        [
            "hpo",
            "create",
            "--study-name",
            "study-a",
            "--config",
            "configs/base.json",
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
