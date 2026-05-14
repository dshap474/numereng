from __future__ import annotations

import json
from pathlib import Path

import pytest

import numereng.api as api_module
from numereng.cli.main import main


def test_cli_serve_package_create_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        api_module,
        "serve_package_create",
        lambda request: api_module.ServePackageResponse(
            package_id=request.package_id,
            experiment_id=request.experiment_id,
            tournament="classic",
            data_version=request.data_version,
            package_path=str(tmp_path / "pkg-1"),
            status="created",
            components=request.components,
            blend_rule=request.blend_rule,
            neutralization=request.neutralization,
            artifacts={},
            created_at="2026-04-11T00:00:00Z",
            updated_at="2026-04-11T00:00:00Z",
            provenance={},
        ),
    )
    components_path = tmp_path / "components.json"
    components_path.write_text(json.dumps([{"weight": 1.0, "config_path": "/tmp/component.json"}]), encoding="utf-8")

    exit_code = main(
        [
            "serve",
            "package",
            "create",
            "--experiment-id",
            "exp-1",
            "--package-id",
            "pkg-1",
            "--components",
            str(components_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["package_id"] == "pkg-1"


def test_cli_serve_package_create_accepts_inline_components_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        api_module,
        "serve_package_create",
        lambda request: api_module.ServePackageResponse(
            package_id=request.package_id,
            experiment_id=request.experiment_id,
            tournament="classic",
            data_version=request.data_version,
            package_path=str(tmp_path / "pkg-1"),
            status="created",
            components=request.components,
            blend_rule=request.blend_rule,
            neutralization=request.neutralization,
            artifacts={},
            created_at="2026-04-11T00:00:00Z",
            updated_at="2026-04-11T00:00:00Z",
            provenance={},
        ),
    )

    exit_code = main(
        [
            "serve",
            "package",
            "create",
            "--experiment-id",
            "exp-1",
            "--package-id",
            "pkg-1",
            "--components",
            json.dumps(
                [
                    {"component_id": "a", "run_id": "run-a", "weight": 0.5},
                    {"component_id": "b", "run_id": "run-b", "weight": 0.5},
                ]
            ),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert [item["run_id"] for item in payload["components"]] == ["run-a", "run-b"]


def test_cli_serve_live_build_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    package = api_module.ServePackageResponse(
        package_id="pkg-1",
        experiment_id="exp-1",
        tournament="classic",
        data_version="v5.2",
        package_path="/tmp/pkg-1",
        status="live_built",
        components=[api_module.ServeComponentRequest(weight=1.0, config_path="/tmp/component.json")],
        blend_rule=api_module.ServeBlendRuleRequest(),
        neutralization=None,
        artifacts={},
        created_at="2026-04-11T00:00:00Z",
        updated_at="2026-04-11T00:00:00Z",
        provenance={},
    )
    monkeypatch.setattr(
        api_module,
        "serve_live_build",
        lambda request: api_module.ServeLiveBuildResponse(
            package=package,
            current_round=777,
            live_dataset_name="v5.2/live.parquet",
            live_benchmark_dataset_name="v5.2/live_benchmark_models.parquet",
            live_dataset_path="/tmp/live.parquet",
            live_benchmark_dataset_path="/tmp/live_benchmark_models.parquet",
            component_prediction_paths=["/tmp/component.parquet"],
            blended_predictions_path="/tmp/blended.parquet",
            submission_predictions_path="/tmp/submission.parquet",
        ),
    )

    exit_code = main(["serve", "live", "build", "--experiment-id", "exp-1", "--package-id", "pkg-1"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["current_round"] == 777


def test_cli_serve_package_inspect_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    package = api_module.ServePackageResponse(
        package_id="pkg-1",
        experiment_id="exp-1",
        tournament="classic",
        data_version="v5.2",
        package_path="/tmp/pkg-1",
        status="inspected",
        components=[api_module.ServeComponentRequest(weight=1.0, config_path="/tmp/component.json")],
        blend_rule=api_module.ServeBlendRuleRequest(),
        neutralization=None,
        artifacts={},
        created_at="2026-04-11T00:00:00Z",
        updated_at="2026-04-11T00:00:00Z",
        provenance={},
    )
    monkeypatch.setattr(
        api_module,
        "serve_package_inspect",
        lambda request: api_module.ServePackageInspectResponse(
            package=package,
            checked_at="2026-04-11T00:00:01Z",
            local_live_compatible=True,
            model_upload_compatible=False,
            artifact_backed=False,
            artifact_ready=False,
            artifact_live_ready=False,
            pickle_upload_ready=False,
            deployment_classification="local_live_only",
            local_live_blockers=[],
            model_upload_blockers=["serving_model_upload_custom_modules_not_supported"],
            artifact_blockers=["serving_component_config_backed_only"],
            warnings=[],
            components=[
                api_module.ServeComponentInspectionResponse(
                    component_id="dummy",
                    local_live_compatible=True,
                    model_upload_compatible=False,
                    artifact_backed=False,
                    artifact_ready=False,
                    local_live_blockers=[],
                    model_upload_blockers=["serving_model_upload_custom_modules_not_supported"],
                    artifact_blockers=["serving_component_config_backed_only"],
                    warnings=[],
                )
            ],
            report_path="/tmp/report.json",
        ),
    )

    exit_code = main(["serve", "package", "inspect", "--experiment-id", "exp-1", "--package-id", "pkg-1"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["model_upload_compatible"] is False
    assert payload["deployment_classification"] == "local_live_only"


def test_cli_serve_package_score_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    package = api_module.ServePackageResponse(
        package_id="pkg-1",
        experiment_id="exp-1",
        tournament="classic",
        data_version="v5.2",
        package_path="/tmp/pkg-1",
        status="created",
        components=[api_module.ServeComponentRequest(weight=1.0, config_path="/tmp/component.json")],
        blend_rule=api_module.ServeBlendRuleRequest(),
        neutralization=None,
        artifacts={},
        created_at="2026-04-11T00:00:00Z",
        updated_at="2026-04-11T00:00:00Z",
        provenance={},
    )
    monkeypatch.setattr(
        api_module,
        "serve_package_score",
        lambda request: api_module.ServePackageScoreResponse(
            package=package,
            dataset=request.dataset,
            data_version="v5.2",
            stage=request.stage,
            runtime_requested=request.runtime,
            runtime_used="local",
            predictions_path="/tmp/predictions.parquet",
            score_provenance_path="/tmp/score_provenance.json",
            summaries_path="/tmp/summaries.json",
            metric_series_path="/tmp/metric_series.parquet",
            manifest_path="/tmp/manifest.json",
            row_count=4,
            era_count=2,
        ),
    )

    exit_code = main(
        [
            "serve",
            "package",
            "score",
            "--experiment-id",
            "exp-1",
            "--package-id",
            "pkg-1",
            "--runtime",
            "local",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["runtime_used"] == "local"
    assert payload["row_count"] == 4


def test_cli_serve_package_sync_diagnostics_no_wait(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = api_module.ServePackageResponse(
        package_id="pkg-1",
        experiment_id="exp-1",
        tournament="classic",
        data_version="v5.2",
        package_path="/tmp/pkg-1",
        status="pickle_uploaded",
        components=[api_module.ServeComponentRequest(weight=1.0, config_path="/tmp/component.json")],
        blend_rule=api_module.ServeBlendRuleRequest(),
        neutralization=None,
        artifacts={},
        created_at="2026-04-11T00:00:00Z",
        updated_at="2026-04-11T00:00:00Z",
        provenance={},
    )
    monkeypatch.setattr(
        api_module,
        "serve_package_sync_diagnostics",
        lambda request: api_module.ServePackageSyncDiagnosticsResponse(
            package=package,
            model_id="model-1",
            upload_id="upload-1",
            wait_requested=request.wait,
            diagnostics_status="pending",
            terminal=False,
            timed_out=False,
            synced_at="2026-04-16T12:00:00Z",
            compute_status_path="/tmp/compute_status.json",
            logs_path="/tmp/logs.json",
            raw_path=None,
            summary_path=None,
            per_era_path=None,
        ),
    )

    exit_code = main(
        [
            "serve",
            "package",
            "sync-diagnostics",
            "--experiment-id",
            "exp-1",
            "--package-id",
            "pkg-1",
            "--no-wait",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["wait_requested"] is False
    assert payload["diagnostics_status"] == "pending"


def test_cli_serve_pickle_build_passes_docker_image(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured_request: dict[str, object] = {}
    package = api_module.ServePackageResponse(
        package_id="pkg-1",
        experiment_id="exp-1",
        tournament="classic",
        data_version="v5.2",
        package_path="/tmp/pkg-1",
        status="pickle_built",
        components=[api_module.ServeComponentRequest(weight=1.0, config_path="/tmp/component.json")],
        blend_rule=api_module.ServeBlendRuleRequest(),
        neutralization=None,
        artifacts={},
        created_at="2026-04-11T00:00:00Z",
        updated_at="2026-04-11T00:00:00Z",
        provenance={},
    )

    def _build(request: api_module.ServePickleBuildRequest) -> api_module.ServePickleBuildResponse:
        captured_request["docker_image"] = request.docker_image
        return api_module.ServePickleBuildResponse(
            package=package,
            pickle_path="/tmp/model.pkl",
            docker_image="Python 3.12",
            smoke_verified=True,
        )

    monkeypatch.setattr(api_module, "serve_pickle_build", _build)

    exit_code = main(
        [
            "serve",
            "pickle",
            "build",
            "--experiment-id",
            "exp-1",
            "--package-id",
            "pkg-1",
            "--docker-image",
            "Python 3.12",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured_request["docker_image"] == "Python 3.12"
    assert payload["docker_image"] == "Python 3.12"
    assert payload["smoke_verified"] is True


def test_cli_serve_pickle_upload_wait_diagnostics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured_request: dict[str, object] = {}
    package = api_module.ServePackageResponse(
        package_id="pkg-1",
        experiment_id="exp-1",
        tournament="classic",
        data_version="v5.2",
        package_path="/tmp/pkg-1",
        status="pickle_uploaded",
        components=[api_module.ServeComponentRequest(weight=1.0, config_path="/tmp/component.json")],
        blend_rule=api_module.ServeBlendRuleRequest(),
        neutralization=None,
        artifacts={},
        created_at="2026-04-11T00:00:00Z",
        updated_at="2026-04-11T00:00:00Z",
        provenance={},
    )

    def _upload(request: api_module.ServePickleUploadRequest) -> api_module.ServePickleUploadResponse:
        captured_request["wait_diagnostics"] = request.wait_diagnostics
        return api_module.ServePickleUploadResponse(
            package=package,
            pickle_path="/tmp/model.pkl",
            model_name="main",
            model_id="model-1",
            upload_id="upload-1",
            data_version="v5.2",
            docker_image="Python 3.12",
            diagnostics_synced=True,
            diagnostics_status="pending",
        )

    monkeypatch.setattr(api_module, "serve_pickle_upload", _upload)

    exit_code = main(
        [
            "serve",
            "pickle",
            "upload",
            "--experiment-id",
            "exp-1",
            "--package-id",
            "pkg-1",
            "--model-name",
            "main",
            "--wait-diagnostics",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured_request["wait_diagnostics"] is True
    assert payload["diagnostics_synced"] is True


def test_cli_serve_live_build_rejects_runtime_flag(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "serve",
            "live",
            "build",
            "--experiment-id",
            "exp-1",
            "--package-id",
            "pkg-1",
            "--runtime",
            "pickle",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown arguments: --runtime" in captured.err


def test_cli_serve_pickle_build_rejects_stage_flag(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "serve",
            "pickle",
            "build",
            "--experiment-id",
            "exp-1",
            "--package-id",
            "pkg-1",
            "--stage",
            "post_training_full",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown arguments: --stage" in captured.err
