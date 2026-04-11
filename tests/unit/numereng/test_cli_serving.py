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
