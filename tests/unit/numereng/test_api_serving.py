from __future__ import annotations

from pathlib import Path

import pytest

import numereng.api as api_module
import numereng.api._serving as serving_api_module
from numereng.features.serving import (
    ModelUploadResult,
    PackageDiagnosticsSyncResult,
    PackageScoreResult,
    PickleBuildResult,
    ServingBlendRule,
    ServingComponentInspection,
    ServingComponentSpec,
    ServingInspectionResult,
    ServingRuntimeError,
    SubmissionPackageRecord,
)
from numereng.platform.errors import PackageError


def _package(tmp_path: Path) -> SubmissionPackageRecord:
    return SubmissionPackageRecord(
        package_id="pkg-1",
        experiment_id="exp-1",
        tournament="classic",
        data_version="v5.2",
        package_path=tmp_path / "pkg-1",
        status="created",
        components=(ServingComponentSpec(component_id="dummy", weight=1.0, config_path=tmp_path / "c.json"),),
        blend_rule=ServingBlendRule(),
        neutralization=None,
        artifacts={},
        created_at="2026-04-11T00:00:00Z",
        updated_at="2026-04-11T00:00:00Z",
        provenance={},
    )


def test_api_serve_package_create_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(serving_api_module, "create_submission_package", lambda **_: _package(tmp_path))

    response = api_module.serve_package_create(
        api_module.ServePackageCreateRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            components=[api_module.ServeComponentRequest(weight=1.0, config_path=str(tmp_path / "c.json"))],
            workspace_root=str(tmp_path),
        )
    )

    assert response.package_id == "pkg-1"
    assert response.experiment_id == "exp-1"


def test_api_serve_live_build_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serving_api_module,
        "build_live_submission_package",
        lambda **_: (_ for _ in ()).throw(ServingRuntimeError("serving_live_dataset_unavailable")),
    )

    with pytest.raises(PackageError, match="serving_live_dataset_unavailable"):
        api_module.serve_live_build(
            api_module.ServeLiveBuildRequest(experiment_id="exp-1", package_id="pkg-1", workspace_root=".")
        )


def test_api_serve_package_inspect_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(
        serving_api_module,
        "inspect_package",
        lambda **_: ServingInspectionResult(
            package=package,
            checked_at="2026-04-11T00:00:00Z",
            local_live_compatible=True,
            model_upload_compatible=True,
            artifact_backed=True,
            artifact_ready=True,
            artifact_live_ready=True,
            pickle_upload_ready=False,
            deployment_classification="artifact_backed_live_ready",
            local_live_blockers=(),
            model_upload_blockers=(),
            artifact_blockers=(),
            warnings=(),
            components=(
                ServingComponentInspection(
                    component_id="dummy",
                    local_live_compatible=True,
                    model_upload_compatible=True,
                    artifact_backed=True,
                    artifact_ready=True,
                    model_upload_blockers=(),
                    artifact_blockers=(),
                ),
            ),
            report_path=tmp_path / "report.json",
        ),
    )

    response = api_module.serve_package_inspect(
        api_module.ServePackageInspectRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            workspace_root=str(tmp_path),
        )
    )

    assert response.local_live_compatible is True
    assert response.model_upload_compatible is True
    assert response.deployment_classification == "artifact_backed_live_ready"
    assert response.components[0].component_id == "dummy"


def test_api_serve_pickle_upload_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(
        serving_api_module,
        "upload_submission_pickle",
        lambda **_: ModelUploadResult(
            package=package,
            model_name="main",
            model_id="model-1",
            pickle_path=tmp_path / "model.pkl",
            upload_id="pickle-1",
            data_version="v5.2",
            docker_image=None,
        ),
    )

    response = api_module.serve_pickle_upload(
        api_module.ServePickleUploadRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            model_name="main",
            workspace_root=str(tmp_path),
        )
    )

    assert response.upload_id == "pickle-1"
    assert response.model_id == "model-1"


def test_api_serve_package_score_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(
        serving_api_module,
        "score_submission_package",
        lambda **_: PackageScoreResult(
            package=package,
            dataset="validation",
            data_version="v5.2",
            stage="post_training_full",
            runtime_requested="auto",
            runtime_used="pickle",
            predictions_path=tmp_path / "predictions.parquet",
            score_provenance_path=tmp_path / "score_provenance.json",
            summaries_path=tmp_path / "summaries.json",
            metric_series_path=tmp_path / "metric_series.parquet",
            manifest_path=tmp_path / "manifest.json",
            row_count=4,
            era_count=2,
        ),
    )

    response = api_module.serve_package_score(
        api_module.ServePackageScoreRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            workspace_root=str(tmp_path),
        )
    )

    assert response.runtime_used == "pickle"
    assert response.row_count == 4


def test_api_serve_package_sync_diagnostics_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(
        serving_api_module,
        "sync_submission_package_diagnostics",
        lambda **_: PackageDiagnosticsSyncResult(
            package=package,
            model_id="model-1",
            upload_id="upload-1",
            wait_requested=True,
            diagnostics_status="succeeded",
            terminal=True,
            timed_out=False,
            synced_at="2026-04-16T12:00:00Z",
            compute_status_path=tmp_path / "compute_status.json",
            logs_path=tmp_path / "logs.json",
            raw_path=tmp_path / "raw.json",
            summary_path=tmp_path / "summary.json",
            per_era_path=tmp_path / "per_era.parquet",
        ),
    )

    response = api_module.serve_package_sync_diagnostics(
        api_module.ServePackageSyncDiagnosticsRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            workspace_root=str(tmp_path),
        )
    )

    assert response.diagnostics_status == "succeeded"
    assert response.terminal is True


def test_api_serve_pickle_upload_waits_for_diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr(
        serving_api_module,
        "upload_submission_pickle",
        lambda **_: ModelUploadResult(
            package=package,
            model_name="main",
            model_id="model-1",
            pickle_path=tmp_path / "model.pkl",
            upload_id="pickle-1",
            data_version="v5.2",
            docker_image="Python 3.12",
        ),
    )
    monkeypatch.setattr(
        serving_api_module,
        "sync_submission_package_diagnostics",
        lambda **_: PackageDiagnosticsSyncResult(
            package=package,
            model_id="model-1",
            upload_id="pickle-1",
            wait_requested=True,
            diagnostics_status="pending",
            terminal=False,
            timed_out=True,
            synced_at="2026-04-16T12:00:00Z",
            compute_status_path=tmp_path / "compute_status.json",
            logs_path=tmp_path / "logs.json",
            raw_path=None,
            summary_path=None,
            per_era_path=None,
        ),
    )

    response = api_module.serve_pickle_upload(
        api_module.ServePickleUploadRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            model_name="main",
            wait_diagnostics=True,
            workspace_root=str(tmp_path),
        )
    )

    assert response.diagnostics_synced is True
    assert response.diagnostics_status == "pending"


def test_api_serve_pickle_build_passes_docker_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = _package(tmp_path)
    captured: dict[str, object] = {}

    def _build(**kwargs):
        captured.update(kwargs)
        return PickleBuildResult(
            package=package,
            pickle_path=tmp_path / "model.pkl",
            docker_image="Python 3.12",
            smoke_verified=True,
        )

    monkeypatch.setattr(serving_api_module, "build_submission_pickle", _build)

    response = api_module.serve_pickle_build(
        api_module.ServePickleBuildRequest(
            experiment_id="exp-1",
            package_id="pkg-1",
            docker_image="Python 3.12",
            workspace_root=str(tmp_path),
        )
    )

    assert captured["docker_image"] == "Python 3.12"
    assert response.docker_image == "Python 3.12"
    assert response.smoke_verified is True
