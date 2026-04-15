from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

import numereng.api as api_module
from numereng import __version__
from numereng.api import (
    AwsImageBuildPushRequest,
    AwsTrainCancelRequest,
    AwsTrainExtractRequest,
    AwsTrainLogsRequest,
    AwsTrainPullRequest,
    AwsTrainStatusRequest,
    AwsTrainSubmitRequest,
    CloudAwsResponse,
    CloudEc2Response,
    CloudModalResponse,
    ExperimentArchiveRequest,
    ExperimentArchiveResponse,
    ExperimentCreateRequest,
    ExperimentGetRequest,
    ExperimentListRequest,
    ExperimentListResponse,
    ExperimentPackRequest,
    ExperimentPackResponse,
    ExperimentPromoteRequest,
    ExperimentPromoteResponse,
    ExperimentReportRequest,
    ExperimentReportResponse,
    ExperimentScoreRoundRequest,
    ExperimentScoreRoundResponse,
    ExperimentTrainRequest,
    ExperimentTrainResponse,
    HealthResponse,
    ModalDataSyncRequest,
    ModalDeployRequest,
    ModalTrainCancelRequest,
    ModalTrainLogsRequest,
    ModalTrainPullRequest,
    ModalTrainStatusRequest,
    ModalTrainSubmitRequest,
    MonitorSnapshotRequest,
    MonitorSnapshotResponse,
    NumeraiCurrentRoundRequest,
    NumeraiCurrentRoundResponse,
    NumeraiDatasetDownloadRequest,
    NumeraiDatasetDownloadResponse,
    NumeraiDatasetListRequest,
    NumeraiDatasetListResponse,
    NumeraiModelsRequest,
    NumeraiModelsResponse,
    NumeraiTournament,
    PackageError,
    RemoteRepoSyncRequest,
    RemoteRepoSyncResponse,
    RemoteTargetListRequest,
    RemoteTargetListResponse,
    RemoteTrainLaunchRequest,
    RemoteTrainLaunchResponse,
    RemoteVizBootstrapRequest,
    RemoteVizBootstrapResponse,
    ResearchInitRequest,
    ResearchInitResponse,
    ResearchProgramListRequest,
    ResearchProgramListResponse,
    ResearchProgramShowRequest,
    ResearchProgramShowResponse,
    ResearchRunRequest,
    ResearchRunResponse,
    ResearchStatusRequest,
    ResearchStatusResponse,
    RunCancelRequest,
    RunCancelResponse,
    RunLifecycleRequest,
    RunLifecycleResponse,
    ScoreRunRequest,
    ScoreRunResponse,
    StoreDoctorRequest,
    StoreDoctorResponse,
    StoreIndexRequest,
    StoreIndexResponse,
    StoreInitRequest,
    StoreInitResponse,
    StoreMaterializeVizArtifactsRequest,
    StoreMaterializeVizArtifactsResponse,
    StoreRebuildRequest,
    StoreRebuildResponse,
    StoreRunExecutionBackfillRequest,
    StoreRunExecutionBackfillResponse,
    StoreRunLifecycleRepairRequest,
    StoreRunLifecycleRepairResponse,
    SubmissionRequest,
    SubmissionResponse,
    TrainRunRequest,
    TrainRunResponse,
    VizAppRequest,
    build_monitor_snapshot,
    cancel_run,
    create_viz_app,
    download_numerai_dataset,
    experiment_archive,
    experiment_create,
    experiment_get,
    experiment_list,
    experiment_pack,
    experiment_promote,
    experiment_report,
    experiment_score_round,
    experiment_train,
    experiment_unarchive,
    get_health,
    get_numerai_current_round,
    get_run_lifecycle,
    list_numerai_datasets,
    list_numerai_models,
    remote_bootstrap_viz,
    remote_list_targets,
    remote_repo_sync,
    remote_train_launch,
    research_init,
    research_program_list,
    research_program_show,
    research_run,
    research_status,
    run_bootstrap_check,
    run_training,
    score_run,
    store_backfill_run_execution,
    store_doctor,
    store_index_run,
    store_init,
    store_materialize_viz_artifacts,
    store_rebuild,
    store_repair_run_lifecycles,
    submit_predictions,
)
from numereng.features.agentic_research import (
    ResearchBestRun,
    ResearchInitResult,
    ResearchRunResult,
    ResearchStatusResult,
)
from numereng.features.agentic_research.utils.types import (
    ResearchProgramCatalogEntry,
    ResearchProgramConfigPolicy,
    ResearchProgramDefinition,
    ResearchProgramDetails,
    ResearchProgramMetricPolicy,
    ResearchProgramRoundPolicy,
)
from numereng.features.cloud.aws import CloudAwsError, CloudEc2Error
from numereng.features.cloud.modal import CloudModalError
from numereng.features.experiments import (
    ExperimentArchiveResult,
    ExperimentNotFoundError,
    ExperimentPackResult,
    ExperimentPromotionResult,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    ExperimentScoreRoundResult,
    ExperimentTrainResult,
    ExperimentValidationError,
)
from numereng.features.remote_ops.contracts import (
    RemoteRepoSyncResult,
    RemoteTargetRecord,
    RemoteTrainLaunchResult,
    RemoteVizBootstrapResult,
    RemoteVizBootstrapTargetResult,
)
from numereng.features.store import (
    StoreDoctorResult,
    StoreError,
    StoreIndexResult,
    StoreInitResult,
    StoreMaterializeVizArtifactsResult,
    StoreRebuildResult,
    StoreRunExecutionBackfillResult,
)
from numereng.features.submission import (
    SubmissionLiveUniverseUnavailableError,
    SubmissionModelNotFoundError,
    SubmissionPredictionsReadError,
    SubmissionResult,
    SubmissionRunIdInvalidError,
    SubmissionRunPredictionsNotLiveEligibleError,
    SubmissionRunPredictionsPathUnsafeError,
)
from numereng.features.telemetry import get_launch_metadata
from numereng.features.telemetry.contracts import RunCancelResult, RunLifecycleRecord, RunLifecycleRepairResult
from numereng.features.training import (
    ScoreRunResult,
    TrainingDataError,
    TrainingError,
    TrainingModelError,
    TrainingRunResult,
)
from numereng.platform.errors import ForumScraperError, NumeraiClientError

CloudApiFunc = Callable[[Any], CloudEc2Response]
ManagedCloudApiFunc = Callable[[Any], CloudAwsResponse]
ModalCloudApiFunc = Callable[[Any], CloudModalResponse]


class _FakeNumeraiClient:
    def __init__(self) -> None:
        self.last_round: int | None = None
        self.last_download: tuple[str, str | None, int | None] | None = None

    def list_datasets(self, *, round_num: int | None = None) -> list[str]:
        self.last_round = round_num
        return ["v5.2/train_int8.parquet"]

    def download_dataset(
        self,
        *,
        filename: str,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        self.last_download = (filename, dest_path, round_num)
        return dest_path or filename

    def get_models(self) -> dict[str, str]:
        return {"main": "model-1"}

    def get_current_round(self) -> int | None:
        return 777


def test_get_health_returns_public_model() -> None:
    response = get_health()
    assert isinstance(response, HealthResponse)
    assert response.status == "ok"
    assert response.package == "numereng"
    assert response.version == __version__


def test_run_bootstrap_check_success_returns_health_payload() -> None:
    response = run_bootstrap_check()

    assert isinstance(response, HealthResponse)
    assert response.status == "ok"
    assert response.package == "numereng"
    assert response.version == __version__


def test_run_bootstrap_check_translates_internal_error() -> None:
    with pytest.raises(PackageError, match="bootstrap_check_failed"):
        run_bootstrap_check(fail=True)


def test_workspace_init_returns_synced_workspace_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from numereng.features.workspace.runtime import WorkspaceSyncResult
    from numereng.features.workspace.service import WorkspaceInitResult

    def fake_init_workspace_record(**kwargs: Any) -> WorkspaceInitResult:
        assert kwargs["runtime_source"] == "path"
        assert kwargs["runtime_path"] == "/tmp/numereng-src"
        assert kwargs["with_training"] is True
        return WorkspaceInitResult(
            workspace_root=Path("/tmp/numerai-dev"),
            store_root=Path("/tmp/numerai-dev/.numereng"),
            created_paths=(Path("/tmp/numerai-dev/experiments"),),
            skipped_existing_paths=(),
            installed_skill_ids=("numereng-experiment-ops",),
            sync_result=WorkspaceSyncResult(
                workspace_root=Path("/tmp/numerai-dev"),
                store_root=Path("/tmp/numerai-dev/.numereng"),
                workspace_project_path=Path("/tmp/numerai-dev/pyproject.toml"),
                python_version_path=Path("/tmp/numerai-dev/.python-version"),
                venv_path=Path("/tmp/numerai-dev/.venv"),
                created_paths=(Path("/tmp/numerai-dev/pyproject.toml"),),
                updated_paths=(),
                runtime_source="path",
                runtime_path=Path("/tmp/numereng-src"),
                extras=("training",),
                dependency_spec="numereng[training]",
                installed_numereng_version="1.2.3",
                verified_dependencies=("numereng", "cloudpickle"),
            ),
        )

    monkeypatch.setattr("numereng.api._workspace.init_workspace_record", fake_init_workspace_record)

    response = api_module.workspace_init(
        api_module.WorkspaceInitRequest(
            workspace_root="/tmp/numerai-dev",
            runtime_source="path",
            runtime_path="/tmp/numereng-src",
            with_training=True,
        )
    )

    assert response.runtime_source == "path"
    assert response.runtime_path == "/tmp/numereng-src"
    assert response.venv_path == "/tmp/numerai-dev/.venv"
    assert response.verified_dependencies == ["numereng", "cloudpickle"]


def test_workspace_sync_returns_synced_runtime_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from numereng.features.workspace.runtime import WorkspaceSyncResult

    def fake_sync_workspace_environment_record(**kwargs: Any) -> WorkspaceSyncResult:
        assert kwargs["workspace_root"] == "/tmp/numerai-dev"
        return WorkspaceSyncResult(
            workspace_root=Path("/tmp/numerai-dev"),
            store_root=Path("/tmp/numerai-dev/.numereng"),
            workspace_project_path=Path("/tmp/numerai-dev/pyproject.toml"),
            python_version_path=Path("/tmp/numerai-dev/.python-version"),
            venv_path=Path("/tmp/numerai-dev/.venv"),
            created_paths=(),
            updated_paths=(Path("/tmp/numerai-dev/pyproject.toml"),),
            runtime_source="pypi",
            runtime_path=None,
            extras=(),
            dependency_spec="numereng==1.2.3",
            installed_numereng_version="1.2.3",
            verified_dependencies=("numereng", "cloudpickle"),
        )

    monkeypatch.setattr(
        "numereng.api._workspace.sync_workspace_environment_record",
        fake_sync_workspace_environment_record,
    )

    response = api_module.workspace_sync(api_module.WorkspaceSyncRequest(workspace_root="/tmp/numerai-dev"))

    assert response.runtime_source == "pypi"
    assert response.updated_paths == ["/tmp/numerai-dev/pyproject.toml"]


def test_lazy_public_api_translates_missing_runtime_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    api_module.__dict__.pop("serve_pickle_build", None)

    def fake_import_module(name: str):
        if name == "numereng.api._serving":
            raise ModuleNotFoundError("No module named 'cloudpickle'", name="cloudpickle")
        return importlib.import_module(name)

    monkeypatch.setattr("numereng.api.import_module", fake_import_module)

    with pytest.raises(PackageError, match="runtime_dependency_missing:cloudpickle:run_numereng_workspace_sync"):
        getattr(api_module, "serve_pickle_build")


def test_public_pipeline_module_is_importable() -> None:
    pipeline_module = importlib.import_module("numereng.api.pipeline")

    assert hasattr(pipeline_module, "run_training_pipeline")


def test_build_monitor_snapshot_returns_public_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "numereng_viz.build_monitor_snapshot",
        lambda **_: {
            "generated_at": "2026-03-25T00:00:00+00:00",
            "source": {
                "kind": "local",
                "id": "local",
                "label": "Local store",
                "host": "host",
                "store_root": ".numereng",
                "state": "live",
            },
            "summary": {
                "total_experiments": 1,
                "active_experiments": 1,
                "completed_experiments": 0,
                "live_experiments": 1,
                "live_runs": 1,
                "queued_runs": 0,
                "attention_count": 0,
            },
            "experiments": [],
            "live_experiments": [],
            "live_runs": [],
            "recent_activity": [],
        },
    )

    response = build_monitor_snapshot(MonitorSnapshotRequest(store_root=".numereng", refresh_cloud=False))

    assert isinstance(response, MonitorSnapshotResponse)
    assert response.source.kind == "local"
    assert response.summary.live_runs == 1


def test_create_viz_app_returns_fastapi_app(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_app = object()

    monkeypatch.setattr(
        "numereng_viz.create_app", lambda **kwargs: fake_app if kwargs["workspace_root"] == "/tmp/ws" else None
    )

    response = create_viz_app(VizAppRequest(workspace_root="/tmp/ws"))

    assert response is fake_app


def test_remote_list_targets_returns_public_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "numereng.api._remote.list_remote_targets_record",
        lambda: (
            RemoteTargetRecord(
                id="pc",
                label="Windows PC",
                kind="ssh",
                shell="powershell",
                repo_root=r"C:\Users\<you>\remote-access\numereng",
                store_root=r"C:\Users\<you>\remote-access\numereng\.numereng",
                runner_cmd="uv run numereng",
                python_cmd="uv run python",
                tags=("pc", "windows"),
            ),
        ),
    )

    response = remote_list_targets(RemoteTargetListRequest())

    assert isinstance(response, RemoteTargetListResponse)
    assert response.targets[0].python_cmd == "uv run python"
    assert response.targets[0].id == "pc"


def test_remote_repo_sync_returns_public_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "numereng.api._remote.sync_remote_repo_record",
        lambda **_: RemoteRepoSyncResult(
            target_id="pc",
            repo_root=r"C:\Users\<you>\remote-access\numereng",
            manifest_hash="hash",
            local_commit_sha="abc123",
            dirty=False,
            synced_files=10,
            deleted_files=1,
            synced_at="2026-03-27T00:00:00+00:00",
            local_marker_path=r"/tmp/.numereng/remote_ops/sync/pc/repo.json",
            remote_marker_path=r"C:\Users\<you>\remote-access\numereng\.numereng\remote_ops\sync\pc\repo.json",
        ),
    )

    response = remote_repo_sync(RemoteRepoSyncRequest(target_id="pc"))

    assert isinstance(response, RemoteRepoSyncResponse)
    assert response.target_id == "pc"
    assert response.synced_files == 10


def test_remote_bootstrap_viz_returns_public_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    state_path = store_root / "remote_ops" / "bootstrap" / "viz.json"
    monkeypatch.setattr(
        "numereng.api._remote.bootstrap_viz_remotes_record",
        lambda **_: RemoteVizBootstrapResult(
            store_root=store_root,
            state_path=state_path,
            bootstrapped_at="2026-03-28T00:00:00+00:00",
            ready_count=1,
            degraded_count=1,
            targets=(
                RemoteVizBootstrapTargetResult(
                    target=RemoteTargetRecord(
                        id="pc",
                        label="Windows PC",
                        kind="ssh",
                        shell="powershell",
                        repo_root=r"C:\Users\<you>\remote-access\numereng",
                        store_root=r"C:\Users\<you>\remote-access\numereng\.numereng",
                        runner_cmd="uv run numereng",
                        python_cmd="uv run python",
                        tags=("pc",),
                    ),
                    bootstrap_status="ready",
                    last_bootstrap_at="2026-03-28T00:00:00+00:00",
                    last_bootstrap_error=None,
                    repo_synced=True,
                    repo_sync_skipped=False,
                    doctor_ok=True,
                    issues=(),
                ),
                RemoteVizBootstrapTargetResult(
                    target=RemoteTargetRecord(
                        id="offline",
                        label="Offline Host",
                        kind="ssh",
                        shell="posix",
                        repo_root="/srv/numereng",
                        store_root="/srv/numereng/.numereng",
                        runner_cmd="uv run numereng",
                        python_cmd="uv run python",
                        tags=("ci",),
                    ),
                    bootstrap_status="degraded",
                    last_bootstrap_at="2026-03-28T00:00:00+00:00",
                    last_bootstrap_error="monitor_snapshot_failed",
                    repo_synced=False,
                    repo_sync_skipped=True,
                    doctor_ok=False,
                    issues=("monitor_snapshot_failed",),
                ),
            ),
        ),
    )

    response = remote_bootstrap_viz(RemoteVizBootstrapRequest(store_root=str(store_root)))

    assert isinstance(response, RemoteVizBootstrapResponse)
    assert response.ready_count == 1
    assert response.degraded_count == 1
    assert response.targets[1].bootstrap_status == "degraded"


def test_remote_train_launch_returns_public_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "numereng.api._remote.remote_run_train_record",
        lambda **_: RemoteTrainLaunchResult(
            target_id="pc",
            launch_id="launch-1",
            remote_config_path=r"C:\Users\<you>\remote-access\numereng\.numereng\tmp\remote-configs\cfg.json",
            remote_log_path=r"C:\Users\<you>\remote-access\numereng\.numereng\remote_ops\launches\launch-1.log",
            remote_metadata_path=r"C:\Users\<you>\remote-access\numereng\.numereng\remote_ops\launches\launch-1.json",
            remote_pid=4321,
            launched_at="2026-03-27T00:00:00+00:00",
            sync_repo_policy="auto",
            repo_synced=True,
            experiment_synced=False,
        ),
    )

    response = remote_train_launch(RemoteTrainLaunchRequest(target_id="pc", config_path="cfg.json"))

    assert isinstance(response, RemoteTrainLaunchResponse)
    assert response.remote_pid == 4321
    assert response.repo_synced is True


def test_old_deep_run_module_is_not_publicly_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("numereng.api.run")


def test_submission_request_requires_exactly_one_source() -> None:
    with pytest.raises(ValidationError, match="exactly one of run_id or predictions_path is required"):
        SubmissionRequest(model_name="main")

    with pytest.raises(ValidationError, match="exactly one of run_id or predictions_path is required"):
        SubmissionRequest(model_name="main", run_id="run-1", predictions_path="predictions.csv")


def test_submission_request_requires_neutralizer_path_when_neutralizing() -> None:
    with pytest.raises(ValidationError, match="neutralizer_path is required when neutralize is true"):
        SubmissionRequest(model_name="main", run_id="run-1", neutralize=True)


def test_submission_request_allows_non_classic_tournament() -> None:
    request = SubmissionRequest(model_name="main", run_id="run-1", tournament="signals")

    assert request.tournament == "signals"


def test_submit_predictions_file_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_predictions_file(
        *,
        predictions_path: str,
        model_name: str,
        tournament: NumeraiTournament,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        assert predictions_path == "predictions.csv"
        assert model_name == "main"
        assert tournament == "classic"
        assert allow_non_live_artifact is False
        assert neutralize is False
        assert neutralizer_path is None
        assert neutralization_proportion == 0.5
        assert neutralization_mode == "era"
        assert neutralizer_cols is None
        assert neutralization_rank_output is True
        return SubmissionResult(
            submission_id="submission-1",
            model_name="main",
            model_id="model-1",
            predictions_path=Path("/tmp/predictions.csv"),
        )

    monkeypatch.setattr(api_module, "submit_predictions_file", fake_submit_predictions_file)
    request = SubmissionRequest(model_name="main", predictions_path="predictions.csv")

    response = submit_predictions(request)

    assert isinstance(response, SubmissionResponse)
    assert response.submission_id == "submission-1"
    assert response.model_id == "model-1"
    assert response.model_name == "main"
    assert response.predictions_path == "/tmp/predictions.csv"
    assert response.run_id is None


def test_submit_predictions_file_passes_allow_non_live_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_predictions_file(
        *,
        predictions_path: str,
        model_name: str,
        tournament: NumeraiTournament,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            predictions_path,
            model_name,
            tournament,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        assert allow_non_live_artifact is True
        return SubmissionResult(
            submission_id="submission-1",
            model_name="main",
            model_id="model-1",
            predictions_path=Path("/tmp/predictions.csv"),
        )

    monkeypatch.setattr(api_module, "submit_predictions_file", fake_submit_predictions_file)
    request = SubmissionRequest(
        model_name="main",
        predictions_path="predictions.csv",
        allow_non_live_artifact=True,
    )

    response = submit_predictions(request)
    assert response.submission_id == "submission-1"


def test_submit_predictions_preserves_explicit_empty_neutralizer_cols(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_predictions_file(
        *,
        predictions_path: str,
        model_name: str,
        tournament: NumeraiTournament,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            predictions_path,
            model_name,
            tournament,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralization_rank_output,
        )
        assert neutralizer_cols == ()
        return SubmissionResult(
            submission_id="submission-1",
            model_name="main",
            model_id="model-1",
            predictions_path=Path("/tmp/predictions.csv"),
        )

    monkeypatch.setattr(api_module, "submit_predictions_file", fake_submit_predictions_file)

    response = submit_predictions(
        SubmissionRequest(
            model_name="main",
            predictions_path="predictions.csv",
            neutralizer_cols=[],
        )
    )

    assert response.submission_id == "submission-1"


def test_submit_predictions_run_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_run_predictions(
        *,
        run_id: str,
        model_name: str,
        tournament: NumeraiTournament,
        store_root: str,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        assert run_id == "run-1"
        assert model_name == "main"
        assert tournament == "classic"
        assert store_root == ".numereng"
        assert allow_non_live_artifact is False
        assert neutralize is False
        assert neutralizer_path is None
        assert neutralization_proportion == 0.5
        assert neutralization_mode == "era"
        assert neutralizer_cols is None
        assert neutralization_rank_output is True
        return SubmissionResult(
            submission_id="submission-2",
            model_name="main",
            model_id="model-1",
            predictions_path=Path("/tmp/run-1-preds.csv"),
            run_id="run-1",
        )

    monkeypatch.setattr(api_module, "submit_run_predictions", fake_submit_run_predictions)
    request = SubmissionRequest(model_name="main", run_id="run-1")

    response = submit_predictions(request)

    assert response.submission_id == "submission-2"
    assert response.run_id == "run-1"


def test_submit_predictions_translates_feature_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_predictions_file(
        *,
        predictions_path: str,
        model_name: str,
        tournament: NumeraiTournament,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            predictions_path,
            model_name,
            tournament,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        raise SubmissionModelNotFoundError("main")

    monkeypatch.setattr(api_module, "submit_predictions_file", fake_submit_predictions_file)
    request = SubmissionRequest(model_name="main", predictions_path="predictions.csv")

    with pytest.raises(PackageError, match="submission_model_not_found"):
        submit_predictions(request)


def test_submit_predictions_translates_run_not_live_eligible_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_run_predictions(
        *,
        run_id: str,
        model_name: str,
        tournament: NumeraiTournament,
        store_root: str,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            run_id,
            model_name,
            tournament,
            store_root,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        raise SubmissionRunPredictionsNotLiveEligibleError("submission_run_predictions_not_live_eligible")

    monkeypatch.setattr(api_module, "submit_run_predictions", fake_submit_run_predictions)
    request = SubmissionRequest(model_name="main", run_id="run-1")

    with pytest.raises(PackageError, match="submission_run_predictions_not_live_eligible"):
        submit_predictions(request)


def test_submit_predictions_translates_run_id_invalid_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_run_predictions(
        *,
        run_id: str,
        model_name: str,
        tournament: NumeraiTournament,
        store_root: str,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            run_id,
            model_name,
            tournament,
            store_root,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        raise SubmissionRunIdInvalidError("submission_run_id_invalid")

    monkeypatch.setattr(api_module, "submit_run_predictions", fake_submit_run_predictions)
    request = SubmissionRequest(model_name="main", run_id="run-1")

    with pytest.raises(PackageError, match="submission_run_id_invalid"):
        submit_predictions(request)


def test_submit_predictions_translates_run_predictions_path_unsafe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_run_predictions(
        *,
        run_id: str,
        model_name: str,
        tournament: NumeraiTournament,
        store_root: str,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            run_id,
            model_name,
            tournament,
            store_root,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        raise SubmissionRunPredictionsPathUnsafeError("submission_run_predictions_path_unsafe")

    monkeypatch.setattr(api_module, "submit_run_predictions", fake_submit_run_predictions)
    request = SubmissionRequest(model_name="main", run_id="run-1")

    with pytest.raises(PackageError, match="submission_run_predictions_path_unsafe"):
        submit_predictions(request)


def test_submit_predictions_translates_predictions_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_predictions_file(
        *,
        predictions_path: str,
        model_name: str,
        tournament: NumeraiTournament,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            predictions_path,
            model_name,
            tournament,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        raise SubmissionPredictionsReadError("submission_predictions_read_failed")

    monkeypatch.setattr(api_module, "submit_predictions_file", fake_submit_predictions_file)
    request = SubmissionRequest(model_name="main", predictions_path="predictions.csv")

    with pytest.raises(PackageError, match="submission_predictions_read_failed"):
        submit_predictions(request)


def test_submit_predictions_translates_live_universe_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_predictions_file(
        *,
        predictions_path: str,
        model_name: str,
        tournament: NumeraiTournament,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            predictions_path,
            model_name,
            tournament,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        raise SubmissionLiveUniverseUnavailableError("submission_live_universe_unavailable")

    monkeypatch.setattr(api_module, "submit_predictions_file", fake_submit_predictions_file)
    request = SubmissionRequest(model_name="main", predictions_path="predictions.csv")

    with pytest.raises(PackageError, match="submission_live_universe_unavailable"):
        submit_predictions(request)


def test_submit_predictions_translates_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_predictions_file(
        *,
        predictions_path: str,
        model_name: str,
        tournament: NumeraiTournament,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            predictions_path,
            model_name,
            tournament,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        raise NumeraiClientError("numerai_upload_predictions_failed")

    monkeypatch.setattr(api_module, "submit_predictions_file", fake_submit_predictions_file)
    request = SubmissionRequest(model_name="main", predictions_path="predictions.csv")

    with pytest.raises(PackageError, match="numerai_upload_predictions_failed"):
        submit_predictions(request)


def test_submit_predictions_translates_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_predictions_file(
        *,
        predictions_path: str,
        model_name: str,
        tournament: NumeraiTournament,
        allow_non_live_artifact: bool,
        neutralize: bool,
        neutralizer_path: str | None,
        neutralization_proportion: float,
        neutralization_mode: str,
        neutralizer_cols: tuple[str, ...] | None,
        neutralization_rank_output: bool,
    ) -> SubmissionResult:
        _ = (
            predictions_path,
            model_name,
            tournament,
            allow_non_live_artifact,
            neutralize,
            neutralizer_path,
            neutralization_proportion,
            neutralization_mode,
            neutralizer_cols,
            neutralization_rank_output,
        )
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module, "submit_predictions_file", fake_submit_predictions_file)
    request = SubmissionRequest(model_name="main", predictions_path="predictions.csv")

    with pytest.raises(PackageError, match="submission_unexpected_error:RuntimeError"):
        submit_predictions(request)


def test_list_numerai_datasets_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeNumeraiClient()
    captured_tournament: NumeraiTournament | None = None

    def fake_create_client(*, tournament: NumeraiTournament = "classic") -> _FakeNumeraiClient:
        nonlocal captured_tournament
        captured_tournament = tournament
        return fake_client

    monkeypatch.setattr(api_module, "_create_numerai_client", fake_create_client)

    response = list_numerai_datasets(NumeraiDatasetListRequest(round_num=12, tournament="signals"))

    assert isinstance(response, NumeraiDatasetListResponse)
    assert response.datasets == ["v5.2/train_int8.parquet"]
    assert fake_client.last_round == 12
    assert captured_tournament == "signals"


def test_download_numerai_dataset_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeNumeraiClient()
    captured_tournament: NumeraiTournament | None = None

    def fake_create_client(*, tournament: NumeraiTournament = "classic") -> _FakeNumeraiClient:
        nonlocal captured_tournament
        captured_tournament = tournament
        return fake_client

    monkeypatch.setattr(api_module, "_create_numerai_client", fake_create_client)

    response = download_numerai_dataset(
        NumeraiDatasetDownloadRequest(
            filename="v5.2/train_int8.parquet",
            tournament="crypto",
            dest_path="cache/train.parquet",
            round_num=9,
        )
    )

    assert isinstance(response, NumeraiDatasetDownloadResponse)
    assert response.path == "cache/train.parquet"
    assert fake_client.last_download == ("v5.2/train_int8.parquet", "cache/train.parquet", 9)
    assert captured_tournament == "crypto"


def test_download_numerai_dataset_default_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeNumeraiClient()
    monkeypatch.setattr(
        api_module,
        "_create_numerai_client",
        lambda *, tournament="classic": fake_client,
    )

    response = download_numerai_dataset(
        NumeraiDatasetDownloadRequest(
            filename="v5.2/validation_int8.parquet",
            round_num=3,
        )
    )

    assert isinstance(response, NumeraiDatasetDownloadResponse)
    assert response.path == ".numereng/datasets/v5.2/validation_int8.parquet"
    assert fake_client.last_download == (
        "v5.2/validation_int8.parquet",
        ".numereng/datasets/v5.2/validation_int8.parquet",
        3,
    )


def test_list_numerai_models_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeNumeraiClient()
    captured_tournament: NumeraiTournament | None = None

    def fake_create_client(*, tournament: NumeraiTournament = "classic") -> _FakeNumeraiClient:
        nonlocal captured_tournament
        captured_tournament = tournament
        return fake_client

    monkeypatch.setattr(api_module, "_create_numerai_client", fake_create_client)

    response = list_numerai_models(NumeraiModelsRequest(tournament="signals"))

    assert isinstance(response, NumeraiModelsResponse)
    assert response.models == {"main": "model-1"}
    assert captured_tournament == "signals"


def test_get_numerai_current_round_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeNumeraiClient()
    captured_tournament: NumeraiTournament | None = None

    def fake_create_client(*, tournament: NumeraiTournament = "classic") -> _FakeNumeraiClient:
        nonlocal captured_tournament
        captured_tournament = tournament
        return fake_client

    monkeypatch.setattr(api_module, "_create_numerai_client", fake_create_client)

    response = get_numerai_current_round(NumeraiCurrentRoundRequest(tournament="crypto"))

    assert isinstance(response, NumeraiCurrentRoundResponse)
    assert response.round_num == 777
    assert captured_tournament == "crypto"


def test_scrape_numerai_forum_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_scrape_forum_posts(
        *,
        output_dir: str,
        state_path: str | None,
        full_refresh: bool,
    ) -> dict[str, object]:
        assert output_dir == "docs/numerai/forum"
        assert state_path == "tmp/forum_state.json"
        assert full_refresh is True
        return {
            "output_dir": output_dir,
            "posts_dir": "docs/numerai/forum/posts",
            "index_path": "docs/numerai/forum/INDEX.md",
            "manifest_path": "docs/numerai/forum/.forum_scraper_manifest.json",
            "state_path": "tmp/forum_state.json",
            "mode": "full",
            "pages_fetched": 10,
            "fetched_posts": 200,
            "new_posts": 200,
            "total_posts": 200,
            "latest_post_id": 200,
            "oldest_post_id": 1,
            "started_at": "2026-03-02T00:00:00Z",
            "completed_at": "2026-03-02T00:01:00Z",
        }

    monkeypatch.setattr(api_module, "scrape_forum_posts", fake_scrape_forum_posts)

    response = api_module.scrape_numerai_forum(
        output_dir="docs/numerai/forum",
        state_path="tmp/forum_state.json",
        full_refresh=True,
    )

    assert isinstance(response, api_module.NumeraiForumScrapeResponse)
    assert response.mode == "full"
    assert response.total_posts == 200


def test_scrape_numerai_forum_translates_platform_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_scrape_forum_posts(
        *,
        output_dir: str,
        state_path: str | None,
        full_refresh: bool,
    ) -> dict[str, object]:
        _ = (output_dir, state_path, full_refresh)
        raise ForumScraperError("forum_scraper_network_error")

    monkeypatch.setattr(api_module, "scrape_forum_posts", fake_scrape_forum_posts)

    with pytest.raises(PackageError, match="forum_scraper_network_error"):
        api_module.scrape_numerai_forum()


def test_numerai_api_translates_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ErrorClient:
        def list_datasets(self, *, round_num: int | None = None) -> list[str]:
            _ = round_num
            raise NumeraiClientError("numerai_list_datasets_failed")

        def download_dataset(
            self,
            *,
            filename: str,
            dest_path: str | None = None,
            round_num: int | None = None,
        ) -> str:
            _ = (filename, dest_path, round_num)
            raise NumeraiClientError("numerai_download_dataset_failed")

        def get_models(self) -> dict[str, str]:
            raise NumeraiClientError("numerai_get_models_failed")

        def get_current_round(self) -> int | None:
            raise NumeraiClientError("numerai_get_current_round_failed")

    monkeypatch.setattr(
        api_module,
        "_create_numerai_client",
        lambda *, tournament="classic": _ErrorClient(),
    )

    with pytest.raises(PackageError, match="numerai_list_datasets_failed"):
        list_numerai_datasets(NumeraiDatasetListRequest())
    with pytest.raises(PackageError, match="numerai_download_dataset_failed"):
        download_numerai_dataset(NumeraiDatasetDownloadRequest(filename="v5.2/train_int8.parquet"))
    with pytest.raises(PackageError, match="numerai_get_models_failed"):
        list_numerai_models(NumeraiModelsRequest())
    with pytest.raises(PackageError, match="numerai_get_current_round_failed"):
        get_numerai_current_round(NumeraiCurrentRoundRequest())


def test_score_run_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score_run_pipeline(*, run_id: str, store_root: str, stage: str) -> ScoreRunResult:
        assert run_id == "run-123"
        assert store_root == ".numereng"
        assert stage == "all"
        return ScoreRunResult(
            run_id=run_id,
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
            metrics_path=Path("/tmp/metrics.json"),
            score_provenance_path=Path("/tmp/score_provenance.json"),
            requested_stage="all",
            refreshed_stages=("run_metric_series", "post_training_core"),
        )

    monkeypatch.setattr(api_module, "score_run_pipeline", fake_score_run_pipeline)

    response = score_run(ScoreRunRequest(run_id="run-123"))
    assert isinstance(response, ScoreRunResponse)
    assert response.run_id == "run-123"
    assert response.predictions_path == "/tmp/preds.parquet"
    assert response.results_path == "/tmp/results.json"
    assert response.metrics_path == "/tmp/metrics.json"
    assert response.score_provenance_path == "/tmp/score_provenance.json"
    assert response.requested_stage == "all"
    assert response.refreshed_stages == ["run_metric_series", "post_training_core"]


def test_score_run_sets_api_launch_metadata_when_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score_run_pipeline(*, run_id: str, store_root: str, stage: str) -> ScoreRunResult:
        _ = (run_id, store_root, stage)
        launch_metadata = get_launch_metadata()
        assert launch_metadata is not None
        assert launch_metadata.source == "api.run.score"
        assert launch_metadata.operation_type == "run"
        assert launch_metadata.job_type == "run"
        return ScoreRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
            metrics_path=Path("/tmp/metrics.json"),
            score_provenance_path=Path("/tmp/score_provenance.json"),
            requested_stage="all",
        )

    monkeypatch.setattr(api_module, "score_run_pipeline", fake_score_run_pipeline)

    response = score_run(ScoreRunRequest(run_id="run-123"))
    assert response.run_id == "run-123"


def test_score_run_translates_run_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score_run_pipeline(*, run_id: str, store_root: str, stage: str) -> ScoreRunResult:
        _ = (run_id, store_root, stage)
        raise TrainingError("training_score_run_not_found:run-404")

    monkeypatch.setattr(api_module, "score_run_pipeline", fake_score_run_pipeline)

    with pytest.raises(PackageError, match="training_score_run_not_found"):
        score_run(ScoreRunRequest(run_id="run-404"))


def test_get_run_lifecycle_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_run_lifecycle_record(*, store_root: str, run_id: str) -> RunLifecycleRecord:
        assert store_root == ".numereng"
        assert run_id == "run-123"
        return RunLifecycleRecord(
            run_id="run-123",
            run_hash="hash-run-123",
            config_hash="cfg-hash",
            job_id="job-123",
            logical_run_id="logical-123",
            attempt_id="attempt-123",
            attempt_no=1,
            source="cli.run.train",
            operation_type="run",
            job_type="run",
            status="running",
            experiment_id=None,
            config_id="configs/run.json",
            config_source="store",
            config_path="configs/run.json",
            config_sha256="sha",
            run_dir="/tmp/.numereng/runs/run-123",
            runtime_path="/tmp/.numereng/runs/run-123/runtime.json",
            backend="local",
            worker_id="local",
            pid=1234,
            host="localhost",
            current_stage="train_model",
            completed_stages=("initializing", "load_data"),
            progress_percent=42.5,
            progress_label="Fold 2 of 4",
            progress_current=1,
            progress_total=4,
            cancel_requested=False,
            cancel_requested_at=None,
            created_at="2026-03-24T00:00:00+00:00",
            queued_at="2026-03-24T00:00:00+00:00",
            started_at="2026-03-24T00:00:01+00:00",
            last_heartbeat_at="2026-03-24T00:00:02+00:00",
            updated_at="2026-03-24T00:00:02+00:00",
            finished_at=None,
            terminal_reason=None,
            terminal_detail={},
            latest_metrics={"corr_mean": 0.1},
            latest_sample={"process_rss_gb": 0.5},
            reconciled=False,
        )

    monkeypatch.setattr(api_module, "get_run_lifecycle_record", fake_get_run_lifecycle_record)

    response = get_run_lifecycle(RunLifecycleRequest(run_id="run-123"))
    assert isinstance(response, RunLifecycleResponse)
    assert response.run_id == "run-123"
    assert response.current_stage == "train_model"
    assert response.progress_percent == 42.5
    assert response.latest_metrics["corr_mean"] == 0.1


def test_cancel_run_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request_run_cancel_record(*, store_root: str, run_id: str) -> RunCancelResult:
        assert store_root == ".numereng"
        assert run_id == "run-123"
        return RunCancelResult(
            run_id="run-123",
            job_id="job-123",
            status="running",
            cancel_requested=True,
            cancel_requested_at="2026-03-24T00:00:03+00:00",
            accepted=True,
        )

    monkeypatch.setattr(api_module, "request_run_cancel_record", fake_request_run_cancel_record)

    response = cancel_run(RunCancelRequest(run_id="run-123"))
    assert isinstance(response, RunCancelResponse)
    assert response.accepted is True
    assert response.cancel_requested is True


def test_store_repair_run_lifecycles_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_reconcile_run_lifecycles_record(
        *,
        store_root: str,
        run_id: str | None,
        active_only: bool,
    ) -> RunLifecycleRepairResult:
        assert store_root == ".numereng"
        assert run_id is None
        assert active_only is True
        return RunLifecycleRepairResult(
            store_root=Path("/tmp/.numereng"),
            scanned_count=2,
            unchanged_count=1,
            reconciled_count=1,
            reconciled_stale_count=1,
            reconciled_canceled_count=0,
            run_ids=("run-stale",),
        )

    monkeypatch.setattr(api_module, "reconcile_run_lifecycles_record", fake_reconcile_run_lifecycles_record)

    response = store_repair_run_lifecycles(StoreRunLifecycleRepairRequest())
    assert isinstance(response, StoreRunLifecycleRepairResponse)
    assert response.reconciled_stale_count == 1
    assert response.run_ids == ["run-stale"]


def test_run_training_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        assert config_path == "configs/run.json"
        assert output_dir is None
        assert post_training_scoring is None
        assert engine_mode is None
        assert window_size_eras is None
        assert embargo_eras is None
        assert experiment_id is None
        return TrainingRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)
    response = run_training(TrainRunRequest(config_path="configs/run.json"))
    assert isinstance(response, TrainRunResponse)
    assert response.run_id == "run-123"
    assert response.predictions_path == "/tmp/preds.parquet"
    assert response.results_path == "/tmp/results.json"


def test_run_training_sets_api_launch_metadata_when_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
            experiment_id,
        )
        launch_metadata = get_launch_metadata()
        assert launch_metadata is not None
        assert launch_metadata.source == "api.run.train"
        assert launch_metadata.operation_type == "run"
        assert launch_metadata.job_type == "run"
        return TrainingRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    response = run_training(TrainRunRequest(config_path="configs/run.json"))
    assert response.run_id == "run-123"


def test_train_run_request_allows_resolver_semantic_validation() -> None:
    request = TrainRunRequest(config_path="configs/run.json", engine_mode="custom")
    assert request.engine_mode == "custom"
    assert request.post_training_scoring is None
    assert request.window_size_eras is None
    assert request.embargo_eras is None


def test_train_run_request_allows_custom_knobs_for_resolver_to_evaluate() -> None:
    request = TrainRunRequest(
        config_path="configs/run.json",
        window_size_eras=128,
        embargo_eras=8,
    )
    assert request.engine_mode is None
    assert request.window_size_eras == 128
    assert request.embargo_eras == 8


def test_train_run_request_accepts_post_training_scoring_policy() -> None:
    request = TrainRunRequest(config_path="configs/run.json", post_training_scoring="round_core")
    assert request.post_training_scoring == "round_core"


def test_train_run_request_rejects_non_json_config_path() -> None:
    with pytest.raises(ValidationError, match="config_path must reference a .json file"):
        TrainRunRequest(config_path="configs/run.yaml")


def test_experiment_train_request_rejects_non_json_config_path() -> None:
    with pytest.raises(ValidationError, match="config_path must reference a .json file"):
        ExperimentTrainRequest(experiment_id="2026-02-22_test-exp", config_path="configs/run.yaml")


def test_experiment_train_request_rejects_submission_profile_rename() -> None:
    with pytest.raises(
        ValidationError,
        match="training profile 'submission' was renamed to 'full_history_refit'",
    ):
        ExperimentTrainRequest(
            experiment_id="2026-02-22_test-exp",
            config_path="configs/run.json",
            profile=cast(Any, "submission"),
        )


def test_experiment_train_request_accepts_full_history_refit_profile() -> None:
    request = ExperimentTrainRequest(
        experiment_id="2026-02-22_test-exp",
        config_path="configs/run.json",
        profile="full_history_refit",
    )

    assert request.profile == "full_history_refit"


def test_run_training_translates_backend_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
            experiment_id,
        )
        raise TrainingModelError("training_model_backend_missing_lightgbm")

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    with pytest.raises(PackageError, match="training_model_backend_missing"):
        run_training(TrainRunRequest(config_path="configs/run.json"))


def test_run_training_translates_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
            experiment_id,
        )
        raise ValueError("bad integer")

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    with pytest.raises(PackageError, match="training_config_invalid"):
        run_training(TrainRunRequest(config_path="configs/run.json"))


def test_run_training_translates_training_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
            experiment_id,
        )
        raise TrainingError("training_store_index_failed:run-123")

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    with pytest.raises(PackageError, match="training_run_failed"):
        run_training(TrainRunRequest(config_path="configs/run.json"))


def test_run_training_passes_unlabeled_target_training_data_error_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
            experiment_id,
        )
        raise TrainingDataError("training_target_rows_all_unlabeled:split=train:target=target_jeremy_20:fold=0")

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    with pytest.raises(
        PackageError, match="training_target_rows_all_unlabeled:split=train:target=target_jeremy_20:fold=0"
    ):
        run_training(TrainRunRequest(config_path="configs/run.json"))


def test_run_training_passes_round_policy_error_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
            experiment_id,
        )
        raise TrainingError("training_post_training_scoring_round_requires_experiment_workflow")

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    with pytest.raises(PackageError, match="training_post_training_scoring_round_requires_experiment_workflow"):
        run_training(TrainRunRequest(config_path="configs/run.json", post_training_scoring="round_core"))


def test_run_training_translates_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
            experiment_id,
        )
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    with pytest.raises(PackageError, match="training_unexpected_error:RuntimeError"):
        run_training(TrainRunRequest(config_path="configs/run.json"))


def test_run_training_passes_engine_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        assert config_path == "configs/run.json"
        assert output_dir == "out"
        assert post_training_scoring is None
        assert engine_mode == "custom"
        assert window_size_eras == 144
        assert embargo_eras == 9
        assert experiment_id is None
        return TrainingRunResult(
            run_id="run-xyz",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    response = run_training(
        TrainRunRequest(
            config_path="configs/run.json",
            output_dir="out",
            engine_mode="custom",
            window_size_eras=144,
            embargo_eras=9,
        )
    )
    assert response.run_id == "run-xyz"
    assert response.results_path == "/tmp/results.json"


def test_run_training_passes_experiment_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_training_pipeline(
        *,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
        )
        assert experiment_id == "2026-02-22_test-exp"
        return TrainingRunResult(
            run_id="run-exp",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(api_module, "run_training_pipeline", fake_run_training_pipeline)

    response = run_training(
        TrainRunRequest(
            config_path="configs/run.json",
            experiment_id="2026-02-22_test-exp",
        )
    )
    assert response.run_id == "run-exp"


def test_experiment_create_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_create_experiment(
        *,
        store_root: str,
        experiment_id: str,
        name: str | None,
        hypothesis: str | None,
        tags: list[str] | None,
    ) -> ExperimentRecord:
        assert store_root == ".numereng"
        assert experiment_id == "2026-02-22_test-exp"
        assert name == "Test Experiment"
        assert hypothesis == "Test hypothesis"
        assert tags == ["quick", "baseline"]
        return ExperimentRecord(
            experiment_id=experiment_id,
            name=name or experiment_id,
            status="draft",
            hypothesis=hypothesis,
            tags=("quick", "baseline"),
            created_at="2026-02-22T00:00:00+00:00",
            updated_at="2026-02-22T00:00:00+00:00",
            champion_run_id=None,
            runs=(),
            metadata={},
            manifest_path=Path("/tmp/experiments/2026-02-22_test-exp/experiment.json"),
        )

    monkeypatch.setattr(api_module, "create_experiment_record", fake_create_experiment)
    response = experiment_create(
        ExperimentCreateRequest(
            experiment_id="2026-02-22_test-exp",
            name="Test Experiment",
            hypothesis="Test hypothesis",
            tags=["quick", "baseline"],
        )
    )
    assert response.experiment_id == "2026-02-22_test-exp"
    assert response.status == "draft"
    assert response.tags == ["quick", "baseline"]


def test_experiment_list_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_list_experiments(*, store_root: str, status: str | None) -> tuple[ExperimentRecord, ...]:
        assert store_root == ".numereng"
        assert status == "active"
        return (
            ExperimentRecord(
                experiment_id="2026-02-22_test-exp",
                name="Test Experiment",
                status="active",
                hypothesis=None,
                tags=("quick",),
                created_at="2026-02-22T00:00:00+00:00",
                updated_at="2026-02-22T00:05:00+00:00",
                champion_run_id="run-1",
                runs=("run-1", "run-2"),
                metadata={"foo": "bar"},
                manifest_path=Path("/tmp/experiments/2026-02-22_test-exp/experiment.json"),
            ),
        )

    monkeypatch.setattr(api_module, "list_experiment_records", fake_list_experiments)
    response = experiment_list(ExperimentListRequest(status="active"))
    assert isinstance(response, ExperimentListResponse)
    assert len(response.experiments) == 1
    assert response.experiments[0].experiment_id == "2026-02-22_test-exp"


def test_experiment_archive_and_unarchive_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_archive_experiment(*, store_root: str, experiment_id: str) -> ExperimentArchiveResult:
        assert store_root == ".numereng"
        assert experiment_id == "2026-02-22_test-exp"
        return ExperimentArchiveResult(
            experiment_id=experiment_id,
            status="archived",
            manifest_path=Path("/tmp/experiments/_archive/2026-02-22_test-exp/experiment.json"),
            archived=True,
        )

    def fake_unarchive_experiment(*, store_root: str, experiment_id: str) -> ExperimentArchiveResult:
        assert store_root == ".numereng"
        assert experiment_id == "2026-02-22_test-exp"
        return ExperimentArchiveResult(
            experiment_id=experiment_id,
            status="active",
            manifest_path=Path("/tmp/experiments/2026-02-22_test-exp/experiment.json"),
            archived=False,
        )

    monkeypatch.setattr(api_module, "archive_experiment_record", fake_archive_experiment)
    monkeypatch.setattr(api_module, "unarchive_experiment_record", fake_unarchive_experiment)

    archived = experiment_archive(ExperimentArchiveRequest(experiment_id="2026-02-22_test-exp"))
    restored = experiment_unarchive(ExperimentArchiveRequest(experiment_id="2026-02-22_test-exp"))

    assert isinstance(archived, ExperimentArchiveResponse)
    assert archived.archived is True
    assert archived.status == "archived"
    assert isinstance(restored, ExperimentArchiveResponse)
    assert restored.archived is False
    assert restored.status == "active"


def test_experiment_train_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_train_experiment(
        *,
        store_root: str,
        experiment_id: str,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
    ) -> ExperimentTrainResult:
        assert store_root == ".numereng"
        assert experiment_id == "2026-02-22_test-exp"
        assert config_path == "configs/run.json"
        assert output_dir == "out"
        assert post_training_scoring is None
        assert engine_mode == "custom"
        assert window_size_eras == 128
        assert embargo_eras == 8
        return ExperimentTrainResult(
            experiment_id=experiment_id,
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(api_module, "train_experiment_record", fake_train_experiment)
    response = experiment_train(
        ExperimentTrainRequest(
            experiment_id="2026-02-22_test-exp",
            config_path="configs/run.json",
            output_dir="out",
            engine_mode="custom",
            window_size_eras=128,
            embargo_eras=8,
        )
    )
    assert isinstance(response, ExperimentTrainResponse)
    assert response.run_id == "run-123"


def test_experiment_train_sets_api_launch_metadata_when_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_train_experiment(
        *,
        store_root: str,
        experiment_id: str,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
    ) -> ExperimentTrainResult:
        _ = (
            store_root,
            experiment_id,
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
        )
        launch_metadata = get_launch_metadata()
        assert launch_metadata is not None
        assert launch_metadata.source == "api.experiment.train"
        assert launch_metadata.operation_type == "run"
        assert launch_metadata.job_type == "run"
        return ExperimentTrainResult(
            experiment_id=experiment_id,
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(api_module, "train_experiment_record", fake_train_experiment)

    response = experiment_train(
        ExperimentTrainRequest(
            experiment_id="2026-02-22_test-exp",
            config_path="configs/run.json",
        )
    )

    assert isinstance(response, ExperimentTrainResponse)
    assert response.run_id == "run-123"


def test_experiment_train_passes_post_training_scoring_override(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_train_experiment(
        *,
        store_root: str,
        experiment_id: str,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
    ) -> ExperimentTrainResult:
        _ = (store_root, experiment_id, config_path, output_dir, engine_mode, window_size_eras, embargo_eras)
        assert post_training_scoring == "round_full"
        return ExperimentTrainResult(
            experiment_id=experiment_id,
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(api_module, "train_experiment_record", fake_train_experiment)

    response = experiment_train(
        ExperimentTrainRequest(
            experiment_id="2026-02-22_test-exp",
            config_path="configs/run.json",
            post_training_scoring="round_full",
        )
    )

    assert response.run_id == "run-123"


def test_experiment_score_round_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score_experiment_round(
        *,
        store_root: str,
        experiment_id: str,
        round: str,
        stage: str,
    ) -> ExperimentScoreRoundResult:
        assert store_root == ".numereng"
        assert experiment_id == "2026-02-22_test-exp"
        assert round == "r1"
        assert stage == "post_training_full"
        return ExperimentScoreRoundResult(
            experiment_id=experiment_id,
            round=round,
            stage="post_training_full",
            run_ids=("run-1", "run-2"),
        )

    monkeypatch.setattr(api_module, "score_experiment_round_record", fake_score_experiment_round)

    response = experiment_score_round(
        ExperimentScoreRoundRequest(
            experiment_id="2026-02-22_test-exp",
            round="r1",
            stage="post_training_full",
        )
    )

    assert isinstance(response, ExperimentScoreRoundResponse)
    assert response.experiment_id == "2026-02-22_test-exp"
    assert response.round == "r1"
    assert response.stage == "post_training_full"
    assert response.run_ids == ["run-1", "run-2"]


def test_experiment_promote_and_report_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_promote_experiment(
        *,
        store_root: str,
        experiment_id: str,
        run_id: str | None,
        metric: str,
    ) -> ExperimentPromotionResult:
        assert store_root == ".numereng"
        assert experiment_id == "2026-02-22_test-exp"
        assert run_id is None
        assert metric == "bmc_last_200_eras.mean"
        return ExperimentPromotionResult(
            experiment_id=experiment_id,
            champion_run_id="run-2",
            metric=metric,
            metric_value=0.123,
            auto_selected=True,
        )

    def fake_report_experiment(
        *,
        store_root: str,
        experiment_id: str,
        metric: str,
        limit: int,
    ) -> ExperimentReport:
        assert store_root == ".numereng"
        assert experiment_id == "2026-02-22_test-exp"
        assert metric == "bmc_last_200_eras.mean"
        assert limit == 5
        return ExperimentReport(
            experiment_id=experiment_id,
            metric=metric,
            total_runs=2,
            champion_run_id="run-2",
            rows=(
                ExperimentReportRow(
                    run_id="run-2",
                    status="FINISHED",
                    created_at="2026-02-22T00:00:00+00:00",
                    metric_value=0.123,
                    corr_mean=0.11,
                    mmc_mean=0.09,
                    cwmm_mean=0.04,
                    bmc_mean=0.12,
                    bmc_last_200_eras_mean=0.123,
                    is_champion=True,
                ),
            ),
        )

    monkeypatch.setattr(api_module, "promote_experiment_record", fake_promote_experiment)
    monkeypatch.setattr(api_module, "report_experiment_record", fake_report_experiment)

    promote_response = experiment_promote(ExperimentPromoteRequest(experiment_id="2026-02-22_test-exp"))
    assert isinstance(promote_response, ExperimentPromoteResponse)
    assert promote_response.champion_run_id == "run-2"

    report_response = experiment_report(
        ExperimentReportRequest(
            experiment_id="2026-02-22_test-exp",
            limit=5,
        )
    )
    assert isinstance(report_response, ExperimentReportResponse)
    assert report_response.total_runs == 2
    assert report_response.rows[0].run_id == "run-2"


def test_experiment_pack_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pack_experiment(*, store_root: str, experiment_id: str) -> ExperimentPackResult:
        assert store_root == ".numereng"
        assert experiment_id == "2026-02-22_test-exp"
        return ExperimentPackResult(
            experiment_id=experiment_id,
            output_path=Path("/tmp/experiments/2026-02-22_test-exp/EXPERIMENT.pack.md"),
            experiment_path=Path("/tmp/experiments/2026-02-22_test-exp"),
            source_markdown_path=Path("/tmp/experiments/2026-02-22_test-exp/EXPERIMENT.md"),
            run_count=2,
            packed_at="2026-02-22T00:05:00+00:00",
        )

    monkeypatch.setattr(api_module, "pack_experiment_record", fake_pack_experiment)

    response = experiment_pack(ExperimentPackRequest(experiment_id="2026-02-22_test-exp"))

    assert isinstance(response, ExperimentPackResponse)
    assert response.run_count == 2
    assert response.output_path.endswith("EXPERIMENT.pack.md")


def test_experiment_pack_translates_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pack_experiment(*, store_root: str, experiment_id: str) -> ExperimentPackResult:
        _ = (store_root, experiment_id)
        raise ExperimentValidationError("experiment_doc_missing:/tmp/EXPERIMENT.md")

    monkeypatch.setattr(api_module, "pack_experiment_record", fake_pack_experiment)

    with pytest.raises(PackageError, match="experiment_doc_missing"):
        experiment_pack(ExperimentPackRequest(experiment_id="2026-02-22_test-exp"))


def test_experiment_get_translates_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_experiment(*, store_root: str, experiment_id: str) -> ExperimentRecord:
        _ = (store_root, experiment_id)
        raise ExperimentNotFoundError("experiment_not_found:missing")

    monkeypatch.setattr(api_module, "get_experiment_record", fake_get_experiment)

    with pytest.raises(PackageError, match="experiment_not_found:missing"):
        experiment_get(ExperimentGetRequest(experiment_id="missing"))


def test_experiment_train_translates_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_train_experiment(
        *,
        store_root: str,
        experiment_id: str,
        config_path: str,
        output_dir: str | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
    ) -> ExperimentTrainResult:
        _ = (
            store_root,
            experiment_id,
            config_path,
            output_dir,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
        )
        raise ExperimentValidationError("experiment_output_dir_must_match_store_root")

    monkeypatch.setattr(api_module, "train_experiment_record", fake_train_experiment)

    with pytest.raises(PackageError, match="experiment_output_dir_must_match_store_root"):
        experiment_train(
            ExperimentTrainRequest(
                experiment_id="2026-02-22_test-exp",
                config_path="configs/run.json",
                output_dir=".numereng-mismatch",
            )
        )


def test_research_api_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_init_research_program(
        *,
        store_root: str,
        experiment_id: str,
        program_id: str,
        improvement_threshold: float | None,
    ) -> ResearchInitResult:
        _ = store_root
        return ResearchInitResult(
            root_experiment_id=experiment_id,
            program_id=program_id,
            program_title="Numerai Experiment Loop",
            status="initialized",
            active_experiment_id=experiment_id,
            active_path_id="p00",
            improvement_threshold=improvement_threshold or 0.0002,
            current_phase=None,
            agentic_research_dir=Path("/tmp/agentic_research"),
            program_path=Path("/tmp/agentic_research/program.json"),
            lineage_path=Path("/tmp/agentic_research/lineage.json"),
            session_program_path=Path("/tmp/agentic_research/session_program.md"),
        )

    monkeypatch.setattr(
        api_module,
        "init_research_program",
        fake_init_research_program,
    )
    monkeypatch.setattr(
        api_module,
        "get_research_program_status",
        lambda *, store_root, experiment_id: ResearchStatusResult(
            root_experiment_id=experiment_id,
            program_id="numerai-experiment-loop",
            program_title="Numerai Experiment Loop",
            status="running",
            active_experiment_id=experiment_id,
            active_path_id="p00",
            next_round_number=2,
            total_rounds_completed=1,
            total_paths_created=1,
            improvement_threshold=0.0002,
            last_checkpoint="round_completed",
            stop_reason=None,
            best_overall=ResearchBestRun(run_id="run-4", bmc_last_200_eras_mean=0.123),
            current_round=None,
            current_phase=None,
            program_path=Path("/tmp/agentic_research/program.json"),
            lineage_path=Path("/tmp/agentic_research/lineage.json"),
            session_program_path=Path("/tmp/agentic_research/session_program.md"),
        ),
    )
    monkeypatch.setattr(
        api_module,
        "run_research_program",
        lambda *, store_root, experiment_id, max_rounds, max_paths: ResearchRunResult(
            root_experiment_id=experiment_id,
            program_id="numerai-experiment-loop",
            program_title="Numerai Experiment Loop",
            status="stopped",
            active_experiment_id=experiment_id,
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

    init_payload = research_init(
        ResearchInitRequest(
            experiment_id="2026-02-22_test-exp",
            program_id="numerai-experiment-loop",
        )
    )
    assert isinstance(init_payload, ResearchInitResponse)
    assert init_payload.active_path_id == "p00"
    assert init_payload.program_id == "numerai-experiment-loop"

    status_payload = research_status(ResearchStatusRequest(experiment_id="2026-02-22_test-exp"))
    assert isinstance(status_payload, ResearchStatusResponse)
    assert status_payload.program_title == "Numerai Experiment Loop"
    assert status_payload.best_overall.run_id == "run-4"

    run_payload = research_run(ResearchRunRequest(experiment_id="2026-02-22_test-exp", max_rounds=1))
    assert isinstance(run_payload, ResearchRunResponse)
    assert run_payload.program_id == "numerai-experiment-loop"
    assert run_payload.stop_reason == "max_rounds_reached"


def test_research_program_catalog_api_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "list_research_program_records",
        lambda user_programs_dir=None: (
            ResearchProgramCatalogEntry(
                program_id="numerai-experiment-loop",
                title="Numerai Experiment Loop",
                description="Builtin numerai loop.",
                source="builtin",
                planner_contract="config_mutation",
                phase_aware=False,
                source_path="/tmp/numerai-experiment-loop.md",
            ),
        ),
    )
    monkeypatch.setattr(
        api_module,
        "get_research_program_record",
        lambda program_id, user_programs_dir=None: ResearchProgramDetails(
            definition=ResearchProgramDefinition(
                program_id=program_id,
                title="Numerai Experiment Loop",
                description="Builtin numerai loop.",
                source="builtin",
                planner_contract="config_mutation",
                scoring_stage="post_training_full",
                metric_policy=ResearchProgramMetricPolicy(
                    primary="bmc_last_200_eras.mean",
                    tie_break="bmc.mean",
                    sanity_checks=("corr.mean",),
                ),
                round_policy=ResearchProgramRoundPolicy(
                    plateau_non_improving_rounds=2,
                    require_scale_confirmation=True,
                    scale_confirmation_rounds=1,
                ),
                improvement_threshold_default=0.0002,
                config_policy=ResearchProgramConfigPolicy(
                    allowed_paths=("model.params.*",),
                    max_candidate_configs=1,
                    min_changes=1,
                    max_changes=2,
                ),
                prompt_template="Context:\n$CONTEXT_JSON\n\n$VALIDATION_FEEDBACK_BLOCK\n",
                source_path="/tmp/numerai-experiment-loop.md",
            ),
            raw_markdown=(
                "---\nid: numerai-experiment-loop\n---\nContext:\n$CONTEXT_JSON\n\n$VALIDATION_FEEDBACK_BLOCK\n"
            ),
        ),
    )

    list_payload = research_program_list(ResearchProgramListRequest())
    show_payload = research_program_show(ResearchProgramShowRequest(program_id="numerai-experiment-loop"))

    assert isinstance(list_payload, ResearchProgramListResponse)
    assert list_payload.programs[0].program_id == "numerai-experiment-loop"
    assert isinstance(show_payload, ResearchProgramShowResponse)
    assert show_payload.planner_contract == "config_mutation"


def test_store_init_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_init_store_db(*, store_root: str) -> StoreInitResult:
        assert store_root == ".numereng"
        return StoreInitResult(
            store_root=Path("/tmp/.numereng"),
            db_path=Path("/tmp/.numereng/numereng.db"),
            created=True,
            schema_migration="2026_02_store_index_v3_experiments",
        )

    monkeypatch.setattr(api_module, "init_store_db", fake_init_store_db)

    response = store_init(StoreInitRequest())

    assert isinstance(response, StoreInitResponse)
    assert response.created is True
    assert response.schema_migration == "2026_02_store_index_v3_experiments"


def test_store_index_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_index_run(*, store_root: str, run_id: str) -> StoreIndexResult:
        assert store_root == ".numereng"
        assert run_id == "run-123"
        return StoreIndexResult(
            run_id="run-123",
            status="FINISHED",
            metrics_indexed=12,
            artifacts_indexed=5,
            run_path=Path("/tmp/.numereng/runs/run-123"),
            warnings=(),
        )

    monkeypatch.setattr(api_module, "index_run", fake_index_run)

    response = store_index_run(StoreIndexRequest(run_id="run-123"))

    assert isinstance(response, StoreIndexResponse)
    assert response.run_id == "run-123"
    assert response.metrics_indexed == 12
    assert response.artifacts_indexed == 5
    assert response.warnings == []


def test_store_rebuild_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_rebuild_run_index(*, store_root: str) -> StoreRebuildResult:
        assert store_root == ".numereng"
        return StoreRebuildResult(
            store_root=Path("/tmp/.numereng"),
            db_path=Path("/tmp/.numereng/numereng.db"),
            scanned_runs=3,
            indexed_runs=2,
            failed_runs=1,
            failures=(),
        )

    monkeypatch.setattr(api_module, "rebuild_run_index", fake_rebuild_run_index)

    response = store_rebuild(StoreRebuildRequest())

    assert isinstance(response, StoreRebuildResponse)
    assert response.scanned_runs == 3
    assert response.indexed_runs == 2
    assert response.failed_runs == 1


def test_store_doctor_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_doctor_store(*, store_root: str, fix_strays: bool) -> StoreDoctorResult:
        assert store_root == ".numereng"
        assert fix_strays is False
        return StoreDoctorResult(
            store_root=Path("/tmp/.numereng"),
            db_path=Path("/tmp/.numereng/numereng.db"),
            ok=True,
            issues=(),
            stats={"filesystem_runs": 2, "indexed_runs": 2},
            stray_cleanup_applied=False,
            deleted_paths=(),
            missing_paths=(),
        )

    monkeypatch.setattr(api_module, "doctor_store", fake_doctor_store)

    response = store_doctor(StoreDoctorRequest())

    assert isinstance(response, StoreDoctorResponse)
    assert response.ok is True
    assert response.stats["filesystem_runs"] == 2
    assert response.stray_cleanup_applied is False


def test_store_doctor_fix_strays_flag_passes_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_doctor_store(*, store_root: str, fix_strays: bool) -> StoreDoctorResult:
        assert store_root == ".numereng"
        assert fix_strays is True
        return StoreDoctorResult(
            store_root=Path("/tmp/.numereng"),
            db_path=Path("/tmp/.numereng/numereng.db"),
            ok=True,
            issues=(),
            stats={"filesystem_runs": 2, "indexed_runs": 2},
            stray_cleanup_applied=True,
            deleted_paths=("/tmp/.numereng/modal_smoke_data",),
            missing_paths=("/tmp/.numereng/smoke_live_check",),
        )

    monkeypatch.setattr(api_module, "doctor_store", fake_doctor_store)

    response = store_doctor(StoreDoctorRequest(fix_strays=True))

    assert response.stray_cleanup_applied is True
    assert response.deleted_paths == ["/tmp/.numereng/modal_smoke_data"]
    assert response.missing_paths == ["/tmp/.numereng/smoke_live_check"]


def test_store_backfill_run_execution_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_backfill_run_execution(
        *, store_root: str, run_id: str | None, all_runs: bool
    ) -> StoreRunExecutionBackfillResult:
        assert store_root == ".numereng"
        assert run_id is None
        assert all_runs is True
        return StoreRunExecutionBackfillResult(
            store_root=Path("/tmp/.numereng"),
            scanned_runs=3,
            updated_runs=2,
            skipped_runs=1,
            ambiguous_runs=("run-3",),
            updated_run_ids=("run-1", "run-2"),
            skipped_run_ids=("run-3",),
        )

    monkeypatch.setattr(api_module, "backfill_run_execution", fake_backfill_run_execution)

    response = store_backfill_run_execution(StoreRunExecutionBackfillRequest(all=True))

    assert isinstance(response, StoreRunExecutionBackfillResponse)
    assert response.updated_count == 2
    assert response.ambiguous_runs == ["run-3"]
    assert response.updated_run_ids == ["run-1", "run-2"]


def test_store_materialize_viz_artifacts_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_materialize_viz_artifacts(
        *,
        store_root: str,
        kind: str,
        run_id: str | None,
        experiment_id: str | None,
        all_runs: bool,
    ) -> StoreMaterializeVizArtifactsResult:
        assert store_root == ".numereng"
        assert kind == "per-era-corr"
        assert run_id is None
        assert experiment_id == "exp-1"
        assert all_runs is False
        return StoreMaterializeVizArtifactsResult(
            store_root=Path("/tmp/.numereng"),
            kind="per-era-corr",
            scoped_run_count=3,
            created_count=2,
            skipped_count=1,
            failed_count=0,
            failures=(),
        )

    monkeypatch.setattr(api_module, "materialize_viz_artifacts", fake_materialize_viz_artifacts)

    response = store_materialize_viz_artifacts(
        StoreMaterializeVizArtifactsRequest(kind="per-era-corr", experiment_id="exp-1")
    )

    assert isinstance(response, StoreMaterializeVizArtifactsResponse)
    assert response.created_count == 2
    assert response.skipped_count == 1


def test_store_api_translates_store_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_index_run(*, store_root: str, run_id: str) -> StoreIndexResult:
        _ = (store_root, run_id)
        raise StoreError("store_run_not_found:run-x")

    monkeypatch.setattr(api_module, "index_run", fake_index_run)

    with pytest.raises(PackageError, match="store_run_not_found:run-x"):
        store_index_run(StoreIndexRequest(run_id="run-x"))


class _CloudServiceRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def _ok(self, method: str, request: object) -> CloudEc2Response:
        self.calls.append((method, request))
        return CloudEc2Response(action=method, message="ok", result={"method": method})

    def init_iam(self, request: object) -> CloudEc2Response:
        return self._ok("init_iam", request)

    def setup_data(self, request: object) -> CloudEc2Response:
        return self._ok("setup_data", request)

    def provision(self, request: object) -> CloudEc2Response:
        return self._ok("provision", request)

    def package_build_upload(self, request: object) -> CloudEc2Response:
        return self._ok("package_build_upload", request)

    def config_upload(self, request: object) -> CloudEc2Response:
        return self._ok("config_upload", request)

    def push(self, request: object) -> CloudEc2Response:
        return self._ok("push", request)

    def install(self, request: object) -> CloudEc2Response:
        return self._ok("install", request)

    def train_start(self, request: object) -> CloudEc2Response:
        return self._ok("train_start", request)

    def train_poll(self, request: object) -> CloudEc2Response:
        return self._ok("train_poll", request)

    def logs(self, request: object) -> CloudEc2Response:
        return self._ok("logs", request)

    def pull(self, request: object) -> CloudEc2Response:
        return self._ok("pull", request)

    def terminate(self, request: object) -> CloudEc2Response:
        return self._ok("terminate", request)

    def status(self, request: object) -> CloudEc2Response:
        return self._ok("status", request)

    def s3_list(self, request: object) -> CloudEc2Response:
        return self._ok("s3_list", request)

    def s3_copy(self, request: object) -> CloudEc2Response:
        return self._ok("s3_copy", request)

    def s3_remove(self, request: object) -> CloudEc2Response:
        return self._ok("s3_remove", request)


@pytest.mark.parametrize(
    ("api_func", "req", "method"),
    [
        (api_module.cloud_ec2_init_iam, api_module.Ec2InitIamRequest(), "init_iam"),
        (api_module.cloud_ec2_setup_data, api_module.Ec2SetupDataRequest(data_version="v5.2"), "setup_data"),
        (api_module.cloud_ec2_provision, api_module.Ec2ProvisionRequest(run_id="run-1"), "provision"),
        (
            api_module.cloud_ec2_package_build_upload,
            api_module.Ec2PackageBuildUploadRequest(run_id="run-1"),
            "package_build_upload",
        ),
        (
            api_module.cloud_ec2_config_upload,
            api_module.Ec2ConfigUploadRequest(run_id="run-1", config_path="config.json"),
            "config_upload",
        ),
        (api_module.cloud_ec2_push, api_module.Ec2PushRequest(run_id="run-1", instance_id="i-1"), "push"),
        (api_module.cloud_ec2_install, api_module.Ec2InstallRequest(run_id="run-1", instance_id="i-1"), "install"),
        (
            api_module.cloud_ec2_train_start,
            api_module.Ec2TrainStartRequest(run_id="run-1", instance_id="i-1"),
            "train_start",
        ),
        (
            api_module.cloud_ec2_train_poll,
            api_module.Ec2TrainPollRequest(run_id="run-1", instance_id="i-1"),
            "train_poll",
        ),
        (api_module.cloud_ec2_logs, api_module.Ec2LogsRequest(instance_id="i-1"), "logs"),
        (api_module.cloud_ec2_pull, api_module.Ec2PullRequest(run_id="run-1", instance_id="i-1"), "pull"),
        (api_module.cloud_ec2_terminate, api_module.Ec2TerminateRequest(instance_id="i-1"), "terminate"),
        (api_module.cloud_ec2_status, api_module.Ec2StatusRequest(run_id="run-1"), "status"),
        (api_module.cloud_ec2_s3_list, api_module.Ec2S3ListRequest(prefix="runs/"), "s3_list"),
        (
            api_module.cloud_ec2_s3_copy,
            api_module.Ec2S3CopyRequest(src="s3://bucket/a", dst="s3://bucket/b"),
            "s3_copy",
        ),
        (api_module.cloud_ec2_s3_remove, api_module.Ec2S3RemoveRequest(uri="s3://bucket/a"), "s3_remove"),
    ],
)
def test_cloud_ec2_api_functions_delegate_to_service(
    monkeypatch: pytest.MonkeyPatch,
    api_func: CloudApiFunc,
    req: object,
    method: str,
) -> None:
    recorder = _CloudServiceRecorder()
    monkeypatch.setattr(api_module, "_create_cloud_ec2_service", lambda: recorder)

    response = api_func(req)
    assert response.action == method
    assert recorder.calls[0][0] == method


@pytest.mark.parametrize(
    ("api_func", "req"),
    [
        (api_module.cloud_ec2_status, api_module.Ec2StatusRequest(run_id="run-1")),
        (api_module.cloud_ec2_s3_list, api_module.Ec2S3ListRequest(prefix="runs/")),
        (api_module.cloud_ec2_train_start, api_module.Ec2TrainStartRequest(run_id="run-1", instance_id="i-1")),
    ],
)
def test_cloud_ec2_api_translates_cloud_error(
    monkeypatch: pytest.MonkeyPatch,
    api_func: CloudApiFunc,
    req: object,
) -> None:
    class _FailingCloudService:
        def status(self, request: object) -> CloudEc2Response:
            _ = request
            raise CloudEc2Error("cloud_failed")

        def s3_list(self, request: object) -> CloudEc2Response:
            _ = request
            raise CloudEc2Error("cloud_failed")

        def train_start(self, request: object) -> CloudEc2Response:
            _ = request
            raise CloudEc2Error("cloud_failed")

    monkeypatch.setattr(api_module, "_create_cloud_ec2_service", lambda: _FailingCloudService())

    with pytest.raises(PackageError, match="cloud_failed"):
        api_func(req)


def test_cloud_ec2_config_upload_request_rejects_non_json_config() -> None:
    with pytest.raises(ValidationError, match="config_path must reference a .json file"):
        api_module.Ec2ConfigUploadRequest(run_id="run-1", config_path="config.yaml")


class _CloudAwsServiceRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def _ok(self, method: str, request: object) -> CloudAwsResponse:
        self.calls.append((method, request))
        return CloudAwsResponse(action=method, message="ok", result={"method": method})

    def image_build_push(self, request: object) -> CloudAwsResponse:
        return self._ok("image_build_push", request)

    def train_submit(self, request: object) -> CloudAwsResponse:
        return self._ok("train_submit", request)

    def train_status(self, request: object) -> CloudAwsResponse:
        return self._ok("train_status", request)

    def train_logs(self, request: object) -> CloudAwsResponse:
        return self._ok("train_logs", request)

    def train_cancel(self, request: object) -> CloudAwsResponse:
        return self._ok("train_cancel", request)

    def train_pull(self, request: object) -> CloudAwsResponse:
        return self._ok("train_pull", request)

    def train_extract(self, request: object) -> CloudAwsResponse:
        return self._ok("train_extract", request)


def test_cloud_aws_train_submit_request_rejects_non_json_config_values() -> None:
    with pytest.raises(ValidationError, match="config_path must reference a .json file"):
        AwsTrainSubmitRequest(run_id="run-1", config_path="train.yaml")

    with pytest.raises(ValidationError, match="config_s3_uri must reference a .json object"):
        AwsTrainSubmitRequest(run_id="run-1", config_s3_uri="s3://bucket/runs/run-1/config.yaml")


def test_cloud_modal_train_submit_request_profile_contract() -> None:
    request = ModalTrainSubmitRequest(config_path="config.json", profile="full_history_refit")
    assert request.profile == "full_history_refit"

    with pytest.raises(
        ValidationError,
        match="training profile 'submission' was renamed to 'full_history_refit'",
    ):
        ModalTrainSubmitRequest(config_path="config.json", profile=cast(Any, "submission"))


@pytest.mark.parametrize(
    ("api_func", "req", "method"),
    [
        (api_module.cloud_aws_image_build_push, AwsImageBuildPushRequest(), "image_build_push"),
        (
            api_module.cloud_aws_train_submit,
            AwsTrainSubmitRequest(
                run_id="run-1",
                config_s3_uri="s3://bucket/runs/run-1/config.json",
                image_uri="123456.dkr.ecr.us-east-2.amazonaws.com/numereng:v1",
            ),
            "train_submit",
        ),
        (api_module.cloud_aws_train_status, AwsTrainStatusRequest(run_id="run-1"), "train_status"),
        (api_module.cloud_aws_train_logs, AwsTrainLogsRequest(run_id="run-1"), "train_logs"),
        (api_module.cloud_aws_train_cancel, AwsTrainCancelRequest(run_id="run-1"), "train_cancel"),
        (api_module.cloud_aws_train_pull, AwsTrainPullRequest(run_id="run-1"), "train_pull"),
        (api_module.cloud_aws_train_extract, AwsTrainExtractRequest(run_id="run-1"), "train_extract"),
    ],
)
def test_cloud_aws_api_functions_delegate_to_service(
    monkeypatch: pytest.MonkeyPatch,
    api_func: ManagedCloudApiFunc,
    req: object,
    method: str,
) -> None:
    recorder = _CloudAwsServiceRecorder()
    monkeypatch.setattr(api_module, "_create_cloud_aws_managed_service", lambda: recorder)

    response = api_func(req)
    assert response.action == method
    assert recorder.calls[0][0] == method


@pytest.mark.parametrize(
    ("api_func", "req"),
    [
        (api_module.cloud_aws_train_status, AwsTrainStatusRequest(run_id="run-1")),
        (api_module.cloud_aws_train_logs, AwsTrainLogsRequest(run_id="run-1")),
        (api_module.cloud_aws_train_pull, AwsTrainPullRequest(run_id="run-1")),
        (api_module.cloud_aws_train_extract, AwsTrainExtractRequest(run_id="run-1")),
    ],
)
def test_cloud_aws_api_translates_cloud_error(
    monkeypatch: pytest.MonkeyPatch,
    api_func: ManagedCloudApiFunc,
    req: object,
) -> None:
    class _FailingCloudAwsService:
        def train_status(self, request: object) -> CloudAwsResponse:
            _ = request
            raise CloudAwsError("managed_cloud_failed")

        def train_logs(self, request: object) -> CloudAwsResponse:
            _ = request
            raise CloudAwsError("managed_cloud_failed")

        def train_pull(self, request: object) -> CloudAwsResponse:
            _ = request
            raise CloudAwsError("managed_cloud_failed")

        def train_extract(self, request: object) -> CloudAwsResponse:
            _ = request
            raise CloudAwsError("managed_cloud_failed")

    monkeypatch.setattr(api_module, "_create_cloud_aws_managed_service", lambda: _FailingCloudAwsService())

    with pytest.raises(PackageError, match="managed_cloud_failed"):
        api_func(req)


@pytest.mark.parametrize(
    ("api_func", "req"),
    [
        (api_module.cloud_aws_image_build_push, AwsImageBuildPushRequest()),
        (
            api_module.cloud_aws_train_submit,
            AwsTrainSubmitRequest(
                run_id="run-1",
                config_s3_uri="s3://bucket/runs/run-1/config.json",
                image_uri="123456.dkr.ecr.us-east-2.amazonaws.com/numereng:v1",
            ),
        ),
        (api_module.cloud_aws_train_status, AwsTrainStatusRequest(run_id="run-1")),
        (api_module.cloud_aws_train_logs, AwsTrainLogsRequest(run_id="run-1")),
        (api_module.cloud_aws_train_cancel, AwsTrainCancelRequest(run_id="run-1")),
        (api_module.cloud_aws_train_pull, AwsTrainPullRequest(run_id="run-1")),
        (api_module.cloud_aws_train_extract, AwsTrainExtractRequest(run_id="run-1")),
    ],
)
def test_cloud_aws_api_translates_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
    api_func: ManagedCloudApiFunc,
    req: object,
) -> None:
    class _UnexpectedCloudAwsService:
        def image_build_push(self, request: object) -> CloudAwsResponse:
            _ = request
            raise RuntimeError("boom")

        def train_submit(self, request: object) -> CloudAwsResponse:
            _ = request
            raise RuntimeError("boom")

        def train_status(self, request: object) -> CloudAwsResponse:
            _ = request
            raise RuntimeError("boom")

        def train_logs(self, request: object) -> CloudAwsResponse:
            _ = request
            raise RuntimeError("boom")

        def train_cancel(self, request: object) -> CloudAwsResponse:
            _ = request
            raise RuntimeError("boom")

        def train_pull(self, request: object) -> CloudAwsResponse:
            _ = request
            raise RuntimeError("boom")

        def train_extract(self, request: object) -> CloudAwsResponse:
            _ = request
            raise RuntimeError("boom")

    monkeypatch.setattr(api_module, "_create_cloud_aws_managed_service", lambda: _UnexpectedCloudAwsService())

    with pytest.raises(PackageError, match="cloud_aws_unexpected_error:boom"):
        api_func(req)


class _CloudModalServiceRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def _ok(self, method: str, request: object) -> CloudModalResponse:
        self.calls.append((method, request))
        return CloudModalResponse(action=method, message="ok", result={"method": method})

    def train_submit(self, request: object) -> CloudModalResponse:
        return self._ok("train_submit", request)

    def train_status(self, request: object) -> CloudModalResponse:
        return self._ok("train_status", request)

    def train_logs(self, request: object) -> CloudModalResponse:
        return self._ok("train_logs", request)

    def train_cancel(self, request: object) -> CloudModalResponse:
        return self._ok("train_cancel", request)

    def train_pull(self, request: object) -> CloudModalResponse:
        return self._ok("train_pull", request)

    def deploy(self, request: object) -> CloudModalResponse:
        return self._ok("deploy", request)

    def data_sync(self, request: object) -> CloudModalResponse:
        return self._ok("data_sync", request)


@pytest.mark.parametrize(
    ("api_func", "req", "method"),
    [
        (
            api_module.cloud_modal_data_sync,
            ModalDataSyncRequest(config_path="config.json", volume_name="numereng-v52"),
            "data_sync",
        ),
        (
            api_module.cloud_modal_deploy,
            ModalDeployRequest(ecr_image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest"),
            "deploy",
        ),
        (
            api_module.cloud_modal_train_submit,
            ModalTrainSubmitRequest(config_path="config.json"),
            "train_submit",
        ),
        (api_module.cloud_modal_train_status, ModalTrainStatusRequest(call_id="fc-1"), "train_status"),
        (api_module.cloud_modal_train_logs, ModalTrainLogsRequest(call_id="fc-1"), "train_logs"),
        (api_module.cloud_modal_train_cancel, ModalTrainCancelRequest(call_id="fc-1"), "train_cancel"),
        (api_module.cloud_modal_train_pull, ModalTrainPullRequest(call_id="fc-1"), "train_pull"),
    ],
)
def test_cloud_modal_api_functions_delegate_to_service(
    monkeypatch: pytest.MonkeyPatch,
    api_func: ModalCloudApiFunc,
    req: object,
    method: str,
) -> None:
    recorder = _CloudModalServiceRecorder()
    monkeypatch.setattr(api_module, "_create_cloud_modal_service", lambda: recorder)

    response = api_func(req)
    assert response.action == method
    assert recorder.calls[0][0] == method


@pytest.mark.parametrize(
    ("api_func", "req"),
    [
        (
            api_module.cloud_modal_data_sync,
            ModalDataSyncRequest(config_path="config.json", volume_name="numereng-v52"),
        ),
        (
            api_module.cloud_modal_deploy,
            ModalDeployRequest(ecr_image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest"),
        ),
        (api_module.cloud_modal_train_status, ModalTrainStatusRequest(call_id="fc-1")),
        (api_module.cloud_modal_train_logs, ModalTrainLogsRequest(call_id="fc-1")),
        (api_module.cloud_modal_train_pull, ModalTrainPullRequest(call_id="fc-1")),
    ],
)
def test_cloud_modal_api_translates_cloud_error(
    monkeypatch: pytest.MonkeyPatch,
    api_func: ModalCloudApiFunc,
    req: object,
) -> None:
    class _FailingCloudModalService:
        def data_sync(self, request: object) -> CloudModalResponse:
            _ = request
            raise CloudModalError("modal_cloud_failed")

        def deploy(self, request: object) -> CloudModalResponse:
            _ = request
            raise CloudModalError("modal_cloud_failed")

        def train_status(self, request: object) -> CloudModalResponse:
            _ = request
            raise CloudModalError("modal_cloud_failed")

        def train_logs(self, request: object) -> CloudModalResponse:
            _ = request
            raise CloudModalError("modal_cloud_failed")

        def train_pull(self, request: object) -> CloudModalResponse:
            _ = request
            raise CloudModalError("modal_cloud_failed")

    monkeypatch.setattr(api_module, "_create_cloud_modal_service", lambda: _FailingCloudModalService())

    with pytest.raises(PackageError, match="modal_cloud_failed"):
        api_func(req)


@pytest.mark.parametrize(
    ("api_func", "req"),
    [
        (
            api_module.cloud_modal_data_sync,
            ModalDataSyncRequest(config_path="config.json", volume_name="numereng-v52"),
        ),
        (
            api_module.cloud_modal_deploy,
            ModalDeployRequest(ecr_image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/numereng-training:latest"),
        ),
        (api_module.cloud_modal_train_submit, ModalTrainSubmitRequest(config_path="config.json")),
        (api_module.cloud_modal_train_status, ModalTrainStatusRequest(call_id="fc-1")),
        (api_module.cloud_modal_train_logs, ModalTrainLogsRequest(call_id="fc-1")),
        (api_module.cloud_modal_train_cancel, ModalTrainCancelRequest(call_id="fc-1")),
        (api_module.cloud_modal_train_pull, ModalTrainPullRequest(call_id="fc-1")),
    ],
)
def test_cloud_modal_api_translates_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
    api_func: ModalCloudApiFunc,
    req: object,
) -> None:
    class _UnexpectedCloudModalService:
        def data_sync(self, request: object) -> CloudModalResponse:
            _ = request
            raise RuntimeError("boom")

        def deploy(self, request: object) -> CloudModalResponse:
            _ = request
            raise RuntimeError("boom")

        def train_submit(self, request: object) -> CloudModalResponse:
            _ = request
            raise RuntimeError("boom")

        def train_status(self, request: object) -> CloudModalResponse:
            _ = request
            raise RuntimeError("boom")

        def train_logs(self, request: object) -> CloudModalResponse:
            _ = request
            raise RuntimeError("boom")

        def train_cancel(self, request: object) -> CloudModalResponse:
            _ = request
            raise RuntimeError("boom")

        def train_pull(self, request: object) -> CloudModalResponse:
            _ = request
            raise RuntimeError("boom")

    monkeypatch.setattr(api_module, "_create_cloud_modal_service", lambda: _UnexpectedCloudModalService())

    with pytest.raises(PackageError, match="cloud_modal_unexpected_error:boom"):
        api_func(req)


def test_api_exports_are_explicit() -> None:
    exports = api_module.__all__
    export_set = set(exports)

    required_exports = {
        "PackageError",
        "HealthResponse",
        "get_health",
        "run_bootstrap_check",
        "run_training",
        "score_run",
        "submit_predictions",
        "store_init",
        "store_index_run",
        "store_rebuild",
        "store_doctor",
        "list_numerai_datasets",
        "download_numerai_dataset",
        "list_numerai_models",
        "get_numerai_current_round",
        "scrape_numerai_forum",
        "NumeraiForumScrapeResponse",
        "scrape_forum_posts",
        "_create_numerai_client",
    }

    assert required_exports.issubset(export_set)
    assert len(exports) == len(export_set)
    assert all(hasattr(api_module, name) for name in exports)
