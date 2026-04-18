# Python API

`numereng` exposes a stable public Python facade for the same workflows available from the CLI.

## Stable Imports

- `import numereng.api`
- `import numereng.api.pipeline`
- `import numereng.api.contracts`

Do not treat underscore-prefixed modules under `src/numereng/api/` as stable imports.

## Core Pattern

1. build a typed request from `numereng.api.contracts`
2. call the matching function from `numereng.api` or `numereng.api.pipeline`
3. consume the typed response

## Canonical Training Entry Point

Use `numereng.api.pipeline.run_training_pipeline()` for the explicit public train workflow facade.

```python
from numereng.api.contracts import TrainRunRequest
from numereng.api.pipeline import run_training_pipeline

response = run_training_pipeline(
    TrainRunRequest(config_path="configs/run.json")
)

print(response.run_id)
print(response.predictions_path)
```

## Major Function Groups

### Runs

- `run_training(TrainRunRequest)`
- `score_run(ScoreRunRequest)`
- `submit_predictions(SubmissionRequest)`
- `cancel_run(RunCancelRequest)`

### Baselines And Dataset Tools

- `baseline_build(BaselineBuildRequest)`
- `dataset_tools_build_downsampled_full(DatasetToolsBuildDownsampleRequest)`

### Experiments And Research

- `experiment_create(ExperimentCreateRequest)`
- `experiment_list(ExperimentListRequest | None = None)`
- `experiment_get(ExperimentGetRequest)`
- `experiment_train(ExperimentTrainRequest)`
- `experiment_report(ExperimentReportRequest)`
- `experiment_promote(ExperimentPromoteRequest)`
- `experiment_pack(ExperimentPackRequest)`
- `experiment_archive(ExperimentArchiveRequest)`
- `experiment_unarchive(ExperimentUnarchiveRequest)`
- `experiment_score_round(ExperimentScoreRoundRequest)`
- `research_program_list(ResearchProgramListRequest | None = None)`
- `research_program_show(ResearchProgramShowRequest)`
- `research_init(ResearchInitRequest)`
- `research_status(ResearchStatusRequest)`
- `research_run(ResearchRunRequest)`

### HPO And Ensembles

- `hpo_create(HpoStudyCreateRequest)`
- `hpo_list(HpoStudyListRequest | None = None)`
- `hpo_get(HpoStudyGetRequest)`
- `hpo_trials(HpoStudyTrialsRequest)`
- `ensemble_build(EnsembleBuildRequest)`
- `ensemble_list(EnsembleListRequest | None = None)`
- `ensemble_get(EnsembleGetRequest)`

### Serving And Submission

- `serve_package_create(ServePackageCreateRequest)`
- `serve_package_inspect(ServePackageInspectRequest)`
- `serve_package_list(ServePackageListRequest | None = None)`
- `serve_package_score(ServePackageScoreRequest)`
- `serve_package_sync_diagnostics(ServePackageSyncDiagnosticsRequest)`
- `serve_live_build(ServeLiveBuildRequest)`
- `serve_live_submit(ServeLiveSubmitRequest)`
- `serve_pickle_build(ServePickleBuildRequest)`
- `serve_pickle_upload(ServePickleUploadRequest)`
- `neutralize_apply(NeutralizeRequest)`

### Store, Monitor, Remote, Cloud, Numerai

- `store_init(StoreInitRequest | None = None)`
- `store_index_run(StoreIndexRequest)`
- `store_rebuild(StoreRebuildRequest | None = None)`
- `store_doctor(StoreDoctorRequest | None = None)`
- `store_backfill_run_execution(StoreRunExecutionBackfillRequest)`
- `store_repair_run_lifecycles(StoreRunLifecycleRepairRequest)`
- `store_materialize_viz_artifacts(StoreMaterializeVizArtifactsRequest)`
- `build_monitor_snapshot(MonitorSnapshotRequest | None = None)`
- `remote_list_targets(RemoteTargetListRequest | None = None)`
- `remote_bootstrap_viz(RemoteVizBootstrapRequest | None = None)`
- `remote_doctor(RemoteDoctorRequest)`
- `remote_repo_sync(RemoteRepoSyncRequest)`
- `remote_experiment_sync(RemoteExperimentSyncRequest)`
- `remote_experiment_launch(RemoteExperimentLaunchRequest)`
- `remote_experiment_status(RemoteExperimentStatusRequest)`
- `remote_experiment_maintain(RemoteExperimentMaintainRequest)`
- `remote_experiment_stop(RemoteExperimentStopRequest)`
- `remote_experiment_pull(RemoteExperimentPullRequest)`
- `remote_config_push(RemoteConfigPushRequest)`
- `remote_train_launch(RemoteTrainLaunchRequest)`
- cloud EC2, managed AWS, and Modal functions exposed through `numereng.api.cloud` and the top-level facade
- `list_numerai_datasets(NumeraiDatasetListRequest | None = None)`
- `download_numerai_dataset(NumeraiDatasetDownloadRequest)`
- `list_numerai_models(NumeraiModelsRequest | None = None)`
- `get_numerai_current_round(NumeraiCurrentRoundRequest | None = None)`
- `scrape_numerai_forum(...)`

## Contracts

All stable request and response models live in `numereng.api.contracts`.

Key examples:

- `TrainRunRequest`
- `ExperimentCreateRequest`
- `ResearchInitRequest`
- `HpoStudyCreateRequest`
- `EnsembleBuildRequest`
- `ServePackageCreateRequest`
- `StoreDoctorRequest`
- `MonitorSnapshotRequest`
- `RemoteExperimentLaunchRequest`

## Error Model

Public API functions raise `PackageError` for boundary failures. Feature-internal exceptions are translated before they cross the API boundary.
