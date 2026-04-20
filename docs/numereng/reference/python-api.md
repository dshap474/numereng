# Python API

`numereng` exposes a stable public Python facade for the same workflows available from the CLI.

## Stable Imports

- `import numereng.api`
- `import numereng.api.pipeline`
- `import numereng.api.contracts`
- `import numereng.api.cloud`

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

`numereng.api.run_training()` is also stable, but `run_training_pipeline()` is the clearest end-to-end training entry point.

## Stable Function Groups

### Runs

- `run_training(TrainRunRequest)`
- `score_run(ScoreRunRequest)`
- `submit_predictions(SubmissionRequest)`
- `cancel_run(RunCancelRequest)`

### Docs And Viz

- `sync_docs(DocsSyncRequest | None = None)`
- `create_viz_app(VizAppRequest | None = None)`

### Baselines And Dataset Tools

- `baseline_build(BaselineBuildRequest)`
- `dataset_tools_build_downsampled_full(DatasetToolsBuildDownsampleRequest)`

### Experiments And Research

- `experiment_create(ExperimentCreateRequest)`
- `experiment_list(ExperimentListRequest | None = None)`
- `experiment_get(ExperimentGetRequest)`
- `experiment_train(ExperimentTrainRequest)`
- `experiment_run_plan(ExperimentRunPlanRequest)`
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
- `ensemble_select(EnsembleSelectRequest)`
- `ensemble_list(EnsembleListRequest | None = None)`
- `ensemble_get(EnsembleGetRequest)`

### Serving And Neutralization

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

### Store And Monitor

- `store_init(StoreInitRequest | None = None)`
- `store_index_run(StoreIndexRequest)`
- `store_rebuild(StoreRebuildRequest | None = None)`
- `store_doctor(StoreDoctorRequest | None = None)`
- `store_backfill_run_execution(StoreRunExecutionBackfillRequest)`
- `store_repair_run_lifecycles(StoreRunLifecycleRepairRequest)`
- `store_materialize_viz_artifacts(StoreMaterializeVizArtifactsRequest)`
- `build_monitor_snapshot(MonitorSnapshotRequest | None = None)`

### Remote

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

### Cloud

Cloud helpers are available from both `numereng.api` and `numereng.api.cloud`.

EC2:

- `cloud_ec2_init_iam(Ec2InitIamRequest)`
- `cloud_ec2_setup_data(Ec2SetupDataRequest)`
- `cloud_ec2_provision(Ec2ProvisionRequest)`
- `cloud_ec2_package_build_upload(Ec2PackageBuildUploadRequest)`
- `cloud_ec2_config_upload(Ec2ConfigUploadRequest)`
- `cloud_ec2_push(Ec2PushRequest)`
- `cloud_ec2_install(Ec2InstallRequest)`
- `cloud_ec2_train_start(Ec2TrainStartRequest)`
- `cloud_ec2_train_poll(Ec2TrainPollRequest)`
- `cloud_ec2_logs(Ec2LogsRequest)`
- `cloud_ec2_pull(Ec2PullRequest)`
- `cloud_ec2_terminate(Ec2TerminateRequest)`
- `cloud_ec2_status(Ec2StatusRequest)`
- `cloud_ec2_s3_list(Ec2S3ListRequest)`
- `cloud_ec2_s3_copy(Ec2S3CopyRequest)`
- `cloud_ec2_s3_remove(Ec2S3RemoveRequest)`

Managed AWS:

- `cloud_aws_image_build_push(AwsImageBuildPushRequest)`
- `cloud_aws_train_submit(AwsTrainSubmitRequest)`
- `cloud_aws_train_status(AwsTrainStatusRequest)`
- `cloud_aws_train_logs(AwsTrainLogsRequest)`
- `cloud_aws_train_cancel(AwsTrainCancelRequest)`
- `cloud_aws_train_pull(AwsTrainPullRequest)`
- `cloud_aws_train_extract(AwsTrainExtractRequest)`

Modal:

- `cloud_modal_deploy(ModalDeployRequest)`
- `cloud_modal_data_sync(ModalDataSyncRequest)`
- `cloud_modal_train_submit(ModalTrainSubmitRequest)`
- `cloud_modal_train_status(ModalTrainStatusRequest)`
- `cloud_modal_train_logs(ModalTrainLogsRequest)`
- `cloud_modal_train_cancel(ModalTrainCancelRequest)`
- `cloud_modal_train_pull(ModalTrainPullRequest)`

### Numerai

- `list_numerai_datasets(NumeraiDatasetListRequest | None = None)`
- `download_numerai_dataset(NumeraiDatasetDownloadRequest)`
- `list_numerai_models(NumeraiModelsRequest | None = None)`
- `get_numerai_current_round(NumeraiCurrentRoundRequest | None = None)`
- `scrape_numerai_forum(output_dir="docs/numerai/forum", state_path=None, full_refresh=False)`

### Support Utilities

- `get_health()`
- `run_bootstrap_check(fail=False)`
- `get_run_lifecycle(RunLifecycleRequest)`

## Contracts

All stable request and response models live in `numereng.api.contracts`.

Key examples:

- `TrainRunRequest`
- `DocsSyncRequest`
- `ExperimentCreateRequest`
- `ExperimentRunPlanRequest`
- `ResearchInitRequest`
- `HpoStudyCreateRequest`
- `EnsembleBuildRequest`
- `EnsembleSelectRequest`
- `Ec2ProvisionRequest`
- `AwsTrainSubmitRequest`
- `ModalTrainSubmitRequest`
- `ServePackageCreateRequest`
- `VizAppRequest`
- `StoreDoctorRequest`
- `MonitorSnapshotRequest`
- `RunLifecycleRequest`
- `RemoteExperimentLaunchRequest`

## Error Model

Public API functions raise `PackageError` for boundary failures. Feature-internal exceptions are translated before they cross the API boundary.

## Undocumented Exports

`numereng.api` currently re-exports additional compatibility and feature-adapter names such as `*_record`, `*_api`, `*_study`, and raw feature helpers. Those exports are intentionally undocumented here and are not part of the user-facing API contract for this guide.
