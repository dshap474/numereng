# Python API

Numereng exposes a stable public Python facade for the same workflows available from the CLI.

## Stable Public Surfaces

- `import numereng.api`
- `import numereng.api.pipeline`
- `import numereng.api.contracts`

Underscore-prefixed modules under `src/numereng/api/` are implementation modules, not stable import paths.

## Core Pattern

1. construct a typed request object from `numereng.api.contracts`
2. call the matching function from `numereng.api` or `numereng.api.pipeline`
3. consume the typed response object

## Public Workflow Entry Point

`numereng.api.pipeline.run_training_pipeline()` is the explicit public training workflow facade.

Example:

```python
from numereng.api.contracts import TrainRunRequest
from numereng.api.pipeline import run_training_pipeline

response = run_training_pipeline(
    TrainRunRequest(config_path="configs/run.json")
)

print(response.run_id)
print(response.predictions_path)
print(response.results_path)
```

## Common Public Functions

### Runs

- `run_training(TrainRunRequest)`
- `score_run(ScoreRunRequest)`
- `submit_predictions(SubmissionRequest)`

### Experiments

- `experiment_create(ExperimentCreateRequest)`
- `experiment_list(ExperimentListRequest | None = None)`
- `experiment_get(ExperimentGetRequest)`
- `experiment_train(ExperimentTrainRequest)`
- `experiment_report(ExperimentReportRequest)`
- `experiment_promote(ExperimentPromoteRequest)`

### HPO

- `hpo_create(HpoStudyCreateRequest)`
- `hpo_list(HpoStudyListRequest | None = None)`
- `hpo_get(HpoStudyGetRequest)`
- `hpo_trials(HpoStudyTrialsRequest)`

HPO v2 requests are nested:

- `study_id`, `study_name`, `config_path`, optional `experiment_id`
- `objective`
- explicit `search_space`
- `sampler`
- `stopping`

`HpoStudyResponse` now returns the full nested `spec` plus summary fields such as `attempted_trials`, `completed_trials`, `failed_trials`, and `stop_reason`.

### Ensembles

- `ensemble_build(EnsembleBuildRequest)`
- `ensemble_list(EnsembleListRequest | None = None)`
- `ensemble_get(EnsembleGetRequest)`

### Neutralization And Store

- `neutralize_apply(NeutralizeRequest)`
- `store_init(StoreInitRequest | None = None)`
- `store_index_run(StoreIndexRequest)`
- `store_rebuild(StoreRebuildRequest | None = None)`
- `store_doctor(StoreDoctorRequest | None = None)`

### Numerai Operations

- `list_numerai_datasets(NumeraiDatasetListRequest | None = None)`
- `download_numerai_dataset(NumeraiDatasetDownloadRequest)`
- `list_numerai_models(NumeraiModelsRequest | None = None)`
- `get_numerai_current_round(NumeraiCurrentRoundRequest | None = None)`
- `scrape_numerai_forum(output_dir=..., state_path=..., full_refresh=...)`

### Dataset Tools

- `dataset_tools_build_downsampled_full(DatasetToolsBuildDownsampleRequest)`

### Cloud

The public API facade also exposes typed cloud entrypoints for:

- EC2
- managed AWS training
- Modal deploy/data/train workflows

Use `numereng.api.contracts` for the request models and `numereng.api.cloud`/`numereng.api` exports for the corresponding functions.

## Contracts

The public request/response types live in `numereng.api.contracts`.

Examples:

- `TrainRunRequest`
- `TrainRunResponse`
- `ScoreRunRequest`
- `ExperimentCreateRequest`
- `HpoStudyCreateRequest`
- `EnsembleBuildRequest`
- `NeutralizeRequest`
- `SubmissionRequest`

## Error Model

Public API functions raise `PackageError` for boundary failures. Feature-internal exceptions are translated before they cross the API boundary.
